#!/usr/bin/env python3
"""
Esperimenti con protocollo holdout 80/20 (train/validation), senza test set.

Cosa esegue, per ogni finestra temporale L:
  1. costruisce la partizione 80/20 per traiettoria e stima lo scaler sul solo training;
  2. valuta le quattro baseline banali sulla validation (stesse coppie dei modelli);
  3. addestra tutte le configurazioni della griglia, registrando a ogni epoca le
     metriche di training e di validation;
  4. ripristina i pesi dell'epoca con validation minima;
  5. valuta ogni configurazione sulla validation (next-step e rollout autoregressivo)
     con intervallo di confidenza bootstrap;
  6. seleziona una configurazione per famiglia con una regola dichiarata in anticipo;
  7. salva metriche, tabelle, pesi dei modelli selezionati e figure.

Esempio:
    python run_holdout.py --csv /content/work/trajectories.csv \
        --windows 50 100 --out-dir ../artifacts/holdout

Prova rapida senza GPU (geometria ridotta, serve solo a verificare che giri):
    python run_holdout.py --csv sintetico.csv --quick --out-dir /tmp/holdout_test
"""

import argparse
import json
import os
import time

import numpy as np

from data_preprocessing import (PreprocessConfig, set_global_seeds, dump_json,
                                load_raw_trajectories, validate_trajectories,
                                reproducibility_report)
from models import build_model, save_model_bundle, count_params
from training import TrainConfig
from experiments import default_hp_grid, build_ablation_variants
from baselines import (evaluate_rollout_baselines, evaluate_nextstep_baselines,
                       best_baseline, skill_score)
from holdout_protocol import prepare_holdout, save_holdout_meta
from holdout_training import HoldoutTrainer, evaluate_rollout_detailed
from holdout_plots import make_all_figures, diagnose_curve, plot_curve_taxonomy


# Etichette leggibili per la griglia

def hp_label(kind: str, hp: dict) -> str:
    m = hp["model"]
    if kind == "rnn":
        return f"{m.rnn_type} {m.units}, lr {hp['lr']:g}"
    return f"d_model {m.d_model}, dff {m.dff}, lr {hp['lr']:g}"


# Regola di selezione, dichiarata prima di guardare i risultati

def select_config(entries: dict, kind: str) -> str:
    """Sceglie una configurazione per famiglia.

    Stadio 1: si ordina per RMSE del rollout sulla validation, che e' il compito reale.
    Stadio 2: fra le configurazioni il cui intervallo di confidenza bootstrap al 95%
              si sovrappone a quello della migliore - cioe' quelle che i dati non
              sanno distinguere - si tiene quella con l'errore next-step piu' basso.
    La regola e' la stessa della versione a cross-validation; e' cambiata solo l'origine
    dell'incertezza: bootstrap sulle traiettorie di validation invece della dispersione
    fra i fold.
    """
    names = [n for n, e in entries.items() if e["kind"] == kind]
    if not names:
        return ""
    def ro(n):
        return entries[n]["val"].get("rollout_rmse_physical", np.inf)
    names.sort(key=ro)
    best = names[0]
    b_lo, b_hi = entries[best]["val"].get("rollout_rmse_physical_ci95", (ro(best), ro(best)))
    tied = [n for n in names
            if _overlap(entries[n]["val"].get("rollout_rmse_physical_ci95",
                                              (ro(n), ro(n))), (b_lo, b_hi))]
    if len(tied) <= 1:
        return best
    return min(tied, key=lambda n: entries[n]["val"].get("nextstep_rmse", np.inf))


def _overlap(a, b) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


# Tabelle

def _tex_text(s) -> str:
    """Rende sicuro in LaTeX un testo pensato per le figure.

    Le etichette delle configurazioni contengono caratteri che in LaTeX hanno un
    significato speciale - `d_model` e' il caso concreto: l'underscore aprirebbe un
    pedice in modo matematico e la compilazione si fermerebbe.
    """
    s = str(s)
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, "\\" + ch)
    return s


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def build_tables(results, out_dir, window):
    """Tabella dei risultati in CSV e LaTeX, modelli e baseline nella stessa scala."""
    rows = []
    for name, e in results["configs"].items():
        v = e["val"]
        ci = v.get("rollout_rmse_physical_ci95", (np.nan, np.nan))
        rows.append({
            "config": name, "descrizione": e.get("label", ""),
            "epoca_migliore": e["info"]["best_epoch"],
            "train_mse": e["at_best"]["train_mse"],
            "val_mse": e["at_best"]["val_mse"],
            "gap": e["at_best"]["gap_mse"],
            "val_nextstep_rmse_fis": v.get("nextstep_rmse"),
            "val_rollout_rmse_fis": v.get("rollout_rmse_physical"),
            "ci95_basso": ci[0], "ci95_alto": ci[1],
            "nrmse": v.get("rollout_nrmse"),
            "skill": v.get("skill_vs_baseline"),
            "rmse_magnetizzazioni": v.get("rmse_physical_magnetizations"),
            "rmse_correlazioni": v.get("rmse_physical_correlations"),
        })
    for bname, m in results.get("baselines", {}).items():
        rows.append({
            "config": f"baseline:{bname}", "descrizione": "trivial predictor",
            "epoca_migliore": "--", "train_mse": None, "val_mse": None, "gap": None,
            "val_nextstep_rmse_fis": m.get("nextstep_rmse"),
            "val_rollout_rmse_fis": m.get("rollout_rmse_physical"),
            "ci95_basso": None, "ci95_alto": None,
            "nrmse": m.get("rollout_nrmse"), "skill": m.get("skill_vs_baseline"),
            "rmse_magnetizzazioni": m.get("rmse_physical_magnetizations"),
            "rmse_correlazioni": m.get("rmse_physical_correlations"),
        })

    csv_path = os.path.join(out_dir, f"table_holdout_L{window}.csv")
    cols = list(rows[0].keys())
    import csv as _csv
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r[c] is None else
                        (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c])
                        for c in cols])

    tex_path = os.path.join(out_dir, f"table_holdout_L{window}.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\\begin{table}[t]\n\\centering\n\\small\n")
        fh.write(f"\\caption{{Holdout 80/20 results for context window $L={window}$. "
                 "All metrics are computed on the validation partition; there is no test "
                 "set. The interval is the 95\\% bootstrap confidence interval of the "
                 "rollout RMSE, obtained by resampling the rollout pairs. Rollout and "
                 "next-step RMSE are in physical units; the skill score is relative to "
                 "the strongest trivial baseline, so a negative value means the baseline "
                 "is better.}\n")   # una graffa sola: questo pezzo non e' una f-string
        fh.write(f"\\label{{tab:resL{window}}}\n")
        fh.write("\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n")
        fh.write("\\begin{tabular}{llrrrrrr}\n\\hline\n")
        fh.write("Config. & Description & Ep. & Val MSE & Gap & "
                 "Next-step RMSE & Rollout RMSE (95\\% CI) & Skill \\\\\n\\hline\n")
        for r in rows:
            ci = ("" if r["ci95_basso"] is None
                  else f" [{_fmt(r['ci95_basso'])}, {_fmt(r['ci95_alto'])}]")
            fh.write(f"{_tex_text(r['config'])} & {_tex_text(r['descrizione'])} & "
                     f"{r['epoca_migliore']} & {_fmt(r['val_mse'], 5)} & "
                     f"{_fmt(r['gap'], 5)} & {_fmt(r['val_nextstep_rmse_fis'])} & "
                     f"{_fmt(r['val_rollout_rmse_fis'])}{ci} & "
                     f"{_fmt(r['skill'], 3)} \\\\\n")
        fh.write("\\hline\n\\end{tabular}}\n\\end{table}\n")
    return csv_path, tex_path


# Ablation dello schedule, sotto il protocollo holdout

def run_ablation(L, args, data, bl_ro, selected, grid, out_dir, train_cfg_kwargs):
    """Varianti dello schedule a budget di epoche costante, sulla configurazione
    selezionata di ciascuna famiglia.

    Rispetto alla versione precedente cambia solo dove si misura: non piu' la media sui
    cinque fold, ma la partizione di validation con intervallo bootstrap. Il vincolo
    importante resta: tutte le varianti consumano lo stesso numero totale di epoche, cosi'
    una variante non vince solo perche' si addestra di piu'.
    """
    varianti = build_ablation_variants(train_cfg_kwargs["schedule"],
                                       ss_passes=args.ss_passes)
    print(f"\n{'-'*72}\nABLATION dello schedule (L={L}) - {len(varianti)} varianti "
          f"x {len(selected)} architetture\n{'-'*72}")
    ris = {}
    for kind, name in selected.items():
        if not name:
            continue
        hp = next(h for h in grid[kind] if h["name"] == name)
        ris[kind] = {"config": name, "varianti": {}}
        for vname, (sched_kw, extra_kw) in varianti.items():
            kw = dict(train_cfg_kwargs)
            kw.update(sched_kw)
            kw.update(extra_kw)
            set_global_seeds(args.seed)
            model = build_model(kind, data.X_train.shape[-1], hp["model"])
            tcfg = TrainConfig(batch_size=args.batch_size, lr=hp["lr"],
                               seed=args.seed, verbose=False, **kw)
            trainer = HoldoutTrainer(model, tcfg)
            hist, info = trainer.fit_monitored(
                data.X_train, data.Y_train, data.X_val, data.Y_val,
                X_train_eval=data.X_train_eval, Y_train_eval=data.Y_train_eval,
                scaler=data.scaler, rollout_every=0,
                restore_best=not args.no_restore_best)
            from training import evaluate_nextstep, arrays_to_dataset
            ns = evaluate_nextstep(model, arrays_to_dataset(
                data.X_val, data.Y_val, args.batch_size, shuffle=False),
                scaler=data.scaler)
            ro = evaluate_rollout_detailed(model, data.ctx_val, data.fut_val,
                                           scaler=data.scaler,
                                           feature_idx=data.feature_idx,
                                           n_qubits=args.n_qubits, baselines=bl_ro,
                                           n_boot=args.n_boot, seed=args.seed)
            ci = ro.get("rollout_rmse_physical_ci95", (np.nan, np.nan))
            ris[kind]["varianti"][vname] = {
                "nextstep_rmse": float(ns.get("rmse_physical", ns["rmse"])),
                "rollout_rmse_phys": float(ro["rollout_rmse_physical"]),
                "ci95": [float(ci[0]), float(ci[1])],
                "skill": float(ro.get("skill_vs_baseline", np.nan)),
                "best_epoch": int(info["best_epoch"]),
            }
            print(f"  [{kind:11s}] {vname:38s} rollout {ro['rollout_rmse_physical']:.4f} "
                  f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    _write_ablation_tables(ris, out_dir, L, sum(int(n) for _, n in
                                                train_cfg_kwargs["schedule"]))
    return ris


def _write_ablation_tables(ris, out_dir, L, total_epochs):
    kinds = list(ris.keys())
    if not kinds:
        return
    varianti = list(ris[kinds[0]]["varianti"].keys())
    import csv as _csv
    with open(os.path.join(out_dir, f"table_ablation_L{L}.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["variante"] + [f"{k}_{m}" for k in kinds
                                   for m in ("nextstep_rmse", "rollout_rmse", "ci95", "skill")])
        for v in varianti:
            row = [v]
            for k in kinds:
                a = ris[k]["varianti"][v]
                row += [f"{a['nextstep_rmse']:.4f}", f"{a['rollout_rmse_phys']:.4f}",
                        f"[{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]", f"{a['skill']:+.3f}"]
            w.writerow(row)

    best = {k: min(ris[k]["varianti"],
                   key=lambda v: ris[k]["varianti"][v]["rollout_rmse_phys"])
            for k in kinds}
    out = ["\\begin{table}[t]\\centering\\footnotesize",
           "\\setlength{\\tabcolsep}{4pt}",
           f"\\caption{{Ablation of the multi-regime schedule at constant budget "
           f"({total_epochs} epochs, $L={L}$). Metrics on the validation partition; "
           f"the interval is the 95\\% bootstrap CI of the rollout RMSE.}}",
           "\\label{tab:ablation}",
           "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%",
           "\\begin{tabular}{l" + "cc" * len(kinds) + "}", "\\hline",
           " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{k.upper()}}}" for k in kinds)
           + " \\\\",
           "Training schedule & " + " & ".join(
               "Next-step RMSE & Rollout RMSE (phys)" for _ in kinds) + " \\\\",
           "\\hline"]
    for v in varianti:
        cells = []
        for k in kinds:
            a = ris[k]["varianti"][v]
            ro = (f"{a['rollout_rmse_phys']:.4f} "
                  f"[{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]")
            if v == best[k]:
                ro = f"\\textbf{{{ro}}}"
            cells += [f"{a['nextstep_rmse']:.4f}", ro]
        out.append(_tex_text(v) + " & " + " & ".join(cells) + " \\\\")
    out += ["\\hline", "\\end{tabular}}", "\\end{table}"]
    with open(os.path.join(out_dir, f"table_ablation_L{L}.tex"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(out))


# Numeri per il testo del report

def write_report_numbers(summary_per_window, out_dir):
    """Tutti i numeri che compaiono nella prosa del report, in un file solo.

    Due formati con lo stesso contenuto: `numeri_report.txt` da leggere e copiare a
    mano, e `report_numbers_holdout.tex` con le stesse cifre gia' pronte come comandi
    LaTeX, per chi preferisce fare \\input invece di trascrivere.
    """
    righe, macro = [], []
    CIFRE = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
             "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    # Il report e' in inglese, la console e le figure in italiano: la diagnosi viene
    # tradotta solo qui, dove serve per il testo del documento.
    DIAGNOSI_EN = {
        "sovradattamento": "over-fitting",
        "sottoadattamento (budget insufficiente)": "under-fitting (insufficient budget)",
        "validation piu' bassa del training": "validation below training",
        "buon adattamento": "good fit",
        "curva troppo corta per una diagnosi": "curve too short for a diagnosis",
    }

    def nome_macro(chiave: str) -> str:
        """Nome di comando LaTeX valido: solo lettere, quindi le cifre diventano parole."""
        base = "".join(w.capitalize() for w in chiave.replace(".", "").split("_"))
        return "".join(CIFRE.get(c, c) for c in base if c.isalnum())

    def add(chiave, valore, descrizione, fmt="{:.4f}"):
        testo = fmt.format(valore) if isinstance(valore, float) else str(valore)
        nome = nome_macro(chiave)
        righe.append(f"{chiave:34s} = {testo:>12s}   # {descrizione}   ->  \\{nome}")
        # I nomi delle baseline (context_mean, train_mean) e le diagnosi contengono
        # caratteri speciali per LaTeX: il corpo della macro va reso sicuro.
        macro.append(f"\\newcommand{{\\{nome}}}{{{_tex_text(testo)}}}")

    for tag, res in summary_per_window.items():
        L = res["window"]
        add(f"L{L}_n_train", res["n_train_trajectories"], "traiettorie di training", "{}")
        add(f"L{L}_n_val", res["n_val_trajectories"], "traiettorie di validation", "{}")
        add(f"L{L}_best_baseline", res["best_baseline"], "baseline piu' forte", "{}")
        bb = res["baselines"][res["best_baseline"]]["rollout_rmse_physical"]
        add(f"L{L}_baseline_rmse", bb, "RMSE della baseline piu' forte")
        for kind, name in res["selected"].items():
            if not name:
                continue
            e = res["configs"][name]
            v = e["val"]
            p = f"L{L}_{kind}"
            add(f"{p}_config", name, "configurazione selezionata", "{}")
            add(f"{p}_best_epoch", e["info"]["best_epoch"], "epoca conservata", "{}")
            add(f"{p}_train_mse", e["at_best"]["train_mse"], "MSE training all'epoca migliore", "{:.5f}")
            add(f"{p}_val_mse", e["at_best"]["val_mse"], "MSE validation all'epoca migliore", "{:.5f}")
            add(f"{p}_gap", e["at_best"]["gap_mse"], "divario di generalizzazione", "{:+.5f}")
            verdetto = e["diagnosi"]["verdetto"]
            add(f"{p}_diagnosi", DIAGNOSI_EN.get(verdetto, verdetto),
                f"diagnosi della curva (in italiano: {verdetto})", "{}")
            add(f"{p}_rollout_rmse", v["rollout_rmse_physical"], "RMSE rollout (fisico)")
            ci = v.get("rollout_rmse_physical_ci95", [float("nan")] * 2)
            add(f"{p}_ci_low", ci[0], "estremo inferiore IC 95%")
            add(f"{p}_ci_high", ci[1], "estremo superiore IC 95%")
            add(f"{p}_nextstep_rmse", v["nextstep_rmse"], "RMSE next-step (fisico)")
            add(f"{p}_nrmse", v["rollout_nrmse"], "NRMSE", "{:.3f}")
            add(f"{p}_skill", v.get("skill_vs_baseline", float("nan")), "skill score", "{:+.3f}")
            add(f"{p}_rmse_mag", v.get("rmse_physical_magnetizations", float("nan")), "RMSE magnetizzazioni")
            add(f"{p}_rmse_corr", v.get("rmse_physical_correlations", float("nan")), "RMSE correlazioni")
            add(f"{p}_first_step", v.get("first_step_rmse_physical", float("nan")), "errore al primo passo", "{:.5f}")
            add(f"{p}_final_step", v.get("final_step_rmse_physical", float("nan")), "errore all'ultimo passo", "{:.5f}")
            add(f"{p}_growth", v.get("error_growth_factor", float("nan")), "fattore di crescita", "{:.1f}")
            add(f"{p}_amplitude_ratio", v.get("amplitude_ratio", float("nan")), "rapporto di ampiezza", "{:.3f}")
            for s, c in sorted((v.get("correlation_at_step") or {}).items(),
                               key=lambda kv: int(kv[0])):
                add(f"{p}_corr_step{s}", c, f"correlazione al passo {s}", "{:.3f}")
        righe.append("")

    txt = os.path.join(out_dir, "numeri_report.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("# Numeri per il testo del report - protocollo holdout 80/20\n")
        fh.write("# Le tabelle complete sono in table_holdout_L*.tex; qui ci sono le\n")
        fh.write("# cifre che compaiono nella prosa.\n\n")
        fh.write("\n".join(righe))
    tex = os.path.join(out_dir, "report_numbers_holdout.tex")
    with open(tex, "w", encoding="utf-8") as fh:
        fh.write("% Generato da run_holdout.py - stesse cifre di numeri_report.txt\n")
        fh.write("\n".join(macro) + "\n")
    return txt, tex


# Esecuzione di una finestra

def run_window(L, args, raw_cache):
    print(f"\n{'='*72}\nFINESTRA L={L}\n{'='*72}")
    cfg = PreprocessConfig(
        csv_path=args.csv, input_window=L, horizon=args.horizon,
        feature_set=args.feature_set, seed=args.seed,
        train_frac=1.0 - args.val_frac, val_frac=args.val_frac, test_frac=0.0,
        rollout_origins=args.origins, train_stride=args.train_stride,
        max_trajectories=args.max_trajectories, n_points=args.n_points,
        n_qubits=args.n_qubits,
    )
    data = prepare_holdout(cfg, val_frac=args.val_frac, verbose=True,
                           strict_checks=not args.quick, raw_cache=raw_cache)
    out_dir = os.path.join(args.out_dir, f"L{L}")
    os.makedirs(out_dir, exist_ok=True)
    save_holdout_meta(data, out_dir)

    # Baseline sulla validation, sulle stesse identiche coppie usate per i modelli
    print("[baseline] valutazione sulla validation")
    bl_ro = evaluate_rollout_baselines(data.ctx_val, data.fut_val, scaler=data.scaler,
                                       feature_idx=data.feature_idx,
                                       n_qubits=cfg.n_qubits)
    bl_ns = evaluate_nextstep_baselines(data.X_val, data.Y_val, scaler=data.scaler)
    bb_name, bb_val = best_baseline(bl_ro)
    baselines = {}
    for name, m in bl_ro.items():
        baselines[name] = {
            "rollout_rmse_physical": float(m["rollout_rmse_physical"]),
            "rollout_nrmse": float(m.get("rollout_nrmse", np.nan)),
            "per_horizon_rmse_physical": np.asarray(
                m.get("per_horizon_rmse_physical", m["per_horizon_rmse"])).tolist(),
            "rmse_physical_magnetizations": float(m.get("rmse_physical_magnetizations", np.nan)),
            "rmse_physical_correlations": float(m.get("rmse_physical_correlations", np.nan)),
            # unita' fisiche, come per i modelli: mescolare scalato e fisico nella
            # stessa colonna renderebbe la tabella incomparabile riga per riga
            "nextstep_rmse": float(bl_ns.get(name, {}).get(
                "rmse_physical", bl_ns.get(name, {}).get("rmse", np.nan))),
            "skill_vs_baseline": skill_score(m["rollout_rmse_physical"], bb_val),
        }
        print(f"  {name:14s} rollout RMSE = {m['rollout_rmse_physical']:.4f}")
    print(f"  baseline di riferimento: {bb_name} ({bb_val:.4f})")

    schedule = args.schedule
    train_cfg_kwargs = {"schedule": schedule, "mask_prob": args.mask_prob,
                        "mask_loss_weight": args.mask_loss_weight,
                        "mask_mode": args.mask_mode, "ss_mode": args.ss_mode,
                        "ss_passes": args.ss_passes}

    grid = default_hp_grid()
    entries = {}
    trained = {}
    for kind in ("rnn", "transformer"):
        for hp in grid[kind]:
            name = hp["name"]
            print(f"\n[{kind} {name}] {hp_label(kind, hp)}")
            set_global_seeds(args.seed)
            model = build_model(kind, data.X_train.shape[-1], hp["model"])
            tcfg = TrainConfig(batch_size=args.batch_size, lr=hp["lr"], seed=args.seed,
                               verbose=True, **train_cfg_kwargs)
            trainer = HoldoutTrainer(model, tcfg)
            t0 = time.time()
            history, info = trainer.fit_monitored(
                data.X_train, data.Y_train, data.X_val, data.Y_val,
                X_train_eval=data.X_train_eval, Y_train_eval=data.Y_train_eval,
                ctx_val=data.ctx_val, fut_val=data.fut_val,
                ctx_train=data.ctx_train_eval, fut_train=data.fut_train_eval,
                scaler=data.scaler, feature_idx=data.feature_idx,
                n_qubits=cfg.n_qubits, rollout_every=args.rollout_every,
                rollout_max_pairs=args.rollout_monitor_pairs,
                restore_best=not args.no_restore_best)

            # Valutazione finale, con i pesi dell'epoca migliore
            from training import evaluate_nextstep, arrays_to_dataset
            ns = evaluate_nextstep(model, arrays_to_dataset(
                data.X_val, data.Y_val, args.batch_size, shuffle=False),
                scaler=data.scaler)
            ro = evaluate_rollout_detailed(model, data.ctx_val, data.fut_val,
                                           scaler=data.scaler,
                                           feature_idx=data.feature_idx,
                                           n_qubits=cfg.n_qubits, baselines=bl_ro,
                                           n_boot=args.n_boot, seed=args.seed)
            val = {k: v for k, v in ro.items() if k != "per_pair_mse_physical"}
            val["per_horizon_rmse_physical"] = np.asarray(
                ro.get("per_horizon_rmse_physical", ro["per_horizon_rmse"])).tolist()
            val.pop("per_horizon_rmse", None)
            val["nextstep_mse"] = float(ns["mse"])
            val["nextstep_rmse"] = float(ns.get("rmse_physical", ns["rmse"]))
            val["rollout_rmse_physical_ci95"] = list(
                ro.get("rollout_rmse_physical_ci95", (np.nan, np.nan)))

            at_best = history[info["best_epoch"]] if info["best_epoch"] >= 0 else history[-1]
            entries[name] = {
                "name": name, "kind": kind, "label": hp_label(kind, hp),
                "lr": hp["lr"], "n_params": int(count_params(model)),
                "history": history, "info": info, "val": val,
                "at_best": {k: at_best[k] for k in
                            ("epoch", "train_mse", "val_mse", "gap_mse")},
                "train_seconds": round(time.time() - t0, 1),
                "diagnosi": diagnose_curve(history),
            }
            trained[name] = (model, hp)
            print(f"  -> epoca migliore {info['best_epoch']}, "
                  f"val MSE {at_best['val_mse']:.5f}, gap {at_best['gap_mse']:+.5f}, "
                  f"rollout RMSE {val['rollout_rmse_physical']:.4f} "
                  f"(skill {val.get('skill_vs_baseline', float('nan')):+.3f})")
            d = entries[name]["diagnosi"]
            print(f"  -> curva: {d['verdetto'].upper()}")
            for nota in d["note"]:
                print(f"       - {nota}")

    selected = {k: select_config(entries, k) for k in ("rnn", "transformer")}
    print(f"\n[selezione] RNN -> {selected['rnn']} | "
          f"Transformer -> {selected['transformer']}")

    # Salvataggio dei modelli selezionati. I pesi in memoria sono gia' quelli
    # dell'epoca con validation minima, ripristinati alla fine dell'addestramento.
    models_dir = os.path.join(out_dir, "models")
    from training import plot_rollout
    for kind, name in selected.items():
        if not name or name not in trained:
            continue
        model, hp = trained[name]
        extra = {
            "protocol": "holdout_80_20", "val_frac": args.val_frac,
            "input_window": L, "horizon": int(data.fut_val.shape[1]),
            "origins": list(data.origins), "feature_names": data.feature_names,
            "scaler": data.scaler.to_dict(),
            "train_traj_ids": data.train_ids.tolist(),
            "val_traj_ids": data.val_ids.tolist(),
            "best_epoch": entries[name]["info"]["best_epoch"],
            "hp": {"name": name, "lr": hp["lr"]},
            "train_cfg": {k: v for k, v in train_cfg_kwargs.items()},
        }
        save_model_bundle(model, kind, data.X_train.shape[-1], hp["model"], models_dir,
                          tag=f"best_{kind}_{name}_L{L}", extra=extra)
        try:
            plot_rollout(model, data.ctx_val, data.fut_val, data.time_grid,
                         data.feature_names,
                         os.path.join(out_dir, f"fig_rollout_{kind}_{name}_L{L}.jpg"),
                         traj_idx=0, origin=data.origins[0])
        except Exception as exc:
            print(f"  [figura] rollout {name} non prodotta: {exc}")

    # Figura illustrativa dei tre regimi (serve al report, non dipende dai risultati)
    try:
        from training import plot_regimes_illustration
        plot_regimes_illustration(
            data.X_train, os.path.join(out_dir, f"fig_regimes_L{L}.jpg"),
            TrainConfig(**train_cfg_kwargs))
    except Exception as exc:
        print(f"  [figura] regimi non prodotta: {exc}")

    ablation = None
    if args.ablation and (args.ablation_window in (None, L)):
        ablation = run_ablation(L, args, data, bl_ro, selected, grid, out_dir,
                                train_cfg_kwargs)

    results = {
        "window": L,
        "horizon": data.horizon,
        "ablation": ablation,
        "protocol": "holdout_80_20",
        "val_frac": args.val_frac,
        "n_train_trajectories": int(len(data.train_ids)),
        "n_val_trajectories": int(len(data.val_ids)),
        "schedule": schedule,
        "configs": entries,
        "baselines": baselines,
        "selected": selected,
        "best_baseline": bb_name,
        "target_sigma_physical": float(
            entries[selected["rnn"]]["val"].get("target_sigma_physical", np.nan))
        if selected["rnn"] else None,
    }

    figs = make_all_figures(results, out_dir)
    print(f"[figure] {len(figs)} salvate in {out_dir}")
    csvp, texp = build_tables(results, out_dir, L)
    print(f"[tabelle] {os.path.basename(csvp)}, {os.path.basename(texp)}")

    dump_json(json.loads(json.dumps(results, default=_json_default)),
              os.path.join(out_dir, f"results_holdout_L{L}.json"))
    return results, data


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="percorso di trajectories.csv")
    p.add_argument("--out-dir", default="../artifacts/holdout")
    p.add_argument("--windows", type=int, nargs="+", default=[50, 100])
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--origins", nargs="+", default=[0, 250, 500])
    p.add_argument("--epochs", nargs=3, type=int, default=[10, 10, 5],
                   metavar=("TF", "MM", "SS"),
                   help="epoche di teacher forcing, masked modeling, scheduled sampling")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=None)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--mask-loss-weight", type=float, default=3.0)
    p.add_argument("--mask-mode", default="reconstruct", choices=["denoise_next", "reconstruct"])
    p.add_argument("--ss-mode", default="true", choices=["true", "two_pass"])
    p.add_argument("--ss-passes", type=int, default=3)
    p.add_argument("--rollout-every", type=int, default=1,
                   help="ogni quante epoche misurare il rollout di validation (0 = mai)")
    p.add_argument("--rollout-monitor-pairs", type=int, default=40)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-restore-best", action="store_true",
                   help="tiene i pesi dell'ultima epoca invece di quelli dell'epoca migliore")
    p.add_argument("--feature-set", default="all")
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--n-points", type=int, default=1001)
    p.add_argument("--n-qubits", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablation", action="store_true",
                   help="esegue anche l'ablation dello schedule sulla configurazione "
                        "selezionata di ogni famiglia (5 varianti a budget costante)")
    p.add_argument("--ablation-window", type=int, default=None,
                   help="finestra su cui eseguire l'ablation (default: tutte quelle richieste)")
    p.add_argument("--quick", action="store_true",
                   help="prova rapida: poche epoche, orizzonte corto, una sola origine")
    args = p.parse_args()
    args.origins = [0 if str(o) in ("0", "start") else int(o) for o in args.origins]
    if args.quick:
        args.epochs = [2, 1, 1]
        args.horizon = min(args.horizon, 20)
        args.origins = [0]
        args.rollout_monitor_pairs = 8
        args.n_boot = 100
    args.schedule = [("teacher_forcing", args.epochs[0]),
                     ("masked_modeling", args.epochs[1]),
                     ("scheduled_sampling", args.epochs[2])]
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_global_seeds(args.seed)
    dump_json(reproducibility_report(args.seed),
              os.path.join(args.out_dir, "reproducibility.json"))

    # Tavola di riferimento dei sei casi canonici: non dipende dai risultati e
    # serve da chiave di lettura accanto alle curve vere.
    plot_curve_taxonomy(os.path.join(args.out_dir, "fig_come_leggere_le_curve.jpg"))

    # Il CSV viene letto e validato una sola volta e riusato per tutte le finestre.
    cfg0 = PreprocessConfig(csv_path=args.csv, n_points=args.n_points,
                            n_qubits=args.n_qubits, seed=args.seed)
    traj, time_grid, starts = load_raw_trajectories(cfg0)
    report = validate_trajectories(traj, time_grid, starts, cfg0,
                                   strict=not args.quick)
    raw_cache = (traj, time_grid, report)

    summary = {}
    per_window = {}
    for L in args.windows:
        res, _ = run_window(L, args, raw_cache)
        per_window[f"L{L}"] = res
        summary[f"L{L}"] = {
            "selected": res["selected"],
            "best_baseline": res["best_baseline"],
            "val_rollout_rmse": {n: res["configs"][n]["val"]["rollout_rmse_physical"]
                                 for n in res["configs"]},
            "val_mse_at_best": {n: res["configs"][n]["at_best"]["val_mse"]
                                for n in res["configs"]},
            "best_epoch": {n: res["configs"][n]["info"]["best_epoch"]
                           for n in res["configs"]},
            "diagnosi_curve": {n: res["configs"][n]["diagnosi"]["verdetto"]
                               for n in res["configs"]},
        }
    dump_json(summary, os.path.join(args.out_dir, "summary_holdout.json"))
    txt, tex = write_report_numbers(per_window, args.out_dir)
    print(f"\nFatto. Riepilogo in summary_holdout.json")
    print(f"Numeri per il report: {os.path.basename(txt)} (e {os.path.basename(tex)})")


if __name__ == "__main__":
    main()
