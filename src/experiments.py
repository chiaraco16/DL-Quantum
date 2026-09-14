"""
DL-Quantum -- Step 3: sweep degli iperparametri, cross-validation e ablation.

Ogni configurazione di ogni famiglia, per ogni finestra temporale, viene valutata in
cross-validation sulle traiettorie; di ciascuna metrica si riportano media e intervallo
di confidenza al 95% calcolati sui fold.

Scelte che rendono confrontabili i numeri prodotti:
* i fold partizionano le traiettorie, non le finestre, quindi nessuna finestra di
  validazione proviene da una simulazione vista in addestramento;
* lo scaler viene ristimato dentro ogni fold sui soli dati di training, cosi' che le
  statistiche del validation non entrino nella normalizzazione;
* le baseline banali sono ricalcolate sugli stessi fold dei modelli, non su una
  partizione diversa;
* il rollout parte da piu' origini temporali, per non giudicare i modelli sul solo
  transitorio iniziale;
* la selezione confronta gli intervalli di confidenza e dichiara esplicitamente quando
  due configurazioni non sono distinguibili;
* le metriche sono riportate anche in unita' fisiche e separate per gruppo di
  osservabili, per far emergere eventuali squilibri fra magnetizzazioni e correlazioni.

Produce: tabelle CSV e LaTeX, risultati per fold in JSON, figure a 300 dpi e i bundle
dei modelli migliori con i rispettivi metadati.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os

import numpy as np

from data_preprocessing import (
    N_POINTS_DEFAULT, PreprocessConfig, build_datasets, set_global_seeds, FeatureScaler,
    feature_group_ids, kfold_trajectory_indices, window_split, make_forecast_pairs,
    reproducibility_report, dump_json,
)
from models import (RNNConfig, TransformerConfig, build_model, count_params,
                    save_model_bundle)
from training import (TrainConfig, MultiRegimeTrainer, arrays_to_dataset,
                      evaluate_nextstep, evaluate_rollout)
from baselines import (evaluate_rollout_baselines, evaluate_nextstep_baselines,
                       skill_score, best_baseline)



# Griglia degli iperparametri: domini espliciti, prodotto cartesiano completo.
#
# I due fattori variati non sono gli stessi nelle due famiglie, ed e' una scelta, non
# una svista. Per la RNN il secondo fattore e' il TIPO DI CELLA (LSTM contro GRU) a
# capacita' fissata: le due celle differiscono per come regolano la memoria, ed e' la
# domanda piu' interessante a parita' di parametri. Per il Transformer, dove la cella
# non esiste, il secondo fattore e' la CAPACITA' del modello (d_model). In entrambi i
# casi il primo fattore e' il learning rate. Numero di teste e di layer restano fissi
# per non moltiplicare il costo dello sweep.
HP_RNN_TYPE = ["LSTM", "GRU"]
HP_RNN_UNITS = [64]
HP_RNN_LR = [1e-2, 1e-3]

HP_D_MODEL = [32, 64]
HP_TR_LR = [1e-2, 1e-3]
HP_NUM_HEADS = 4
HP_NUM_LAYERS = 2


def default_hp_grid(full_grid: bool = False):
    """Genera il prodotto cartesiano completo dei domini di iperparametri."""
    grid = {"rnn": [], "transformer": []}
    units_domain = [32, 64] if full_grid else HP_RNN_UNITS

    i = 0
    for rnn_type in HP_RNN_TYPE:
        for units in units_domain:
            for lr in HP_RNN_LR:
                i += 1
                grid["rnn"].append({
                    "name": f"R{i}",
                    "lr": lr,
                    "model": RNNConfig(rnn_type=rnn_type, units=units, num_layers=1),
                })

    j = 0
    for d_model in HP_D_MODEL:
        for lr in HP_TR_LR:
            j += 1
            grid["transformer"].append({
                "name": f"T{j}",
                "lr": lr,
                "model": TransformerConfig(d_model=d_model,
                                           num_heads=HP_NUM_HEADS,
                                           num_layers=HP_NUM_LAYERS,
                                           dff=2 * d_model),
            })
    return grid


def describe_hp_grid(grid=None) -> str:
    grid = grid or default_hp_grid()
    lines = []
    for kind in ("rnn", "transformer"):
        for hp in grid[kind]:
            m = hp["model"]
            if kind == "rnn":
                desc = f"{m.rnn_type}, unità {m.units}, layer {m.num_layers}"
            else:
                desc = (f"d_model {m.d_model}, teste {m.num_heads}, "
                        f"layer {m.num_layers}, dff {m.dff}")
            lines.append(f"{kind:11s} {hp['name']:4s} {desc}, lr {hp['lr']:g}")
    return "\n".join(lines)


def hp_grid_latex(grid=None, path=None, schedule=None, windows=None, folds=None,
                  batch_size=64, clipnorm=1.0) -> str:
    """Tabella LaTeX delle configurazioni di iperparametri.

    Gli elementi condivisi riportati in didascalia sono passati dal chiamante e non
    scritti a mano: se cambia lo schedule o l'insieme delle finestre, la didascalia
    cambia con loro invece di restare indietro rispetto agli esperimenti.
    """
    grid = grid or default_hp_grid()
    rows = []
    for kind, label in (("rnn", "RNN"), ("transformer", "Transformer")):
        n = len(grid[kind])
        for k, hp in enumerate(grid[kind]):
            m = hp["model"]
            if kind == "rnn":
                desc = (f"{m.rnn_type}, units ${m.units}$, layers ${m.num_layers}$, "
                        f"lr ${_sci(hp['lr'])}$")
            else:
                desc = (f"$d_{{\\text{{model}}}}\\,{m.d_model}$, heads ${m.num_heads}$, "
                        f"layers ${m.num_layers}$, dff ${m.dff}$, lr ${_sci(hp['lr'])}$")
            first = f"\\multirow{{{n}}}{{*}}{{{label}}}" if k == 0 else ""
            rows.append(f"{first} & {hp['name']} & {desc} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    shared = [f"Adam, batch ${batch_size}$", f"clipnorm ${clipnorm:g}$",
              "per-feature z-score"]
    if schedule:
        shared.append("schedule $" + "{+}".join(str(n) for _, n in schedule) + "$")
    if windows:
        shared.append("windows $L\\in\\{" + ",".join(str(w) for w in windows) + "\\}$")
    if folds:
        shared.append(f"${folds}$-fold CV")
    out = ["% richiede \\usepackage{booktabs, multirow}",
           "\\begin{table}[t]\\centering",
           "\\caption{Hyper-parameter configurations: full Cartesian product of the "
           "declared domains. Shared by every configuration: "
           + ", ".join(shared) + ".}",
           "\\label{tab:hpcfg}",
           "\\begin{tabular}{lll}", "\\toprule",
           "Model & Config & Hyper-parameters \\\\", "\\midrule",
           *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    txt = "\n".join(out)
    if path:
        with open(path, "w") as f:
            f.write(txt)
    return txt


def _sci(x: float) -> str:
    e = int(round(math.log10(x)))
    return f"10^{{{e}}}" if abs(x - 10 ** e) < 1e-12 else f"{x:g}"



#Registro delle metriche valutate

METRICS = [
    ("nextstep_rmse",     "Next-step RMSE (scaled)",            True),
    ("nextstep_mae",      "Next-step MAE (scaled)",             True),
    ("rollout_rmse_phys", "Rollout RMSE (phys)",                True),
    ("rollout_mae_phys",  "Rollout MAE (phys)",                 True),
    ("rollout_nrmse",     "Rollout NRMSE (RMSE/$\\sigma$)",      True),
    ("skill_vs_baseline", "Skill score vs baseline",           False),
    #Le etichette finiscono anche nelle tabelle CSV: niente virgole nel testo.
    ("rollout_rmse_mag",  "Rollout RMSE mag (phys)",            True),
    ("rollout_rmse_corr", "Rollout RMSE corr (phys)",           True),
]
METRIC_KEYS = [k for k, _, _ in METRICS]
LOWER_IS_BETTER = {k: lb for k, _, lb in METRICS}

#Valori critici della t di Student a due code (95%)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 24: 2.064, 29: 2.045}


def _ci95(vals):
    """Calcola la media e la semi-ampiezza dell'intervallo di confidenza al 95%."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
    m = float(v.mean())
    if n < 2:
        return m, 0.0
    s = float(v.std(ddof=1))
    t = _T95.get(n - 1, 1.96)
    return m, t * s / math.sqrt(n)


def _overlap(a, b) -> bool:
    """Verifica se i due intervalli di confidenza si sovrappongono."""
    lo_a, hi_a = a[0] - a[1], a[0] + a[1]
    lo_b, hi_b = b[0] - b[1], b[0] + b[1]
    return not (hi_a < lo_b or hi_b < lo_a)


def _select_best(cv_kind: dict, select_on: str, tie_break_on: str) -> tuple:
    """Selezione della configurazione migliore in due stadi.

    1) si ordina per la metrica di selezione `select_on` (media sui fold);
    2) fra le configurazioni il cui IC al 95% si sovrappone a quello del leader
       -- cioe' statisticamente indistinguibili sul rollout -- si sceglie quella
       col miglior `tie_break_on` (di default il next-step RMSE).

    A parita' statistica sul rollout si preferisce cosi' il modello che ha appreso
    meglio la dinamica a un passo. Ritorna (best_name, significance); significance
    confronta il best selezionato con la migliore delle altre configurazioni
    secondo `select_on`, cosi' che `best` e `significance['best']` coincidano sempre.
    """
    def stat(name, key):
        a = cv_kind[name]["agg"][key]
        return (a["mean"], a["ci95"])

    lower = LOWER_IS_BETTER[select_on]
    ranked = sorted(cv_kind, key=lambda n: stat(n, select_on)[0], reverse=not lower)
    leader = ranked[0]
    tied = [n for n in ranked if _overlap(stat(n, select_on), stat(leader, select_on))]
    tb_lower = LOWER_IS_BETTER.get(tie_break_on, True)
    best = (min if tb_lower else max)(tied, key=lambda n: stat(n, tie_break_on)[0])

    sig = {"best": best, "leader": leader, "tie_broken": bool(best != leader),
           "tie_candidates": list(tied)}
    others = [n for n in ranked if n != best]
    if others:
        second = others[0]
        overlap = _overlap(stat(best, select_on), stat(second, select_on))
        sig.update({"second": second, "ci_overlap": bool(overlap),
                    "significant": bool(not overlap)})
    return best, sig


#Funzioni di supporto per training e valutazione

def _pack_metrics(ns, ro) -> dict:
    return {
        "nextstep_rmse": float(ns["rmse"]),
        "nextstep_mae": float(ns["mae"]),
        "rollout_rmse_phys": float(ro.get("rollout_rmse_physical", ro["rollout_rmse"])),
        "rollout_mae_phys": float(ro.get("rollout_mae_physical", ro["rollout_mae"])),
        "rollout_nrmse": float(ro.get("rollout_nrmse", np.nan)),
        "skill_vs_baseline": float(ro.get("skill_vs_baseline", np.nan)),
        "rollout_rmse_mag": float(ro.get("rmse_physical_magnetizations", np.nan)),
        "rollout_rmse_corr": float(ro.get("rmse_physical_correlations", np.nan)),
    }


def _train_eval(kind, model_cfg, lr, train_cfg_kwargs, Xtr, Ytr, Xva, Yva,
                ctx, fut, scaler, feature_idx, n_qubits, seed, batch_size=64,
                fold_baselines=None):
    set_global_seeds(seed)
    F = Xtr.shape[-1]
    model = build_model(kind, F, model_cfg)
    tcfg = TrainConfig(batch_size=batch_size, lr=lr, seed=seed, verbose=False,
                       **train_cfg_kwargs)
    hist = MultiRegimeTrainer(model, tcfg).fit(Xtr, Ytr, Xva, Yva)
    ns = evaluate_nextstep(model, arrays_to_dataset(Xva, Yva, batch_size, shuffle=False),
                           scaler=scaler)
    ro = evaluate_rollout(model, ctx, fut, scaler=scaler, feature_idx=feature_idx,
                          n_qubits=n_qubits, baselines=fold_baselines)
    return _pack_metrics(ns, ro), model, hist, ro


def _make_fold(prep, tr_ids, va_ids, origins):
    """Crea un fold adattando lo scaler esclusivamente sui dati di training."""
    L = prep.cfg.input_window
    P = prep.traj_selected.shape[1]
    H = min(prep.cfg.horizon, P - L)
    gid = feature_group_ids(prep.feature_idx, prep.cfg.n_qubits)
    raw = prep.traj_selected
    scaler = FeatureScaler(prep.cfg.scaler_type, group_ids=gid).fit(raw[tr_ids])
    Xtr, Ytr = window_split(scaler.transform(raw[tr_ids]), prep.cfg,
                            stride=prep.cfg.resolved_train_stride())
    Xva, Yva = window_split(scaler.transform(raw[va_ids]), prep.cfg,
                            stride=prep.cfg.resolved_stride())
    ctx, fut = make_forecast_pairs(scaler.transform(raw[va_ids]), L, H, origin=origins)
    return scaler, Xtr, Ytr, Xva, Yva, ctx, fut


def run_cv_baselines(prep, folds, seed, origins, verbose=True):
    """Baseline banali valutate sugli stessi fold usati per i modelli.

    Lo skill score di ogni baseline e' calcolato rispetto alla migliore fra le baseline
    dello stesso fold: quella di riferimento ha per costruzione skill nullo, e il valore
    diventa leggibile sulla stessa scala usata per i modelli.
    """
    dev_ids = np.sort(np.concatenate([prep.splits["train"], prep.splits["val"]]))
    per_fold = {}
    for k, tr_ids, va_ids in kfold_trajectory_indices(dev_ids, folds, seed):
        scaler, Xtr, Ytr, Xva, Yva, ctx, fut = _make_fold(prep, tr_ids, va_ids, origins)
        ro = evaluate_rollout_baselines(ctx, fut, scaler=scaler,
                                        feature_idx=prep.feature_idx,
                                        n_qubits=prep.cfg.n_qubits)
        ns = evaluate_nextstep_baselines(Xva, Yva, scaler=scaler)
        bb_name, bb_val = best_baseline(ro)
        for name, m in ro.items():
            nsm = ns.get(name, {})
            rec = {
                "nextstep_rmse": float(nsm.get("rmse", np.nan)),
                "nextstep_mae": float(nsm.get("mae", np.nan)),
                "rollout_rmse_phys": float(m["rollout_rmse_physical"]),
                "rollout_mae_phys": float(m["rollout_mae_physical"]),
                "rollout_nrmse": float(m.get("rollout_nrmse", np.nan)),
                "skill_vs_baseline": skill_score(m["rollout_rmse_physical"], bb_val),
                "rollout_rmse_mag": float(m.get("rmse_physical_magnetizations", np.nan)),
                "rollout_rmse_corr": float(m.get("rmse_physical_correlations", np.nan)),
                "fold": k,
            }
            per_fold.setdefault(name, []).append(rec)
    agg = {}
    for name, recs in per_fold.items():
        agg[name] = {"per_fold": recs,
                     "agg": {key: dict(zip(("mean", "ci95"),
                                           _ci95([r[key] for r in recs])))
                             for key in METRIC_KEYS}}
    if verbose:
        for name in sorted(agg, key=lambda n: agg[n]["agg"]["rollout_rmse_phys"]["mean"]):
            a = agg[name]["agg"]
            print(f"  [baseline    {name:13s}] rollout_rmse_phys(CV) = "
                  f"{a['rollout_rmse_phys']['mean']:.4f} +/- {a['rollout_rmse_phys']['ci95']:.4f}"
                  f"   NRMSE = {a['rollout_nrmse']['mean']:.3f}")
    return agg


def run_cv_for_config(kind, hp, prep, folds, train_cfg_kwargs, seed,
                      origins, batch_size=64):
    dev_ids = np.sort(np.concatenate([prep.splits["train"], prep.splits["val"]]))
    per_fold = []
    for k, tr_ids, va_ids in kfold_trajectory_indices(dev_ids, folds, seed):
        scaler, Xtr, Ytr, Xva, Yva, ctx, fut = _make_fold(prep, tr_ids, va_ids, origins)
        fold_bl = evaluate_rollout_baselines(ctx, fut, scaler=scaler,
                                             feature_idx=prep.feature_idx,
                                             n_qubits=prep.cfg.n_qubits)
        met, _, _, _ = _train_eval(kind, hp["model"], hp["lr"], train_cfg_kwargs,
                                   Xtr, Ytr, Xva, Yva, ctx, fut, scaler,
                                   prep.feature_idx, prep.cfg.n_qubits, seed,
                                   batch_size, fold_baselines=fold_bl)
        met["fold"] = k
        per_fold.append(met)
    agg = {key: dict(zip(("mean", "ci95"), _ci95([f[key] for f in per_fold])))
           for key in METRIC_KEYS}
    return per_fold, agg


def train_final_and_test(kind, hp, prep, train_cfg_kwargs, seed, out_models_dir,
                         batch_size=64, save_extra=None):
    """Riaddestra la configurazione selezionata e la valuta una sola volta sul test.

    La cross-validation gira sull'unione train+val e serve a scegliere la configurazione;
    il modello finale viene poi addestrato sulla sola partizione di train, con la
    partizione di validation usata per registrare la curva di apprendimento. Il test set
    non entra in nessuna decisione: viene letto solo qui, a scelta gia' fatta.
    """
    set_global_seeds(seed)
    F = prep.X_train.shape[-1]
    model = build_model(kind, F, hp["model"])
    tcfg = TrainConfig(batch_size=batch_size, lr=hp["lr"], seed=seed, verbose=False,
                       **train_cfg_kwargs)
    hist = MultiRegimeTrainer(model, tcfg).fit(prep.X_train, prep.Y_train,
                                               prep.X_val, prep.Y_val)
    test_bl = evaluate_rollout_baselines(prep.ctx_test, prep.fut_test,
                                         scaler=prep.scaler,
                                         feature_idx=prep.feature_idx,
                                         n_qubits=prep.cfg.n_qubits)
    ns = evaluate_nextstep(model, arrays_to_dataset(prep.X_test, prep.Y_test,
                                                    batch_size, shuffle=False),
                           scaler=prep.scaler)
    ro = evaluate_rollout(model, prep.ctx_test, prep.fut_test, scaler=prep.scaler,
                          feature_idx=prep.feature_idx, n_qubits=prep.cfg.n_qubits,
                          baselines=test_bl)
    if out_models_dir:
        extra = {"input_window": prep.cfg.input_window,
                 "horizon": int(prep.fut_test.shape[1]),
                 "origins": list(prep.origins),
                 "feature_names": prep.feature_names,
                 "scaler": prep.scaler.to_dict(),
                 "test_traj_ids": prep.splits["test"].tolist(),
                 "hp": {"name": hp["name"], "lr": hp["lr"]},
                 "train_cfg": train_cfg_kwargs}
        if save_extra:
            extra.update(save_extra)
        save_model_bundle(model, kind, F, hp["model"], out_models_dir,
                          tag=f"best_{kind}_{hp['name']}_L{prep.cfg.input_window}",
                          extra=extra)
    out = _pack_metrics(ns, ro)
    out["per_horizon_rmse"] = np.asarray(ro["per_horizon_rmse"]).tolist()
    out["per_horizon_rmse_physical"] = np.asarray(
        ro.get("per_horizon_rmse_physical", ro["per_horizon_rmse"])).tolist()
    out["baselines_test"] = {n: {"rollout_rmse_phys": float(m["rollout_rmse_physical"]),
                                 "rollout_nrmse": float(m.get("rollout_nrmse", np.nan)),
                                 "per_horizon_rmse_physical": np.asarray(
                                     m.get("per_horizon_rmse_physical",
                                           m["per_horizon_rmse"])).tolist()}
                             for n, m in test_bl.items()}
    out["history"] = hist
    return out, model


def run_full_sweep(csv_path, windows, folds, schedule, feature_set="all",
                   max_trajectories=None, horizon=200, seed=42, out_dir=".",
                   hp_grid=None, batch_size=64, origins=("start",),
                   train_stride=None, select_on="rollout_rmse_phys",
                   tie_break_on="nextstep_rmse",
                   train_cfg_kwargs=None, verbose=True):
    hp_grid = hp_grid or default_hp_grid()
    train_cfg_kwargs = dict(train_cfg_kwargs or {})
    train_cfg_kwargs["schedule"] = schedule
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "models")
    results = {"config": {"windows": list(windows), "folds": folds,
                          "schedule": schedule, "feature_set": feature_set,
                          "seed": seed, "origins": list(origins),
                          "select_on": select_on,
                          "tie_break_on": tie_break_on,
                          "train_cfg": {k: v for k, v in train_cfg_kwargs.items()
                                        if k != "schedule"},
                          "reproducibility": reproducibility_report(seed)},
               "windows": {}}

    for L in windows:
        if verbose:
            print(f"\n########## FINESTRA TEMPORALE L={L} ##########")
        max_rows = max_trajectories * N_POINTS_DEFAULT if max_trajectories else None
        prep = build_datasets(PreprocessConfig(
            csv_path=csv_path, input_window=L, horizon=horizon, feature_set=feature_set,
            max_trajectories=max_trajectories, n_folds=folds, seed=seed,
            max_rows=max_rows, rollout_origins=list(origins),
            train_stride=train_stride), verbose=False)
        eff_origins = prep.origins
        if verbose:
            print(f"  origini di rollout effettive: {eff_origins}")

        os.makedirs(models_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(models_dir, f"test_pack_L{L}.npz"),
            X_test=prep.X_test, Y_test=prep.Y_test,
            ctx_test=prep.ctx_test, fut_test=prep.fut_test,
            time_grid=prep.time_grid, feature_idx=prep.feature_idx,
            scaler_mean=prep.scaler.mean_, scaler_std=prep.scaler.std_,
            test_traj_ids=prep.splits["test"],
            n_qubits=prep.cfg.n_qubits, origins=np.asarray(prep.origins),
            feature_names=np.array(prep.feature_names))

        wres = {"cv": {"rnn": {}, "transformer": {}}, "final": {}, "best": {},
                "significance": {}}

        wres["baselines_cv"] = run_cv_baselines(prep, folds, seed, eff_origins,
                                                verbose=verbose)

        for kind in ["rnn", "transformer"]:
            for hp in hp_grid[kind]:
                per_fold, agg = run_cv_for_config(kind, hp, prep, folds,
                                                  train_cfg_kwargs, seed,
                                                  eff_origins, batch_size)
                wres["cv"][kind][hp["name"]] = {
                    "hp": {"lr": hp["lr"], "model": asdict(hp["model"]),
                           "params": count_params(build_model(
                               kind, len(prep.feature_idx), hp["model"]))},
                    "per_fold": per_fold, "agg": agg}
                if verbose:
                    m = agg[select_on]
                    print(f"  [{kind:11s} {hp['name']}] {select_on}(CV) = "
                          f"{m['mean']:.4f} +/- {m['ci95']:.4f}   "
                          f"NRMSE = {agg['rollout_nrmse']['mean']:.3f}   "
                          f"skill = {agg['skill_vs_baseline']['mean']:+.3f}")

            best_name, sig = _select_best(wres["cv"][kind], select_on, tie_break_on)
            wres["best"][kind] = best_name
            wres["significance"][kind] = sig
            if verbose and sig.get("tie_broken"):
                print(f"  ~~ {kind}: IC sovrapposti su {select_on}; tie-break su "
                      f"{tie_break_on} -> scelto {best_name} "
                      f"(leader per media: {sig['leader']}).")
            if verbose and sig.get("ci_overlap"):
                print(f"  !! {kind}: {best_name} e {sig.get('second')} hanno IC al 95% "
                      f"SOVRAPPOSTI su {select_on}: differenza non statisticamente "
                      f"significativa.")

            best_hp = next(h for h in hp_grid[kind] if h["name"] == best_name)
            final, _ = train_final_and_test(kind, best_hp, prep, train_cfg_kwargs,
                                            seed, models_dir, batch_size)
            wres["final"][kind] = final
            if verbose:
                print(f"  -> best {kind}: {best_name} | TEST rollout_rmse_phys = "
                      f"{final['rollout_rmse_phys']:.4f}  NRMSE = "
                      f"{final['rollout_nrmse']:.3f}  skill = "
                      f"{final['skill_vs_baseline']:+.3f}")

        results["windows"][str(L)] = wres

    dump_json(results, os.path.join(out_dir, "results_full.json"))
    return results


def build_ablation_variants(schedule, ss_passes=3):
    """Varianti dello schedule confrontate a parità di budget di epoche.

    Tutte le varianti consumano lo stesso numero totale di epoche dello schedule di
    riferimento: le epoche sottratte a un regime vengono redistribuite sugli altri. Senza
    questo vincolo la variante piu' ricca vincerebbe semplicemente perche' si addestra di
    piu'. L'ablation misura quindi il contributo dei regimi e della loro variante
    implementativa, non l'ordine in cui compaiono ne' la ripartizione delle epoche fra
    loro, che restano fissati.
    """
    d = dict(schedule)
    a = d.get("teacher_forcing", 0)
    b = d.get("masked_modeling", 0)
    c = d.get("scheduled_sampling", 0)
    total = a + b + c
    half = c // 2
    return {
        "TF only": (dict(schedule=[("teacher_forcing", total)]), {}),
        "TF + MM": (dict(schedule=[("teacher_forcing", a + half),
                                   ("masked_modeling", b + (c - half))]), {}),
        #I nomi delle varianti finiscono nella prima colonna della tabella di ablation
        #del documento finale: restano in inglese come il resto delle etichette.
        "TF + MM + SS (1 pass)": (dict(schedule=list(schedule)),
                                  dict(ss_mode="two_pass")),
        f"TF + MM + SS ({ss_passes} passes)": (
            dict(schedule=list(schedule)),
            dict(ss_mode="true", ss_passes=ss_passes)),
        f"TF + MM reconstruct + SS ({ss_passes} passes)": (
            dict(schedule=list(schedule)),
            dict(ss_mode="true", ss_passes=ss_passes, mask_mode="reconstruct")),
    }


def run_regime_ablation(csv_path, window, folds, schedule, feature_set="all",
                        max_trajectories=None, horizon=200, seed=42, out_dir=".",
                        hp_grid=None, batch_size=64, origins=("start",),
                        best_names=None, ss_passes=3, verbose=True):
    """Ablation dello schedule multi-regime sulla configurazione migliore di ogni famiglia."""
    os.makedirs(out_dir, exist_ok=True)
    hp_grid = hp_grid or default_hp_grid()
    total_epochs = sum(n for _, n in schedule)
    variants = build_ablation_variants(schedule, ss_passes=ss_passes)

    max_rows = max_trajectories * N_POINTS_DEFAULT if max_trajectories else None
    prep = build_datasets(PreprocessConfig(
        csv_path=csv_path, input_window=window, horizon=horizon,
        feature_set=feature_set, max_trajectories=max_trajectories, n_folds=folds,
        seed=seed, max_rows=max_rows, rollout_origins=list(origins)), verbose=False)
    eff_origins = prep.origins

    out = {"window": window, "total_epochs": total_epochs, "variants": {}}
    for kind in ["rnn", "transformer"]:
        name = (best_names or {}).get(kind) or hp_grid[kind][0]["name"]
        hp = next(h for h in hp_grid[kind] if h["name"] == name)
        out["variants"][kind] = {"config": name, "results": {}}
        for vname, (sched_kwargs, extra) in variants.items():
            kwargs = dict(sched_kwargs)
            kwargs.update(extra)
            per_fold, agg = run_cv_for_config(kind, hp, prep, folds, kwargs, seed,
                                              eff_origins, batch_size)
            out["variants"][kind]["results"][vname] = {"per_fold": per_fold, "agg": agg}
            if verbose:
                a = agg["rollout_rmse_phys"]
                print(f"  [ablation {kind:11s} {vname:33s}] rollout_rmse_phys = "
                      f"{a['mean']:.4f} +/- {a['ci95']:.4f}  |  next-step RMSE = "
                      f"{agg['nextstep_rmse']['mean']:.4f}")
    dump_json(out, os.path.join(out_dir, "ablation_full.json"))
    return out


def _table_columns(wres, include_baselines=True):
    cols = []
    if include_baselines:
        for name in sorted(wres.get("baselines_cv", {})):
            cols.append(("baseline", name))
    for kind in ["rnn", "transformer"]:
        for name in sorted(wres["cv"][kind]):
            cols.append((kind, name))
    matrix = {key: [] for key in METRIC_KEYS}
    for key in METRIC_KEYS:
        for kind, name in cols:
            src = wres["baselines_cv"][name] if kind == "baseline" else wres["cv"][kind][name]
            a = src["agg"][key]
            matrix[key].append((_num(a["mean"]), _num(a["ci95"])))
    return cols, matrix


def _num(x):
    try:
        return float(x) if x is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _best_index(values, lower: bool):
    finite = [j for j, (m, _) in enumerate(values) if np.isfinite(m)]
    if not finite:
        return None
    return min(finite, key=lambda j: values[j][0]) if lower else \
        max(finite, key=lambda j: values[j][0])


def build_table1_csv(wres, path, include_baselines=True):
    cols, matrix = _table_columns(wres, include_baselines)
    header = ["metric"] + [f"{k.upper()}_{n}" for k, n in cols]
    lines = [",".join(header)]
    for key, disp, lower in METRICS:
        row = [disp.replace("$\\sigma$", "sigma")]
        best = _best_index(matrix[key], lower)
        for j, (m, ci) in enumerate(matrix[key]):
            star = "*" if j == best else ""
            row.append("nan" if not np.isfinite(m) else f"{m:.4f}+-{ci:.4f}{star}")
        lines.append(",".join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _tex(s: str) -> str:
    return str(s).replace("_", "-")


def _group_positions(groups) -> list:
    """Colonna iniziale di ciascun gruppo di intestazioni (la prima e' l'etichetta)."""
    pos, out = 2, []
    for _, n in groups:
        out.append(pos)
        pos += n
    return out


def build_table1_latex(wres, path, window, n_folds=None, include_baselines=True,
                       label=None):
    cols, matrix = _table_columns(wres, include_baselines)
    groups = []
    for kind, _ in cols:
        if not groups or groups[-1][0] != kind:
            groups.append([kind, 0])
        groups[-1][1] += 1
    colspec = "l" + "c" * len(cols)
    fold_txt = f", {n_folds}-fold CV" if n_folds else ""
    group_label = {"baseline": "Trivial baselines", "rnn": "RNN",
                   "transformer": "Transformer"}

    out = ["% richiede \\usepackage{booktabs, multirow, makecell, graphicx}",
           "\\begin{table*}[t]\\centering\\footnotesize",
           "\\setlength{\\tabcolsep}{2.5pt}\\renewcommand{\\arraystretch}{1.1}",
           f"\\caption{{Cross-validated results (mean $\\pm$ 95\\% CI{fold_txt}) for time "
           f"window $L={window}$; best per row in bold. NRMSE is the rollout RMSE divided by the "
           f"standard deviation of the target, so NRMSE $=1$ means an error as large as the "
           f"natural spread of the signal. The skill score is "
           f"$1-$RMSE$/$RMSE$_{{\\text{{best baseline}}}}$ and is positive only if the model "
           f"beats every trivial predictor.}}",
           f"\\label{{{label or f'tab:main{window}'}}}",
           "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
           " & ".join([""] + [f"\\multicolumn{{{n}}}{{c}}{{{group_label[kind]}}}" for kind, n in groups]) + " \\\\",
           # I \cmidrule seguono la riga di intestazione e non vanno chiusi da \\:
           # un secondo \\ aprirebbe una riga vuota fra i filetti e le etichette.
           "".join([f"\\cmidrule(lr){{{pos}-{pos + n - 1}}}"
                    for (kind, n), pos in zip(groups, _group_positions(groups))]),
           "Metric & " + " & ".join(_tex(n) for _, n in cols) + " \\\\",
           "\\midrule"]
    for key, disp, lower in METRICS:
        best = _best_index(matrix[key], lower)
        cells = []
        for j, (m, ci) in enumerate(matrix[key]):
            if not np.isfinite(m):
                cells.append("--")
                continue
            mean = f"\\textbf{{{m:.4f}}}" if j == best else f"{m:.4f}"
            cells.append(f"\\makecell{{{mean}\\\\[-2pt]"
                         f"{{\\scriptsize$\\pm${ci:.4f}}}}}")
        out.append(disp + " & " + " & ".join(cells) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}}")
    out.append("\\end{table*}")
    txt = "\n".join(out)
    with open(path, "w") as f:
        f.write(txt)
    return path


def build_ablation_table(ablation, path_csv, path_tex):
    variants = list(next(iter(ablation["variants"].values()))["results"].keys())
    lines = ["variant," + ",".join(
        f"{k}_{m}" for k in ablation["variants"] for m in
        ("nextstep_rmse", "rollout_rmse_phys", "skill"))]
    for v in variants:
        row = [v]
        for kind in ablation["variants"]:
            a = ablation["variants"][kind]["results"][v]["agg"]
            row += [f"{a['nextstep_rmse']['mean']:.4f}+-{a['nextstep_rmse']['ci95']:.4f}",
                    f"{a['rollout_rmse_phys']['mean']:.4f}+-{a['rollout_rmse_phys']['ci95']:.4f}",
                    f"{a['skill_vs_baseline']['mean']:+.3f}"]
        lines.append(",".join(row))
    with open(path_csv, "w") as f:
        f.write("\n".join(lines))

    kinds = list(ablation["variants"].keys())
    out = ["% richiede \\usepackage{booktabs}",
           "\\begin{table}[t]\\centering\\footnotesize",
           "\\setlength{\\tabcolsep}{4pt}",
           f"\\caption{{Ablation of the multi-regime schedule at constant budget "
           f"({ablation['total_epochs']} epochs, $L={ablation['window']}$, mean $\\pm$ 95\\% CI). "
           f"TF = teacher forcing, MM = masked modeling, SS = scheduled sampling.}}",
           "\\label{tab:ablation}",
           "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%",
           "\\begin{tabular}{l" + "cc" * len(kinds) + "}", "\\toprule",
           " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{k.upper()}}}" for k in kinds) + " \\\\",
           "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(kinds))),
           "Training schedule & " + " & ".join(
               "Next-step RMSE & Rollout RMSE (phys)" for _ in kinds) + " \\\\",
           "\\midrule"]
    best = {}
    for kind in kinds:
        r = ablation["variants"][kind]["results"]
        best[kind] = min(r, key=lambda v: r[v]["agg"]["rollout_rmse_phys"]["mean"])
    for v in variants:
        cells = []
        for kind in kinds:
            a = ablation["variants"][kind]["results"][v]["agg"]
            ns = f"{a['nextstep_rmse']['mean']:.4f} $\\pm$ {a['nextstep_rmse']['ci95']:.4f}"
            ro = f"{a['rollout_rmse_phys']['mean']:.4f} $\\pm$ {a['rollout_rmse_phys']['ci95']:.4f}"
            if v == best[kind]:
                ro = f"\\textbf{{{ro}}}"
            cells += [ns, ro]
        out.append(v.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    with open(path_tex, "w") as f:
        f.write("\n".join(out))
    return path_csv, path_tex


def plot_results(results, out_dir, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for L, wres in results["windows"].items():
        cols, matrix = _table_columns(wres, include_baselines=False)
        labels = [f"{k[:2].upper()}-{n}" for k, n in cols]
        colors = ["#1f77b4" if k == "rnn" else "#d62728" for k, _ in cols]

        means = [m for m, _ in matrix["rollout_rmse_phys"]]
        cis = [c for _, c in matrix["rollout_rmse_phys"]]
        bl = wres.get("baselines_cv", {})
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.bar(range(len(cols)), means, yerr=cis, capsize=4, color=colors)
        if bl:
            bname = min(bl, key=lambda n: bl[n]["agg"]["rollout_rmse_phys"]["mean"])
            bval = bl[bname]["agg"]["rollout_rmse_phys"]["mean"]
            ax.axhline(bval, color="k", ls="--", lw=1.4,
                       label=f"best trivial baseline ({bname}) = {bval:.4f}")
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels)
        ax.set_ylabel("Rollout RMSE (phys)")
        ax.set_title(f"Hyper-parameter comparison (CV mean $\\pm$ 95% CI) - L={L}")
        p = os.path.join(out_dir, f"fig_hp_comparison_L{L}.jpg")
        fig.savefig(p, dpi=dpi, format="jpg", bbox_inches="tight"); plt.close(fig)
        saved.append(p)

        fig, ax = plt.subplots(figsize=(6.4, 4))
        groups = ["rollout_rmse_mag", "rollout_rmse_corr"]
        x = np.arange(len(groups)); w = 0.35
        for i, kind in enumerate(["rnn", "transformer"]):
            bn = wres["best"][kind]
            vals = [wres["cv"][kind][bn]["agg"][g]["mean"] for g in groups]
            errs = [wres["cv"][kind][bn]["agg"][g]["ci95"] for g in groups]
            ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=4,
                   label=f"{kind} ({bn})", color="#1f77b4" if i == 0 else "#d62728")
        ax.set_xticks(x); ax.set_xticklabels(["magnetizations", "correlations"])
        ax.set_ylabel("Rollout RMSE (phys)"); ax.legend()
        ax.set_title(f"Error by feature group - L={L}")
        p = os.path.join(out_dir, f"fig_group_breakdown_L{L}.jpg")
        fig.savefig(p, dpi=dpi, format="jpg", bbox_inches="tight"); plt.close(fig)
        saved.append(p)

        fig, ax = plt.subplots(figsize=(8, 4.4))
        for kind, col in [("rnn", "#1f77b4"), ("transformer", "#d62728")]:
            ph = wres["final"][kind].get("per_horizon_rmse_physical")
            if ph:
                ax.plot(range(1, len(ph) + 1), ph, lw=1.6, color=col,
                        label=f"{kind} ({wres['best'][kind]})")
        for bname, bm in wres["final"]["rnn"].get("baselines_test", {}).items():
            ph = bm.get("per_horizon_rmse_physical")
            if ph:
                ax.plot(range(1, len(ph) + 1), ph, lw=1.0, ls="--", alpha=0.8,
                        label=f"baseline: {bname}")
        ax.set_xlabel("forecast horizon (steps)")
        ax.set_ylabel("RMSE (phys)")
        ax.set_title(f"Autoregressive error growth (TEST) - L={L}")
        ax.legend(fontsize=7, ncol=2)
        p = os.path.join(out_dir, f"fig_error_growth_L{L}.jpg")
        fig.savefig(p, dpi=dpi, format="jpg", bbox_inches="tight"); plt.close(fig)
        saved.append(p)

        fig, ax = plt.subplots(figsize=(9, 4))
        for kind, col in [("rnn", "#1f77b4"), ("transformer", "#d62728")]:
            h = wres["final"][kind].get("history", [])
            if h:
                ax.plot([r["epoch"] for r in h], [r["val_mse"] for r in h],
                        "-o", ms=3, color=col, label=f"{kind} val MSE")
                ax.plot([r["epoch"] for r in h], [r["train_loss"] for r in h],
                        "--", lw=1, color=col, alpha=0.6, label=f"{kind} train loss")
        h0 = wres["final"]["rnn"].get("history", [])
        shades = {"teacher_forcing": "#d9f0d3", "masked_modeling": "#fde0dd",
                  "scheduled_sampling": "#e0ecf4"}
        start = 0
        for i, r in enumerate(h0):
            if i == len(h0) - 1 or h0[i + 1]["regime"] != r["regime"]:
                ax.axvspan(start - 0.5, r["epoch"] + 0.5,
                           color=shades.get(r["regime"], "#eee"), alpha=0.55, zorder=0)
                start = r["epoch"] + 1
        ax.set_xlabel("epoch"); ax.set_ylabel("loss / MSE"); ax.legend(fontsize=7)
        ax.set_title(f"Learning curves of the selected models - L={L}")
        p = os.path.join(out_dir, f"fig_training_curves_L{L}.jpg")
        fig.savefig(p, dpi=dpi, format="jpg", bbox_inches="tight"); plt.close(fig)
        saved.append(p)

    return saved


def plot_ablation(ablation, out_dir, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    kinds = list(ablation["variants"].keys())
    variants = list(ablation["variants"][kinds[0]]["results"].keys())
    x = np.arange(len(variants)); w = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, kind in enumerate(kinds):
        r = ablation["variants"][kind]["results"]
        vals = [r[v]["agg"]["rollout_rmse_phys"]["mean"] for v in variants]
        errs = [r[v]["agg"]["rollout_rmse_phys"]["ci95"] for v in variants]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=4,
               label=kind, color="#1f77b4" if i == 0 else "#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace(" + ", "\n+ ") for v in variants], fontsize=8)
    ax.set_ylabel("Rollout RMSE (phys)")
    ax.set_title(f"Ablation of the training schedule "
                 f"({ablation['total_epochs']} epochs, L={ablation['window']})")
    ax.legend()
    p = os.path.join(out_dir, "fig_ablation_regimes.jpg")
    fig.savefig(p, dpi=dpi, format="jpg", bbox_inches="tight"); plt.close(fig)
    return [p]