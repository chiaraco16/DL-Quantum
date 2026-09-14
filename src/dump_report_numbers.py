"""
DL-Quantum -- Estrazione in chiaro dei numeri dei risultati.

Stesso contenuto di make_report_macros.py, che produce macro LaTeX, ma in forma
leggibile: un prospetto ordinato di configurazione, baseline, metriche per finestra,
esiti dei confronti di significativita' e ablation. Serve a rileggere i risultati di
un'esecuzione senza aprire i JSON e a confrontarli riga per riga con quanto riportato
nel documento finale.

Uso:
    python dump_report_numbers.py
    python dump_report_numbers.py --results ../artifacts/step3/results_full.json \\
        --ablation ../artifacts/step3/ablation_full.json \\
        --step2 ../artifacts/step2/step2_summary.json \\
        --out ../artifacts/step3/report_numbers.txt

Dipende solo dalla libreria standard.
"""
from __future__ import annotations
import argparse, json, os

PRETTY = {"rnn": "RNN", "transformer": "Transformer"}


def g(d, *keys, default=None):
    """get annidato sicuro"""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/d"
    return "n/d" if v != v else f"{v:.{nd}f}"


def line(label, value, width=30):
    return f"  {label:.<{width}} {value}"


def _disgiunti(a, b) -> bool:
    """True se i due intervalli di confidenza non si sovrappongono."""
    lo_a, hi_a = a.get("mean", 0) - a.get("ci95", 0), a.get("mean", 0) + a.get("ci95", 0)
    lo_b, hi_b = b.get("mean", 0) - b.get("ci95", 0), b.get("mean", 0) + b.get("ci95", 0)
    return hi_a < lo_b or hi_b < lo_a


def build(results, ablation=None, step2=None):
    L = []
    W = L.append
    W("=" * 74)
    W(" DL-QUANTUM -- PROSPETTO DEI RISULTATI")
    W(" Fonte: results_full.json" + (" + ablation_full.json" if ablation else "")
      + (" + step2_summary.json" if step2 else ""))
    W("=" * 74)

    cfg = results.get("config", {})
    windows = list(results["windows"].keys())
    n_rnn = len(g(results, "windows", windows[0], "cv", "rnn", default={}))
    n_tr = len(g(results, "windows", windows[0], "cv", "transformer", default={}))
    folds = cfg.get("folds", "?")
    origins = cfg.get("origins", [])
    sched = cfg.get("schedule", [])
    tc = cfg.get("train_cfg", {})
    n_runs = (n_rnn + n_tr) * len(windows) * (folds if isinstance(folds, int) else 1)

    W("")
    W("CONFIGURAZIONE")
    W(line("schedule (epoche)", "+".join(str(n) for _, n in sched)
           + f"  (= {sum(n for _, n in sched)} epoche)"))
    W(line("finestre temporali L", ", ".join(str(w) for w in windows)))
    W(line("fold cross-validation", folds))
    W(line("origini rollout", ", ".join(str(o) for o in origins)))
    W(line("mask_prob / ss_passes", f"{tc.get('mask_prob','?')} / {tc.get('ss_passes','?')}"))
    W(line("seed / determinismo", f"{cfg.get('seed','?')} / "
           + str(g(cfg, "reproducibility", "op_determinism", default="?"))))
    W(line("config HP (RNN / Tr)", f"{n_rnn} / {n_tr}"))
    n_final = 2 * len(windows)
    W(line("addestramenti di CV",
           f"{n_runs}  ({n_rnn}+{n_tr} config x {len(windows)} finestre x {folds} fold)"))
    W(line("addestramenti finali", f"{n_final}  (miglior config per famiglia e finestra)"))

    best_overall = (None, None, float("inf"))
    for i, w in enumerate(windows):
        wres = results["windows"][w]
        W("")
        W("-" * 74)
        W(f" FINESTRA L = {w}")
        W("-" * 74)

        bl = wres.get("baselines_cv", {})
        if bl:
            bn = min(bl, key=lambda n: g(bl, n, "agg", "rollout_rmse_phys", "mean", default=1e9))
            W(line("miglior baseline banale",
                   f"{bn}   rollout RMSE {f(g(bl,bn,'agg','rollout_rmse_phys','mean'))}"
                   f"   NRMSE {f(g(bl,bn,'agg','rollout_nrmse','mean'),3)}"))

        for kind in ("rnn", "transformer"):
            bnm = g(wres, "best", kind)
            agg = g(wres, "cv", kind, bnm, "agg", default={})
            fin = g(wres, "final", kind, default={})
            sig = g(wres, "significance", kind, default={})
            W("")
            W(f"  [{PRETTY[kind]}]  best config: {bnm}")
            W(line("CV  rollout RMSE +/- IC95",
                   f"{f(g(agg,'rollout_rmse_phys','mean'))} +/- {f(g(agg,'rollout_rmse_phys','ci95'))}", 30))
            W(line("TEST rollout RMSE (fis.)", f(fin.get("rollout_rmse_phys"))))
            W(line("TEST next-step RMSE", f(fin.get("nextstep_rmse"))))
            W(line("TEST NRMSE", f(fin.get("rollout_nrmse"), 3)))
            W(line("TEST skill vs baseline", f(fin.get("skill_vs_baseline"), 3)))
            W(line("TEST RMSE mag / corr",
                   f"{f(fin.get('rollout_rmse_mag'))} / {f(fin.get('rollout_rmse_corr'))}"))
            ph = fin.get("per_horizon_rmse_physical") or fin.get("per_horizon_rmse") or []
            if ph:
                W(line("errore rollout 1o / ultimo passo",
                       f"{f(ph[0],5)} -> {f(ph[-1],5)}  (fattore {f(ph[-1]/max(ph[0],1e-12),1)})", 34))
            if sig:
                significant = sig.get("significant")
                W(line("significativita' best vs 2o",
                       f"{sig.get('best')} vs {sig.get('second')}: "
                       + ("SIGNIFICATIVO (IC disgiunti)" if significant
                          else "NON significativo (IC sovrapposti)"), 34))
                if sig.get("tie_broken"):
                    W(line("  nota selezione",
                           f"IC sovrapposti fra {', '.join(sig.get('tie_candidates', []))}; "
                           f"spareggio sul next-step RMSE: scelta {bnm} "
                           f"(migliore media di rollout: {sig.get('leader')})", 34))

            rr = fin.get("rollout_rmse_phys")
            if isinstance(rr, (int, float)) and rr == rr and rr < best_overall[2]:
                best_overall = (kind, w, rr)

        r = g(wres, "final", "rnn", "rollout_rmse_phys", default=1e9)
        t = g(wres, "final", "transformer", "rollout_rmse_phys", default=1e9)
        winner = "rnn" if r <= t else "transformer"
        n_beats = sum(1 for k in ("rnn", "transformer")
                      if g(wres, "final", k, "skill_vs_baseline", default=-9) > 0)
        W("")
        W(line("modello migliore", PRETTY[winner], 30))
        W(line("battono il miglior baseline",
               {0: "nessuna delle due", 1: "una sola delle due",
                2: "entrambe"}[n_beats], 30))

    W("")
    W("=" * 74)
    W(line("MIGLIORE ASSOLUTO",
           f"{PRETTY.get(best_overall[0],'?')} a L={best_overall[1]}  "
           f"rollout RMSE {f(best_overall[2])}", 30))
    tot = sum(1 for w in windows for k in ("rnn", "transformer")
              if g(results, "windows", w, "final", k, "skill_vs_baseline", default=-9) > 0)
    n_models = 2 * len(windows)
    W(line("verdetto complessivo",
           f"{tot} modelli su {n_models} superano il miglior predittore banale"
           if tot else
           "nessun modello appreso supera il miglior predittore banale", 30))
    W("=" * 74)

    if ablation:
        W("")
        W("-" * 74)
        W(f" ABLATION DELLO SCHEDULE  (L = {ablation.get('window','?')}, "
          f"{ablation.get('total_epochs','?')} epoche, budget costante)")
        W("-" * 74)
        for kind, blk in ablation.get("variants", {}).items():
            res = blk.get("results", {})
            if not res:
                continue
            best = min(res, key=lambda v: g(res, v, "agg", "rollout_rmse_phys", "mean", default=1e9))
            W("")
            W(f"  [{PRETTY.get(kind,kind)}]  best variante: {best}")
            for v, blkv in res.items():
                a = g(blkv, "agg", "rollout_rmse_phys", default={})
                ns = g(blkv, "agg", "nextstep_rmse", default={})
                mark = "  <== best" if v == best else ""
                W(line(v, f"rollout {f(a.get('mean'))} +/- {f(a.get('ci95'))}"
                       f"   next-step {f(ns.get('mean'))}{mark}", 36))
            ref = g(res, "TF only", "agg", "rollout_rmse_phys", default={})
            top = g(res, best, "agg", "rollout_rmse_phys", default={})
            if ref:
                gain = ref["mean"] - top.get("mean", ref["mean"])
                #Un guadagno con intervalli sovrapposti non e' distinguibile dalla
                #variabilita' fra i fold: il giudizio va riportato accanto al numero.
                esito = ("IC disgiunti: differenza significativa" if _disgiunti(top, ref)
                         else "IC sovrapposti: differenza non significativa")
                W(line("  guadagno best vs 'TF only'", f"{f(gain,5)}  ({esito})", 36))

    if step2:
        W("")
        W("-" * 74)
        W(" RUN DIMOSTRATIVA DELLO STEP 2  (singolo addestramento, senza cross-validation)")
        W("-" * 74)
        for kind in ("rnn", "transformer"):
            s = step2.get(kind, {})
            if not s:
                continue
            W(line(f"{PRETTY.get(kind,kind)}: params",
                   f"{s.get('params','?')}   next {f(s.get('nextstep'))}   "
                   f"rollout {f(s.get('rollout_phys'))}   NRMSE {f(s.get('nrmse'),3)}   "
                   f"skill {f(s.get('skill'),3)}", 26))

    W("")
    W("(I valori di questo prospetto vanno confrontati riga per riga con quelli citati "
      "nel documento finale.)")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../artifacts/step3/results_full.json")
    ap.add_argument("--ablation", default="../artifacts/step3/ablation_full.json")
    ap.add_argument("--step2", default="../artifacts/step2/step2_summary.json")
    ap.add_argument("--out", default="../artifacts/step3/report_numbers.txt")
    args = ap.parse_args()

    results = json.load(open(args.results))
    ablation = json.load(open(args.ablation)) if os.path.exists(args.ablation) else None
    step2 = json.load(open(args.step2)) if os.path.exists(args.step2) else None

    txt = build(results, ablation, step2)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(txt + "\n")
        print(f"\n[salvato] {args.out}")


if __name__ == "__main__":
    main()
