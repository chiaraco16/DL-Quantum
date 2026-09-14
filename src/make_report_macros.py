"""
DL-Quantum -- generatore delle macro LaTeX con i numeri dei risultati.

Legge i risultati completi e, se disponibile, l'ablation, e scrive un file di
`\\newcommand` con tutte le cifre citate nella discussione: errori, intervalli di
confidenza, skill score, esito dei confronti di significativita'.

Il file puo' essere incluso nel documento, cosi' che il testo discorsivo si aggiorni
insieme alle tabelle, oppure usato come riferimento per controllare riga per riga i
numeri riportati nel report. In entrambi i casi lo scopo e' lo stesso: nessuna cifra
del testo deve poter divergere da quella delle tabelle.

    python make_report_macros.py --results ../artifacts/step3/results_full.json \
        --ablation ../artifacts/step3/ablation_full.json \
        --out ../artifacts/step3/report_numbers.tex
"""
from __future__ import annotations

import argparse
import json
import os

#Un suffisso alfabetico per ciascuna finestra: i nomi delle macro LaTeX non possono
#contenere cifre, quindi L=50 diventa "A", L=100 diventa "B" e cosi' via.
IDX = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NAME = {"rnn": "Rnn", "transformer": "Tr"}
#Il testo delle macro finisce nel documento finale, che e' in inglese: le stringhe
#esposte al lettore sono in inglese, i commenti del codice restano in italiano.
PRETTY = {"rnn": "the RNN", "transformer": "the Transformer"}


def _f(x, nd=4):
    """Formatta un numero in modo sicuro sia per il testo che per le formule.

    Usa la clausola ensuremath per evitare che il segno meno di un valore negativo
    venga interpretato come un semplice trattino di sillabazione nel testo.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"\\ensuremath{{{v:.{nd}f}}}"


def _cmd(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def build(results, ablation=None):
    out = ["% ------------------------------------------------------------------\n",
           "% Generated automatically from the result files -- do not edit by hand\n",
           "% ------------------------------------------------------------------\n"]

    cfg = results.get("config", {})
    windows = list(results["windows"].keys())
    n_cfg_rnn = len(results["windows"][windows[0]]["cv"]["rnn"])
    n_cfg_tr = len(results["windows"][windows[0]]["cv"]["transformer"])
    folds = cfg.get("folds", "?")
    origins = cfg.get("origins", [])
    n_runs = (n_cfg_rnn + n_cfg_tr) * len(windows) * (folds if isinstance(folds, int) else 1)

    out.append(_cmd("nFolds", folds))
    out.append(_cmd("nOrigins", len(origins)))
    out.append(_cmd("originList", ", ".join(str(o) for o in origins)))
    out.append(_cmd("nConfigRnn", n_cfg_rnn))
    out.append(_cmd("nConfigTr", n_cfg_tr))
    out.append(_cmd("nWindows", len(windows)))
    out.append(_cmd("nRuns", n_runs))
    out.append(_cmd("windowList", " and ".join(f"$L={w}$" for w in windows)))
    sched = cfg.get("schedule", [])
    if sched:
        out.append(_cmd("scheduleStr",
                        "\\ensuremath{" + "{+}".join(str(n) for _, n in sched) + "}"))
        out.append(_cmd("nEpochs", sum(n for _, n in sched)))
    tc = cfg.get("train_cfg", {})
    out.append(_cmd("ssPasses", tc.get("ss_passes", 3)))
    out.append(_cmd("maskProb", "\\ensuremath{%g}" % tc.get("mask_prob", 0.15)))
    out.append(_cmd("reproStatus",
                    cfg.get("reproducibility", {}).get("op_determinism", "unknown")))

    best_overall = (None, None, float("inf"))
    for i, w in enumerate(windows):
        s = IDX[i]
        wres = results["windows"][w]
        out.append(f"% ---- results for time window L={w} ----\n")
        out.append(_cmd(f"window{s}", w))

        bl = wres.get("baselines_cv", {})
        if bl:
            bname = min(bl, key=lambda n: bl[n]["agg"]["rollout_rmse_phys"]["mean"])
            out.append(_cmd(f"blBest{s}", bname.replace("_", "-")))
            out.append(_cmd(f"blBestRmse{s}",
                            _f(bl[bname]["agg"]["rollout_rmse_phys"]["mean"])))
            out.append(_cmd(f"blBestNrmse{s}",
                            _f(bl[bname]["agg"]["rollout_nrmse"]["mean"], 3)))

        for kind in ("rnn", "transformer"):
            k = NAME[kind]
            bn = wres["best"][kind]
            agg = wres["cv"][kind][bn]["agg"]
            fin = wres["final"][kind]
            out.append(_cmd(f"best{k}{s}", bn))
            out.append(_cmd(f"cv{k}Rmse{s}", _f(agg["rollout_rmse_phys"]["mean"])))
            out.append(_cmd(f"cv{k}Ci{s}", _f(agg["rollout_rmse_phys"]["ci95"])))
            out.append(_cmd(f"test{k}Rmse{s}", _f(fin["rollout_rmse_phys"])))
            out.append(_cmd(f"test{k}Nrmse{s}", _f(fin["rollout_nrmse"], 3)))
            out.append(_cmd(f"test{k}Skill{s}", _f(fin["skill_vs_baseline"], 3)))
            out.append(_cmd(f"test{k}Next{s}", _f(fin["nextstep_rmse"])))
            out.append(_cmd(f"test{k}Mag{s}", _f(fin["rollout_rmse_mag"])))
            out.append(_cmd(f"test{k}Corr{s}", _f(fin["rollout_rmse_corr"])))
            sig = wres.get("significance", {}).get(kind, {})
            out.append(_cmd(f"signif{k}{s}",
                            "statistically significant" if sig.get("significant")
                            else "\\emph{not} statistically significant"))
            out.append(_cmd(f"runnerUp{k}{s}", sig.get("second", "n/a")))
            #La selezione e' a due stadi: se piu' configurazioni hanno intervalli di
            #confidenza sovrapposti sul rollout, la scelta passa al next-step RMSE.
            #Queste due macro permettono al testo di dichiararlo invece di presentare
            #come vincitrice una configurazione che in tabella non ha la media migliore.
            out.append(_cmd(f"leader{k}{s}", sig.get("leader", bn)))
            out.append(_cmd(f"tieBreak{k}{s}",
                            "selected by the next-step RMSE tie-break among configurations "
                            "with overlapping intervals" if sig.get("tie_broken")
                            else "first on mean rollout error as well"))

            try:
                ph = fin.get("per_horizon_rmse_physical") or []
                if ph:
                    out.append(_cmd(f"err{k}First{s}", _f(ph[0], 5)))
                    out.append(_cmd(f"err{k}Last{s}", _f(ph[-1], 5)))
                    out.append(_cmd(f"err{k}Growth{s}", _f(ph[-1] / max(ph[0], 1e-12), 1)))
            except Exception:
                pass
            if fin["rollout_rmse_phys"] < best_overall[2]:
                best_overall = (kind, w, fin["rollout_rmse_phys"])

        r, t = wres["final"]["rnn"], wres["final"]["transformer"]
        winner = "rnn" if r["rollout_rmse_phys"] <= t["rollout_rmse_phys"] else "transformer"
        out.append(_cmd(f"winner{s}", PRETTY[winner]))
        # \beatsBaseline descrive il modello vincitore della finestra; \verdict descrive
        # quante delle due architetture superano il miglior predittore banale: uno skill
        # positivo su una sola delle due non autorizza a dire "entrambe".
        out.append(_cmd(f"beatsBaseline{s}",
                        "beats" if wres["final"][winner]["skill_vs_baseline"] > 0
                        else "does \\emph{not} beat"))
        n_beats = sum(1 for m in (r, t) if m["skill_vs_baseline"] > 0)
        out.append(_cmd(f"verdict{s}", {
            0: "neither architecture beats the best trivial predictor",
            1: "only one of the two architectures beats the best trivial predictor",
            2: "both architectures beat every trivial predictor"}[n_beats]))

    out.append(_cmd("bestOverall", PRETTY.get(best_overall[0], "n/a")))
    out.append(_cmd("bestOverallWindow", best_overall[1] or "n/a"))
    out.append(_cmd("bestOverallRmse", _f(best_overall[2])))

    #Il verdetto complessivo conta quanti modelli finali hanno skill positivo: uno
    #skill positivo su un solo modello non autorizza la formula al plurale.
    n_tot = 2 * len(windows)
    n_beats_all = sum(1 for w in windows for kind in ("rnn", "transformer")
                      if results["windows"][w]["final"][kind]["skill_vs_baseline"] > 0)
    out.append(_cmd("nBeatsBaseline", n_beats_all))
    out.append(_cmd("nModelsTotal", n_tot))
    if n_beats_all == 0:
        verdict_all = "no learned model beats the best trivial predictor"
    elif n_beats_all == n_tot:
        verdict_all = "every learned model beats the best trivial predictor"
    else:
        verdict_all = (f"{n_beats_all} of the {n_tot} learned models beat the best "
                       "trivial predictor")
    out.append(_cmd("verdictBest", verdict_all))

    if ablation:
        out.append("% ---- ablation study ----\n")
        out.append(_cmd("ablWindow", ablation.get("window", "?")))
        out.append(_cmd("ablEpochs", ablation.get("total_epochs", "?")))
        for kind, blk in ablation.get("variants", {}).items():
            k = NAME[kind]
            res = blk["results"]
            best = min(res, key=lambda v: res[v]["agg"]["rollout_rmse_phys"]["mean"])
            out.append(_cmd(f"abl{k}Best", best.replace("_", "-")))
            out.append(_cmd(f"abl{k}BestRmse",
                            _f(res[best]["agg"]["rollout_rmse_phys"]["mean"])))
            # Variante "teacher forcing + masked modeling", cioe' lo schedule completo
            # privato dello scheduled sampling: non e' il masked modeling da solo.
            tfmm = res.get("TF + MM", {}).get("agg", {})
            if tfmm:
                out.append(_cmd(f"abl{k}TfMm", _f(tfmm["rollout_rmse_phys"]["mean"])))
            tfo = res.get("TF only", {}).get("agg", {})
            if tfo:
                a = tfo["rollout_rmse_phys"]
                b = res[best]["agg"]["rollout_rmse_phys"]
                out.append(_cmd(f"abl{k}TfOnly", _f(a["mean"])))
                delta = a["mean"] - b["mean"]
                out.append(_cmd(f"abl{k}Gain", _f(delta, 5)))
                out.append(_cmd(f"abl{k}Helps",
                                "improves on" if delta > 0 else "does not improve on"))

                overlap = not (b["mean"] + b["ci95"] < a["mean"] - a["ci95"]
                               or a["mean"] + a["ci95"] < b["mean"] - b["ci95"])
                out.append(_cmd(f"abl{k}Signif",
                                "\\emph{not} statistically significant" if overlap
                                else "statistically significant"))
                if overlap:
                    verdict = ("is \\emph{statistically indistinguishable} from the "
                               "reference configuration: the difference lies within the "
                               "confidence intervals")
                elif delta > 0:
                    verdict = ("\\emph{improves} overall performance, with disjoint "
                               "confidence intervals")
                else:
                    verdict = "brings no significant improvement"
                out.append(_cmd(f"abl{k}Verdict", verdict))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../artifacts/step3/results_full.json")
    ap.add_argument("--ablation", default="../artifacts/step3/ablation_full.json")
    ap.add_argument("--out", default="../artifacts/step3/report_numbers.tex")
    args = ap.parse_args()

    results = json.load(open(args.results))
    ablation = json.load(open(args.ablation)) if os.path.exists(args.ablation) else None
    txt = build(results, ablation)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt)
    n = txt.count("\\newcommand")
    print(f"scritte {n} macro -> {args.out}")


if __name__ == "__main__":
    main()