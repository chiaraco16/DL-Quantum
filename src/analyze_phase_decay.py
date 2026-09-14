"""
DL-Quantum -- Analisi della degradazione del rollout: ampiezza contro fase.

Sul rollout lungo lo skill score dei modelli e' negativo. Il numero da solo non
dice perche', e ci sono due meccanismi possibili, con firme opposte:

* collasso di ampiezza -- il modello smorza verso la media. Allora la deviazione
  standard delle previsioni tende a zero e l'errore tende alla dispersione del
  target;
* perdita di fase -- il modello continua a oscillare con l'ampiezza giusta ma
  sfasato. Per y = A sin(wt + f) e una previsione A sin(wt + f + d) con d
  scorrelato, l'errore quadratico medio vale A^2, mentre il predittore costante
  ottimale vale A^2/2: il rapporto atteso fra i due RMSE e' sqrt(2) ~= 1.414.

Distinguere i due casi cambia la lettura del risultato. Nel primo il modello non
ha imparato la dinamica; nel secondo l'ha imparata e la perde per decoerenza, e
oltre l'orizzonte di coerenza la media condizionata e' la previsione ottima in
norma L2 -- nessun predittore che continui a oscillare puo' batterla.

Lo script misura, in funzione dell'orizzonte, la deviazione standard delle
previsioni e del target, la loro correlazione e il rapporto fra l'RMSE del
modello e quello del predittore costante ottimale. Scrive `phase_analysis.json`
accanto agli altri risultati.

Uso:
    python analyze_phase_decay.py
    python analyze_phase_decay.py --models ../artifacts/step3/models --window 50 \\
        --out ../artifacts/step3/phase_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from data_preprocessing import set_global_seeds
from models import load_model_bundle
from training import autoregressive_rollout

#Orizzonti a cui viene riportato il dettaglio nella tabella per passo.
CHECKPOINTS = (0, 9, 24, 49, 99, 149, 199)


def _slice_stats(pred: np.ndarray, true: np.ndarray) -> dict:
    """Statistiche di una singola fetta temporale, appiattita su finestre e feature."""
    p, y = pred.ravel(), true.ravel()
    rmse_model = float(np.sqrt(((p - y) ** 2).mean()))
    #Predittore costante ottimale in L2 a quell'orizzonte: la media del target.
    rmse_const = float(np.sqrt(((y - y.mean()) ** 2).mean()))
    return {
        "std_pred": float(p.std()),
        "std_true": float(y.std()),
        "corr": float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 and y.std() > 0 else float("nan"),
        "rmse_model": rmse_model,
        "rmse_constant": rmse_const,
        "ratio": rmse_model / max(rmse_const, 1e-12),
    }


def analyze(models_dir: str, window: int, seed: int = 42) -> dict:
    set_global_seeds(seed)
    pack_path = os.path.join(models_dir, f"test_pack_L{window}.npz")
    if not os.path.exists(pack_path):
        raise FileNotFoundError(
            f"{pack_path} non trovato. Eseguire prima run_step3_experiments.py, "
            "che genera il pacchetto di test insieme ai modelli.")
    pack = np.load(pack_path, allow_pickle=True)
    ctx, fut, std_ = pack["ctx_test"], pack["fut_test"], pack["scaler_std"]

    out = {"window": int(window), "n_windows": int(ctx.shape[0]),
           "horizon": int(fut.shape[1]), "sqrt2": float(np.sqrt(2.0)), "models": {}}

    tags = sorted(os.path.basename(p)[:-len("_spec.json")]
                  for p in glob.glob(os.path.join(models_dir, f"*_L{window}_spec.json")))
    if not tags:
        raise FileNotFoundError(f"Nessun modello salvato per L={window} in {models_dir}.")

    for tag in tags:
        model = load_model_bundle(models_dir, tag)
        pred = np.asarray(autoregressive_rollout(model, ctx, fut.shape[1]))
        #In unita' fisiche: la media additiva si cancella nelle differenze e nelle
        #deviazioni standard, quindi basta riscalare per la sola sigma.
        p, y = pred * std_, fut * std_

        per_h = {str(h): _slice_stats(p[:, h, :], y[:, h, :])
                 for h in CHECKPOINTS if h < p.shape[1]}
        overall = _slice_stats(p, y)
        out["models"][tag] = {
            "per_horizon": per_h,
            "overall": overall,
            "amplitude_ratio": float(p.std() / max(y.std(), 1e-12)),
            "rmse_ratio_vs_constant": overall["ratio"],
        }
        print(f"\n=== {tag} ===")
        print(f"{'passo':>6} {'std(prev)':>10} {'std(vero)':>10} {'corr':>7} "
              f"{'RMSE mod':>9} {'RMSE cost':>10} {'rapporto':>9}")
        for h, d in per_h.items():
            print(f"{h:>6} {d['std_pred']:>10.5f} {d['std_true']:>10.5f} {d['corr']:>7.3f} "
                  f"{d['rmse_model']:>9.5f} {d['rmse_constant']:>10.5f} {d['ratio']:>9.3f}")
        print(f"  ampiezza conservata: std(prev)/std(vero) = "
              f"{out['models'][tag]['amplitude_ratio']:.3f}")
        print(f"  rapporto RMSE modello / costante = {overall['ratio']:.3f}  "
              f"(sqrt(2) = 1.414 se l'ampiezza e' corretta e la fase scorrelata)")
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(here, "..", "artifacts", "step3", "models"))
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = analyze(args.models, args.window, args.seed)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.models)),
                                   "phase_analysis.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    #Piu' finestre possono essere analizzate in esecuzioni successive: il file
    #accumula i risultati per finestra invece di sovrascriverli.
    acc = {}
    if os.path.exists(out):
        try:
            acc = json.load(open(out))
        except (ValueError, OSError):
            acc = {}
    acc[str(args.window)] = res
    with open(out, "w") as f:
        json.dump(acc, f, indent=2)
    print(f"\n[salvato] {out}")


if __name__ == "__main__":
    main()
