"""
DL-Quantum -- Step 1: preparazione dei dati, controlli di integrità ed EDA.

Per ciascuna finestra temporale richiesta:
  1. carica il CSV e lo riorganizza in tensore (traiettorie, istanti, osservabili);
  2. esegue i controlli di integrità fisica e temporale;
  3. misura lo squilibrio di ampiezza fra le osservabili;
  4. costruisce le partizioni per traiettoria, lo scaler e le finestre;
  5. valuta le baseline banali sul test set, come riferimento per gli step successivi;
  6. salva la cache dei tensori e, per la prima finestra, le figure esplorative.

Esempio d'uso:
    python run_step1_preprocessing.py --csv ../trajectories.csv --out ../artifacts
"""
from __future__ import annotations

import argparse
import os

from data_preprocessing import (N_POINTS_DEFAULT, PreprocessConfig, build_datasets, run_eda,
                                save_prepared, set_global_seeds,
                                reproducibility_report, dump_json)
from baselines import evaluate_rollout_baselines, summarize_baselines

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.environ.get("QUANTUM_CSV", os.path.join(_HERE, "..", "trajectories.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "artifacts"))
    ap.add_argument("--windows", default="50,100")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--origins", default="0,250,500")
    ap.add_argument("--feature-set", default="all")
    ap.add_argument("--max-traj", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seeds(args.seed)
    eda_dir = os.path.join(args.out, "eda")
    cache_dir = os.path.join(args.out, "cache")
    os.makedirs(eda_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    origins = [int(o) for o in args.origins.split(",")]
    windows = [int(w) for w in args.windows.split(",")]
    first = True

    for L in windows:
        print("=" * 70)
        print(f"Elaborazione dei dati in corso per la finestra L={L}")
        print("=" * 70)
        cfg = PreprocessConfig(
            csv_path=args.csv, input_window=L, horizon=args.horizon,
            feature_set=args.feature_set, max_trajectories=args.max_traj,
            seed=args.seed, rollout_origins=origins,
            max_rows=args.max_traj * N_POINTS_DEFAULT if args.max_traj else None)
        prep = build_datasets(cfg, verbose=True)

        print("\n--- controlli di integrità ---")
        for k, v in prep.report.items():
            if k.startswith("ok_") or k in ("n_nan", "n_inf", "n_duplicate_rows",
                                            "dt_inferred", "t_fin"):
                print(f"  {k:28s} {v}")

        print("\n--- sbilanciamento fra osservabili ---")
        print(f"  std max / std min = {prep.imbalance['std_ratio_max_over_min']:.3f}")
        for grp, label in [("magnetizations", "magnetizzazioni"),
                           ("correlations", "correlazioni")]:
            # Con un sottoinsieme di feature uno dei due gruppi puo' essere assente.
            v = prep.imbalance.get(f"{grp}_std_mean")
            if v is not None:
                print(f"  std media {label:16s} = {v:.5f}")

        print("\n--- baseline banali sul test set ---")
        bl = evaluate_rollout_baselines(prep.ctx_test, prep.fut_test,
                                        scaler=prep.scaler,
                                        feature_idx=prep.feature_idx,
                                        n_qubits=cfg.n_qubits)
        print(summarize_baselines(bl))

        npz = save_prepared(prep, cache_dir)
        print(f"\n  cache salvata in -> {npz}")

        if first:
            # Le figure esplorative descrivono i dati, non la finestra: basta generarle
            # una volta sola, sulla prima configurazione elaborata.
            figs = run_eda(prep, eda_dir)
            print(f"  EDA completata -> {len(figs)} figure generate a 300 dpi in {eda_dir}")
            first = False

    dump_json(reproducibility_report(args.seed),
              os.path.join(args.out, "reproducibility.json"))
    print("\nFase di preprocessing completata con successo.")


if __name__ == "__main__":
    main()