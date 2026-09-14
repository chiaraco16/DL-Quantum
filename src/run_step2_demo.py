"""
DL-Quantum -- Step 2: prova del training loop multi-regime su una singola finestra.

Una run dimostrativa, non un esperimento: serve a verificare che il ciclo unificato
(teacher forcing, masked modeling, scheduled sampling) converga e a produrre le figure
qualitative che accompagnano la descrizione dei regimi. Le conclusioni quantitative
vengono dallo Step 3, che ripete tutto in cross-validation.

  1. addestra una RNN e un Transformer con lo stesso schedule;
  2. salva curve di apprendimento, illustrazione dei regimi e grafici di rollout;
  3. valuta i due modelli sul test set accanto alle baseline banali.

Esempio d'uso:
    python run_step2_demo.py --csv ../trajectories.csv --window 50 --schedule 6,6,3
"""
from __future__ import annotations

import argparse
import os

from data_preprocessing import (N_POINTS_DEFAULT, PreprocessConfig, build_datasets,
                                set_global_seeds, dump_json)
from models import RNNConfig, TransformerConfig, build_model, count_params
from training import (TrainConfig, MultiRegimeTrainer, arrays_to_dataset,
                      evaluate_nextstep, evaluate_rollout, plot_history,
                      plot_rollout, plot_regimes_illustration)
from baselines import evaluate_rollout_baselines, summarize_baselines

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.environ.get("QUANTUM_CSV", os.path.join(_HERE, "..", "trajectories.csv"))


def parse_schedule(s):
    a, b, c = (int(x) for x in s.split(","))
    return [("teacher_forcing", a), ("masked_modeling", b), ("scheduled_sampling", c)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "artifacts"))
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--origins", default="0,250,500")
    ap.add_argument("--schedule", default="6,6,3")
    ap.add_argument("--max-traj", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seeds(args.seed)
    out_dir = os.path.join(args.out, "step2")
    os.makedirs(out_dir, exist_ok=True)

    origins = [int(o) for o in args.origins.split(",")]
    prep = build_datasets(PreprocessConfig(
        csv_path=args.csv, input_window=args.window, horizon=args.horizon,
        feature_set="all", max_trajectories=args.max_traj, seed=args.seed,
        rollout_origins=origins,
        max_rows=args.max_traj * N_POINTS_DEFAULT if args.max_traj else None), verbose=True)

    F = prep.X_train.shape[-1]
    tcfg = TrainConfig(schedule=parse_schedule(args.schedule), seed=args.seed,
                       verbose=True)
    plot_regimes_illustration(prep.X_train, os.path.join(out_dir, "fig_regimes.jpg"), tcfg)

    bl = evaluate_rollout_baselines(prep.ctx_test, prep.fut_test, scaler=prep.scaler,
                                    feature_idx=prep.feature_idx,
                                    n_qubits=prep.cfg.n_qubits)
    print("\n--- baseline di confronto sul test set ---")
    print(summarize_baselines(bl))

    summary = {}
    for kind, cfg in [("rnn", RNNConfig(rnn_type="LSTM", units=64, num_layers=1)),
                      ("transformer", TransformerConfig(d_model=64, num_heads=4,
                                                        num_layers=2, dff=128))]:
        print("\n" + "=" * 70)
        print(f"Elaborazione in corso per l'architettura: {kind}  ({cfg})")
        print("=" * 70)
        set_global_seeds(args.seed)
        model = build_model(kind, F, cfg)
        print(f"  parametri addestrabili: {count_params(model):,}")
        hist = MultiRegimeTrainer(model, tcfg).fit(prep.X_train, prep.Y_train,
                                                   prep.X_val, prep.Y_val)
        ns = evaluate_nextstep(model, arrays_to_dataset(prep.X_test, prep.Y_test,
                                                        64, shuffle=False),
                               scaler=prep.scaler)
        ro = evaluate_rollout(model, prep.ctx_test, prep.fut_test, scaler=prep.scaler,
                              feature_idx=prep.feature_idx,
                              n_qubits=prep.cfg.n_qubits, baselines=bl)
        print(f"  TEST next-step RMSE (scalato) = {ns['rmse']:.4f}   MAE = {ns['mae']:.4f}")
        print(f"  TEST rollout  RMSE (fisico)    = {ro['rollout_rmse_physical']:.4f}   "
              f"NRMSE = {ro['rollout_nrmse']:.3f}   skill = {ro['skill_vs_baseline']:+.3f}")
        plot_history(hist, os.path.join(out_dir, f"fig_training_curve_{kind}.jpg"))
        plot_rollout(model, prep.ctx_test, prep.fut_test, prep.time_grid,
                     prep.feature_names, os.path.join(out_dir, f"fig_rollout_{kind}.jpg"),
                     origin=prep.origins[0])
        summary[kind] = {"params": count_params(model), "nextstep": ns["rmse"],
                         "rollout_phys": ro["rollout_rmse_physical"],
                         "nrmse": ro["rollout_nrmse"],
                         "skill": ro["skill_vs_baseline"], "history": hist}

    dump_json(summary, os.path.join(out_dir, "step2_summary.json"))
    print(f"\nElaborazione completata con successo. Risultati e figure salvati in {out_dir}")


if __name__ == "__main__":
    main()