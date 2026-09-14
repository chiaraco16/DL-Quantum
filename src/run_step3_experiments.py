"""
DL-Quantum -- Step 3: sweep degli iperparametri, cross-validation e ablation.

  1. legge da riga di comando la configurazione dell'esperimento (finestre, fold,
     schedule, orizzonte, origini del rollout, varianti del masking e dello
     scheduled sampling); `--quick` esegue la stessa pipeline su scala ridotta per
     verificarne il funzionamento in pochi minuti;
  2. esegue lo sweep completo su entrambe le famiglie di modelli;
  3. genera tabelle CSV e LaTeX e le figure a 300 dpi;
  4. con `--ablation`, confronta le varianti dello schedule multi-regime a parità
     di budget di epoche.

Esempio d'uso:
    python run_step3_experiments.py --csv ../trajectories.csv --windows 50,100 \\
        --folds 5 --schedule 6,6,3 --ablation
"""
from __future__ import annotations

import argparse
import os

from data_preprocessing import set_global_seeds, reproducibility_report, dump_json
from experiments import (default_hp_grid, describe_hp_grid, hp_grid_latex,
                         run_full_sweep, run_regime_ablation, build_table1_csv,
                         build_table1_latex, build_ablation_table, plot_results,
                         plot_ablation, METRICS)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.environ.get("QUANTUM_CSV", os.path.join(_HERE, "..", "trajectories.csv"))


def parse_schedule(s):
    a, b, c = (int(x) for x in s.split(","))
    return [("teacher_forcing", a), ("masked_modeling", b), ("scheduled_sampling", c)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "artifacts", "step3"))
    ap.add_argument("--windows", default="50,100")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--schedule", default="6,6,3")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--origins", default="0,250,500",
                    help="origini temporali del rollout separate da virgola")
    ap.add_argument("--feature-set", default="all")
    ap.add_argument("--max-traj", type=int, default=None)
    ap.add_argument("--train-stride", type=int, default=None,
                    help="passo di scorrimento delle finestre di training")
    ap.add_argument("--select-on", default="rollout_rmse_phys",
                    choices=[k for k, _, _ in METRICS])
    ap.add_argument("--tie-break-on", default="nextstep_rmse",
                    choices=[k for k, _, _ in METRICS],
                    help="metrica di spareggio fra configurazioni con intervalli sovrapposti")
    ap.add_argument("--ss-mode", default="true", choices=["true", "two_pass"])
    ap.add_argument("--ss-passes", type=int, default=3)
    ap.add_argument("--mask-mode", default="denoise_next",
                    choices=["denoise_next", "reconstruct"])
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--full-grid", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                    help="attiva lo studio di ablation sullo schedule multi-regime")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="modalità di test rapido su scala ridotta")
    args = ap.parse_args()

    set_global_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)
    hp_grid = default_hp_grid(full_grid=args.full_grid)
    train_cfg_kwargs = {"ss_mode": args.ss_mode, "ss_passes": args.ss_passes,
                        "mask_mode": args.mask_mode, "mask_prob": args.mask_prob}

    if args.quick:
        from models import RNNConfig, TransformerConfig
        windows = [24]
        folds = 2
        schedule = parse_schedule("1,1,1")
        max_traj = 24
        horizon = 40
        origins = [0, 20]
        hp_grid = {
            "rnn": [{"name": "R1", "lr": 1e-2, "model": RNNConfig("LSTM", 8, 1)},
                    {"name": "R2", "lr": 1e-3, "model": RNNConfig("GRU", 12, 1)}],
            "transformer": [
                {"name": "T1", "lr": 1e-2, "model": TransformerConfig(8, 2, 1, 16)},
                {"name": "T2", "lr": 1e-3, "model": TransformerConfig(16, 2, 1, 32)}],
        }
    else:
        windows = [int(w) for w in args.windows.split(",")]
        folds = args.folds
        schedule = parse_schedule(args.schedule)
        max_traj = args.max_traj
        horizon = args.horizon
        origins = [int(o) for o in args.origins.split(",")]

    if len(windows) < (1 if args.quick else 2):
        raise SystemExit("Il confronto richiede almeno due finestre temporali "
                         "(--windows 50,100).")
    if sum(n for _, n in schedule) <= 0:
        raise SystemExit("Lo schedule deve prevedere almeno un'epoca.")

    print("=" * 68)
    print("CONFIGURAZIONE GRIGLIA IPERPARAMETRI")
    print("=" * 68)
    print(describe_hp_grid(hp_grid))
    hp_grid_latex(hp_grid, os.path.join(args.out, "table_hp_configs.tex"),
                  schedule=schedule, windows=windows, folds=folds,
                  batch_size=args.batch_size)

    results = run_full_sweep(
        csv_path=args.csv, windows=windows, folds=folds, schedule=schedule,
        feature_set=args.feature_set, max_trajectories=max_traj, horizon=horizon,
        seed=args.seed, out_dir=args.out, hp_grid=hp_grid,
        batch_size=args.batch_size, origins=origins,
        train_stride=args.train_stride, select_on=args.select_on,
        tie_break_on=args.tie_break_on,
        train_cfg_kwargs=train_cfg_kwargs, verbose=True)

    print("\n" + "=" * 68)
    print("GENERAZIONE TABELLA PRINCIPALE")
    print("=" * 68)
    for i, (L, wres) in enumerate(results["windows"].items()):
        csv_p = build_table1_csv(wres, os.path.join(args.out, f"table1_L{L}.csv"))
        tex_p = build_table1_latex(wres, os.path.join(args.out, f"table1_L{L}.tex"),
                                   L, folds,
                                   label=f"tab:main{'ABCD'[i] if i < 4 else i}")
        print(f"\n--- Finestra L={L} ---  ({csv_p} , {tex_p})")
        with open(csv_p) as f:
            for line in f:
                print("  " + line.rstrip())

    figs = plot_results(results, args.out)

    if args.ablation:
        print("\n" + "=" * 68)
        print("STUDIO DI ABLATION DELLO SCHEDULE MULTI-REGIME")
        print("=" * 68)
        best_names = {k: results["windows"][str(windows[0])]["best"][k]
                      for k in ("rnn", "transformer")}
        abl = run_regime_ablation(
            csv_path=args.csv, window=windows[0], folds=folds, schedule=schedule,
            feature_set=args.feature_set, max_trajectories=max_traj, horizon=horizon,
            seed=args.seed, out_dir=args.out, hp_grid=hp_grid,
            batch_size=args.batch_size, origins=origins, best_names=best_names,
            ss_passes=args.ss_passes, verbose=True)
        build_ablation_table(abl, os.path.join(args.out, "table_ablation.csv"),
                             os.path.join(args.out, "table_ablation.tex"))
        figs += plot_ablation(abl, args.out)

    dump_json(reproducibility_report(args.seed),
              os.path.join(args.out, "reproducibility.json"))

    print(f"\n[output] {len(figs)} figure salvate a 300 dpi in -> {args.out}")
    print("\nElaborazione completata. Seme di riproducibilità:", args.seed)
    print("Stato di determinismo operativo:", reproducibility_report(args.seed)["op_determinism"])


if __name__ == "__main__":
    main()