"""
DL-Quantum -- Inferenza sul test set a partire dai modelli salvati.

Carica i pesi dei modelli selezionati e il pacchetto compresso con i dati di test gia'
preparati (`test_pack_L*.npz`), senza rileggere il CSV originale: l'inferenza parte in
pochi secondi e non dipende dalla presenza del dataset completo.

Per ciascun modello calcola la previsione a un passo e il rollout autoregressivo
sull'intero orizzonte, li confronta con i predittori banali sugli stessi dati, stampa
una tabella riassuntiva ordinata per errore di rollout e salva la figura del rollout.

Esempio d'uso:
    python run_inference.py --models ../artifacts/step3/models --window 50
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from data_preprocessing import FeatureScaler, set_global_seeds, n_features
from models import load_model_bundle
from training import evaluate_nextstep, evaluate_rollout, arrays_to_dataset, plot_rollout
from baselines import evaluate_rollout_baselines, evaluate_nextstep_baselines, skill_score


def load_test_pack(models_dir: str, window: int):
    """Carica il pacchetto di test, con ripiego sulla versione ridotta.

    Il pacchetto completo pesa decine di MB e non sempre accompagna il codice.
    In sua assenza viene usato quello ridotto, se presente, dichiarandolo: le
    metriche calcolate su un sottoinsieme non coincidono con quelle del
    documento finale, ottenute sul test set completo.
    """
    path = os.path.join(models_dir, f"test_pack_L{window}.npz")
    if not os.path.exists(path):
        demo = os.path.join(models_dir, f"test_pack_L{window}_demo.npz")
        if not os.path.exists(demo):
            raise FileNotFoundError(
                f"{path} non trovato. Eseguire prima run_step3_experiments.py, "
                f"che genera il pacchetto di test insieme ai modelli.")
        path = demo
    d = np.load(path, allow_pickle=True)
    if "subset" in d.files:
        n_ro, tot_ro, n_ns, tot_ns = (int(v) for v in d["subset"])
        print(f"\n  ATTENZIONE: pacchetto di test RIDOTTO "
              f"({n_ro}/{tot_ro} coppie di rollout, {n_ns}/{tot_ns} finestre a un passo).\n"
              f"  Le metriche qui sotto NON coincidono con quelle del documento, calcolate\n"
              f"  sul test set completo. Il pacchetto integrale si rigenera con "
              f"run_step3_experiments.py.")
    scaler = FeatureScaler("per_feature_zscore")
    scaler.mean_ = d["scaler_mean"].astype("float32")
    scaler.std_ = d["scaler_std"].astype("float32")
    return d, scaler


def resolve_n_qubits(pack, n_feat: int) -> int:
    """Numero di qubit del pacchetto, ricavato dal file o dedotto dal numero di feature."""
    if "n_qubits" in pack.files:
        return int(pack["n_qubits"])
    for nq in range(2, 65):
        if n_features(nq) == n_feat:
            return nq
    raise ValueError(f"Impossibile dedurre il numero di qubit da {n_feat} feature.")


def find_bundles(models_dir: str, window: int):
    return [os.path.basename(p).replace("_spec.json", "")
            for p in sorted(glob.glob(os.path.join(models_dir, f"best_*_L{window}_spec.json")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../artifacts/step3/models")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", default=None, help="cartella di destinazione delle figure")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seeds(args.seed)
    out_dir = args.out or args.models
    os.makedirs(out_dir, exist_ok=True)

    pack, scaler = load_test_pack(args.models, args.window)
    X_test, Y_test = pack["X_test"], pack["Y_test"]
    ctx, fut = pack["ctx_test"], pack["fut_test"]
    feature_idx = pack["feature_idx"]
    feature_names = [str(s) for s in pack["feature_names"]]
    time_grid = pack["time_grid"]
    n_qubits = resolve_n_qubits(pack, len(feature_idx))
    origin = int(np.asarray(pack["origins"]).ravel()[0]) if "origins" in pack.files else 0

    print("=" * 74)
    print(f"INFERENZA SUL TEST SET  (L={args.window})")
    print("=" * 74)
    test_ids = np.asarray(pack["test_traj_ids"]).tolist()
    print(f"traiettorie di test : {len(test_ids)}  "
          f"(id: {test_ids[:8]}{'...' if len(test_ids) > 8 else ''})")
    print(f"finestre a un passo : {X_test.shape}")
    print(f"coppie di rollout   : {ctx.shape} -> {fut.shape}  "
          f"(orizzonte H={fut.shape[1]} passi)")

    bl_ro = evaluate_rollout_baselines(ctx, fut, scaler=scaler,
                                       feature_idx=feature_idx, n_qubits=n_qubits)
    bl_ns = evaluate_nextstep_baselines(X_test, Y_test, scaler=scaler)
    best_bl = min(bl_ro, key=lambda k: bl_ro[k]["rollout_rmse_physical"])
    best_bl_val = bl_ro[best_bl]["rollout_rmse_physical"]

    rows = []
    for name, m in bl_ro.items():
        rows.append((f"baseline: {name}",
                     bl_ns.get(name, {}).get("rmse", float("nan")),
                     m["rollout_rmse_physical"], m["rollout_nrmse"],
                     skill_score(m["rollout_rmse_physical"], best_bl_val)))

    tags = find_bundles(args.models, args.window)
    if not tags:
        raise FileNotFoundError(
            f"Nessun modello 'best_*_L{args.window}' presente in {args.models}")

    for tag in tags:
        model, spec = load_model_bundle(args.models, tag, return_spec=True)
        ns = evaluate_nextstep(model, arrays_to_dataset(X_test, Y_test, 64, shuffle=False),
                               scaler=scaler)
        ro = evaluate_rollout(model, ctx, fut, scaler=scaler, feature_idx=feature_idx,
                              n_qubits=n_qubits, baselines=bl_ro)
        rows.append((tag, ns["rmse"], ro["rollout_rmse_physical"],
                     ro["rollout_nrmse"], ro["skill_vs_baseline"]))
        fig_path = os.path.join(out_dir, f"rollout_{tag}.jpg")
        plot_rollout(model, ctx, fut, time_grid, feature_names, fig_path,
                     traj_idx=0, feats=(0, 1, 10, 11), origin=origin)
        print(f"\n[{tag}]  parametri: {model.count_params():,}"
              f"   figura -> {os.path.basename(fig_path)}")

    print("\n" + "-" * 74)
    print(f"{'modello / baseline':32s} {'RMSE 1 passo':>13s} {'RMSE rollout':>13s} "
          f"{'NRMSE':>8s} {'skill':>8s}")
    print("-" * 74)
    for name, nsr, ror, nr, sk in sorted(rows, key=lambda r: r[2]):
        nsr_s = f"{'--':>13s}" if not np.isfinite(nsr) else f"{nsr:13.4f}"
        print(f"{name:32s} {nsr_s} {ror:13.4f} {nr:8.3f} {sk:+8.3f}")
    print("-" * 74)
    print("NRMSE = RMSE del rollout diviso per la dispersione del target: sotto 1 il "
          "modello e' piu' informativo del predittore costante.")
    print(f"Skill score = 1 - RMSE / RMSE della baseline migliore ({best_bl}): "
          f"positivo solo se il modello batte ogni predittore banale.")


if __name__ == "__main__":
    main()
