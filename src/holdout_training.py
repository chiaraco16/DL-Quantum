"""
Addestramento strumentato per il protocollo holdout 80/20.

Estende `MultiRegimeTrainer` senza modificarlo: lo schedule dei tre regimi, i tre
passi compilati e lo scheduled sampling a punto fisso restano identici. Cambia solo
cosa viene MISURATO a ogni epoca, perche' e' la misura che rende leggibile la curva.

Le quattro quantita' registrate a ogni epoca
--------------------------------------------
1. `train_loss`  - il valore dell'obiettivo effettivamente minimizzato in quell'epoca.
   Cambia definizione fra i regimi: MSE semplice in teacher forcing, MSE pesato su
   input mascherati in masked modeling, MSE su input parzialmente autogenerati in
   scheduled sampling. Serve a verificare che l'ottimizzazione stia scendendo DENTRO
   ciascun regime; NON e' confrontabile con la validation, e i suoi salti ai confini
   fra regimi sono cambi di obiettivo, non peggioramenti del modello.

2. `train_mse`   - MSE next-step in teacher forcing, calcolato su un sottocampione di
   traiettorie di TRAINING mai usato per aggiornare i pesi in quell'istante ma che
   proviene dalla partizione di training. E' la meta' "train" della coppia
   confrontabile.

3. `val_mse`     - la stessa identica metrica, sulle traiettorie di validation. La
   differenza `val_mse - train_mse` e' il divario di generalizzazione: e' questo, e
   non il divario rispetto a `train_loss`, che dice se il modello sta sovradattando.

4. `val_rollout_rmse` - RMSE del rollout autoregressivo su un piccolo campione fisso
   di coppie di validation, in unita' fisiche. E' il compito reale: previsione a H
   passi senza mai rivedere un valore osservato. Va guardata insieme a `val_mse`
   perche' le due possono divergere - errore a un passo che scende mentre l'errore a
   orizzonte lungo resta fermo e' esattamente la firma dell'accumulo d'errore.

Selezione dell'epoca
--------------------
Con un vero insieme di validation l'epoca migliore si puo' scegliere, invece di
subirla: si tiene la fotografia dei pesi all'epoca di `val_mse` minima e si ripristina
alla fine (`restore_best=True`). Il budget di epoche resta uguale per tutte le
configurazioni, quindi il confronto rimane a parita' di calcolo.
"""

from typing import Optional
import time

import numpy as np
import tensorflow as tf

from data_preprocessing import set_global_seeds, feature_group_ids
from training import (MultiRegimeTrainer, arrays_to_dataset, evaluate_nextstep,
                      autoregressive_rollout)


# Valutazione del rollout con errore per singola coppia

def evaluate_rollout_detailed(model, context, future, scaler=None, feature_idx=None,
                              n_qubits: int = 10, baselines: Optional[dict] = None,
                              n_boot: int = 1000, seed: int = 42) -> dict:
    """Metriche del rollout piu' l'errore di ogni singola coppia (contesto, futuro).

    Senza cross-validation non ci sono piu' cinque fold su cui calcolare un intervallo
    di confidenza. L'incertezza si stima allora dove effettivamente risiede: nella
    variabilita' fra le traiettorie di validation. Si ricampionano con reimmissione le
    coppie di rollout e si ricalcola l'RMSE su ogni ricampionamento; il percentile 2.5
    e 97.5 della distribuzione risultante e' l'intervallo di confidenza al 95%.
    Il ricampionamento e' sulle coppie, che e' l'unita' indipendente: coppie diverse
    provengono da traiettorie diverse o da origini temporali distanti.
    """
    H = future.shape[1]
    pred = autoregressive_rollout(model, context, H)
    out = _rollout_metrics(pred, future, scaler, feature_idx, n_qubits, n_boot, seed)
    if baselines:
        best_key = min(baselines, key=lambda k: baselines[k]["rollout_rmse_physical"])
        best = float(baselines[best_key]["rollout_rmse_physical"])
        out["best_baseline"] = best_key
        out["best_baseline_rmse_physical"] = best
        out["skill_vs_baseline"] = float(
            1.0 - out["rollout_rmse_physical"] / max(best, 1e-12))
    return out


def _rollout_metrics(pred, future, scaler, feature_idx, n_qubits, n_boot, seed) -> dict:
    err = pred - future
    out = {
        "rollout_rmse": float(np.sqrt((err ** 2).mean())),
        "rollout_mae": float(np.abs(err).mean()),
        "per_horizon_rmse": np.sqrt((err ** 2).mean(axis=(0, 2))),
    }
    if scaler is None:
        return out
    pred_p = scaler.inverse_transform(pred)
    fut_p = scaler.inverse_transform(future)
    errp = pred_p - fut_p
    # MSE di ogni coppia: e' la quantita' che viene ricampionata dal bootstrap.
    per_pair_mse = (errp ** 2).mean(axis=(1, 2))
    out["rollout_rmse_physical"] = float(np.sqrt(per_pair_mse.mean()))
    out["rollout_mae_physical"] = float(np.abs(errp).mean())
    out["per_horizon_rmse_physical"] = np.sqrt((errp ** 2).mean(axis=(0, 2)))
    out["per_pair_mse_physical"] = per_pair_mse
    lo, hi = bootstrap_rmse_ci(per_pair_mse, n_boot=n_boot, seed=seed)
    out["rollout_rmse_physical_ci95"] = (lo, hi)
    sigma = float(np.sqrt(max((fut_p ** 2).mean() - fut_p.mean() ** 2, 0.0)))
    out["rollout_nrmse"] = float(out["rollout_rmse_physical"] / max(sigma, 1e-12))
    out["target_sigma_physical"] = sigma
    if feature_idx is not None:
        gid = feature_group_ids(feature_idx, n_qubits)
        for g, name in [(0, "magnetizations"), (1, "correlations")]:
            m = gid == g
            if m.any():
                out[f"rmse_physical_{name}"] = float(np.sqrt((errp[:, :, m] ** 2).mean()))

    # Ampiezza e fase: servono a distinguere il collasso sulla media dalla perdita di
    # allineamento temporale. Se il modello collassasse sulla media, il rapporto fra le
    # dispersioni tenderebbe a zero; se invece conserva l'oscillazione ma ne perde la
    # fase, il rapporto resta vicino a 1 e a crollare e' la correlazione.
    H = fut_p.shape[1]
    out["amplitude_ratio"] = float(pred_p.std() / max(fut_p.std(), 1e-12))
    passi = sorted({0, H // 8, H // 2, H - 1})
    out["correlation_at_step"] = {}
    for s in passi:
        a = pred_p[:, s, :].ravel()
        b = fut_p[:, s, :].ravel()
        if a.std() < 1e-12 or b.std() < 1e-12:
            out["correlation_at_step"][int(s)] = float("nan")
        else:
            out["correlation_at_step"][int(s)] = float(np.corrcoef(a, b)[0, 1])
    ph = out["per_horizon_rmse_physical"]
    out["first_step_rmse_physical"] = float(ph[0])
    out["final_step_rmse_physical"] = float(ph[-1])
    out["error_growth_factor"] = float(ph[-1] / max(ph[0], 1e-12))
    return out


def bootstrap_rmse_ci(per_pair_mse: np.ndarray, n_boot: int = 1000,
                      seed: int = 42, alpha: float = 0.05):
    """Intervallo di confidenza percentile dell'RMSE, per ricampionamento delle coppie."""
    per_pair_mse = np.asarray(per_pair_mse, dtype=np.float64)
    n = len(per_pair_mse)
    if n < 2:
        v = float(np.sqrt(per_pair_mse.mean())) if n else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.sqrt(per_pair_mse[idx].mean(axis=1))
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


# Trainer strumentato

class HoldoutTrainer(MultiRegimeTrainer):
    """`MultiRegimeTrainer` con misura per epoca su training e validation.

    Non ridefinisce ne' i passi compilati ne' lo scheduled sampling: riusa
    `_run_epoch` della classe base e aggiunge solo la valutazione a fine epoca.
    """

    def fit_monitored(self, X_train, Y_train, X_val, Y_val,
                      X_train_eval=None, Y_train_eval=None,
                      ctx_val=None, fut_val=None,
                      ctx_train=None, fut_train=None,
                      scaler=None, feature_idx=None, n_qubits: int = 10,
                      rollout_every: int = 1, rollout_max_pairs: int = 40,
                      restore_best: bool = True, select_metric: str = "val_mse"):
        set_global_seeds(self.cfg.seed)
        self._build_steps(X_train.shape[-1])
        bs = self.cfg.batch_size

        train_ds = arrays_to_dataset(X_train, Y_train, bs, shuffle=True, seed=self.cfg.seed)
        val_ds = arrays_to_dataset(X_val, Y_val, bs, shuffle=False)
        if X_train_eval is None:
            X_train_eval, Y_train_eval = X_train, Y_train
        train_eval_ds = arrays_to_dataset(X_train_eval, Y_train_eval, bs, shuffle=False)

        # Campione fisso di coppie per il monitoraggio del rollout: fisso perche' la
        # curva deve raccontare come cambia il modello, non come cambia il campione.
        mon = _monitor_pairs(ctx_val, fut_val, rollout_max_pairs, self.cfg.seed)
        mon_tr = _monitor_pairs(ctx_train, fut_train, rollout_max_pairs, self.cfg.seed)

        history = []
        best = {"epoch": -1, "value": np.inf, "weights": None}
        epoch = 0
        for regime, n_ep in self.cfg.schedule:
            for e in range(n_ep):
                if regime == "scheduled_sampling":
                    frac = (e + 1) / max(n_ep, 1)
                    tf_prob = self.cfg.ss_tf_prob_start + frac * (
                        self.cfg.ss_tf_prob_end - self.cfg.ss_tf_prob_start)
                else:
                    tf_prob = self.cfg.ss_tf_prob_start

                t0 = time.time()
                tr_loss = self._run_epoch(train_ds, regime, tf_prob)

                # metriche confrontabili: stessa definizione su train e validation
                tr_m = evaluate_nextstep(self.model, train_eval_ds, scaler=scaler)
                va_m = evaluate_nextstep(self.model, val_ds, scaler=scaler)

                rec = {
                    "epoch": epoch, "regime": regime,
                    "train_loss": float(tr_loss),
                    "train_mse": float(tr_m["mse"]), "train_mae": float(tr_m["mae"]),
                    "val_mse": float(va_m["mse"]), "val_mae": float(va_m["mae"]),
                    "gap_mse": float(va_m["mse"] - tr_m["mse"]),
                    "tf_prob": round(float(tf_prob), 3),
                }
                if scaler is not None:
                    rec["train_rmse_physical"] = float(tr_m.get("rmse_physical", np.nan))
                    rec["val_rmse_physical"] = float(va_m.get("rmse_physical", np.nan))

                do_rollout = mon is not None and (
                    rollout_every > 0 and (epoch % rollout_every == 0 or _is_last(
                        self.cfg.schedule, epoch)))
                if do_rollout:
                    rec["val_rollout_rmse"] = _quick_rollout_rmse(
                        self.model, mon[0], mon[1], scaler)
                    if mon_tr is not None:
                        rec["train_rollout_rmse"] = _quick_rollout_rmse(
                            self.model, mon_tr[0], mon_tr[1], scaler)

                rec["sec"] = round(time.time() - t0, 1)
                history.append(rec)

                value = rec.get(select_metric, rec["val_mse"])
                if value < best["value"]:
                    best = {"epoch": epoch, "value": float(value),
                            "weights": [w.copy() for w in self.model.get_weights()]}

                if self.cfg.verbose:
                    extra = (f"  val_roll={rec['val_rollout_rmse']:.4f}"
                             if "val_rollout_rmse" in rec else "")
                    print(f"  [{epoch:02d}] {regime:18s} obj={tr_loss:.5f}  "
                          f"train_mse={rec['train_mse']:.5f}  val_mse={rec['val_mse']:.5f}  "
                          f"gap={rec['gap_mse']:+.5f}{extra}  ({rec['sec']}s)")
                epoch += 1

        restored = False
        if restore_best and best["weights"] is not None and best["epoch"] != epoch - 1:
            self.model.set_weights(best["weights"])
            restored = True
        info = {
            "best_epoch": int(best["epoch"]),
            "best_value": float(best["value"]),
            "select_metric": select_metric,
            "restored_best_weights": bool(restored),
            "total_epochs": int(epoch),
        }
        return history, info


def _is_last(schedule, epoch: int) -> bool:
    return epoch == sum(int(n) for _, n in schedule) - 1


def _monitor_pairs(ctx, fut, max_pairs: int, seed: int):
    if ctx is None or fut is None or len(ctx) == 0:
        return None
    if len(ctx) <= max_pairs:
        return ctx, fut
    rng = np.random.default_rng(seed + 13)
    sel = np.sort(rng.choice(len(ctx), size=max_pairs, replace=False))
    return ctx[sel], fut[sel]


def _quick_rollout_rmse(model, ctx, fut, scaler) -> float:
    """RMSE del rollout sul campione di monitoraggio, in unita' fisiche se possibile."""
    pred = autoregressive_rollout(model, ctx, fut.shape[1])
    if scaler is not None:
        err = scaler.inverse_transform(pred) - scaler.inverse_transform(fut)
    else:
        err = pred - fut
    return float(np.sqrt((err ** 2).mean()))
