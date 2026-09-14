"""
DL-Quantum -- Predittori non parametrici di riferimento (baseline banali).

Un RMSE letto da solo non e' interpretabile: serve un termine di paragone che non
impari nulla dai dati. Questo modulo raccoglie i predittori banali contro cui vengono
confrontati i modelli parametrici, sia sul task a un passo sia sul rollout
autoregressivo, e il Forecast Skill Score che ne misura il divario.

Predittori implementati (identici nei due task, cambia solo l'orizzonte):

* persistence   : x_hat[t+k] = ultimo valore osservato
* drift         : x_hat[t+k] = ultimo valore + k volte l'ultimo incremento (random walk con drift)
* context_mean  : x_hat[t+k] = media dei valori gia' osservati
* train_mean    : x_hat[t+k] = media del set di addestramento (0 in unita' z-score)

Sul task a un passo la media di contesto e' calcolata in forma espansiva (media dei
soli istanti fino a t incluso): usare la media dell'intera finestra darebbe al
predittore banale informazione futura e ne falserebbe il confronto.

Il modulo dipende solo da numpy ed e' eseguibile senza TensorFlow.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# Baseline sul rollout: dal contesto (B, L, F) a H passi futuri (B, H, F)
# --------------------------------------------------------------------------

def baseline_persistence(context: np.ndarray, horizon: int) -> np.ndarray:
    last = context[:, -1:, :]
    return np.repeat(last, horizon, axis=1)


def baseline_drift(context: np.ndarray, horizon: int) -> np.ndarray:
    if context.shape[1] < 2:
        raise ValueError("La baseline drift richiede un contesto di almeno 2 istanti.")
    last = context[:, -1, :]
    slope = context[:, -1, :] - context[:, -2, :]
    steps = np.arange(1, horizon + 1, dtype=context.dtype)[None, :, None]
    return last[:, None, :] + slope[:, None, :] * steps


def baseline_context_mean(context: np.ndarray, horizon: int) -> np.ndarray:
    mu = context.mean(axis=1, keepdims=True)
    return np.repeat(mu, horizon, axis=1)


def baseline_train_mean(context: np.ndarray, horizon: int,
                        train_mean_scaled: Optional[np.ndarray] = None) -> np.ndarray:
    B, _, F = context.shape
    mu = np.zeros(F, dtype=context.dtype) if train_mean_scaled is None \
        else np.asarray(train_mean_scaled, dtype=context.dtype)
    return np.broadcast_to(mu[None, None, :], (B, horizon, F)).copy()


ROLLOUT_BASELINES = {
    "persistence": baseline_persistence,
    "drift": baseline_drift,
    "context_mean": baseline_context_mean,
    "train_mean": baseline_train_mean,
}


# --------------------------------------------------------------------------
# Metriche
# --------------------------------------------------------------------------

def _metrics(pred: np.ndarray, true: np.ndarray, scaler=None,
             feature_idx=None, n_qubits: int = 10) -> dict:
    err = pred - true
    out = {"rollout_rmse": float(np.sqrt((err ** 2).mean())),
           "rollout_mae": float(np.abs(err).mean()),
           "per_horizon_rmse": np.sqrt((err ** 2).mean(axis=(0, 2)))}
    if scaler is not None:
        # Riporta le previsioni nella scala fisica delle osservabili: e' l'unica in cui
        # l'errore e' confrontabile con l'ampiezza reale di magnetizzazioni e correlazioni.
        pred_p = scaler.inverse_transform(pred)
        true_p = scaler.inverse_transform(true)
        errp = pred_p - true_p
        out["rollout_rmse_physical"] = float(np.sqrt((errp ** 2).mean()))
        out["rollout_mae_physical"] = float(np.abs(errp).mean())
        out["per_horizon_rmse_physical"] = np.sqrt((errp ** 2).mean(axis=(0, 2)))
        sigma = float(np.sqrt(max((true_p ** 2).mean() - true_p.mean() ** 2, 0.0)))
        out["rollout_nrmse"] = float(out["rollout_rmse_physical"] / max(sigma, 1e-12))
        out["target_sigma_physical"] = sigma
        if feature_idx is not None:
            from data_preprocessing import feature_group_ids
            gid = feature_group_ids(feature_idx, n_qubits)
            for g, name in [(0, "magnetizations"), (1, "correlations")]:
                m = gid == g
                if m.any():
                    out[f"rmse_physical_{name}"] = float(np.sqrt((errp[:, :, m] ** 2).mean()))
    else:
        out["rollout_rmse_physical"] = out["rollout_rmse"]
        out["rollout_mae_physical"] = out["rollout_mae"]
    return out


def evaluate_rollout_baselines(context: np.ndarray, future: np.ndarray, scaler=None,
                               feature_idx=None, n_qubits: int = 10,
                               which=("persistence", "drift", "context_mean", "train_mean")) -> dict:
    """Metriche di rollout per ogni baseline, sull'intero orizzonte H."""
    H = future.shape[1]
    res = {}
    for name in which:
        pred = ROLLOUT_BASELINES[name](context, H)
        res[name] = _metrics(pred, future, scaler, feature_idx, n_qubits)
    return res


# --------------------------------------------------------------------------
# Baseline sul task a un passo
# --------------------------------------------------------------------------

def _nextstep_predictions(X: np.ndarray) -> dict:
    """Previsioni a un passo dei quattro predittori banali, tutte causali.

    `X` ha forma (n_finestre, L, F) e il target e' `X` traslato di un istante, quindi
    la previsione in posizione t puo' usare solo gli istanti fino a t incluso.
    """
    persistence = X
    train_mean = np.zeros_like(X)

    # drift: estrapolazione lineare dell'ultimo incremento; in t=0 non esiste un
    # incremento precedente, quindi degrada a persistence.
    drift = np.empty_like(X)
    drift[:, 0, :] = X[:, 0, :]
    drift[:, 1:, :] = 2.0 * X[:, 1:, :] - X[:, :-1, :]

    # context_mean in forma espansiva: media dei soli istanti gia' osservati.
    counts = np.arange(1, X.shape[1] + 1, dtype=X.dtype)[None, :, None]
    context_mean = np.cumsum(X, axis=1) / counts

    return {"persistence": persistence, "drift": drift,
            "context_mean": context_mean, "train_mean": train_mean}


def evaluate_nextstep_baselines(X: np.ndarray, Y: np.ndarray, scaler=None) -> dict:
    """
    Metriche dei predittori banali sul task a singolo passo (t -> t+1).

    Con dati in unita' z-score il predittore `train_mean` ha un RMSE prossimo a 1 per
    costruzione, perche' l'errore coincide con la dispersione standardizzata del target:
    e' il riferimento naturale per capire se un modello ha imparato qualcosa.
    """
    res = {}
    for name, pred in _nextstep_predictions(X).items():
        err = pred - Y
        d = {"mse": float((err ** 2).mean()),
             "mae": float(np.abs(err).mean()),
             "rmse": float(np.sqrt((err ** 2).mean()))}
        if scaler is not None:
            errp = err * scaler.std_
            d["rmse_physical"] = float(np.sqrt((errp ** 2).mean()))
            d["mae_physical"] = float(np.abs(errp).mean())
        res[name] = d
    return res


# --------------------------------------------------------------------------
# Skill score
# --------------------------------------------------------------------------

def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """
    Forecast Skill Score: quanto errore il modello elimina rispetto alla baseline.

        SS = 1 - RMSE_modello / RMSE_baseline

    SS > 0 : il modello batte la baseline;
    SS = 0 : prestazioni equivalenti;
    SS < 0 : la baseline e' preferibile al modello.
    """
    return float(1.0 - model_rmse / max(baseline_rmse, 1e-12))


def best_baseline(baseline_metrics: dict, key: str = "rollout_rmse_physical") -> tuple:
    name = min(baseline_metrics, key=lambda k: baseline_metrics[k][key])
    return name, float(baseline_metrics[name][key])


def summarize_baselines(baseline_metrics: dict, key: str = "rollout_rmse_physical") -> str:
    lines = ["baseline                RMSE(fis.)     NRMSE"]
    for name, m in sorted(baseline_metrics.items(), key=lambda kv: kv[1][key]):
        nr = m.get("rollout_nrmse", float("nan"))
        lines.append(f"  {name:20s} {m[key]:.5f}     {nr:.3f}")
    return "\n".join(lines)
