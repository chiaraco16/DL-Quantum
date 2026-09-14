"""
DL-Quantum -- training loop multi-regime e valutazione.

Un unico ciclo di addestramento scritto con `tf.GradientTape` attraversa in sequenza
tre regimi, senza mai ricostruire ne' ricompilare il modello:

* teacher_forcing    : l'input e' la finestra di ground truth, il target la stessa
                       finestra traslata di un istante. E' il regime piu' stabile ma
                       addestra il modello su una condizione che al rollout non esiste.
* masked_modeling    : una frazione `mask_prob` degli istanti di input viene azzerata
                       (in unita' z-score lo zero coincide con la media di training,
                       quindi il valore mascherato non introduce un livello anomalo) e
                       la loss pesa di piu' le posizioni coinvolte. Il modello impara a
                       ricostruire il segnale da un contesto incompleto.
* scheduled_sampling : una frazione crescente degli istanti di input viene sostituita
                       dalle previsioni del modello stesso, avvicinando le condizioni
                       di addestramento a quelle del rollout autoregressivo e riducendo
                       l'exposure bias.

Non e' previsto early stopping: il budget di epoche e' fisso e uguale per tutte le
configurazioni, cosi' il confronto fra iperparametri avviene a parita' di calcolo e la
selezione resta affidata alla cross-validation. Ordine, durata dei regimi e parametri
di ciascuno sono definiti in `TrainConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import time

import numpy as np
import tensorflow as tf

from data_preprocessing import set_global_seeds, feature_group_ids


#Configurazione dell'addestramento


@dataclass
class TrainConfig:
    schedule: Sequence[tuple] = field(default_factory=lambda: [
        ("teacher_forcing", 6), ("masked_modeling", 6), ("scheduled_sampling", 3)])
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"
    clipnorm: float = 1.0
    # masked modeling
    mask_prob: float = 0.15           # frazione attesa di istanti azzerati
    mask_loss_weight: float = 3.0     # peso aggiuntivo: 1+3 = 4 volte le altre posizioni
    mask_mode: str = "denoise_next"   # "denoise_next": pesa la predizione fatta a partire
                                      #   da un istante mascherato (denoising);
                                      # "reconstruct": pesa la predizione del valore
                                      #   mascherato stesso (ricostruzione).
    # scheduled sampling
    ss_mode: str = "true"             # "true": iterazione a punto fisso con ss_passes
                                      #   passaggi, equivale alla sostituzione sequenziale;
                                      # "two_pass": un solo passaggio di sostituzione,
                                      #   variante piu' economica e meno fedele.
    ss_passes: int = 3
    ss_tf_prob_start: float = 1.0     # probabilita' iniziale di tenere il valore reale
    ss_tf_prob_end: float = 0.5       # probabilita' finale (decadimento lineare)
    seed: int = 42
    verbose: bool = True


def make_optimizer(cfg: TrainConfig):
    kw = {"learning_rate": cfg.lr}
    if cfg.clipnorm:
        kw["clipnorm"] = cfg.clipnorm
    return {"adam": tf.keras.optimizers.Adam,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "sgd": tf.keras.optimizers.SGD}[cfg.optimizer](**kw)


#Utility per tf.data

def arrays_to_dataset(X, Y, batch_size=64, shuffle=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), Y.astype("float32")))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



#Funzioni di loss

def _mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def _mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def _weighted_mse(y_true, y_pred, w):
    se = tf.square(y_true - y_pred)
    return tf.reduce_sum(w * se) / (tf.reduce_sum(w) *
                                    tf.cast(tf.shape(y_true)[-1], tf.float32))


#Trainer unificato


class MultiRegimeTrainer:
    """Modello, ottimizzatore ed esecuzione dello schedule dei regimi.

    I tre passi di addestramento sono compilati una volta sola, alla prima chiamata di
    `fit`, con una firma che lascia indeterminate la dimensione del batch e la lunghezza
    della finestra: l'ultimo batch di ogni epoca e' piu' corto degli altri e senza firma
    esplicita ogni epoca ne provocherebbe una nuova tracciatura.
    """

    def __init__(self, model: tf.keras.Model, cfg: TrainConfig, n_features=None):
        self.model = model
        self.cfg = cfg
        self.opt = make_optimizer(cfg)
        self._compiled_for = None
        if n_features is not None:
            self._build_steps(int(n_features))

    def _build_steps(self, n_features: int):
        if self._compiled_for == n_features:
            return
        seq = tf.TensorSpec(shape=(None, None, n_features), dtype=tf.float32)
        scalar_f = tf.TensorSpec(shape=(), dtype=tf.float32)
        scalar_b = tf.TensorSpec(shape=(), dtype=tf.bool)
        scalar_i = tf.TensorSpec(shape=(), dtype=tf.int32)

        @tf.function(input_signature=[seq, seq])
        def step_teacher(x, y):
            with tf.GradientTape() as tape:
                pred = self.model(x, training=True)
                loss = _mse(y, pred)
            self._apply(tape, loss)
            return loss

        @tf.function(input_signature=[seq, seq, scalar_f, scalar_f, scalar_b])
        def step_mask(x, y, mask_prob, mask_w, reconstruct):
            # m[b,t]=1 se l'istante t dell'input e' mascherato. La maschera e' estratta
            # a ogni batch e a ogni epoca: il modello non puo' memorizzare quali
            # posizioni mancano.
            m = tf.cast(tf.random.uniform(tf.shape(x)[:2])[..., None] < mask_prob, x.dtype)
            x_masked = x * (1.0 - m)
            # y[t] = x[t+1]: la predizione del valore mascherato x[t+1] si trova in
            # posizione t, quindi la maschera va traslata indietro di uno per pesare la
            # ricostruzione invece del denoising.
            m_shift = tf.concat([m[:, 1:, :], tf.zeros_like(m[:, :1, :])], axis=1)
            m_eff = tf.where(reconstruct, m_shift, m)
            w = 1.0 + mask_w * m_eff
            with tf.GradientTape() as tape:
                pred = self.model(x_masked, training=True)
                loss = _weighted_mse(y, pred, w)
            self._apply(tape, loss)
            return loss

        @tf.function(input_signature=[seq, seq, scalar_f, scalar_i])
        def step_sched(x, y, tf_prob, n_passes):
            x_mix = self._scheduled_inputs(x, tf_prob, n_passes)
            with tf.GradientTape() as tape:
                pred = self.model(x_mix, training=True)
                loss = _mse(y, pred)
            self._apply(tape, loss)
            return loss

        self._step_teacher = step_teacher
        self._step_mask = step_mask
        self._step_sched = step_sched
        self._compiled_for = n_features

    def _scheduled_inputs(self, x, tf_prob, n_passes, use_gt=None):
        """Costruisce l'input misto dello scheduled sampling.

        Le posizioni estratte con probabilita' `tf_prob` restano ground truth, le altre
        vengono sostituite dalla previsione del modello per quell'istante. La
        sostituzione sequenziale vera e' costosa (un passaggio per istante), ma poiche'
        il modello e' causale la mappa di sostituzione e' strettamente triangolare
        inferiore: iterandola K volte si ottiene esattamente il risultato sequenziale
        sulle prime K+1 posizioni, e con K = L-1 su tutta la finestra. Il numero di
        iterazioni diventa cosi' un iperparametro che scambia fedelta' con costo
        (`ss_passes`). Le previsioni entrano con `stop_gradient`: il gradiente scorre
        solo attraverso la passata di addestramento, non attraverso la generazione
        degli input.
        """
        if use_gt is None:
            use_gt = tf.cast(tf.random.uniform(tf.shape(x)[:2])[..., None] < tf_prob,
                             x.dtype)
        x_mix = x
        for _ in tf.range(n_passes):
            pred = tf.stop_gradient(self.model(x_mix, training=False))
            shifted = tf.concat([x[:, :1, :], pred[:, :-1, :]], axis=1)
            x_mix = use_gt * x + (1.0 - use_gt) * shifted
        return x_mix

    def _apply(self, tape, loss):
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def _run_epoch(self, ds, regime, tf_prob):
        meter = tf.keras.metrics.Mean()
        for xb, yb in ds:
            if regime == "teacher_forcing":
                loss = self._step_teacher(xb, yb)
            elif regime == "masked_modeling":
                loss = self._step_mask(
                    xb, yb,
                    tf.constant(self.cfg.mask_prob, tf.float32),
                    tf.constant(self.cfg.mask_loss_weight, tf.float32),
                    tf.constant(self.cfg.mask_mode == "reconstruct", tf.bool))
            elif regime == "scheduled_sampling":
                # "two_pass" = una sola iterazione di sostituzione (una passata di
                # generazione piu' quella di addestramento), "true" = ss_passes iterazioni.
                n_passes = self.cfg.ss_passes if self.cfg.ss_mode == "true" else 1
                loss = self._step_sched(xb, yb,
                                        tf.constant(tf_prob, tf.float32),
                                        tf.constant(int(n_passes), tf.int32))
            else:
                raise ValueError(f"Regime sconosciuto: {regime!r}")
            meter.update_state(loss)
        return float(meter.result())

    def fit(self, X_train, Y_train, X_val, Y_val):
        set_global_seeds(self.cfg.seed)
        self._build_steps(X_train.shape[-1])
        train_ds = arrays_to_dataset(X_train, Y_train, self.cfg.batch_size,
                                     shuffle=True, seed=self.cfg.seed)
        val_ds = arrays_to_dataset(X_val, Y_val, self.cfg.batch_size, shuffle=False)

        history = []
        epoch = 0
        for regime, n_ep in self.cfg.schedule:
            for e in range(n_ep):
                if regime == "scheduled_sampling":
                    # Decadimento lineare della probabilita' di usare il valore reale:
                    # con 3 epoche e gli estremi di default la sequenza e' 0.83, 0.67, 0.50.
                    # L'ultima epoca lascia meta' degli istanti alla ground truth: e' una
                    # transizione volutamente conservativa verso il regime free-running.
                    frac = (e + 1) / n_ep
                    tf_prob = self.cfg.ss_tf_prob_start + frac * (
                        self.cfg.ss_tf_prob_end - self.cfg.ss_tf_prob_start)
                else:
                    tf_prob = self.cfg.ss_tf_prob_start
                t0 = time.time()
                tr_loss = self._run_epoch(train_ds, regime, tf_prob)
                va = evaluate_nextstep(self.model, val_ds)
                rec = {"epoch": epoch, "regime": regime, "train_loss": tr_loss,
                       "val_mse": va["mse"], "val_mae": va["mae"],
                       "tf_prob": round(float(tf_prob), 3),
                       "sec": round(time.time() - t0, 1)}
                history.append(rec)
                if self.cfg.verbose:
                    print(f"  [{epoch:02d}] {regime:18s} "
                          f"train_loss={tr_loss:.5f}  val_mse={va['mse']:.5f}  "
                          f"val_mae={va['mae']:.5f}  tf_p={tf_prob:.2f}  ({rec['sec']}s)")
                epoch += 1
        return history


#Valutazione a un passo (next-step)

def evaluate_nextstep(model, ds, scaler=None) -> dict:
    """Calcola le metriche di errore aggregate su un dataset batched."""
    se_sum = ae_sum = n = 0.0
    per_feat_se = None
    for xb, yb in ds:
        pred = model(xb, training=False)
        diff = yb - pred
        se_sum += float(tf.reduce_sum(tf.square(diff)))
        ae_sum += float(tf.reduce_sum(tf.abs(diff)))
        n += float(tf.size(diff))
        pf = tf.reduce_sum(tf.square(diff), axis=[0, 1])
        per_feat_se = pf.numpy() if per_feat_se is None else per_feat_se + pf.numpy()
    if per_feat_se is None or n == 0.0:
        raise ValueError("evaluate_nextstep: dataset vuoto.")
    F = per_feat_se.shape[0]
    out = {"mse": se_sum / n, "mae": ae_sum / n, "rmse": float(np.sqrt(se_sum / n)),
           "per_feature_mse": per_feat_se / (n / F)}
    if scaler is not None:
        rmse_f = np.sqrt(out["per_feature_mse"])
        out["rmse_physical"] = float(np.sqrt(np.mean((rmse_f * scaler.std_) ** 2)))
    return out


#Valutazione del rollout autoregressivo

def autoregressive_rollout(model, context, horizon: int, batch_size: int = 256) -> np.ndarray:
    """Rollout libero: H previsioni consecutive a partire dal solo contesto.

    Dopo il primo passo il modello non vede piu' alcun valore reale: la finestra scorre
    e ogni nuova previsione viene reimmessa in coda al posto del valore osservato. E' la
    condizione in cui l'errore si accumula e in cui l'exposure bias diventa visibile.
    """
    outs = []
    for i in range(0, len(context), batch_size):
        seq = tf.convert_to_tensor(context[i:i + batch_size], dtype=tf.float32)
        preds = []
        for _ in range(horizon):
            nxt = model(seq, training=False)[:, -1, :]
            preds.append(nxt)
            seq = tf.concat([seq[:, 1:, :], nxt[:, None, :]], axis=1)
        outs.append(tf.stack(preds, axis=1).numpy())
    return np.concatenate(outs, axis=0)


def evaluate_rollout(model, context, future, scaler=None, feature_idx=None,
                     n_qubits=10, baselines: Optional[dict] = None) -> dict:
    """Metriche del rollout, in unita' scalate e fisiche.

    Oltre all'errore aggregato produce la curva per orizzonte, la scomposizione fra
    magnetizzazioni e correlazioni, l'NRMSE (errore diviso per la dispersione del
    target) e, se vengono passate le baseline, lo skill score rispetto alla migliore.
    """
    H = future.shape[1]
    pred = autoregressive_rollout(model, context, H)
    err = pred - future
    per_h_rmse = np.sqrt((err ** 2).mean(axis=(0, 2)))
    out = {"rollout_rmse": float(np.sqrt((err ** 2).mean())),
           "rollout_mae": float(np.abs(err).mean()),
           "per_horizon_rmse": per_h_rmse}
    if scaler is not None:
        pred_p = scaler.inverse_transform(pred)
        fut_p = scaler.inverse_transform(future)
        errp = pred_p - fut_p
        out["rollout_rmse_physical"] = float(np.sqrt((errp ** 2).mean()))
        out["rollout_mae_physical"] = float(np.abs(errp).mean())
        out["per_horizon_rmse_physical"] = np.sqrt((errp ** 2).mean(axis=(0, 2)))
        sigma = float(np.sqrt(max((fut_p ** 2).mean() - fut_p.mean() ** 2, 0.0)))
        out["rollout_nrmse"] = float(out["rollout_rmse_physical"] / max(sigma, 1e-12))
        out["target_sigma_physical"] = sigma
        if feature_idx is not None:
            gid = feature_group_ids(feature_idx, n_qubits)
            for g, name in [(0, "magnetizations"), (1, "correlations")]:
                m = gid == g
                if m.any():
                    out[f"rmse_physical_{name}"] = float(
                        np.sqrt((errp[:, :, m] ** 2).mean()))
    if baselines:
        best_key = min(baselines, key=lambda k: baselines[k]["rollout_rmse_physical"])
        best = baselines[best_key]["rollout_rmse_physical"]
        out["best_baseline"] = best_key
        out["best_baseline_rmse_physical"] = float(best)
        out["skill_vs_baseline"] = float(1.0 - out.get("rollout_rmse_physical",
                                                       out["rollout_rmse"]) / max(best, 1e-12))
    return out


#Funzioni di generazione delle figure

def plot_rollout(model, context, future, time_grid, feature_names, out_path,
                 traj_idx=0, feats=(0, 1, 10, 11), dpi=300, origin=0):
    """Confronto grafico fra traiettoria reale e rollout su alcune osservabili.

    `origin` e' l'istante da cui parte il contesto della coppia mostrata: serve solo a
    etichettare correttamente l'asse dei tempi quando le coppie di rollout non partono
    dall'inizio della traiettoria.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = context.shape[1]
    H = future.shape[1]
    F = context.shape[2]
    feats = [c for c in feats if c < F] or list(range(min(4, F)))
    pred = autoregressive_rollout(model, context[traj_idx:traj_idx + 1], H)[0]
    o = int(origin) if int(origin) + L + H <= len(time_grid) else 0
    t_ctx = time_grid[o:o + L]
    t_fut = time_grid[o + L:o + L + H]

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in feats:
        gt_full = np.concatenate([context[traj_idx, :, c], future[traj_idx, :, c]])
        t_full = np.concatenate([t_ctx, t_fut])
        line, = ax.plot(t_full, gt_full, lw=1.3, label=f"{feature_names[c]} (true)")
        ax.plot(t_fut, pred[:, c], ".", ms=3, color=line.get_color(),
                label=f"{feature_names[c]} (predicted)")
    ax.axvline(t_ctx[-1], color="k", ls="--", lw=0.8)
    ax.text(t_ctx[-1], ax.get_ylim()[1], " context | forecast", va="top", fontsize=8)
    ax.set_title(f"Autoregressive rollout (context L={L} -> forecast H={H}), "
                 f"trajectory #{traj_idx}")
    ax.set_xlabel("time"); ax.set_ylabel("value (scaled)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_history(history, out_path, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    va = [h["val_mse"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ep, tr, "-o", ms=3, label="train loss")
    ax.plot(ep, va, "-s", ms=3, label="validation MSE")
    colors = {"teacher_forcing": "#d9f0d3", "masked_modeling": "#fde0dd",
              "scheduled_sampling": "#e0ecf4"}
    start = 0
    for i, h in enumerate(history):
        if i == len(history) - 1 or history[i + 1]["regime"] != h["regime"]:
            ax.axvspan(start - 0.5, h["epoch"] + 0.5,
                       color=colors.get(h["regime"], "#eeeeee"), alpha=0.6, zorder=0)
            ax.text((start + h["epoch"]) / 2, ax.get_ylim()[1], h["regime"],
                    ha="center", va="top", fontsize=7)
            start = h["epoch"] + 1
    ax.set_xlabel("epoch"); ax.set_ylabel("loss / MSE"); ax.legend()
    ax.set_title("Multi-regime learning curve")
    fig.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_regimes_illustration(X, out_path, cfg: TrainConfig, feature=0, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data_preprocessing import make_timestep_mask

    x = X[:1]
    L = x.shape[1]
    mask = make_timestep_mask(1, L, cfg.mask_prob, seed=cfg.seed)[0]
    x_masked = x.copy()
    x_masked[0, mask, :] = 0.0
    rng = np.random.default_rng(cfg.seed)
    use_gt = rng.random(L) < cfg.ss_tf_prob_end

    fig, ax = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    ax[0].plot(x[0, :, feature], "-o", ms=3, color="#1f77b4")
    ax[0].set_title("(a) teacher forcing: the model is fed the ground-truth window",
                    fontsize=10)
    ax[1].plot(x_masked[0, :, feature], "-o", ms=3, color="#d62728")
    ax[1].plot(np.where(mask)[0], x_masked[0, mask, feature], "x", ms=9, color="k",
               label=f"masked timesteps (p={cfg.mask_prob})")
    ax[1].legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax[1].set_title(f"(b) masked modeling: zeroed timesteps, weighted "
                    f"{1 + cfg.mask_loss_weight:g}x in the loss", fontsize=10)
    ax[2].plot(x[0, :, feature], "-", lw=1, color="#999999", label="ground truth")
    ax[2].plot(np.where(use_gt)[0], x[0, use_gt, feature], "o", ms=4,
               color="#1f77b4", label="input = true value")
    ax[2].plot(np.where(~use_gt)[0], x[0, ~use_gt, feature], "s", ms=4,
               color="#ff7f0e", label="input = model prediction")
    ax[2].legend(fontsize=7, loc="lower right", framealpha=0.9, ncol=3)
    ax[2].set_title(f"(c) scheduled sampling: mixed inputs, "
                    f"p(teacher) = {cfg.ss_tf_prob_end}", fontsize=10)
    ax[2].set_xlabel("timestep within the window")
    fig.suptitle("The three training regimes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    return out_path