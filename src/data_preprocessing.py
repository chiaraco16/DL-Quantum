"""
DL-Quantum -- Step 1: ingestione, validazione e preparazione dei dati.

Il modulo copre l'intero percorso dal CSV grezzo ai tensori di addestramento per le
traiettorie del modello PXP a 10 qubit (55 osservabili: 10 magnetizzazioni e 45
correlazioni): lettura e riorganizzazione in tensore 3D, controlli di integrita' fisica
e temporale, partizionamento per traiettoria, standardizzazione stimata sul solo
training, estrazione delle finestre scorrevoli e costruzione delle coppie
(contesto, futuro) per il rollout.

Due scelte attraversano tutto il file. La prima e' che ogni partizione avviene a
livello di traiettoria e mai di finestra: due finestre della stessa traiettoria
condividono la dinamica, quindi separarle fra train e test darebbe un errore
ottimisticamente distorto. La seconda e' il controllo del determinismo: semi fissati
per Python, NumPy e TensorFlow e, dove la piattaforma lo consente, kernel deterministici,
in modo che due esecuzioni della stessa configurazione producano gli stessi numeri.
"""
from __future__ import annotations

import os
import json
import random
import warnings
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd

N_QUBITS_DEFAULT = 10
N_POINTS_DEFAULT = 1001
DT_DEFAULT = 0.02
T_FIN_DEFAULT = 20.0
EPS = 1e-8


def n_features(n_qubits: int = N_QUBITS_DEFAULT) -> int:
    return n_qubits + n_qubits * (n_qubits - 1) // 2


def build_feature_names(n_qubits: int = N_QUBITS_DEFAULT):
    mag = [f"m_{i}" for i in range(1, n_qubits + 1)]
    cor = [f"c_{i}_{j}" for i in range(1, n_qubits + 1) for j in range(i + 1, n_qubits + 1)]
    return mag, cor, mag + cor



#Riproducibilità

_OP_DETERMINISM_ENABLED = False
OP_DETERMINISM_STATUS = "non_tentato"


def set_global_seeds(seed: int = 42, verbose_on_failure: bool = True) -> int:
    global _OP_DETERMINISM_ENABLED, OP_DETERMINISM_STATUS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except Exception as exc:
        OP_DETERMINISM_STATUS = f"tensorflow_non_disponibile({type(exc).__name__})"
        if verbose_on_failure:
            warnings.warn("[Riproducibilità] TensorFlow non importabile: impostati solo i seed di Python/NumPy.")
        return seed

    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

    if not _OP_DETERMINISM_ENABLED:
        try:
            tf.config.experimental.enable_op_determinism()
            _OP_DETERMINISM_ENABLED = True
            OP_DETERMINISM_STATUS = "enabled"
        except Exception as exc:
            OP_DETERMINISM_STATUS = f"FALLITO ({type(exc).__name__}: {exc})"
            if verbose_on_failure:
                warnings.warn(
                    "[Riproducibilità] tf.config.experimental.enable_op_determinism() FALLITO: "
                    f"{exc}. I risultati NON saranno riproducibili bit-a-bit su questa macchina.")
    return seed


def reproducibility_report(seed: int) -> dict:
    #Report generato automaticamente per certificare i parametri di riproducibilità
    import sys
    rep = {"seed": seed,
           "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
           "op_determinism": OP_DETERMINISM_STATUS,
           "python": sys.version.split()[0],
           "numpy": np.__version__}
    try:
        import tensorflow as tf
        rep["tensorflow"] = tf.__version__
        rep["devices"] = [d.device_type for d in tf.config.list_physical_devices()]
    except Exception:
        rep["tensorflow"] = None
    return rep



#JSON valido

def json_safe(obj):
    """
    Garantisce la serializzazione in standard JSON puro.
    Converte i tipi NumPy nei corrispettivi nativi Python e mappa i valori
    non finiti (NaN, Inf) in `null`, prevenendo errori di parsing nei tool 
    di visualizzazione web-based.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def dump_json(obj, path):
    #Esporta il dizionario in formato JSON
    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2, allow_nan=False, default=str)
    return path



#Configurazione della Pipeline

@dataclass
class PreprocessConfig:
    csv_path: str
    n_qubits: int = N_QUBITS_DEFAULT
    n_points: int = N_POINTS_DEFAULT
    dt: float = DT_DEFAULT
    t_fin: float = T_FIN_DEFAULT
    feature_set: Union[str, Sequence[int]] = "all"
    input_window: int = 50
    stride: Optional[int] = None            #Stride in fase di valutazione: None -> L (nessuna sovrapposizione)
    train_stride: Optional[int] = None      #Stride in addestramento: None -> L. Valori < L generano sovrapposizione.
    horizon: int = 100
    max_windows: Optional[int] = None
    subsample_time: int = 1
    max_trajectories: Optional[int] = None
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    n_folds: int = 5
    scaler_type: str = "per_feature_zscore"
    rollout_origins: Sequence[Union[str, int]] = field(default_factory=lambda: ["start"])
    seed: int = 42
    dtype: str = "float32"
    max_rows: Optional[int] = None

    def resolved_stride(self) -> int:
        return self.input_window if self.stride is None else self.stride

    def resolved_train_stride(self) -> int:
        return self.input_window if self.train_stride is None else self.train_stride



#Caricamento e Reshaping

def load_raw_trajectories(cfg: PreprocessConfig):
    """
    Carica i dati grezzi dal file CSV e li riorganizza in un tensore 3D continuo: 
    (numero_traiettorie, punti_temporali, numero_feature).
    """
    dtype = np.dtype(cfg.dtype)
    df = pd.read_csv(cfg.csv_path, header=None, index_col=False, dtype=dtype, nrows=cfg.max_rows)
    arr = df.values
    if arr.shape[1] != 1 + n_features(cfg.n_qubits):
        raise ValueError(
            f"Attese {1 + n_features(cfg.n_qubits)} colonne "
            f"(1 tempo + {n_features(cfg.n_qubits)} feature fisiche), trovate {arr.shape[1]}.")
    time_col = arr[:, 0]
    feats = arr[:, 1:]
    starts = np.where(time_col == 0.0)[0]
    n_traj = len(starts)
    if n_traj * cfg.n_points != arr.shape[0]:
        raise ValueError(
            f"Il numero di righe {arr.shape[0]} non corrisponde a N_traj*N_points"
            f"({n_traj}*{cfg.n_points}); il file potrebbe essere troncato o corrotto.")
    traj = feats.reshape(n_traj, cfg.n_points, n_features(cfg.n_qubits))
    time_grid = time_col[: cfg.n_points].copy()
    return traj.astype(dtype, copy=False), time_grid.astype(dtype, copy=False), starts



#Controlli di Integrità e Consistenza

def validate_trajectories(traj, time_grid, starts, cfg: PreprocessConfig, strict: bool = True) -> dict:
    """
    Controlli di integrita' sul tensore grezzo, prima di qualunque modellazione.

    Verifica la geometria attesa, l'assenza di NaN e infiniti, la regolarita' del
    reticolo temporale e i limiti fisici |m|, |c| <= 1 delle osservabili. L'uniformita'
    di dt non e' un dettaglio formale: un passo del modello corrisponde a un intervallo
    di tempo fisico, e se il campionamento non fosse uniforme la stessa rete
    rappresenterebbe salti temporali diversi in punti diversi della traiettoria.
    Con `strict=True` una violazione interrompe l'esecuzione; i limiti fisici restano
    invece un controllo non bloccante, perche' un piccolo sforamento numerico e' atteso.
    """
    report: dict = {}
    N, P, F = traj.shape
    report["n_traiettorie"] = int(N)
    report["n_punti"] = int(P)
    report["n_feature"] = int(F)

    def _check(cond, msg):
        cond = bool(cond)
        if not cond:
            if strict:
                raise AssertionError(msg)
            warnings.warn(msg)
        return cond

    report["ok_n_feature"] = _check(F == n_features(cfg.n_qubits),
                                     f"Numero di feature {F} != {n_features(cfg.n_qubits)}")
    report["ok_n_punti"] = _check(P == cfg.n_points, f"N_punti {P} != {cfg.n_points}")

    expected_starts = np.arange(N) * P
    report["ok_trajectory_starts"] = _check(
        np.array_equal(starts, expected_starts),
        "Le righe di inizio traiettoria sono irregolari: un tempo '0' appare a metà "
        "o una traiettoria non possiede esattamente N_points righe.")

    n_nan = int(np.isnan(traj).sum())
    n_inf = int(np.isinf(traj).sum())
    report["n_nan"] = n_nan
    report["n_inf"] = n_inf
    report["ok_finite"] = _check(n_nan == 0 and n_inf == 0,
                                 f"Trovati {n_nan} valori NaN e {n_inf} valori Inf.")
    report["n_duplicate_rows"] = int(N * P - np.unique(traj.reshape(N * P, F), axis=0).shape[0]) \
        if N * P <= 500000 else -1

    d = np.diff(time_grid)
    report["dt_inferred"] = float(np.median(d))
    report["t_fin"] = float(time_grid[-1])
    report["ok_time_start_zero"] = _check(np.isclose(time_grid[0], 0.0), "time_grid[0] != 0")
    report["ok_uniform_dt"] = _check(np.allclose(d, d[0], atol=1e-4), "Time steps non uniformi")
    report["ok_dt_value"] = _check(np.isclose(report["dt_inferred"], cfg.dt, atol=1e-3),
                                   f"Il dt inferito {report['dt_inferred']} != {cfg.dt}")

    #Limiti fisici |m|,|c| <= 1: controllo non bloccante (soft check)
    mag = traj[:, :, : cfg.n_qubits]
    cor = traj[:, :, cfg.n_qubits:]
    report["mag_min"], report["mag_max"] = float(mag.min()), float(mag.max())
    report["cor_min"], report["cor_max"] = float(cor.min()), float(cor.max())
    report["ok_mag_bounds"] = bool(mag.min() >= -1 - 1e-3 and mag.max() <= 1 + 1e-3)
    report["ok_cor_bounds"] = bool(cor.min() >= -1 - 1e-3 and cor.max() <= 1 + 1e-3)
    if not (report["ok_mag_bounds"] and report["ok_cor_bounds"]):
        warnings.warn("Alcune osservabili eccedono il limite fisico |x|<=1; verificare le unità di simulazione.")

    x0_spread = float(traj[:, 0, :].std(axis=0).mean())
    report["x0_spread_across_traj"] = x0_spread
    report["ok_random_initial_states"] = _check(
        x0_spread > 0, "Tutte le traiettorie condividono lo stesso stato iniziale (comportamento inatteso).")

    report["all_checks_passed"] = all(v for k, v in report.items() if k.startswith("ok_"))
    return report



#Selezione delle variabili fisiche

def resolve_feature_indices(cfg: PreprocessConfig) -> np.ndarray:
    F = n_features(cfg.n_qubits)
    nq = cfg.n_qubits
    fs = cfg.feature_set
    if isinstance(fs, (list, tuple, np.ndarray)):
        idx = np.asarray(fs, dtype=int)
        if idx.min() < 0 or idx.max() >= F:
            raise ValueError("Indici delle feature espliciti fuori intervallo.")
        return idx
    if fs == "all":
        return np.arange(F)
    if fs == "magnetizations":
        return np.arange(nq)
    if fs == "correlations":
        return np.arange(nq, F)
    raise ValueError(f"Feature_set sconosciuto={fs!r}")


def feature_group_ids(feature_idx: np.ndarray, n_qubits: int) -> np.ndarray:
    return (np.asarray(feature_idx) >= n_qubits).astype(int)



#Partizionamento traiettorie e cross-validation

def split_trajectories(n_traj: int, cfg: PreprocessConfig):
    """Partiziona gli indici di traiettoria in train/validation/test.

    La separazione e' per traiettoria e non per finestra: finestre estratte dalla stessa
    simulazione condividono lo stesso stato iniziale e la stessa dinamica, quindi
    distribuirle fra le partizioni equivarrebbe a valutare il modello su dati che ha
    gia' visto.
    """
    total = cfg.train_frac + cfg.val_frac + cfg.test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Le frazioni train/val/test devono sommare a 1, trovato {total}.")
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_traj)
    n_test = int(round(cfg.test_frac * n_traj))
    n_val = int(round(cfg.val_frac * n_traj))
    test = np.sort(perm[:n_test])
    val = np.sort(perm[n_test:n_test + n_val])
    train = np.sort(perm[n_test + n_val:])
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    return train, val, test


def kfold_trajectory_indices(traj_ids: np.ndarray, n_folds: int, seed: int) -> Iterator[tuple]:
    rng = np.random.default_rng(seed)
    ids = np.array(traj_ids)
    rng.shuffle(ids)
    num_val_samples = len(ids) // n_folds
    for k in range(n_folds):
        val = ids[k * num_val_samples:(k + 1) * num_val_samples]
        train = np.concatenate([ids[:k * num_val_samples],
                                ids[(k + 1) * num_val_samples:]], axis=0)
        yield k, np.sort(train), np.sort(val)



#Feature scaler

class FeatureScaler:
    r"""
    Standardizzazione delle osservabili, con media e deviazione stimate sul solo training.

    Lo z-score per singola feature, $z = (x - \mu)/\sigma$, e' la scelta di riferimento:
    lo squilibrio di ampiezza fra le 55 osservabili e' soprattutto interno ai due gruppi,
    non fra magnetizzazioni e correlazioni, e una normalizzazione per gruppo lo lascerebbe
    intatto. Il punto e' che la loss e' quadratica: se le deviazioni standard differiscono
    di un fattore $r$, i contributi alla loss differiscono di $r^2$, e le osservabili piu'
    ampie determinerebbero da sole la direzione del gradiente. Dopo lo z-score per feature
    il rapporto scende a 1 per costruzione. Le varianti `global` e `per_group` restano
    disponibili come termine di confronto.
    """
    def __init__(self, scaler_type: str = "per_feature_zscore",
                 group_ids: Optional[np.ndarray] = None):
        self.scaler_type = scaler_type
        self.group_ids = None if group_ids is None else np.asarray(group_ids)
        self.mean_ = None
        self.std_ = None

    def fit(self, train_data: np.ndarray) -> "FeatureScaler":
        flat = train_data.reshape(-1, train_data.shape[-1]).astype(np.float64)
        F = flat.shape[1]
        if self.scaler_type == "per_feature_zscore":
            self.mean_ = flat.mean(0)
            self.std_ = flat.std(0)
        elif self.scaler_type == "global":
            self.mean_ = np.full(F, flat.mean())
            self.std_ = np.full(F, flat.std())
        elif self.scaler_type == "per_group":
            if self.group_ids is None:
                raise ValueError("lo scaler per_group necessita di group_ids.")
            self.mean_ = np.empty(F)
            self.std_ = np.empty(F)
            for g in np.unique(self.group_ids):
                m = self.group_ids == g
                self.mean_[m] = flat[:, m].mean()
                self.std_[m] = flat[:, m].std()
        else:
            raise ValueError(f"scaler_type sconosciuto={self.scaler_type!r}")
        self.std_ = np.where(self.std_ < EPS, 1.0, self.std_)
        self.mean_ = self.mean_.astype(np.float32)
        self.std_ = self.std_.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std_ + self.mean_).astype(np.float32)

    def to_dict(self) -> dict:
        return {"scaler_type": self.scaler_type,
                "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        s = cls(d["scaler_type"])
        s.mean_ = np.asarray(d["mean"], dtype=np.float32)
        s.std_ = np.asarray(d["std"], dtype=np.float32)
        return s



#Estrazione finestre temporali (Sliding Windows)

def build_window_index(n_traj: int, P: int, L: int, stride: int,
                       max_windows: Optional[int] = None, seed: int = 42) -> np.ndarray:
    """Elenco di coppie (traiettoria, istante iniziale) delle finestre ammissibili.

    L'ultimo inizio consentito e' P-L-1 e non P-L, perche' il target della finestra e'
    la finestra stessa traslata di un istante: serve un valore osservato oltre la fine.
    Le finestre non attraversano mai il confine fra due traiettorie.
    """
    if L < 2:
        raise ValueError("input_window (L) deve essere >= 2.")
    max_start = P - L - 1
    if max_start < 0:
        raise ValueError(f"Finestra L={L} troppo lunga per la lunghezza della sequenza P={P}.")
    origins = []
    for t in range(n_traj):
        s = 0
        while s <= max_start:
            origins.append((t, s))
            s += stride
    origins = np.asarray(origins, dtype=np.int64)
    if max_windows is not None and len(origins) > max_windows:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(origins), size=max_windows, replace=False)
        origins = origins[np.sort(sel)]
    return origins


def windows_to_arrays(data: np.ndarray, index: np.ndarray, L: int, warn_gb: float = 2.0):
    n = len(index)
    F = data.shape[-1]
    est_gb = 2 * n * L * F * np.dtype(data.dtype).itemsize / 1e9
    if est_gb > warn_gb:
        warnings.warn(
            f"windows_to_arrays allocherebbe ~{est_gb:.1f} GB. Considerare uno stride maggiore "
            f"`max_windows`, o l'uso del costruttore lazy make_tf_dataset.")
    t_idx = index[:, 0]
    s_idx = index[:, 1]
    offs = np.arange(L)
    rows = s_idx[:, None] + offs[None, :]
    X = data[t_idx[:, None], rows]
    Y = data[t_idx[:, None], rows + 1]
    return np.ascontiguousarray(X), np.ascontiguousarray(Y)


def window_split(data_scaled_subset: np.ndarray, cfg: PreprocessConfig,
                 stride: Optional[int] = None, max_windows: Optional[int] = None):
    n_traj, P, _ = data_scaled_subset.shape
    stride = cfg.resolved_stride() if stride is None else stride
    idx = build_window_index(n_traj, P, cfg.input_window, stride,
                             max_windows=max_windows, seed=cfg.seed)
    return windows_to_arrays(data_scaled_subset, idx, cfg.input_window)



#Creazione dei tensori di Rollout

def make_forecast_pairs(data_scaled_subset: np.ndarray, context_L: int, horizon_H: int,
                        origin: Union[str, int, Sequence] = 0):
    """
    Coppie (contesto, futuro) per la valutazione del rollout autoregressivo.

    Per ogni origine `o` il contesto sono gli istanti [o, o+L) e il target gli H istanti
    successivi, [o+L, o+L+H). Le origini multiple evitano che il giudizio dipenda dal
    solo transitorio iniziale: la dinamica del PXP attraversa regimi diversi, e un
    modello puo' essere accurato sulle prime oscillazioni e non sul resto.

    Va tenuto presente che il target dipende da L: a parita' di origine, una finestra di
    contesto piu' lunga sposta in avanti l'intervallo da prevedere. Il confronto fra L
    diversi non e' quindi sullo stesso identico problema di previsione, ed e' un limite
    da dichiarare quando si commentano i risultati.
    """
    N, P, F = data_scaled_subset.shape
    origins = origin if isinstance(origin, (list, tuple, np.ndarray)) else [origin]
    ctx_list, fut_list = [], []
    for o_raw in origins:
        o = 0 if o_raw in ("start", 0) else int(o_raw)
        if o + context_L + horizon_H > P:
            raise ValueError(
                f"origine({o})+L({context_L})+H({horizon_H}) > P({P}); accorciare l'orizzonte "
                f"or rimuovere questa origine.")
        ctx_list.append(data_scaled_subset[:, o:o + context_L, :])
        fut_list.append(data_scaled_subset[:, o + context_L:o + context_L + horizon_H, :])
    context = np.concatenate(ctx_list, axis=0)
    future = np.concatenate(fut_list, axis=0)
    return np.ascontiguousarray(context), np.ascontiguousarray(future)


def resolve_origins(cfg: PreprocessConfig, P: int) -> list:
    """Mantiene solo le origini che rientrano nella sequenza per gli attuali (L, H)."""
    H = min(cfg.horizon, P - cfg.input_window)
    out = []
    for o_raw in cfg.rollout_origins:
        o = 0 if o_raw in ("start", 0) else int(o_raw)
        if o + cfg.input_window + H <= P:
            out.append(o)
    return out or [0]



#Helper per la mascheratura (Masked Modeling)


def make_timestep_mask(n_windows: int, L: int, mask_prob: float = 0.15,
                       seed: int = 42) -> np.ndarray:
    """Maschera booleana riproducibile (True = istante mascherato), forma (n_windows, L).

    Serve alle figure illustrative: durante l'addestramento la maschera viene estratta
    direttamente in TensorFlow, batch per batch, per non ripetere lo stesso schema.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n_windows, L)) < mask_prob



#Generazione pipeline tf.data (Ottimizzazione memoria)

def make_tf_dataset(data_scaled_subset: np.ndarray, index: np.ndarray, L: int,
                    batch_size: int = 64, shuffle: bool = True, seed: int = 42,
                    drop_remainder: bool = False):
    """Costruttore lazy delle finestre: le estrae al volo invece di materializzarle.

    Alternativa a `windows_to_arrays` per configurazioni in cui l'insieme delle finestre
    non entrerebbe in memoria (stride piccolo e molte traiettorie). Gli esperimenti di
    questo progetto usano la versione materializzata, piu' veloce alla scala usata qui.
    """
    import tensorflow as tf
    data_t = tf.constant(data_scaled_subset)
    idx_t = tf.constant(index)
    offs = tf.range(L)

    def _gather(pair):
        t, s = pair[0], pair[1]
        rows = s + offs
        seq = tf.gather(data_t[t], rows)
        tgt = tf.gather(data_t[t], rows + 1)
        return seq, tgt

    ds = tf.data.Dataset.from_tensor_slices(idx_t)
    if shuffle:
        ds = ds.shuffle(len(index), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_gather, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
    return ds



#Analisi dello sbilanciamento delle feature

def imbalance_report(traj_selected: np.ndarray, feature_idx: np.ndarray,
                     n_qubits: int) -> dict:
    flat = traj_selected.reshape(-1, traj_selected.shape[-1])
    std = flat.std(0)
    gid = feature_group_ids(feature_idx, n_qubits)
    rep = {
        "per_feature_std": std.tolist(),
        "std_max": float(std.max()),
        "std_min": float(std.min()),
        "std_ratio_max_over_min": float(std.max() / max(std.min(), EPS)),
    }
    for g, name in [(0, "magnetizations"), (1, "correlations")]:
        m = gid == g
        if m.any():
            rep[f"{name}_std_mean"] = float(std[m].mean())
            rep[f"{name}_abs_mean"] = float(np.abs(flat[:, m]).mean())
    return rep



#Orchestratore principale (Pipeline builder)

@dataclass
class PreparedData:
    cfg: PreprocessConfig
    report: dict
    feature_idx: np.ndarray
    feature_names: list
    time_grid: np.ndarray
    traj_selected: np.ndarray
    scaler: FeatureScaler
    splits: dict
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    ctx_test: np.ndarray
    fut_test: np.ndarray
    imbalance: dict
    origins: list

    def summary(self) -> str:
        def mb(a):
            return a.nbytes / 1e6
        ratio = self.imbalance["std_ratio_max_over_min"]
        after = " -> 1.00x dopo lo z-score per singola feature" \
            if self.scaler.scaler_type == "per_feature_zscore" else ""
        lines = [
            f"finestra di input (L)  : {self.cfg.input_window}  "
            f"(stride train {self.cfg.resolved_train_stride()}, "
            f"stride valutazione {self.cfg.resolved_stride()})",
            f"osservabili            : {len(self.feature_idx)}  ({self.cfg.feature_set})",
            f"traiettorie (tr/va/te) : {len(self.splits['train'])}/"
            f"{len(self.splits['val'])}/{len(self.splits['test'])}",
            f"X_train / Y_train      : {self.X_train.shape} / {self.Y_train.shape}"
            f"  ({mb(self.X_train)+mb(self.Y_train):.0f} MB)",
            f"X_val   / Y_val        : {self.X_val.shape} / {self.Y_val.shape}",
            f"X_test  / Y_test       : {self.X_test.shape} / {self.Y_test.shape}",
            f"rollout ctx/fut (test) : {self.ctx_test.shape} / {self.fut_test.shape}"
            f"  (origini {self.origins})",
            f"scaler                 : {self.scaler.scaler_type} "
            f"(stimato su {len(self.splits['train'])} traiettorie di train)",
            f"rapporto std fra feature: {ratio:.2f}x prima dello scaling{after}",
        ]
        return "\n".join(lines)


def build_datasets(cfg: PreprocessConfig, verbose: bool = True,
                   strict_checks: bool = True) -> PreparedData:
    set_global_seeds(cfg.seed)

    traj, time_grid, starts = load_raw_trajectories(cfg)
    report = validate_trajectories(traj, time_grid, starts, cfg, strict=strict_checks)
    if verbose:
        print(f"[Caricamento] tensore grezzo {traj.shape} | controlli superati: {report['all_checks_passed']}")

    if cfg.max_trajectories is not None:
        traj = traj[: cfg.max_trajectories]
    fidx = resolve_feature_indices(cfg)
    traj = traj[:, :, fidx]
    _, _, all_names = build_feature_names(cfg.n_qubits)
    feat_names = [all_names[i] for i in fidx]
    if cfg.subsample_time > 1:
        traj = traj[:, :: cfg.subsample_time, :]
        time_grid = time_grid[:: cfg.subsample_time]
    P = traj.shape[1]

    imb = imbalance_report(traj, fidx, cfg.n_qubits)
    tr, va, te = split_trajectories(traj.shape[0], cfg)

    scaler = FeatureScaler(cfg.scaler_type,
                           group_ids=feature_group_ids(fidx, cfg.n_qubits)).fit(traj[tr])
    data_s = scaler.transform(traj)

    Xtr, Ytr = window_split(data_s[tr], cfg, stride=cfg.resolved_train_stride(),
                            max_windows=cfg.max_windows)
    Xva, Yva = window_split(data_s[va], cfg, stride=cfg.resolved_stride())
    Xte, Yte = window_split(data_s[te], cfg, stride=cfg.resolved_stride())

    H = min(cfg.horizon, P - cfg.input_window)
    origins = resolve_origins(cfg, P)
    ctx, fut = make_forecast_pairs(data_s[te], cfg.input_window, H, origin=origins)

    prepared = PreparedData(
        cfg=cfg, report=report, feature_idx=fidx, feature_names=feat_names,
        time_grid=time_grid, traj_selected=traj, scaler=scaler,
        splits={"train": tr, "val": va, "test": te},
        X_train=Xtr, Y_train=Ytr, X_val=Xva, Y_val=Yva, X_test=Xte, Y_test=Yte,
        ctx_test=ctx, fut_test=fut, imbalance=imb, origins=origins,
    )
    if verbose:
        print(prepared.summary())
    return prepared



#Salvataggio e persistenza dei tensori

def save_prepared(prepared: PreparedData, out_dir: str, tag: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    tag = tag or f"L{prepared.cfg.input_window}_{prepared.cfg.feature_set}"
    npz = os.path.join(out_dir, f"prepared_{tag}.npz")
    np.savez_compressed(
        npz,
        X_train=prepared.X_train, Y_train=prepared.Y_train,
        X_val=prepared.X_val, Y_val=prepared.Y_val,
        X_test=prepared.X_test, Y_test=prepared.Y_test,
        ctx_test=prepared.ctx_test, fut_test=prepared.fut_test,
        time_grid=prepared.time_grid,
        train_ids=prepared.splits["train"], val_ids=prepared.splits["val"],
        test_ids=prepared.splits["test"], feature_idx=prepared.feature_idx,
    )
    cfg_meta = asdict(prepared.cfg)
    # Si registra il solo nome del file: il percorso assoluto della macchina su cui e'
    # stata lanciata la pipeline renderebbe i metadati non portabili.
    cfg_meta["csv_path"] = os.path.basename(cfg_meta.get("csv_path", ""))
    meta = {
        "config": cfg_meta,
        "feature_names": prepared.feature_names,
        "integrity_report": prepared.report,
        "imbalance_report": prepared.imbalance,
        "scaler": prepared.scaler.to_dict(),
        "origins": list(prepared.origins),
        "reproducibility": reproducibility_report(prepared.cfg.seed),
    }
    dump_json(meta, os.path.join(out_dir, f"meta_{tag}.json"))
    return npz



#Analisi esplorativa e plotting (EDA)

def run_eda(prepared: PreparedData, out_dir: str, dpi: int = 300) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    traj = prepared.traj_selected
    t = prepared.time_grid
    fidx = prepared.feature_idx
    nq = prepared.cfg.n_qubits
    gid = feature_group_ids(fidx, nq)
    names = prepared.feature_names
    saved = []

    def _save(fig, name):
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=dpi, format="jpg", bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    mag_cols = np.where(gid == 0)[0]
    cor_cols = np.where(gid == 1)[0]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tr0 = traj[0]
    for c in mag_cols[:4]:
        ax[0].plot(t, tr0[:, c], lw=1, label=names[c])
    ax[0].set_title("Sample trajectory - magnetizations")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("value"); ax[0].legend(fontsize=7)
    for c in cor_cols[:4]:
        ax[1].plot(t, tr0[:, c], lw=1, label=names[c])
    ax[1].set_title("Sample trajectory - correlations")
    ax[1].set_xlabel("time"); ax[1].legend(fontsize=7)
    fig.suptitle("EDA 1 - example dynamics (trajectory #0)")
    _save(fig, "eda1_sample_trajectory.jpg")

    std = traj.reshape(-1, traj.shape[-1]).std(0)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    colors = ["#1f77b4" if g == 0 else "#d62728" for g in gid]
    ax.bar(range(len(std)), std, color=colors)
    ax.set_title(f"EDA 2 - per-feature standard deviation (blue=mag, red=corr) | "
                 f"max/min ratio = {std.max()/max(std.min(),EPS):.2f}x")
    ax.set_xlabel("feature index"); ax.set_ylabel("std")
    _save(fig, "eda2_feature_std_imbalance.jpg")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(traj[:, :, mag_cols].ravel(), bins=80, alpha=0.6, density=True,
            label="magnetizations", color="#1f77b4")
    ax.hist(traj[:, :, cor_cols].ravel(), bins=80, alpha=0.6, density=True,
            label="correlations", color="#d62728")
    ax.set_title("EDA 3 - value distribution per group")
    ax.set_xlabel("value"); ax.set_ylabel("density"); ax.legend()
    _save(fig, "eda3_value_distributions.jpg")

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in list(mag_cols[:2]) + list(cor_cols[:2]):
        mu = traj[:, :, c].mean(0)
        sd = traj[:, :, c].std(0)
        ax.plot(t, mu, lw=1.2, label=names[c])
        ax.fill_between(t, mu - sd, mu + sd, alpha=0.15)
    ax.set_title("EDA 4 - ensemble mean and standard deviation over time")
    ax.set_xlabel("time"); ax.set_ylabel("value"); ax.legend(fontsize=7)
    _save(fig, "eda4_ensemble_envelope.jpg")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(traj[:, 0, :].ravel(), bins=60, color="#555555")
    ax.set_title(f"EDA 5 - spread of the initial state (t=0) across "
                 f"{traj.shape[0]} trajectories")
    ax.set_xlabel("value at t=0"); ax.set_ylabel("count")
    _save(fig, "eda5_initial_state_spread.jpg")

    #Matrice di correlazione tra le osservabili fisiche
    flat = traj.reshape(-1, traj.shape[-1])
    sub = flat[:: max(1, len(flat) // 20000)]
    C = np.corrcoef(sub.T)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.axhline(nq - 0.5, color="k", lw=0.8); ax.axvline(nq - 0.5, color="k", lw=0.8)
    ax.set_title("EDA 6 - inter-feature correlation matrix")
    ax.set_xlabel("feature index"); ax.set_ylabel("feature index")
    fig.colorbar(im, ax=ax, shrink=0.85)
    _save(fig, "eda6_feature_correlation.jpg")

    return saved


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg = PreprocessConfig(csv_path=sys.argv[1])
        prep = build_datasets(cfg)
        print("\n" + prep.summary())