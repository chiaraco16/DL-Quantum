"""
Protocollo di valutazione holdout: una sola partizione train/validation.

Perche' questo file esiste
--------------------------
La versione precedente del progetto selezionava le configurazioni con una 5-fold
cross-validation eseguita sull'unione train+val, e teneva un test set separato letto
una volta sola. E' un protocollo corretto, ma ha un difetto didattico: non esiste
un insieme di validazione fisso, quindi non esiste *una* curva di validazione da
guardare accanto alla curva di training. L'errore di generalizzazione si legge solo
alla fine, come media sui fold.

Qui il protocollo e' quello elementare e completamente ispezionabile:

    80% delle traiettorie -> training     (il modello ci addestra i pesi)
    20% delle traiettorie -> validation   (il modello non ci addestra mai;
                                           serve a misurare, a scegliere l'epoca
                                           migliore e a scegliere la configurazione)

Non c'e' un test set. E' una scelta consapevole e va dichiarata: le metriche finali
vengono dalla stessa partizione usata per selezionare, quindi sono ottimisticamente
distorte. La distorsione e' piccola qui (si sceglie fra 8 configurazioni, non fra
migliaia) ma non e' zero, e il report deve dirlo invece di nasconderlo.

Cosa resta invariato rispetto a prima
-------------------------------------
- La separazione e' per TRAIETTORIA, non per finestra: due finestre estratte dalla
  stessa simulazione condividono stato iniziale e dinamica, e distribuirle fra le
  due partizioni equivarrebbe a validare su dati gia' visti.
- Lo scaler (z-score per feature) e' stimato SOLO sulle traiettorie di training e
  poi applicato a entrambe le partizioni. La validation non contribuisce ne' alla
  media ne' alla deviazione standard.
- Le finestre non attraversano mai il confine fra due traiettorie.

Cosa viene aggiunto
-------------------
Un sottoinsieme di traiettorie di TRAINING della stessa numerosita' della validation
("train_eval"), usato per calcolare sul training le stesse identiche metriche che si
calcolano sulla validation. Senza questo, le due curve non sarebbero confrontabili:
la loss di training e' quella del regime corrente (mascherata o autogenerata), la
metrica di validation e' sempre next-step teacher forcing. Confrontare le due
direttamente e' l'errore di lettura piu' comune su questo tipo di grafico.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Sequence
import os

import numpy as np

from data_preprocessing import (
    PreprocessConfig, load_raw_trajectories, validate_trajectories,
    resolve_feature_indices, build_feature_names, feature_group_ids,
    imbalance_report, FeatureScaler, window_split, make_forecast_pairs,
    resolve_origins, set_global_seeds, dump_json,
)


# Partizionamento

def split_train_val(n_traj: int, val_frac: float = 0.20, seed: int = 42):
    """Partiziona gli indici di traiettoria in due soli insiemi: train e validation.

    La permutazione dipende dal solo `seed`: con lo stesso seme la partizione e'
    identica fra esecuzioni e fra finestre temporali diverse, quindi il confronto
    fra L=50 e L=100 avviene sulle stesse traiettorie e non su partizioni diverse.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac deve stare in (0,1), ricevuto {val_frac}.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_traj)
    n_val = int(round(val_frac * n_traj))
    if n_val < 2 or n_val >= n_traj:
        raise ValueError(f"val_frac={val_frac} produce {n_val} traiettorie di validation "
                         f"su {n_traj}: partizione degenere.")
    val = np.sort(perm[:n_val])
    train = np.sort(perm[n_val:])
    assert len(set(train.tolist()) & set(val.tolist())) == 0
    return train, val


def subsample_ids(ids: np.ndarray, size: int, seed: int = 42) -> np.ndarray:
    """Sottoinsieme riproducibile di traiettorie, usato per il monitoraggio del training.

    Il monitoraggio sul training non gira su tutte le 320 traiettorie ma su un campione
    della stessa numerosita' della validation: la metrica costa quanto quella di
    validation e, soprattutto, le due sono confrontabili anche come rumore campionario.
    """
    ids = np.asarray(ids)
    if size >= len(ids):
        return np.sort(ids)
    rng = np.random.default_rng(seed + 777)
    sel = rng.choice(len(ids), size=size, replace=False)
    return np.sort(ids[sel])


@dataclass
class HoldoutData:
    """Tutti i tensori del protocollo 80/20, gia' scalati."""
    cfg: PreprocessConfig
    val_frac: float
    report: dict
    imbalance: dict
    feature_idx: np.ndarray
    feature_names: list
    time_grid: np.ndarray
    traj_selected: np.ndarray
    scaler: FeatureScaler
    train_ids: np.ndarray
    val_ids: np.ndarray
    train_eval_ids: np.ndarray
    # finestre next-step (input -> stessa finestra traslata di un istante)
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_train_eval: np.ndarray
    Y_train_eval: np.ndarray
    # coppie (contesto, futuro) per il rollout autoregressivo
    ctx_val: np.ndarray
    fut_val: np.ndarray
    ctx_train_eval: np.ndarray
    fut_train_eval: np.ndarray
    origins: list

    @property
    def horizon(self) -> int:
        return int(self.fut_val.shape[1])

    def summary(self) -> str:
        def mb(*arrs):
            return sum(a.nbytes for a in arrs) / 1e6
        L = self.cfg.input_window
        return "\n".join([
            f"finestra di input (L)     : {L}  (stride train {self.cfg.resolved_train_stride()}, "
            f"stride valutazione {self.cfg.resolved_stride()})",
            f"osservabili               : {len(self.feature_idx)} ({self.cfg.feature_set})",
            f"traiettorie train / val   : {len(self.train_ids)} / {len(self.val_ids)}  "
            f"({100*(1-self.val_frac):.0f}% / {100*self.val_frac:.0f}%)",
            f"sottocampione di train    : {len(self.train_eval_ids)} traiettorie "
            f"(monitoraggio, stessa numerosita' della validation)",
            f"X_train / Y_train         : {self.X_train.shape} / {self.Y_train.shape} "
            f"({mb(self.X_train, self.Y_train):.0f} MB)",
            f"X_val   / Y_val           : {self.X_val.shape} / {self.Y_val.shape}",
            f"rollout val  ctx/fut      : {self.ctx_val.shape} / {self.fut_val.shape} "
            f"(origini {self.origins}, H={self.horizon})",
            f"scaler                    : {self.scaler.scaler_type}, stimato sulle "
            f"{len(self.train_ids)} traiettorie di training",
            f"nessun test set           : le metriche finali vengono dalla validation "
            f"(distorsione da selezione dichiarata nel report)",
        ])

    def meta(self) -> dict:
        cfg_meta = asdict(self.cfg)
        cfg_meta["csv_path"] = os.path.basename(cfg_meta.get("csv_path", ""))
        return {
            "protocol": "holdout_train_val",
            "val_frac": self.val_frac,
            "n_train_trajectories": int(len(self.train_ids)),
            "n_val_trajectories": int(len(self.val_ids)),
            "train_ids": self.train_ids.tolist(),
            "val_ids": self.val_ids.tolist(),
            "train_eval_ids": self.train_eval_ids.tolist(),
            "config": cfg_meta,
            "feature_names": self.feature_names,
            "integrity_report": self.report,
            "imbalance_report": self.imbalance,
            "scaler": self.scaler.to_dict(),
            "origins": list(self.origins),
            "horizon": self.horizon,
        }


def prepare_holdout(cfg: PreprocessConfig, val_frac: float = 0.20,
                    verbose: bool = True, strict_checks: bool = True,
                    raw_cache: Optional[tuple] = None) -> HoldoutData:
    """Costruisce i tensori del protocollo 80/20 a partire dal CSV grezzo.

    `raw_cache` permette di riusare il tensore grezzo gia' caricato quando si eseguono
    piu' finestre temporali di seguito: il CSV pesa centinaia di MB e rileggerlo per
    ogni valore di L sarebbe tempo speso senza ragione.
    """
    set_global_seeds(cfg.seed)

    if raw_cache is None:
        traj, time_grid, starts = load_raw_trajectories(cfg)
        report = validate_trajectories(traj, time_grid, starts, cfg, strict=strict_checks)
    else:
        traj, time_grid, report = raw_cache
    if verbose:
        print(f"[dati] tensore grezzo {traj.shape} | "
              f"controlli superati: {report.get('all_checks_passed')}")

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

    # 1) partizione per traiettoria
    tr_ids, va_ids = split_train_val(traj.shape[0], val_frac=val_frac, seed=cfg.seed)
    tr_eval_ids = subsample_ids(tr_ids, size=len(va_ids), seed=cfg.seed)

    # 2) scaler stimato SOLO sul training, poi applicato a tutto
    gid = feature_group_ids(fidx, cfg.n_qubits)
    scaler = FeatureScaler(cfg.scaler_type, group_ids=gid).fit(traj[tr_ids])
    data_s = scaler.transform(traj)

    # 3) finestre next-step. Il training usa lo stride di addestramento, le partizioni
    #    di valutazione usano lo stride di valutazione (nessuna sovrapposizione).
    Xtr, Ytr = window_split(data_s[tr_ids], cfg, stride=cfg.resolved_train_stride(),
                            max_windows=cfg.max_windows)
    Xva, Yva = window_split(data_s[va_ids], cfg, stride=cfg.resolved_stride())
    Xtre, Ytre = window_split(data_s[tr_eval_ids], cfg, stride=cfg.resolved_stride())

    # 4) coppie per il rollout autoregressivo, dalle stesse traiettorie
    H = min(cfg.horizon, P - cfg.input_window)
    origins = resolve_origins(cfg, P)
    ctx_va, fut_va = make_forecast_pairs(data_s[va_ids], cfg.input_window, H, origin=origins)
    ctx_tre, fut_tre = make_forecast_pairs(data_s[tr_eval_ids], cfg.input_window, H,
                                           origin=origins)

    data = HoldoutData(
        cfg=cfg, val_frac=val_frac, report=report, imbalance=imb,
        feature_idx=fidx, feature_names=feat_names, time_grid=time_grid,
        traj_selected=traj, scaler=scaler,
        train_ids=tr_ids, val_ids=va_ids, train_eval_ids=tr_eval_ids,
        X_train=Xtr, Y_train=Ytr, X_val=Xva, Y_val=Yva,
        X_train_eval=Xtre, Y_train_eval=Ytre,
        ctx_val=ctx_va, fut_val=fut_va,
        ctx_train_eval=ctx_tre, fut_train_eval=fut_tre,
        origins=origins,
    )
    if verbose:
        print(data.summary())
    return data


def save_holdout_meta(data: HoldoutData, out_dir: str, tag: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    tag = tag or f"L{data.cfg.input_window}"
    path = os.path.join(out_dir, f"holdout_meta_{tag}.json")
    dump_json(data.meta(), path)
    return path
