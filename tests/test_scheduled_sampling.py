"""
Verifica numerica della proprieta' su cui si regge lo scheduled sampling implementato
in `training.MultiRegimeTrainer`:

    l'iterazione a punto fisso di `_scheduled_inputs` con K passaggi riproduce
    esattamente la sostituzione sequenziale sulle prime K+1 posizioni, perche' la
    mappa di sostituzione e' strettamente triangolare inferiore (il modello e'
    causale: la posizione i dipende solo dalle posizioni j < i).

Il test controlla anche che la variante a una sola iterazione non riproduca la
composizione dell'errore: senza questo confronto passerebbe anche un'implementazione
in cui le due modalita' coincidono per errore, rendendo il parametro inutile.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import tensorflow as tf

from data_preprocessing import set_global_seeds
from models import RNNConfig, TransformerConfig, build_model
from training import TrainConfig, MultiRegimeTrainer


def reference_scheduled_sampling(model, x, use_gt):
    """Scheduled sampling vero: sostituzione sequenziale, un istante alla volta."""
    x_mix = np.array(x, dtype=np.float32)
    L = x.shape[1]
    for i in range(1, L):                      #la posizione 0 resta ground truth
        if use_gt[0, i, 0] > 0.5:
            continue
        pred = model(tf.convert_to_tensor(x_mix), training=False).numpy()
        x_mix[:, i, :] = pred[:, i - 1, :]
    return x_mix


def run_case(kind, cfg, L=8, F=3, seed=0):
    set_global_seeds(seed)
    model = build_model(kind, F, cfg)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, L, F)).astype("float32")
    #posizione 0 sempre ground truth,poi si alterna per forzare la composizione
    gt = np.zeros((1, L, 1), dtype="float32")
    gt[0, 0, 0] = 1.0
    gt[0, 5, 0] = 1.0
    use_gt = tf.convert_to_tensor(gt)

    trainer = MultiRegimeTrainer(model, TrainConfig(verbose=False))
    ref = reference_scheduled_sampling(model, x, gt)

    print(f"\n=== {kind} ({cfg}) ===")
    ok_all = True
    for K in range(1, L):
        got = trainer._scheduled_inputs(tf.convert_to_tensor(x),
                                        tf.constant(0.0, tf.float32),
                                        tf.constant(K, tf.int32),
                                        use_gt=use_gt).numpy()
        #le prime K+1 posizioni devono coincidere con il riferimento
        head_err = np.abs(got[:, :K + 1, :] - ref[:, :K + 1, :]).max()
        full_err = np.abs(got - ref).max()
        ok = head_err < 2e-5
        ok_all &= ok
        print(f"  K={K}: max|err| sulle prime {K+1} posizioni = {head_err:.2e}  "
              f"| su tutta la finestra = {full_err:.2e}  -> {'OK' if ok else 'FAIL'}")
    #La variante a una sola iterazione deve differire su tutta la finestra
    got1 = trainer._scheduled_inputs(tf.convert_to_tensor(x),
                                     tf.constant(0.0, tf.float32),
                                     tf.constant(1, tf.int32),
                                     use_gt=use_gt).numpy()
    gotK = trainer._scheduled_inputs(tf.convert_to_tensor(x),
                                     tf.constant(L - 1, tf.int32),
                                     tf.constant(L - 1, tf.int32),
                                     use_gt=use_gt).numpy()
    diff = np.abs(got1 - gotK).max()
    print(f"  differenza fra K=1 e K=L-1: {diff:.4e}  "
          f"-> {'DIVERSE (atteso)' if diff > 1e-6 else 'IDENTICHE (PROBLEMA)'}")
    conv = np.abs(gotK - ref).max()
    print(f"  convergenza completa a K=L-1: max|err| = {conv:.2e}  "
          f"-> {'OK' if conv < 2e-5 else 'FAIL'}")
    return ok_all and diff > 1e-6 and conv < 2e-5


if __name__ == "__main__":
    results = []
    results.append(run_case("transformer",
                            TransformerConfig(d_model=8, num_heads=2, num_layers=1, dff=16,
                                              dropout=0.0)))
    results.append(run_case("rnn", RNNConfig(rnn_type="LSTM", units=6, num_layers=1)))
    results.append(run_case("rnn", RNNConfig(rnn_type="GRU", units=6, num_layers=2)))
    print("\nTUTTI I TEST PASSATI" if all(results) else "\nALMENO UN TEST FALLITO")
    sys.exit(0 if all(results) else 1)
