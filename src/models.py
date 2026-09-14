"""
DL-Quantum -- Step 2: architetture (RNN e Transformer)

Modelli parametrici per il forecasting autoregressivo delle dinamiche quantistiche.
Entrambe le famiglie sono costruite da zero a partire dai soli layer primitivi di
Keras: nessun modello pre-addestrato, nessun peso importato dall'esterno, nessun
blocco Transformer preconfezionato.

1. Rete ricorrente: stack di celle LSTM o GRU assemblato con la Sequential API e
   chiuso da una proiezione lineare per istante. La factory accetta anche SimpleRNN,
   ma gli esperimenti usano solo LSTM e GRU: su orizzonti di 50-100 istanti la cella
   elementare soffre il gradiente evanescente e non e' un termine di paragone utile.
2. Transformer: Model Subclassing con blocchi (proiezione posizionale, self-attention
   causale, feed-forward position-wise) scritti come sottoclassi di `layers.Layer`.
   La mascheratura causale impedisce che l'istante t veda gli istanti successivi.

Nota sui nomi: le classi `Encoder` ed `EncoderLayer` conservano la denominazione
convenzionale dello stack di blocchi self-attention, ma l'attenzione e' mascherata in
senso causale e non esiste cross-attention verso una seconda sequenza. Dal punto di
vista architetturale si tratta quindi del lato decoder di un Transformer, ed e' cosi'
che va letto: "encoder" indica qui la struttura del blocco, non la sua direzionalita'.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



#Configs

@dataclass
class RNNConfig:
    rnn_type: str = "GRU"          #'LSTM' , 'GRU' , 'SimpleRNN'
    units: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    name: str = "rnn_forecaster"


@dataclass
class TransformerConfig:
    d_model: int = 64              #deve essere pari
    num_heads: int = 4
    num_layers: int = 2
    dff: int = 128
    dropout: float = 0.1
    max_len: int = 2048
    key_dim: int | None = None     #None -> d_model
    name: str = "transformer_forecaster"



#RNN forecaster 

def build_rnn_forecaster(n_features: int, cfg: RNNConfig) -> keras.Model:
    """
    Previsore ricorrente: N celle con `return_sequences=True` e una testa lineare.

    Non viene usato alcun wrapper `Bidirectional`: in un task autoregressivo il ramo
    backward leggerebbe gli istanti futuri per produrre la predizione dell'istante
    corrente, cioe' esattamente l'informazione che il modello deve prevedere.

    Args:
        n_features: dimensione dello stato osservato al tempo t.
        cfg: configurazione dei layer ricorrenti (tipo di cella, unita', dropout).
    """
    rnn_cls = {"LSTM": layers.LSTM, "GRU": layers.GRU,
               "SimpleRNN": layers.SimpleRNN}[cfg.rnn_type]

    model = keras.Sequential(name=cfg.name)
    #Input shape:(Batch, Sequence_Length, n_features)
    model.add(layers.Input(shape=(None, n_features)))
    for i in range(cfg.num_layers):
        model.add(rnn_cls(cfg.units,
                          return_sequences=True,
                          dropout=cfg.dropout,
                          recurrent_dropout=cfg.recurrent_dropout,
                          name=f"{cfg.rnn_type.lower()}_{i}"))

    #Proiezione lineare indipendente per ogni time-step (equivalente a TimeDistributed).
    #Assenza di attivazione finale poiché il task è una regressione sui valori continui fisici.
    model.add(layers.Dense(n_features, name="head"))
    return model



# Transformer 
def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """
    Matrice di codifica posizionale sinusoidale (length, depth).

    Meta' delle componenti sono seni e meta' coseni, con frequenze geometricamente
    decrescenti: la self-attention e' invariante alle permutazioni degli istanti,
    quindi l'ordine temporale va iniettato esplicitamente nell'input.
    """
    half = depth // 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(half)[np.newaxis, :] / half
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalProjection(layers.Layer):
    """
    Ingresso del Transformer adattato a variabili continue.

    In NLP l'embedding e' una lookup table su token discreti; qui lo stato fisico e'
    gia' un vettore reale, quindi viene portato a `d_model` da una proiezione lineare
    densa prima di sommare la codifica posizionale. La moltiplicazione per
    sqrt(d_model) e' la convenzione del Transformer originale, dove serve a rendere
    l'embedding confrontabile in ampiezza con il segnale sinusoidale: dopo una
    proiezione lineare il fattore potrebbe essere assorbito dai pesi, ma viene
    mantenuto perche' fissa la scala relativa fra contenuto e posizione fin dalla
    prima epoca, quando i pesi sono ancora piccoli.
    """

    def __init__(self, d_model: int, max_len: int = 2048, **kw):
        super().__init__(**kw)
        if d_model % 2 != 0:
            raise ValueError("d_model deve essere pari per la codifica sinusoidale.")
        self.d_model = d_model
        self.max_len = max_len
        self.projection = layers.Dense(d_model, name="input_proj")
        self.pos_encoding = positional_encoding(length=max_len, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.d_model,)

    def get_config(self):
        c = super().get_config()
        c.update({"d_model": self.d_model, "max_len": self.max_len})
        return c


class BaseAttention(layers.Layer):
    """Blocco comune alle varianti di attention: MultiHeadAttention + Add + LayerNorm."""

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()


class CausalSelfAttention(BaseAttention):
    """
    Self-attention con mascheratura causale: la posizione t puo' attendere solo alle
    posizioni <= t. Senza questo vincolo il modello leggerebbe il futuro della finestra
    e il training loop autoregressivo (teacher forcing, scheduled sampling) perderebbe
    ogni significato, perche' la predizione dell'istante t+1 conterrebbe t+1 stesso.
    """

    def call(self, x, training=None):
        # x: (batch, seq_len, d_model)
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True,
                               training=training)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(layers.Layer):
    """
    Rete feed-forward applicata in modo indipendente a ciascun istante: espande lo
    stato da d_model a dff, applica la non linearita' e ricomprime a d_model.
    E' la parte del blocco che mescola le componenti dello stato fisico, mentre
    l'attention mescola gli istanti.
    """

    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x, training=None):
        x = self.add([x, self.seq(x, training=training)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(layers.Layer):
    """
    Blocco elementare: attention causale seguita da feed-forward, entrambe con
    connessione residuale e normalizzazione applicata dopo la somma (post-norm).
    """

    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1, key_dim=None):
        super().__init__()
        # key_dim assente -> d_model: ogni testa lavora sull'intera dimensione del
        # modello invece che su d_model/num_heads. Costa piu' parametri ma evita che
        # con d_model=32 e 4 teste ciascuna testa si riduca a 8 dimensioni.
        self.self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=key_dim if key_dim is not None else d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x, training=None):
        x = self.self_attention(x, training=training)
        x = self.ffn(x, training=training)
        return x


class Encoder(layers.Layer):
    """Stack di blocchi causali preceduto dalla proiezione posizionale."""

    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 dropout_rate=0.1, max_len=2048, key_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_projection = PositionalProjection(d_model=d_model, max_len=max_len)
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff,
                         dropout_rate=dropout_rate, key_dim=key_dim)
            for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        x = self.pos_projection(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        return x


class TransformerForecaster(keras.Model):
    """
    Transformer causale in versione solo-encoder: uno stack di blocchi self-attention
    mascherati che, per ogni istante della finestra, emette la previsione dell'istante
    successivo. La struttura e' quella di un decoder autoregressivo senza cross-attention,
    quindi i blocchi restano formalmente encoder ma il comportamento e' causale.
    """

    def __init__(self, *, n_features, num_layers, d_model, num_heads, dff,
                 dropout_rate=0.1, max_len=2048, key_dim=None, name=None):
        super().__init__(name=name)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               dropout_rate=dropout_rate, max_len=max_len,
                               key_dim=key_dim)
        self.final_layer = layers.Dense(n_features)

    def call(self, x, training=None):
        # ingresso  (batch, seq_len, n_features)
        # encoder   (batch, seq_len, d_model)
        x = self.encoder(x, training=training)
        # uscita    (batch, seq_len, n_features)
        return self.final_layer(x)


def build_transformer_forecaster(n_features: int, cfg: TransformerConfig) -> keras.Model:
    model = TransformerForecaster(
        n_features=n_features, num_layers=cfg.num_layers, d_model=cfg.d_model,
        num_heads=cfg.num_heads, dff=cfg.dff, dropout_rate=cfg.dropout,
        max_len=cfg.max_len, key_dim=cfg.key_dim, name=cfg.name)
    # Una passata a vuoto materializza i pesi: senza build esplicita un modello
    # sottoclassato non espone trainable_variables e non puo' salvare i pesi.
    _ = model(tf.zeros((1, 8, n_features)), training=False)
    return model



#Factory + salvataggio/caricamento

def build_model(kind: str, n_features: int, cfg) -> keras.Model:
    if kind == "rnn":
        return build_rnn_forecaster(n_features, cfg)
    if kind == "transformer":
        return build_transformer_forecaster(n_features, cfg)
    raise ValueError(f"Tipo di modello sconosciuto: kind={kind!r}")


def count_params(model: keras.Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


def save_model_bundle(model: keras.Model, kind: str, n_features: int, cfg,
                      out_dir: str, tag: str, extra: dict | None = None) -> str:
    """Salva specifica dell'architettura (json), pesi (.weights.h5) e metadati.

    Si salvano i pesi e non il modello serializzato: l'architettura e' ricostruita
    dal codice sorgente tramite la stessa factory usata in addestramento, quindi il
    caricamento non dipende dalla versione di Keras con cui e' stato salvato.
    """
    os.makedirs(out_dir, exist_ok=True)
    spec = {"kind": kind, "n_features": int(n_features), "config": asdict(cfg)}
    if extra:
        spec["extra"] = extra
    with open(os.path.join(out_dir, f"{tag}_spec.json"), "w") as f:
        json.dump(spec, f, indent=2, default=str)
    wpath = os.path.join(out_dir, f"{tag}.weights.h5")
    model.save_weights(wpath)
    return wpath


def load_model_bundle(out_dir: str, tag: str, return_spec: bool = False):
    """Ricostruisce il modello dalla specifica salvata e vi carica i pesi."""
    with open(os.path.join(out_dir, f"{tag}_spec.json")) as f:
        spec = json.load(f)
    cfg_cls = RNNConfig if spec["kind"] == "rnn" else TransformerConfig
    # Si ignorano eventuali campi non piu' presenti nella dataclass, cosi' un bundle
    # salvato con una versione precedente resta caricabile.
    valid = {f.name for f in fields(cfg_cls)}
    cfg_kwargs = {k: v for k, v in spec["config"].items() if k in valid}
    cfg = cfg_cls(**cfg_kwargs)
    model = build_model(spec["kind"], spec["n_features"], cfg)
    model.load_weights(os.path.join(out_dir, f"{tag}.weights.h5"))
    return (model, spec) if return_spec else model
