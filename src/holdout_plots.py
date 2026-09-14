"""
Figure del protocollo holdout 80/20.

Ogni figura risponde a una domanda precisa; se una figura non risponde a nessuna
domanda non va nel report. Le domande, nell'ordine in cui conviene guardarle:

  fig_learning_curves   -> il modello sta imparando? e sta sovradattando?
  fig_val_overlay       -> quale configurazione generalizza meglio, e da quale epoca?
  fig_gap               -> quanto vale il divario di generalizzazione, epoca per epoca?
  fig_nextstep_vs_rollout -> l'errore a un passo e quello a 200 passi vanno insieme?
  fig_train_vs_val_rollout -> il rollout e' difficile anche sui dati di training?
  fig_config_comparison -> le differenze fra configurazioni sono maggiori del rumore?
  fig_error_growth      -> come cresce l'errore con l'orizzonte, e dove supera le baseline?
  fig_group_breakdown   -> l'errore e' distribuito uniformemente sulle osservabili?

Tutte le figure sono salvate in JPEG a 300 dpi come richiede la traccia.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DPI = 300
# Coppia di colori delle curve di loss: blu per il training, arancio per la
# validation. Blu/arancio e' la coppia piu' leggibile anche in stampa in bianco e
# nero e per chi ha difficolta' con la distinzione rosso/verde.
C_TR = "#1f4e79"
C_VA = "#e08214"
REGIME_BANDS = {
    "teacher_forcing": ("#d9f0d3", "teacher forcing"),
    "masked_modeling": ("#fde0dd", "masked modeling"),
    "scheduled_sampling": ("#e0ecf4", "scheduled sampling"),
}
C_TRAIN = "#9aa0a6"
C_VAL = "#1f4e79"
C_OBJ = "#c0c0c0"
FAMILY_COLORS = {"rnn": plt.get_cmap("Blues"), "transformer": plt.get_cmap("Oranges")}


def _save(fig, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, format="jpg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _shade_regimes(ax, history, label=False):
    """Sfondo colorato per i tre regimi dello schedule.

    I nomi dei regimi non vengono scritti dentro i pannelli (si sovrapporrebbero alle
    curve e fra loro quando le bande sono strette): finiscono nella legenda della
    figura, costruita da `_regime_handles`.
    """
    start = 0
    for i, h in enumerate(history):
        last = (i == len(history) - 1) or (history[i + 1]["regime"] != h["regime"])
        if last:
            color, name = REGIME_BANDS.get(h["regime"], ("#eeeeee", h["regime"]))
            ax.axvspan(start - 0.5, h["epoch"] + 0.5, color=color, alpha=0.55, zorder=0)
            if label:
                ax.text((start + h["epoch"]) / 2, 0.98, name, transform=_blend(ax),
                        ha="center", va="top", fontsize=6.5, color="#444444")
            start = h["epoch"] + 1


def _regime_handles(history):
    """Rettangoli colorati da usare in legenda, uno per regime presente nello schedule."""
    from matplotlib.patches import Patch
    seen, handles = [], []
    for h in history:
        if h["regime"] in seen:
            continue
        seen.append(h["regime"])
        color, name = REGIME_BANDS.get(h["regime"], ("#eeeeee", h["regime"]))
        handles.append(Patch(facecolor=color, alpha=0.8, label=name))
    return handles


def _blend(ax):
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transData, ax.transAxes)


def _series(history, key):
    ep = [h["epoch"] for h in history if h.get(key) is not None]
    va = [h[key] for h in history if h.get(key) is not None]
    return ep, va


def _family_color(kind, i, n):
    cmap = FAMILY_COLORS.get(kind, plt.get_cmap("Greys"))
    return cmap(0.35 + 0.55 * (i / max(n - 1, 1)))


# 0. Diagnosi automatica della forma della curva

def diagnose_curve(history, tail_frac: float = 0.33) -> dict:
    """Classifica la curva di validation nei casi canonici del manuale.

    Le soglie sono dichiarate qui una volta sola e valgono per tutte le
    configurazioni: la diagnosi e' una regola, non un giudizio a occhio.

    - `risalita`  : di quanto la validation risale dopo il suo minimo, in rapporto
                    al minimo stesso. Sopra il 5% si parla di sovradattamento.
    - `pendenza`  : calo relativo medio per epoca nell'ultimo terzo. Sopra l'1,5%
                    per epoca la curva sta ancora scendendo: budget insufficiente.
    - `divario`   : (val - train) / val a fine addestramento.
    - `rumore`    : dispersione delle differenze fra epoche consecutive, in
                    rapporto al livello della curva.
    """
    tr = np.array([h["train_mse"] for h in history], dtype=float)
    va = np.array([h["val_mse"] for h in history], dtype=float)
    n = len(va)
    if n < 4:
        return {"verdetto": "curva troppo corta per una diagnosi", "note": []}

    i_min = int(np.argmin(va))
    v_min = float(va[i_min])
    risalita = float((va[-1] - v_min) / max(v_min, 1e-12))

    k = max(3, int(round(tail_frac * n)))
    coda = va[-k:]
    pendenza = float((coda[0] - coda[-1]) / max(coda[0], 1e-12) / max(k - 1, 1))
    divario = float((va[-1] - tr[-1]) / max(abs(va[-1]), 1e-12))
    rumore = float(np.std(np.diff(coda)) / max(np.mean(coda), 1e-12))

    note = []
    if risalita > 0.05 and i_min < n - 2:
        verdetto = "sovradattamento"
        note.append(f"la validation risale del {100*risalita:.0f}% dopo l'epoca {i_min}")
        note.append("i pesi conservati sono quelli dell'epoca del minimo, non dell'ultima")
    elif pendenza > 0.015:
        verdetto = "sottoadattamento (budget insufficiente)"
        note.append(f"nell'ultimo terzo la validation scende ancora "
                    f"del {100*pendenza:.1f}% per epoca")
        note.append("le conclusioni valgono sotto questo budget di epoche, non in assoluto")
    elif divario < -0.02:
        verdetto = "validation piu' bassa del training"
        note.append("accade quando la partizione di validation e' piu' facile di quella "
                    "di training, o quando il training e' misurato con regimi piu' duri")
    else:
        verdetto = "buon adattamento"
        note.append(f"la validation si e' stabilizzata (calo {100*pendenza:.1f}% per epoca "
                    f"nell'ultimo terzo)")
    if rumore > 0.10:
        note.append(f"curva rumorosa (dispersione {100*rumore:.0f}% del livello): "
                    "campione di validation piccolo o learning rate alto")
    if divario > 0.25:
        note.append(f"divario finale {100*divario:.0f}% della validation")

    return {"verdetto": verdetto, "note": note, "epoca_minimo": i_min,
            "risalita": risalita, "pendenza_coda": pendenza,
            "divario_finale": divario, "rumore": rumore}


# 0-bis. Come si leggono le curve: tavola dei casi canonici

def plot_curve_taxonomy(out_path):
    """I sei casi canonici delle curve di loss, disegnati e diagnosticati.

    Non dipende dai risultati: e' la tavola di riferimento da mettere accanto alle
    curve vere per dire in quale caso ci si trova. Le curve sono generate
    analiticamente, non sono dati.
    """
    e = np.arange(0, 40)
    def dec(a, b, tau, off=0.0):
        return a * np.exp(-e / tau) + b + off

    casi = [
        ("Sottoadattamento",
         dec(1.0, 0.42, 30), dec(1.0, 0.50, 30),
         "Entrambe alte e ancora in discesa alla fine.",
         "Piu' epoche, piu' capacita', learning rate diverso."),
        ("Buon adattamento",
         dec(1.0, 0.12, 9), dec(1.0, 0.17, 9),
         "Entrambe scendono e si appiattiscono, divario piccolo e stabile.",
         "Nessuno: e' il caso che si cerca."),
        ("Sovradattamento",
         dec(1.0, 0.06, 8),
         dec(1.0, 0.16, 7) + np.clip((e - 14) / 40, 0, None) ** 1.6 * 1.1,
         "Il training continua a scendere, la validation risale dopo un minimo.",
         "Fermarsi al minimo, ridurre la capacita', regolarizzare."),
        ("Validation piu' bassa del training",
         dec(1.0, 0.22, 10), dec(0.9, 0.14, 10),
         "La validation sta sotto: partizione piu' facile, o training misurato "
         "con regimi piu' duri.",
         "Verificare la partizione e come sono definite le due metriche."),
        ("Validation rumorosa",
         dec(1.0, 0.14, 9),
         dec(1.0, 0.20, 9) + np.sin(e * 1.7) * 0.09 * np.exp(-e / 30),
         "Oscillazioni ampie epoca per epoca sulla sola validation.",
         "Insieme di validation troppo piccolo: aumentarlo, o mediare piu' seed."),
        ("Training rumoroso",
         dec(1.0, 0.15, 9) + np.sin(e * 2.3) * 0.13 * np.exp(-e / 22),
         dec(1.0, 0.21, 9),
         "Oscillazioni sul training: passo di aggiornamento troppo aggressivo.",
         "Learning rate piu' basso, batch piu' grande, gradient clipping."),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), squeeze=False)
    for k, (titolo, tr, va, sintomo, rimedio) in enumerate(casi):
        ax = axes[k // 3][k % 3]
        ax.plot(e, tr, lw=2.4, color=C_TR, label="training loss")
        ax.plot(e, va, lw=2.4, color=C_VA, label="validation loss")
        if titolo == "Sovradattamento":
            i = int(np.argmin(va))
            ax.plot(i, va[i], "o", ms=6, mfc="none", mec="#b30000", mew=1.8)
            ax.annotate("minimo", (i, va[i]), textcoords="offset points",
                        xytext=(6, 14), fontsize=8, color="#b30000")
        ax.set_title(titolo, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("epoche", fontsize=9)
        ax.set_ylabel("loss", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(0, 1.35)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        import textwrap
        ax.text(0.0, -0.26, textwrap.fill(sintomo, 52), transform=ax.transAxes,
                fontsize=8.4, va="top", color="#333333", linespacing=1.4)
        ax.text(0.0, -0.52, textwrap.fill("Rimedio: " + rimedio, 52),
                transform=ax.transAxes, fontsize=8.4, va="top", color="#6b7680",
                linespacing=1.4)
        if k == 0:
            ax.legend(fontsize=8.5, loc="lower left", frameon=False)
    fig.suptitle("Come si legge una curva di loss: i sei casi canonici", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.subplots_adjust(hspace=0.95, wspace=0.28)
    return _save(fig, out_path)


# 1-bis. Le due curve, senza altro: la figura che il docente si aspetta

def plot_loss_curves_simple(results, out_path, only_selected=False):
    """Training loss e validation loss, una coppia per configurazione.

    Nessuna banda, nessuna scala logaritmica, nessuna terza curva: e' la figura
    canonica dei manuali, quella su cui si dice «qui sovradatta, qui no». Entrambe
    le curve sono l'MSE next-step in teacher forcing, l'una sul training e l'altra
    sulla validation: e' la stessa quantita' misurata su dati diversi, che e' la
    condizione perche' il confronto voglia dire qualcosa.
    """
    cfgs = results["configs"]
    if only_selected:
        names = [v for v in results.get("selected", {}).values() if v in cfgs]
    else:
        names = list(cfgs.keys())
    n = len(names)
    ncol = min(4, n) if not only_selected else min(2, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(4.6 * ncol, 3.5 * nrow), squeeze=False)
    for k, name in enumerate(names):
        ax = axes[k // ncol][k % ncol]
        entry = cfgs[name]
        hist = entry["history"]
        ep, tr = _series(hist, "train_mse")
        ep2, va = _series(hist, "val_mse")
        ax.plot(ep, tr, lw=2.3, color=C_TR, label="training loss")
        ax.plot(ep2, va, lw=2.3, color=C_VA, label="validation loss")

        be = entry.get("info", {}).get("best_epoch")
        if be is not None and 0 <= be < len(va):
            ax.plot(be, va[be], "o", ms=7, mfc="none", mec="#b30000", mew=1.8,
                    zorder=4)
            # l'etichetta va a sinistra del punto quando il minimo cade nella parte
            # destra del pannello, altrimenti uscirebbe dal riquadro
            destra = be > 0.6 * max(ep2[-1], 1)
            ax.annotate(f"minimo, ep. {be}", (be, va[be]),
                        textcoords="offset points",
                        xytext=(-10 if destra else 10, 13),
                        ha="right" if destra else "left",
                        fontsize=8, color="#b30000")

        d = entry.get("diagnosi") or diagnose_curve(hist)
        ax.text(0.97, 0.95, d["verdetto"], transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color="#222222",
                bbox=dict(boxstyle="round,pad=0.35", fc="#f2f4f6", ec="#cfd6dc",
                          lw=0.8))

        ax.set_title(f"{name} - {entry.get('label','')}", fontsize=10, loc="left")
        ax.set_xlabel("epoche"); ax.set_ylabel("loss (MSE next-step)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.22, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"Training loss e validation loss - holdout 80/20, "
                 f"L={results['window']}", fontsize=12)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    return _save(fig, out_path)


# 1. Curve di apprendimento, una per configurazione

def plot_learning_curves(results, out_path, log_scale=True):
    """Griglia di pannelli: training e validation a confronto, per ogni configurazione.

    In ogni pannello ci sono tre curve e vanno lette in quest'ordine:
      - grigia tratteggiata: l'obiettivo minimizzato nell'epoca corrente. Salta ai
        confini fra regimi perche' cambia la definizione, non perche' il modello
        peggiori.
      - grigia continua: MSE next-step sul sottocampione di training.
      - blu continua: MSE next-step sulla validation. Confrontabile con la precedente.
    La linea verticale tratteggiata segna l'epoca con validation minima, cioe' i pesi
    che vengono effettivamente conservati.
    """
    cfgs = results["configs"]
    names = list(cfgs.keys())
    n = len(names)
    ncol = 4 if n >= 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             squeeze=False, sharex=True)
    for k, name in enumerate(names):
        ax = axes[k // ncol][k % ncol]
        entry = cfgs[name]
        hist = entry["history"]
        _shade_regimes(ax, hist)

        ep, obj = _series(hist, "train_loss")
        ax.plot(ep, obj, "--", lw=1.0, color=C_OBJ, label="obiettivo del regime")
        ep, tr = _series(hist, "train_mse")
        ax.plot(ep, tr, "-", lw=1.4, color=C_TRAIN, label="train MSE (next-step)")
        ep, va = _series(hist, "val_mse")
        ax.plot(ep, va, "-o", ms=2.6, lw=1.6, color=C_VAL, label="validation MSE")

        be = entry.get("info", {}).get("best_epoch")
        if be is not None and be >= 0:
            ax.axvline(be, color="#b30000", ls=":", lw=1.2)
            ax.text(be, 0.99, f"ep. {be} ", transform=_blend(ax), ha="right",
                    fontsize=6.5, color="#b30000", va="top")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"{name} ({entry.get('label', entry.get('kind', ''))})", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        if k % ncol == 0:
            ax.set_ylabel("MSE (unita' scalate)")
        if k // ncol == nrow - 1:
            ax.set_xlabel("epoca")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    rh = _regime_handles(cfgs[names[0]]["history"])
    fig.legend(handles + rh, labels + [h.get_label() for h in rh],
               loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Curve di apprendimento - holdout 80/20, L={results['window']}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    return _save(fig, out_path)


# 2. Validation di tutte le configurazioni sullo stesso asse

def plot_val_overlay(results, out_path, log_scale=True):
    cfgs = results["configs"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    any_hist = next(iter(cfgs.values()))["history"]
    _shade_regimes(ax, any_hist)
    by_kind = {}
    for name, e in cfgs.items():
        by_kind.setdefault(e["kind"], []).append(name)
    for kind, names in by_kind.items():
        for i, name in enumerate(names):
            ep, va = _series(cfgs[name]["history"], "val_mse")
            ax.plot(ep, va, "-", lw=1.6, color=_family_color(kind, i, len(names)),
                    label=f"{name}")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("epoca"); ax.set_ylabel("validation MSE (unita' scalate)")
    ax.grid(alpha=0.25, lw=0.5)
    h, l = ax.get_legend_handles_labels()
    rh = _regime_handles(any_hist)
    ax.legend(h + rh, l + [p.get_label() for p in rh], fontsize=8, ncol=4)
    ax.set_title(f"Validation MSE per configurazione - L={results['window']}")
    fig.tight_layout()
    return _save(fig, out_path)


# 3. Divario di generalizzazione

def plot_gap(results, out_path):
    """validation - training, sulla stessa metrica. Positivo e crescente = sovradattamento."""
    cfgs = results["configs"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    any_hist = next(iter(cfgs.values()))["history"]
    _shade_regimes(ax, any_hist)
    by_kind = {}
    for name, e in cfgs.items():
        by_kind.setdefault(e["kind"], []).append(name)
    for kind, names in by_kind.items():
        for i, name in enumerate(names):
            ep, g = _series(cfgs[name]["history"], "gap_mse")
            ax.plot(ep, g, "-", lw=1.5, color=_family_color(kind, i, len(names)),
                    label=name)
    ax.axhline(0.0, color="k", lw=0.9)
    ax.set_xlabel("epoca")
    ax.set_ylabel("val MSE  -  train MSE")
    ax.grid(alpha=0.25, lw=0.5)
    h, l = ax.get_legend_handles_labels()
    rh = _regime_handles(any_hist)
    ax.legend(h + rh, l + [p.get_label() for p in rh], fontsize=8, ncol=4)
    ax.set_title(f"Divario di generalizzazione - L={results['window']}\n"
                 "sopra lo zero il modello va peggio su dati mai visti", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)


# 4. Un passo contro duecento passi

def plot_nextstep_vs_rollout(results, out_path):
    """Le due metriche sulla stessa figura, ognuna sul proprio asse.

    Se la curva blu scende e quella rossa resta piatta, il modello sta migliorando
    sul compito facile (prevedere l'istante successivo avendo davanti la storia vera)
    senza migliorare su quello vero (prevedere 200 istanti riusando le proprie
    previsioni). E' la definizione operativa dell'accumulo d'errore.
    """
    sel = results.get("selected", {})
    entries = [(k, results["configs"][v]) for k, v in sel.items()
               if v in results["configs"]]
    if not entries:
        entries = list(results["configs"].items())[:2]
    fig, axes = plt.subplots(1, len(entries), figsize=(5.6 * len(entries), 4.0),
                             squeeze=False)
    for j, (kind, entry) in enumerate(entries):
        ax = axes[0][j]
        hist = entry["history"]
        _shade_regimes(ax, hist)
        ep, va = _series(hist, "val_mse")
        ax.plot(ep, va, "-o", ms=3, lw=1.6, color=C_VAL, label="validation MSE (1 passo)")
        ax.set_xlabel("epoca")
        ax.set_ylabel("MSE next-step (scalato)", color=C_VAL)
        ax.tick_params(axis="y", labelcolor=C_VAL)
        ax2 = ax.twinx()
        ep2, ro = _series(hist, "val_rollout_rmse")
        ax2.plot(ep2, ro, "-s", ms=3, lw=1.6, color="#b30000",
                 label=f"validation rollout RMSE (H={results.get('horizon', '?')})")
        ax2.set_ylabel("RMSE rollout (unita' fisiche)", color="#b30000")
        ax2.tick_params(axis="y", labelcolor="#b30000")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_title(f"{entry['name']} - {kind}", fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center")
    rh = _regime_handles(entries[0][1]["history"])
    fig.legend(rh, [p.get_label() for p in rh], loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Errore a un passo contro errore a orizzonte lungo - L={results['window']}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return _save(fig, out_path)


# 5. Rollout su training e su validation

def plot_train_vs_val_rollout(results, out_path):
    """Se le due curve coincidono, il rollout non e' difficile perche' i dati sono nuovi:
    e' difficile in se'. E' l'argomento che separa 'sovradattamento' da 'compito duro'."""
    sel = results.get("selected", {})
    entries = [(k, results["configs"][v]) for k, v in sel.items()
               if v in results["configs"]]
    if not entries:
        entries = list(results["configs"].items())[:2]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _shade_regimes(ax, entries[0][1]["history"])
    styles = ["-", "--"]
    for j, (kind, entry) in enumerate(entries):
        hist = entry["history"]
        ep, tr = _series(hist, "train_rollout_rmse")
        ep2, va = _series(hist, "val_rollout_rmse")
        ax.plot(ep, tr, styles[j % 2], lw=1.5, color=C_TRAIN,
                label=f"{entry['name']} - train")
        ax.plot(ep2, va, styles[j % 2], lw=1.8, color="#b30000",
                label=f"{entry['name']} - validation")
    bl = results.get("baselines", {})
    if bl:
        best = min(bl, key=lambda k: bl[k]["rollout_rmse_physical"])
        ax.axhline(bl[best]["rollout_rmse_physical"], color="#333333", ls=":", lw=1.2,
                   label=f"baseline migliore ({best})")
    ax.set_xlabel("epoca"); ax.set_ylabel("RMSE rollout (unita' fisiche)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"Rollout su training e su validation - L={results['window']}")
    fig.tight_layout()
    return _save(fig, out_path)


# 6. Confronto fra configurazioni, con incertezza

def plot_config_comparison(results, out_path):
    """Barre con intervallo di confidenza bootstrap al 95%.

    Due configurazioni i cui intervalli si sovrappongono non sono distinguibili con
    questi dati: la differenza fra le loro medie e' compatibile con il rumore
    campionario delle traiettorie di validation.
    """
    cfgs = results["configs"]
    names = list(cfgs.keys())
    sel = set(results.get("selected", {}).values())
    vals, los, his, colors = [], [], [], []
    for name in names:
        m = cfgs[name]["val"]
        v = m.get("rollout_rmse_physical", np.nan)
        ci = m.get("rollout_rmse_physical_ci95", (v, v))
        vals.append(v); los.append(v - ci[0]); his.append(ci[1] - v)
        colors.append(_family_color(cfgs[name]["kind"], 0, 1))
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 0.40 * len(names) + 2.4))
    # Punto con intervallo invece di barra: le differenze fra configurazioni sono
    # piccole rispetto al valore assoluto, e una barra ancorata a zero le nasconde.
    for i in range(len(names)):
        ax.errorbar(vals[i], y[i], xerr=[[los[i]], [his[i]]], fmt="o", ms=7,
                    color=colors[i], ecolor="#444444", elinewidth=1.3, capsize=4,
                    markeredgecolor="black" if names[i] in sel else "none",
                    markeredgewidth=1.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  ({cfgs[n].get('label','')})" for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_ylim(len(names) - 0.4, -0.8)

    bl = results.get("baselines", {})
    for name, m in bl.items():
        xv = m["rollout_rmse_physical"]
        ax.axvline(xv, ls=":", lw=1.1, color="#666666", zorder=1)
        ax.text(xv, 0.985, f" {name}", transform=_blend(ax), rotation=90,
                fontsize=6.5, color="#555555", va="top", ha="left")
    allv = [v for v in vals if np.isfinite(v)] + \
           [m["rollout_rmse_physical"] for m in bl.values()]
    lo, hi = min(allv), max(allv)
    pad = 0.12 * (hi - lo + 1e-9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("RMSE del rollout sulla validation (unita' fisiche) - piu' basso e' meglio")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_title(f"Configurazioni a confronto, IC 95% bootstrap - L={results['window']}\n"
                 "bordo nero = configurazione selezionata; intervalli sovrapposti = "
                 "differenza non distinguibile", fontsize=9)
    fig.tight_layout()
    return _save(fig, out_path)


# 7. Crescita dell'errore con l'orizzonte

def plot_error_growth(results, out_path):
    sel = results.get("selected", {})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for kind, name in sel.items():
        m = results["configs"][name]["val"]
        cur = np.asarray(m.get("per_horizon_rmse_physical", m.get("per_horizon_rmse")))
        ax.plot(np.arange(1, len(cur) + 1), cur, lw=1.8,
                color=_family_color(kind, 0, 1), label=f"{name} ({kind})")
    for bname, m in results.get("baselines", {}).items():
        cur = np.asarray(m.get("per_horizon_rmse_physical", m.get("per_horizon_rmse")))
        ax.plot(np.arange(1, len(cur) + 1), cur, lw=1.1, ls="--", alpha=0.8,
                label=f"baseline {bname}")
    sigma = results.get("target_sigma_physical")
    if sigma:
        ax.axhline(sigma, color="k", lw=1.0, ls=":")
        ax.text(1, sigma, " dispersione del segnale (sigma)", fontsize=7, va="bottom")
    ax.set_xlabel("passi di previsione oltre il contesto")
    ax.set_ylabel("RMSE (unita' fisiche)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"Crescita dell'errore con l'orizzonte - validation, L={results['window']}")
    fig.tight_layout()
    return _save(fig, out_path)


# 8. Errore per gruppo di osservabili

def plot_group_breakdown(results, out_path):
    sel = results.get("selected", {})
    items = [(results["configs"][n]["name"], results["configs"][n]["val"])
             for n in sel.values() if n in results["configs"]]
    for bname, m in results.get("baselines", {}).items():
        items.append((f"bl:{bname}", m))
    labels = [i[0] for i in items]
    mag = [i[1].get("rmse_physical_magnetizations", np.nan) for i in items]
    cor = [i[1].get("rmse_physical_correlations", np.nan) for i in items]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.15 * len(labels) + 3, 4.2))
    ax.bar(x - w / 2, mag, w, label="magnetizzazioni (10)", color="#1f4e79")
    ax.bar(x + w / 2, cor, w, label="correlazioni (45)", color="#e07b39")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("RMSE rollout (unita' fisiche)")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)
    ax.set_title(f"Errore per gruppo di osservabili - validation, L={results['window']}",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)


def make_all_figures(results, out_dir) -> list:
    os.makedirs(out_dir, exist_ok=True)
    L = results["window"]
    paths = []
    plans = [
        (plot_loss_curves_simple, f"fig_loss_curves_L{L}.jpg"),
        (lambda r, p: plot_loss_curves_simple(r, p, only_selected=True),
         f"fig_loss_curves_selezionati_L{L}.jpg"),
        (plot_learning_curves, f"fig_learning_curves_L{L}.jpg"),
        (plot_val_overlay, f"fig_val_overlay_L{L}.jpg"),
        (plot_gap, f"fig_gap_L{L}.jpg"),
        (plot_nextstep_vs_rollout, f"fig_nextstep_vs_rollout_L{L}.jpg"),
        (plot_train_vs_val_rollout, f"fig_train_vs_val_rollout_L{L}.jpg"),
        (plot_config_comparison, f"fig_config_comparison_L{L}.jpg"),
        (plot_error_growth, f"fig_error_growth_L{L}.jpg"),
        (plot_group_breakdown, f"fig_group_breakdown_L{L}.jpg"),
    ]
    for fn, fname in plans:
        try:
            paths.append(fn(results, os.path.join(out_dir, fname)))
        except Exception as exc:                      # una figura che fallisce non
            print(f"  [figura] {fname} non prodotta: {exc}")   # deve fermare la pipeline
    return paths
