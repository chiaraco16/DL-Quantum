"""
Verifica della riproducibilità' dei risultati.

Confronta due esecuzioni indipendenti dello stesso sweep:
  * tutte le metriche in results_full.json devono coincidere (NaN == NaN)
  * Table 1 deve essere identica carattere per carattere
  * i pesi salvati devono essere identici bit per bit 
I tempi di esecuzione (sec) sono ovviamente esclusi dal confronto.

"""
import glob
import hashlib
import json
import math
import os
import sys

IGNORED_KEYS = {"sec", "reproducibility"}


def _equal(a, b, path=""):
    diffs = []
    if isinstance(a, dict):
        if set(a) != set(b):
            diffs.append((path, f"chiavi diverse: {set(a) ^ set(b)}"))
            return diffs
        for k in a:
            if k in IGNORED_KEYS:
                continue
            diffs += _equal(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((path, f"lunghezze diverse {len(a)} vs {len(b)}"))
            return diffs
        for i, (u, v) in enumerate(zip(a, b)):
            diffs += _equal(u, v, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return diffs
        if a != b:
            diffs.append((path, f"{a} != {b}"))
    else:
        if a != b:
            diffs.append((path, f"{a!r} != {b!r}"))
    return diffs


def main(dir_a, dir_b):
    ok = True

    pa, pb = os.path.join(dir_a, "results_full.json"), os.path.join(dir_b, "results_full.json")
    if os.path.exists(pa) and os.path.exists(pb):
        diffs = _equal(json.load(open(pa)), json.load(open(pb)))
        print(f"[metriche]  results_full.json: "
              f"{'IDENTICHE' if not diffs else str(len(diffs)) + ' DIFFERENZE'}")
        for p, d in diffs[:10]:
            print(f"            {p}: {d}")
        ok &= not diffs
    else:
        print("[metriche]  results_full.json mancante in almeno una delle due cartelle")
        ok = False

    for name in sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_a, "table*.csv"))):
        fa, fb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        if os.path.exists(fb):
            same = open(fa).read() == open(fb).read()
            print(f"[tabella ]  {name}: {'IDENTICA' if same else 'DIVERSA'}")
            ok &= same

    for p in sorted(glob.glob(os.path.join(dir_a, "models", "*.weights.h5"))):
        name = os.path.basename(p)
        q = os.path.join(dir_b, "models", name)
        if os.path.exists(q):
            ha = hashlib.md5(open(p, "rb").read()).hexdigest()
            hb = hashlib.md5(open(q, "rb").read()).hexdigest()
            print(f"[pesi    ]  {name}: {'IDENTICI (bit-per-bit)' if ha == hb else 'DIVERSI'}")
            ok &= ha == hb

    print("\n" + ("RIPRODUCIBILITA' VERIFICATA" if ok else "RIPRODUCIBILITA' NON VERIFICATA"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
