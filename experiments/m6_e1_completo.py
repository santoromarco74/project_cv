"""M6 — E1 completo: curva RMSE vs degradazione, SIFT contro ORB. Primo CSV vero.

Due sweep, ciascuno con una variabile indipendente sola (§8):

- `degradazione`: la trasformazione geometrica è fissa, cresce solo il degrado
  radiometrico. È la curva RMSE vs degradazione, la prima figura della relazione.
- `ampiezza`: nessuna degradazione, cresce solo la rotazione. Dice fino a dove
  regge il matching quando l'unica difficoltà è geometrica.

Ogni combinazione produce una riga di CSV, comprese quelle fallite: un
esperimento che non produce una stima ha comunque prodotto un dato (I7).

    python -m experiments.m6_e1_completo                     # griglia intera
    python -m experiments.m6_e1_completo --crop ribba        # una prova rapida
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from src.estimate import stima as stima_ransac
from src.evaluate import append_csv, parametri_matcher, valuta
from src.io_geo import read_jgw
from src.matchers.classic import crea_matcher
from src.prep.crop import CROPS
from src.prep.synth import Trasformazione, genera_coppia, scala_degradazione
from src.preprocess import applica

# Trasformazione di riferimento dello sweep sulla degradazione: una similarità
# realistica, né identità né caso estremo.
BASE = Trasformazione(rot_deg=15, scala=1.15, tx=30, ty=20)

# Campionamento fitto sopra 1.0: lì sta la soglia di rottura, e con passi da 0.2
# la curva la salterebbe mostrando una discesa dolce che non esiste.
LIVELLI_DEGRADO = (0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
AMPIEZZE_ROT = (0, 5, 15, 30, 45, 90)
PREPROCESS = ("none", "clahe", "sauvola")
MATCHER = ("sift", "orb")

# Soglia di successo per E1. La ground truth qui è esatta, quindi si può essere
# severi: 1 px = 0.254 m, cioè metà del pavimento del riferimento reale (§5.3).
SOGLIA_M = 0.2545


def una_riga(
    img, t, degrado, nome_crop, nome_matcher, nome_preprocess, modello, seed, W_hist
) -> dict:
    b, H_true = genera_coppia(img, t, scala_degradazione(degrado) if degrado else None, seed)
    h, w = img.shape[:2]

    # Il preprocessing si applica alle DUE immagini allo stesso modo: è parte
    # della pipeline, non un ritocco di una delle due.
    a_prep = applica(img, modo=nome_preprocess)
    b_prep = applica(b, modo=nome_preprocess)

    t0 = time.perf_counter()
    pts_a, pts_b, meta = crea_matcher(nome_matcher).match(a_prep, b_prep)
    t_match = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    st = stima_ransac(pts_a, pts_b, modello=modello, seed=seed)
    t_stima = (time.perf_counter() - t0) * 1000

    riga = valuta(st, H_true, w, h, W_hist=W_hist, soglia_m=SOGLIA_M)
    riga |= {
        "esperimento": "E1",
        "crop": nome_crop,
        "matcher": nome_matcher,
        "preprocess": nome_preprocess,
        "morph_open": 0,
        "morph_close": 0,
        "degrado": degrado,
        "rot_deg": t.rot_deg,
        "scala": t.scala,
        "tx": t.tx,
        "ty": t.ty,
        "prospettiva": t.prospettiva,
        "seed": seed,
        "n_kp_a": meta.get("n_kp_a"),
        "n_kp_b": meta.get("n_kp_b"),
        "t_match_ms": round(t_match, 1),
        "t_stima_ms": round(t_stima, 1),
    } | parametri_matcher(meta)
    return riga


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="tutti", help="nome del crop, o 'tutti'")
    ap.add_argument("--modello", default="homography")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default="results/runs.csv")
    ap.add_argument("--jgw", default="data/raw/L675_004900.jgw")
    ap.add_argument(
        "--riparti",
        action="store_true",
        help="cancella il CSV prima di scrivere (I9: stesso input, stesso file)",
    )
    args = ap.parse_args(argv)

    if args.riparti and os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    W_hist = read_jgw(args.jgw) if os.path.exists(args.jgw) else None
    nomi = [c.nome for c in CROPS] if args.crop == "tutti" else [args.crop]

    prove = []
    for nome in nomi:
        for m in MATCHER:
            for p in PREPROCESS:
                for livello in LIVELLI_DEGRADO:
                    prove.append((nome, m, p, BASE, livello))
                for rot in AMPIEZZE_ROT:
                    prove.append((nome, m, p, Trasformazione(rot_deg=rot, scala=1.0), 0.0))

    print(f"{len(prove)} prove · matcher {MATCHER} · preprocess {PREPROCESS} · seed {args.seed}")
    print(f"CSV: {args.out_csv}\n")

    immagini: dict[str, np.ndarray] = {}
    falliti = 0
    t_inizio = time.perf_counter()
    for i, (nome, m, p, t, livello) in enumerate(prove, 1):
        if nome not in immagini:
            img = cv2.imread(f"data/crops/{nome}.png", cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"data/crops/{nome}.png — genera prima i crop (M1)")
            immagini[nome] = img

        riga = una_riga(
            immagini[nome], t, livello, nome, m, p, args.modello, args.seed, W_hist
        )
        append_csv(args.out_csv, riga)
        if not riga["success"]:
            falliti += 1
        if i % 20 == 0 or i == len(prove):
            print(f"  {i}/{len(prove)} · {falliti} sotto soglia o falliti")

    print(f"\nfatto in {time.perf_counter() - t_inizio:.0f} s · {len(prove)} righe scritte")
    print(f"Ora: python -m src.report --csv {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
