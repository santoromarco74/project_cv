"""M9 — E3: LoFTR su E1 ed E2, stesse metriche e stessi crop del classico.

Il confronto ha valore solo se classico e neurale girano nella stessa pipeline
cambiando il solo `--matcher` (§8). Qui è così: si costruiscono le stesse
`Opzioni` di M6 e M8, con `matcher="loftr"`.

Due differenze da dichiarare, perché sono reali e influenzano il confronto:

1. LoFTR lavora su una griglia a 1/8 e a piena risoluzione su CPU costa minuti
   e molti GB. Gira su lato massimo 640 px, con i keypoint riportati alle
   coordinate originali. SIFT e ORB girano a piena risoluzione.
2. Il costo per registrazione è di un altro ordine di grandezza (~12 s contro
   0.1-0.6 s). Fa parte del risultato (§8, E3), non è un dettaglio da nascondere.

Su E1 si usa il solo preprocessing `none`, che è il dominio nativo di LoFTR e
la colonna su cui la curva di M6 è già disegnata.

    python -m experiments.m9_e3_loftr
"""
from __future__ import annotations

import argparse
import os
import time

import cv2

from src.evaluate import append_csv, valuta
from src.groundtruth import h_true_from_jgw
from src.io_geo import read_jgw
from src.pipeline import Opzioni, registra
from src.prep.crop import CROPS
from src.prep.synth import Trasformazione, genera_coppia, scala_degradazione

# Stessi punti di M6 e M8, così le righe finiscono sulle stesse figure.
LIVELLI_DEGRADO = (0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
AMPIEZZE_ROT = (0, 5, 15, 30, 45, 90)
BASE = Trasformazione(rot_deg=15, scala=1.15, tx=30, ty=20)
SOGLIA_E1_M = 0.2545
SOGLIA_E2_M = 2.0

PREPROCESS_E2 = (("clahe", 0, 0), ("sauvola", 0, 0), ("sauvola", 0, 1))
MODELLI = ("similarity", "affine", "homography")
VETTORIALI = (("vec", "18"), ("vec1812", "18+12"))


def _riga_base(opz: Opzioni, ris) -> dict:
    return {
        "matcher": "loftr",
        "preprocess": opz.preprocess + ("+close" if opz.morph_close else ""),
        "morph_open": opz.morph_open,
        "morph_close": opz.morph_close,
        "loftr_conf": opz.loftr_conf,
        "loftr_max_lato": opz.loftr_max_lato,
        "seed": opz.seed,
        "n_kp_a": ris.meta.get("n_kp_a"),
        "n_kp_b": ris.meta.get("n_kp_b"),
        "t_init_ms": ris.meta["t_init_ms"],
        "t_match_ms": ris.meta["t_match_ms"],
        "t_stima_ms": ris.meta["t_stima_ms"],
    }


def e1(args, immagini) -> int:
    prove = [(n, BASE, liv) for n in immagini for liv in LIVELLI_DEGRADO]
    prove += [
        (n, Trasformazione(rot_deg=r, scala=1.0), 0.0) for n in immagini for r in AMPIEZZE_ROT
    ]
    print(f"E1 · {len(prove)} prove", flush=True)
    for i, (nome, t, livello) in enumerate(prove, 1):
        img = immagini[nome]
        b, H_true = genera_coppia(img, t, scala_degradazione(livello) if livello else None, args.seed)
        opz = Opzioni(matcher="loftr", preprocess="none", model="homography", seed=args.seed)
        ris = registra(img, b, opz)
        riga = valuta(
            ris.stima, H_true, img.shape[1], img.shape[0],
            W_hist=args.W_foglio, soglia_m=SOGLIA_E1_M,
        )
        riga |= _riga_base(opz, ris) | {
            "esperimento": "E1",
            "crop": nome,
            "degrado": livello,
            "rot_deg": t.rot_deg,
            "scala": t.scala,
            "tx": t.tx,
            "ty": t.ty,
            "prospettiva": t.prospettiva,
        }
        append_csv(args.out_csv, riga)
        if i % 10 == 0 or i == len(prove):
            print(f"  E1 {i}/{len(prove)}", flush=True)
    return len(prove)


def e2(args, immagini) -> int:
    prove = [
        (nome, suffisso, codici, prep, modello)
        for nome in immagini
        for suffisso, codici in VETTORIALI
        for prep in PREPROCESS_E2
        for modello in MODELLI
    ]
    print(f"E2 · {len(prove)} prove", flush=True)
    riusciti = 0
    for i, (nome, suffisso, codici, (prep, m_open, m_close), modello) in enumerate(prove, 1):
        hist = immagini[nome]
        chiave = f"{nome}_{suffisso}"
        vec = cv2.imread(os.path.join(args.crops_dir, f"{chiave}.png"), cv2.IMREAD_COLOR)
        if vec is None:
            raise FileNotFoundError(f"{chiave}.png — genera prima i raster (M7)")

        opz = Opzioni(
            matcher="loftr",
            preprocess=prep,
            morph_open=m_open,
            morph_close=m_close,
            model=modello,
            seed=args.seed,
        )
        ris = registra(hist, vec, opz)
        crop_jgw = os.path.join(args.crops_dir, f"{nome}.jgw")
        riga = valuta(
            ris.stima,
            h_true_from_jgw(crop_jgw, os.path.join(args.crops_dir, f"{chiave}.jgw")),
            hist.shape[1],
            hist.shape[0],
            W_hist=read_jgw(crop_jgw),
            soglia_m=SOGLIA_E2_M,
        )
        riga |= _riga_base(opz, ris) | {
            "esperimento": "E2",
            "crop": nome,
            "codici": codici,
        }
        append_csv(args.out_csv, riga)
        riusciti += bool(riga["success"])
        if i % 10 == 0 or i == len(prove):
            print(f"  E2 {i}/{len(prove)} · {riusciti} sotto {SOGLIA_E2_M} m", flush=True)
    return len(prove)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="tutti")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--out-csv", default="results/runs.csv")
    ap.add_argument("--jgw", default="data/raw/L675_004900.jgw")
    ap.add_argument("--solo", choices=("e1", "e2"), default=None)
    args = ap.parse_args(argv)

    args.W_foglio = read_jgw(args.jgw) if os.path.exists(args.jgw) else None
    nomi = [c.nome for c in CROPS] if args.crop == "tutti" else [args.crop]
    immagini = {}
    for nome in nomi:
        img = cv2.imread(os.path.join(args.crops_dir, f"{nome}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"{nome}.png — genera prima i crop (M1)")
        immagini[nome] = img

    t0 = time.perf_counter()
    totale = 0
    if args.solo != "e2":
        totale += e1(args, immagini)
    if args.solo != "e1":
        totale += e2(args, immagini)
    print(f"\nfatto in {time.perf_counter() - t0:.0f} s · {totale} righe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
