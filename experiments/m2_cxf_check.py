"""M2 — verifiche guardabili del parser CXF.

Due figure:

1. `m2_cxf_overview.png` — i poligoni del CXF in coordinate CRS, colorati per
   codice. È un controllo del *parser*: se `nflag` fosse ignorato, qui si
   vedrebbero segmenti che schizzano fuori dal foglio.

2. `m2_cornice.png` — inquadramento del vettoriale sul raster, per decidere fra
   le due letture della dimensione del foglio (8489x5648 del file contro
   8000x5322 di §5.2 / I10). Due pannelli:
     - foglio intero ridotto: serve solo a vedere se il vettoriale cade *dove*
       sta il disegno. Uno scarto di scala del 6% sarebbero ~500 px, visibili
       anche così. Non è una verifica di allineamento (I5, §5.5).
     - un ritaglio a **piena risoluzione**: questa è la verifica vera, l'unica
       su cui si possono trarre conclusioni.

Nessuna metrica indiretta: correlazione e chamfer su questi dati non
discriminano (§5.5). Si guarda.

    python -m experiments.m2_cxf_check
"""
from __future__ import annotations

import argparse
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.io_geo import (  # noqa: E402
    CODICE_ACQUA_STRADA,
    CODICE_ALTRO,
    CODICE_PARTICELLA,
    apply_affine,
    cxf_extent,
    read_cxf,
    read_jgw,
)

COLORI = {
    CODICE_PARTICELLA: ("#1f77b4", "18 — particelle"),
    CODICE_ACQUA_STRADA: ("#2ca02c", "12 — acqua / strada / bordo"),
    CODICE_ALTRO: ("#d62728", "25 — altro"),
}


def figura_overview(bordi, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    visti = set()
    for b in bordi:
        colore, etichetta = COLORI.get(b.codice, ("#777777", f"{b.codice} — ?"))
        ax.plot(
            b.pts[:, 0],
            b.pts[:, 1],
            color=colore,
            linewidth=0.5,
            label=etichetta if b.codice not in visti else None,
        )
        visti.add(b.codice)
    xmin, ymin, xmax, ymax = cxf_extent(bordi)
    ax.set_aspect("equal")
    ax.set_title(
        f"CXF L675_004900 — {len(bordi)} record BORDO, "
        f"{sum(len(b.pts) for b in bordi)} vertici\n"
        f"X {xmin:.0f}…{xmax:.0f}   Y {ymin:.0f}…{ymax:.0f}  "
        "(Cassini-Soldner Forte Diamante, metri)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    _salva(fig, out_path)


def figura_cornice(bordi, sheet_path: str, jgw_path: str, crop: str, out_path: str) -> None:
    """Vettoriale proiettato in pixel sul raster: colpo d'occhio + piena risoluzione."""
    W = read_jgw(jgw_path)
    Winv = np.linalg.inv(W)
    sheet = cv2.imread(sheet_path, cv2.IMREAD_COLOR)
    if sheet is None:
        raise FileNotFoundError(sheet_path)
    h, w = sheet.shape[:2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

    scala = 8
    ax1.imshow(cv2.cvtColor(cv2.resize(sheet, (w // scala, h // scala)), cv2.COLOR_BGR2RGB))
    for b in bordi:
        px = apply_affine(Winv, b.pts) / scala
        ax1.plot(px[:, 0], px[:, 1], color="#d62728", linewidth=0.4, alpha=0.8)
    ax1.set_title(
        f"Inquadramento — foglio {w}x{h} px, ridotto {scala}x\n"
        "colpo d'occhio sulla scala, NON una verifica di allineamento (I5)",
        fontsize=9,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])

    crop_png = os.path.join("data/crops", f"{crop}.png")
    crop_jgw = os.path.join("data/crops", f"{crop}.jgw")
    img = cv2.imread(crop_png, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{crop_png} — genera prima i crop con src.prep.crop")
    Wc_inv = np.linalg.inv(read_jgw(crop_jgw))
    ch, cw = img.shape[:2]
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for b in bordi:
        px = apply_affine(Wc_inv, b.pts)
        dentro = (px[:, 0] > -50) & (px[:, 0] < cw + 50) & (px[:, 1] > -50) & (px[:, 1] < ch + 50)
        if dentro.any():
            ax2.plot(px[:, 0], px[:, 1], color="#d62728", linewidth=1.0, alpha=0.9)
    ax2.set_xlim(0, cw)
    ax2.set_ylim(ch, 0)
    ax2.set_title(
        f"{crop} — {cw}x{ch} px a piena risoluzione (1 px = 0.2545 m)\n"
        "questa è la verifica su cui si può concludere qualcosa",
        fontsize=9,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])
    _salva(fig, out_path)


def _salva(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"figura: {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cxf", default="data/raw/L675_004900.cxf")
    ap.add_argument("--sheet", default="data/raw/L675_004900.jpg")
    ap.add_argument("--jgw", default="data/raw/L675_004900.jgw")
    ap.add_argument("--crop", default="ribba", help="crop usato per il pannello a piena risoluzione")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args(argv)

    bordi = read_cxf(args.cxf).bordi
    figura_overview(bordi, os.path.join(args.out, "m2_cxf_overview.png"))
    figura_cornice(
        bordi, args.sheet, args.jgw, args.crop, os.path.join(args.out, "m2_cornice.png")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
