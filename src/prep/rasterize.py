"""CXF -> raster allineato via world file. Verifica VISIVA a piena risoluzione (I5).

Produce l'immagine "moderna" di E2: i poligoni del CXF disegnati su una griglia
regolare, con il proprio world file. Da lì `groundtruth.h_true` compone le due
affini e la ground truth è nota senza un solo GCP (I2).

**La griglia non è quella del crop storico.** Se lo fosse, `H_true` sarebbe
l'identità e l'esperimento premierebbe qualunque metodo con un bias verso il
"non muovere niente". Il raster moderno ha risoluzione e origine proprie, così
`H_true` è una similarità vera — che è anche il caso realistico, visto che la
cartografia vigente non nasce sulla griglia della scansione d'impianto.
`--stessa-griglia` esiste per il caso degenere, ma non è il default.

**Cosa NON si fa qui**: misurare l'allineamento con correlazione o chamfer.
Su questi dati producono falsi positivi convincenti (§5.5). La verifica è
visiva, a piena risoluzione, e sta in experiments/m7_rasterize_check.py.

    python -m src.prep.rasterize --crop ribba --codici 18
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from src.io_geo import (
    CODICE_ACQUA_STRADA,
    CODICE_PARTICELLA,
    apply_affine,
    raster_extent,
    read_cxf,
    read_jgw,
    write_jgw,
)

# Risoluzione del raster moderno. Diversa da quella dello storico (0.254453) di
# proposito: vedi la nota sulla griglia nel docstring del modulo.
RISOLUZIONE_M = 0.20

# Quanto il raster moderno deborda dal crop storico. Senza margine il contenuto
# ai bordi del crop non avrebbe controparte, e si misurerebbe il ritaglio invece
# del matching.
MARGINE_M = 20.0


def griglia(
    xmin: float, ymin: float, xmax: float, ymax: float, risoluzione: float
) -> tuple[np.ndarray, int, int]:
    """Affine e dimensioni di una griglia nord-up che copre l'estensione data.

    Come nel JGW, (C, F) è il centro del pixel (0,0) e la riga cresce verso sud.
    """
    w = int(np.ceil((xmax - xmin) / risoluzione)) + 1
    h = int(np.ceil((ymax - ymin) / risoluzione)) + 1
    W = np.array(
        [[risoluzione, 0.0, xmin], [0.0, -risoluzione, ymax], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return W, w, h


def rasterizza(
    bordi,
    W: np.ndarray,
    width: int,
    height: int,
    codici: tuple[int, ...] = (CODICE_PARTICELLA,),
    spessore: int = 1,
) -> np.ndarray:
    """Disegna i poligoni sulla griglia. Inchiostro scuro su fondo chiaro.

    I bordi si disegnano come polilinee e non come poligoni pieni: sull'impianto
    il confine è un tratto, non una campitura, e riempire cambierebbe il tipo di
    struttura che il matcher vede.

    La stessa polarità dello storico (scuro su chiaro) serve a poter confrontare
    le due immagini senza inversioni implicite lungo la pipeline.
    """
    Winv = np.linalg.inv(W)
    img = np.full((height, width), 255, dtype=np.uint8)
    for b in bordi:
        if b.codice not in codici:
            continue
        px = apply_affine(Winv, b.pts)
        cv2.polylines(
            img,
            [np.round(px).astype(np.int32)],
            isClosed=bool(b.chiuso),
            color=0,
            thickness=spessore,
            lineType=cv2.LINE_AA,
        )
    return img


def per_crop(
    cxf_path: str,
    crop_jgw: str,
    crop_size: tuple[int, int],
    codici: tuple[int, ...] = (CODICE_PARTICELLA,),
    risoluzione: float = RISOLUZIONE_M,
    margine_m: float = MARGINE_M,
    spessore: int = 1,
    stessa_griglia: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterizza la porzione di CXF che copre un crop storico.

    Ritorna (immagine, affine del raster moderno).
    """
    W_hist = read_jgw(crop_jgw)
    w_hist, h_hist = crop_size
    xmin, ymin, xmax, ymax = raster_extent(W_hist, w_hist, h_hist)

    if stessa_griglia:
        W, w, h = W_hist, w_hist, h_hist
    else:
        W, w, h = griglia(
            xmin - margine_m, ymin - margine_m, xmax + margine_m, ymax + margine_m, risoluzione
        )

    bordi = read_cxf(cxf_path).bordi
    return rasterizza(bordi, W, w, h, codici=codici, spessore=spessore), W


def _parse_codici(testo: str) -> tuple[int, ...]:
    """'18', '18,12' o '12' — l'ablation prevista da §5.4."""
    return tuple(int(c) for c in testo.split(",") if c.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rasterizza il CXF sulla zona di un crop (M7)")
    ap.add_argument("--crop", default="ribba")
    ap.add_argument("--cxf", default="data/raw/L675_004900.cxf")
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--codici", default="18", help="18 = particelle, 12 = acqua/strada, o '18,12'")
    ap.add_argument("--risoluzione", type=float, default=RISOLUZIONE_M, help="m/px del moderno")
    ap.add_argument("--margine", type=float, default=MARGINE_M, help="m di bordo oltre il crop")
    ap.add_argument("--spessore", type=int, default=1, help="spessore del tratto in px")
    ap.add_argument("--stessa-griglia", action="store_true", help="caso degenere: H_true = identità")
    ap.add_argument("--out-dir", default="data/crops")
    args = ap.parse_args(argv)

    crop_png = os.path.join(args.crops_dir, f"{args.crop}.png")
    crop_jgw = os.path.join(args.crops_dir, f"{args.crop}.jgw")
    img_hist = cv2.imread(crop_png, cv2.IMREAD_GRAYSCALE)
    if img_hist is None:
        raise FileNotFoundError(f"{crop_png} — genera prima i crop (M1)")

    codici = _parse_codici(args.codici)
    img, W = per_crop(
        args.cxf,
        crop_jgw,
        (img_hist.shape[1], img_hist.shape[0]),
        codici=codici,
        risoluzione=args.risoluzione,
        margine_m=args.margine,
        spessore=args.spessore,
        stessa_griglia=args.stessa_griglia,
    )

    suffisso = "vec" if args.codici == "18" else f"vec{args.codici.replace(',', '')}"
    base = os.path.join(args.out_dir, f"{args.crop}_{suffisso}")
    os.makedirs(args.out_dir, exist_ok=True)
    cv2.imwrite(f"{base}.png", img)
    write_jgw(f"{base}.jgw", W)

    xmin, ymin, xmax, ymax = raster_extent(W, img.shape[1], img.shape[0])
    inchiostro = float((img < 128).mean())
    etichette = {CODICE_PARTICELLA: "particelle", CODICE_ACQUA_STRADA: "acqua/strada"}
    print(f"crop storico : {crop_png}  {img_hist.shape[1]}x{img_hist.shape[0]} px")
    print(f"codici       : {codici} ({', '.join(etichette.get(c, '?') for c in codici)})")
    print(f"raster vett. : {img.shape[1]}x{img.shape[0]} px @ {args.risoluzione} m/px")
    print(f"estensione   : X {xmin:.2f} .. {xmax:.2f}   Y {ymin:.2f} .. {ymax:.2f}")
    print(f"inchiostro   : {inchiostro * 100:.2f}% dei pixel")
    print(f"scritti      : {base}.png + {base}.jgw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
