"""Ritagli dal foglio intero. Coordinate dei 5 crop in CLAUDE.md §5.6.

Ogni crop viene salvato come PNG (lossless, colore originale) accompagnato dal
proprio world file: la georeferenziazione resta un metadato ACCANTO al dato, che
solo evaluate.py leggerà (I3). La pipeline vede il PNG e basta.

    python -m src.prep.crop --sheet data/raw/L675_004900.jpg \\
                            --jgw   data/raw/L675_004900.jgw \\
                            --out   data/crops
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

from src.io_geo import apply_affine, raster_extent, read_jgw, write_jgw


@dataclass(frozen=True)
class CropSpec:
    """Un ritaglio: offset e dimensioni in pixel, più l'estensione attesa da §5.6.

    L'estensione attesa non serve al ritaglio — si ricalcola dal JGW — ma è il
    controllo che il world file del crop sia composto correttamente.
    """

    nome: str
    x0: int
    y0: int
    w: int
    h: int
    x_attesa: tuple[float, float]
    y_attesa: tuple[float, float]


# CLAUDE.md §5.6: fascia collinare, dove il vettoriale copre a pieno. Fuori restano
# Celle Ligure (vuoto in entrambi) e l'ente urbano (non coperto dal catasto terreni).
CROPS: tuple[CropSpec, ...] = (
    CropSpec("tassarole", 1500, 300, 1024, 1024, (-31098, -30838), (-11616, -11355)),
    # Recentrato sul punto reale dell'etichetta "Cannei" nel CXF (TESTO,
    # X=-30561.997 Y=-11516.857 -> pixel 3609,936): il box originario
    # (2900,200,1200x1000) conteneva il punto ma sbilanciato al 59%/74%,
    # con margine superfluo verso Tassarole/Piano. Uniformato a 1024x1024
    # come gli altri crop.
    CropSpec("cannei", 3100, 420, 1024, 1024, (-30691, -30431), (-11646, -11386)),
    # x0/y0 corretti sul punto reale dell'etichetta "Ribba" nel CXF (TESTO,
    # X=-30392.563 Y=-11604.184 -> pixel 4274,1279): l'offset originario
    # inquadrava perlopiù Cannei, con "Ribba" tagliata a 50 px dal bordo destro.
    CropSpec("ribba", 3850, 700, 1024, 1024, (-30500, -30240), (-11717, -11457)),
    # y0 corretto sul punto reale dell'etichetta "Vedra'" nel CXF (TESTO,
    # X=-30276.945 Y=-11728.872 -> pixel 4728,1769): l'offset originario
    # (y0=2000) escludeva il punto, che cadeva 231 px SOPRA il bordo
    # superiore del box -- non sfiorato, proprio fuori. x0 invariato: sul
    # quell'asse il punto era già ben posizionato (51.6% dal bordo sinistro).
    CropSpec("vedra", 4200, 1450, 1024, 1024, (-30411, -30151), (-11908, -11648)),
    CropSpec("aspera", 5600, 2600, 1024, 1024, (-30055, -29795), (-12201, -11940)),
)

# Tolleranza sul confronto con le estensioni tabulate in §5.6, che sono arrotondate
# al metro.
TOLLERANZA_ESTENSIONE_M = 1.0


def crop_world_file(W: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Affine del ritaglio: stessa scala e rotazione, origine traslata di (x0, y0).

    Con col = col' + x0 e row = row' + y0 si ottiene W' = W @ T, cioè
    C' = C + A*x0 + B*y0 e F' = F + D*x0 + E*y0 (qui B = D = 0).
    """
    T = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
    return W @ T


def read_sheet(path: str) -> np.ndarray:
    """Carica il foglio. Nessuna dimensione hardcodata: si legge dal file."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"impossibile leggere il raster: {path}")
    return img


def extract(sheet: np.ndarray, W: np.ndarray, spec: CropSpec) -> tuple[np.ndarray, np.ndarray]:
    """Ritaglia e compone il world file del ritaglio. Ritorna (immagine, affine)."""
    h, w = sheet.shape[:2]
    if spec.x0 < 0 or spec.y0 < 0 or spec.x0 + spec.w > w or spec.y0 + spec.h > h:
        raise ValueError(
            f"{spec.nome}: ritaglio ({spec.x0},{spec.y0},{spec.w},{spec.h}) "
            f"fuori dal foglio {w}x{h}"
        )
    patch = sheet[spec.y0 : spec.y0 + spec.h, spec.x0 : spec.x0 + spec.w].copy()
    return patch, crop_world_file(W, spec.x0, spec.y0)


def check_extent(spec: CropSpec, Wc: np.ndarray) -> tuple[float, float, float, float]:
    """Verifica che l'estensione ricalcolata coincida con quella tabulata in §5.6."""
    xmin, ymin, xmax, ymax = raster_extent(Wc, spec.w, spec.h)
    attesi = (min(spec.x_attesa), min(spec.y_attesa), max(spec.x_attesa), max(spec.y_attesa))
    for valore, atteso, etichetta in zip(
        (xmin, ymin, xmax, ymax), attesi, ("xmin", "ymin", "xmax", "ymax")
    ):
        if abs(valore - atteso) > TOLLERANZA_ESTENSIONE_M:
            raise AssertionError(
                f"{spec.nome}: {etichetta} = {valore:.2f} m, atteso {atteso:.0f} m da §5.6 "
                f"(scarto {valore - atteso:+.2f} m)"
            )
    return xmin, ymin, xmax, ymax


def generate(
    sheet_path: str, jgw_path: str, out_dir: str, only: str | None = None
) -> list[dict]:
    """Genera i crop richiesti. Ritorna una riga di riepilogo per ciascuno."""
    W = read_jgw(jgw_path)
    sheet = read_sheet(sheet_path)
    os.makedirs(out_dir, exist_ok=True)

    specs = [s for s in CROPS if only is None or s.nome == only]
    if not specs:
        raise ValueError(f"crop sconosciuto: {only!r} (disponibili: {[s.nome for s in CROPS]})")

    righe = []
    for spec in specs:
        patch, Wc = extract(sheet, W, spec)
        xmin, ymin, xmax, ymax = check_extent(spec, Wc)
        png = os.path.join(out_dir, f"{spec.nome}.png")
        jgw = os.path.join(out_dir, f"{spec.nome}.jgw")
        cv2.imwrite(png, patch)
        write_jgw(jgw, Wc)
        righe.append(
            {
                "nome": spec.nome,
                "png": png,
                "jgw": jgw,
                "px": f"{spec.x0},{spec.y0},{spec.w}x{spec.h}",
                "x": (xmin, xmax),
                "y": (ymin, ymax),
                "m": f"{xmax - xmin:.0f}x{ymax - ymin:.0f}",
            }
        )
    return righe


def contact_sheet(sheet: np.ndarray, out_path: str, scala: int = 8) -> None:
    """Figura di riepilogo: i 5 ritagli affiancati e la loro posizione sul foglio.

    Il foglio è mostrato ridotto SOLO per indicare dove cadono i ritagli. Non è, e
    non va usata come, una verifica di allineamento: quella si fa a piena
    risoluzione (I5, §5.5).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    h, w = sheet.shape[:2]
    small = cv2.cvtColor(cv2.resize(sheet, (w // scala, h // scala)), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.6, 1.0])

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(small)
    ax.set_title(
        f"Foglio 49 Varazze — {w}x{h} px (ridotto {scala}x, solo per collocare i ritagli)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spec in CROPS:
        ax.add_patch(
            mpatches.Rectangle(
                (spec.x0 / scala, spec.y0 / scala),
                spec.w / scala,
                spec.h / scala,
                fill=False,
                edgecolor="red",
                linewidth=1.4,
            )
        )
        ax.text(
            spec.x0 / scala,
            spec.y0 / scala - 6,
            spec.nome,
            color="red",
            fontsize=8,
            weight="bold",
        )

    for i, spec in enumerate(CROPS):
        axi = fig.add_subplot(gs[1, i])
        patch = sheet[spec.y0 : spec.y0 + spec.h, spec.x0 : spec.x0 + spec.w]
        axi.imshow(cv2.cvtColor(cv2.resize(patch, (256, 256)), cv2.COLOR_BGR2RGB))
        axi.set_title(f"{spec.nome}\n{spec.w}x{spec.h} px", fontsize=8)
        axi.set_xticks([])
        axi.set_yticks([])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera i ritagli del foglio (CLAUDE.md §5.6)")
    ap.add_argument("--sheet", default="data/raw/L675_004900.jpg", help="raster del foglio")
    ap.add_argument("--jgw", default="data/raw/L675_004900.jgw", help="world file del foglio")
    ap.add_argument("--out", default="data/crops", help="cartella di destinazione")
    ap.add_argument("--only", default=None, help="genera un solo crop, per nome")
    ap.add_argument(
        "--figure",
        default="results/figures/m1_crops.png",
        help="figura di riepilogo ('' per non generarla)",
    )
    args = ap.parse_args(argv)

    W = read_jgw(args.jgw)
    sheet = read_sheet(args.sheet)
    h, w = sheet.shape[:2]
    xmin, ymin, xmax, ymax = raster_extent(W, w, h)
    px = apply_affine(W, [[0, 0]])[0]
    print(f"foglio      : {args.sheet}  {w} x {h} px")
    print(f"pixel (0,0) : X {px[0]:.6f}  Y {px[1]:.6f}")
    print(f"estensione  : X {xmin:.2f} .. {xmax:.2f}   Y {ymin:.2f} .. {ymax:.2f}")
    print()

    righe = generate(args.sheet, args.jgw, args.out, args.only)
    print(f"{'nome':<11} {'pixel':<20} {'X':<24} {'Y':<24} {'m'}")
    for r in righe:
        print(
            f"{r['nome']:<11} {r['px']:<20} "
            f"{r['x'][0]:>11.2f}…{r['x'][1]:<11.2f} "
            f"{r['y'][0]:>11.2f}…{r['y'][1]:<11.2f} {r['m']}"
        )
    print(f"\n{len(righe)} crop scritti in {args.out} (PNG + .jgw affiancato)")

    if args.figure:
        contact_sheet(sheet, args.figure)
        print(f"figura: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
