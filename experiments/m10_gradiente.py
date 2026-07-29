"""M10 — luminosità, gradiente e descrittore, su un pezzo vero di mappa.

Chiarisce una confusione comune: la luminosità è un numero per pixel, il
gradiente è un vettore per pixel (quanto e in che direzione cambia la
luminosità lì), il descrittore SIFT è UN SOLO vettore a 128 numeri che
raccoglie i gradienti di molti pixel vicini in un istogramma di direzioni.

Quattro pannelli su un dettaglio reale:
  1. luminosità   — il valore di grigio, pixel per pixel
  2. Gx, Gy       — le due derivate: variazione orizzontale e verticale
  3. magnitudine  — quanto è forte il bordo in quel punto (√(Gx²+Gy²))
  4. direzione    — verso cui punta il bordo, colorata (ruota HSV)

    python -m experiments.m10_gradiente --crop ribba
"""
from __future__ import annotations

import argparse
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def figura(crop: str, crops_dir: str, out_path: str, riquadro=(430, 300, 140, 140)) -> None:
    img = cv2.imread(os.path.join(crops_dir, f"{crop}.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{crop}: manca il ritaglio (M1)")
    x, y, w, h = riquadro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y : y + h, x : x + w].astype(np.float32)

    # Le derivate: quanto cambia la luminosità andando a destra (Gx) e in basso
    # (Gy). Sobel le stima con un kernel 3x3, non con la differenza fra due
    # pixel nudi, perché è meno sensibile al rumore di scansione.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitudine = np.hypot(gx, gy)
    direzione = (np.degrees(np.arctan2(gy, gx)) + 180) % 360  # 0-360

    # direzione come tinta HSV, magnitudine come luminosità del colore: dove il
    # gradiente è quasi nullo (carta uniforme) il colore è nero indipendentemente
    # dalla direzione, che lì non significa nulla.
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (direzione / 2).astype(np.uint8)  # OpenCV vuole H in [0,180)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitudine / magnitudine.max() * 255, 0, 255).astype(np.uint8)
    direzione_colorata = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    fig, assi = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (immagine, titolo, cmap) in zip(
        assi,
        (
            (gray, "1 · luminosità\n(un numero per pixel: 0=nero, 255=bianco)", "gray"),
            (magnitudine, "2 · magnitudine del gradiente\n(quanto è forte il bordo, lì)", "inferno"),
            (direzione_colorata, "3 · direzione del gradiente\n(colore = verso; nero = carta uniforme)", None),
            (None, "4 · il descrittore SIFT\n(prossimo pannello)", None),
        ),
    ):
        if immagine is not None:
            ax.imshow(immagine, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titolo, fontsize=9.5)

    # Pannello 4: l'istogramma di orientazioni pesato dalla magnitudine, la
    # stessa aritmetica del descrittore SIFT (qui su tutta la finestra, non
    # suddivisa in 4x4 celle come nel descrittore vero, per tenerlo leggibile).
    assi[3].remove()
    ax4 = fig.add_subplot(1, 4, 4, projection="polar")
    bins = np.linspace(0, 2 * np.pi, 9)
    pesi, _ = np.histogram(np.radians(direzione), bins=bins, weights=magnitudine)
    ax4.bar(bins[:-1], pesi, width=2 * np.pi / 8, align="edge", color="#2f6b4f", edgecolor="white")
    ax4.set_title("4 · l'istogramma di orientazioni\n(8 direzioni, pesate per magnitudine)", fontsize=9.5, pad=18)
    ax4.set_yticklabels([])

    fig.suptitle(
        f"Da un pixel al descrittore — {crop}, dettaglio {w}x{h} px a piena risoluzione.\n"
        "Un numero per pixel (1) → un vettore per pixel (2+3) → un istogramma per l'intero riquadro (4).\n"
        "Il vero descrittore SIFT ripete il pannello 4 su 16 sotto-finestre e le incolla: 16 × 8 = 128 numeri.",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"figura: {out_path}")
    print(f"magnitudine: min {magnitudine.min():.1f}  max {magnitudine.max():.1f}  media {magnitudine.mean():.1f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="ribba")
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--out", default="results/figures/m10_gradiente.png")
    args = ap.parse_args(argv)
    figura(args.crop, args.crops_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
