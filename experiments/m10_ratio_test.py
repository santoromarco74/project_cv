"""M10 — il ratio test di Lowe, illustrato con i rapporti d1/d2 veri.

Per ogni keypoint SIFT dell'immagine A, calcola le distanze dai due migliori
candidati in B e mostra la distribuzione di d1/d2, con la soglia di §7.1 (0.75)
segnata sopra. È la stessa quantità che `SiftMatcher.match` calcola internamente
per filtrare — qui la si guarda invece di limitarsi a usarla.

Confronta lo stesso ritaglio in due condizioni:
- **stesso dominio** (E1): storico contro sé stesso ruotato — il caso in cui il
  ratio test dovrebbe avere vita facile;
- **cross-domain** (E2): storico contro il raster del CXF — il caso difficile.

    python -m experiments.m10_ratio_test --crop ribba
"""
from __future__ import annotations

import argparse
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.prep.synth import Trasformazione, genera_coppia  # noqa: E402

SOGLIA = 0.75


def _rapporti(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, int, int]:
    """d1/d2 per ogni keypoint di A che ha almeno due candidati in B."""
    sift = cv2.SIFT_create()
    gray = lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i
    kp_a, des_a = sift.detectAndCompute(gray(img_a), None)
    kp_b, des_b = sift.detectAndCompute(gray(img_b), None)
    if des_a is None or des_b is None or len(kp_b) < 2:
        return np.array([]), len(kp_a), len(kp_b)
    coppie = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    rapporti = np.array([m.distance / n.distance for m, n in coppie if n.distance > 0])
    return rapporti, len(kp_a), len(kp_b)


def _pannello(ax, rapporti: np.ndarray, titolo: str) -> None:
    if len(rapporti) == 0:
        ax.set_title(f"{titolo}\n(nessun keypoint)", fontsize=10)
        return
    accettati = rapporti < SOGLIA
    ax.hist(rapporti[~accettati], bins=40, range=(0, 1.4), color="#b23a2e", alpha=0.75, label="scartati")
    ax.hist(rapporti[accettati], bins=40, range=(0, 1.4), color="#2f6b4f", alpha=0.85, label="accettati")
    ax.axvline(SOGLIA, color="black", linestyle="--", linewidth=1.2)
    ax.text(SOGLIA + 0.02, ax.get_ylim()[1] * 0.92, f"soglia {SOGLIA}", fontsize=8)
    ax.set_xlabel("d1 / d2  (distanza dal migliore ÷ distanza dal secondo migliore)")
    ax.set_ylabel("keypoint")
    pct = 100 * accettati.mean()
    ax.set_title(f"{titolo}\n{len(rapporti)} keypoint · {pct:.1f}% sopravvive al ratio test", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def figura(crop: str, crops_dir: str, out_path: str) -> None:
    hist = cv2.imread(os.path.join(crops_dir, f"{crop}.png"), cv2.IMREAD_COLOR)
    vec = cv2.imread(os.path.join(crops_dir, f"{crop}_vec1812.png"), cv2.IMREAD_COLOR)
    if hist is None or vec is None:
        raise FileNotFoundError(f"{crop}: mancano il ritaglio o il raster vettoriale (M1/M7)")

    ruotato, _ = genera_coppia(hist, Trasformazione(rot_deg=15, scala=1.15, tx=30, ty=20))
    r_e1, n_a1, n_b1 = _rapporti(hist, ruotato)
    r_e2, n_a2, n_b2 = _rapporti(hist, vec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _pannello(ax1, r_e1, f"stesso dominio — {crop} contro sé stesso ruotato\n({n_a1} vs {n_b1} keypoint)")
    _pannello(ax2, r_e2, f"cross-domain — {crop} contro il vettoriale\n({n_a2} vs {n_b2} keypoint)")
    fig.suptitle(
        "Il ratio test di Lowe: quanto il migliore candidato si stacca dal secondo\n"
        "verde = accettato (d1/d2 < 0.75) · rosso = scartato",
        fontsize=12,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"figura: {out_path}")
    print(f"E1 (stesso dominio): {100 * (r_e1 < SOGLIA).mean():.1f}% accettati su {len(r_e1)}")
    print(f"E2 (cross-domain)  : {100 * (r_e2 < SOGLIA).mean():.1f}% accettati su {len(r_e2)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="ribba")
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--out", default="results/figures/m10_ratio_test.png")
    args = ap.parse_args(argv)
    figura(args.crop, args.crops_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
