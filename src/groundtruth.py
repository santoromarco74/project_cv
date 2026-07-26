"""Trasformazione di riferimento fra due raster georeferenziati nello stesso sistema.

    H_true = W_moderno⁻¹ ∘ W_storico

Entrambe le affini sono note dai world file (CLAUDE.md §5.3): la ground truth è
esatta, senza un solo GCP annotato (I2). `H_true` porta un pixel dell'immagine
storica sul pixel corrispondente dell'immagine moderna.

⚠ I3 — questo modulo è importato SOLO da `evaluate.py`. Mai da `pipeline.py`,
mai da un matcher. Se un pixel di georeferenziazione filtra nel matching il
risultato è privo di significato. La separazione è verificata da un test
(`test_i3_groundtruth_fuori_dalla_pipeline`).
"""
from __future__ import annotations

import numpy as np

from src.io_geo import pixel_size_m, read_jgw


def h_true(W_hist: np.ndarray, W_modern: np.ndarray) -> np.ndarray:
    """Compone le due affini: pixel storico -> CRS -> pixel moderno.

    W_hist e W_modern vanno da pixel a CRS (vedi io_geo.read_jgw), quindi
    l'andata è W_modern⁻¹ @ W_hist. Il risultato è una 3x3 in coordinate
    omogenee, direttamente confrontabile con l'H stimata dalla pipeline.
    """
    return np.linalg.inv(W_modern) @ W_hist


def h_true_from_jgw(jgw_hist: str, jgw_modern: str) -> np.ndarray:
    """Come h_true, leggendo le due affini dai world file."""
    return h_true(read_jgw(jgw_hist), read_jgw(jgw_modern))


def transform(H: np.ndarray, pts) -> np.ndarray:
    """Applica una 3x3 (affine o omografia) a punti (N,2), con divisione omogenea."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homo = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return homo[:, :2] / homo[:, 2:3]


def checkpoints(width: int, height: int, n: int = 10, margine: float = 0.05) -> np.ndarray:
    """Griglia regolare di checkpoint nell'immagine storica (§7.4), bordi esclusi.

    Ritorna (n*n, 2) in coordinate pixel (col, row). Il margine è la frazione di
    lato esclusa da ogni bordo: i punti sul bordo sono i più sensibili
    all'estrapolazione e non sono rappresentativi.
    """
    cols = np.linspace(margine * width, (1 - margine) * width, n)
    rows = np.linspace(margine * height, (1 - margine) * height, n)
    cc, rr = np.meshgrid(cols, rows)
    return np.column_stack([cc.ravel(), rr.ravel()])


def errore_px_to_m(errore_px: float, W_hist: np.ndarray) -> float:
    """Converte un errore in pixel dell'immagine storica in metri (§7.4).

    La risoluzione si legge dall'affine, non da una costante: 0.254453 m/px è il
    valore di questo foglio, non una proprietà del codice.
    """
    sx, sy = pixel_size_m(W_hist)
    return float(errore_px * (sx + sy) / 2.0)


def residuo_andata_ritorno(H: np.ndarray, pts) -> float:
    """Errore massimo di un giro completo H⁻¹(H(p)) - p, in pixel.

    Criterio d'accettazione di M3: sotto 1e-9. Non misura la qualità della
    georeferenziazione, misura che la composizione sia algebricamente corretta.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ritorno = transform(np.linalg.inv(H), transform(H, pts))
    return float(np.abs(ritorno - pts).max())


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Compone la trasformazione di riferimento (M3)")
    ap.add_argument("--jgw-hist", required=True, help="world file dell'immagine storica")
    ap.add_argument("--jgw-modern", required=True, help="world file dell'immagine moderna")
    ap.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), default=(1024, 1024))
    args = ap.parse_args(argv)

    W_hist = read_jgw(args.jgw_hist)
    W_modern = read_jgw(args.jgw_modern)
    H = h_true(W_hist, W_modern)

    np.set_printoptions(precision=6, suppress=True)
    print("H_true (pixel storico -> pixel moderno):")
    print(H)
    sx, sy = pixel_size_m(W_hist)
    print(f"\nrisoluzione storica : {sx:.6f} x {sy:.6f} m/px")
    pts = checkpoints(*args.size)
    print(f"checkpoint          : {len(pts)} su griglia {args.size[0]}x{args.size[1]}")
    print(f"residuo andata/ritorno: {residuo_andata_ritorno(H, pts):.3e} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
