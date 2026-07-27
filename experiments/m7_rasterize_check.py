"""M7 — verifica della rasterizzazione del CXF. Visiva, a piena risoluzione (I5).

§5.5 è esplicito: su questi dati correlazione e chamfer producono falsi positivi
convincenti, e l'unica metrica è l'RMSE su checkpoint contro `H_true` (che è E2,
cioè M8). Qui non si misura l'allineamento: si guarda.

L'unico numero che compare è algebrico ed esatto — gli angoli del crop storico,
portati dal suo world file in coordinate CRS e riportati indietro dal world file
del raster vettoriale, devono cadere dove previsto. Verifica la composizione
delle affini, non la qualità della registrazione.

    python -m experiments.m7_rasterize_check --crop ribba
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from src.groundtruth import h_true, transform
from src.io_geo import apply_affine, read_jgw

# Lato del riquadro di dettaglio a 1:1, in pixel del crop storico.
LATO_DETTAGLIO = 320


def riquadro_dettaglio(vec_su_hist: np.ndarray, lato: int = LATO_DETTAGLIO, passo: int = 64):
    """Sceglie il riquadro con più tratto vettoriale.

    Un riquadro fisso finisce su una zona vuota appena si cambia crop — su
    `vedra`, che è quasi tutto carta bianca, mostrerebbe due immagini vuote e
    non verificherebbe niente. La scelta è deterministica: a parità di conteggio
    vince il riquadro più in alto a sinistra.
    """
    h, w = vec_su_hist.shape[:2]
    lato = min(lato, h, w)
    # cv2.integral non accetta int32 in ingresso: uint8 con valori 0/1, e la
    # somma esce comunque in int32.
    integrale = cv2.integral((vec_su_hist < 128).astype(np.uint8))

    migliore, conteggio_max = (0, 0), -1
    for y in range(0, h - lato + 1, passo):
        for x in range(0, w - lato + 1, passo):
            conteggio = (
                integrale[y + lato, x + lato]
                - integrale[y, x + lato]
                - integrale[y + lato, x]
                + integrale[y, x]
            )
            if conteggio > conteggio_max:
                migliore, conteggio_max = (x, y), conteggio
    return (*migliore, lato, lato)


def verifica_affini(crop_jgw: str, vec_jgw: str, size_hist, size_vec) -> np.ndarray:
    """Controllo algebrico della composizione: esatto, non statistico."""
    W_hist, W_vec = read_jgw(crop_jgw), read_jgw(vec_jgw)
    H = h_true(W_hist, W_vec)

    w, h = size_hist
    angoli = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
    via_h = transform(H, angoli)
    # giro lungo: pixel storico -> CRS -> pixel vettoriale
    via_crs = apply_affine(np.linalg.inv(W_vec), apply_affine(W_hist, angoli))
    scarto = float(np.abs(via_h - via_crs).max())

    print(f"H_true (storico -> vettoriale), scarto composizione: {scarto:.2e} px")
    print(f"  scala   : {np.hypot(H[0, 0], H[1, 0]):.6f} px moderni per px storico")
    print("  angoli del crop storico nel raster vettoriale:")
    for (cx, cy), (vx, vy) in zip(angoli, via_h):
        dentro = 0 <= vx < size_vec[0] and 0 <= vy < size_vec[1]
        print(f"    ({cx:6.0f},{cy:6.0f}) -> ({vx:8.2f},{vy:8.2f})  {'dentro' if dentro else 'FUORI'}")
    assert scarto < 1e-9, "la composizione delle affini non torna"
    return H


def figura(hist: np.ndarray, vec: np.ndarray, H: np.ndarray, out_path: str, titolo: str) -> None:
    """Storico, vettoriale, sovrapposizione a piena risoluzione e dettaglio 1:1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = hist.shape[:2]
    # H porta pixel storici su pixel vettoriali: per riportare il vettoriale
    # sulla griglia dello storico serve l'inversa.
    vec_su_hist = cv2.warpPerspective(
        vec, np.linalg.inv(H), (w, h), flags=cv2.INTER_NEAREST, borderValue=255
    )

    overlay = cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR)
    overlay[vec_su_hist < 128] = (0, 0, 220)  # BGR: rosso sul tratto vettoriale

    x, y, dw, dh = riquadro_dettaglio(vec_su_hist)
    dettaglio = overlay[y : y + dh, x : x + dw].copy()  # ritagliato prima del riquadro
    cv2.rectangle(overlay, (x, y), (x + dw, y + dh), (200, 120, 0), 3)

    fig, assi = plt.subplots(1, 4, figsize=(20, 5.6))
    for ax, (immagine, sottotitolo) in zip(
        assi,
        (
            (cv2.cvtColor(hist, cv2.COLOR_GRAY2RGB), f"storico ({w}x{h})"),
            (cv2.cvtColor(vec_su_hist, cv2.COLOR_GRAY2RGB), "vettoriale, riportato sulla griglia storica"),
            (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "sovrapposizione, piena risoluzione"),
            (cv2.cvtColor(dettaglio, cv2.COLOR_BGR2RGB), f"dettaglio 1:1 ({dw}x{dh}) a ({x},{y})"),
        ),
    ):
        ax.imshow(immagine)
        ax.set_title(sottotitolo, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(titolo, fontsize=12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"figura: {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="ribba")
    ap.add_argument("--suffisso", default="vec", help="vec = solo codice 18, vec1812 = 18+12")
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args(argv)

    hist_png = os.path.join(args.crops_dir, f"{args.crop}.png")
    vec_png = os.path.join(args.crops_dir, f"{args.crop}_{args.suffisso}.png")
    hist = cv2.imread(hist_png, cv2.IMREAD_GRAYSCALE)
    vec = cv2.imread(vec_png, cv2.IMREAD_GRAYSCALE)
    if hist is None:
        raise FileNotFoundError(f"{hist_png} — genera prima i crop (M1)")
    if vec is None:
        raise FileNotFoundError(f"{vec_png} — genera prima il raster (python -m src.prep.rasterize)")

    H = verifica_affini(
        os.path.join(args.crops_dir, f"{args.crop}.jgw"),
        os.path.join(args.crops_dir, f"{args.crop}_{args.suffisso}.jgw"),
        (hist.shape[1], hist.shape[0]),
        (vec.shape[1], vec.shape[0]),
    )
    figura(
        hist,
        vec,
        H,
        os.path.join(args.out, f"m7_{args.crop}_{args.suffisso}.png"),
        f"M7 — {args.crop} · CXF ({args.suffisso}) sul crop storico · "
        "verifica visiva a piena risoluzione, nessuna metrica indiretta (§5.5)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
