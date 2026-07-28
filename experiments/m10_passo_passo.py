"""M10 — una registrazione seguita passo per passo, per la relazione.

Serve a rispondere alla domanda "ma in concreto cosa fa il programma?" con le
immagini invece che con le parole. Segue una singola esecuzione dall'inizio alla
fine e mostra che cosa entra e che cosa esce da ogni stadio:

  1. la scansione storica            4. gli abbinamenti trovati (quasi tutti sbagliati)
  2. la stessa, dopo la pulizia      5. quelli sopravvissuti alla votazione di RANSAC
  3. il vettoriale disegnato         6. la sovrapposizione finale

    python -m experiments.m10_passo_passo --crop ribba
"""
from __future__ import annotations

import argparse
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluate import valuta  # noqa: E402
from src.groundtruth import h_true_from_jgw  # noqa: E402
from src.io_geo import read_jgw  # noqa: E402
from src.pipeline import Opzioni, registra  # noqa: E402
from src.preprocess import applica  # noqa: E402


def _affianca(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """Mette due immagini una accanto all'altra. Ritorna anche l'offset della seconda."""
    a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else a.copy()
    b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR) if b.ndim == 2 else b.copy()
    h = max(a.shape[0], b.shape[0])
    tela = np.full((h, a.shape[1] + b.shape[1], 3), 255, np.uint8)
    tela[: a.shape[0], : a.shape[1]] = a
    tela[: b.shape[0], a.shape[1] :] = b
    return tela, a.shape[1]


def _disegna_linee(tela, offset, pts_a, pts_b, colore, spessore=1, quanti=60, seed=42):
    """Traccia le linee di corrispondenza fra i due lati, su un campione casuale."""
    if len(pts_a) == 0:
        return tela
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts_a), size=min(quanti, len(pts_a)), replace=False)
    for i in idx:
        xa, ya = pts_a[i]
        xb, yb = pts_b[i]
        cv2.line(
            tela,
            (int(round(xa)), int(round(ya))),
            (int(round(xb + offset)), int(round(yb))),
            colore,
            spessore,
            cv2.LINE_AA,
        )
    return tela


def figura(crop: str, crops_dir: str, out_path: str, opz: Opzioni) -> dict:
    hist = cv2.imread(os.path.join(crops_dir, f"{crop}.png"), cv2.IMREAD_COLOR)
    vec = cv2.imread(os.path.join(crops_dir, f"{crop}_vec1812.png"), cv2.IMREAD_COLOR)
    if hist is None or vec is None:
        raise FileNotFoundError(f"{crop}: mancano il ritaglio o il raster vettoriale (M1/M7)")

    ris = registra(hist, vec, opz)
    H_true = h_true_from_jgw(
        os.path.join(crops_dir, f"{crop}.jgw"), os.path.join(crops_dir, f"{crop}_vec1812.jgw")
    )
    metriche = valuta(
        ris.stima,
        H_true,
        hist.shape[1],
        hist.shape[0],
        W_hist=read_jgw(os.path.join(crops_dir, f"{crop}.jgw")),
        soglia_m=2.0,
    )

    hist_pulito = applica(hist, modo=opz.preprocess, morph_close=opz.morph_close)
    vec_pulito = applica(vec, modo=opz.preprocess, morph_close=opz.morph_close)

    # 4 e 5: gli abbinamenti, prima e dopo la votazione
    tela_tutti, off = _affianca(hist_pulito, vec_pulito)
    _disegna_linee(tela_tutti, off, ris.pts_hist, ris.pts_modern, (0, 0, 220))

    inlier = (
        np.flatnonzero(ris.stima.inliers) if ris.stima.inliers is not None else np.array([], int)
    )
    tela_inlier, off2 = _affianca(hist_pulito, vec_pulito)
    _disegna_linee(
        tela_inlier, off2, ris.pts_hist[inlier], ris.pts_modern[inlier], (0, 160, 0), spessore=2
    )

    # 6: la sovrapposizione
    h, w = vec.shape[:2]
    warp = cv2.warpPerspective(hist, ris.H, (w, h), borderValue=(255, 255, 255))
    overlay = vec.copy()
    overlay[cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) < 128] = (0, 0, 220)

    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.72, 1.0])

    def _pannello(ax, img, titolo, didascalia):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap="gray")
        ax.set_title(titolo, fontsize=11, weight="bold")
        ax.set_xlabel(didascalia, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    _pannello(
        fig.add_subplot(gs[0, 0]),
        hist,
        "1 · la scansione storica",
        "il foglio disegnato a mano,\ncosì come arriva dall'archivio",
    )
    _pannello(
        fig.add_subplot(gs[0, 1]),
        hist_pulito,
        f"2 · la stessa, ripulita ({opz.preprocess})",
        "ogni pixel è deciso: inchiostro o carta.\nLa soglia è calcolata zona per zona",
    )
    _pannello(
        fig.add_subplot(gs[0, 2]),
        vec_pulito,
        "3 · il vettoriale, disegnato",
        "le coordinate del CXF tracciate su\nun'immagine: ora sono confrontabili",
    )

    _pannello(
        fig.add_subplot(gs[1, :]),
        tela_tutti,
        f"4 · gli abbinamenti trovati ({ris.stima.n_matches} in tutto, ne sono disegnati 60 a caso)",
        "ogni linea unisce due punti che il programma ritiene la stessa cosa. "
        "Se fossero giusti sarebbero tutti paralleli: non lo sono, "
        f"perché solo il {100 * ris.stima.inlier_ratio:.0f}% è corretto",
    )
    _pannello(
        fig.add_subplot(gs[2, 0]),
        tela_inlier,
        f"5 · dopo la votazione ({ris.stima.n_inliers} sopravvissuti)",
        "RANSAC tiene solo gli abbinamenti che\nconcordano su una stessa trasformazione",
    )

    ax6 = fig.add_subplot(gs[2, 1:])
    _pannello(
        ax6,
        overlay,
        "6 · il risultato",
        "lo storico, deformato dalla trasformazione stimata, sovrapposto in rosso al vettoriale",
    )

    esito = "riuscita" if metriche["success"] else "fallita"
    fig.suptitle(
        f"Una registrazione dall'inizio alla fine — ritaglio «{crop}», "
        f"{opz.matcher.upper()} · {opz.preprocess} · {opz.model}\n"
        f"errore finale {metriche['rmse_m']:.2f} m su un riferimento che ne vale ~0.5: "
        f"registrazione {esito}",
        fontsize=13,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"figura: {out_path}")
    print(
        f"  match {ris.stima.n_matches} · inlier {ris.stima.n_inliers} "
        f"({ris.stima.inlier_ratio:.3f}) · RMSE {metriche['rmse_m']:.3f} m"
    )
    return metriche


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="ribba")
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--matcher", default="orb")
    ap.add_argument("--preprocess", default="sauvola")
    ap.add_argument("--model", default="similarity")
    ap.add_argument("--out", default="results/figures/m10_passo_passo.png")
    args = ap.parse_args(argv)

    figura(
        args.crop,
        args.crops_dir,
        args.out,
        Opzioni(matcher=args.matcher, preprocess=args.preprocess, model=args.model),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
