"""M10 — quante iterazioni servono a RANSAC, in funzione dell'inlier ratio.

La formula (§7.2):

    k >= ln(1-p) / ln(1 - w^s)

con p = confidence (0.995 nel progetto), w = inlier ratio vero, s = punti
minimi del modello (2 similarità, 3 affine, 4 omografia). Il grafico mostra
k(w) per i tre modelli, con il tetto di 5000 iterazioni del codice (`max_iter`
di src/estimate.py) e gli inlier ratio realmente osservati in E1 ed E2 segnati
sopra: non punti scelti per fare bella figura, ma le mediane misurate nelle
griglie di M6 e M8.

    python -m experiments.m10_ransac_iterazioni
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Gli stessi valori di default di src/estimate.py (confidence, max_iter): non
# duplicati per caso, sono il contratto reale usato da tutte le griglie.
CONFIDENCE = 0.995
MAX_ITER = 5000
MODELLI = {"similarità": 2, "affine": 3, "omografia": 4}
COLORI = {"similarità": "#2f6b4f", "affine": "#c98a1f", "omografia": "#b23a2e"}


def k_necessario(w: np.ndarray, s: int, p: float = CONFIDENCE) -> np.ndarray:
    """k >= ln(1-p) / ln(1 - w^s), con guardia sui casi degeneri (w=0 o w=1)."""
    w = np.clip(w, 1e-6, 1 - 1e-9)
    return np.log(1 - p) / np.log(1 - w**s)


def punti_osservati(csv_path: str) -> list[dict]:
    """Inlier ratio realmente misurati nel CSV, non inventati per la figura."""
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    punti = []
    e1 = df[(df.esperimento == "E1") & (df.matcher == "sift") & (df.degrado == 0) & (df.rot_deg == 15)]
    if len(e1):
        punti.append({"etichetta": "E1 · SIFT, senza degrado", "w": e1.inlier_ratio.median(), "s": 2})
    e2 = df[
        (df.esperimento == "E2") & (df.matcher == "orb") & (df.preprocess == "sauvola") & (df.modello == "similarity")
    ]
    if len(e2):
        punti.append({"etichetta": "E2 · ORB+Sauvola, mediana", "w": e2.inlier_ratio.median(), "s": 2})
    # stessa identica configurazione del punto precedente (ORB+Sauvola+similarità),
    # isolata sul solo ritaglio aspera: non una media su preprocessing diversi,
    # altrimenti il numero non sarebbe più confrontabile con quello citato in
    # relazione per lo stesso caso (§9.5).
    peggio = df[
        (df.esperimento == "E2")
        & (df.crop == "aspera")
        & (df.codici == "18")
        & (df.matcher == "orb")
        & (df.preprocess == "sauvola")
        & (df.modello == "similarity")
    ]
    if len(peggio):
        punti.append({"etichetta": "E2 · aspera, solo codice 18", "w": float(peggio.inlier_ratio.iloc[0]), "s": 2})
    return punti


def figura(out_path: str, csv_path: str) -> None:
    w = np.linspace(0.005, 1.0, 400)
    fig, ax = plt.subplots(figsize=(9.5, 6))

    for nome, s in MODELLI.items():
        ax.plot(w, k_necessario(w, s), label=f"{nome}  (s={s})", color=COLORI[nome], linewidth=2)

    ax.axhline(MAX_ITER, color="black", linestyle="--", linewidth=1.1)
    ax.text(0.55, MAX_ITER * 1.15, f"tetto del codice: max_iter = {MAX_ITER}", fontsize=9)

    # scostamenti verticali scelti a mano per non incrociare la linea del tetto
    scarti = {"E2 · ORB+Sauvola, mediana": -22}
    for p in punti_osservati(csv_path):
        k = float(k_necessario(np.array([p["w"]]), p["s"])[0])
        ax.plot(p["w"], k, "o", color="black", markersize=6, zorder=5)
        ax.annotate(
            f"{p['etichetta']}\nw={p['w']:.3f} → k≈{k:,.0f}".replace(",", " "),
            (p["w"], k),
            textcoords="offset points",
            xytext=(10, scarti.get(p["etichetta"], 12)),
            fontsize=8,
        )

    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)  # spazio per il titolo su due righe
    ax.set_xlabel("inlier ratio vero w")
    ax.set_ylabel("iterazioni k necessarie (scala logaritmica)")
    ax.set_title(
        "Iterazioni richieste da RANSAC per una confidenza del 99.5%\n"
        "k = ln(1-p) / ln(1 - w^s): più il modello ha parametri, più k esplode quando w è piccolo",
        fontsize=11.5,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"figura: {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/runs.csv")
    ap.add_argument("--out", default="results/figures/m10_ransac_iterazioni.png")
    args = ap.parse_args(argv)
    figura(args.out, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
