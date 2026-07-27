"""Dataset sintetico E1: crop vs se stesso trasformato con H nota. Ground truth esatta.

L'immagine B si ottiene da A con una trasformazione decisa da noi, quindi
`H_true` è nota per costruzione e ha incertezza zero — a differenza di E2, dove
il riferimento ha un pavimento di ~0.5 m (§5.3).

A cosa serve (§8): isolare il matcher dal domain gap. Qui tutto **deve**
funzionare. Se E1 fallisce, il bug è nel codice, non nei dati.

Determinismo (I9): ogni sorgente di casualità passa da un `Generator` seminato.
Stessi parametri e stesso seed = stessi pixel, bit per bit.

    python -m src.prep.synth --crop data/crops/ribba.png --rot 15 --scala 1.2
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Trasformazione:
    """Parametri geometrici della trasformazione sintetica.

    Rotazione e scala agiscono attorno al centro dell'immagine: attorno
    all'origine porterebbero il contenuto fuori dal riquadro anche per angoli
    piccoli, e si finirebbe per misurare quanta immagine è rimasta dentro invece
    della qualità del matching.
    """

    rot_deg: float = 0.0
    scala: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    prospettiva: float = 0.0  # omografia lieve; 1e-4 è già una deformazione visibile


@dataclass(frozen=True)
class Degradazione:
    """Degradazione radiometrica opzionale, indipendente dalla geometria."""

    rumore: float = 0.0  # deviazione standard del rumore gaussiano, in livelli
    blur: float = 0.0  # sigma del blur gaussiano, in pixel
    contrasto: float = 1.0  # <1 schiaccia il contrasto
    luminosita: float = 0.0  # offset additivo, in livelli


def matrice(t: Trasformazione, width: int, height: int) -> np.ndarray:
    """Costruisce H (A -> B) dai parametri, centrata sull'immagine."""
    cx, cy = width / 2.0, height / 2.0
    ang = np.deg2rad(t.rot_deg)
    cos, sin = np.cos(ang) * t.scala, np.sin(ang) * t.scala

    centra = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    ruota_scala = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)
    prospettiva = np.array(
        [[1, 0, 0], [0, 1, 0], [t.prospettiva, t.prospettiva, 1]], dtype=np.float64
    )
    decentra = np.array([[1, 0, cx + t.tx], [0, 1, cy + t.ty], [0, 0, 1]], dtype=np.float64)

    H = decentra @ prospettiva @ ruota_scala @ centra
    return H / H[2, 2]


def degrada(img: np.ndarray, d: Degradazione, rng: np.random.Generator) -> np.ndarray:
    """Applica la degradazione radiometrica. Ordine: contrasto, blur, rumore.

    Il rumore va per ultimo: è rumore di acquisizione, non un dettaglio
    dell'immagine, quindi non deve essere sfocato dal blur.
    """
    out = img.astype(np.float32)
    if d.contrasto != 1.0 or d.luminosita != 0.0:
        out = out * d.contrasto + d.luminosita
    if d.blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), d.blur)
    if d.rumore > 0:
        out = out + rng.normal(0.0, d.rumore, size=out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def genera_coppia(
    img: np.ndarray,
    t: Trasformazione,
    d: Degradazione | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Da un'immagine A produce (B, H_true) con B = warp(A, H_true).

    B ha le stesse dimensioni di A: la parte che esce dal riquadro si perde, ed è
    voluto — è il caso realistico di due immagini che si sovrappongono solo in
    parte.
    """
    h, w = img.shape[:2]
    H = matrice(t, w, h)
    warped = cv2.warpPerspective(
        img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    if d is not None:
        warped = degrada(warped, d, np.random.default_rng(seed))
    return warped, H


CONTRASTO_MINIMO = 0.15


def scala_degradazione(livello: float) -> Degradazione:
    """Degradazione parametrizzata da un solo numero, tipicamente in [0, 1.5].

    Serve alla curva RMSE vs degradazione di M6: una variabile indipendente
    sola, monotona, così l'asse x della figura ha un significato.

    Il livello 1 non è un massimo: su questi dati SIFT lo regge ancora (RMSE
    ~0.16 px), e la soglia di rottura sta poco sopra. Livelli maggiori sono
    ammessi, con il contrasto limitato dal basso — a `1 - 0.5·livello` puro
    diventerebbe nullo a 2 e negativo oltre, cioè un'immagine uniforme o
    invertita: non più una degradazione ma un'immagine diversa.
    """
    return Degradazione(
        rumore=20.0 * livello,
        blur=2.0 * livello,
        contrasto=max(CONTRASTO_MINIMO, 1.0 - 0.5 * livello),
        luminosita=20.0 * livello,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera una coppia sintetica E1 (M4)")
    ap.add_argument("--crop", default="data/crops/ribba.png")
    ap.add_argument("--rot", type=float, default=10.0, help="rotazione in gradi")
    ap.add_argument("--scala", type=float, default=1.0)
    ap.add_argument("--tx", type=float, default=0.0)
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--prospettiva", type=float, default=0.0)
    ap.add_argument("--degrado", type=float, default=0.0, help="livello in [0,1]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/figures/e1_coppia.png")
    args = ap.parse_args(argv)

    img = cv2.imread(args.crop, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.crop)
    t = Trasformazione(args.rot, args.scala, args.tx, args.ty, args.prospettiva)
    d = scala_degradazione(args.degrado) if args.degrado > 0 else None
    b, H = genera_coppia(img, t, d, args.seed)

    np.set_printoptions(precision=6, suppress=True)
    print(f"crop    : {args.crop}  {img.shape[1]}x{img.shape[0]}")
    print(f"geometria: {t}")
    print(f"degrado  : {d}")
    print("H_true (A -> B):")
    print(H)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, np.hstack([img, b]))
    print(f"figura: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
