"""Disegno delle corrispondenze sulle immagini.

Distinto da `report.py`, che produce grafici *dal CSV*: qui si disegna *sopra*
le immagini, e serve sia alla CLI (`--out-figure`) sia agli esperimenti. Vive
in un modulo a parte perché entrambi ne hanno bisogno e perché così non tira
dentro pandas, che la CLI non usa.
"""
from __future__ import annotations

import cv2
import numpy as np


def colori_per_quota(pts: np.ndarray, altezza: int) -> list[tuple[int, int, int]]:
    """Un colore per corrispondenza, da una scala continua sulla quota del punto
    di partenza.

    Tutte le linee dello stesso colore sono illeggibili appena il fascio si
    incrocia, ma alternare tinte a caso distinguerebbe due linee vicine senza
    dire nient'altro. Legando la tinta alla posizione verticale nell'immagine di
    sinistra la figura diventa anche diagnostica: se gli agganci sono corretti i
    colori arrivano a destra nello stesso ordine, e una corrispondenza sbagliata
    si nota perché rompe la sequenza, non perché è di un colore qualsiasi.
    """
    quota = np.clip(pts[:, 1] / max(altezza - 1, 1), 0.0, 1.0)
    # La tinta si ferma a 150 dei 180 gradini di OpenCV: il giro completo
    # riporterebbe il fondo scala sul rosso da cui era partito. Il valore è
    # tenuto sotto il massimo perché il fondo è carta chiara: a piena
    # luminosità la fascia gialla sparisce contro il beige.
    hsv = np.stack(
        [
            (quota * 150).astype(np.uint8),
            np.full(len(quota), 235, np.uint8),
            np.full(len(quota), 195, np.uint8),
        ],
        axis=1,
    ).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return [tuple(int(c) for c in riga) for riga in bgr]


def affianca_corrispondenze(
    img_a: np.ndarray, img_b: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray
) -> np.ndarray:
    """Le due immagini affiancate, con una linea colorata per corrispondenza.

    Le due immagini possono avere dimensioni diverse: su E2 il raster
    vettoriale ha una griglia propria e un margine oltre il ritaglio (§9.1),
    quindi la tela si dimensiona sulla più alta delle due.

    Si disegna a mano invece di usare `cv2.drawMatches` perché quella accetta
    un solo `matchColor` per tutta la figura, ed è proprio il vincolo da cui
    nasce il problema.
    """
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    vis = np.zeros((max(ha, hb), wa + wb, 3), np.uint8)
    vis[:ha, :wa] = img_a
    vis[:hb, wa:] = img_b

    for (xa, ya), (xb, yb), colore in zip(pts_a, pts_b, colori_per_quota(pts_a, ha)):
        p = (int(round(float(xa))), int(round(float(ya))))
        q = (int(round(float(xb))) + wa, int(round(float(yb))))
        cv2.line(vis, p, q, colore, 1, cv2.LINE_AA)
        cv2.circle(vis, p, 3, colore, -1, cv2.LINE_AA)
        cv2.circle(vis, q, 3, colore, -1, cv2.LINE_AA)
    return vis
