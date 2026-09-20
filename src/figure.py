"""Disegno delle corrispondenze sulle immagini.

Distinto da `report.py`, che produce grafici *dal CSV*: qui si disegna *sopra*
le immagini, e serve sia alla CLI (`--out-figure`) sia agli esperimenti. Vive
in un modulo a parte perché entrambi ne hanno bisogno e perché così non tira
dentro pandas, che la CLI non usa.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.preprocess import to_gray


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
    if len(pts) == 0:
        # `cv2.cvtColor` rifiuta un array vuoto. I chiamanti oggi hanno tutti
        # una guardia a monte, ma una funzione che colora le corrispondenze
        # deve poter rispondere "nessun colore" invece di sollevare.
        return []
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

    Due accorgimenti per lo scarto di altezza, che su E2 è di 480 px su 1504.
    La tela nasce **bianca**, non nera: azzerata lascerebbe un terzo della
    figura nero sotto il pannello più corto, e si leggerebbe come un difetto
    dell'immagine invece che come lo spazio vuoto che è. E l'immagine più corta
    è **centrata** in verticale invece di essere appoggiata in alto: appoggiata
    lascia tutto il vuoto da un lato solo, e il pannello sembra tagliato a
    metà.

    Si disegna a mano invece di usare `cv2.drawMatches` perché quella accetta
    un solo `matchColor` per tutta la figura, ed è proprio il vincolo da cui
    nasce il problema.
    """
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    altezza = max(ha, hb)
    vis = np.full((altezza, wa + wb, 3), 255, np.uint8)
    dy_a, dy_b = (altezza - ha) // 2, (altezza - hb) // 2
    vis[dy_a : dy_a + ha, :wa] = img_a
    vis[dy_b : dy_b + hb, wa:] = img_b

    # la tinta resta legata alla quota *nell'immagine di partenza*, non sulla
    # tela: è quella che rende la sequenza dei colori diagnostica
    for (xa, ya), (xb, yb), colore in zip(pts_a, pts_b, colori_per_quota(pts_a, ha)):
        p = (int(round(float(xa))), int(round(float(ya))) + dy_a)
        q = (int(round(float(xb))) + wa, int(round(float(yb))) + dy_b)
        cv2.line(vis, p, q, colore, 1, cv2.LINE_AA)
        cv2.circle(vis, p, 3, colore, -1, cv2.LINE_AA)
        cv2.circle(vis, q, 3, colore, -1, cv2.LINE_AA)
    return vis


def sovrapponi_tratti(a: np.ndarray, b: np.ndarray, soglia: int = 128) -> np.ndarray:
    """Due strati di tratto sovrapposti, con l'accordo e lo scarto distinguibili.

    Dipingere uno strato sopra l'altro con un colore pieno — `overlay[maschera]
    = rosso` — perde proprio l'informazione che la figura deve mostrare: il
    rosso *sostituisce* il pixel, quindi rosso su tratto nero (i due strati
    coincidono, il caso buono) e rosso su carta bianca (il tratto storico è
    finito dove il vettoriale non ha niente) diventano indistinguibili.

    Qui i due strati vanno invece su canali complementari, partendo dal bianco
    e spegnendo:

        solo `a`      → rosso
        solo `b`      → ciano
        entrambi      → nero
        nessuno       → bianco

    Così un disallineamento si vede come **frangia**: i due bordi si separano
    in un filo rosso e un filo ciano, e l'occhio lo cattura a colpo d'occhio
    anche per uno scarto di un paio di pixel. Una registrazione corretta invece
    annerisce, e la figura somiglia a una mappa pulita.

    La coppia rosso/ciano non è arbitraria: serve una partizione dei canali
    BGR, e delle tre possibili è la sola che resta leggibile su fondo bianco.
    Blu contro giallo sarebbe più sicura per chi confonde rosso e verde, ma il
    giallo su carta chiara sparisce; verde contro magenta è peggiore proprio
    sull'asse rosso-verde. Rosso e ciano conserva inoltre la convenzione già
    usata nel resto del progetto, dove il tratto storico è rosso.
    """
    ink_a = to_gray(a) < soglia
    ink_b = to_gray(b) < soglia
    if ink_a.shape != ink_b.shape:
        raise ValueError(f"strati di dimensioni diverse: {ink_a.shape} contro {ink_b.shape}")

    vis = np.full((*ink_a.shape, 3), 255, np.uint8)
    vis[..., 0][ink_a] = 0  # B  -> `a` da solo resta rosso
    vis[..., 1][ink_a] = 0  # G
    vis[..., 2][ink_b] = 0  # R  -> `b` da solo resta ciano
    return vis
