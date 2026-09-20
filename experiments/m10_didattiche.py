"""M10 — le quattro figure didattiche del capitolo 2/7 della relazione.

Non sono un esperimento: non producono righe di CSV né toccano la griglia
sperimentale. Sono lo stesso identico artefatto di milestone (CLAUDE.md §10
ha una riga sola per M10) che `scripts/riproduci.py` invoca come un'unica
fase, `didattiche` — qui vivono in un file solo invece che in quattro,
perché quattro `import matplotlib; matplotlib.use("Agg")` e una quinta copia
della conversione a grayscale (§7.1) non aggiungevano nulla alla milestone.

  - **passo-passo**: una registrazione seguita dall'inizio alla fine, sei
    pannelli — la scansione storica, la pulizia, il vettoriale disegnato, gli
    abbinamenti prima e dopo RANSAC, il risultato finale.
  - **ratio-test**: il ratio test di Lowe illustrato con i rapporti d1/d2
    veri, stesso dominio contro cross-domain.
  - **gradiente**: luminosità, gradiente e descrittore su un dettaglio reale
    — chiarisce la differenza fra un numero per pixel, un vettore per pixel e
    un istogramma per l'intera finestra.
  - **RANSAC-iterazioni**: quante iterazioni servono in funzione dell'inlier
    ratio, con i valori realmente osservati in E1 ed E2 segnati sopra.

    python -m experiments.m10_didattiche --crop ribba
"""
from __future__ import annotations

import argparse
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluate import valuta  # noqa: E402
from src.figure import sovrapponi_tratti  # noqa: E402
from src.groundtruth import riferimento_da_jgw  # noqa: E402
from src.pipeline import Opzioni, registra  # noqa: E402
from src.prep.synth import Trasformazione, genera_coppia  # noqa: E402
from src.preprocess import applica, to_gray  # noqa: E402

# --------------------------------------------------------------- passo-passo

# La stessa soglia di successo della griglia E2 (experiments/m8_e2_griglia.py):
# la figura descrive una riga di quell'esperimento, e dichiarare due soglie
# diverse per la stessa prova la renderebbe incomparabile.
SOGLIA_M = 2.0


def _affianca(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Mette due immagini una accanto all'altra, centrate in verticale.

    Ritorna la tela, l'offset orizzontale della seconda immagine e i due
    scostamenti verticali. Centrare serve perché le due immagini hanno altezze
    diverse — 1024 px lo storico contro 1504 il vettoriale (§9.1) — e
    appoggiarle entrambe in alto lascerebbe 480 righe vuote tutte sotto la più
    corta, cioè un terzo del pannello bianco da un lato solo.
    """
    a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else a.copy()
    b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR) if b.ndim == 2 else b.copy()
    ha, hb = a.shape[0], b.shape[0]
    h = max(ha, hb)
    tela = np.full((h, a.shape[1] + b.shape[1], 3), 255, np.uint8)
    dy_a, dy_b = (h - ha) // 2, (h - hb) // 2
    tela[dy_a : dy_a + ha, : a.shape[1]] = a
    tela[dy_b : dy_b + hb, a.shape[1] :] = b
    return tela, a.shape[1], dy_a, dy_b


def _disegna_linee(
    tela, offset, pts_a, pts_b, colore, spessore=1, quanti=60, seed=42, dy_a=0, dy_b=0
):
    """Traccia le linee di corrispondenza fra i due lati, su un campione casuale.

    `dy_a` e `dy_b` sono gli scostamenti verticali che `_affianca` ha applicato
    centrando le due immagini: senza, le linee partirebbero dove i punti
    *stavano* e non dove si vedono.
    """
    if len(pts_a) == 0:
        return tela
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts_a), size=min(quanti, len(pts_a)), replace=False)
    for i in idx:
        xa, ya = pts_a[i]
        xb, yb = pts_b[i]
        cv2.line(
            tela,
            (int(round(xa)), int(round(ya)) + dy_a),
            (int(round(xb + offset)), int(round(yb)) + dy_b),
            colore,
            spessore,
            cv2.LINE_AA,
        )
    return tela


def _riquadro_numeri(ax, ris, metriche, riuscita: bool, tinta: str) -> None:
    """Il risultato in cifre, accanto al pannello 6.

    La figura fa vedere che i tratti si sovrappongono; questo riquadro dice di
    quanto sbaglia, che è il numero su cui si giudica l'intera pipeline (§4).
    Sta qui e non solo nel titolo generale perché il titolo si legge una volta,
    mentre questo resta accanto all'immagine che descrive.
    """
    ax.axis("off")
    ax.text(
        0.5, 0.97, "registrazione riuscita" if riuscita else "registrazione fallita",
        ha="center", va="top", fontsize=15, weight="bold", color=tinta,
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.82, f"{metriche['rmse_m']:.2f} m",
        ha="center", va="top", fontsize=40, weight="bold", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.66,
        "errore finale (RMSE) su un riferimento\nche di suo ne vale circa 0.5",
        ha="center", va="top", fontsize=10, transform=ax.transAxes,
    )
    righe = (
        ("corrispondenze trovate", f"{ris.stima.n_matches}"),
        ("inlier dopo RANSAC", f"{ris.stima.n_inliers}  ({100 * ris.stima.inlier_ratio:.1f}%)"),
        ("errore in pixel", f"{metriche['rmse_px']:.2f} px"),
        ("errore massimo", f"{metriche['err_max_px']:.2f} px"),
        ("soglia dichiarata", f"{SOGLIA_M:.2f} m"),
    )
    for i, (etichetta, valore) in enumerate(righe):
        y = 0.48 - i * 0.09
        ax.text(0.05, y, etichetta, fontsize=10, transform=ax.transAxes)
        ax.text(0.95, y, valore, fontsize=10, weight="bold", ha="right", transform=ax.transAxes)
    ax.add_patch(
        plt.Rectangle(
            (0.01, 0.01), 0.98, 0.99, transform=ax.transAxes,
            fill=False, linewidth=1.2, edgecolor="#999999",
        )
    )


def _pannello_immagine(ax, img, titolo, didascalia, cornice=None, grande=False):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap="gray")
    ax.set_title(titolo, fontsize=14 if grande else 11, weight="bold")
    ax.set_xlabel(didascalia, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if cornice is not None:
        for lato in ax.spines.values():
            lato.set_visible(True)
            lato.set_linewidth(4)
            lato.set_edgecolor(cornice)


def figura_passo_passo(crop: str, crops_dir: str, out_path: str, opz: Opzioni) -> dict:
    hist = cv2.imread(os.path.join(crops_dir, f"{crop}.png"), cv2.IMREAD_COLOR)
    vec = cv2.imread(os.path.join(crops_dir, f"{crop}_vec1812.png"), cv2.IMREAD_COLOR)
    if hist is None or vec is None:
        raise FileNotFoundError(f"{crop}: mancano il ritaglio o il raster vettoriale (M1/M7)")

    ris = registra(hist, vec, opz)
    H_true, W_dest = riferimento_da_jgw(
        os.path.join(crops_dir, f"{crop}.jgw"), os.path.join(crops_dir, f"{crop}_vec1812.jgw")
    )
    metriche = valuta(
        ris.stima,
        H_true,
        hist.shape[1],
        hist.shape[0],
        W_dest=W_dest,
        soglia_m=SOGLIA_M,
    )

    hist_pulito = applica(hist, modo=opz.preprocess, morph_close=opz.morph_close)
    vec_pulito = applica(vec, modo=opz.preprocess, morph_close=opz.morph_close)

    # 4 e 5: gli abbinamenti, prima e dopo la votazione
    tela_tutti, off, dy_a, dy_b = _affianca(hist_pulito, vec_pulito)
    _disegna_linee(
        tela_tutti, off, ris.pts_hist, ris.pts_modern, (0, 0, 220), dy_a=dy_a, dy_b=dy_b
    )

    inlier = (
        np.flatnonzero(ris.stima.inliers) if ris.stima.inliers is not None else np.array([], int)
    )
    tela_inlier, off2, dy_a2, dy_b2 = _affianca(hist_pulito, vec_pulito)
    _disegna_linee(
        tela_inlier, off2, ris.pts_hist[inlier], ris.pts_modern[inlier], (0, 160, 0),
        spessore=2, dy_a=dy_a2, dy_b=dy_b2,
    )

    # 6: la sovrapposizione
    h, w = vec.shape[:2]
    warp = cv2.warpPerspective(hist, ris.H, (w, h), borderValue=(255, 255, 255))
    overlay = sovrapponi_tratti(warp, vec)

    # Sei colonne perché la griglia deve dividersi sia in tre (la prima riga) sia
    # in due (la seconda), e matplotlib non mescola conteggi diversi di colonne.
    #
    # I pannelli 4 e 5 sono la stessa coppia di immagini prima e dopo la
    # votazione, e confrontarli è il punto: stanno **affiancati**, così il
    # confronto è un movimento dell'occhio e non una risalita oltre una
    # didascalia. Affiancati prendono anche metà larghezza ciascuno invece di un
    # terzo, quindi si ingrandiscono. L'ultima riga è la più alta e ospita il
    # risultato, che è ciò che il lettore deve portarsi via.
    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.9, 1.7])

    _pannello_immagine(
        fig.add_subplot(gs[0, 0:2]),
        hist,
        "1 · la scansione storica",
        "il foglio disegnato a mano,\ncosì come arriva dall'archivio",
    )
    _pannello_immagine(
        fig.add_subplot(gs[0, 2:4]),
        hist_pulito,
        f"2 · la stessa, ripulita ({opz.preprocess})",
        "ogni pixel è deciso: inchiostro o carta.\nLa soglia è calcolata zona per zona",
    )
    _pannello_immagine(
        fig.add_subplot(gs[0, 4:6]),
        vec_pulito,
        "3 · il vettoriale, disegnato",
        "le coordinate del CXF tracciate su\nun'immagine: ora sono confrontabili",
    )

    _pannello_immagine(
        fig.add_subplot(gs[1, 0:3]),
        tela_tutti,
        f"4 · gli abbinamenti trovati ({ris.stima.n_matches} in tutto, ne sono disegnati 60 a caso)",
        "ogni linea unisce due punti che il programma ritiene la stessa cosa.\n"
        "Se fossero giusti sarebbero tutti paralleli: "
        f"solo il {100 * ris.stima.inlier_ratio:.0f}% lo è",
    )
    _pannello_immagine(
        fig.add_subplot(gs[1, 3:6]),
        tela_inlier,
        f"5 · dopo la votazione ({ris.stima.n_inliers} sopravvissuti)",
        "RANSAC tiene solo gli abbinamenti che concordano su una stessa\n"
        "trasformazione. Il confronto col pannello a sinistra è il punto",
    )

    riuscita = bool(metriche["success"])
    tinta = "#1b7a3d" if riuscita else "#a62828"
    _pannello_immagine(
        fig.add_subplot(gs[2, 0:4]),
        overlay,
        "6 · il risultato",
        "nero dove i due tratti coincidono; rosso il solo storico deformato, "
        "ciano il solo vettoriale",
        cornice=tinta,
        grande=True,
    )

    # L'immagine del risultato è quadrata e occupa due colonne: la terza
    # resterebbe vuota. Ci va il numero, che del risultato è la parte che si
    # cita — la figura mostra *che* ha funzionato, il riquadro dice *quanto* (I6).
    _riquadro_numeri(fig.add_subplot(gs[2, 4:6]), ris, metriche, riuscita, tinta)

    fig.suptitle(
        f"Una registrazione dall'inizio alla fine — ritaglio «{crop}», "
        f"{opz.matcher.upper()} · {opz.preprocess} · {opz.model}",
        fontsize=14,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # rect lascia il posto al suptitle: senza, tight_layout gli fa sovrascrivere
    # i titoli della prima riga di pannelli
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"figura: {out_path}")
    print(
        f"  match {ris.stima.n_matches} · inlier {ris.stima.n_inliers} "
        f"({ris.stima.inlier_ratio:.3f}) · RMSE {metriche['rmse_m']:.3f} m"
    )
    return metriche


# ---------------------------------------------------------------- ratio-test

SOGLIA = 0.75


def _rapporti(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, int, int]:
    """d1/d2 per ogni keypoint di A che ha almeno due candidati in B."""
    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(to_gray(img_a), None)
    kp_b, des_b = sift.detectAndCompute(to_gray(img_b), None)
    if des_a is None or des_b is None or len(kp_b) < 2:
        return np.array([]), len(kp_a), len(kp_b)
    coppie = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    rapporti = np.array([m.distance / n.distance for m, n in coppie if n.distance > 0])
    return rapporti, len(kp_a), len(kp_b)


def _pannello_istogramma_ratio(ax, rapporti: np.ndarray, titolo: str) -> None:
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


def figura_ratio_test(crop: str, crops_dir: str, out_path: str) -> None:
    hist = cv2.imread(os.path.join(crops_dir, f"{crop}.png"), cv2.IMREAD_COLOR)
    vec = cv2.imread(os.path.join(crops_dir, f"{crop}_vec1812.png"), cv2.IMREAD_COLOR)
    if hist is None or vec is None:
        raise FileNotFoundError(f"{crop}: mancano il ritaglio o il raster vettoriale (M1/M7)")

    ruotato, _ = genera_coppia(hist, Trasformazione(rot_deg=15, scala=1.15, tx=30, ty=20))
    r_e1, n_a1, n_b1 = _rapporti(hist, ruotato)
    r_e2, n_a2, n_b2 = _rapporti(hist, vec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _pannello_istogramma_ratio(ax1, r_e1, f"stesso dominio — {crop} contro sé stesso ruotato\n({n_a1} vs {n_b1} keypoint)")
    _pannello_istogramma_ratio(ax2, r_e2, f"cross-domain — {crop} contro il vettoriale\n({n_a2} vs {n_b2} keypoint)")
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


# ---------------------------------------------------------------- gradiente


def figura_gradiente(crop: str, crops_dir: str, out_path: str, riquadro=(430, 300, 140, 140)) -> None:
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


# ------------------------------------------------------------ RANSAC-iterazioni

# Gli stessi valori di default di src/pipeline.py::estimate (confidence,
# max_iter): non duplicati per caso, sono il contratto reale usato da tutte
# le griglie.
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


def figura_ransac_iterazioni(out_path: str, csv_path: str) -> None:
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


# --------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="ribba")
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--matcher", default="orb", help="solo per la figura passo-passo")
    ap.add_argument("--preprocess", default="sauvola", help="solo per la figura passo-passo")
    ap.add_argument("--model", default="similarity", help="solo per la figura passo-passo")
    ap.add_argument("--csv", default="results/runs.csv", help="solo per la figura RANSAC-iterazioni")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args(argv)

    figura_passo_passo(
        args.crop,
        args.crops_dir,
        os.path.join(args.out_dir, "m10_passo_passo.png"),
        Opzioni(matcher=args.matcher, preprocess=args.preprocess, model=args.model),
    )
    figura_ratio_test(args.crop, args.crops_dir, os.path.join(args.out_dir, "m10_ratio_test.png"))
    figura_gradiente(args.crop, args.crops_dir, os.path.join(args.out_dir, "m10_gradiente.png"))
    figura_ransac_iterazioni(os.path.join(args.out_dir, "m10_ransac_iterazioni.png"), args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
