"""Pulizia del segnale: grayscale, denoise, binarizzazione, morfologia (§7.1).

È il cuore dell'approccio A — "massimizzare l'efficacia dei metodi classici
tramite una pulizia ottimale del segnale". Ogni stadio è attivabile da CLI
perché l'ablation è un risultato, non un parametro da fissare una volta.

**Convenzione di polarità.** L'ingresso ha inchiostro scuro su carta chiara, e
l'uscita mantiene la stessa polarità: le funzioni ritornano sempre un'immagine
in cui l'inchiostro è scuro. Internamente la morfologia lavora sulla maschera
con l'inchiostro in primo piano, altrimenti "apertura" e "chiusura" farebbero
l'opposto di quello che dice il loro nome: `apri` toglie il pepe dal tratto,
`chiudi` salda le interruzioni.

**La domanda sperimentale** (§7.1): SIFT lavora su grayscale, non su binaria.
Binarizzare potrebbe distruggere proprio la texture su cui SIFT si basa. Per
questo CLAHE è un'alternativa alla binarizzazione, non un suo preludio, e le
due strade vanno misurate entrambe.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

MODI = ("none", "clahe", "otsu", "sauvola")
DENOISE = ("none", "mediana", "bilaterale")

# Modi che producono un'immagine binaria. Solo con questi la morfologia e la
# rimozione delle componenti hanno un effetto: su un grayscale `applica` le
# ignora. Sta qui, in un posto solo, perché serve anche a chi registra i
# parametri nel CSV — vedi pipeline.registra.
MODI_BINARI = ("otsu", "sauvola")


def produce_binaria(modo: str) -> bool:
    """Vero se `applica(..., modo=modo)` ritorna un'immagine binaria."""
    return modo in MODI_BINARI


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def denoise(gray: np.ndarray, metodo: str = "none", forza: int = 3) -> np.ndarray:
    """Mediana (via il sale e pepe) o bilaterale (liscia preservando i bordi)."""
    if metodo == "none":
        return gray
    if metodo == "mediana":
        return cv2.medianBlur(gray, forza if forza % 2 else forza + 1)
    if metodo == "bilaterale":
        return cv2.bilateralFilter(gray, d=forza * 2 + 1, sigmaColor=50, sigmaSpace=50)
    raise ValueError(f"denoise sconosciuto: {metodo!r} (disponibili: {DENOISE})")


def clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Equalizzazione adattiva del contrasto. Resta grayscale: non binarizza."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def otsu(gray: np.ndarray) -> np.ndarray:
    """Soglia globale di Otsu — la baseline.

    Su carta ingiallita con gradiente di illuminazione una soglia sola non può
    funzionare bene: è il risultato atteso (§7.1), non un difetto da aggiustare.
    """
    _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria


def sauvola(
    gray: np.ndarray, finestra: int = 25, k: float = 0.2, R: float = 128.0
) -> np.ndarray:
    """Soglia locale di Sauvola: T = m·[1 + k·(s/R − 1)] (§7.1).

    Calcolata con immagini integrali: media e varianza di ogni finestra costano
    O(1) per pixel invece di O(finestra²), quindi la dimensione della finestra
    non incide sul tempo.

    Dove la carta è chiara e uniforme s è piccolo, T scende sotto m e il rumore
    di fondo non passa; dove c'è tratto s cresce e T si alza verso m. È questo
    che la rende robusta al gradiente di illuminazione, che Otsu non vede.
    """
    if finestra % 2 == 0:
        finestra += 1
    r = finestra // 2
    g = gray.astype(np.float64)

    pad = cv2.copyMakeBorder(g, r, r, r, r, cv2.BORDER_REFLECT_101)
    somma, somma2 = cv2.integral2(pad)

    h, w = g.shape
    f = finestra

    def _box(S: np.ndarray) -> np.ndarray:
        return S[f : f + h, f : f + w] - S[0:h, f : f + w] - S[f : f + h, 0:w] + S[0:h, 0:w]

    area = float(f * f)
    media = _box(somma) / area
    varianza = np.maximum(_box(somma2) / area - media**2, 0.0)
    scarto = np.sqrt(varianza)

    soglia = media * (1.0 + k * (scarto / R - 1.0))
    # inchiostro = più scuro della soglia locale; l'uscita resta scura su chiaro
    return np.where(g < soglia, 0, 255).astype(np.uint8)


def _maschera_inchiostro(binaria: np.ndarray) -> np.ndarray:
    """Da 'inchiostro scuro su chiaro' a 'inchiostro = 255', per la morfologia."""
    return cv2.bitwise_not(binaria)


def morfologia(binaria: np.ndarray, apri: int = 0, chiudi: int = 0, se: int = 3) -> np.ndarray:
    """Apertura e chiusura sull'inchiostro (§7.1).

    `apri` toglie il pepe: componenti più sottili dell'elemento strutturante
    spariscono. `chiudi` salda le interruzioni del tratto, che sull'impianto
    sono frequenti perché il pennino stacca.

    L'ordine è apertura poi chiusura: chiudere per primo salderebbe anche il
    pepe al tratto, rendendolo non più rimovibile.
    """
    if apri <= 0 and chiudi <= 0:
        return binaria
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (se, se))
    ink = _maschera_inchiostro(binaria)
    if apri > 0:
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=apri)
    if chiudi > 0:
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=chiudi)
    return cv2.bitwise_not(ink)


def rimuovi_componenti(binaria: np.ndarray, area_min: int = 0) -> np.ndarray:
    """Elimina le componenti connesse di inchiostro più piccole di `area_min`.

    Le cifre scritte a mano e i puntini di grana sono rumore per il matching
    strutturale (§7.1). Attenzione però: sono anche i blob su cui SIFT trova
    keypoint, quindi toglierli non è gratis — va misurato, non assunto.
    """
    if area_min <= 0:
        return binaria
    ink = _maschera_inchiostro(binaria)
    n, etichette, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    tieni = np.zeros(n, dtype=bool)
    tieni[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_min  # 0 = sfondo
    return cv2.bitwise_not(np.where(tieni[etichette], 255, 0).astype(np.uint8))


def percentuale_inchiostro(binaria: np.ndarray) -> float:
    """Frazione di pixel di inchiostro. Serve a confrontare le binarizzazioni."""
    return float((binaria < 128).mean())


def applica(
    img: np.ndarray,
    modo: str = "sauvola",
    denoise_metodo: str = "none",
    morph_open: int = 0,
    morph_close: int = 0,
    area_min: int = 0,
    clip: float = 2.0,
    finestra: int = 25,
    k: float = 0.2,
) -> np.ndarray:
    """Pipeline di preprocessing, nell'ordine dei flag CLI di §9.

    Morfologia e rimozione componenti hanno senso solo su un'immagine binaria:
    con `--preprocess none` o `clahe` vengono ignorate, e la cosa va detta
    invece di applicarle a un grayscale ottenendo qualcosa di privo di senso.
    """
    if modo not in MODI:
        raise ValueError(f"preprocess sconosciuto: {modo!r} (disponibili: {MODI})")

    gray = denoise(to_gray(img), denoise_metodo)
    if modo == "none":
        return gray
    if modo == "clahe":
        return clahe(gray, clip=clip)

    binaria = otsu(gray) if modo == "otsu" else sauvola(gray, finestra=finestra, k=k)
    binaria = morfologia(binaria, apri=morph_open, chiudi=morph_close)
    return rimuovi_componenti(binaria, area_min)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocessing di un crop (§7.1)")
    ap.add_argument("--crop", default="data/crops/ribba.png")
    ap.add_argument("--preprocess", dest="modo", default="sauvola", choices=MODI)
    ap.add_argument("--denoise", default="none", choices=DENOISE)
    ap.add_argument("--morph-open", type=int, default=0)
    ap.add_argument("--morph-close", type=int, default=0)
    ap.add_argument("--area-min", type=int, default=0)
    ap.add_argument("--finestra", type=int, default=25, help="finestra di Sauvola")
    ap.add_argument("--k", type=float, default=0.2, help="k di Sauvola")
    ap.add_argument("--out", default="results/figures/preprocess.png")
    args = ap.parse_args(argv)

    img = cv2.imread(args.crop, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.crop)
    out = applica(
        img,
        modo=args.modo,
        denoise_metodo=args.denoise,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        area_min=args.area_min,
        finestra=args.finestra,
        k=args.k,
    )
    if produce_binaria(args.modo):
        print(f"inchiostro: {percentuale_inchiostro(out) * 100:.2f}% dei pixel")
    elif args.morph_open or args.morph_close or args.area_min:
        print("nota: morfologia e rimozione componenti ignorate, l'uscita non è binaria")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"figura: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
