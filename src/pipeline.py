"""Stima della trasformazione (RANSAC, §7.3) e orchestrazione della pipeline.

    preprocess -> matcher -> corrispondenze -> RANSAC -> H_est

I due tempi stavano in `estimate.py` e `pipeline.py` separati; sono confluiti
qui perché sono entrambi "lato algoritmo" secondo I3 (vedi sotto) e nessuno
dei due importa mai `groundtruth`/`io_geo` — accorparli non sposta quel
confine, che resta fra questo file ed `evaluate.py`.

Tre modelli di trasformazione, dal più vincolato al più libero: similarità
(4 gdl), affine (6), omografia (8). Più gradi di libertà significano più
capacità di adattarsi ai dati — anche a corrispondenze sbagliate: il
confronto fra modelli è uno degli assi sperimentali di E2, non un dettaglio
implementativo.

Determinismo (I9): `cv2.setRNGSeed` prima di ogni chiamata a RANSAC. Stesso
input e stessi parametri = stessa H, cifra per cifra.

Fallimento pulito (§7.3): sotto i 4 match si ritorna `success=False`, non
un'eccezione. La riga del CSV va scritta comunque — un fallimento è un dato (I7).

⚠ I3 — questo modulo NON conosce il world file. Riceve due array di pixel e
non sa da dove vengano né dove siano nel mondo. È il motivo per cui gli import
qui sopra non includono `io_geo` né `groundtruth`, ed è verificato da un test
statico (`test_i3_groundtruth_fuori_dalla_pipeline`). Se un pixel di
georeferenziazione filtrasse nel matching, il risultato sarebbe privo di
significato.

⚠ I4 — non sa nemmeno quale matcher sta usando: arriva dalla fabbrica per nome.
È ciò che rende confrontabili classico e neurale, cambiando il solo `--matcher`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.matchers.classic import crea_matcher
from src.preprocess import applica, produce_binaria

# --------------------------------------------------------------------- stima

MODELLI = ("similarity", "affine", "homography")

# Sotto questa soglia non si stima nulla. È il minimo dell'omografia (4 punti);
# similarità e affine ne richiederebbero meno, ma con 2 o 3 corrispondenze RANSAC
# non ha nulla da votare e il risultato non sarebbe una stima.
MIN_MATCH = 4


@dataclass
class Stima:
    """Esito di una stima. `H` è None se non è riuscita."""

    success: bool
    H: np.ndarray | None = None
    inliers: np.ndarray | None = None  # maschera booleana (N,)
    n_matches: int = 0
    n_inliers: int = 0
    modello: str = ""
    motivo: str = ""  # perché è fallita, quando è fallita
    meta: dict = field(default_factory=dict)

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


def _a_3x3(M: np.ndarray) -> np.ndarray:
    """Porta una 2x3 (affine) in coordinate omogenee."""
    return np.vstack([M, [0.0, 0.0, 1.0]])


def stima(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    modello: str = "homography",
    ransac_thresh: float = 3.0,
    max_iter: int = 5000,
    confidence: float = 0.995,
    seed: int = 42,
) -> Stima:
    """Stima la trasformazione che porta pts_a su pts_b.

    `ransac_thresh` è in pixel: la distanza di riproiezione sotto la quale una
    corrispondenza è considerata inlier.
    """
    if modello not in MODELLI:
        raise ValueError(f"modello sconosciuto: {modello!r} (disponibili: {MODELLI})")

    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    n = len(pts_a)
    if n != len(pts_b):
        raise ValueError(f"corrispondenze sbilanciate: {n} contro {len(pts_b)}")
    if n < MIN_MATCH:
        return Stima(
            success=False,
            n_matches=n,
            modello=modello,
            motivo=f"solo {n} corrispondenze, ne servono almeno {MIN_MATCH}",
        )

    cv2.setRNGSeed(seed)  # I9
    a = pts_a.reshape(-1, 1, 2).astype(np.float32)
    b = pts_b.reshape(-1, 1, 2).astype(np.float32)

    if modello == "similarity":
        M, mask = cv2.estimateAffinePartial2D(
            a,
            b,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iter,
            confidence=confidence,
        )
        H = None if M is None else _a_3x3(M)
    elif modello == "affine":
        M, mask = cv2.estimateAffine2D(
            a,
            b,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iter,
            confidence=confidence,
        )
        H = None if M is None else _a_3x3(M)
    else:
        H, mask = cv2.findHomography(
            a,
            b,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iter,
            confidence=confidence,
        )

    if H is None:
        return Stima(
            success=False,
            n_matches=n,
            modello=modello,
            motivo="RANSAC non ha trovato un modello",
        )

    inliers = np.zeros(n, dtype=bool) if mask is None else mask.ravel().astype(bool)
    return Stima(
        success=True,
        H=np.asarray(H, dtype=np.float64),
        inliers=inliers,
        n_matches=n,
        n_inliers=int(inliers.sum()),
        modello=modello,
        meta={"ransac_thresh": ransac_thresh, "seed": seed},
    )


# --------------------------------------------------------------- orchestrazione


@dataclass(frozen=True)
class Opzioni:
    """I parametri del contratto CLI (§9) che riguardano l'algoritmo."""

    matcher: str = "sift"
    preprocess: str = "sauvola"
    morph_open: int = 0
    morph_close: int = 0
    model: str = "homography"
    ratio: float = 0.75
    ransac_thresh: float = 3.0
    seed: int = 42
    # Parametri specifici di LoFTR. Stanno qui, e non dentro il matcher, per lo
    # stesso motivo per cui ci sta `ratio`: sono variabili sperimentali, e una
    # variabile sperimentale che non passa dalle opzioni non finisce nel CSV.
    loftr_conf: float = 0.5
    loftr_max_lato: int = 640


@dataclass
class Risultato:
    """H stimata e tutto ciò che serve a scriverne una riga di CSV."""

    stima: Stima
    pts_hist: np.ndarray
    pts_modern: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def H(self) -> np.ndarray | None:
        return self.stima.H

    @property
    def success(self) -> bool:
        return self.stima.success


# Matcher già costruiti, riusati fra una registrazione e l'altra.
#
# `LoftrMatcher` tiene il modello in `self._modello`, ma finché l'istanza veniva
# creata e buttata a ogni chiamata quella cache non serviva a niente: ogni
# registrazione rifaceva `torch.load` del checkpoint da 90 MB, il
# `load_state_dict` e il trasferimento sul device. In E3 sono 170 registrazioni,
# quindi 170 caricamenti.
#
# Il riuso non cambia i risultati (I9): nessuno dei tre matcher porta stato da
# una chiamata all'altra — gli oggetti di OpenCV sono riusabili per costruzione,
# LoFTR gira in `eval()` dentro `inference_mode`. La chiave tiene separati
# matcher con parametri diversi, che restano istanze distinte.
_CACHE_MATCHER: dict[tuple, object] = {}


def _chiave_matcher(opz: Opzioni) -> tuple:
    """I parametri che definiscono un matcher: due opzioni diverse, due istanze."""
    if opz.matcher == "sift":
        return ("sift", opz.ratio)
    if opz.matcher == "loftr":
        return ("loftr", opz.loftr_conf, opz.loftr_max_lato)
    return (opz.matcher,)


def _costruisci_matcher(opz: Opzioni):
    """Ogni matcher riceve i suoi parametri e nessun altro.

    Il ratio test è di SIFT: passarlo a ORB sarebbe un parametro finto, che
    finirebbe nel CSV come se avesse avuto un effetto.
    """
    chiave = _chiave_matcher(opz)
    if chiave not in _CACHE_MATCHER:
        if opz.matcher == "sift":
            costruito = crea_matcher("sift", ratio=opz.ratio)
        elif opz.matcher == "loftr":
            costruito = crea_matcher(
                "loftr", soglia_conf=opz.loftr_conf, max_lato=opz.loftr_max_lato
            )
        else:
            costruito = crea_matcher(opz.matcher)
        _CACHE_MATCHER[chiave] = costruito
    return _CACHE_MATCHER[chiave]


def svuota_cache_matcher() -> None:
    """Libera i matcher in cache. Utile nei test e per rilasciare il modello LoFTR."""
    _CACHE_MATCHER.clear()


def registra(img_hist: np.ndarray, img_modern: np.ndarray, opz: Opzioni | None = None) -> Risultato:
    """Registra `img_hist` su `img_modern`. Ritorna H_est e i metadati.

    Il preprocessing si applica a entrambe allo stesso modo: fa parte della
    pipeline, non un ritocco di una delle due immagini.
    """
    opz = opz or Opzioni()

    t0 = time.perf_counter()
    a = applica(img_hist, modo=opz.preprocess, morph_open=opz.morph_open, morph_close=opz.morph_close)
    b = applica(img_modern, modo=opz.preprocess, morph_open=opz.morph_open, morph_close=opz.morph_close)
    t_prep = (time.perf_counter() - t0) * 1000

    # Costruzione e preparazione stanno FUORI dal cronometro del matching. Con
    # LoFTR, includerle significava contare il caricamento del checkpoint dentro
    # il tempo di inferenza — e quel tempo è uno dei risultati del confronto
    # classico contro neurale (§10.1), non un dettaglio. Grazie alla cache il
    # costo si paga alla prima registrazione e poi `t_init_ms` va a zero.
    t0 = time.perf_counter()
    matcher = _costruisci_matcher(opz)
    prepara = getattr(matcher, "prepara", None)
    if prepara is not None:
        prepara()
    t_init = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pts_a, pts_b, meta_match = matcher.match(a, b)
    t_match = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    st = stima(
        pts_a,
        pts_b,
        modello=opz.model,
        ransac_thresh=opz.ransac_thresh,
        seed=opz.seed,
    )
    t_stima = (time.perf_counter() - t0) * 1000

    # Con `none` e `clahe` l'uscita del preprocessing non è binaria e `applica`
    # ignora la morfologia. Riportare i valori richiesti darebbe righe di CSV che
    # dichiarano una chiusura mai avvenuta: due configurazioni distinte in
    # tabella, con risultati identici per forza, e nessun modo di accorgersene
    # guardando il file. Si registra quello che è stato applicato.
    morfologia_attiva = produce_binaria(opz.preprocess)

    return Risultato(
        stima=st,
        pts_hist=pts_a,
        pts_modern=pts_b,
        meta=meta_match
        | {
            "preprocess": opz.preprocess,
            "morph_open": opz.morph_open if morfologia_attiva else 0,
            "morph_close": opz.morph_close if morfologia_attiva else 0,
            "morfologia_ignorata": not morfologia_attiva
            and bool(opz.morph_open or opz.morph_close),
            "modello": opz.model,
            "seed": opz.seed,
            "t_prep_ms": round(t_prep, 1),
            "t_init_ms": round(t_init, 1),
            "t_match_ms": round(t_match, 1),
            "t_stima_ms": round(t_stima, 1),
        },
    )
