"""Orchestrazione: due immagini entrano, una H stimata esce.

    preprocess -> matcher -> corrispondenze -> RANSAC -> H_est

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

import numpy as np

from src.estimate import Stima
from src.estimate import stima as stima_ransac
from src.matchers.classic import crea_matcher
from src.preprocess import applica


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


def _costruisci_matcher(opz: Opzioni):
    """Ogni matcher riceve i suoi parametri e nessun altro.

    Il ratio test è di SIFT: passarlo a ORB sarebbe un parametro finto, che
    finirebbe nel CSV come se avesse avuto un effetto.
    """
    if opz.matcher == "sift":
        return crea_matcher("sift", ratio=opz.ratio)
    if opz.matcher == "loftr":
        return crea_matcher("loftr", soglia_conf=opz.loftr_conf, max_lato=opz.loftr_max_lato)
    return crea_matcher(opz.matcher)


def registra(img_hist: np.ndarray, img_modern: np.ndarray, opz: Opzioni | None = None) -> Risultato:
    """Registra `img_hist` su `img_modern`. Ritorna H_est e i metadati.

    Il preprocessing si applica a entrambe allo stesso modo: fa parte della
    pipeline, non è un ritocco di una delle due immagini.
    """
    opz = opz or Opzioni()

    t0 = time.perf_counter()
    a = applica(img_hist, modo=opz.preprocess, morph_open=opz.morph_open, morph_close=opz.morph_close)
    b = applica(img_modern, modo=opz.preprocess, morph_open=opz.morph_open, morph_close=opz.morph_close)
    t_prep = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pts_a, pts_b, meta_match = _costruisci_matcher(opz).match(a, b)
    t_match = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    st = stima_ransac(
        pts_a,
        pts_b,
        modello=opz.model,
        ransac_thresh=opz.ransac_thresh,
        seed=opz.seed,
    )
    t_stima = (time.perf_counter() - t0) * 1000

    return Risultato(
        stima=st,
        pts_hist=pts_a,
        pts_modern=pts_b,
        meta=meta_match
        | {
            "preprocess": opz.preprocess,
            "morph_open": opz.morph_open,
            "morph_close": opz.morph_close,
            "modello": opz.model,
            "seed": opz.seed,
            "t_prep_ms": round(t_prep, 1),
            "t_match_ms": round(t_match, 1),
            "t_stima_ms": round(t_stima, 1),
        },
    )
