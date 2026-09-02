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
from src.preprocess import applica, produce_binaria


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
    pipeline, non è un ritocco di una delle due immagini.
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
    st = stima_ransac(
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
