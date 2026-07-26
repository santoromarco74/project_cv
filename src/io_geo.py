"""Lettura dei dati georeferenziati: world file (JGW), metadati (TXT), vettoriale (CXF).

Nessuna dipendenza geospaziale: raster e vettoriale sono già nello stesso
sistema (Cassini-Soldner Forte Diamante), non serve riproiettare. Vedi CLAUDE.md §5.
"""
from __future__ import annotations
import numpy as np


def read_jgw(path: str) -> np.ndarray:
    """Legge un world file a 6 righe e ritorna l'affine pixel->CRS come matrice 3x3.

    Righe JGW: A, D, B, E, C, F  ->  X = A*col + B*row + C ; Y = D*col + E*row + F
    """
    raise NotImplementedError  # M1


def parse_cxf(path: str):
    """Parser del formato CXF (testuale AdE). Ritorna la lista dei record BORDO.

    ATTENZIONE (CLAUDE.md §5.4): dopo i due punti di etichetta c'è <nflag>,
    numero di indici extra che PRECEDONO il blocco coordinate. Ignorarlo sfasa
    il parser su 33 record su 871 e fa esplodere l'estensione. Dopo il parsing,
    ASSERIRE che ogni coordinata cada dentro l'estensione del JGW.

    Codici: 18 = particella, 12 = acqua/strada, 25 = altro (da scartare).
    """
    raise NotImplementedError  # M2
