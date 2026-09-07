"""Interfaccia comune dei matcher. Garantisce che classico e neurale girino
nella STESSA pipeline : l'unica variabile è --matcher."""
from __future__ import annotations
from typing import Protocol
import numpy as np


class Matcher(Protocol):
    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Ritorna (punti_a Nx2, punti_b Nx2, metadati). """
        ...


# `prepara()` è un metodo OPZIONALE, fuori dal Protocol perché non tutti i
# matcher hanno qualcosa da preparare: SIFT e ORB sono pronti appena costruiti.
# Serve a chi paga un costo di inizializzazione una volta sola — LoFTR carica un
# checkpoint da 90 MB — e permette alla pipeline di pagarlo fuori dal cronometro
# del matching, che altrimenti misurerebbe il caricamento invece dell'inferenza.
# La pipeline lo cerca con getattr: un matcher che non ce l'ha non deve
# implementare un metodo vuoto per rispettare l'interfaccia.
