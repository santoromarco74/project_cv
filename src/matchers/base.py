"""Interfaccia comune dei matcher. Garantisce che classico e neurale girino
nella STESSA pipeline (CLAUDE.md I4): l'unica variabile è --matcher."""
from __future__ import annotations
from typing import Protocol
import numpy as np


class Matcher(Protocol):
    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Ritorna (punti_a Nx2, punti_b Nx2, metadati). Vedi CLAUDE.md §7.2."""
        ...
