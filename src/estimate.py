"""Stima della trasformazione da corrispondenze, con RANSAC (§7.3).

Tre modelli, dal più vincolato al più libero: similarità (4 gdl), affine (6),
omografia (8). Più gradi di libertà significano più capacità di adattarsi ai
dati — anche a corrispondenze sbagliate: il confronto fra modelli è uno degli
assi sperimentali di E2, non un dettaglio implementativo.

Determinismo (I9): `cv2.setRNGSeed` prima di ogni chiamata a RANSAC. Stesso
input e stessi parametri = stessa H, cifra per cifra.

Fallimento pulito (§7.3): sotto i 4 match si ritorna `success=False`, non
un'eccezione. La riga del CSV va scritta comunque — un fallimento è un dato (I7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

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
