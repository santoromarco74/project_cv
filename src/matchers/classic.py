"""SiftMatcher, OrbMatcher (OpenCV + ratio test di Lowe / cross-check).

Entrambi rispettano l'interfaccia `Matcher` (I4): la pipeline non sa quale sta
usando, e LoftrMatcher entrerà dalla stessa porta a M9.

Nota su questi dati: storico e vettoriale sono *line drawings* quasi privi di
texture, il caso peggiore per descrittori a blob/corner (§7.2). Su E1 —
immagine contro se stessa trasformata — questo non si vede: è proprio il punto,
E1 isola il matcher dal domain gap.
"""
from __future__ import annotations

import cv2
import numpy as np


def _grigio(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


class SiftMatcher:
    """SIFT + BFMatcher + ratio test di Lowe (§7.2)."""

    nome = "sift"

    def __init__(self, ratio: float = 0.75, n_features: int = 0):
        self.ratio = ratio
        self.n_features = n_features
        self._sift = cv2.SIFT_create(nfeatures=n_features)

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        kp_a, des_a = self._sift.detectAndCompute(_grigio(img_a), None)
        kp_b, des_b = self._sift.detectAndCompute(_grigio(img_b), None)
        meta = {
            "matcher": self.nome,
            "ratio": self.ratio,
            "n_kp_a": len(kp_a),
            "n_kp_b": len(kp_b),
        }
        if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
            return np.empty((0, 2)), np.empty((0, 2)), meta | {"n_matches": 0}

        coppie = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
        buoni = [
            m
            for m, n in (c for c in coppie if len(c) == 2)
            if m.distance < self.ratio * n.distance
        ]
        pts_a = np.array([kp_a[m.queryIdx].pt for m in buoni], dtype=np.float64).reshape(-1, 2)
        pts_b = np.array([kp_b[m.trainIdx].pt for m in buoni], dtype=np.float64).reshape(-1, 2)
        meta["n_matches"] = len(buoni)
        # Distribuzione delle distanze fra descrittori: §7.2 chiede di misurare il
        # fallimento, non solo di constatarlo.
        if buoni:
            meta["dist_mediana"] = float(np.median([m.distance for m in buoni]))
        return pts_a, pts_b, meta


class OrbMatcher:
    """ORB + BFMatcher Hamming + cross-check (§7.2)."""

    nome = "orb"

    def __init__(self, n_features: int = 5000):
        self.n_features = n_features
        self._orb = cv2.ORB_create(nfeatures=n_features)

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        kp_a, des_a = self._orb.detectAndCompute(_grigio(img_a), None)
        kp_b, des_b = self._orb.detectAndCompute(_grigio(img_b), None)
        meta = {"matcher": self.nome, "n_kp_a": len(kp_a), "n_kp_b": len(kp_b)}
        if des_a is None or des_b is None:
            return np.empty((0, 2)), np.empty((0, 2)), meta | {"n_matches": 0}

        trovati = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_a, des_b)
        pts_a = np.array([kp_a[m.queryIdx].pt for m in trovati], dtype=np.float64).reshape(-1, 2)
        pts_b = np.array([kp_b[m.trainIdx].pt for m in trovati], dtype=np.float64).reshape(-1, 2)
        meta["n_matches"] = len(trovati)
        if trovati:
            meta["dist_mediana"] = float(np.median([m.distance for m in trovati]))
        return pts_a, pts_b, meta


MATCHER = {"sift": SiftMatcher, "orb": OrbMatcher}


def crea_matcher(nome: str, **kwargs):
    """Fabbrica: il `--matcher <nome>` del contratto CLI (§9) arriva qui.

    `loftr` non è in questa tabella: vive in matchers/loftr.py con import lazy e
    viene risolto solo se richiesto (I4).
    """
    if nome == "loftr":
        from src.matchers.loftr import LoftrMatcher  # import lazy, isolato (I4)

        return LoftrMatcher(**kwargs)
    if nome not in MATCHER:
        raise ValueError(f"matcher sconosciuto: {nome!r} (disponibili: {sorted(MATCHER)}, loftr)")
    return MATCHER[nome](**kwargs)
