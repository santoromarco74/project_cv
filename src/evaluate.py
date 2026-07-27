"""Metriche: RMSE su checkpoint, inlier ratio, success (§7.4).

⚠ I3 — questo è l'unico modulo che importa `groundtruth`. La pipeline non sa che
esiste un world file; qui si confronta ciò che la pipeline ha stimato con il
riferimento, e si producono numeri (I6).

L'RMSE si misura su una griglia di checkpoint nell'immagine storica, non sulle
corrispondenze: gli inlier di RANSAC sono i punti su cui il modello è già stato
adattato, misurarci sopra l'errore direbbe quanto bene il modello si spiega da
solo. I checkpoint sono indipendenti dalla stima.

Correlazione e chamfer NON sono metriche di valutazione su questi dati (§5.5).

In M4 questo modulo serve a E1, dove `H_true` viene dalla generazione sintetica.
Da M7 la stessa funzione riceve la `H_true` composta dai world file: è la stessa
metrica, cambia solo da dove arriva il riferimento.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from src.groundtruth import checkpoints, errore_px_to_m, transform

# Schema del CSV (§7.4). Fisso e ordinato: le tabelle della relazione sono
# aggregazioni di questo file, non numeri ricopiati a mano, quindi le colonne
# non possono cambiare nome fra un esperimento e l'altro.
COLONNE = (
    "esperimento",
    "crop",
    "matcher",
    "preprocess",
    "morph_open",
    "morph_close",
    "modello",
    "degrado",
    "rot_deg",
    "scala",
    "tx",
    "ty",
    "prospettiva",
    "seed",
    "n_kp_a",
    "n_kp_b",
    "n_matches",
    "n_inliers",
    "inlier_ratio",
    "rmse_px",
    "rmse_m",
    "err_max_px",
    "success_stima",
    "success",
    "motivo",
    "t_match_ms",
    "t_stima_ms",
)


def errori_px(H_est: np.ndarray, H_true: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distanza fra dove H_est manda ogni checkpoint e dove lo manda H_true."""
    return np.linalg.norm(transform(H_est, pts) - transform(H_true, pts), axis=1)


def rmse_px(H_est: np.ndarray, H_true: np.ndarray, pts: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errori_px(H_est, H_true, pts) ** 2)))


def valuta(
    stima,
    H_true: np.ndarray,
    width: int,
    height: int,
    W_hist: np.ndarray | None = None,
    soglia_m: float | None = None,
    n_checkpoint: int = 10,
) -> dict:
    """Riga di metriche per un singolo esperimento (§7.4).

    `W_hist` serve solo a convertire i pixel in metri: senza, le colonne in metri
    restano vuote e il resto funziona lo stesso. `soglia_m` decide `success`;
    senza soglia, `success` è solo la riuscita della stima.
    """
    riga = {
        "success_stima": bool(stima.success),
        "modello": stima.modello,
        "n_matches": stima.n_matches,
        "n_inliers": stima.n_inliers,
        "inlier_ratio": round(stima.inlier_ratio, 6),
        "motivo": stima.motivo,
        "rmse_px": None,
        "rmse_m": None,
        "err_max_px": None,
        "success": False,
    }
    if not stima.success or stima.H is None:
        return riga

    pts = checkpoints(width, height, n=n_checkpoint)
    errori = errori_px(stima.H, H_true, pts)
    riga["rmse_px"] = float(np.sqrt(np.mean(errori**2)))
    riga["err_max_px"] = float(errori.max())
    if W_hist is not None:
        riga["rmse_m"] = errore_px_to_m(riga["rmse_px"], W_hist)

    if soglia_m is None:
        riga["success"] = True
    elif riga["rmse_m"] is not None:
        riga["success"] = riga["rmse_m"] < soglia_m
    else:
        riga["success"] = False
    return riga


def append_csv(path: str, riga: dict) -> None:
    """Aggiunge una riga al CSV, scrivendo l'intestazione se il file è nuovo.

    Le chiavi fuori da COLONNE vengono ignorate e quelle mancanti restano vuote:
    un esperimento che non usa la degradazione non deve rompere lo schema, e uno
    che inventa una colonna non deve sporcarlo.

    Le righe di fallimento si scrivono come le altre (§7.3, I7): un esperimento
    che non produce una stima ha comunque prodotto un dato.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nuovo = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLONNE, extrasaction="ignore")
        if nuovo:
            w.writeheader()
        w.writerow({k: riga.get(k, "") for k in COLONNE})
