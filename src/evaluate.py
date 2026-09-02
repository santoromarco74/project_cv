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
    "codici",  # E2: quali codici CXF sono stati rasterizzati (§5.4)
    "modello",
    "degrado",
    "rot_deg",
    "scala",
    "tx",
    "ty",
    "prospettiva",
    "ratio",
    "loftr_conf",
    "loftr_max_lato",
    "ransac_thresh",
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
    # Costo di inizializzazione del matcher, separato dall'inferenza. Per SIFT e
    # ORB è trascurabile; per LoFTR è il caricamento del checkpoint, che prima
    # stava dentro `t_match_ms` e ne gonfiava il confronto con i classici
    # (§10.1). Con la cache dei matcher si paga alla prima registrazione e poi
    # è zero: tenerlo in colonna a parte è ciò che rende leggibile la differenza.
    "t_init_ms",
    "t_match_ms",
    "t_stima_ms",
)


def parametri_matcher(meta: dict) -> dict:
    """I parametri del matcher che vanno nel CSV, letti dai suoi metadati.

    Una variabile che influenza il risultato e non finisce nel CSV rende due
    esperimenti diversi indistinguibili in tabella (I6, I9): con `--ratio 0.95`
    la riga era identica a una con 0.75. Si legge da `meta`, cioè da quello che
    il matcher dice di aver usato, e non dagli argomenti della CLI, che dicono
    solo quello che è stato chiesto.

    Ogni matcher riporta i suoi e nessun altro. Scrivere `ratio` accanto a una
    riga di ORB significherebbe mettere a CSV un parametro finto, che nella
    tabella sembra aver avuto un effetto: è la stessa ragione per cui la
    pipeline non passa il ratio a ORB.
    """
    fuori = {}
    if "ratio" in meta:  # SIFT
        fuori["ratio"] = meta["ratio"]
    if "soglia_conf" in meta:  # LoFTR
        fuori["loftr_conf"] = meta["soglia_conf"]
    if "max_lato" in meta:
        fuori["loftr_max_lato"] = meta["max_lato"]
    return fuori


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
    restano vuote e il resto funziona lo stesso.

    `success` segue §7.4 alla lettera: "stima riuscita **e** RMSE sotto soglia
    dichiarata". Senza soglia dichiarata non è definito, e resta vuoto — non
    True. Metterci True significherebbe scrivere `success` accanto a un errore
    di 200 m, che è esattamente il tipo di numero che poi finisce in una tabella.
    """
    riga = {
        "success_stima": bool(stima.success),
        "modello": stima.modello,
        "n_matches": stima.n_matches,
        "n_inliers": stima.n_inliers,
        "inlier_ratio": round(stima.inlier_ratio, 6),
        "motivo": stima.motivo,
        # La soglia di RANSAC arriva dalla stima e non da chi chiama: così ogni
        # esperimento la registra senza doversene ricordare. Resta vuota quando
        # RANSAC non è mai partito (meno di 4 corrispondenze), perché in quel
        # caso non c'è nessuna soglia che abbia agito.
        "ransac_thresh": stima.meta.get("ransac_thresh"),
        "rmse_px": None,
        "rmse_m": None,
        "err_max_px": None,
        "success": False if soglia_m is not None else "",
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
        riga["success"] = ""  # non dichiarata: la domanda non ha risposta
    else:
        riga["success"] = riga["rmse_m"] is not None and riga["rmse_m"] < soglia_m
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
    if not nuovo:
        _verifica_intestazione(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLONNE, extrasaction="ignore")
        if nuovo:
            w.writeheader()
        w.writerow({k: riga.get(k, "") for k in COLONNE})


def _verifica_intestazione(path: str) -> None:
    """Rifiuta di appendere a un CSV con uno schema diverso.

    Se COLONNE cambia (una colonna nuova, un ordine diverso) e si continua ad
    appendere a un file vecchio, le righe finiscono disallineate rispetto
    all'intestazione: i valori ci sono tutti, sono solo sotto la colonna
    sbagliata. Nessuna aggregazione se ne accorge, e i numeri della relazione
    diventano falsi in modo silenzioso. Meglio fermarsi e rigenerare.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        intestazione = next(csv.reader(fh), None)
    if intestazione is not None and tuple(intestazione) != COLONNE:
        mancanti = set(COLONNE) - set(intestazione)
        avanzo = set(intestazione) - set(COLONNE)
        raise ValueError(
            f"{path}: lo schema del CSV non coincide con COLONNE "
            f"(in più nello schema: {sorted(mancanti) or '—'}, "
            f"nel file e non nello schema: {sorted(avanzo) or '—'}). "
            "Rigenera il file invece di appendere: le righe finirebbero sotto "
            "le colonne sbagliate senza che nulla se ne accorga."
        )
