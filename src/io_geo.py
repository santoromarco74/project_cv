"""Lettura dei dati georeferenziati: world file (JGW), metadati (TXT), vettoriale (CXF).

Nessuna dipendenza geospaziale: raster e vettoriale sono già nello stesso
sistema (Cassini-Soldner Forte Diamante), non serve riproiettare. Vedi CLAUDE.md §5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Convenzione world file (CLAUDE.md §5.3): le sei righe sono A, D, B, E, C, F e
#   X = A*col + B*row + C ,  Y = D*col + E*row + F
# dove (C, F) è il CENTRO del pixel (0,0), non il suo spigolo. La differenza è
# mezzo pixel (0.127 m a questa risoluzione): irrilevante per il matching,
# fastidiosa quando si confrontano estensioni con QGIS, che ragiona per spigoli.


def read_jgw(path: str) -> np.ndarray:
    """Legge un world file a 6 righe e ritorna l'affine pixel->CRS come matrice 3x3.

    Righe JGW: A, D, B, E, C, F  ->  X = A*col + B*row + C ; Y = D*col + E*row + F

    La matrice ritornata agisce su coordinate omogenee (col, row, 1):
        [X, Y, 1]^T = W @ [col, row, 1]^T
    """
    with open(path, encoding="latin-1") as fh:
        values = [float(line) for line in fh if line.strip()]
    if len(values) != 6:
        raise ValueError(f"{path}: attese 6 righe nel world file, trovate {len(values)}")
    a, d, b, e, c, f = values
    return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]], dtype=np.float64)


def write_jgw(path: str, W: np.ndarray) -> None:
    """Scrive un world file dalla matrice 3x3, nell'ordine A, D, B, E, C, F."""
    a, b, c = W[0]
    d, e, f = W[1]
    with open(path, "w", encoding="latin-1", newline="\n") as fh:
        for value in (a, d, b, e, c, f):
            fh.write(f"{value:.10g}\n")


def apply_affine(W: np.ndarray, pts) -> np.ndarray:
    """Applica un'affine 3x3 a un array (N,2). Ritorna un array (N,2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    return homo @ W[:2].T


def pixel_size_m(W: np.ndarray) -> tuple[float, float]:
    """Dimensione del pixel in metri (x, y), come moduli delle colonne dell'affine."""
    return float(np.hypot(W[0, 0], W[1, 0])), float(np.hypot(W[0, 1], W[1, 1]))


def raster_extent(W: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    """Estensione (xmin, ymin, xmax, ymax) del raster, nella convenzione del JGW.

    Gli angoli sono i CENTRI dei pixel estremi (0,0) e (width-1, height-1), coerente
    con read_jgw. Per l'estensione "a spigoli" alla QGIS va aggiunto mezzo pixel
    su ogni lato.
    """
    corners = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64
    )
    xy = apply_affine(W, corners)
    return (
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    )


# --------------------------------------------------------------------------- TXT


@dataclass(frozen=True)
class SheetMetadata:
    """Metadati del foglio dal TXT AdE. Definiscono il pavimento dell'RMSE (§5.3)."""

    nome_mappa: str
    comune: str
    foglio: str
    denominatore_scala: int
    data_orientamento: str
    ricampionato: bool
    n_coppie_omologhe: int
    scarto_max_m: float
    scarto_medio_m: float
    dev_std_m: float


def read_metadata_txt(path: str) -> SheetMetadata:
    """Legge il TXT dei metadati AdE.

    ⚠ Trappola di formato: la riga di legenda dichiara 12 campi, la riga dati ne
    ha 13. Fra `Comune` e `Foglio` c'è una colonna Sezione senza etichetta
    (`VARAZZE|_|49`). Allineando da sinistra si legge Foglio="_" e Allegato="49",
    che è sbagliato. Qui si allinea DA DESTRA: i valori che servono davvero
    (coppie omologhe e scarti) stanno in coda, e la coda è stabile.
    """
    with open(path, encoding="latin-1") as fh:
        rows = [
            line.strip()
            for line in fh
            if "|" in line and not line.lstrip().startswith("Nome mappa")
        ]
    if not rows:
        raise ValueError(f"{path}: nessuna riga dati trovata")
    fields = [f.strip() for f in rows[-1].rstrip("|").split("|")]
    if len(fields) < 12:
        raise ValueError(f"{path}: attesi almeno 12 campi, trovati {len(fields)}")

    # Coda, allineata da destra: ... | denom | data | ricamp | coppie | max | medio | std
    dev_std, scarto_medio, scarto_max, n_coppie = fields[-1], fields[-2], fields[-3], fields[-4]
    ricampionato, data_orient, denominatore = fields[-5], fields[-6], fields[-7]
    # Testa, allineata da sinistra: nome | comune | [sezione] | foglio | ...
    nome, comune = fields[0], fields[1]
    foglio = fields[3] if len(fields) >= 13 else fields[2]

    return SheetMetadata(
        nome_mappa=nome,
        comune=comune,
        foglio=foglio,
        denominatore_scala=int(denominatore),
        data_orientamento=data_orient,
        ricampionato=ricampionato.upper().startswith("S"),
        n_coppie_omologhe=int(n_coppie),
        scarto_max_m=float(scarto_max),
        scarto_medio_m=float(scarto_medio),
        dev_std_m=float(dev_std),
    )


def parse_cxf(path: str):
    """Parser del formato CXF (testuale AdE). Ritorna la lista dei record BORDO.

    ATTENZIONE (CLAUDE.md §5.4): dopo i due punti di etichetta c'è <nflag>,
    numero di indici extra che PRECEDONO il blocco coordinate. Ignorarlo sfasa
    il parser su 33 record su 871 e fa esplodere l'estensione. Dopo il parsing,
    ASSERIRE che ogni coordinata cada dentro l'estensione del JGW.

    Codici: 18 = particella, 12 = acqua/strada, 25 = altro (da scartare).
    """
    raise NotImplementedError  # M2
