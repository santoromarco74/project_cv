"""Lettura dei dati georeferenziati: world file (JGW), metadati (TXT), vettoriale (CXF).

Nessuna dipendenza geospaziale: raster e vettoriale sono già nello stesso
sistema (Cassini-Soldner Forte Diamante), non serve riproiettare. 
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Convenzione world file : le sei righe sono A, D, B, E, C, F e
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


# --------------------------------------------------------------------------- CXF

# Keyword note del formato. Servono a due cose: riconoscere l'inizio di un record
# e, quando il record non è gestito, ritrovare la sincronia saltando alla keyword
# successiva. Alcune compaiono con backslash di continuazione (LINEA\, TESTO\,
# SIMBOLO\).
CXF_KEYWORDS = frozenset({"MAPPA", "BORDO", "LINEA", "TESTO", "SIMBOLO", "FIDUCIALE", "EOF"})

# Codici dei BORDO (§5.4). Il parser non filtra: la scelta di quali usare è
# l'ablation di M7, non una decisione del lettore.
CODICE_PARTICELLA = 18
CODICE_ACQUA_STRADA = 12
CODICE_ALTRO = 25


@dataclass(frozen=True)
class BordoRecord:
    """Un poligono BORDO del CXF."""

    nome: str
    codice: int
    angolo: float
    label_xy: tuple[float, float]
    nflag: int
    flags: tuple[int, ...]
    pts: np.ndarray  # (N, 2), coordinate CRS

    @property
    def chiuso(self) -> bool:
        return len(self.pts) > 1 and bool(np.array_equal(self.pts[0], self.pts[-1]))


@dataclass(frozen=True)
class CxfFile:
    """Contenuto utile di un CXF: intestazione MAPPA e record BORDO."""

    nome_mappa: str
    denominatore_scala: float
    bordi: list[BordoRecord]
    saltati: dict[str, int]  # keyword non gestite, per record


def _is_keyword(riga: str) -> bool:
    return riga.rstrip("\\") in CXF_KEYWORDS


def read_cxf(path: str) -> CxfFile:
    """Legge un CXF e ritorna intestazione, record BORDO e conteggio dei saltati.

    Il formato è testuale, un campo per riga, CRLF, latin-1. Qui si normalizzano
    i newline: lo stesso file passato per git (che converte in LF) deve dare lo
    stesso risultato dell'originale.

    Il layout di un BORDO è:

        BORDO / nome / codice / angolo / x_etichetta / y_etichetta /
        x_etichetta / y_etichetta / nflag / N / [nflag interi] / (x, y) x N

    `nome` è testo libero: nel foglio 49 ci sono record chiamati "ACQUA",
    "STRADA", "1015", "L675_004900" e persino "A". Per questo il parser è
    sequenziale e non cerca la stringa "BORDO" nel file: un nome può somigliare
    a una keyword.

    ⚠ `nflag` è la trappola: vale 0 in 838 record su 871, ma 1, 2 o 5 nei
    restanti 33. Leggere le coordinate subito dopo N sfasa il parser su quei 33
    e fa esplodere l'estensione da ~1.6 km a ~31 km, con coordinate positive
    dove dovrebbero essere tutte negative. Vedi `assert_dentro_estensione`.

    Ogni x e ogni y stanno su righe separate: un vertice occupa due righe.
    """
    with open(path, encoding="latin-1") as fh:
        righe = [r.strip() for r in fh.read().replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    nome_mappa, denominatore = "", float("nan")
    bordi: list[BordoRecord] = []
    saltati: dict[str, int] = {}

    i, n = 0, len(righe)
    while i < n:
        token = righe[i]
        if not token:
            i += 1
            continue
        if not _is_keyword(token):
            # Riga orfana: non siamo dentro un record noto, si cerca la sincronia.
            i += 1
            continue

        if token == "MAPPA":
            nome_mappa = righe[i + 1]
            denominatore = float(righe[i + 2])
            i += 3
        elif token == "BORDO":
            try:
                record, i = _read_bordo(righe, i)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}: record BORDO malformato alla riga {i + 1}: {exc}") from exc
            bordi.append(record)
        else:
            # LINEA, TESTO, SIMBOLO, FIDUCIALE, EOF: non servono al progetto (§5.4).
            # Si salta alla keyword successiva invece di indovinarne il layout.
            chiave = token.rstrip("\\")
            saltati[chiave] = saltati.get(chiave, 0) + 1
            i += 1
            while i < n and not _is_keyword(righe[i]):
                i += 1

    return CxfFile(nome_mappa, denominatore, bordi, saltati)


def _read_bordo(righe: list[str], i: int) -> tuple[BordoRecord, int]:
    """Legge un record BORDO che comincia alla riga `i`. Ritorna (record, riga_dopo)."""
    nome = righe[i + 1]
    codice = int(righe[i + 2])
    angolo = float(righe[i + 3])
    label = (float(righe[i + 4]), float(righe[i + 5]))
    # righe i+6, i+7: il punto di etichetta ripetuto
    nflag = int(righe[i + 8])
    n_vertici = int(righe[i + 9])

    j = i + 10
    flags = tuple(int(righe[j + k]) for k in range(nflag))  # <- la trappola di §5.4
    j += nflag

    coord = np.array([float(righe[j + k]) for k in range(2 * n_vertici)], dtype=np.float64)
    pts = coord.reshape(n_vertici, 2)
    return BordoRecord(nome, codice, angolo, label, nflag, flags, pts), j + 2 * n_vertici


def parse_cxf(path: str) -> list[BordoRecord]:
    """Parser del formato CXF (testuale AdE). Ritorna la lista dei record BORDO.

    Dopo i due punti di etichetta c'è <nflag>,
    numero di indici extra che PRECEDONO il blocco coordinate. Ignorarlo sfasa
    il parser su 33 record su 871 e fa esplodere l'estensione. Dopo il parsing,
    ASSERIRE che ogni coordinata cada dentro l'estensione del JGW.

    Codici: 18 = particella, 12 = acqua/strada, 25 = altro (da scartare).
    """
    return read_cxf(path).bordi


def cxf_extent(bordi) -> tuple[float, float, float, float]:
    """Estensione (xmin, ymin, xmax, ymax) dei poligoni."""
    pts = np.vstack([b.pts for b in bordi])
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def assert_dentro_estensione(bordi, extent, tolleranza_m: float = 0.0) -> None:
    """Verifica obbligatoria di M2 (§5.4): ogni coordinata dentro l'estensione JGW.

    È il test che smaschera lo sfasamento da `nflag`: se il parser si disallinea,
    legge indici di tratto come coordinate e l'estensione esplode.
    """
    xmin, ymin, xmax, ymax = extent
    for b in bordi:
        x, y = b.pts[:, 0], b.pts[:, 1]
        fuori = (
            (x < xmin - tolleranza_m)
            | (x > xmax + tolleranza_m)
            | (y < ymin - tolleranza_m)
            | (y > ymax + tolleranza_m)
        )
        if fuori.any():
            k = int(np.argmax(fuori))
            raise AssertionError(
                f"BORDO {b.nome!r} (codice {b.codice}, nflag {b.nflag}): vertice {k} = "
                f"({x[k]:.3f}, {y[k]:.3f}) fuori dall'estensione del raster "
                f"X [{xmin:.3f}, {xmax:.3f}] Y [{ymin:.3f}, {ymax:.3f}]. "
                f"Sintomo tipico di nflag ignorato, che sfasa il parser."
            )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Ispeziona un CXF e verifica che i BORDO siano dentro l'estensione del JGW")
    ap.add_argument("--cxf", default="data/raw/L675_004900.cxf")
    ap.add_argument("--jgw", default="data/raw/L675_004900.jgw")
    ap.add_argument("--sheet", default="data/raw/L675_004900.jpg", help="serve solo per le dimensioni")
    args = ap.parse_args(argv)

    cxf = read_cxf(args.cxf)
    bordi = cxf.bordi
    print(f"mappa       : {cxf.nome_mappa}  1:{cxf.denominatore_scala:.0f}")
    print(f"record BORDO: {len(bordi)}")

    from collections import Counter

    codici = Counter(b.codice for b in bordi)
    etichette = {
        CODICE_PARTICELLA: "particelle",
        CODICE_ACQUA_STRADA: "acqua/strada/bordo",
        CODICE_ALTRO: "altro (da scartare)",
    }
    for codice, quanti in sorted(codici.items()):
        print(f"  codice {codice:<3}: {quanti:>4}  {etichette.get(codice, '?')}")
    print(f"nflag       : {dict(sorted(Counter(b.nflag for b in bordi).items()))}")
    print(f"chiusi      : {sum(b.chiuso for b in bordi)} / {len(bordi)}")
    print(f"vertici     : {sum(len(b.pts) for b in bordi)}")
    print(f"record saltati: {dict(sorted(cxf.saltati.items()))}")

    xmin, ymin, xmax, ymax = cxf_extent(bordi)
    print(f"estensione  : X {xmin:.3f} .. {xmax:.3f}   Y {ymin:.3f} .. {ymax:.3f}")

    if os.path.exists(args.jgw) and os.path.exists(args.sheet):
        from PIL import Image  # legge le dimensioni dall'header, non decodifica

        Image.MAX_IMAGE_PIXELS = None
        W = read_jgw(args.jgw)
        with Image.open(args.sheet) as im:
            w, h = im.size
        extent = raster_extent(W, w, h)
        print(f"raster      : X {extent[0]:.3f} .. {extent[2]:.3f}   Y {extent[1]:.3f} .. {extent[3]:.3f}")
        assert_dentro_estensione(bordi, extent)
        print("verifica    : ogni coordinata è dentro l'estensione del raster ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
