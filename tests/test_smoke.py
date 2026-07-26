"""Test minimi, senza framework pesanti. Si estendono milestone per milestone.

Assert nudi, nessuna dipendenza oltre a numpy: si esegue sia direttamente
(`python tests/test_smoke.py`) sia sotto pytest, se un giorno lo si aggiunge.

    python -m tests.test_smoke

I test che hanno bisogno dei dati AdE (non versionati, vedi data/README.md) si
saltano da soli se `data/raw/` non c'è, dichiarandolo.

# M3: punto trasformato avanti e indietro torna a sé entro 1e-9.
# M3: dualità (se pertinente ai crop), baseline casuale per ogni misura indiretta.
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_geo import (  # noqa: E402
    apply_affine,
    assert_dentro_estensione,
    cxf_extent,
    parse_cxf,
    pixel_size_m,
    raster_extent,
    read_cxf,
    read_jgw,
    read_metadata_txt,
)
from src.groundtruth import (  # noqa: E402
    checkpoints,
    errore_px_to_m,
    h_true,
    residuo_andata_ritorno,
    transform,
)
from src.prep.crop import CROPS, crop_world_file  # noqa: E402

RAW = "data/raw"
JGW = os.path.join(RAW, "L675_004900.jgw")
TXT = os.path.join(RAW, "L675_004900_metadata.txt")
CXF = os.path.join(RAW, "L675_004900.cxf")
JPG = os.path.join(RAW, "L675_004900.jpg")
ZIP = os.path.join(RAW, "Richiesta_921360_L675.zip")
GEOJSON = os.path.join(RAW, "L675_004900_vettoriale_qgis.geojson")

# Dimensione del foglio, letta dal file — qui serve solo come riferimento per il
# test sull'estensione, e vale 8489x5648 (non 8000x5322 come in §5.2 / I10).
def _dimensioni_foglio() -> tuple[int, int]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(_serve(JPG)) as im:
        return im.size

# Valori dichiarati in CLAUDE.md §5.3, usati come riferimento indipendente dal file.
JGW_ATTESO = (0.254453, 0.0, 0.0, -0.254453, -31480.044315, -11278.758056)


class Skip(Exception):
    """Dato non disponibile: il test si dichiara saltato, non fallito."""


def _serve(path: str) -> str:
    if not os.path.exists(path):
        raise Skip(f"manca {path} (vedi data/README.md)")
    return path


# ------------------------------------------------------------------ M1: world file


def test_read_jgw_ordine_campi():
    """Le 6 righe sono A, D, B, E, C, F: l'ordine è la trappola classica del JGW."""
    W = read_jgw(_serve(JGW))
    a, d, b, e, c, f = JGW_ATTESO
    atteso = np.array([[a, b, c], [d, e, f], [0, 0, 1]])
    assert np.allclose(W, atteso, atol=0, rtol=0), f"world file letto male:\n{W}"


def test_pixel_noto():
    """Criterio d'accettazione M1: il pixel (0,0) cade sull'origine dichiarata in §5.3."""
    W = read_jgw(_serve(JGW))
    xy = apply_affine(W, [[0, 0]])[0]
    assert abs(xy[0] - (-31480.044315)) < 1e-9, xy
    assert abs(xy[1] - (-11278.758056)) < 1e-9, xy
    # e un pixel qualsiasi segue la risoluzione dichiarata
    xy = apply_affine(W, [[1000, 400]])[0]
    assert abs(xy[0] - (-31480.044315 + 1000 * 0.254453)) < 1e-9, xy
    assert abs(xy[1] - (-11278.758056 - 400 * 0.254453)) < 1e-9, xy


def test_risoluzione():
    sx, sy = pixel_size_m(read_jgw(_serve(JGW)))
    assert abs(sx - 0.254453) < 1e-12 and abs(sy - 0.254453) < 1e-12, (sx, sy)


def test_crop_world_file_componibile():
    """Il world file del crop deve mandare il pixel (0,0) del crop dove il foglio
    manda il pixel (x0, y0). È tutta la ground truth di M3 in una riga."""
    W = read_jgw(_serve(JGW))
    for spec in CROPS:
        Wc = crop_world_file(W, spec.x0, spec.y0)
        dal_crop = apply_affine(Wc, [[0, 0]])[0]
        dal_foglio = apply_affine(W, [[spec.x0, spec.y0]])[0]
        assert np.allclose(dal_crop, dal_foglio, atol=1e-9), (spec.nome, dal_crop, dal_foglio)
        # e la scala non cambia
        assert np.allclose(pixel_size_m(Wc), pixel_size_m(W), atol=1e-12), spec.nome


def test_crop_estensioni_come_in_5_6():
    """Le estensioni ricalcolate coincidono con la tabella di §5.6 (arrotondata al metro)."""
    W = read_jgw(_serve(JGW))
    for spec in CROPS:
        Wc = crop_world_file(W, spec.x0, spec.y0)
        xmin, ymin, xmax, ymax = raster_extent(Wc, spec.w, spec.h)
        assert abs(xmin - min(spec.x_attesa)) < 1.0, (spec.nome, xmin)
        assert abs(xmax - max(spec.x_attesa)) < 1.0, (spec.nome, xmax)
        assert abs(ymin - min(spec.y_attesa)) < 1.0, (spec.nome, ymin)
        assert abs(ymax - max(spec.y_attesa)) < 1.0, (spec.nome, ymax)


# ------------------------------------------------------------------ M1: metadati


def test_metadata_txt_allineato_da_destra():
    """La legenda dichiara 12 campi, i dati ne hanno 13: c'è una Sezione senza
    etichetta fra Comune e Foglio. Se si allinea da sinistra, Foglio diventa '_'."""
    md = read_metadata_txt(_serve(TXT))
    assert md.nome_mappa == "L675_004900", md
    assert md.comune == "VARAZZE", md
    assert md.foglio == "49", md
    assert md.denominatore_scala == 2000, md
    assert md.ricampionato is True, md
    # I quattro numeri che definiscono il pavimento dell'RMSE (§5.3)
    assert md.n_coppie_omologhe == 76, md
    assert abs(md.scarto_max_m - 1.28) < 1e-9, md
    assert abs(md.scarto_medio_m - 0.56) < 1e-9, md
    assert abs(md.dev_std_m - 0.25) < 1e-9, md


# ------------------------------------------------------------------ M2: parser CXF


def test_cxf_conteggi():
    """I numeri dichiarati in §5.4, uno per uno."""
    cxf = read_cxf(_serve(CXF))
    assert cxf.nome_mappa == "L675_004900", cxf.nome_mappa
    assert cxf.denominatore_scala == 2000.0, cxf.denominatore_scala
    bordi = cxf.bordi
    assert len(bordi) == 871, len(bordi)
    assert Counter(b.codice for b in bordi) == {18: 438, 12: 432, 25: 1}
    assert sum(b.chiuso for b in bordi) == 838, sum(b.chiuso for b in bordi)
    assert sum(len(b.pts) for b in bordi) == 14272


def test_cxf_nflag():
    """La trappola di §5.4: nflag vale 0 su 838 record, ma 1, 2 o 5 sui restanti 33."""
    bordi = parse_cxf(_serve(CXF))
    assert Counter(b.nflag for b in bordi) == {0: 838, 1: 27, 2: 5, 5: 1}
    # e dove nflag > 0 gli indici sono stati davvero letti, non saltati a caso
    for b in bordi:
        assert len(b.flags) == b.nflag, (b.nome, b.nflag, b.flags)


def test_cxf_dentro_estensione_jgw():
    """Verifica obbligatoria di M2 (§5.4)."""
    bordi = parse_cxf(_serve(CXF))
    W = read_jgw(_serve(JGW))
    w, h = _dimensioni_foglio()
    assert_dentro_estensione(bordi, raster_extent(W, w, h))


def test_cxf_nessuna_coordinata_positiva():
    """Sentinella anti-sfasamento: in Cassini-Soldner Forte Diamante il foglio 49 sta
    tutto nel terzo quadrante. Una coordinata positiva (tipo 68, 82) è un indice di
    cambio tratto letto come coordinata, cioè nflag ignorato."""
    bordi = parse_cxf(_serve(CXF))
    pts = np.vstack([b.pts for b in bordi])
    assert pts.max() < 0, f"coordinata non negativa: {pts.max()}"
    xmin, ymin, xmax, ymax = cxf_extent(bordi)
    # l'estensione dichiarata in §5.2, al metro
    assert abs(xmin - (-31205)) < 1 and abs(xmax - (-29548)) < 1, (xmin, xmax)
    assert abs(ymin - (-12469)) < 1 and abs(ymax - (-11360)) < 1, (ymin, ymax)


def test_cxf_indipendente_dai_newline():
    """Git normalizza i CRLF: il CXF nel working tree è LF, quello nello zip è CRLF.
    Devono dare lo stesso identico risultato."""
    bordi_lf = parse_cxf(_serve(CXF))
    with zipfile.ZipFile(_serve(ZIP)) as z:
        grezzo = z.read("L675_004900.cxf")
    assert b"\r\n" in grezzo, "atteso CRLF nella copia dello zip"
    tmp = os.path.join(os.path.dirname(CXF), ".crlf_check.cxf")
    with open(tmp, "wb") as fh:
        fh.write(grezzo)
    try:
        bordi_crlf = parse_cxf(tmp)
    finally:
        os.remove(tmp)
    assert len(bordi_lf) == len(bordi_crlf)
    for a, b in zip(bordi_lf, bordi_crlf):
        assert a.nome == b.nome and a.codice == b.codice and a.nflag == b.nflag
        assert np.array_equal(a.pts, b.pts), a.nome


def test_cxf_contro_export_indipendente():
    """Regressione contro l'export GeoJSON fatto a suo tempo in QGIS: 871 feature,
    stessi attributi, stesse coordinate."""
    bordi = parse_cxf(_serve(CXF))
    with open(_serve(GEOJSON), encoding="utf-8") as fh:
        feature = json.load(fh)["features"]
    assert len(feature) == len(bordi) == 871
    for b, f in zip(bordi, feature):
        geom = f["geometry"]
        coord = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"]
        assert f["properties"]["nome"] == b.nome, (b.nome, f["properties"])
        assert f["properties"]["codice"] == str(b.codice), (b.nome, f["properties"])
        assert np.allclose(np.asarray(coord), b.pts, atol=1e-9), b.nome


# ------------------------------------------------------------------ M3: ground truth


def _affine(a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]])


def _affine_moderna_finta() -> np.ndarray:
    """Un'affine 'moderna' non banale: risoluzione diversa e rotazione di 12°.

    Serve a testare la composizione prima che M7 produca il raster del CXF. Se
    storica e moderna avessero la stessa scala e nessuna rotazione, H_true
    sarebbe una pura traslazione e il test non direbbe granché.
    """
    ang = np.deg2rad(12.0)
    s = 0.40  # m/px
    return _affine(
        s * np.cos(ang), -s * np.sin(ang), -30900.0, -s * np.sin(ang), -s * np.cos(ang), -11500.0
    )


def test_h_true_identita():
    """Stesso world file per entrambe: H_true è l'identità."""
    W = read_jgw(_serve(JGW))
    assert np.allclose(h_true(W, W), np.eye(3), atol=1e-12)


def test_h_true_passa_per_il_crs():
    """H_true deve mandare un pixel storico dove lo manda il giro lungo per il CRS."""
    W_hist = read_jgw(_serve(JGW))
    W_modern = _affine_moderna_finta()
    H = h_true(W_hist, W_modern)
    pts = checkpoints(8489, 5648, n=5)
    diretto = transform(H, pts)
    # giro lungo: pixel storico -> CRS -> pixel moderno
    crs = apply_affine(W_hist, pts)
    lungo = apply_affine(np.linalg.inv(W_modern), crs)
    assert np.allclose(diretto, lungo, atol=1e-9), np.abs(diretto - lungo).max()


def test_m3_andata_e_ritorno():
    """Criterio d'accettazione di M3: il punto trasformato avanti e indietro
    torna a sé entro 1e-9."""
    W_hist = read_jgw(_serve(JGW))
    for W_modern in (_affine_moderna_finta(), W_hist):
        H = h_true(W_hist, W_modern)
        residuo = residuo_andata_ritorno(H, checkpoints(1024, 1024))
        assert residuo < 1e-9, residuo


def test_h_true_fra_due_crop_e_una_traslazione():
    """Due crop dello stesso foglio condividono scala e orientamento: fra loro
    H_true è la traslazione esatta dei rispettivi offset in pixel."""
    W_a = read_jgw(_serve(os.path.join("data/crops", "tassarole.jgw")))
    W_b = read_jgw(_serve(os.path.join("data/crops", "ribba.jgw")))
    H = h_true(W_a, W_b)
    atteso = _affine(1, 0, 1500 - 3300, 0, 1, 300 - 600)  # offset di §5.6
    assert np.allclose(H, atteso, atol=1e-6), H


def test_errore_in_metri():
    """1 px dell'immagine storica vale 0.254453 m — letto dall'affine, non da una
    costante nel codice."""
    W = read_jgw(_serve(JGW))
    assert abs(errore_px_to_m(1.0, W) - 0.254453) < 1e-9
    assert abs(errore_px_to_m(4.0, W) - 4 * 0.254453) < 1e-9


def test_checkpoint_dentro_i_bordi():
    pts = checkpoints(1024, 1024, n=10, margine=0.05)
    assert pts.shape == (100, 2)
    assert pts.min() >= 0.05 * 1024 - 1e-9 and pts.max() <= 0.95 * 1024 + 1e-9


# ------------------------------------------------------------------ invarianti I3, I4


def _import_di(path: str) -> set[str]:
    """Moduli importati da un file sorgente, secondo l'AST."""
    import ast

    with open(path, encoding="utf-8") as fh:
        albero = ast.parse(fh.read(), filename=path)
    moduli: set[str] = set()
    for nodo in ast.walk(albero):
        if isinstance(nodo, ast.Import):
            moduli.update(a.name for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            moduli.add(nodo.module)
    return moduli


def test_i3_groundtruth_fuori_dalla_pipeline():
    """I3: la georeferenziazione non entra nell'algoritmo. groundtruth.py e
    io_geo.py possono essere importati da evaluate.py, non dalla pipeline né dai
    matcher."""
    lato_algoritmo = [
        "src/pipeline.py",
        "src/preprocess.py",
        "src/estimate.py",
        "src/matchers/base.py",
        "src/matchers/classic.py",
        "src/matchers/loftr.py",
    ]
    vietati = {"src.groundtruth", "src.io_geo", "groundtruth", "io_geo"}
    for path in lato_algoritmo:
        if not os.path.exists(path):
            continue
        colpevoli = _import_di(path) & vietati
        assert not colpevoli, f"{path} importa {colpevoli}: viola I3"


def test_i4_torch_solo_in_loftr():
    """I4: nessun `import torch` (o kornia) fuori da matchers/loftr.py, dove
    l'import è comunque lazy — dentro una funzione, non a livello di modulo."""
    import ast

    for cartella, _, file in os.walk("src"):
        for nome in file:
            if not nome.endswith(".py"):
                continue
            path = os.path.join(cartella, nome)
            deep = {m for m in _import_di(path) if m.split(".")[0] in {"torch", "kornia"}}
            if path.replace(os.sep, "/") != "src/matchers/loftr.py":
                assert not deep, f"{path} importa {deep}: viola I4"
                continue
            # in loftr.py l'import c'è, ma non a livello di modulo
            with open(path, encoding="utf-8") as fh:
                albero = ast.parse(fh.read(), filename=path)
            for nodo in albero.body:
                if isinstance(nodo, (ast.Import, ast.ImportFrom)):
                    nomi = (
                        [a.name for a in nodo.names]
                        if isinstance(nodo, ast.Import)
                        else [nodo.module or ""]
                    )
                    assert not any(
                        n.split(".")[0] in {"torch", "kornia"} for n in nomi
                    ), f"{path}: import di {nomi} a livello di modulo, deve essere lazy (I4)"


# ------------------------------------------------------------------ runner


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    falliti = saltati = 0
    for nome, fn in tests:
        try:
            fn()
        except Skip as exc:
            saltati += 1
            print(f"SKIP {nome}: {exc}")
        except AssertionError as exc:
            falliti += 1
            print(f"FAIL {nome}: {exc}")
        else:
            print(f"ok   {nome}")
    print(f"\n{len(tests) - falliti - saltati} ok, {falliti} falliti, {saltati} saltati")
    return 1 if falliti else 0


if __name__ == "__main__":
    raise SystemExit(main())
