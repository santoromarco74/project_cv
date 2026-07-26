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

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_geo import (  # noqa: E402
    apply_affine,
    pixel_size_m,
    raster_extent,
    read_jgw,
    read_metadata_txt,
)
from src.prep.crop import CROPS, crop_world_file  # noqa: E402

RAW = "data/raw"
JGW = os.path.join(RAW, "L675_004900.jgw")
TXT = os.path.join(RAW, "L675_004900_metadata.txt")

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
