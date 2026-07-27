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

import cv2
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
from src.estimate import stima as stima_ransac  # noqa: E402
from src.evaluate import COLONNE, append_csv, valuta  # noqa: E402
from src.matchers.classic import crea_matcher  # noqa: E402
from src.preprocess import (  # noqa: E402
    applica,
    morfologia,
    otsu,
    percentuale_inchiostro,
    rimuovi_componenti,
    sauvola,
)
from src.prep.crop import CROPS, crop_world_file  # noqa: E402
from src.prep.rasterize import griglia, per_crop, rasterizza  # noqa: E402
from src.prep.synth import (  # noqa: E402
    Trasformazione,
    genera_coppia,
    matrice,
    scala_degradazione,
)

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


# ------------------------------------------------------------------ M4: E1 sintetico


def _crop_di_prova(lato: int = 512) -> np.ndarray:
    """Un pezzo di crop reale, ridotto per tenere veloce la suite."""
    import cv2

    path = os.path.join("data/crops", "ribba.png")
    img = cv2.imread(_serve(path), cv2.IMREAD_COLOR)
    if img is None:
        raise Skip(f"{path} illeggibile")
    return img[:lato, :lato].copy()


def test_synth_identita():
    """Trasformazione neutra: H è l'identità e B è identica ad A."""
    img = _crop_di_prova(128)
    b, H = genera_coppia(img, Trasformazione())
    assert np.allclose(H, np.eye(3), atol=1e-12), H
    assert np.array_equal(b, img)


def test_synth_h_coerente_con_i_parametri():
    """Il centro resta il centro più la traslazione: rotazione e scala sono
    centrate sull'immagine, non sull'origine."""
    w = h = 200
    t = Trasformazione(rot_deg=30, scala=1.4, tx=10, ty=-5)
    H = matrice(t, w, h)
    centro = np.array([[w / 2, h / 2]])
    assert np.allclose(transform(H, centro)[0], [w / 2 + 10, h / 2 - 5], atol=1e-9)


def test_synth_deterministico():
    """I9: stesso seed, stessi pixel, bit per bit. Seed diverso, immagine diversa."""
    img = _crop_di_prova(128)
    t = Trasformazione(rot_deg=7, scala=1.1)
    d = scala_degradazione(0.6)
    a1, h1 = genera_coppia(img, t, d, seed=42)
    a2, h2 = genera_coppia(img, t, d, seed=42)
    a3, _ = genera_coppia(img, t, d, seed=43)
    assert np.array_equal(a1, a2) and np.array_equal(h1, h2)
    assert not np.array_equal(a1, a3), "seed diverso ma stessa immagine: il rumore non è seminato"


def test_estimate_fallimento_pulito():
    """§7.3: sotto i 4 match si ritorna success=False, non si solleva un'eccezione.
    La riga del CSV va scritta comunque (I7)."""
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 5.0]])
    st = stima_ransac(pts, pts, modello="homography")
    assert st.success is False and st.H is None
    assert st.n_matches == 3 and "almeno 4" in st.motivo
    assert st.inlier_ratio == 0.0


def test_estimate_modello_sconosciuto():
    pts = np.zeros((10, 2))
    try:
        stima_ransac(pts, pts, modello="proiettiva")
    except ValueError as exc:
        assert "proiettiva" in str(exc)
    else:
        raise AssertionError("atteso ValueError su modello sconosciuto")


def test_estimate_deterministico():
    """I9: RANSAC con lo stesso seed dà la stessa H, cifra per cifra."""
    rng = np.random.default_rng(0)
    pts_a = rng.uniform(0, 500, size=(200, 2))
    H = matrice(Trasformazione(rot_deg=12, scala=1.1, tx=20), 500, 500)
    pts_b = transform(H, pts_a)
    s1 = stima_ransac(pts_a, pts_b, modello="homography", seed=42)
    s2 = stima_ransac(pts_a, pts_b, modello="homography", seed=42)
    assert np.array_equal(s1.H, s2.H), "stesso seed, H diversa"


def test_matcher_rispettano_l_interfaccia():
    """I4: la pipeline non deve sapere quale matcher sta usando. Entrambi
    ritornano (N,2), (N,2), dict con le stesse chiavi minime."""
    img = _crop_di_prova(256)
    b, _ = genera_coppia(img, Trasformazione(rot_deg=5))
    for nome in ("sift", "orb"):
        pts_a, pts_b, meta = crea_matcher(nome).match(img, b)
        assert pts_a.ndim == 2 and pts_a.shape[1] == 2, (nome, pts_a.shape)
        assert pts_a.shape == pts_b.shape, (nome, pts_a.shape, pts_b.shape)
        assert meta["matcher"] == nome and "n_matches" in meta, meta


def test_m4_e1_sift_recupera_h():
    """Criterio d'accettazione di M4: su E1, SIFT recupera H con RMSE ≈ 0.

    Soglia a 0.5 px = 0.13 m, cioè un quarto del pavimento del riferimento reale
    (§5.3). Se questo test fallisce, il bug è nel codice e ci si ferma (§8).
    """
    img = _crop_di_prova(512)
    W_hist = read_jgw(_serve(JGW))
    h, w = img.shape[:2]
    for t in (
        Trasformazione(tx=25, ty=-15),
        Trasformazione(rot_deg=12, scala=1.15, tx=20, ty=10),
        Trasformazione(rot_deg=8, scala=1.05, prospettiva=8e-5),
    ):
        b, H_true = genera_coppia(img, t)
        pts_a, pts_b, _ = crea_matcher("sift").match(img, b)
        st = stima_ransac(pts_a, pts_b, modello="homography", seed=42)
        riga = valuta(st, H_true, w, h, W_hist=W_hist)
        assert riga["success_stima"], (t, st.motivo)
        assert riga["rmse_px"] < 0.5, (t, riga["rmse_px"])
        assert riga["inlier_ratio"] > 0.5, (t, riga["inlier_ratio"])


# ------------------------------------------------------------------ M5: preprocessing


def _sauvola_ingenuo(gray, finestra=25, k=0.2, R=128.0):
    """Sauvola scritto nel modo ovvio: una finestra per pixel, O(finestra²).

    Serve solo come riferimento per verificare la versione con immagini
    integrali. Se le due non coincidono, l'ottimizzazione ha introdotto un bug —
    ed è il tipo di bug che non si vede a occhio sull'immagine.
    """
    import cv2

    r = finestra // 2
    pad = cv2.copyMakeBorder(gray.astype(np.float64), r, r, r, r, cv2.BORDER_REFLECT_101)
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            fin = pad[i : i + finestra, j : j + finestra]
            m, s = fin.mean(), fin.std()
            out[i, j] = 0 if gray[i, j] < m * (1 + k * (s / R - 1)) else 255
    return out


def test_sauvola_uguale_alla_versione_ingenua():
    """Le immagini integrali danno O(1) per pixel: devono dare gli stessi pixel."""
    rng = np.random.default_rng(7)
    gray = rng.integers(90, 230, size=(40, 40), dtype=np.uint8)
    gray[12:20, 5:35] = 30  # un tratto scuro
    veloce = sauvola(gray, finestra=25, k=0.2)
    lento = _sauvola_ingenuo(gray, finestra=25, k=0.2)
    assert np.array_equal(veloce, lento), int((veloce != lento).sum())


def test_binarizzazioni_sono_binarie_e_scure_sul_tratto():
    """Polarità: l'inchiostro resta scuro, come nell'originale."""
    gray = np.full((60, 60), 220, dtype=np.uint8)
    gray[20:40, 20:40] = 40  # tratto
    for out in (otsu(gray), sauvola(gray)):
        assert set(np.unique(out)) <= {0, 255}, np.unique(out)
        assert out[30, 30] == 0, "il tratto dovrebbe essere inchiostro (scuro)"
        assert out[5, 5] == 255, "la carta dovrebbe essere sfondo (chiaro)"


def test_chiusura_salda_le_interruzioni():
    """`chiudi` deve saldare un tratto spezzato: è il suo scopo dichiarato."""
    img = np.full((40, 40), 255, dtype=np.uint8)
    img[20, 5:35] = 0  # tratto orizzontale
    img[20, 19:21] = 255  # con un buco di 2 px
    saldato = morfologia(img, chiudi=1)
    assert saldato[20, 19] == 0 and saldato[20, 20] == 0, "il buco non è stato saldato"


def test_apertura_toglie_il_pepe():
    img = np.full((40, 40), 255, dtype=np.uint8)
    img[10:30, 10:30] = 0  # blocco di inchiostro
    img[5, 5] = 0  # un granello isolato
    pulito = morfologia(img, apri=1)
    assert pulito[5, 5] == 255, "il granello non è stato rimosso"
    assert pulito[20, 20] == 0, "l'apertura ha mangiato il tratto vero"


def test_rimuovi_componenti_per_area():
    img = np.full((60, 60), 255, dtype=np.uint8)
    img[10:14, 10:14] = 0  # 16 px
    img[30:50, 30:50] = 0  # 400 px
    out = rimuovi_componenti(img, area_min=100)
    assert out[11, 11] == 255, "la componente piccola doveva sparire"
    assert out[40, 40] == 0, "la componente grande doveva restare"


def test_applica_modi_e_morfologia_ignorata():
    """Su `none` e `clahe` l'uscita è grayscale: morfologia e pulizia non si
    applicano, e non devono nemmeno provarci."""
    # una rampa, non due soli livelli: su un'immagine bicolore "l'uscita non è
    # binaria" non sarebbe distinguibile da "l'uscita è binaria"
    rampa = np.tile(np.linspace(120, 240, 50, dtype=np.uint8), (50, 1))
    img = cv2.cvtColor(rampa, cv2.COLOR_GRAY2BGR)
    img[20:30, 20:30] = 30
    grigio = applica(img, modo="none", morph_close=3, area_min=50)
    assert grigio.ndim == 2 and len(np.unique(grigio)) > 2
    assert np.array_equal(grigio, applica(img, modo="none"))
    assert applica(img, modo="clahe").ndim == 2
    for modo in ("otsu", "sauvola"):
        assert set(np.unique(applica(img, modo=modo))) <= {0, 255}
    try:
        applica(img, modo="adattiva")
    except ValueError as exc:
        assert "adattiva" in str(exc)
    else:
        raise AssertionError("atteso ValueError su preprocess sconosciuto")


def test_sauvola_regge_il_gradiente_dove_otsu_cede():
    """Il risultato atteso di §7.1, misurato invece che assunto.

    Con un gradiente di illuminazione, la soglia globale di Otsu finisce per
    classificare come inchiostro la carta scurita; Sauvola, che guarda una
    finestra locale, non se ne accorge.
    """
    rng = np.random.default_rng(3)
    gray = rng.integers(200, 240, size=(200, 200)).astype(np.uint8)
    gray[::20, :] = 40  # tratto regolare
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    scurito = np.clip(gray * (1.0 - 0.9 * (xx / (w - 1) + yy / (h - 1)) / 2), 0, 255).astype(
        np.uint8
    )

    base = percentuale_inchiostro(sauvola(gray))
    assert percentuale_inchiostro(otsu(scurito)) > 0.30, "atteso il collasso di Otsu"
    assert abs(percentuale_inchiostro(sauvola(scurito)) - base) < 0.02, "Sauvola non è stabile"


# ------------------------------------------------------------------ M6: CSV


def _csv_temporaneo() -> str:
    import tempfile

    return os.path.join(tempfile.mkdtemp(prefix="histreg-test-"), "runs.csv")


def test_csv_intestazione_una_volta_sola():
    """Il CSV si scrive in append: l'intestazione va messa solo alla creazione."""
    path = _csv_temporaneo()
    for i in range(3):
        append_csv(path, {"esperimento": "E1", "crop": f"c{i}", "rmse_px": i})
    with open(path, encoding="utf-8") as fh:
        righe = fh.read().strip().split("\n")
    assert len(righe) == 4, righe
    assert righe[0].split(",") == list(COLONNE)
    assert sum(r.startswith("esperimento,") for r in righe) == 1


def test_csv_schema_stabile():
    """Chiavi mancanti restano vuote, chiavi estranee non entrano: le colonne non
    possono cambiare fra un esperimento e l'altro, o le aggregazioni non tornano."""
    path = _csv_temporaneo()
    append_csv(path, {"esperimento": "E1", "colonna_inventata": 99})
    with open(path, encoding="utf-8") as fh:
        intestazione, riga = fh.read().strip().split("\n")
    assert "colonna_inventata" not in intestazione
    assert len(riga.split(",")) == len(COLONNE)


def test_csv_scrive_anche_i_fallimenti():
    """§7.3 e I7: una stima fallita produce comunque la sua riga."""
    path = _csv_temporaneo()
    pts = np.zeros((2, 2))
    st = stima_ransac(pts, pts, modello="homography")
    riga = valuta(st, np.eye(3), 100, 100)
    append_csv(path, riga | {"esperimento": "E1", "crop": "finto"})
    with open(path, encoding="utf-8") as fh:
        contenuto = fh.read().strip().split("\n")
    assert len(contenuto) == 2, contenuto
    assert "False" in contenuto[1], contenuto[1]


def test_csv_deterministico():
    """I9: stesso input e stessi parametri = stesso CSV, cifra per cifra."""
    img = _crop_di_prova(256)
    t = Trasformazione(rot_deg=12, scala=1.1)

    def una_passata() -> str:
        path = _csv_temporaneo()
        for livello in (0.0, 0.5):
            b, H_true = genera_coppia(img, t, scala_degradazione(livello), seed=42)
            pts_a, pts_b, meta = crea_matcher("sift").match(img, b)
            st = stima_ransac(pts_a, pts_b, seed=42)
            riga = valuta(st, H_true, 256, 256)
            # i tempi sono l'unica colonna non riproducibile: si escludono
            append_csv(path, riga | {"esperimento": "E1", "degrado": livello})
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    assert una_passata() == una_passata(), "stesso input, CSV diverso"


# ------------------------------------------------------------------ M7: rasterizzazione


def test_griglia_copre_l_estensione():
    """La griglia deve contenere l'estensione richiesta, alla risoluzione data."""
    W, w, h = griglia(-100.0, -50.0, 0.0, 50.0, risoluzione=0.25)
    assert w >= 100 / 0.25 and h >= 100 / 0.25, (w, h)
    xmin, ymin, xmax, ymax = raster_extent(W, w, h)
    assert xmin <= -100 + 1e-9 and xmax >= -1e-9, (xmin, xmax)
    assert ymin <= -50 + 1e-9 and ymax >= 50 - 1e-9, (ymin, ymax)
    assert np.allclose(pixel_size_m(W), (0.25, 0.25))
    # riga che cresce verso sud, come nel JGW
    assert W[1, 1] < 0


def test_rasterizza_filtra_per_codice():
    """Il filtro 18 / 12 è l'ablation di §5.4: deve cambiare davvero l'immagine."""
    bordi = parse_cxf(_serve(CXF))
    W, w, h = griglia(-30660.0, -11712.0, -30360.0, -11412.0, risoluzione=1.0)
    solo18 = rasterizza(bordi, W, w, h, codici=(18,))
    solo12 = rasterizza(bordi, W, w, h, codici=(12,))
    insieme = rasterizza(bordi, W, w, h, codici=(18, 12))
    for img in (solo18, solo12, insieme):
        assert set(np.unique(img)) <= set(range(256)) and img.dtype == np.uint8
        assert (img < 128).any(), "nessun tratto disegnato"
        assert img[0, 0] == 255, "lo sfondo deve restare chiaro (polarità dello storico)"
    assert not np.array_equal(solo18, solo12)
    assert (insieme < 128).sum() > (solo18 < 128).sum(), "18+12 deve avere più tratto di 18"


def test_rasterizza_deterministico():
    bordi = parse_cxf(_serve(CXF))
    W, w, h = griglia(-30660.0, -11712.0, -30360.0, -11412.0, risoluzione=1.0)
    assert np.array_equal(rasterizza(bordi, W, w, h), rasterizza(bordi, W, w, h))


def test_per_crop_h_true_non_degenere():
    """Il raster moderno ha griglia propria: H_true è una similarità vera, non
    l'identità. Altrimenti l'esperimento premierebbe il 'non muovere niente'."""
    crop_jgw = os.path.join("data/crops", "ribba.jgw")
    img, W_vec = per_crop(_serve(CXF), _serve(crop_jgw), (1024, 1024), risoluzione=0.20)
    H = h_true(read_jgw(crop_jgw), W_vec)
    assert not np.allclose(H, np.eye(3)), "H_true degenerata a identità"
    # la scala è il rapporto delle due risoluzioni
    assert abs(np.hypot(H[0, 0], H[1, 0]) - 0.254453 / 0.20) < 1e-6, H

    # e gli angoli del crop cadono dentro il raster moderno, grazie al margine
    angoli = np.array([[0, 0], [1023, 0], [0, 1023], [1023, 1023]], dtype=np.float64)
    dentro = transform(H, angoli)
    assert (dentro >= 0).all(), dentro
    assert (dentro[:, 0] < img.shape[1]).all() and (dentro[:, 1] < img.shape[0]).all(), dentro


def test_per_crop_stessa_griglia_e_identita():
    """Il caso degenere esiste ed è esplicito: serve a documentare perché non è
    il default."""
    crop_jgw = os.path.join("data/crops", "ribba.jgw")
    _, W_vec = per_crop(_serve(CXF), _serve(crop_jgw), (1024, 1024), stessa_griglia=True)
    assert np.allclose(h_true(read_jgw(crop_jgw), W_vec), np.eye(3), atol=1e-12)


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
