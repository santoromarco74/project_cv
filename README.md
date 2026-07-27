# histreg

Registrazione di mappe catastali storiche su cartografia moderna.
Progetto d'esame di Computer Vision (Università di Pavia, 6 CFU).

Confronto fra matching classico (SIFT/ORB + RANSAC) e detector-free deep
learning (LoFTR) nell'allineare un Originale di Impianto catastale disegnato
a mano sulla corrispondente cartografia vettoriale vigente.

## Dati

Il progetto usa il foglio 49 del Comune di Varazze (`L675_004900`), scaricato
dal servizio "Consultazione dei fogli di mappa catastale" dell'Agenzia delle
Entrate (accesso con SPID/CIE/CNS, gratuito).

**I file cartografici NON sono inclusi nel repository** (soggetti alle
condizioni d'uso del servizio AdE). Per ricostruire il dataset vedere
`data/README.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python -m src.main --hist data/crops/ribba.png --modern data/crops/ribba_vec.png \
    --matcher sift --preprocess sauvola --model homography \
    --jgw-hist data/raw/L675_004900.jgw --out-csv results/runs.csv
```

Elenco completo delle opzioni: `python -m src.main --help`.

## Stato (M1 → M5)

`src/main.py` arriverà con le milestone successive. Quello che gira oggi:

```bash
# M1 — genera i 5 ritagli di CLAUDE.md §5.6 (PNG + world file affiancato)
python -m src.prep.crop

# M2 — statistiche e verifiche del parser CXF
python -m src.io_geo

# M2 — figure di verifica (overview del vettoriale, inquadramento sul raster)
python -m experiments.m2_cxf_check

# M3 — trasformazione di riferimento fra due raster georeferenziati
python -m src.groundtruth --jgw-hist data/crops/tassarole.jgw \
                          --jgw-modern data/crops/ribba.jgw

# M4 — coppia sintetica E1 con H nota
python -m src.prep.synth --crop data/crops/ribba.png --rot 15 --scala 1.2

# M4 — E1: SIFT recupera H su un ventaglio di trasformazioni
python -m experiments.m4_e1_smoke
python -m experiments.m4_e1_smoke --crop tutti --matcher orb --modello affine

# M5 — preprocessing: Otsu vs Sauvola vs CLAHE, morfologia
python -m src.preprocess --crop data/crops/ribba.png --preprocess sauvola --morph-close 1
python -m experiments.m5_preprocess --crop tutti

# test
python -m tests.test_smoke
```

## Struttura

Vedere `CLAUDE.md` per la specifica completa del progetto (architettura,
invarianti, milestone, design sperimentale).
