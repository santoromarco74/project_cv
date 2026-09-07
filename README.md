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

## Riprodurre tutto

I dati AdE non stanno nel repository (§5.8), quindi vanno rimessi in `data/raw/`
seguendo `data/README.md`. Fatto quello, l'intera catena — ritagli,
rasterizzazione, esperimenti, figure, tabelle, relazione — sta in un comando:

```bash
python -m scripts.riproduci --controlla   # verifica le precondizioni, non esegue
python -m scripts.riproduci --lista       # le fasi, in ordine, con i tempi
python -m scripts.riproduci               # tutto tranne E3      (~18 min)
python -m scripts.riproduci --con-loftr   # tutto, E3 compreso   (~60 min)
```

Ogni comando viene stampato prima di essere eseguito, e ogni fase dichiara gli
artefatti che deve produrre: se un comando esce con codice 0 ma non ha scritto
quello che doveva, la corsa si ferma lì e dice da dove riprendere
(`--da <fase>`). In coda si verifica che ogni figura citata dalla relazione
esista davvero e che nessun segnaposto di tabella sia rimasto vuoto.

Senza `--con-loftr` la relazione esce completa tranne la figura del capitolo 10,
e lo script lo dice invece di lasciarlo scoprire aprendo il PDF.

## Stato (M1 → M10)

`src/main.py` implementa il contratto di §9. Gli esperimenti passano dalla stessa
pipeline: cambia solo `--matcher`. I singoli comandi, se servono uno alla volta:

```bash
# M1 — genera i 5 ritagli (PNG + world file affiancato)
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
python -m experiments.m5_preprocess --crop tutti --dettaglio ribba

# M6 — E1 completo: la griglia intera, un CSV (~4 min)
python -m experiments.m6_e1_completo --riparti

# M6 — figure e tabelle, generate dal CSV
python -m src.report --csv results/runs.csv

# M7 — rasterizzazione del CXF sulla zona di un crop
python -m src.prep.rasterize --crop ribba --codici 18
python -m experiments.m7_rasterize_check --crop ribba

# M8 — una singola registrazione dalla CLI di §9
python -m src.main --hist data/crops/ribba.png --modern data/crops/ribba_vec.png \
    --matcher sift --preprocess sauvola --model homography \
    --jgw-hist data/crops/ribba.jgw --jgw-modern data/crops/ribba_vec.jgw \
    --soglia-m 2.0 --out-figure results/figures/registrazione.png

# M8 — la griglia cross-domain completa (~6 min), in append al CSV di E1
python -m experiments.m8_e2_griglia
python -m experiments.m8_e2_griglia --diagnosi-ratio   # perché SIFT cede su E2

# M9 — LoFTR (modulo B). Serve torch+kornia e i pesi: vedi weights/README.md
python -m scripts.scarica_pesi
python -m src.main --hist data/crops/ribba.png --modern data/crops/ribba_vec.png \
    --matcher loftr --preprocess sauvola --model similarity
python -m experiments.m9_e3_loftr      # E1 + E2 con LoFTR (~40 min su CPU)

# M10 — figure e tabelle dal CSV, poi la relazione con le tabelle iniettate
python -m src.report --csv results/runs.csv
python -m experiments.m10_passo_passo  # la figura guidata del capitolo 2
python -m experiments.m10_ratio_test   # il ratio test di Lowe, con i rapporti veri
python -m experiments.m10_gradiente    # luminosità, gradiente e descrittore, passo per passo
python -m experiments.m10_ransac_iterazioni  # iterazioni RANSAC vs inlier ratio, con i dati veri
python -m scripts.componi_relazione    # -> relazione/RELAZIONE.md
python -m scripts.relazione_html       # -> RELAZIONE.html, autonomo, da stampare in PDF

# test
python -m tests.test_smoke
```


