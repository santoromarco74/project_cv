# CLAUDE.md — histreg

Registrazione di mappe catastali storiche su cartografia moderna.
Costituzione del progetto. Se una richiesta contraddice un invariante, fermati e segnalalo invece di procedere.

---

## 1. Contesto

Progetto d'esame per **Computer Vision**, Università di Pavia (ING-INF/05, 6 CFU, prof. Luca Lombardi).
Traccia: *"More complex problems... the solution of a computer vision problem. In this case standard librery or tools may be used (opencv, mathlab, Image magick)"*.

**Approvato dal docente nella forma A+B (Comparative).** Due componenti:

- **A — Core-Vision (classico)**: preprocessing + SIFT/ORB + RANSAC, massimizzando l'efficacia dei metodi classici tramite pulizia ottimale del segnale su mappe storiche degradate.
- **B — Comparative (ibrido)**: A, più confronto quantitativo (RMSE, inlier ratio) con un matcher detector-free deep learning (LoFTR).

**B resta comunque isolato dietro interfaccia (I4)**: non perché sia opzionale, ma perché la pipeline non deve dipendere da torch/kornia per funzionare, e A deve poter girare da solo se l'ambiente deep non è disponibile.

Caso di studio: **Comune di Varazze, foglio 49** (`L675_004900`), Originale di Impianto dell'Agenzia delle Entrate.

---

## 2. Invarianti (non negoziabili)

**I1 — Usare `L675_004900`, MAI `L675_00490Z`.**
Vedi §5.1. Sono due sistemi di riferimento diversi. È l'errore che ha già fatto perdere un pomeriggio.

**I2 — La ground truth viene dal JGW, non da annotazione manuale.**
Composizione analitica di due affini note (§5.3). Nessun picking di GCP a mano. La tentazione di annotare punti in QGIS è il segnale che si è sbagliato qualcosa.

**I3 — Il JGW non entra mai nell'algoritmo.**
Serve *solo* a calcolare la trasformazione di riferimento e valutare l'errore. Separazione fisica: `groundtruth.py` è importato da `evaluate.py`, mai da `pipeline.py`. Se un pixel di georeferenziazione filtra nel matching, il risultato è privo di significato.

**I4 — Il modulo deep learning è isolato.**
Interfaccia `Matcher` con `SiftMatcher`, `OrbMatcher`, `LoftrMatcher`. La pipeline non sa quale sta usando. Nessun `import torch` fuori da `matchers/loftr.py`, e quell'import è lazy.

**I5 — Ogni claim di allineamento si verifica a piena risoluzione, mai su immagini ridotte.**
Vedi §5.5. Un errore di 3 m su un overlay ridotto 8× è mezzo pixel: invisibile. Le metriche indirette su questo dato hanno rapporto segnale/rumore pessimo (§5.5) e producono falsi positivi convincenti. Ispezione visiva a 1:500 in QGIS prima di qualunque conclusione.

**I6 — Ogni esperimento produce numeri, non impressioni.**
RMSE in metri, inlier ratio, n° match, success rate, tempo. Su CSV. Nessun "sembra funzionare meglio".

**I7 — Il fallimento è un risultato.**
È plausibile che i descrittori classici cedano sul cross-domain (§7.2). Se succede, si misura e si spiega — non si truccano i parametri finché non esce un numero presentabile. E1 (§8) esiste per garantire risultati validi comunque.

**I8 — Metodo del bisturi.**
Ogni milestone è un programma che gira e produce un artefatto guardabile. Un commit per milestone. Niente refactoring speculativo.

**I9 — Determinismo.**
Seed fisso per RANSAC e per ogni generazione sintetica. Stesso input + stessi parametri = stesso CSV, cifra per cifra.

**I10 — Ritagli, non fogli interi.**
Il foglio 49 è 8489×5648 px. Tiling e gestione memoria non aggiungono nulla al tema d'esame. Crop di ~1000-1200 px per lato.

---

## 3. Stack

- **Python 3.12**, WSL2/Ubuntu
- `opencv-python` (SIFT, ORB, RANSAC, warp), `numpy`, `pandas`, `matplotlib`, `scipy`, `Pillow`
- **Nessun `pyproj`, nessun `geopandas`.** Raster e vettoriale sono già nello stesso sistema (§5.2): non serve riproiettare nulla. Il parser CXF è ~30 righe scritte a mano (§5.4).
- **Modulo B soltanto**: `kornia` + `torch`. Import lazy, isolato (I4).
- `requirements.txt` pinnato. La traccia impone *"if external package are used they should be present in the final version of the project"*: pesi LoFTR (~90 MB) vendorati in `weights/`, non scaricati a runtime. Se troppo pesanti, documentare script di download + checksum.

---

## 4. Struttura del repository

```
histreg/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              L675_004900.jpg/.jgw/.txt/.cxf
│   ├── crops/            ritagli generati
│   └── README.md         provenienza, data di scarico, condizioni d'uso
├── weights/              pesi LoFTR (vendorati, vedi I-stack)
├── src/
│   ├── main.py           CLI
│   ├── io_geo.py         lettura JGW, TXT, parser CXF
│   ├── groundtruth.py    trasformazione di riferimento (I3: solo evaluate)
│   ├── prep/
│   │   ├── crop.py       ritagli dal foglio
│   │   ├── rasterize.py  CXF → raster allineato
│   │   └── synth.py      dataset sintetico (E1)
│   ├── preprocess.py     grayscale, denoise, Otsu/Sauvola, morfologia
│   ├── matchers/
│   │   ├── base.py       interfaccia Matcher
│   │   ├── classic.py    SiftMatcher, OrbMatcher
│   │   └── loftr.py      LoftrMatcher (modulo B, lazy)
│   ├── estimate.py       RANSAC + modelli
│   ├── pipeline.py       orchestrazione (non conosce il JGW)
│   ├── evaluate.py       RMSE, inlier ratio, success rate
│   └── report.py         figure e tabelle
├── experiments/
├── results/
└── relazione/
```

---

## 5. Dati — sezione critica, leggere tutta

### 5.1 I due file CXF: la trappola

Il servizio AdE rilascia due file per il foglio 49, e `_SistemaDiRappresentazione.txt` li distingue così:

```
L675_004900 (nativamente CASSINI-SOLDNER zona G0007 - Forte Diamante - 02)
L675_00490Z (nativamente ROMA40-GAUSS BOAGA DA RILIEVO AEROFOTOGRAMMETRICO OVEST | EPSG:3003)
```

**Usare esclusivamente `L675_004900`.** È il solo che condivide il sistema del JGW dell'Originale di Impianto. `L675_00490Z` è un allegato in Roma40-Gauss Boaga: caricarlo insieme al raster produce due layer lontanissimi e ore di diagnosi sbagliata. È già successo.

### 5.2 Sistema di riferimento

Raster e vettoriale (`004900`) sono entrambi in **Cassini-Soldner zona G0007, origine Forte Diamante**. Nessuna riproiezione, nessun EPSG reale.

Estensioni verificate:
- JPG via JGW: X −31480.04…−29320.25, Y −12715.65…−11278.76 (foglio 8489×5648 px)
- CXF: X −31205…−29548, Y −12469…−11360 (contenuto nel raster ✓)

Gli estremi sono nella convenzione del world file: sono i **centri** dei pixel d'angolo, non gli spigoli (mezzo pixel = 0.127 m di differenza rispetto a come li riporta QGIS). La dimensione del raster si legge sempre dal file, mai da una costante.

**Nota per QGIS**: etichettare entrambi i layer come EPSG:3003 è una bugia utile — geodeticamente falso, ma finché *entrambi* portano la stessa etichetta QGIS non riproietta e le coordinate restano grezze. Non usare quel progetto per misure geodetiche.

### 5.3 Ground truth — verificata

`L675_004900.jgw` è un world file: trasformazione affine pixel → coordinate.

```
0.254453        A  → dimensione pixel x (m)
0.0             D  → rotazione
0.0             B  → rotazione
-0.254453       E  → dimensione pixel y (negativa)
-31480.044315   C  → x del centro del pixel (0,0)
-11278.758056   F  → y del centro del pixel (0,0)
```

`X = A·col + B·row + C`, `Y = D·col + E·row + F`. Risoluzione **0.254453 m/px**, scala 1:2000.

Rasterizzando il CXF nello stesso sistema con trasformazione pixel→CRS nota:

```
H_true = W_moderno⁻¹ ∘ W_storico          (entrambe affini note)
```

**La ground truth è esatta, gratis, senza un solo GCP annotato.** `evaluate.py` confronta `H_est` con `H_true` e produce RMSE **in metri** (errore in px × 0.254453).

**Verificata visivamente in QGIS a 1:500: il vettoriale ricalca l'inchiostro.** Non è un'assunzione.

**Pavimento da dichiarare in relazione.** Il TXT: foglio ricampionato su **76 coppie omologhe**, scarto massimo **1.28 m**, scarto medio **0.56 m**, deviazione standard **0.25 m**. Il JGW non è verità assoluta: è un riferimento con incertezza ~0.5 m. Un RMSE sotto quella soglia misura il rumore del riferimento, non l'errore dell'algoritmo. Va usato come pavimento nelle tabelle.

### 5.4 Parser CXF — struttura decodificata dal file

Formato testuale, una riga per campo, CRLF, latin-1. Nessuna libreria: ~30 righe a mano.

```
MAPPA
  <nome>            L675_004900
  <denom. scala>    2000.000
BORDO
  <nome>            "1", "1015"… = n° particella | "ACQUA" | "STRADA" | nome mappa (bordo foglio)
  <codice>          18 = particella (438) · 12 = acqua/strada/bordo (432) · 25 = altro (1)
  <angolo>          0.000
  <x,y>             punto di etichetta
  <x,y>             ripetuto
  <nflag>           ⚠ numero di indici extra che seguono N
  <N>               numero di vertici
  [nflag interi]    indici di cambio tratto (tipico su STRADA: tratteggio)
  <x,y> × N         vertici; poligono chiuso in 838/871 casi
```

**⚠ Il `nflag` è la trappola.** Vale 0 in 838 record su 871, ma 1, 2 o 5 nei restanti 33. Se lo si ignora e si leggono le coordinate subito dopo `N`, il parser sfasa su quei 33 record e l'estensione del foglio esplode da ~1.6 km a ~31 km. Il sintomo è coordinate positive (tipo `68`, `82`) dove dovrebbero essere tutte negative. **Test obbligatorio**: dopo il parsing, asserire che ogni coordinata cada dentro l'estensione del JGW.

Record `LINEA`, `TESTO`, `SIMBOLO`, `FIDUCIALE` esistono e possono comparire con backslash di continuazione (`LINEA\`). Per il progetto si ignorano: servono solo i `BORDO`.

Filtro per la rasterizzazione: **codice 18** = particelle (il segnale). **Codice 12** = acque e strade. **Codice 25** = un solo poligono, si scarta. Ablation prevista: solo 18 / 18+12 / solo 12 — le strade nello storico sono disegnate come *fasce* a doppio bordo, nel vettoriale sono poligoni: la densità di tratto differisce proprio dove SIFT cerca struttura.

### 5.5 Lezione metodologica — vincolante per M6

Durante la preparazione dei dati si è tentato di misurare l'allineamento con correlazione incrociata e chamfer matching su distance transform. **Entrambi hanno prodotto falsi positivi convincenti**: shift ottimali incoerenti fra zone (−238, −330, +11 px), ottimi saturati sui bordi della ricerca, e un profilo apparentemente sistematico che confermava un disallineamento inesistente.

La diagnosi: **linee tirate a caso sul foglio ottengono l'11.5% dei pixel entro 2 px dall'inchiostro; il vettoriale vero il 16-20%.** Con quel rapporto segnale/rumore nessuna di quelle metriche discrimina. La carta ha inchiostro sparso ovunque (testi, simboli, tratteggi, grana, macchie) e la superficie di correlazione su disegni al tratto è piatta e multi-picco.

**Regole che ne derivano:**
- Correlazione e chamfer su questi dati **non sono metriche di valutazione**. L'unica metrica è l'RMSE su checkpoint contro `H_true` (§7.4).
- Prima di ogni conclusione sull'allineamento: ispezione a piena risoluzione o QGIS a 1:500.
- Ogni misura indiretta va accompagnata da un **baseline casuale**. Se il segnale non batte nettamente il caso, la misura si butta.
- Cinque minuti in QGIS hanno battuto un'ora di misure indirette.

### 5.6 Selezione dei crop

**Zone da evitare:**
- **Territorio di Celle Ligure** (fascia bianca a ovest): fuori giurisdizione, vuoto in entrambi. Nessun contenuto da allineare.
- **Ente urbano** (campiture rosa sulla costa, zona Perno/Cucco/centro): nel catasto terreni le zone urbane sono rimandate al catasto urbano, quindi il vettoriale **non le copre**, mentre l'impianto ha tratto fittissimo. Domain gap massimo per il motivo sbagliato — non misura nulla di interessante.

**Crop consigliati** (dove il vettoriale copre a pieno, fascia collinare):

| nome | pixel (x0,y0,w,h) | coordinate X | coordinate Y | m |
|---|---|---|---|---|
| `tassarole` | 1500, 300, 1024×1024 | −31098…−30838 | −11616…−11355 | 261×261 |
| `cannei` | 2900, 200, 1200×1000 | −30742…−30437 | −11584…−11330 | 305×254 |
| `ribba` | 3300, 600, 1024×1024 | −30640…−30380 | −11692…−11431 | 261×261 |
| `vedra` | 4200, 2000, 1024×1024 | −30411…−30151 | −12048…−11788 | 261×261 |
| `aspera` | 5600, 2600, 1024×1024 | −30055…−29795 | −12201…−11940 | 261×261 |

Documentare offset e dimensioni in `data/crops/README.md` per la riproducibilità.

### 5.7 Contenuto: cosa il vettoriale NON è

Il CXF è la **cartografia vigente**, non la digitalizzazione dell'impianto. La geometria discende dall'impianto ma ha un secolo di aggiornamenti: nel CXF ci sono particelle con numeri a quattro cifre (`1015`, `1026`, `1041`) che sull'impianto non esistono, dove la numerazione arriva a ~336.

**Conseguenza per E2**: parte delle linee vettoriali non ha corrispondenza nell'inchiostro. Non è errore di allineamento, è storia. Va detto in relazione e va tenuto presente nell'interpretare gli inlier ratio bassi.

**Conseguenza per l'inquadramento**: i dati AdE forniscono georeferenziazione nota, e *proprio per questo* permettono di valutare quantitativamente un metodo che altrove — scansioni d'archivio, catasti preunitari, mappe senza world file — dovrebbe operare senza riferimento. È un dataset di **validazione**, non di applicazione. Scrivere in relazione "serve ad allineare mappe storiche disorientate" sarebbe falso *su questi dati*. Dirlo prima che lo chieda il docente.

### 5.8 Licenza

Le scansioni vengono dal servizio AdE previa accettazione delle condizioni d'uso. **Non committare JPG e CXF in un repo pubblico** finché le condizioni non sono state lette e confermate compatibili. `data/raw/` in `.gitignore`; `data/README.md` documenta comune, foglio, data di scarico, così il set è ricostruibile.

---

## 6. Pipeline

```
storico (raster)  ─┐
                   ├→ preprocess → matcher → corrispondenze → RANSAC → H_est → warp
moderno (raster)  ─┘                                                      │
                                                                          ▼
                                              H_true (da JGW)  →  evaluate → RMSE, inlier ratio
```

`pipeline.py` riceve due immagini e restituisce `H_est` + metadati. Non conosce il JGW (I3). `evaluate.py` è l'unico che vede entrambi.

---

## 7. Specifiche algoritmiche

### 7.1 Preprocessing (`preprocess.py`)
Cuore dell'approccio A: *"massimizzare l'efficacia dei metodi classici tramite una pulizia ottimale del segnale"*. Ogni stadio attivabile da CLI: l'ablation è un risultato.

- Grayscale
- Denoise: mediana, bilaterale
- **Binarizzazione**: Otsu globale (baseline; fallirà su carta ingiallita con gradiente di illuminazione — risultato atteso, non "aggiustarlo") vs **Sauvola** locale, `T = m·[1 + k·(s/R − 1)]`, R=128, k=0.2, finestra 25, con immagini integrali (O(1) per pixel).
- **Morfologia**: apertura (via il pepe), chiusura (salda le interruzioni del tratto). SE 3×3, iterazioni da CLI.
- Rimozione componenti troppo piccole: le cifre scritte a mano sono rumore per il matching strutturale.
- **CLAHE come alternativa alla binarizzazione**: SIFT lavora su grayscale, non su binaria. Binarizzare potrebbe distruggere proprio la texture su cui SIFT si basa. **Testare entrambe le strade** — è una delle domande sperimentali, non un dettaglio.

### 7.2 Matching (`matchers/`)

```python
class Matcher(Protocol):
    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """→ (punti_a Nx2, punti_b Nx2, metadati)"""
```

- `SiftMatcher`: SIFT + BFMatcher/FLANN + **ratio test di Lowe** (0.7-0.8 da CLI)
- `OrbMatcher`: ORB + BFMatcher Hamming + cross-check
- `LoftrMatcher` (**modulo B**): kornia, pesi `outdoor`, import lazy

**Rischio principale del progetto.** Il cross-domain è brutale: disegno a mano su carta invecchiata vs raster vettoriale di linee pulite. Entrambi *line drawings* quasi privi di texture — il caso peggiore per descrittori a blob/corner. È plausibile che SIFT produca pochissimi match corretti su E2. Non è un bug: è la risposta alla domanda di ricerca, **purché misurata** (inlier ratio, RMSE, distribuzione delle distanze fra descrittori) e non solo constatata. E1 garantisce risultati solidi comunque.

### 7.3 Stima (`estimate.py`)
RANSAC (`cv2.findHomography` / `estimateAffinePartial2D`), seed fisso (I9). Modelli: similarità (4 gdl), affine (6), omografia (8). Parametri da CLI: soglia reprojection, iterazioni, confidence. Sotto i 4 match: fallimento pulito (`success=False`, riga CSV comunque scritta), non eccezione opaca.

### 7.4 Valutazione (`evaluate.py`)
Su griglia regolare di **checkpoint** nell'immagine storica (10×10, bordi esclusi):

- `RMSE_m` = √(media(‖H_est·p − H_true·p‖²)) × 0.254453
- `inlier_ratio`, `n_matches`, `n_inliers`
- `success` = stima riuscita **e** RMSE sotto soglia dichiarata
- `time_ms` per fase

Una riga CSV per (esperimento, crop, matcher, preprocessing, modello). Le tabelle della relazione sono aggregazioni del CSV, non numeri ricopiati a mano.

---

## 8. Design sperimentale

### E1 — Sintetico, stesso dominio (`prep/synth.py`)
Crop storico vs **se stesso trasformato con H nota** (rotazione, scala, traslazione, omografia lieve; degradazione opzionale: rumore, blur, contrasto).
Ground truth esatta per costruzione, incertezza zero.
Isola il matcher dal domain gap. Qui tutto **deve** funzionare: stabilisce il tetto di prestazione e valida la correttezza della pipeline. Se E1 fallisce, il bug è nel codice.
Variabile indipendente: ampiezza trasformazione, livello degradazione. Curva RMSE vs degradazione = prima figura.

**Non saltare E1.** È ciò che rende il progetto presentabile qualunque cosa dia E2, e ciò che distingue "non funziona" da "l'ho scritto male".

### E2 — Cross-domain reale
Crop storico vs raster del CXF, stesso sistema.
Ground truth da JGW (§5.3), pavimento ~0.5 m.
Griglia: {SIFT, ORB} × {grayscale+CLAHE, Sauvola, Sauvola+morfologia} × {similarità, affine, omografia} × 5 crop (§5.6).
Interpretare gli inlier ratio alla luce di §5.7: parte delle linee vettoriali non ha corrispondenza storica.

### E3 — Comparative (LoFTR)
LoFTR su E1 ed E2, stesse metriche, stessi crop. Confronto anche sul costo computazionale (tempo per registrazione, CPU vs eventuale GPU). È il cuore della parte comparativa approvata: il confronto ha valore solo se classico e neurale girano nella **stessa** pipeline, cambiando il solo `--matcher`.

---

## 9. Contratto CLI

```
python -m src.main --hist <crop.png> --modern <raster.png> [opzioni]

  --hist <path>             immagine storica (obbligatorio)
  --modern <path>           immagine di riferimento (obbligatorio)
  --matcher <sift|orb|loftr>              default: sift    (loftr = modulo B)
  --preprocess <none|clahe|otsu|sauvola>  default: sauvola
  --morph-close <n>         default 0
  --morph-open <n>          default 0
  --model <similarity|affine|homography>  default: homography
  --ratio <float>           ratio test di Lowe, default 0.75
  --ransac-thresh <float>   px, default 3.0
  --seed <int>              default 42
  --jgw-hist <path>         world file storico   (solo evaluate)
  --jgw-modern <path>       world file moderno   (solo evaluate)
  --out-csv <path>          default results/runs.csv (append)
  --out-figure <path>       overlay warp + match
  --verbose
```

Senza `--jgw-*` la pipeline gira e produce `H_est`, semplicemente non calcola RMSE. È la prova architetturale di I3.

---

## 10. Milestone

| # | Obiettivo | Criterio di accettazione |
|---|---|---|
| **M1** | I/O + crop | Legge JPG e JGW, genera i 5 crop di §5.6. `io_geo.py` ritorna l'affine corretta, verificata a mano su un pixel noto. |
| **M2** | Parser CXF | 871 BORDO, 438 codice 18, gestione `nflag`. **Assert**: ogni coordinata dentro l'estensione JGW. |
| **M3** | Ground truth | `groundtruth.py` compone le due affini. Test: punto trasformato avanti e indietro torna a sé entro 1e-9. |
| **M4** | E1 sintetico | `synth.py` genera coppie con H nota. SIFT recupera H con RMSE ≈ 0. **Se qui non torna, si ferma tutto.** |
| **M5** | Preprocessing | Otsu vs Sauvola vs CLAHE, morfologia. Figure affiancate sui crop reali. |
| **M6** | E1 completo | Curva RMSE vs degradazione, SIFT vs ORB. Primo CSV vero. |
| **M7** | Rasterizzazione CXF | Vettoriale → raster allineato. **Verifica visiva a piena risoluzione** (I5), non metriche indirette (§5.5). |
| **M8** | E2 cross-domain | Griglia completa. Qualunque sia il risultato, è il risultato. |
| **M9** | E3 (LoFTR) | LoFTR via kornia su E1 ed E2. Stesse metriche e stessi crop del classico, tabella comparativa affiancata. |
| **M10** | Relazione | Tabelle e figure generate da `report.py` dal CSV. |

---

## 11. Fuori scope

- PostGIS, API, GUI, viewer, Foliarium
- OCR dei numeri di particella
- Linking particella ↔ partita
- Mosaicatura di più fogli, tiling, immagini oltre ~2000 px per lato
- Fine-tuning o training (LoFTR pretrained o niente)
- Studi con utenti, tempo risparmiato all'operatore
- Docker, CI, packaging
- `pyproj`, riproiezioni, EPSG reali (§5.2)

Materiale da tesi: annotare in `IDEE_TESI.md` e proseguire.

---

## 12. Relazione

La traccia chiede: *"a relation is requested with the modality of use of the application"*.

1. Problema: cos'è un Originale di Impianto, perché registrarlo è difficile
2. Dati: foglio 49 Varazze, i due CXF e perché uno solo è utilizzabile (§5.1), crop
3. **Ground truth dal world file**: composizione analitica, e la sua incertezza dichiarata (0.56 m medi, 1.28 m max, 76 coppie omologhe)
4. Pipeline
5. Preprocessing: Otsu vs Sauvola vs CLAHE, effetto della morfologia — con immagini
6. Matching: SIFT, ORB, ratio test; RANSAC e modelli
7. E1: risultati sintetici, curva RMSE vs degradazione
8. E2: risultati cross-domain — **incluso il fallimento, se c'è, con l'analisi del perché** (line drawings privi di texture, descrittori a blob)
9. E3: confronto classico vs LoFTR — RMSE, inlier ratio, robustezza al domain gap, costo computazionale
10. **Modalità d'uso**: tutte le opzioni CLI con esempi eseguibili — la traccia lo chiede esplicitamente, non è un'appendice
11. Limiti: incertezza del riferimento, divergenza di contenuto (§5.7), inquadramento come validazione e non applicazione

Un progetto che misura e spiega perché i descrittori classici cedono sul cross-domain vale più di uno che mostra solo il caso riuscito. Ma serve E1 a dimostrare che la pipeline è corretta.
