# Registrazione di mappe catastali storiche su cartografia moderna

**Progetto d'esame di Computer Vision — Università di Pavia**
Caso di studio: Comune di Varazze, foglio 49 (`L675_004900`), Originale di Impianto dell'Agenzia delle Entrate.

> Questo documento è generato da `scripts/componi_relazione.py`: tutte le tabelle
> sono aggregazioni di `results/runs.csv` calcolate al momento della
> composizione, non numeri ricopiati a mano. Le figure si rigenerano con i
> comandi indicati in §10.

---

## 1. Il problema

Un **Originale di Impianto** è il foglio di mappa disegnato a mano al momento
della formazione del catasto terreni, nella prima metà del Novecento. È un
disegno al tratto su carta: confini di particella, corsi d'acqua, strade, numeri
scritti a penna, e un secolo di ingiallimento, macchie e pieghe.

Registrare un foglio di questo tipo su cartografia moderna significa stimare la
trasformazione geometrica che porta i pixel della scansione sulle coordinate
della cartografia vigente. È difficile per una ragione precisa e poco intuitiva:
**entrambe le immagini sono disegni al tratto quasi privi di texture**. I
descrittori classici come SIFT e ORB nascono per agganciare blob e angoli in
immagini fotografiche; su due reticoli di linee sottili hanno poco a cui
aggrapparsi. A questo si aggiunge un divario di dominio brutale — inchiostro
sbiadito su carta invecchiata contro linee vettoriali nere su fondo bianco — e
il fatto che i due documenti, come si vedrà in §11, **non rappresentano
esattamente la stessa realtà**.

Il progetto affronta il problema in due componenti, secondo la traccia approvata:

- **A — Core-Vision classico**: preprocessing accurato, SIFT/ORB, RANSAC, con
  l'obiettivo di spingere al massimo i metodi classici tramite una pulizia
  ottimale del segnale.
- **B — Comparativo**: confronto quantitativo con un matcher *detector-free*
  di deep learning (LoFTR), nella stessa identica pipeline.

---

## 2. I dati

### 2.1 Il foglio e il vettoriale

Dal servizio "Consultazione dei fogli di mappa catastale" dell'Agenzia delle
Entrate si ottengono, per il foglio 49 di Varazze:

| file | contenuto |
|---|---|
| `L675_004900.jpg` | scansione dell'Originale di Impianto, **8489 × 5648 px** |
| `L675_004900.jgw` | world file: la georeferenziazione della scansione |
| `L675_004900.txt` | metadati del ricampionamento |
| `L675_004900.cxf` | cartografia **vigente** in formato vettoriale, stesso sistema |

Raster e vettoriale sono entrambi in **Cassini-Soldner zona G0007, origine Forte
Diamante**. Non è un EPSG standard e non serve riproiettare nulla: le coordinate
dei due file sono già confrontabili. Il progetto non usa `pyproj` né `geopandas`.

### 2.2 La trappola dei due CXF

Il servizio rilascia **due** file vettoriali per lo stesso foglio, e il file
`_SistemaDiRappresentazione.txt` li distingue così:

```
L675_004900 (nativamente CASSINI-SOLDNER zona G0007 - Forte Diamante - 02)
L675_00490Z (nativamente ROMA40-GAUSS BOAGA DA RILIEVO AEROFOTOGRAMMETRICO OVEST | EPSG:3003)
```

Solo `L675_004900` condivide il sistema di riferimento del JGW. Caricare
`L675_00490Z` insieme al raster produce due strati a chilometri di distanza e
una diagnosi che parte dal posto sbagliato. È un errore che costa un pomeriggio
e che nessun messaggio d'errore segnala: i file si aprono entrambi senza
problemi, semplicemente descrivono posti diversi.

### 2.3 Il parser CXF e il campo che sfasa tutto

Il CXF è un formato testuale elementare: un campo per riga, CRLF, codifica
latin-1. Il parser è scritto a mano in una trentina di righe, senza librerie.
La struttura di un record `BORDO` (un poligono) è:

```
BORDO
  <nome>       "1", "1015"… = n° particella | "ACQUA" | "STRADA" | nome mappa
  <codice>     18 = particella · 12 = acqua/strada/bordo · 25 = altro
  <angolo>
  <x> <y>      punto di etichetta (su due righe)
  <x> <y>      ripetuto
  <nflag>      ⚠ numero di indici extra che seguono
  <N>          numero di vertici
  [nflag interi]
  <x> <y> × N  i vertici
```

**Il campo `nflag` è la trappola.** Vale 0 in 838 record su 871, ma 1, 2 o 5 nei
restanti 33. Chi lo ignora e legge le coordinate subito dopo `N` ottiene un
parser che funziona benissimo per il 96% dei record e sfasa sul restante 4%: il
sintomo è che l'estensione del foglio passa da ~1.6 km a ~31 km, con coordinate
positive dove dovrebbero essere tutte negative.

Il progetto ha tre difese contro questo errore, tutte automatiche:

1. **Assert di estensione**: dopo il parsing, ogni coordinata deve cadere dentro
   l'estensione del raster calcolata dal JGW.
2. **Sentinella di segno**: nessuna coordinata può essere positiva, perché il
   foglio sta interamente nel terzo quadrante del sistema.
3. **Regressione contro un export indipendente**: il parser riproduce, entro
   1e-9, un GeoJSON esportato in precedenza da QGIS — 871 feature, stessi
   attributi, stesse coordinate.

Un dettaglio del formato merita una nota, perché è lo stesso tipo di insidia:
il nome di un `BORDO` è testo libero, e nel foglio 49 esistono record chiamati
`ACQUA`, `STRADA`, `L675_004900` e persino `A`. Un parser che cercasse la
stringa `BORDO` nel file invece di leggerlo sequenzialmente rischierebbe di
scambiare un nome per una parola chiave.

### 2.4 I ritagli

Il foglio intero è troppo grande per essere utile allo scopo: tiling e gestione
della memoria non aggiungono nulla al tema. Si lavora su cinque ritagli di circa
1000-1200 px per lato, scelti nella fascia collinare dove il vettoriale copre
per intero il contenuto disegnato.

Due zone sono state **escluse deliberatamente**:

- il **territorio di Celle Ligure**, fascia bianca a ovest: fuori giurisdizione,
  vuoto in entrambe le rappresentazioni;
- l'**ente urbano** sulla costa: nel catasto terreni le zone urbane sono
  rimandate al catasto urbano, quindi il vettoriale non le copre, mentre
  l'impianto ha lì il tratto più fitto. È il divario di dominio massimo, ma per
  la ragione sbagliata: non misurerebbe la capacità del metodo, solo l'assenza
  di un riferimento.

<!-- TABELLA: crop -->

![I cinque ritagli](../results/figures/m1_crops.png)

Ogni ritaglio è salvato come PNG accompagnato dal **proprio world file**,
composto per traslazione dell'origine pixel:

```
C' = C + A·x0 + B·y0        F' = F + D·x0 + E·y0
```

---

## 3. La ground truth: esatta e gratuita

Questo è il punto metodologico su cui poggia tutto il resto.

Il world file `L675_004900.jgw` è una trasformazione affine da pixel a
coordinate:

```
0.254453        A → dimensione pixel x (m)
0.0             D → rotazione
0.0             B → rotazione
-0.254453       E → dimensione pixel y (negativa)
-31480.044315   C → x del centro del pixel (0,0)
-11278.758056   F → y del centro del pixel (0,0)
```

cioè `X = A·col + B·row + C` e `Y = D·col + E·row + F`, con risoluzione
**0.254453 m/px** (scala 1:2000).

Rasterizzando il CXF su una griglia con trasformazione pixel→CRS anch'essa nota,
si ottengono due affini note, e la trasformazione di riferimento è la loro
composizione:

```
H_true = W_moderno⁻¹ ∘ W_storico
```

**Non è stato annotato un solo punto di controllo a mano.** La ground truth è
analitica, esatta per costruzione, e `evaluate.py` la usa per produrre l'RMSE in
metri su una griglia di checkpoint. La correttezza della composizione è
verificata da un test: un punto trasformato avanti e indietro torna su sé stesso
entro **1.1e-13 px**, contro la soglia dichiarata di 1e-9.

### 3.1 L'incertezza del riferimento, dichiarata

Il JGW non è verità assoluta. I metadati del foglio dicono che la scansione è
stata ricampionata su **76 coppie omologhe**, con:

| grandezza | valore |
|---|---|
| scarto massimo | 1.28 m |
| scarto medio | 0.56 m |
| deviazione standard | 0.25 m |

Questo definisce un **pavimento**: un RMSE inferiore a ~0.5 m non misura più
l'errore dell'algoritmo, misura il rumore del riferimento. Va tenuto presente
ogni volta che si legge una cifra nelle tabelle di §8.

### 3.2 Separazione fisica fra algoritmo e riferimento

Un rischio serio in un lavoro di questo tipo è che l'informazione di
georeferenziazione filtri nella pipeline che dovrebbe stimarla — a quel punto il
risultato non significherebbe più nulla. Il progetto lo impedisce per costruzione:

- `pipeline.py` riceve due array di pixel e non importa né `io_geo` né
  `groundtruth`;
- solo `evaluate.py` vede entrambi i mondi;
- un **test statico** analizza l'AST dei moduli dell'algoritmo e fallisce se uno
  di essi importa la georeferenziazione;
- la CLI, invocata **senza** i parametri `--jgw-*`, produce comunque `H_est` e
  semplicemente non calcola l'RMSE. È la prova architetturale, eseguibile in
  qualsiasi momento.

---

## 4. La pipeline

```
storico (raster)  ─┐
                   ├→ preprocess → matcher → corrispondenze → RANSAC → H_est
moderno (raster)  ─┘                                                    │
                                                                        ▼
                                        H_true (dai world file) → evaluate → RMSE
```

Il preprocessing si applica **a entrambe le immagini allo stesso modo**: fa parte
della pipeline, non è un ritocco di una delle due. Il matcher è scelto per nome
da una fabbrica, e la pipeline non sa quale stia usando: è ciò che rende
confrontabili il classico e il neurale cambiando un solo parametro.

Sotto le 4 corrispondenze la stima fallisce in modo pulito — `success=False` con
il motivo — e la riga di CSV viene scritta lo stesso. Un esperimento che non
produce una stima ha comunque prodotto un dato.

Ogni esecuzione è **deterministica**: il seed di RANSAC è fissato con
`cv2.setRNGSeed`, ogni generazione sintetica passa da un generatore seminato.
Eseguendo due volte la stessa griglia si ottengono CSV identici cifra per cifra
su ogni colonna tranne i tempi.

---

## 5. Preprocessing

Il preprocessing è il cuore della componente classica. Ogni stadio è attivabile
da riga di comando, perché l'ablazione è un risultato e non un parametro da
fissare una volta per tutte.

- **Otsu** — soglia globale, la baseline.
- **Sauvola** — soglia locale, `T = m·[1 + k·(s/R − 1)]` con R=128, k=0.2,
  finestra 25, calcolata con immagini integrali: media e varianza di ogni
  finestra costano O(1) per pixel, quindi la dimensione della finestra non incide
  sul tempo. La correttezza dell'ottimizzazione è verificata confrontandola
  pixel per pixel con una versione ingenua O(finestra²) — è il tipo di errore
  che non si vedrebbe mai a occhio sull'immagine.
- **CLAHE** — equalizzazione adattiva, **alternativa** alla binarizzazione e non
  suo preludio: SIFT lavora su grayscale, e binarizzare potrebbe distruggere
  proprio la texture su cui si basa. Quale delle due strade convenga è una
  domanda sperimentale, non un dettaglio implementativo.
- **Morfologia** — apertura (toglie il pepe) e chiusura (salda le interruzioni
  del tratto, frequenti perché il pennino stacca), applicate all'inchiostro in
  primo piano. L'ordine è apertura → chiusura: chiudere per primo salderebbe il
  pepe al tratto rendendolo non più rimovibile.

![Confronto dei preprocessing](../results/figures/m5_preprocess_ribba.png)

Le figure hanno una riga di dettagli **a piena risoluzione**, perché su una
binarizzazione il ridimensionamento inganna sempre: un tratto interrotto e uno
saldato diventano indistinguibili a un quarto di scala.

### 5.1 Quanto costa binarizzare

Contando i keypoint SIFT rilevati su ciascuna variante, rapportati al grayscale:

| variante | keypoint (rapporto sul grayscale) |
|---|---|
| CLAHE | 1.17× – 1.55× |
| Sauvola | 0.75× – 0.79× |
| Sauvola + chiusura | 0.68× – 0.77× |
| Otsu | 0.61× – 0.72× |

Il dato è consistente su tutti e cinque i ritagli: **binarizzare costa fra il 21
e il 39% dei keypoint, CLAHE ne aggiunge fra il 17 e il 55%**. Non è ancora la
risposta — quella è l'RMSE di §7 e §8 — ma indica la direzione.

### 5.2 Una previsione verificata invece che assunta

L'impostazione del progetto prevedeva che Otsu fallisse "su carta ingiallita con
gradiente di illuminazione". **Sui ritagli reali questo non accade**: sono
illuminati in modo uniforme e Otsu si comporta benissimo.

Invece di dare per buona la previsione o di dichiararla smentita, la si è messa
alla prova in modo controllato, aggiungendo un gradiente di illuminazione
sintetico di intensità crescente:

| gradiente | Otsu | Sauvola |
|---|---|---|
| 0.0 | 8.42% | 7.09% |
| 0.3 | 8.99% | 6.94% |
| 0.6 | **46.58%** | 6.79% |
| 0.9 | **53.25%** | 6.63% |

(percentuale di pixel classificati come inchiostro, ritaglio `ribba`)

![Otsu contro Sauvola sotto gradiente](../results/figures/m5_gradiente_ribba.png)

A gradiente 0.6 la soglia globale collassa — classifica come inchiostro metà
immagine, cioè la carta scurita — mentre Sauvola resta piatta lungo tutta la
scala. **La previsione è corretta in generale, ma il suo antecedente su questi
dati non si verifica.** Sauvola è più robusta, e lo si dimostra; su questo foglio
specifico quella robustezza non serve, e anche questo è un dato.

---

## 6. Matching e stima

L'interfaccia comune è minima: due immagini entrano, due insiemi di punti
corrispondenti e un dizionario di metadati escono.

- **SIFT** + BFMatcher + **ratio test di Lowe** (0.75 di default).
- **ORB** + BFMatcher Hamming + **cross-check**.
- **LoFTR** (componente B) — matcher *detector-free*: non cerca keypoint
  ripetibili per poi descriverli, ma mette in corrispondenza direttamente due
  griglie dense di feature con attenzione incrociata. È esattamente ciò che
  dovrebbe aiutare dove i rilevatori a blob soffrono.

La stima usa RANSAC su tre modelli di complessità crescente: **similarità**
(4 gradi di libertà), **affine** (6), **omografia** (8). Il confronto fra modelli
non è un dettaglio: come si vedrà in §8, su questi dati è il fattore che decide
fra successo e fallimento.

---

## 7. E1 — esperimento sintetico, stesso dominio

Il primo esperimento confronta un ritaglio con **sé stesso trasformato con una
`H` nota**: rotazione, scala, traslazione, omografia lieve, più una degradazione
radiometrica opzionale (rumore, sfocatura, contrasto, luminosità). La ground
truth è esatta per costruzione e il divario di dominio è assente.

Serve a due cose, entrambe indispensabili: stabilire il **tetto di prestazione**
dei matcher, e verificare che la pipeline sia **corretta**. Se E1 fallisce, il
problema è nel codice.

<!-- TABELLA: e1 -->

![RMSE contro degradazione](../results/figures/m6_rmse_vs_degradazione.png)

### 7.1 Il tetto di prestazione

In assenza di degradazione tutti e tre i matcher recuperano la trasformazione
con errore ampiamente sub-pixel — nel caso peggiore su 35 combinazioni di
ritaglio e trasformazione, SIFT si ferma a 0.276 px (0.070 m). Su un riferimento
il cui pavimento è ~0.5 m, questo è due ordini di grandezza sotto: è il segnale
che la pipeline è corretta.

Il caso geometricamente più difficile è sempre la **rotazione a 45°**, su tutti i
ritagli: è il costo dell'interpolazione del warp e della quantizzazione
dell'orientamento dei descrittori, non un difetto.

### 7.2 La rottura è un precipizio, non una discesa

La scala di degradazione è stata estesa oltre il valore nominale 1.0 proprio
perché a quel livello nulla si rompeva ancora. Il comportamento reale è netto
(SIFT, senza preprocessing):

| degradazione | RMSE mediano | successo | corrispondenze |
|---|---|---|---|
| 0.00 | 0.108 px | 100% | 2085 |
| 0.75 | 0.224 px | 60% | 487 |
| 1.00 | 0.344 px | 60% | 389 |
| 1.10 | 3.168 px | 40% | 215 |
| 1.40 | 9.009 px | 0% | 9 |

Con passi di 0.2 fermi a 1.0, la curva avrebbe mostrato un degrado dolce che non
esiste. Il campionamento è stato infittito sopra 1.0 per catturare la soglia.

### 7.3 Nota sull'aggregazione

Oltre la soglia di rottura RANSAC **restituisce comunque una `H`**, ma sbagliata
di migliaia di pixel: il massimo osservato è 12615 px. Una sola stima di questo
tipo trascina la media dell'intero gruppo, producendo "RMSE medi" di centinaia di
pixel che non descrivono né i casi buoni né i cattivi.

Per questo le tabelle riportano la **mediana** accompagnata dal **tasso di
successo**, che è la grandezza che descrive i casi cattivi. Le curve usano
mediana, banda interquartile e scala logaritmica.

Le tabelle di §8 e §9 aggregano in due modi diversi, e le colonne lo dicono nel
nome: `rmse_m_mediano` è la mediana su **tutte** le prove — grande di proposito,
perché su E2 una configurazione può avere zero successi e allora "l'errore sulle
riuscite" non esisterebbe — mentre `rmse_m_mediano_ok` è la mediana sulle sole
prove riuscite. Confrontando le due tabelle si trovano quindi cifre leggermente
diverse per la stessa configurazione: non è un'incoerenza, sono due domande
diverse.

---

## 8. E2 — cross-domain reale

Il secondo esperimento è quello vero: ritaglio storico contro **raster del CXF**,
nello stesso sistema di riferimento.

### 8.1 La rasterizzazione e la sua verifica

I poligoni del CXF sono disegnati come **polilinee e non come campiture**: sul
foglio d'impianto il confine è un tratto, e riempire i poligoni cambierebbe il
tipo di struttura che il matcher vede.

Una scelta merita di essere motivata: **la griglia del raster vettoriale non è
quella del ritaglio storico**. Ha risoluzione propria (0.20 m/px contro
0.254453) e un margine di 20 m. Se le due griglie coincidessero, `H_true`
sarebbe l'identità, e l'esperimento premierebbe qualunque metodo con un bias
verso il "non muovere niente" — un risultato che sembrerebbe ottimo e non
significherebbe nulla. Con griglie diverse `H_true` è una similarità vera, con
fattore di scala 1.272265.

La verifica dell'allineamento è **visiva e a piena risoluzione**, non metrica.

![Vettoriale sul crop storico](../results/figures/m7_ribba_vec.png)

Il motivo di questa scelta è documentato in §11.2: su questi dati le metriche
indirette di allineamento producono falsi positivi convincenti.

### 8.2 Risultati

<!-- TABELLA: e2 -->

Su 180 prove, 47 raggiungono un RMSE sotto i 2 m. **Il cross-domain non fallisce
del tutto**, ma il quadro ribalta E1 su ogni asse.

**La migliore combinazione è ORB + Sauvola + similarità: 90% di successo, RMSE
mediano 0.41 m.** È *sotto* il pavimento del riferimento: la registrazione è
buona quanto questa ground truth consente di misurare.

![Verifica a piena risoluzione](../results/figures/m8_verifica_ribba.png)

### 8.3 Perché ORB batte SIFT, contro ogni aspettativa

Su E1 SIFT domina; su E2 crolla al 30% di successo mentre ORB arriva al 90%. La
causa non è la qualità dei descrittori ma il **numero di candidati**: il ratio
test di Lowe lascia a SIFT 74-121 corrispondenze, il cross-check di ORB ne lascia
circa 730. Con inlier ratio dell'1-5%, RANSAC ha bisogno di candidati, non di
candidati puliti.

L'ipotesi naturale è che il ratio 0.75, tarato su immagini dello stesso dominio,
sia semplicemente troppo severo. **È stata verificata, ed è falsa:**

<!-- TABELLA: diagnosi_ratio -->

Allentando il ratio fino a 0.99 le corrispondenze passano da ~100 a ~2500, ma
l'RMSE resta a centinaia di metri, e su un ritaglio peggiora addirittura da
0.64 m a 163 m. **I match aggiuntivi non contengono segnale**: sono rumore che
sposta il consenso di RANSAC su un modello sbagliato. Il limite di SIFT su questi
dati è nei descrittori, non nel filtro che li seleziona — che è una conclusione
diversa, e più forte.

### 8.4 Il fattore che decide: il modello geometrico

<!-- TABELLA: e2_fattori -->

Sugli stessi identici insiemi di corrispondenze, il modello geometrico cambia
tutto: la **similarità** (4 gradi di libertà) riesce nel 50% delle prove,
l'affine nel 33%, l'**omografia** (8 gradi) nel 18%. Con inlier ratio dell'ordine
del 2-5%, più gradi di libertà significano più modi di accordarsi con gli
outlier: RANSAC trova un consenso, ma quello sbagliato.

È il risultato più trasferibile dell'intero lavoro: **su dati cross-domain con
inlier ratio bassi, il modello più vincolato non è una semplificazione, è una
necessità.**

La tabella mostra anche l'esito dell'ablazione sui codici CXF: rasterizzare
**particelle + acque/strade** (18+12) batte le sole particelle (18), 40% contro
27% di successo. Le strade e i corsi d'acqua aggiungono struttura proprio dove
il tratto storico è più marcato.

### 8.5 I ritagli non sono equivalenti

<!-- TABELLA: e2_per_crop -->

Un caso è istruttivo: **`aspera` fallisce con le sole particelle (125.9 m) e
riesce includendo acque e strade (0.98 m)**. È il ritaglio che tocca la costa,
dove buona parte del contenuto sono la linea di riva e i corsi d'acqua: senza il
codice 12 il vettoriale è quasi vuoto proprio dove l'impianto ha il tratto.

`vedra`, il più povero di tratto, ha l'inlier ratio più basso (0.024) ma riesce
comunque: la struttura c'è, è solo poca.

---

## 9. E3 — classico contro neurale

LoFTR entra nella pipeline dalla stessa porta degli altri: cambia solo il valore
di `--matcher`. Stessi ritagli, stesse metriche, stesse soglie.

<!-- TABELLA: e3 -->

![Confronto classico/neurale](../results/figures/m9_e3_confronto.png)

### 9.1 Due differenze da dichiarare

Il confronto è onesto solo se si dichiara ciò che non è simmetrico:

1. **LoFTR gira a lato massimo 640 px**, mentre SIFT e ORB girano a piena
   risoluzione. LoFTR lavora su una griglia a 1/8 e il costo cresce col quadrato
   del lato: a 1024 px su CPU diventa proibitivo. I keypoint vengono riportati
   alle coordinate originali, quindi `H_est` resta nei pixel di partenza.
2. **Il costo per registrazione è di un altro ordine di grandezza**: circa 5
   secondi contro 0.14 (ORB) e 0.71 (SIFT). Fa parte del risultato.

### 9.2 LoFTR non ribalta il cross-domain

Sul tasso di successo LoFTR **pareggia** ORB (90%), con RMSE mediano peggiore
(0.593 contro 0.404 m) e circa 36 volte il tempo. La promessa del detector-free
— funzionare dove i rilevatori a blob non hanno nulla da agganciare — **non si
realizza su questi dati**.

Dove è invece nettamente superiore è nella **qualità** delle corrispondenze: su
E2 l'inlier ratio mediano è 0.288 contro 0.016 di ORB e 0.118 di SIFT. Sono due
strade opposte allo stesso risultato: LoFTR trova poche corrispondenze molto
pulite, ORB ne trova una massa e lascia a RANSAC il lavoro di setacciarle. Il
solo tasso di successo nasconde questa differenza.

### 9.3 Anche LoFTR ha bisogno della binarizzazione

| preprocessing | corrispondenze mediane | successo | RMSE mediano |
|---|---|---|---|
| CLAHE | 7 | 0 – 10% | 236 – 488 m |
| Sauvola | 477 | 60 – 80% | 0.48 – 0.85 m |
| Sauvola + chiusura | 487 | 60 – 90% | 0.63 – 1.02 m |

Il vantaggio del pre-addestramento su immagini naturali **non sopravvive al
divario di dominio**: a colmarlo è la binarizzazione, non la rete. È forse il
risultato più interessante della componente comparativa, perché suggerisce che
su questo tipo di dati la leva efficace resti il preprocessing.

### 9.4 Fragilità alla degradazione

Su E1, tasso di successo al crescere della degradazione (senza preprocessing):

| degradazione | LoFTR | ORB | SIFT |
|---|---|---|---|
| 0.00 | 100% | 90% | 100% |
| 0.50 | 100% | 60% | 80% |
| 1.00 | 20% | 60% | 60% |
| 1.20 | 0% | 20% | 40% |

LoFTR parte alla pari con SIFT e **crolla prima di entrambi i classici**. Il
rumore gaussiano non appartiene alla distribuzione su cui è stato addestrato.

---

## 10. Modalità d'uso

### 10.1 Installazione

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Per il solo approccio classico bastano `numpy`, `opencv-python`, `Pillow`,
`scipy`, `pandas`, `matplotlib`. `torch` e `kornia` servono **solo** a
`--matcher loftr`; su CPU conviene l'indice dedicato:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install kornia==0.7.3
python -m scripts.scarica_pesi          # pesi LoFTR + verifica del checksum
```

I dati cartografici non sono versionati: `data/README.md` documenta come
ricostruirli.

### 10.2 La CLI

```
python -m src.main --hist <crop.png> --modern <raster.png> [opzioni]

  --hist <path>             immagine storica (obbligatorio)
  --modern <path>           immagine di riferimento (obbligatorio)
  --matcher <sift|orb|loftr>              default: sift
  --preprocess <none|clahe|otsu|sauvola>  default: sauvola
  --morph-close <n>         default 0
  --morph-open <n>          default 0
  --model <similarity|affine|homography>  default: homography
  --ratio <float>           ratio test di Lowe, default 0.75
  --ransac-thresh <float>   px, default 3.0
  --seed <int>              default 42
  --jgw-hist <path>         world file storico   (solo evaluate)
  --jgw-modern <path>       world file moderno   (solo evaluate)
  --soglia-m <float>        soglia di successo in metri
  --out-csv <path>          default results/runs.csv (in append)
  --out-figure <path>       overlay del warp + corrispondenze
  --verbose
```

**Esempio 1 — la configurazione migliore su dati reali:**

```bash
python -m src.main \
    --hist data/crops/ribba.png --modern data/crops/ribba_vec.png \
    --matcher orb --preprocess sauvola --model similarity \
    --jgw-hist data/crops/ribba.jgw --jgw-modern data/crops/ribba_vec.jgw \
    --soglia-m 2.0 --out-figure results/figures/registrazione.png
```

**Esempio 2 — senza world file: la pipeline gira lo stesso.** È la prova che la
georeferenziazione non entra nell'algoritmo:

```bash
python -m src.main --hist data/crops/ribba.png --modern data/crops/ribba_vec.png \
    --matcher sift --preprocess sauvola
# → H_est prodotta, "nessun world file: RMSE non calcolato"
```

**Esempio 3 — lo stesso confronto con il matcher neurale**, cambiando una parola:

```bash
python -m src.main --hist data/crops/ribba.png --modern data/crops/ribba_vec.png \
    --matcher loftr --preprocess sauvola --model similarity
```

### 10.3 Preparazione dei dati ed esperimenti

```bash
# ritagli dal foglio (PNG + world file affiancato)
python -m src.prep.crop

# ispezione e verifica del parser CXF
python -m src.io_geo

# rasterizzazione del vettoriale sulla zona di un ritaglio
python -m src.prep.rasterize --crop ribba --codici 18,12
python -m experiments.m7_rasterize_check --crop ribba

# confronto dei preprocessing
python -m experiments.m5_preprocess --crop tutti

# gli esperimenti: E1, E2, la diagnosi del ratio, E3
python -m experiments.m6_e1_completo --riparti
python -m experiments.m8_e2_griglia
python -m experiments.m8_e2_griglia --diagnosi-ratio
python -m experiments.m9_e3_loftr

# figure e tabelle, tutte generate dal CSV
python -m src.report --csv results/runs.csv

# questa relazione, con le tabelle iniettate dal CSV
python -m scripts.componi_relazione

# i test
python -m tests.test_smoke
```

---

## 11. Limiti e onestà dei risultati

### 11.1 Il vettoriale non è la digitalizzazione dell'impianto

Il CXF è la **cartografia vigente**, non una trascrizione del foglio storico. La
geometria discende dall'impianto, ma ha subito un secolo di frazionamenti e
aggiornamenti: nel CXF compaiono particelle con numeri a quattro cifre (1015,
1026, 1041) che sull'impianto non esistono, dove la numerazione arriva a ~336.

**Parte delle linee vettoriali non ha alcuna corrispondenza nell'inchiostro**, e
si vede a occhio nelle sovrapposizioni a piena risoluzione. Non è disallineamento:
è storia. È anche una delle ragioni per cui gli inlier ratio di E2 sono
strutturalmente bassi, e vanno letti con questa chiave.

### 11.2 Le metriche indirette non funzionano su questi dati

Durante la preparazione si è tentato di misurare l'allineamento con correlazione
incrociata e chamfer matching su distance transform. **Entrambe hanno prodotto
falsi positivi convincenti**: shift ottimali incoerenti fra zone (−238, −330,
+11 px), ottimi saturati sui bordi dello spazio di ricerca, e un profilo
apparentemente sistematico che confermava un disallineamento inesistente.

La diagnosi è quantitativa: **linee tirate a caso sul foglio ottengono l'11.5%
dei pixel entro 2 px dall'inchiostro; il vettoriale vero il 16-20%.** Con quel
rapporto segnale/rumore nessuna di quelle metriche discrimina — la carta ha
inchiostro sparso ovunque (testi, simboli, tratteggi, grana, macchie) e la
superficie di correlazione su disegni al tratto è piatta e multi-picco.

Da qui la regola seguita in tutto il progetto: **l'unica metrica di valutazione è
l'RMSE su checkpoint contro `H_true`**; ogni claim di allineamento si verifica a
piena risoluzione; ogni misura indiretta va accompagnata da una baseline casuale,
e se il segnale non batte nettamente il caso, la misura si butta.

### 11.3 Validazione, non applicazione

Questo va detto prima che lo chieda chi legge. I dati dell'Agenzia delle Entrate
forniscono una georeferenziazione **già nota**, e *proprio per questo*
permettono di valutare quantitativamente un metodo che altrove — scansioni
d'archivio, catasti preunitari, mappe senza world file — dovrebbe operare senza
alcun riferimento.

È un dataset di **validazione**, non di applicazione. Scrivere che il lavoro
"serve ad allineare mappe storiche disorientate" sarebbe falso *su questi dati*:
qui il riferimento c'è, ed è quello che rende possibile la misura.

### 11.4 Ampiezza del campione

Ogni punto delle curve di E1 è la mediana di 5 prove, una per ritaglio; ogni
cella delle tabelle di E2 aggrega 10 prove. Le non-monotonie che compaiono qua e
là — per esempio nel tasso di successo attorno a degradazione 1.1 — sono rumore
di campionamento, non struttura, e non vanno raccontate come fenomeni.

### 11.5 Un errore documentato nel materiale di partenza

La specifica del progetto riportava il foglio come 8000×5322 px, con l'estensione
corrispondente. Il file reale misura **8489×5648 px**: i pixel in più sono il
margine di carta della scansione. L'ipotesi alternativa — che il world file
appartenesse a un rendering diverso, e che quindi il pixel valesse 0.2398 m
anziché 0.254453 — avrebbe comportato un errore di scala del 6% su *tutti* i
risultati in metri.

È stata risolta con una verifica visiva a piena risoluzione: con il JGW applicato
al file reale il vettoriale ricalca l'inchiostro, mentre un errore del 6%
significherebbe circa 500 px di divergenza sul foglio. Il codice legge sempre la
dimensione dal file, mai da una costante.

---

## 12. Conclusioni

Il lavoro misura, e spiega, dove i metodi classici cedono sul cross-domain — e
dove invece reggono meglio del previsto.

1. **La pipeline è corretta**: E1 recupera trasformazioni note con errore
   sub-pixel (0.046 m mediani nella configurazione migliore), due ordini di grandezza sotto
   il pavimento del riferimento.
2. **La registrazione cross-domain riesce**, ma non con la configurazione che ci
   si aspetterebbe: ORB + Sauvola + similarità raggiunge il 90% di successo con
   RMSE mediano 0.41 m, al limite di ciò che questa ground truth può misurare.
3. **Il modello geometrico conta più del matcher**: a parità di corrispondenze,
   passare da omografia a similarità porta il successo dal 18% al 50%. Con inlier
   ratio bassi, vincolare è necessario.
4. **Il preprocessing conta più della rete**: su E2 tutti e tre i matcher, LoFTR
   incluso, funzionano solo con Sauvola e falliscono con CLAHE.
5. **Il matcher neurale non ribalta il risultato**: pareggia ORB sul successo,
   con errore maggiore e costo 36 volte superiore — ma con corrispondenze molto
   più pulite, il che indica che il collo di bottiglia è altrove.
6. **Due ipotesi sono state smentite dai dati** e riportate come tali: il ratio
   test troppo severo (§8.3) e il fallimento atteso di Otsu (§5.2).

Un progetto che misura e spiega perché un metodo cede vale più di uno che mostra
solo il caso riuscito. Qui i casi riusciti ci sono, e i punti di rottura sono
documentati con lo stesso rigore.
