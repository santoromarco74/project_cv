# Registrazione di mappe catastali storiche su cartografia moderna

**Progetto d'esame di Computer Vision — Università di Pavia**
Caso di studio: Comune di Varazze, foglio 49 (`L675_004900`), Originale di Impianto dell'Agenzia delle Entrate.

> Questo documento è generato da `scripts/componi_relazione.py`: tutte le tabelle
> sono aggregazioni di `results/runs.csv` calcolate al momento della
> composizione, non numeri ricopiati a mano. Le figure si rigenerano con i
> comandi indicati in §11.

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
il fatto che i due documenti, come si vedrà in §12, **non rappresentano
esattamente la stessa realtà**.

Il progetto affronta il problema in due componenti, secondo la traccia approvata:

- **A — Core-Vision classico**: preprocessing accurato, SIFT/ORB, RANSAC, con
  l'obiettivo di spingere al massimo i metodi classici tramite una pulizia
  ottimale del segnale.
- **B — Comparativo**: confronto quantitativo con un matcher *detector-free*
  di deep learning (LoFTR), nella stessa identica pipeline.

---

## 2. Come funziona, in concreto

Prima di entrare nei dettagli conviene vedere che cosa succede davvero quando si
lancia il programma. Questo capitolo segue **una singola registrazione
dall'inizio alla fine**. Tutti i termini tecnici usati qui sono spiegati nel
glossario alla fine del capitolo, e ripresi più a fondo nei capitoli successivi.

### 2.1 Il punto di partenza: due cose che non si possono confrontare

Abbiamo due descrizioni della stessa zona di Varazze, ma di natura diversa:

- la **scansione del foglio d'impianto**, che è un'immagine: una griglia di
  pixel colorati, dove il programma non sa che cosa sia una particella;
- il **file CXF**, che non è un'immagine ma un elenco di coordinate: *"il
  poligono chiamato 63 ha i vertici nei punti (−30512.4, −11588.1), (…)"*.

Non si possono sovrapporre direttamente, perché non parlano la stessa lingua.

### 2.2 I sei passaggi

![Una registrazione passo per passo](../results/figures/m10_passo_passo.png)

**1 · La scansione storica.** L'immagine così come arriva: carta ingiallita,
tratto a penna sbiadito, numeri di particella scritti a mano, macchie.

**2 · La pulizia.** Il programma decide, per ogni singolo pixel, se è inchiostro
o carta. Il risultato è un'immagine in bianco e nero puro. Non è un abbellimento:
serve a togliere di mezzo il colore della carta, che è diverso in ogni punto del
foglio e non ha niente a che vedere con il disegno.

**3 · Il vettoriale, disegnato.** Le coordinate del CXF vengono tracciate come
linee su un'immagine bianca. Adesso abbiamo **due immagini** della stessa zona, e
il confronto è possibile.

**4 · Gli abbinamenti.** Il programma cerca in entrambe le immagini dei punti
"riconoscibili" — un incrocio di confini, l'angolo di un fabbricato — e per
ciascuno calcola una specie di **impronta numerica** che riassume come appare il
suo vicinato. Poi accoppia i punti di un'immagine con quelli dell'altra che hanno
l'impronta più simile.

Nel pannello 4 ogni linea rossa unisce due punti che il programma considera la
stessa cosa. **Se fossero tutti giusti, le linee sarebbero parallele.** Non lo
sono affatto: su un disegno al tratto moltissimi incroci si somigliano, e in
questo esempio **solo il 6% degli abbinamenti risulta corretto**. Il resto è rumore.

**5 · La votazione.** È il passaggio che salva tutto. Il programma prende a caso
due abbinamenti, calcola quale spostamento-rotazione-ingrandimento
implicherebbero, e poi **conta quanti altri abbinamenti sarebbero d'accordo** con
quella stessa trasformazione. Ripete l'operazione migliaia di volte e tiene la
trasformazione che ha raccolto più consensi.

Gli abbinamenti d'accordo si chiamano *inlier*. Nel pannello 5 ne sono rimasti 54
su 874, e questa volta **sono tutti paralleli**: descrivono tutti lo stesso
movimento. Questa procedura si chiama RANSAC, ed è ciò che permette di trovare la
risposta giusta quando il 94% dei dati è sbagliato.

**6 · Il risultato.** La trasformazione trovata viene applicata all'immagine
storica, che così si sovrappone al vettoriale. Nel pannello 6 il tratto storico
deformato è in rosso sopra le linee nere del vettoriale: dove il rosso segue il
nero, la registrazione è corretta.

### 2.3 Come facciamo a sapere se ha funzionato

Qui sta la particolarità di questo progetto. Entrambi i file portano con sé la
propria **georeferenziazione**: sei numeri (il *world file*) che dicono a quali
coordinate sul terreno corrisponde il pixel in alto a sinistra, e quanti metri
misura il lato di un pixel. Da questi sei numeri si ricava per via puramente
algebrica la trasformazione **esatta**, senza doverne indovinare nemmeno un
pezzo.

Quindi la risposta giusta la conosciamo già, e possiamo dire di **quanti metri**
il programma ha sbagliato. Nell'esempio della figura: 0.47 m, su un riferimento
che di suo ha un'incertezza di circa mezzo metro.

**Il programma che stima non vede quei sei numeri.** Li vede solo il codice che
corregge. È una separazione imposta per costruzione (§4.2): se l'informazione
sulla posizione filtrasse nell'algoritmo, il risultato non significherebbe più
nulla, come uno studente che risolve il problema avendo visto le soluzioni.

### 2.4 Perché è difficile

Se la procedura sembra ragionevole, resta da capire perché non funzioni sempre.
Tre ragioni, tutte misurate nei capitoli §8-§10:

1. **I due disegni sono fatti di linee sottili, non di texture.** I metodi
   classici per riconoscere punti nascono per le fotografie, dove ogni zona ha un
   aspetto diverso dalle altre. Su un reticolo di confini, un incrocio somiglia a
   tutti gli altri incroci.
2. **I due documenti non rappresentano la stessa realtà.** Il CXF è la mappa
   *di oggi*, il foglio è quella *di un secolo fa*: alcuni confini sono cambiati,
   e ci sono linee vettoriali che nell'inchiostro non esistono (§12.1).
3. **L'aspetto è completamente diverso**: inchiostro sbiadito irregolare contro
   linee nette uniformi. È quello che in letteratura si chiama *divario di
   dominio*.

### 2.5 Glossario

| termine | significato |
|---|---|
| **registrazione** | sovrapporre due immagini della stessa zona, trovando la trasformazione che porta l'una sull'altra |
| **raster / immagine** | griglia di pixel |
| **vettoriale** | descrizione per coordinate (elenchi di vertici), non per pixel |
| **rasterizzare** | disegnare un vettoriale su un'immagine |
| **world file** (`.jgw`) | sei numeri che legano i pixel di un'immagine alle coordinate sul terreno |
| **georeferenziazione** | il fatto che un'immagine sappia dove si trova sul terreno |
| **keypoint** | un punto dell'immagine giudicato "riconoscibile" |
| **descrittore** | l'impronta numerica che riassume l'aspetto attorno a un keypoint |
| **SIFT, ORB** | due modi diversi di calcolare keypoint e descrittori |
| **matching** | accoppiare i keypoint di un'immagine con quelli dell'altra |
| **ratio test di Lowe** | il filtro di SIFT: tiene un abbinamento solo se il candidato migliore batte nettamente il secondo (§7.1) |
| **RANSAC** | la procedura a votazione del passo 5, che trova la risposta giusta anche con pochi dati buoni |
| **inlier** | un abbinamento che concorda con la trasformazione scelta |
| **inlier ratio** | la percentuale di inlier: quanto erano buoni i dati di partenza |
| **omografia, affine, similarità** | famiglie di trasformazioni con 8, 6 e 4 parametri liberi (§7.2) |
| **binarizzazione** | decidere, pixel per pixel, se è inchiostro o carta |
| **Otsu, Sauvola** | due modi di scegliere quella soglia (§6.1) |
| **CLAHE** | tecnica che non binarizza, ma aumenta il contrasto locale (§6.1) |
| **LoFTR** | rete neurale che abbina due immagini senza cercare prima i keypoint (§7.3) |
| **RMSE** | l'errore medio commesso, qui espresso in metri |
| **E1, E2, E3** | i tre esperimenti: sintetico, reale, comparativo (§8, §9, §10) |

---

## 3. I dati

### 3.1 Il foglio e il vettoriale

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

### 3.2 La trappola dei due CXF

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

### 3.3 Il parser CXF e il campo che sfasa tutto

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

### 3.4 I ritagli

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

| crop      | x0   | y0   | larghezza | altezza | X (m)           | Y (m)           |
|-----------|------|------|-----------|---------|-----------------|-----------------|
| tassarole | 1500 | 300  | 1024      | 1024    | -31098 … -30838 | -11616 … -11355 |
| cannei    | 2900 | 200  | 1200      | 1000    | -30742 … -30437 | -11584 … -11330 |
| ribba     | 3300 | 600  | 1024      | 1024    | -30640 … -30380 | -11692 … -11431 |
| vedra     | 4200 | 2000 | 1024      | 1024    | -30411 … -30151 | -12048 … -11788 |
| aspera    | 5600 | 2600 | 1024      | 1024    | -30055 … -29795 | -12201 … -11940 |


![I cinque ritagli](../results/figures/m1_crops.png)

Ogni ritaglio è salvato come PNG accompagnato dal **proprio world file**,
composto per traslazione dell'origine pixel:

```
C' = C + A·x0 + B·y0        F' = F + D·x0 + E·y0
```

---

## 4. La ground truth: esatta e gratuita

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

### 4.1 L'incertezza del riferimento, dichiarata

Il JGW non è verità assoluta. I metadati del foglio dicono che la scansione è
stata ricampionata su **76 coppie omologhe**, con:

| grandezza | valore |
|---|---|
| scarto massimo | 1.28 m |
| scarto medio | 0.56 m |
| deviazione standard | 0.25 m |

Questo definisce un **pavimento**: un RMSE inferiore a ~0.5 m non misura più
l'errore dell'algoritmo, misura il rumore del riferimento. Va tenuto presente
ogni volta che si legge una cifra nelle tabelle di §9.

### 4.2 Separazione fisica fra algoritmo e riferimento

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

## 5. La pipeline

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

## 6. Preprocessing

Il preprocessing è il cuore della componente classica: è il passo 2 del capitolo
§2, quello che decide per ogni pixel se è inchiostro o carta. Ogni stadio è
attivabile da riga di comando, perché confrontare le alternative *è* uno dei
risultati, non un parametro da fissare una volta per tutte.

### 6.1 Le tre strade, spiegate

**Il problema.** Su una fotografia normale si potrebbe scegliere un valore di
grigio e dire: "più scuro di così è inchiostro, più chiaro è carta". Su un foglio
d'archivio non funziona, perché la carta non ha un colore solo: è più gialla nelle
zone ingiallite, più scura vicino alle pieghe, più chiara al centro. Una soglia
che va bene in un angolo del foglio sbaglia nell'angolo opposto.

**Otsu — una soglia sola per tutta l'immagine.** Guarda l'istogramma dei grigi e
sceglie il valore che separa meglio le due popolazioni (scuri e chiari). È il
metodo classico di riferimento, e serve qui come termine di paragone. Il suo
limite è strutturale: se la carta cambia luminosità da una parte all'altra, una
soglia sola non può andare bene ovunque.

**Sauvola — una soglia diversa per ogni zona.** Per ogni pixel guarda solo un
quadratino di 25×25 pixel attorno a sé e calcola lì la sua soglia, con la formula

```
T = m · [1 + k · (s/R − 1)]        m = media locale, s = deviazione locale
```

Dove la carta è uniforme, `s` è piccola e la soglia si abbassa, così le
irregolarità di fondo non vengono scambiate per tratto. Dove c'è disegno, `s`
cresce e la soglia si alza verso la media. In pratica: **si adatta alla carta,
zona per zona**.

Fatta ingenuamente costerebbe un ricalcolo per ogni pixel su tutta la finestra;
qui è implementata con le *immagini integrali*, una tabella precalcolata che
permette di ottenere media e varianza di qualunque riquadro con poche
operazioni. Il risultato è identico, ma il costo non dipende più dalla dimensione
della finestra. La correttezza dell'ottimizzazione è verificata confrontandola
pixel per pixel con la versione ingenua — è il genere di errore che non si
vedrebbe mai guardando l'immagine.

**CLAHE — non binarizzare affatto.** È l'alternativa: invece di ridurre tutto a
bianco e nero, aumenta il contrasto **localmente**, così i tratti sbiaditi
diventano più leggibili senza buttare via le sfumature. La ragione per provarlo è
concreta: SIFT lavora sulle sfumature di grigio, e binarizzare potrebbe
distruggere proprio l'informazione su cui si basa. Quale delle due strade
convenga non si può decidere a tavolino, ed è una delle domande sperimentali del
progetto (§6.2 e §9).

**Morfologia — ritoccare la forma del tratto.** Due operazioni elementari:
l'*apertura* cancella i puntini isolati (la grana della carta scambiata per
inchiostro), la *chiusura* ricongiunge i tratti interrotti, frequenti perché il
pennino stacca. L'ordine conta: si apre prima e si chiude poi, perché chiudendo
per primo si salderebbe la grana al tratto, rendendola poi non più rimovibile.

La variante che nelle tabelle compare come **`sauvola+chiusura`** è esattamente
questo: la stessa binarizzazione, più un passaggio di chiusura con un elemento
strutturante di 3×3 pixel. In pratica il tratto viene dilatato di un pixel e poi
ristretto: le interruzioni più strette di quel pixel restano riempite, tutto il
resto torna com'era.

![Effetto della chiusura](../results/figures/m5_chiusura_ribba.png)

Sull'intero ritaglio la chiusura aggiunge 2957 pixel su un milione, lo 0.28%: è
un ritocco, non una trasformazione. Ma il quarto pannello mostra anche un effetto
collaterale che non era previsto e che va detto: **dove due linee corrono
parallele e vicine — il caso delle strade, disegnate a doppio bordo — la chiusura
le fonde in una fascia piena.** La strada smette di essere due tratti sottili e
diventa un nastro nero.

Non è necessariamente un male: nel vettoriale le strade sono poligoni, quindi
riempirle avvicina le due rappresentazioni. Ma cambia la natura di ciò che il
matcher vede, ed è parte della ragione per cui questa variante si comporta
diversamente dalle altre nei risultati di §9 e §10.

![Confronto dei preprocessing](../results/figures/m5_preprocess_ribba.png)

Le figure hanno una riga di dettagli **a piena risoluzione**, perché su una
binarizzazione il ridimensionamento inganna sempre: un tratto interrotto e uno
saldato diventano indistinguibili a un quarto di scala.

### 6.2 Quanto costa binarizzare

Contando i keypoint SIFT rilevati su ciascuna variante, rapportati al grayscale:

| variante | keypoint (rapporto sul grayscale) |
|---|---|
| CLAHE | 1.17× – 1.55× |
| Sauvola | 0.75× – 0.79× |
| Sauvola + chiusura | 0.68× – 0.77× |
| Otsu | 0.61× – 0.72× |

Il dato è consistente su tutti e cinque i ritagli: **binarizzare costa fra il 21
e il 39% dei keypoint, CLAHE ne aggiunge fra il 17 e il 55%**. Non è ancora la
risposta — quella è l'RMSE di §8 e §9 — ma indica la direzione.

### 6.3 Una previsione verificata invece che assunta

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

## 7. Matching e stima

Sono i passi 4 e 5 del capitolo §2. L'interfaccia è la stessa per tutti i
metodi: due immagini entrano, due insiemi di punti corrispondenti escono.

### 7.1 Trovare gli abbinamenti

**SIFT** cerca punti che restano riconoscibili anche se l'immagine viene
ingrandita o ruotata, e per ognuno calcola un vettore di 128 numeri che riassume
come sono orientati i contorni nell'intorno. Due punti che nelle due immagini
hanno vettori simili sono candidati alla stessa cosa reale.

Il problema è che un candidato "abbastanza simile" spesso non basta. Per questo
si usa il **ratio test di Lowe**: per ogni punto si guardano i *due* candidati
migliori nell'altra immagine, `d1` (distanza dal più simile) e `d2` (distanza dal
secondo più simile), e l'abbinamento si accetta solo se

```
d1 < 0.75 · d2
```

cioè se il migliore è nettamente più vicino del secondo, non di poco. L'idea è
che se i due candidati migliori si somigliano fra loro (`d1` e `d2` vicini),
allora quel punto non è davvero distintivo — ci sono almeno due posti dove
potrebbe stare — ed è meglio scartarlo che rischiare un abbinamento a caso.

Due esempi con numeri: se `d1 = 50` e `d2 = 51`, il rapporto è 0.98, molto sopra
la soglia — **scartato**, i due candidati sono quasi indistinguibili. Se `d1 = 50`
e `d2 = 100`, il rapporto è 0.50 — **accettato**, il migliore si stacca
nettamente.

![Il ratio test, con i rapporti veri](../results/figures/m10_ratio_test.png)

La figura mostra questo rapporto calcolato su ogni keypoint di un ritaglio reale,
in due situazioni. **Contro sé stesso ruotato** (stesso dominio, come in E1) la
distribuzione è ampia e un terzo abbondante dei punti supera la soglia: ci sono
molti candidati chiaramente migliori degli altri. **Contro il vettoriale**
(cross-domain, come in E2) la distribuzione si ammassa quasi tutta sopra 0.75, e
**solo l'1.5% dei keypoint sopravvive**. È il numero dietro le poche decine di
corrispondenze che SIFT trova su E2 (§9.3): non è che il ratio test sia mal
tarato, è che nel cross-domain quasi nessun punto ha davvero un solo candidato
migliore di tutti gli altri — sono tutti vagamente simili a molti altri punti.

**ORB** fa la stessa cosa in modo più rapido ed essenziale: descrive ogni punto
con una stringa di bit invece che con 128 numeri, e confronta le stringhe
contando i bit diversi. Al posto del ratio test usa il **cross-check**: tiene
l'abbinamento solo se A sceglie B *e* B sceglie A.

Questi due filtri hanno una conseguenza importante, che emergerà in §9.3: il
ratio test è molto più selettivo del cross-check, e su dati difficili lascia
molte meno corrispondenze.

### 7.2 Scegliere quanto la trasformazione può deformare

RANSAC — la votazione descritta al passo 5 — non cerca "una trasformazione
qualsiasi": cerca la migliore all'interno di una **famiglia** che decidiamo noi.
Ne sono previste tre, di libertà crescente:

| famiglia | parametri | cosa può fare |
|---|---|---|
| **similarità** | 4 | spostare, ruotare, ingrandire — le forme restano simili a sé stesse |
| **affine** | 6 | in più: stirare in una direzione, inclinare (un quadrato diventa un parallelogramma) |
| **omografia** | 8 | in più: la deformazione prospettica di un piano visto di sbieco |

Più parametri significa poter descrivere trasformazioni più complicate. Ma
significa anche, e questo è il punto, **più modi di mettersi d'accordo con dati
sbagliati**: con più libertà RANSAC può trovare un consenso anche fra
abbinamenti casuali. Il confronto fra le tre famiglie non è un dettaglio
implementativo: come si vedrà in §9.4, su questi dati è il fattore che decide fra
successo e fallimento.

Sotto le 4 corrispondenze la stima si ferma dichiarando il fallimento, invece di
restituire un risultato che non avrebbe senso.

### 7.3 Il matcher neurale (componente B)

**LoFTR** affronta il problema da un'angolazione diversa. I metodi classici
procedono in due tempi: prima *trovano* i punti notevoli, poi li descrivono. Se
il primo passo fallisce — e su un disegno al tratto può fallire, perché non ci
sono angoli e macchie ben definiti — il secondo non ha nulla su cui lavorare.

LoFTR salta il primo passo: è *detector-free*. Divide entrambe le immagini in una
griglia regolare e confronta ogni cella di una con tutte le celle dell'altra, con
un meccanismo di attenzione che tiene conto anche del contesto circostante. È una
rete neurale già addestrata su fotografie: qui viene usata così com'è, senza
alcun riaddestramento.

L'aspettativa che giustifica il confronto è precisa: **dovrebbe funzionare
proprio dove i metodi classici soffrono**, cioè dove non ci sono punti notevoli
da trovare. Se questa aspettativa si realizzi su mappe catastali è la domanda
della componente comparativa, e la risposta è in §10.

---

## 8. E1 — esperimento sintetico, stesso dominio

Il primo esperimento confronta un ritaglio con **sé stesso trasformato con una
`H` nota**: rotazione, scala, traslazione, omografia lieve, più una degradazione
radiometrica opzionale (rumore, sfocatura, contrasto, luminosità). La ground
truth è esatta per costruzione e il divario di dominio è assente.

Serve a due cose, entrambe indispensabili: stabilire il **tetto di prestazione**
dei matcher, e verificare che la pipeline sia **corretta**. Se E1 fallisce, il
problema è nel codice.

| matcher | preprocess | prove | successo_pct | rmse_px_mediano_ok | rmse_px_max_ok | inlier_ratio | match_medi | t_ms |
|---------|------------|-------|--------------|--------------------|----------------|--------------|------------|------|
| loftr   | none       | 80    | 46.2         | 0.222              | 0.975          | 0.604        | 1834       | 6262 |
| orb     | clahe      | 80    | 62.5         | 0.443              | 0.857          | 0.593        | 2362       | 174  |
| orb     | none       | 80    | 61.3         | 0.446              | 0.987          | 0.582        | 2080       | 141  |
| orb     | sauvola    | 80    | 48.8         | 0.506              | 0.989          | 0.462        | 1815       | 155  |
| sift    | clahe      | 80    | 75.0         | 0.183              | 0.845          | 0.733        | 1972       | 627  |
| sift    | none       | 80    | 70.0         | 0.168              | 0.969          | 0.664        | 1446       | 479  |
| sift    | sauvola    | 80    | 67.5         | 0.214              | 0.943          | 0.572        | 817        | 633  |


![RMSE contro degradazione](../results/figures/m6_rmse_vs_degradazione.png)

### 8.1 Il tetto di prestazione

In assenza di degradazione tutti e tre i matcher recuperano la trasformazione
con errore ampiamente sub-pixel — nel caso peggiore su 35 combinazioni di
ritaglio e trasformazione, SIFT si ferma a 0.276 px (0.070 m). Su un riferimento
il cui pavimento è ~0.5 m, questo è due ordini di grandezza sotto: è il segnale
che la pipeline è corretta.

Il caso geometricamente più difficile è sempre la **rotazione a 45°**, su tutti i
ritagli: è il costo dell'interpolazione del warp e della quantizzazione
dell'orientamento dei descrittori, non un difetto.

### 8.2 La rottura è un precipizio, non una discesa

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

### 8.3 Nota sull'aggregazione

Oltre la soglia di rottura RANSAC **restituisce comunque una `H`**, ma sbagliata
di migliaia di pixel: il massimo osservato è 12615 px. Una sola stima di questo
tipo trascina la media dell'intero gruppo, producendo "RMSE medi" di centinaia di
pixel che non descrivono né i casi buoni né i cattivi.

Per questo le tabelle riportano la **mediana** accompagnata dal **tasso di
successo**, che è la grandezza che descrive i casi cattivi. Le curve usano
mediana, banda interquartile e scala logaritmica.

Le tabelle di §9 e §10 aggregano in due modi diversi, e le colonne lo dicono nel
nome: `rmse_m_mediano` è la mediana su **tutte** le prove — grande di proposito,
perché su E2 una configurazione può avere zero successi e allora "l'errore sulle
riuscite" non esisterebbe — mentre `rmse_m_mediano_ok` è la mediana sulle sole
prove riuscite. Confrontando le due tabelle si trovano quindi cifre leggermente
diverse per la stessa configurazione: non è un'incoerenza, sono due domande
diverse.

---

## 9. E2 — cross-domain reale

Il secondo esperimento è quello vero: ritaglio storico contro **raster del CXF**,
nello stesso sistema di riferimento.

### 9.1 La rasterizzazione e la sua verifica

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

Il motivo di questa scelta è documentato in §12.2: su questi dati le metriche
indirette di allineamento producono falsi positivi convincenti.

### 9.2 Risultati

| matcher | preprocess       | modello    | prove | successo_pct | rmse_m_mediano | rmse_m_minimo | inlier_ratio | match_mediani |
|---------|------------------|------------|-------|--------------|----------------|---------------|--------------|---------------|
| loftr   | clahe            | affine     | 10    | 0.0          | 242.41         | 124.979       | 0.354        | 7             |
| loftr   | clahe            | homography | 10    | 0.0          | 487.91         | 152.008       | 0.472        | 7             |
| loftr   | clahe            | similarity | 10    | 10.0         | 236.42         | 1.847         | 0.236        | 7             |
| loftr   | sauvola          | affine     | 10    | 70.0         | 0.47           | 0.372         | 0.284        | 477           |
| loftr   | sauvola          | homography | 10    | 60.0         | 0.85           | 0.448         | 0.293        | 477           |
| loftr   | sauvola          | similarity | 10    | 80.0         | 0.5            | 0.23          | 0.276        | 477           |
| loftr   | sauvola+chiusura | affine     | 10    | 70.0         | 0.85           | 0.424         | 0.288        | 487           |
| loftr   | sauvola+chiusura | homography | 10    | 60.0         | 1.02           | 0.468         | 0.291        | 487           |
| loftr   | sauvola+chiusura | similarity | 10    | 90.0         | 0.63           | 0.375         | 0.276        | 487           |
| orb     | clahe            | affine     | 10    | 0.0          | 156.05         | 16.954        | 0.009        | 737           |
| orb     | clahe            | homography | 10    | 0.0          | 195.01         | 32.961        | 0.012        | 737           |
| orb     | clahe            | similarity | 10    | 50.0         | 2.34           | 0.389         | 0.011        | 737           |
| orb     | sauvola          | affine     | 10    | 50.0         | 2.09           | 0.365         | 0.028        | 732           |
| orb     | sauvola          | homography | 10    | 10.0         | 84.94          | 0.701         | 0.016        | 732           |
| orb     | sauvola          | similarity | 10    | 90.0         | 0.41           | 0.242         | 0.054        | 732           |
| orb     | sauvola+chiusura | affine     | 10    | 60.0         | 1.17           | 0.412         | 0.039        | 708           |
| orb     | sauvola+chiusura | homography | 10    | 20.0         | 25.12          | 0.612         | 0.021        | 708           |
| orb     | sauvola+chiusura | similarity | 10    | 70.0         | 0.48           | 0.209         | 0.045        | 708           |
| sift    | clahe            | affine     | 10    | 0.0          | 243.94         | 94.165        | 0.095        | 74            |
| sift    | clahe            | homography | 10    | 0.0          | 193.92         | 139.389       | 0.25         | 74            |
| sift    | clahe            | similarity | 10    | 0.0          | 193.79         | 138.388       | 0.243        | 74            |
| sift    | sauvola          | affine     | 10    | 30.0         | 85.03          | 0.277         | 0.097        | 120           |
| sift    | sauvola          | homography | 10    | 10.0         | 189.96         | 0.587         | 0.132        | 120           |
| sift    | sauvola          | similarity | 10    | 30.0         | 158.27         | 0.229         | 0.098        | 120           |
| sift    | sauvola+chiusura | affine     | 10    | 20.0         | 149.55         | 0.671         | 0.091        | 121           |
| sift    | sauvola+chiusura | homography | 10    | 0.0          | 177.55         | 2.492         | 0.124        | 121           |
| sift    | sauvola+chiusura | similarity | 10    | 30.0         | 163.66         | 0.302         | 0.088        | 121           |


Su 180 prove, 47 raggiungono un RMSE sotto i 2 m. **Il cross-domain non fallisce
del tutto**, ma il quadro ribalta E1 su ogni asse.

**La migliore combinazione è ORB + Sauvola + similarità: 90% di successo, RMSE
mediano 0.41 m.** È *sotto* il pavimento del riferimento: la registrazione è
buona quanto questa ground truth consente di misurare.

![Verifica a piena risoluzione](../results/figures/m8_verifica_ribba.png)

### 9.3 Perché ORB batte SIFT, contro ogni aspettativa

Su E1 SIFT domina; su E2 crolla al 30% di successo mentre ORB arriva al 90%. La
causa non è la qualità dei descrittori ma il **numero di candidati**: il ratio
test di Lowe lascia a SIFT 74-121 corrispondenze, il cross-check di ORB ne lascia
circa 730. Con inlier ratio dell'1-5%, RANSAC ha bisogno di candidati, non di
candidati puliti.

L'ipotesi naturale è che il ratio 0.75, tarato su immagini dello stesso dominio,
sia semplicemente troppo severo. **È stata verificata, ed è falsa:**

| ratio | match_mediani | inlier_mediani | inlier_ratio | rmse_m_mediano | riuscite |
|-------|---------------|----------------|--------------|----------------|----------|
| 0.75  | 101           | 12             | 0.0843       | 208.2          | 1/5      |
| 0.85  | 314           | 38             | 0.081        | 183.8          | 0/5      |
| 0.95  | 1225          | 85             | 0.0641       | 167.5          | 0/5      |
| 0.99  | 2421          | 156            | 0.057        | 182.2          | 0/5      |


Allentando il ratio fino a 0.99 le corrispondenze passano da ~100 a ~2500, ma
l'RMSE resta a centinaia di metri, e su un ritaglio peggiora addirittura da
0.64 m a 163 m. **I match aggiuntivi non contengono segnale**: sono rumore che
sposta il consenso di RANSAC su un modello sbagliato. Il limite di SIFT su questi
dati è nei descrittori, non nel filtro che li seleziona — che è una conclusione
diversa, e più forte.

### 9.4 Il fattore che decide: il modello geometrico

| fattore            | valore     | prove | successo_pct | rmse_m_mediano | inlier_ratio |
|--------------------|------------|-------|--------------|----------------|--------------|
| modello geometrico | affine     | 90    | 33.3         | 36.37          | 0.09         |
| modello geometrico | homography | 90    | 17.8         | 139.14         | 0.135        |
| modello geometrico | similarity | 90    | 50.0         | 1.61           | 0.105        |
| codici CXF         | 18         | 135   | 27.4         | 126.98         | 0.103        |
| codici CXF         | 18+12      | 135   | 40.0         | 17.79          | 0.116        |


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

### 9.5 I ritagli non sono equivalenti

| crop      | codici | n_matches | inlier_ratio | rmse_m  | success |
|-----------|--------|-----------|--------------|---------|---------|
| aspera    | 18     | 666       | 0.009009     | 125.886 | False   |
| aspera    | 18+12  | 766       | 0.009138     | 0.981   | True    |
| cannei    | 18     | 700       | 0.065714     | 0.242   | True    |
| cannei    | 18+12  | 834       | 0.094724     | 0.408   | True    |
| ribba     | 18     | 765       | 0.073203     | 0.497   | True    |
| ribba     | 18+12  | 874       | 0.061785     | 0.472   | True    |
| tassarole | 18     | 673       | 0.054978     | 0.32    | True    |
| tassarole | 18+12  | 788       | 0.053299     | 0.286   | True    |
| vedra     | 18     | 533       | 0.02439      | 0.326   | True    |
| vedra     | 18+12  | 687       | 0.040757     | 0.404   | True    |


Un caso è istruttivo: **`aspera` fallisce con le sole particelle (125.9 m) e
riesce includendo acque e strade (0.98 m)**. È il ritaglio che tocca la costa,
dove buona parte del contenuto sono la linea di riva e i corsi d'acqua: senza il
codice 12 il vettoriale è quasi vuoto proprio dove l'impianto ha il tratto.

`vedra`, il più povero di tratto, ha l'inlier ratio più basso (0.024) ma riesce
comunque: la struttura c'è, è solo poca.

---

## 10. E3 — classico contro neurale

LoFTR entra nella pipeline dalla stessa porta degli altri: cambia solo il valore
di `--matcher`. Stessi ritagli, stesse metriche, stesse soglie.

| esperimento | matcher | config                        | prove | successo_pct | rmse_m_mediano_ok | inlier_ratio | match_mediani | t_ms |
|-------------|---------|-------------------------------|-------|--------------|-------------------|--------------|---------------|------|
| E1          | loftr   | none / homography             | 80    | 46.2         | 0.056             | 0.562        | 861           | 5882 |
| E1          | orb     | clahe / homography            | 80    | 62.5         | 0.113             | 0.691        | 2266          | 174  |
| E1          | sift    | clahe / homography            | 80    | 75.0         | 0.046             | 0.873        | 1169          | 606  |
| E2          | loftr   | sauvola+chiusura / similarity | 10    | 90.0         | 0.593             | 0.276        | 487           | 4959 |
| E2          | orb     | sauvola / similarity          | 10    | 90.0         | 0.404             | 0.054        | 732           | 138  |
| E2          | sift    | sauvola+chiusura / similarity | 10    | 30.0         | 0.466             | 0.088        | 121           | 708  |


![Confronto classico/neurale](../results/figures/m9_e3_confronto.png)

### 10.1 Due differenze da dichiarare

Il confronto è onesto solo se si dichiara ciò che non è simmetrico:

1. **LoFTR gira a lato massimo 640 px**, mentre SIFT e ORB girano a piena
   risoluzione. LoFTR lavora su una griglia a 1/8 e il costo cresce col quadrato
   del lato: a 1024 px su CPU diventa proibitivo. I keypoint vengono riportati
   alle coordinate originali, quindi `H_est` resta nei pixel di partenza.
2. **Il costo per registrazione è di un altro ordine di grandezza**: circa 5
   secondi contro 0.14 (ORB) e 0.71 (SIFT). Fa parte del risultato.

### 10.2 LoFTR non ribalta il cross-domain

Sul tasso di successo LoFTR **pareggia** ORB (90%), con RMSE mediano peggiore
(0.593 contro 0.404 m) e circa 36 volte il tempo. La promessa del detector-free
— funzionare dove i rilevatori a blob non hanno nulla da agganciare — **non si
realizza su questi dati**.

Dove è invece nettamente superiore è nella **qualità** delle corrispondenze: su
E2 l'inlier ratio mediano è 0.288 contro 0.016 di ORB e 0.118 di SIFT. Sono due
strade opposte allo stesso risultato: LoFTR trova poche corrispondenze molto
pulite, ORB ne trova una massa e lascia a RANSAC il lavoro di setacciarle. Il
solo tasso di successo nasconde questa differenza.

### 10.3 Anche LoFTR ha bisogno della binarizzazione

| preprocessing | corrispondenze mediane | successo | RMSE mediano |
|---|---|---|---|
| CLAHE | 7 | 0 – 10% | 236 – 488 m |
| Sauvola | 477 | 60 – 80% | 0.48 – 0.85 m |
| Sauvola + chiusura | 487 | 60 – 90% | 0.63 – 1.02 m |

Il vantaggio del pre-addestramento su immagini naturali **non sopravvive al
divario di dominio**: a colmarlo è la binarizzazione, non la rete. È forse il
risultato più interessante della componente comparativa, perché suggerisce che
su questo tipo di dati la leva efficace resti il preprocessing.

### 10.4 Fragilità alla degradazione

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

## 11. Modalità d'uso

### 11.1 Installazione

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

### 11.2 La CLI

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

### 11.3 Preparazione dei dati ed esperimenti

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

## 12. Limiti e onestà dei risultati

### 12.1 Il vettoriale non è la digitalizzazione dell'impianto

Il CXF è la **cartografia vigente**, non una trascrizione del foglio storico. La
geometria discende dall'impianto, ma ha subito un secolo di frazionamenti e
aggiornamenti: nel CXF compaiono particelle con numeri a quattro cifre (1015,
1026, 1041) che sull'impianto non esistono, dove la numerazione arriva a ~336.

**Parte delle linee vettoriali non ha alcuna corrispondenza nell'inchiostro**, e
si vede a occhio nelle sovrapposizioni a piena risoluzione. Non è disallineamento:
è storia. È anche una delle ragioni per cui gli inlier ratio di E2 sono
strutturalmente bassi, e vanno letti con questa chiave.

### 12.2 Le metriche indirette non funzionano su questi dati

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

### 12.3 Validazione, non applicazione

Questo va detto prima che lo chieda chi legge. I dati dell'Agenzia delle Entrate
forniscono una georeferenziazione **già nota**, e *proprio per questo*
permettono di valutare quantitativamente un metodo che altrove — scansioni
d'archivio, catasti preunitari, mappe senza world file — dovrebbe operare senza
alcun riferimento.

È un dataset di **validazione**, non di applicazione. Scrivere che il lavoro
"serve ad allineare mappe storiche disorientate" sarebbe falso *su questi dati*:
qui il riferimento c'è, ed è quello che rende possibile la misura.

### 12.4 Ampiezza del campione

Ogni punto delle curve di E1 è la mediana di 5 prove, una per ritaglio; ogni
cella delle tabelle di E2 aggrega 10 prove. Le non-monotonie che compaiono qua e
là — per esempio nel tasso di successo attorno a degradazione 1.1 — sono rumore
di campionamento, non struttura, e non vanno raccontate come fenomeni.

### 12.5 Un errore documentato nel materiale di partenza

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

## 13. Conclusioni

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
   test troppo severo (§9.3) e il fallimento atteso di Otsu (§6.3).

Un progetto che misura e spiega perché un metodo cede vale più di uno che mostra
solo il caso riuscito. Qui i casi riusciti ci sono, e i punti di rottura sono
documentati con lo stesso rigore.
