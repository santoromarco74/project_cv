# Registrazione di mappe catastali storiche su cartografia moderna

**Progetto d'esame di Computer Vision — Università di Pavia**
Caso di studio: Comune di Varazze (SV), foglio 49, Originale di Impianto dell'Agenzia delle Entrate.

> Le tabelle di questo documento sono generate automaticamente dai risultati sperimentali raccolti in `results/runs.csv`, non trascritte a mano: se un numero cambia, cambia perché è cambiato l'esperimento.

---

## 1. Il problema

Un **Originale di Impianto** è il foglio di mappa disegnato a mano all'atto della
formazione del catasto terreni, nella prima metà del Novecento: un disegno al
tratto su carta, con i confini delle particelle, i corsi d'acqua, le strade,
numeri scritti a penna, e un secolo di ingiallimento e macchie sopra.

Registrarlo su una cartografia moderna significa trovare la trasformazione
geometrica che porta i pixel della scansione storica a coincidere con le
coordinate della mappa attuale. Il problema è più difficile di quanto sembri,
per una ragione precisa: le tecniche standard per riconoscere punti
corrispondenti fra due immagini nascono per le fotografie, dove ogni zona
dell'immagine ha un aspetto diverso dalle altre. Qui invece **entrambe le
immagini sono disegni al tratto**, quasi senza texture, e per di più con un
aspetto completamente diverso — inchiostro sbiadito da un lato, linee
vettoriali nitide dall'altro. È un caso sfavorevole quasi apposta.

Il progetto affronta il problema in due parti, secondo la traccia d'esame
approvata dal docente:

- **A — approccio classico**: pulizia dell'immagine, ricerca di punti
  corrispondenti con i metodi tradizionali (SIFT, ORB) e stima della
  trasformazione per votazione (RANSAC), cercando di spingere questi metodi al
  massimo tramite una preparazione accurata del segnale.
- **B — confronto**: la stessa pipeline, con un solo componente sostituito da
  un metodo di deep learning (LoFTR), per misurare se una rete pre-addestrata
  colma il divario fra i due stili di disegno meglio dei metodi classici.

Il caso di studio è il foglio 49 del Comune di Varazze, scelto perché
l'Agenzia delle Entrate fornisce, insieme alla scansione, anche la sua
posizione geografica esatta e la cartografia vettoriale aggiornata dello
stesso foglio. Questo permette — ed è il punto metodologico su cui si regge
l'intero lavoro — di **misurare l'errore in metri senza dover indovinare la
risposta giusta** (§3).

---

### 1.1 Gli argomenti del corso che il progetto attraversa

La registrazione di due immagini è un **metodo di allineamento**, che nel
programma del corso compare fra i processi di riconoscimento accanto al
confronto diretto e ai metodi basati su proprietà invarianti. Il progetto tocca
tutte e tre queste voci: allinea, usa descrittori invarianti a scala e
rotazione, e mette alla prova il confronto diretto — che su questi dati non
funziona, per la ragione misurata in §6.

Anche il preprocessing di §4 viene dal programma, e non da una scelta di comodo:
le tre strade messe a confronto — soglia globale, contrasto locale, morfologia —
sono tre lezioni distinte. Qui sotto, dove il progetto tocca il corso.

| lezione | argomento | dove compare |
|---|---|---|
| CV02 | segmentazione, immagini binarie, distribuzione bimodale | Otsu come soglia globale, e Sauvola come sua versione locale (§4) |
| CV03 | istogrammi, LUT, operazioni locali | CLAHE, l'alternativa che non binarizza (§4) |
| CV04 | morfologia matematica binaria | apertura e chiusura con elemento strutturante (§4) |
| CV06 | pattern recognition e ricerca visiva | la ricerca di corrispondenze fra le due immagini (§4) |
| CV08 | analisi di regioni, componenti connesse | `rimuovi_componenti` in `preprocess.py`, che elimina le macchie più piccole di una soglia |


## 2. Come funziona il sistema

![Una registrazione passo per passo](../results/figures/m10_passo_passo.png)

Il programma riceve due immagini della stessa zona — la scansione storica e
un raster ricavato dalla mappa vettoriale moderna — e restituisce la
trasformazione che le sovrappone. Il procedimento, illustrato in figura, si
articola in cinque passi.

**Pulizia.** Ogni pixel viene classificato come inchiostro o come carta,
eliminando le variazioni di colore della carta stessa (§4).

**Ricerca di punti riconoscibili.** Il programma cerca, in entrambe le
immagini, punti "distintivi" — un incrocio di confini, l'angolo di un
fabbricato — e per ciascuno calcola un'impronta numerica che ne descrive
l'aspetto locale. È il ruolo dei descrittori SIFT e ORB (§4).

**Abbinamento.** Ogni punto della prima immagine viene accoppiato con il
punto della seconda che ha l'impronta più simile. Su un disegno al tratto
molti incroci si somigliano, e la maggior parte di questi abbinamenti è
sbagliata: nell'esempio in figura solo il 6% è corretto.

**Votazione (RANSAC).** Il programma estrae a caso due abbinamenti alla
volta, calcola quale trasformazione implicherebbero, e conta quanti altri
abbinamenti sarebbero d'accordo con quella stessa trasformazione. Ripetuto
migliaia di volte, il procedimento tiene la trasformazione più votata. È il
passo che rende il metodo utilizzabile anche quando, come qui, la stragrande
maggioranza degli abbinamenti è rumore.

**Verifica.** La trasformazione vincente viene applicata all'immagine
storica, che così si sovrappone al vettoriale: dove i due disegni coincidono
l'overlay appare nero, dove no si vede una frangia colorata.

**Come si giudica il risultato.** Sia la scansione storica sia il file
vettoriale portano con sé un *world file (JGW)*: sei numeri che indicano a quali
coordinate reali corrisponde ogni pixel. Da questi numeri si calcola per via
puramente algebrica la trasformazione **esatta**, senza dover annotare a mano
un solo punto di controllo (§3). Il programma che stima la trasformazione non
vede mai questi numeri — li usa solo il codice di valutazione, separato — così
il confronto fra stima e verità resta un test indipendente e non un esercizio
con le soluzioni già viste.

---

### 2.1 Perché è difficile

Se la procedura sembra ragionevole, resta da capire perché non funzioni sempre.
Tre ragioni, tutte misurate nel capitolo 5:

1. **I due disegni sono fatti di linee sottili, non di texture.** I metodi
   classici per riconoscere punti nascono per le fotografie, dove ogni zona ha un
   aspetto diverso dalle altre. Su un reticolo di confini, un incrocio somiglia a
   tutti gli altri incroci.
2. **I due documenti non rappresentano la stessa realtà.** Il CXF è la mappa
   *di oggi*, il foglio è quella *di un secolo fa*: alcuni confini sono cambiati,
   e ci sono linee vettoriali che nell'inchiostro non esistono (§6).
3. **L'aspetto è completamente diverso**: inchiostro sbiadito irregolare contro
   linee nette uniformi. È quello che in letteratura si chiama *divario di
   dominio*.

### 2.2 Glossario

| termine | significato |
|---|---|
| **registrazione** | sovrapporre due immagini della stessa zona, trovando la trasformazione che porta l'una sull'altra |
| **raster / immagine** | griglia di pixel |
| **vettoriale** | descrizione per coordinate (elenchi di vertici), non per pixel |
| **rasterizzare** | disegnare un vettoriale su un'immagine |
| **world file** (`.jgw`) | sei numeri che legano i pixel di un'immagine alle coordinate sul terreno |
| **georeferenziazione** | il fatto che un'immagine sappia dove si trova sul terreno |
| **luminosità** | il valore di grigio di un pixel: un solo numero, 0 (nero) – 255 (bianco) |
| **gradiente** | quanto e in che direzione la luminosità cambia in UN pixel, confrontandolo coi vicini (§4) |
| **keypoint** | un punto dell'immagine giudicato "riconoscibile" |
| **descrittore** | l'impronta numerica che riassume l'aspetto attorno a un keypoint, costruita raccogliendo i gradienti di molti pixel vicini |
| **SIFT, ORB** | due modi diversi di calcolare keypoint e descrittori |
| **matching** | accoppiare i keypoint di un'immagine con quelli dell'altra |
| **ratio test di Lowe** | il filtro di SIFT: tiene un abbinamento solo se il candidato migliore batte nettamente il secondo (§4) |
| **RANSAC** | la procedura a votazione del passo 4, che trova la risposta giusta anche con pochi dati buoni (dettaglio in §4) |
| **inlier** | un abbinamento che concorda con la trasformazione scelta |
| **inlier ratio** | la percentuale di inlier: quanto erano buoni i dati di partenza |
| **campione minimo** | il numero di corrispondenze che bastano a calcolare esattamente una trasformazione (2, 3 o 4 a seconda della famiglia, §4) |
| **omografia, affine, similarità** | famiglie di trasformazioni con 8, 6 e 4 parametri liberi (§4) |
| **binarizzazione** | decidere, pixel per pixel, se è inchiostro o carta |
| **Otsu, Sauvola** | due modi di scegliere quella soglia (§4) |
| **CLAHE** | tecnica che non binarizza, ma aumenta il contrasto locale (§4) |
| **LoFTR** | rete neurale che abbina due immagini senza cercare prima i keypoint (§5.3) |
| **RMSE** | l'errore medio commesso, qui espresso in metri |
| **E1, E2, E3** | i tre esperimenti: sintetico, reale, comparativo (§5.1, §5.2, §5.3) |

---

## 3. I dati e la verità di riferimento

Dal servizio dell'Agenzia delle Entrate si ottengono, per il foglio 49 di
Varazze:

| file | contenuto |
|---|---|
| `L675_004900.jpg` | scansione dell'Originale di Impianto, **8489 × 5648 px** |
| `L675_004900.jgw` | world file: la georeferenziazione della scansione |
| `L675_004900.txt` | metadati del ricampionamento |
| `L675_004900.cxf` | cartografia **vigente** in formato vettoriale, stesso sistema |

Raster e vettoriale sono entrambi in **Cassini-Soldner zona G0007, origine Forte
Diamante**. Non è un EPSG standard e non serve riproiettare nulla: le coordinate
dei due file sono già confrontabili — un dettaglio non scontato, perché il
servizio distribuisce per lo stesso foglio anche una seconda variante
vettoriale in un sistema diverso, facile da confondere con la prima e già
causa in passato di un pomeriggio perso dietro un allineamento apparentemente
senza senso. Il progetto non usa `pyproj` né `geopandas`.

Il file CXF è un formato testuale elementare, letto da un parser scritto da
zero in una trentina di righe. Un singolo campo di formato, presente solo in
33 record su 871, se ignorato sposta la lettura delle coordinate di qualche
posizione e fa esplodere l'estensione del foglio da 1.6 a 31 chilometri: il
programma verifica automaticamente, dopo ogni lettura, che ogni coordinata
ricada dentro l'area attesa, proprio per intercettare questo tipo di errore
silenzioso.

Il foglio intero è troppo grande per essere utile allo scopo dell'esame, per
cui si lavora su cinque **ritagli** di circa 1000-1200 pixel di lato, scelti
nella fascia collinare dove la mappa vettoriale copre per intero ciò che è
disegnato nell'impianto (sono state escluse le zone di confine
amministrativo, vuote in entrambe le rappresentazioni, e il centro urbano,
dove il catasto terreni non ha competenza e quindi il vettoriale non ha nulla
da confrontare).

| crop      | x0   | y0   | larghezza | altezza | X (m)           | Y (m)           |
|-----------|------|------|-----------|---------|-----------------|-----------------|
| tassarole | 1500 | 300  | 1024      | 1024    | -31098 … -30838 | -11616 … -11355 |
| cannei    | 3100 | 420  | 1024      | 1024    | -30691 … -30431 | -11646 … -11386 |
| ribba     | 3850 | 700  | 1024      | 1024    | -30500 … -30240 | -11717 … -11457 |
| vedra     | 4200 | 1450 | 1024      | 1024    | -30411 … -30151 | -11908 … -11648 |
| aspera    | 5600 | 2600 | 1024      | 1024    | -30055 … -29795 | -12201 … -11940 |

Ogni ritaglio è salvato come PNG accompagnato dal **proprio world file**,
composto per traslazione dell'origine pixel:


$$C' = C + A\cdot x_0 + B\cdot y_0$$

$$F' = F + D\cdot x_0 + E\cdot y_0$$


### 3.1 La ground truth: esatta e gratuita

Questo è il punto metodologico su cui poggia tutto il resto.

Il world file `L675_004900.jgw` contiene sei numeri che definiscono una
trasformazione affine da pixel a coordinate: la dimensione del pixel sui due
assi — **0.254453 m**, cioè la scala 1:2000 — due termini di rotazione, qui
nulli, e le coordinate del centro del primo pixel, (−31480.044315,
−11278.758056). Da questi sei numeri la posizione sul terreno di qualunque
pixel si ottiene con due moltiplicazioni e due somme.

Quindi nel dettaglio il world file `L675_004900.jgw` è una trasformazione affine da pixel a coordinate:

```
0.254453        A → dimensione pixel x (m)
0.0             D → rotazione
0.0             B → rotazione
-0.254453       E → dimensione pixel y (negativa)
-31480.044315   C → x del centro del pixel (0,0)
-11278.758056   F → y del centro del pixel (0,0)
```

cioè
$$ X = A \cdot col + B \cdot row + C \quad\text{e}\quad Y = D \cdot col + E \cdot row + F $$
con risoluzione **0.254453 m/px** (scala 1:2000).

Rasterizzando il CXF su una griglia di cui conosciamo, allo stesso modo, il legame fra pixel e coordinate, si ottengono due trasformazioni affini note, e la trasformazione di riferimento è la loro composizione:

$$
 H_{\text{true}} = W_{\text{moderno}}^{-1} \circ W_{\text{storico}}
$$

**Non è stato annotato un solo punto di controllo a mano.** La ground truth è
analitica, esatta per costruzione, e `evaluate.py` la usa per produrre l'RMSE in
metri su una griglia regolare di checkpoint nell'immagine storica (10×10 punti,
bordi esclusi):

$$e_i = \left\lVert H_{est}\cdot p_i - H_{true}\cdot p_i \right\rVert_2$$

$$\mathrm{RMSE}_m = \sqrt{\frac{1}{N}\sum_{i=1}^{N} e_i^2} \;\cdot\; r_{\mathrm{dest}}$$

dove $N = 100$ sono i checkpoint e $r_{\mathrm{dest}}$ è la risoluzione in metri
per pixel della griglia di arrivo: 0.254453 m/px in E1, dove la coppia è il
ritaglio contro se stesso trasformato, e 0.20 m/px in E2, dove l'arrivo è il
raster del vettoriale (§5.2).

cioè: si applicano entrambe le trasformazioni — quella stimata e quella vera —
agli stessi 100 punti, si misura di quanto le due risposte divergono, e si
converte in metri. È l'unico numero su cui si giudica l'intera pipeline: ogni
tabella dei capitoli successivi è, in ultima analisi, un'aggregazione di questa
formula su configurazioni diverse.

Questa verità di riferimento non è però infinitamente precisa: i metadati del
foglio dichiarano un errore medio di ricampionamento di 0.56 m (massimo 1.28
m, su 76 punti di taratura). È un **pavimento**: un errore misurato sotto
mezzo metro non descrive più la qualità dell'algoritmo, descrive l'incertezza
del riferimento stesso, e va letto con questa avvertenza in tutte le tabelle
del capitolo 5.

La correttezza della composizione di `H_true` è verificata da un test: un
punto trasformato avanti e indietro torna su sé stesso entro $\mathbf{1.1 \cdot 10^{-13}} px$,
contro la soglia dichiarata di $\mathbf{10^{-9}} px$.

---

## 4. Pulizia delle immagini e ricerca delle corrispondenze

**Perché pulire l'immagine.** Su una fotografia normale una soglia unica
potrebbe separare lo sfondo dal soggetto; su un foglio d'archivio no, perché
il colore della carta cambia da una zona all'altra del foglio (ingiallimento,
pieghe, macchie). Il progetto confronta tre modi di preparare l'immagine
prima della ricerca dei punti corrispondenti:

- **Otsu**: una soglia unica per tutta l'immagine, calcolata automaticamente.
  È il metodo di riferimento più semplice, ed è messo in griglia proprio per
  avere un termine di paragone.
- **Sauvola**: una soglia diversa per ogni zona del foglio, ricalcolata
  localmente. Si adatta a un fondo che cambia luminosità, al prezzo di un
  calcolo più oneroso (reso comunque rapido con una tecnica di somme
  precalcolate).
- **CLAHE**: non separa l'immagine in bianco e nero, ma ne aumenta il
  contrasto zona per zona. Serve a verificare l'ipotesi che ridurre
  l'immagine a due soli livelli distrugga proprio le sfumature su cui i
  descrittori si basano.

![Confronto dei tre preprocessing](../results/figures/m5_preprocess_ribba.png)

Contrariamente all'ipotesi di partenza, sui ritagli reali **Otsu non
fallisce**: il foglio è illuminato in modo abbastanza uniforme da non mettere
in crisi una soglia globale. Messo alla prova con un gradiente di
illuminazione sintetico e crescente, però, Otsu classifica come inchiostro
fino a metà dell'immagine (contro carta), mentre Sauvola resta stabile: il
limite teorico di Otsu è reale, semplicemente questo foglio non lo mette alla
prova. Riportare una previsione smentita dai dati, invece di adattare
l'esperimento per confermarla, è parte del criterio di onestà con cui è
condotto il progetto.

**Trovare e abbinare i punti.** SIFT individua i punti "distintivi" cercando
i massimi locali di un filtro sensibile alle macchie (blob), non ai bordi; li
descrive con un vettore di 128 numeri costruito dai gradienti locali, ed è
per progetto invariante a scala e rotazione. Un abbinamento si accetta solo
se il candidato migliore batte nettamente il secondo migliore (il *ratio
test* di Lowe) — un filtro severo, che su un reticolo di linee sottili quasi
privo di macchie lascia poche centinaia di candidati. **ORB** persegue lo
stesso obiettivo in modo più economico — un descrittore binario invece di 128
numeri — e filtra gli abbinamenti in modo meno severo (basta l'accordo
reciproco fra i due punti), lasciandone passare molti di più.

**Scegliere quanto la trasformazione può deformare.** RANSAC cerca la
trasformazione migliore all'interno di una famiglia scelta a priori: una
**similarità** (può spostare, ruotare, ingrandire), un'**affine** (in più può
stirare e inclinare) o un'**omografia** (in più può simulare una prospettiva).
Più libertà sembra sempre meglio, ma non lo è quando la maggior parte degli
abbinamenti è sbagliata: con più parametri liberi diventa più facile trovare
un consenso anche fra dati casuali. Il capitolo 5 mostra che su questo
problema è proprio questa scelta, più del matcher usato, a decidere fra
successo e fallimento.

---

## 5. Risultati sperimentali

I risultati sono organizzati in tre esperimenti crescenti in difficoltà: un
caso sintetico per verificare che il programma sia corretto, il caso reale
per misurarne le prestazioni, e un confronto con un metodo di deep learning.

### 5.1 Il programma è corretto (esperimento sintetico)

Prima di misurare il caso difficile, occorre verificarne uno facile: ogni
ritaglio storico è stato confrontato con **se stesso** trasformato con una
rotazione, scala o prospettiva nota. Qui la trasformazione vera è nota per
costruzione e non c'è alcun divario fra i due stili di disegno: se il
programma sbagliasse anche qui, il problema sarebbe nel codice, non nel
compito.

La trasformazione **nota** con cui si costruisce la coppia si applica così:
una rotazione di angolo θ e una scala s attorno al centro del ritaglio, poi
una traslazione (tx, ty) — nei casi più impegnativi anche una lieve
componente prospettica:

```
H (similarity):
[ s·cosθ   -s·sinθ   tx ]
[ s·sinθ    s·cosθ   ty ]
[ 0         0         1 ]
```

È la stessa H_true del capitolo 3, qui non ricavata dal world file ma
fissata a priori: nota senza incertezza, così un errore misurato in questo
esperimento è per forza del codice, non del riferimento.

| matcher | preprocess | prove | successo_pct | rmse_px_mediano_ok | rmse_px_max_ok | inlier_ratio | match_medi | t_ms |
|---------|------------|-------|--------------|--------------------|----------------|--------------|------------|------|
| loftr   | none       | 80    | 42.5         | 0.208              | 0.963          | 0.619        | 1909       | 3954 |
| orb     | clahe      | 80    | 68.8         | 0.499              | 1.0            | 0.619        | 2503       | 116  |
| orb     | none       | 80    | 67.5         | 0.438              | 0.999          | 0.608        | 2377       | 107  |
| orb     | sauvola    | 80    | 55.0         | 0.479              | 0.995          | 0.486        | 1945       | 103  |
| sift    | clahe      | 80    | 82.5         | 0.186              | 0.927          | 0.752        | 2289       | 377  |
| sift    | none       | 80    | 80.0         | 0.182              | 0.969          | 0.691        | 1709       | 323  |
| sift    | sauvola    | 80    | 73.8         | 0.206              | 0.927          | 0.614        | 974        | 371  |


Nel caso peggiore l'errore recuperato resta nell'ordine del pixel — pochi
centimetri sul terreno, ben sotto il pavimento di 0.5 m del capitolo 3. Con
un'unica eccezione degna di nota, ripresa al punto 5.3: LoFTR perde
completamente la trasformazione quando la rotazione supera i 30 gradi, mentre
SIFT e ORB restano precisi fino a rotazioni di 90 gradi. La ragione è
strutturale — SIFT e ORB calcolano esplicitamente un orientamento di
riferimento per ogni punto, LoFTR no, ed è quindi vincolato a ciò che ha
incontrato durante l'addestramento su fotografie ordinarie, dove rotazioni
così ampie sono rare.

### 5.2 Il caso reale: storico contro cartografia vettoriale

Il confronto vero è fra ogni ritaglio storico e il raster ottenuto disegnando
il file vettoriale sulla stessa zona. Qui la posizione reciproca fra le due
immagini è nota solo attraverso i world file, mai vista dall'algoritmo.

| matcher | preprocess       | modello    | prove | successo_pct | rmse_m_mediano | rmse_m_minimo | inlier_ratio | match_mediani |
|---------|------------------|------------|-------|--------------|----------------|---------------|--------------|---------------|
| loftr   | clahe            | affine     | 10    | 0.0          | 568.2          | 98.409        | 0.464        | 6             |
| loftr   | clahe            | homography | 10    | 0.0          | 255.6          | 119.479       | 0.619        | 6             |
| loftr   | clahe            | similarity | 10    | 0.0          | 189.12         | 133.99        | 0.31         | 6             |
| loftr   | sauvola          | affine     | 10    | 80.0         | 0.6            | 0.35          | 0.275        | 385           |
| loftr   | sauvola          | homography | 10    | 70.0         | 0.8            | 0.491         | 0.273        | 385           |
| loftr   | sauvola          | similarity | 10    | 90.0         | 0.35           | 0.134         | 0.288        | 385           |
| loftr   | sauvola+chiusura | affine     | 10    | 80.0         | 0.49           | 0.348         | 0.288        | 390           |
| loftr   | sauvola+chiusura | homography | 10    | 80.0         | 0.91           | 0.371         | 0.279        | 390           |
| loftr   | sauvola+chiusura | similarity | 10    | 90.0         | 0.42           | 0.257         | 0.278        | 390           |
| orb     | clahe            | affine     | 10    | 0.0          | 83.17          | 20.138        | 0.009        | 777           |
| orb     | clahe            | homography | 10    | 0.0          | 132.11         | 2.453         | 0.012        | 777           |
| orb     | clahe            | similarity | 10    | 60.0         | 0.65           | 0.21          | 0.016        | 777           |
| orb     | sauvola          | affine     | 10    | 70.0         | 1.17           | 0.246         | 0.034        | 752           |
| orb     | sauvola          | homography | 10    | 50.0         | 2.33           | 0.555         | 0.022        | 752           |
| orb     | sauvola          | similarity | 10    | 90.0         | 0.38           | 0.222         | 0.049        | 752           |
| orb     | sauvola+chiusura | affine     | 10    | 50.0         | 2.76           | 0.225         | 0.032        | 743           |
| orb     | sauvola+chiusura | homography | 10    | 30.0         | 13.4           | 0.715         | 0.022        | 743           |
| orb     | sauvola+chiusura | similarity | 10    | 90.0         | 0.33           | 0.165         | 0.049        | 743           |
| sift    | clahe            | affine     | 10    | 0.0          | 212.1          | 140.125       | 0.09         | 82            |
| sift    | clahe            | homography | 10    | 0.0          | 133.6          | 109.56        | 0.243        | 82            |
| sift    | clahe            | similarity | 10    | 0.0          | 133.96         | 108.773       | 0.268        | 82            |
| sift    | sauvola          | affine     | 10    | 40.0         | 66.83          | 0.238         | 0.07         | 146           |
| sift    | sauvola          | homography | 10    | 10.0         | 131.38         | 0.31          | 0.13         | 146           |
| sift    | sauvola          | similarity | 10    | 20.0         | 128.39         | 0.179         | 0.105        | 146           |
| sift    | sauvola+chiusura | affine     | 10    | 10.0         | 108.57         | 0.959         | 0.078        | 130           |
| sift    | sauvola+chiusura | homography | 10    | 0.0          | 122.86         | 7.008         | 0.117        | 130           |
| sift    | sauvola+chiusura | similarity | 10    | 30.0         | 118.73         | 0.322         | 0.084        | 130           |


Il risultato più importante del progetto è che **la registrazione riesce**,
ma con una configurazione controintuitiva: non SIFT ma **ORB**, non
l'omografia più flessibile ma la **similarità** più vincolata, con la
binarizzazione **Sauvola**. In questa configurazione il 90% delle prove ha
successo, con un errore mediano di 0.33 m — sotto il pavimento
dell'incertezza del riferimento, cioè al limite di quanto questo confronto
può misurare.

![Verifica dell'allineamento a piena risoluzione](../results/figures/m8_verifica_ribba.png)

Perché ORB batte SIFT è chiarito da un numero solo: il filtro di ORB lascia
passare centinaia di abbinamenti in più rispetto al filtro severo di SIFT. Su
un problema dove solo l'1-10% degli abbinamenti è corretto, RANSAC ha
bisogno soprattutto di **quantità di candidati**, non di candidati già
puliti. Si potrebbe pensare che basti rendere meno severo anche il filtro di
SIFT (il *ratio test*, §4): è stato verificato, ed è falso.

| ratio | match_mediani | inlier_mediani | inlier_ratio | rmse_m_mediano | riuscite |
|-------|---------------|----------------|--------------|----------------|----------|
| 0.75  | 168           | 11             | 0.1089       | 135.4          | 1/5      |
| 0.85  | 544           | 43             | 0.0773       | 135.4          | 0/5      |
| 0.95  | 1764          | 106            | 0.0641       | 128.6          | 0/5      |
| 0.99  | 3191          | 155            | 0.057        | 128.8          | 0/5      |


Allentando la soglia da 0.75 a 0.99 le corrispondenze passano da 168 a oltre
3000, ma l'errore mediano resta a oltre 100 metri e le prove riuscite non
aumentano: i candidati aggiuntivi sono rumore puro, non segnale che il filtro
teneva nascosto. Il limite di SIFT su questi dati sta nel descrittore, non nel
filtro che lo seleziona.

Perché la similarità batte l'omografia si spiega in modo analogo: con inlier
ratio così bassi, concedere più libertà geometrica allo stimatore significa
solo dargli più modi di accordarsi con dati sbagliati.

| fattore            | valore     | prove | successo_pct | rmse_m_mediano | inlier_ratio |
|--------------------|------------|-------|--------------|----------------|--------------|
| modello geometrico | affine     | 90    | 36.7         | 28.04          | 0.076        |
| modello geometrico | homography | 90    | 26.7         | 109.17         | 0.135        |
| modello geometrico | similarity | 90    | 52.2         | 0.88           | 0.11         |
| codici CXF         | 18         | 135   | 34.8         | 50.73          | 0.106        |
| codici CXF         | 18+12      | 135   | 42.2         | 28.04          | 0.104        |


Passando dall'omografia alla similarità il tasso di successo sale dal 27% al
52% a parità di tutto il resto — è, fra i risultati del progetto, quello con
la morale più generale: **su dati cross-domain con pochi abbinamenti
corretti, il modello geometrico più vincolato non è una semplificazione, è
una necessità**. La stessa tabella mostra anche che disegnare nel raster
moderno anche acque e strade, non solo i confini di particella, aiuta
(42% di successo contro 35%): sono la parte del vettoriale dove il tratto
storico è più marcato.

Va detto, per onestà, che una parte del vettoriale non ha alcuna
corrispondenza nell'inchiostro storico: il file CXF è la mappa **di oggi**,
non una trascrizione dell'impianto, e un secolo di frazionamenti ha
introdotto confini che nel 1930 non esistevano. Non è un errore di
allineamento, ma va tenuto presente nel leggere le percentuali di
corrispondenze corrette, strutturalmente basse.

### 5.3 Il confronto con il deep learning

L'ultimo esperimento sostituisce, nella stessa identica pipeline, il matcher
classico con **LoFTR**, un metodo di deep learning pre-addestrato che non
cerca prima i punti distintivi e poi li descrive, ma confronta direttamente
le due immagini a griglia. L'idea è che questo approccio possa reggere meglio
proprio dove i metodi classici soffrono, cioè dove non ci sono angoli o
macchie ben definiti da individuare.

| esperimento | matcher | config                        | prove | successo_pct | rmse_m_mediano_ok | inlier_ratio | match_mediani | t_ms |
|-------------|---------|-------------------------------|-------|--------------|-------------------|--------------|---------------|------|
| E1          | loftr   | none / homography             | 80    | 42.5         | 0.053             | 0.622        | 1033          | 3875 |
| E1          | orb     | clahe / homography            | 80    | 68.8         | 0.127             | 0.705        | 2511          | 114  |
| E1          | sift    | clahe / homography            | 80    | 82.5         | 0.047             | 0.873        | 1497          | 378  |
| E2          | loftr   | sauvola / similarity          | 10    | 90.0         | 0.342             | 0.288        | 385           | 3238 |
| E2          | orb     | sauvola+chiusura / similarity | 10    | 90.0         | 0.284             | 0.049        | 743           | 87   |
| E2          | sift    | sauvola / affine              | 10    | 40.0         | 0.861             | 0.07         | 146           | 362  |


![Confronto fra metodi classici e rete neurale](../results/figures/m9_e3_confronto.png)

Il risultato è netto ma non nella direzione attesa: **LoFTR pareggia ORB sul
tasso di successo, con un errore leggermente peggiore e un tempo di calcolo
di un ordine di grandezza superiore**. La promessa del metodo neurale non si
realizza su questo tipo di dati. È però nettamente superiore in un aspetto:
la qualità delle corrispondenze che produce è pulita fin da subito (fino al
30% di abbinamenti corretti, contro il 2-13% dei metodi classici), a
dimostrazione che il collo di bottiglia dell'intero problema è altrove — nel
divario fra i due stili grafici, non nella capacità di descrivere un punto.

Anche LoFTR, inoltre, **funziona solo se l'immagine è binarizzata**: con
CLAHE non trova quasi nessuna corrispondenza utile e non registra mai; con
Sauvola arriva al 90% di successo, esattamente come ORB. Il vantaggio del
pre-addestramento su milioni di fotografie non basta da solo a colmare il
divario di dominio: a farlo è, ancora una volta, il preprocessing.

---

## 6. Limiti dei risultati

Tre avvertenze, dichiarate qui perché condizionano come leggere ogni numero
del capitolo precedente.

**Il vettoriale non è la digitalizzazione dello storico.** È la cartografia
in vigore oggi, aggiornata attraverso un secolo di frazionamenti: alcune
linee del vettoriale non hanno alcuna controparte nell'inchiostro
dell'impianto, non per errore ma per la storia della mappa stessa. È una
delle ragioni per cui le percentuali di corrispondenze corrette restano
basse anche nelle configurazioni che registrano bene.

**Il dataset è di validazione, non di applicazione diretta.** L'Agenzia delle
Entrate fornisce, per questo foglio, una posizione già nota — ed è proprio
questo a rendere possibile misurare l'errore in metri. Su un archivio storico
reale, senza world file né riferimento vettoriale, non si potrebbe applicare
lo stesso criterio di verifica: resterebbe l'ispezione visiva, più lenta e
meno conclusiva. Le decisioni validate qui — modello geometrico vincolato,
binarizzazione locale, diffidenza verso le metriche di allineamento
indirette — restano valide altrove; il modo di verificarle in metri, no.

**Le misure indirette di allineamento non sono affidabili su questi dati.**
Un primo tentativo di verificare l'allineamento con la correlazione fra
immagini, prima di adottare la verità di riferimento del capitolo 3, ha
prodotto risultati che sembravano coerenti ma erano falsi: su un disegno al
tratto, righe tracciate a caso ottengono un punteggio di correlazione già
alto quasi quanto il vettoriale vero, perché l'inchiostro è sparso ovunque
sul foglio. Da qui la regola seguita in tutto il progetto: l'unica misura
accettata è l'errore in metri contro la verità di riferimento esatta, mai una
misura indiretta senza un confronto con un caso casuale.

---

## 7. Uso pratico e conclusioni

**Lo stack tecnologico.** Il nucleo classico è scritto in Python 3.12 con
OpenCV (SIFT, ORB, RANSAC, deformazione dell'immagine), NumPy, pandas,
Matplotlib e Pillow — senza librerie geospaziali come `pyproj` o
`geopandas`: raster e vettoriale sono già nello stesso sistema di
riferimento, quindi non c'è nulla da riproiettare. Il solo componente B
(LoFTR) richiede `torch` e `kornia`, caricati solo se il matcher scelto è
`loftr`: senza quella scelta la pipeline classica gira anche su una
macchina senza le due librerie installate. I pesi del modello (~90 MB)
sono distribuiti insieme al progetto, non scaricati al primo avvio.

### 7.1 Installazione

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Per il solo approccio classico bastano `numpy`, `opencv-python`, `Pillow`,
`scipy`, `pandas`, `matplotlib`.
`torch` e `kornia` servono **solo** a `--matcher loftr`; su CPU conviene l'indice dedicato:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install kornia==0.7.3
python -m scripts.scarica_pesi          # pesi LoFTR + verifica del checksum
```

I dati cartografici non sono versionati: `data/README.md` documenta come
ricostruirli.

### 7.2 La CLI

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
  --esperimento <str>       etichetta nella colonna `esperimento` del CSV
  --crop <str>              etichetta nella colonna `crop` del CSV
  --verbose                 descrizione dettagliata delle operazioni
  --help                    questa lista di comandi
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

### 7.3 Preparazione dei dati ed esperimenti

```bash
# ritagli dal foglio (PNG + world file affiancato)
python -m src.prep.crop

# ispezione e verifica del parser CXF
python -m src.io_geo

# rasterizzazione del vettoriale sulla zona di un ritaglio
python -m src.prep.rasterize --crop ribba --codici 18,12
python -m experiments.m7_rasterize_check --crop ribba

# confronto dei preprocessing
python -m experiments.m5_preprocess --crop tutti --dettaglio ribba

# gli esperimenti: E1, E2, la diagnosi del ratio, E3
python -m experiments.m6_e1_completo --riparti
python -m experiments.m8_e2_griglia
python -m experiments.m8_e2_griglia --diagnosi-ratio
python -m experiments.m9_e3_loftr

# figure e tabelle, tutte generate dal CSV
python -m src.report --csv results/runs.csv

```

### 7.4 Riprodurre tutto in un comando

I comandi di §7.3 vanno eseguiti in quest'ordine, e l'ordine non è arbitrario:
i ritagli prima della rasterizzazione, la rasterizzazione prima di E2, gli
esperimenti prima delle tabelle, le tabelle prima di questo documento. Eseguirli
a mano funziona, ma un passo dimenticato non dà errore: produce un CSV parziale
e tabelle che sembrano complete.

```bash
python -m scripts.riproduci --controlla   # verifica le precondizioni, non esegue
python -m scripts.riproduci --lista       # le fasi, in ordine, con i tempi
python -m scripts.riproduci               # tutto tranne E3      (~18 min)
python -m scripts.riproduci --con-loftr   # tutto, E3 compreso   (~60 min)
python -m scripts.riproduci --da verifica-raster   # senza i dati AdE, dai soli ritagli
```

Lo script stampa ogni comando prima di eseguirlo — il log di una corsa è la
versione eseguita di §7.3 — e dopo ogni fase verifica che gli artefatti attesi
esistano davvero: un comando che esce con codice 0 senza aver scritto quello che
doveva ferma la corsa, che riprende con `--da <fase>`. In coda controlla che ogni
figura citata da questa relazione sia stata prodotta e che nessun segnaposto di
tabella sia rimasto vuoto.

Il controllo delle precondizioni non è formalità. Gli esperimenti di E1 ed E3
convertono l'errore in metri leggendo la risoluzione da un world file, e se
nessuno è disponibile l'errore in metri resta indefinito per ogni riga:
l'esperimento gira fino in fondo e conclude "0 riuscite". Si legge come un
algoritmo che fallisce, ed è invece un file assente, e la difesa è verificare prima.

Il controllo è però **per fase**, e la distinzione conta proprio alla consegna.
Le scansioni catastali non sono ridistribuibili, i ritagli sì: chi riceve il
progetto senza `data/raw/` ha comunque in `data/crops/` i ritagli, i raster del
vettoriale e i rispettivi world file. Solo `crop`, `cxf` e `rasterize` aprono
le scansioni; le altre nove fasi no.

---

**In sintesi:**

1. La pipeline è verificata corretta sul caso sintetico, con errore
   sub-pixel.
2. Sul caso reale la registrazione riesce (90% delle prove, errore mediano
   0.33 m), ma con la combinazione meno prevedibile: descrittore economico
   (ORB), modello geometrico vincolato (similarità), binarizzazione locale
   (Sauvola).
3. Il fattore più determinante non è il matcher scelto, ma **quanta libertà
   si concede al modello geometrico**: con pochi abbinamenti corretti,
   vincolare è necessario, non un compromesso.
4. Un matcher di deep learning pre-addestrato (LoFTR) non supera i metodi
   classici su questo compito: pareggia il migliore di essi, a un costo di
   calcolo molto maggiore, e produce corrispondenze più pulite senza che
   questo si traduca in un risultato finale migliore.
5. Il preprocessing conta più della scelta dell'algoritmo di matching, classico
   o neurale: è la binarizzazione locale a rendere il problema trattabile per
   tutti e tre i metodi messi alla prova.

Per concludere il progetto misura, e spiega, dove i metodi classici cedono sul confronto fra
stili grafici così diversi — e dove reggono meglio del previsto. Quello che
vale oltre questo foglio non sono le cifre in metri, legate a questa scala e
a questo vettoriale, ma le decisioni che le hanno prodotte: vincolare il
modello geometrico quando i dati sono rumorosi, ripulire il segnale con una
soglia locale, e non fidarsi di una misura indiretta che non batta
nettamente un caso casuale.
