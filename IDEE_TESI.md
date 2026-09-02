# IDEE_TESI

Raccoglitore previsto da CLAUDE.md §11: *"Materiale da tesi: annotare qui e proseguire."*

Serve a **non** allargare il progetto d'esame. Ogni voce qui dentro è qualcosa
che è emerso lavorando, che è stato riconosciuto come interessante e che è stato
deliberatamente lasciato fuori. Non è una lista di desideri: ogni punto parte da
un fatto misurato in questo progetto e dice quale domanda apre e cosa servirebbe
per affrontarla.

Le voci più forti sono nel §1, perché nascono da risultati che contraddicono
l'aspettativa di partenza. Il §5 è altrettanto utile: sono le strade già
percorse e già chiuse, e riaprirle sarebbe uno spreco.

---

## 1. Le domande che i risultati lasciano aperte

### 1.1 Perché la binarizzazione colma un divario di dominio che il pre-addestramento non colma

**Il fatto.** Su E2, LoFTR con CLAHE ottiene 7 corrispondenze mediane e un tasso
di successo fra 0 e 10%; con Sauvola ne ottiene 477 e sale al 60-80%, e con
Sauvola più chiusura 487 e fino al 90%. La dipendenza dal preprocessing è
condivisa: SIFT con CLAHE non riesce mai, in nessuno dei tre modelli, e ORB con
CLAHE riesce solo con la similarità (60%) mentre affine e omografia vanno a zero.
Il risultato più interessante della componente comparativa non è che LoFTR vinca
o perda, è che **tutti e tre i matcher dipendono dalla stessa cosa**, e quella
cosa è il preprocessing.

**La domanda.** Una rete addestrata su immagini naturali porta con sé
un'invarianza che sul disegno al tratto non serve, o addirittura danneggia. La
binarizzazione funziona perché rimuove la variabile su cui quell'invarianza è
mal calibrata — la texture della carta — oppure perché rende i due domini
letteralmente più simili nello spazio delle feature? Sono due spiegazioni
diverse e distinguibili: la seconda prevede che la distanza fra le distribuzioni
di feature dei due domini si riduca dopo Sauvola, la prima no.

**Cosa servirebbe.** Misurare la distanza fra le distribuzioni di feature (MMD,
o la distanza di Fréchet sulle attivazioni intermedie di LoFTR) prima e dopo ogni
preprocessing, e correlarla con il tasso di successo. È un esperimento fattibile
con i dati già in casa, ed è il naturale seguito di §10.3 della relazione.

**Perché è materiale da tesi e non da esame.** Richiede di aprire la rete e
guardarci dentro, non di usarla. Fuori dall'approccio A+B approvato.

### 1.2 Che cosa serve davvero a RANSAC: candidati o candidati puliti

**Il fatto.** Su E2, SIFT si ferma al 30% di successo con 74-121 corrispondenze,
ORB arriva al 90% con circa 730. L'ipotesi ovvia — il ratio test di Lowe a 0.75,
tarato su immagini dello stesso dominio, è troppo severo — **è stata verificata
ed è falsa**: allentando fino a 0.99 le corrispondenze passano da ~100 a ~2500 e
l'RMSE resta a centinaia di metri, peggiorando su un ritaglio da 0.64 m a 163 m.
I match aggiuntivi non contengono segnale.

**La domanda.** Se il limite non è nel filtro ma nei descrittori, che cosa rende
i descrittori di ORB utilizzabili dove quelli di SIFT non lo sono, a parità di
immagine? L'ipotesi da testare è che non sia una questione di qualità ma di
**densità di campionamento**: FAST accetta molti più punti candidati, e su un
reticolo di linee la ridondanza compensa l'ambiguità del singolo descrittore.
Se è così, un SIFT con soglia di detezione abbassata fino a produrre ~700
corrispondenze dovrebbe avvicinarsi a ORB — e se non lo fa, l'ipotesi cade.

**Cosa servirebbe.** Uno sweep congiunto su soglia di detezione e ratio, a
parità di numero di corrispondenze in uscita. È un esperimento pulito, con una
variabile controllata, e la risposta è interessante in entrambi i versi.

### 1.3 Descrittori per disegni al tratto

**Il fatto.** Entrambe le immagini sono *line drawings* quasi privi di texture,
il caso peggiore per descrittori a blob e a corner. Il progetto lo ha misurato:
inlier ratio dell'1-5% per ORB su E2, e RANSAC che riesce comunque perché ha
massa di candidati su cui votare.

**La domanda.** Su questi dati la struttura informativa non sta nei punti, sta
nella **topologia del reticolo**: quali particelle confinano con quali, con
quanti lati, in che disposizione. È informazione che nessun descrittore locale
guarda. Un abbinamento basato sul grafo di adiacenza delle particelle — nodi le
particelle, archi i confini condivisi — sarebbe invariante a scala, rotazione e
qualità del tratto per costruzione.

**Cosa servirebbe.** Estrazione affidabile delle regioni chiuse dall'immagine
storica, che è il punto difficile: il tratto a penna si interrompe, e un confine
interrotto fonde due particelle in una. Da qui il legame con §4.1.

**Attenzione.** È il filone più promettente e anche il più impegnativo. Sarebbe
una tesi intera, non un capitolo.

### 1.4 Il modello geometrico conta più del matcher

**Il fatto.** Su E2, passando da omografia a similarità il tasso di successo va
dal 18.9% al 53.3%, a parità di corrispondenze. L'affine sta in mezzo al 32.2%.
Con inlier ratio bassi, vincolare è necessario: gradi di libertà in più
significano capacità di adattarsi anche alle corrispondenze sbagliate.

**La domanda.** La scelta del modello è fatta a mano e fissata prima. Un
criterio automatico — stimare dal più vincolato al più libero e accettare il
grado di libertà aggiuntivo solo se il guadagno supera quello atteso per caso —
renderebbe la pipeline utilizzabile senza sapere in anticipo quanto è deformato
il foglio. È selezione del modello classica (test F, o un criterio informativo),
applicata a un caso in cui la penalizzazione per l'overfitting è misurabile.

**Cosa servirebbe.** Poco: i dati ci sono già. È forse il proseguimento più
economico dell'intero elenco.

### 1.5 Il muro della rotazione, e cosa ci sta dietro

**Il fatto.** Su E1 senza degradazione, dove l'unica variabile è la
trasformazione geometrica, l'RMSE mediano di LoFTR per ampiezza di rotazione è
0.029 px a 0°, 0.200 a 15°, **1.111 a 30°, 3583 a 45° e 12047 a 90°**. SIFT e
ORB, sulle stesse identiche prove, restano sotto il pixel fino a 90°.

**La domanda.** La spiegazione strutturale è chiara — SIFT e ORB stimano un
orientamento per keypoint e ruotano il descrittore, LoFTR non ha keypoint e
l'invarianza può solo averla imparata — ma apre una domanda progettuale: la si
può restituire dall'esterno, senza riaddestrare? Due strade testabili con i
dati già in casa: stimare l'orientamento dominante dell'immagine (dalla
distribuzione dei gradienti, o dall'orientamento prevalente del reticolo di
confini) e pre-ruotare prima di dare in pasto a LoFTR; oppure lanciare LoFTR su
un ventaglio di rotazioni e tenere l'ipotesi con più inlier. La seconda costa
poco da scrivere e molto da eseguire, la prima il contrario.

**Perché conta.** È il punto in cui il classico batte il neurale per una
ragione di progetto e non di taratura, e su questi dati non si vede perché la
rotazione fra storico e vettoriale è piccola. Su un archivio vero — fogli
scansionati come capita — sarebbe il primo problema da risolvere.

### 1.6 Fragilità del neurale alla degradazione

**Il fatto.** Su E1 senza preprocessing, al crescere della degradazione il tasso
di successo di LoFTR va 100% → 100% → 20% → 0%, mentre SIFT tiene 100% → 80% →
60% → 40%. **LoFTR crolla prima di entrambi i classici.** Il rumore gaussiano
non appartiene alla distribuzione su cui è stato addestrato.

**La domanda.** Quanto di questa fragilità è specifico del rumore gaussiano — il
degrado sintetico di `prep/synth.py` — e quanto sopravvive a degradazioni
realistiche di carta d'archivio: pieghe, macchie di umidità, sbiadimento
disomogeneo, strappi? Il degrado sintetico è comodo perché parametrico, ma è
proprio la sua distanza dal degrado vero a rendere il risultato meno solido di
quanto sembri.

**Cosa servirebbe.** Un modello di degradazione derivato da scansioni reali
degradate, non da rumore additivo. E questo richiede scansioni degradate
annotate, cioè il problema di §3.2.

---

## 2. Il problema vero: registrare senza riferimento

Questo è **l'inquadramento onesto** già dichiarato in §12.3 della relazione, ed è
la cosa più importante di questo file.

I dati dell'Agenzia delle Entrate hanno georeferenziazione nota, ed è *proprio
per questo* che permettono di valutare quantitativamente il metodo. Sono un
dataset di **validazione, non di applicazione**: su questi dati specifici la
registrazione non serve a nessuno, perché il world file c'è già.

L'applicazione vera è dove il riferimento **non** esiste: scansioni d'archivio
senza world file, catasti preunitari, mappe storiche digitalizzate da biblioteche
e archivi di stato. Lì il metodo servirebbe davvero, e lì mancano due cose che
qui erano gratis.

### 2.1 Come si dichiara di aver registrato bene, senza ground truth

Senza `H_true` non si può calcolare l'RMSE, e §5.5 ha già dimostrato che le
metriche indirette **non funzionano** su questi dati: correlazione incrociata e
chamfer matching su distance transform hanno prodotto falsi positivi
convincenti, con ottimi incoerenti fra zone e profili apparentemente sistematici
che confermavano un disallineamento inesistente.

La domanda della tesi è quindi: **esiste una misura di confidenza calcolabile
senza riferimento** che discrimini una registrazione buona da una sbagliata su
disegni al tratto? Candidati da testare: la consistenza fra stime indipendenti su
sotto-regioni disgiunte (se quattro quadranti concordano sulla stessa `H`, è
difficile che sia un caso), la stabilità rispetto al seed e ai parametri, la
distribuzione spaziale degli inlier.

Il progetto attuale offre il banco di prova ideale: **si può calcolare la
confidenza senza riferimento e poi verificarla contro l'RMSE vero**, che qui è
noto. È una validazione della metrica, non della registrazione — ed è esattamente
il tipo di cosa che questi dati permettono e che altri dataset no.

### 2.2 Il riferimento ha un pavimento, e va trattato come tale

Il TXT dichiara: foglio ricampionato su **76 coppie omologhe**, scarto massimo
**1.28 m**, scarto medio **0.56 m**, deviazione standard **0.25 m**. Un RMSE
sotto 0.56 m misura il rumore del riferimento, non l'errore dell'algoritmo — e
la configurazione migliore di E2 arriva a **0.41 m**, cioè sotto il pavimento.

La domanda: come si confrontano due metodi che lavorano entrambi **sotto** la
risoluzione del proprio metro di paragone? È un problema statistico reale, non
un cavillo, e la risposta ovvia — servono più coppie omologhe — non è
disponibile. Un approccio possibile è la stima congiunta dell'errore
dell'algoritmo e dell'incertezza del riferimento su più fogli, dove il secondo
termine è condiviso.

---

## 3. Estensioni di scala e di dominio

### 3.1 Dal ritaglio al foglio, e dal foglio al mosaico

CLAUDE.md I10 limita il progetto a ritagli di ~1000-1200 px: il foglio 49 è
8489×5648 px, e tiling e gestione della memoria non aggiungono nulla al tema
d'esame. Restano fuori:

- **registrazione a piena risoluzione** con strategia gerarchica (stima grossolana
  su immagine ridotta, raffinamento locale a piena risoluzione);
- **deformazioni non rigide**: la carta si ritira in modo disomogeneo, e una
  singola omografia non può descriverlo. Thin-plate spline o campo di
  deformazione vincolato, con il problema di stimarli quando gli inlier sono il
  2-5%;
- **mosaicatura di fogli adiacenti**, con vincoli di continuità sui bordi
  condivisi: la registrazione congiunta di N fogli è più vincolata di N
  registrazioni indipendenti, e questo dovrebbe alzare l'accuratezza di tutte.

### 3.2 Generalizzazione: un solo foglio non è un campione

Tutti i risultati vengono da **un comune, un foglio, cinque ritagli**, con le
curve di E1 mediate su 5 prove e le celle di E2 su 10. La relazione lo dichiara
in §12.4. Non si sa nulla su:

- fogli di altri comuni, con altra mano, altro stato di conservazione;
- scale diverse dal 1:2000;
- zone urbane, escluse per costruzione (§5.6) perché il catasto terreni rimanda
  al catasto urbano e il vettoriale non le copre;
- fogli in cui la divergenza di contenuto è maggiore o minore di questo.

Una tesi ha bisogno di decine di fogli. È soprattutto un lavoro di raccolta
dati, e va messo in conto come tale.

---

## 4. Il problema di contenuto, che è storico e non geometrico

### 4.1 Parte del vettoriale non ha corrispondenza nell'inchiostro

Il CXF è la **cartografia vigente**, non la digitalizzazione dell'impianto. La
geometria discende dall'impianto ma ha un secolo di aggiornamenti: nel CXF ci
sono particelle con numeri a quattro cifre (`1015`, `1026`, `1041`) che
sull'impianto non esistono, dove la numerazione arriva a ~336.

**Conseguenza teorica.** Gli outlier di questo problema non sono rumore casuale:
sono **struttura reale che appartiene a una sola delle due immagini**. RANSAC
assume outlier non informativi; qui gli outlier sono linee coerenti, allineate
fra loro, che formano un reticolo plausibile. È il caso in cui l'ipotesi di
RANSAC è violata nel modo peggiore, e il fatto che funzioni comunque merita una
spiegazione più solida di "c'erano abbastanza inlier".

**La domanda.** Stima robusta con outlier strutturati e non casuali. E il
rovescio, molto più interessante: se si sa registrare *nonostante* la divergenza,
si può usare il residuo per **individuare** la divergenza — cioè per datare le
modifiche catastali confrontando impianto e vigente. Da strumento di
registrazione a strumento di analisi storica.

### 4.2 Il resto della lista fuori scope

Da CLAUDE.md §11, tutto materiale ingegneristico e non di ricerca, ma che una
tesi applicativa potrebbe legittimamente includere:

- **OCR dei numeri di particella** sull'impianto, e collegamento
  particella ↔ partita. Reso interessante dal fatto che i numeri sono scritti a
  mano, con grafie d'epoca, e che l'output è vincolato: il numero letto deve
  esistere nel registro. Il vincolo trasforma un problema di OCR aperto in un
  problema di classificazione su un insieme noto.
- **PostGIS, API, viewer, GUI**: consultazione e sovrapposizione interattiva.
- **Docker, CI, packaging**: distribuzione a un'utenza reale.
- **Studi con utenti**: quanto tempo risparmia a un operatore di conservatoria.

---

## 5. Strade già percorse e già chiuse

Il valore di questa sezione è negativo e per questo alto: sono ore già spese.

**Correlazione incrociata e chamfer matching su distance transform come metriche
di allineamento.** Non funzionano su questi dati, e non "funzionano male": danno
**falsi positivi convincenti**. Shift ottimali incoerenti fra zone (−238, −330,
+11 px), ottimi saturati sui bordi della finestra di ricerca, profili
apparentemente sistematici che confermavano un disallineamento inesistente. La
diagnosi è quantitativa: **linee tirate a caso sul foglio ottengono l'11.5% dei
pixel entro 2 px dall'inchiostro, il vettoriale vero il 16-20%**. Con quel
rapporto segnale/rumore nessuna di quelle metriche discrimina. La carta ha
inchiostro sparso ovunque — testi, simboli, tratteggi, grana, macchie — e la
superficie di correlazione su disegni al tratto è piatta e multi-picco.

Se una tesi vuole riprovarci, deve prima spiegare perché il baseline casuale
questa volta sarebbe diverso.

**Allentare il ratio test per recuperare SIFT.** Verificato fino a 0.99: le
corrispondenze crescono di 25 volte, l'RMSE resta a centinaia di metri. I match
aggiuntivi non contengono segnale.

**Fine-tuning di LoFTR su questi dati.** Escluso da CLAUDE.md §11, e a ragione:
non esiste un insieme di addestramento. Le coppie di training andrebbero
generate dalla ground truth stessa, e allora si addestrerebbe la rete su ciò che
poi si userebbe per valutarla. Il problema è metodologico prima che pratico, e
la via d'uscita è §3.2 — più fogli, con separazione netta fra fogli di
addestramento e fogli di test.

**Annotare GCP a mano.** La ground truth viene dalla composizione analitica di
due affini note (I2). Se in un lavoro futuro riemerge la tentazione di annotare
punti in QGIS, è il segnale che si è perso il world file di una delle due
immagini — e la risposta è ritrovarlo, non annotare.

---

## 6. Cosa manca a questo progetto per diventare una tesi

In ordine di costo crescente:

1. **Selezione automatica del modello geometrico** (§1.4) — i dati ci sono già.
2. **Sweep detezione/ratio a corrispondenze costanti** (§1.2) — un esperimento.
3. **Restituire a LoFTR l'invarianza alla rotazione dall'esterno** (§1.5) — il
   limite è già misurato, resta da provare se pre-ruotare basta a toglierlo.
4. **Metrica di confidenza senza riferimento, validata contro l'RMSE noto**
   (§2.1) — è il contributo metodologico più difendibile, e questi dati sono il
   banco di prova giusto.
5. **Decine di fogli** (§3.2) — soprattutto raccolta dati, ma senza non si dice
   nulla di generale.
6. **Abbinamento topologico sul grafo delle particelle** (§1.3) — il filone più
   promettente, e da solo una tesi intera.
