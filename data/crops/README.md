# Ritagli del foglio 49

I PNG **non sono versionati** (derivano dal raster AdE, vedi `../README.md` e
CLAUDE.md §5.8). Questo file documenta come rigenerarli identici.

## Comando

```bash
python -m src.prep.crop --sheet data/raw/L675_004900.jpg \
                        --jgw   data/raw/L675_004900.jgw \
                        --out   data/crops
```

Opzioni: `--only <nome>` per un solo ritaglio, `--figure ''` per saltare la
figura di riepilogo (`results/figures/m1_crops.png`).

## Cosa produce

Per ogni ritaglio due file: `<nome>.png` (PNG lossless, colore originale
conservato — la conversione a grigi è compito di `preprocess.py`) e `<nome>.jgw`,
il world file del ritaglio.

Il world file è composto per traslazione dell'origine pixel:

```
C' = C + A·x0 + B·y0        F' = F + D·x0 + E·y0        (qui B = D = 0)
```

A, D, B, E restano quelli del foglio: il ritaglio non cambia scala né rotazione.

Il `.jgw` accanto al PNG è **metadato, non input dell'algoritmo**: lo legge solo
`evaluate.py` attraverso `groundtruth.py` (I3). `pipeline.py` vede il PNG e basta.
L'estensione `.jgw` è tenuta anche per i PNG (la convenzione sarebbe `.pgw`) per
uniformità con i flag `--jgw-hist` / `--jgw-modern` del contratto CLI (§9).

## Ritagli

Offset e dimensioni da CLAUDE.md §5.6. Le estensioni sono quelle **ricalcolate dal
JGW** e verificate contro la tabella di §5.6 con tolleranza 1 m (`check_extent`):
se un giorno non tornano, il crop o il world file sono sbagliati e il comando
fallisce invece di scrivere dati mal georeferenziati.

| nome | x0 | y0 | w | h | X (m) | Y (m) | dimensione |
|---|---|---|---|---|---|---|---|
| `tassarole` | 1500 | 300 | 1024 | 1024 | −31098.36 … −30838.06 | −11615.40 … −11355.09 | 260×260 m |
| `cannei` | 3100 | 420 | 1024 | 1024 | −30691.24 … −30430.93 | −11645.93 … −11385.63 | 260×260 m |
| `ribba` | 3850 | 700 | 1024 | 1024 | −30500.40 … −30240.09 | −11717.18 … −11456.88 | 260×260 m |
| `vedra` | 4200 | 1450 | 1024 | 1024 | −30411.34 … −30151.04 | −11908.02 … −11647.72 | 260×260 m |
| `aspera` | 5600 | 2600 | 1024 | 1024 | −30055.11 … −29794.80 | −12200.64 … −11940.34 | 260×260 m |

Le estensioni sono espresse nella convenzione del world file: gli estremi sono i
**centri** dei pixel d'angolo, non gli spigoli (mezzo pixel = 0.127 m di
differenza rispetto a come le riporta QGIS).

> **Correzione (offset di `cannei`, `ribba`, `vedra`).** Verificati contro le
> etichette di testo reali nel CXF (record `TESTO`), convertite in pixel col
> JGW:
>
> | crop | punto reale (X,Y) | pixel | box vecchio | problema |
> |---|---|---|---|---|
> | `cannei` | -30561.997, -11516.857 | 3609, 936 | 2900,200,1200×1000 | dentro ma sbilanciato al 59%/74%, margine superfluo verso Tassarole/Piano |
> | `ribba` | -30392.563, -11604.184 | 4274, 1279 | 3300,600,1024×1024 | a 50 px dal bordo destro — inquadrava perlopiù Cannei |
> | `vedra` | -30276.945, -11728.872 | 4728, 1769 | 4200,2000,1024×1024 | **fuori dal box**, 231 px sopra il bordo superiore |
>
> Cannei e Ribba distano ~190 m nella realtà, Ribba e Vedra ~170 m — meno della
> larghezza di un box (260 m): una sovrapposizione residua fra crop adiacenti
> (`cannei`/`ribba` ~19%, `ribba`/`vedra` ~18% dell'area) è quindi geografia, non
> un difetto della correzione. `cannei` è stato anche uniformato a 1024×1024
> come gli altri quattro (prima era 1200×1000).
>
> Conseguenza: `cannei.png`/`ribba.png`/`vedra.png` e i rispettivi `.jgw`, i
> raster vettoriali (`*_vec*.png`) e ogni riga di `results/runs.csv` con
> `crop` in questi tre vanno **rigenerati** — checklist §11.4 di CLAUDE.md. I
> numeri già pubblicati in relazione per questi crop (comprese le percentuali
> di tratto qui sotto) si riferiscono ai box vecchi.

## Foglio di partenza

`L675_004900.jpg` misura **8489 × 5648 px**, letto dal file — mai hardcodato.
Con il JGW (0.254453 m/px) l'estensione è X −31480.04 … −29320.25,
Y −12715.65 … −11278.76.

> **Nota.** CLAUDE.md §5.2 e I10 riportano 8000×5322 px e un'estensione
> corrispondente. Non coincide con il file: i 489 px in più in larghezza e 326 in
> altezza sono i margini di carta della scansione. I ritagli non ne risentono —
> stanno tutti entro col 6624 / riga 3624 — e le loro estensioni tornano al metro.
> La verifica di inquadramento è in `results/figures/m2_cornice.png` (M2).

## Raster vettoriali (M7)

```bash
python -m src.prep.rasterize --crop ribba --codici 18      # -> ribba_vec.png/.jgw
python -m src.prep.rasterize --crop ribba --codici 18,12   # -> ribba_vec1812.png/.jgw
```

Sono le immagini "moderne" di E2: i poligoni del CXF disegnati come polilinee
(non campiture: sull'impianto il confine è un tratto), inchiostro scuro su fondo
chiaro come lo storico.

**La griglia non è quella del crop storico**, ed è una scelta. Il raster
vettoriale ha risoluzione propria (0.20 m/px contro 0.254453) e un margine di
20 m oltre il crop. Se le due griglie coincidessero, `H_true` sarebbe l'identità
e l'esperimento premierebbe qualunque metodo con un bias verso il "non muovere
niente"; con griglie diverse `H_true` è una similarità vera (fattore di scala
1.272265). `--stessa-griglia` esiste per il caso degenere, documentato ma non
usato.

Il margine serve a dare una controparte anche al contenuto sul bordo del crop:
senza, si misurerebbe il ritaglio invece del matching.

Percentuale di tratto per crop:

| crop | codice 18 | codici 18+12 |
|---|---|---|
| `tassarole` | 1.18% | 1.53% |
| `cannei` | *da ricalcolare* | *da ricalcolare* |
| `ribba` | *da ricalcolare* | *da ricalcolare* |
| `vedra` | *da ricalcolare* | *da ricalcolare* |
| `aspera` | 1.61% | 2.45% |

`cannei`, `ribba` e `vedra` sono *da ricalcolare* perché il loro offset è stato
corretto (vedi nota sopra): i valori misurati sui box vecchi — 1.44%/1.90%,
2.32%/3.03%, 0.80%/1.12% rispettivamente — si riferiscono a territorio in parte
diverso e non sono più validi.

## Osservazioni sul contenuto (per E2)

- `vedra` è il ritaglio più povero di tratto: pochi confini, un corso d'acqua,
  molta carta bianca. Utile come caso difficile, ma va tenuto presente quando se
  ne leggono gli inlier ratio. La verifica di M7 lo conferma: il rio e i
  fabbricati 94/96/97 combaciano, ma diversi confini vettoriali attraversano
  carta vuota.
- `aspera` tocca la costa: il bordo destro contiene mare, che nel vettoriale non
  ha corrispondenza.
- In generale, parte delle linee vettoriali non ha controparte nell'inchiostro:
  è la divergenza di contenuto di CLAUDE.md §5.7 (il CXF è la cartografia
  vigente, non la digitalizzazione dell'impianto), e si vede a occhio nelle
  figure `results/figures/m7_*.png`. Non è disallineamento, è un secolo di
  aggiornamenti.
