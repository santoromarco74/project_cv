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
| `cannei` | 2900 | 200 | 1200 | 1000 | −30742.13 … −30437.04 | −11583.85 … −11329.65 | 305×254 m |
| `ribba` | 3300 | 600 | 1024 | 1024 | −30640.35 … −30380.04 | −11691.74 … −11431.43 | 260×260 m |
| `vedra` | 4200 | 2000 | 1024 | 1024 | −30411.34 … −30151.04 | −12047.97 … −11787.66 | 260×260 m |
| `aspera` | 5600 | 2600 | 1024 | 1024 | −30055.11 … −29794.80 | −12200.64 … −11940.34 | 260×260 m |

Le estensioni sono espresse nella convenzione del world file: gli estremi sono i
**centri** dei pixel d'angolo, non gli spigoli (mezzo pixel = 0.127 m di
differenza rispetto a come le riporta QGIS).

## Foglio di partenza

`L675_004900.jpg` misura **8489 × 5648 px**, letto dal file — mai hardcodato.
Con il JGW (0.254453 m/px) l'estensione è X −31480.04 … −29320.25,
Y −12715.65 … −11278.76.

> **Nota.** CLAUDE.md §5.2 e I10 riportano 8000×5322 px e un'estensione
> corrispondente. Non coincide con il file: i 489 px in più in larghezza e 326 in
> altezza sono i margini di carta della scansione. I ritagli non ne risentono —
> stanno tutti entro col 6624 / riga 3624 — e le loro estensioni tornano al metro.
> La verifica di inquadramento è in `results/figures/m2_cornice.png` (M2).

## Osservazioni sul contenuto (per E2)

- `vedra` è il ritaglio più povero di tratto: pochi confini, un corso d'acqua,
  molta carta bianca. Utile come caso difficile, ma va tenuto presente quando se
  ne leggono gli inlier ratio.
- `aspera` tocca la costa: il bordo destro contiene mare, che nel vettoriale non
  ha corrispondenza.
