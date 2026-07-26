# Dati

I file cartografici **non sono versionati** (repo pubblico, condizioni d'uso AdE
da verificare prima di ridistribuire). Questa cartella documenta come ricostruirli.

## Origine

Servizio "Consultazione dei fogli di mappa catastale", Agenzia delle Entrate.
Accesso all'area riservata con SPID / CIE / CNS. Gratuito.

## Foglio usato

| campo | valore |
|---|---|
| Comune | Varazze (SV) |
| Foglio | 49 |
| Nome mappa | L675_004900 |
| Sistema | Cassini-Soldner zona G0007 (Forte Diamante) |
| Scala | 1:2000 |
| Risoluzione | 0.254453 m/px |

## File da scaricare in `raw/`

Dalla funzione **Fogli Originali di Impianto**, sistema **ORIGINARIO**:

- `L675_004900.jpg`  — raster dell'Originale di Impianto, **8489 × 5648 px**
- `L675_004900.jgw`  — world file (georeferenziazione)
- `L675_004900.txt`  — metadati (copiarlo anche in `L675_004900_metadata.txt`:
  quel nome è l'unico escluso dal `.gitignore`, ed è la copia versionata)
- `L675_004900.cxf`  — vettoriale della mappa vigente, stesso sistema

**ATTENZIONE**: NON usare `L675_00490Z.cxf`. È un allegato in Roma40-Gauss
Boaga (EPSG:3003), sistema diverso: non si allinea al raster. Vedi CLAUDE.md §5.1.

Il codice legge sempre la dimensione del raster dal file, mai da una costante:
CLAUDE.md §5.2 e I10 riportano 8000×5322 px, che non corrisponde alla scansione
(i px in più sono margine di carta). Vedi la nota in `crops/README.md`.

## Ricampionamento (dai metadati)

76 coppie omologhe · scarto max 1.28 m · scarto medio 0.56 m · dev. std 0.25 m.
Questo definisce il **pavimento** dell'RMSE: sotto ~0.5 m si misura il rumore
del riferimento, non l'errore dell'algoritmo. Vedi CLAUDE.md §5.3.

## Crop (generati da `src/prep/crop.py`, non versionati)

Coordinate in CLAUDE.md §5.6.
