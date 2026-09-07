# Pesi dei modelli

**Non versionati** (`.gitignore`): 46 MB sono troppi per il repository, quindi si 
prevede questa alternativa esplicita — script di download più checksum — e la
ragione per cui non basta lasciar fare a kornia è che un peso scaricato a runtime
può cambiare, e con lui i risultati in `results/runs.csv`.

## LoFTR

```bash
python -m scripts.scarica_pesi              # scarica e verifica
python -m scripts.scarica_pesi --verifica   # verifica soltanto
python -m scripts.scarica_pesi --registra   # stampa il checksum del file locale
```

| campo | valore |
|---|---|
| file | `weights/loftr_outdoor.ckpt` |
| dimensione | 46.3 MB |
| sha256 | `21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec` |
| origine | `http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt` |
| variante | `outdoor` (§7.2) |

L'URL è lo stesso da cui li prenderebbe kornia
(`kornia/feature/loftr/loftr.py`). `LoftrMatcher` costruisce il modello con
`pretrained=None` e carica questo file: se manca, si ferma e indica lo script
invece di scaricare di nascosto.

Se il checksum non coincide, i risultati prodotti non sono confrontabili con
quelli in `results/runs.csv`, e va detto invece di ignorarlo.

## Dipendenze

`torch` e `kornia` servono **solo** a `--matcher loftr`. La pipeline funziona
senza (I4). Su CPU:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install kornia==0.7.3
```

La riga `torch==2.3.1` in `requirements.txt` installa la build CUDA, che porta
con sé alcuni GB di pacchetti NVIDIA inutili su una macchina senza GPU: per
WSL2/CPU conviene l'indice qui sopra.
