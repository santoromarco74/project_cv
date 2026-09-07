#!/usr/bin/env bash
# Costruisce il pacchetto da inviare al docente, per inclusione esplicita
# (non per esclusione): più facile da controllare a occhio di una lista di
# --exclude, e non rischia di dimenticare un pattern nuovo.
#
# Uso:
#   ./pacchetto_docente.sh /percorso/al/progetto
# (default: cartella corrente)
#
# NON è pensato per essere committato nel repo del progetto — CLAUDE.md §11
# lo mette esplicitamente fuori scope ("Docker, CI, packaging"). Va tenuto a
# parte e lanciato solo quando serve.

set -euo pipefail

SRC="${1:-.}"
OUT="histreg_consegna_$(date +%Y%m%d)"

if [ ! -f "$SRC/README.md" ] || [ ! -d "$SRC/src" ]; then
    echo "ERRORE: $SRC non sembra la root del progetto (manca README.md o src/)" >&2
    exit 1
fi

rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT"

copia() {  # copia() <percorso relativo> — silenzioso se non esiste
    local rel="$1"
    if [ -e "$SRC/$rel" ]; then
        mkdir -p "$OUT/$(dirname "$rel")"
        cp -r "$SRC/$rel" "$OUT/$rel"
    fi
}

# --- codice e documentazione ---
copia src
copia experiments
copia scripts
copia tests
copia requirements.txt
copia README.md
copia LICENSE

# --- relazione (già rigenerata da componi_relazione.py, non a mano) ---
copia relazione/relazione_sorgente.md
copia relazione/RELAZIONE.md
copia relazione/RELAZIONE.html   # se generata da scripts/relazione_html.py

# --- dati: solo i metadati piccoli e versionabili, MAI le scansioni AdE
#     (CLAUDE.md §5.8: condizioni d'uso non verificate per redistribuzione) ---
copia data/README.md
copia data/crops/README.md
find "$SRC/data/raw" -maxdepth 1 -name '*_metadata.txt' -exec cp {} "$OUT/data/raw/" \; 2>/dev/null || true
mkdir -p "$OUT/data/raw"

# --- crop e risultati rigenerati oggi (dopo la correzione di cannei/ribba/vedra) ---
find "$SRC/data/crops" -maxdepth 1 \( -name '*.png' -o -name '*.jgw' \) -exec cp {} "$OUT/data/crops/" \; 2>/dev/null || true
copia results/runs.csv
find "$SRC/results/figures" -maxdepth 1 -name '*.png' -exec bash -c 'mkdir -p "$0/results/figures" && cp "$1" "$0/results/figures/"' "$OUT" {} \; 2>/dev/null || true
find "$SRC/results/figures" -maxdepth 1 -name '*.md' -exec bash -c 'mkdir -p "$0/results/figures" && cp "$1" "$0/results/figures/"' "$OUT" {} \; 2>/dev/null || true

# --- pesi LoFTR: solo istruzioni per riscaricarli, mai il binario (~90 MB) ---
copia weights/README.md

# --- la dichiarazione obbligatoria (Linee Guida IA UniPV) ---
copia DICHIARAZIONE_IA.md

# --- NON copiati, deliberatamente: .git, .claude, CLAUDE.md, IDEE_TESI.md,
#     __pycache__, .venv, .pytest_cache, IDE config, scansioni AdE, pesi binari ---

zip -rq "$OUT.zip" "$OUT"
echo "Pacchetto pronto: $OUT.zip"
echo
echo "Contenuto:"
find "$OUT" -type f | sort
echo
echo "File NON inclusi (verificalo): CLAUDE.md, IDEE_TESI.md, .claude/, .git/, scansioni AdE, pesi .pth/.ckpt"
