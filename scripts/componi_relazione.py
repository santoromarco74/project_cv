"""Compone la relazione iniettando le tabelle generate dal CSV.

CLAUDE.md §7.4: "Le tabelle della relazione sono aggregazioni del CSV, non
numeri ricopiati a mano". Questo script rende quella frase verificabile.

`relazione/relazione_sorgente.md` è il testo, con segnaposto della forma

    <!-- TABELLA: e3 -->

Ogni segnaposto viene sostituito con la tabella corrispondente, calcolata al
momento da `results/runs.csv`. Il risultato è `relazione/RELAZIONE.md`, che è il
documento da leggere e non va modificato a mano: se un numero è sbagliato, è
sbagliato l'esperimento o l'aggregazione, non il testo.

    python -m scripts.componi_relazione
"""
from __future__ import annotations

import argparse
import os
import re

from src import report

# nome del segnaposto -> funzione che produce la tabella
TABELLE = {
    "crop": lambda df: report.tabella_crop(),
    "e1": report.tabella,
    "e2": report.tabella_e2,
    "e2_fattori": report.tabella_e2_fattori,
    "e2_per_crop": report.tabella_e2_per_crop,
    "e3": report.tabella_e3,
    "diagnosi_ratio": report.tabella_diagnosi_ratio,
}

SEGNAPOSTO = re.compile(r"<!--\s*TABELLA:\s*([a-z0-9_]+)\s*-->")


def componi(sorgente: str, csv: str) -> str:
    df = report.carica(csv)
    testo = open(sorgente, encoding="utf-8").read()

    mancanti = []

    def sostituisci(m: re.Match) -> str:
        nome = m.group(1)
        if nome not in TABELLE:
            mancanti.append(nome)
            return m.group(0)
        tabella = TABELLE[nome](df)
        if tabella is None or len(tabella) == 0:
            return f"_(tabella `{nome}` non disponibile: il CSV non contiene le righe necessarie)_"
        return report._markdown(tabella)

    uscita = SEGNAPOSTO.sub(sostituisci, testo)
    if mancanti:
        raise KeyError(f"segnaposto sconosciuti nella relazione: {sorted(set(mancanti))}")
    return uscita


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # nome diverso da RELAZIONE.md non solo per maiuscole: su filesystem
    # case-insensitive (NTFS) i due path collidono sullo stesso file fisico
    ap.add_argument("--sorgente", default=os.path.join("relazione", "relazione_sorgente.md"))
    ap.add_argument("--csv", default=os.path.join("results", "runs.csv"))
    ap.add_argument("--out", default=os.path.join("relazione", "RELAZIONE.md"))
    args = ap.parse_args(argv)

    uscita = componi(args.sorgente, args.csv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(uscita)

    quante = len(SEGNAPOSTO.findall(open(args.sorgente, encoding="utf-8").read()))
    print(f"{args.out}: {quante} tabelle iniettate da {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
