"""Verifica I9: stesso input e stessi parametri devono dare lo stesso CSV.

    python -m scripts.verifica_determinismo

CLAUDE.md I9 dichiara il determinismo come invariante, ma finora nessuno lo
verificava end-to-end: i test controllano il seme di RANSAC su dati sintetici,
non l'esperimento intero sui dati veri. Questo script lo fa: esegue la stessa
fetta di E1 più volte, in **processi separati**, e confronta i CSV.

Le colonne dei tempi sono escluse dal confronto perché cambiano sempre e non
dicono nulla sul determinismo: `t_init_ms`, `t_match_ms`, `t_stima_ms`.

Perché processi separati e non chiamate ripetute nello stesso processo: dentro
un processo solo lo stato globale — il generatore di numeri casuali di OpenCV,
la cache dei matcher, quel che resta in memoria — è condiviso fra le due
esecuzioni, quindi due chiamate uguali possono concordare per la ragione
sbagliata. Processi separati riproducono la condizione vera, che è rilanciare
il comando domani.

L'esito interessante è quello negativo: se due corse identiche divergono, I9 è
violato e va scritto in relazione invece che scoperto da chi prova a
riprodurre.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cambiano a ogni esecuzione per costruzione: confrontarle direbbe sempre
# "diverso" e nasconderebbe l'unica differenza che conta.
COLONNE_TEMPO = ("t_init_ms", "t_match_ms", "t_stima_ms")


def corri(crop: str, destinazione: str, seed: int) -> bool:
    argv = [
        sys.executable, "-m", "experiments.m6_e1_completo",
        "--crop", crop, "--seed", str(seed), "--out-csv", destinazione, "--riparti",
    ]
    print(f"  $ {' '.join(argv[1:])}")
    return subprocess.run(argv, cwd=RADICE, stdout=subprocess.DEVNULL).returncode == 0


def confronta(percorsi: list[str]) -> int:
    """Confronta i CSV a due a due. Ritorna il numero di coppie divergenti."""
    import pandas as pd

    tabelle = []
    for p in percorsi:
        df = pd.read_csv(p)
        tabelle.append(df.drop(columns=[c for c in COLONNE_TEMPO if c in df]))

    divergenti = 0
    for i in range(1, len(tabelle)):
        a, b = tabelle[0], tabelle[i]
        if a.shape != b.shape:
            print(f"\n\033[31m✗ corsa 1 e {i + 1}: forme diverse, {a.shape} contro {b.shape}\033[0m")
            divergenti += 1
            continue
        # NaN != NaN in pandas: senza la seconda maschera ogni cella vuota
        # risulterebbe "diversa" da sé stessa e il test non direbbe mai nulla.
        diverse = (a != b) & ~(a.isna() & b.isna())
        per_colonna = diverse.sum()
        per_colonna = per_colonna[per_colonna > 0]
        if per_colonna.empty:
            print(f"\n\033[32m✓ corsa 1 e {i + 1}: identiche su {len(a)} righe\033[0m")
            continue

        divergenti += 1
        print(f"\n\033[31m✗ corsa 1 e {i + 1}: {int(per_colonna.sum())} celle diverse\033[0m")
        print(per_colonna.to_string())
        # La prima riga divergente, per capire *dove* invece che *quanto*
        prima = diverse.any(axis=1).idxmax()
        colonne = [c for c in a.columns if diverse.loc[prima, c]]
        chiavi = [c for c in ("esperimento", "crop", "matcher", "preprocess", "degrado", "rot_deg")
                  if c in a]
        print("\n  prima riga che diverge:")
        for c in chiavi:
            print(f"    {c:<14} {a.loc[prima, c]}")
        for c in colonne:
            print(f"    {c:<14} {a.loc[prima, c]}  contro  {b.loc[prima, c]}")
    return divergenti


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="ribba", help="un solo ritaglio: basta e avanza")
    ap.add_argument("--corse", type=int, default=3, help="quante esecuzioni confrontare")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tieni", action="store_true", help="non cancellare i CSV prodotti")
    args = ap.parse_args(argv)

    crop_png = os.path.join(RADICE, "data", "crops", f"{args.crop}.png")
    if not os.path.exists(crop_png):
        print(f"manca {crop_png} — genera prima i ritagli: python -m src.prep.crop", file=sys.stderr)
        return 1

    print(f"I9 — determinismo di E1 su `{args.crop}`, {args.corse} corse in processi separati")
    print("(colonne dei tempi escluse dal confronto: cambiano sempre)\n")

    percorsi = []
    for i in range(args.corse):
        destinazione = os.path.join("results", f"_determinismo_{i + 1}.csv")
        print(f"corsa {i + 1}/{args.corse}")
        if not corri(args.crop, destinazione, args.seed):
            print(f"\nla corsa {i + 1} è fallita", file=sys.stderr)
            return 1
        percorsi.append(os.path.join(RADICE, destinazione))

    divergenti = confronta(percorsi)

    if not args.tieni:
        for p in percorsi:
            os.remove(p)

    print()
    if divergenti:
        print("\033[31mI9 NON è rispettato su questa macchina.\033[0m")
        print("Due esecuzioni identiche danno risultati diversi: va dichiarato in")
        print("relazione, perché rende i numeri non riproducibili cifra per cifra.")
        return 1

    print("\033[32mI9 è rispettato: le corse coincidono, cifra per cifra.\033[0m")
    print("Una differenza fra due CSV prodotti in momenti diversi viene quindi")
    print("da qualcos'altro — codice, dipendenze o dati cambiati fra le due volte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
