"""Scarica una volta sola i pesi LoFTR in `weights/`, verificandone il checksum.

CLAUDE.md §3 chiede che i pesi siano presenti nella versione finale del progetto
e **non scaricati a runtime**: un peso che arriva dalla rete al momento
dell'esecuzione può cambiare, e con lui i risultati. Sono ~45 MB, troppo per il
repository, quindi si versiona questo script più il checksum — che è la
seconda opzione prevista da §3.

    python -m scripts.scarica_pesi
    python -m scripts.scarica_pesi --verifica     # controlla e basta

L'URL è lo stesso da cui li prenderebbe kornia (kornia/feature/loftr/loftr.py).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request

URL = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
DESTINAZIONE = os.path.join("weights", "loftr_outdoor.ckpt")

# SHA256 del file usato per gli esperimenti di M9. Se cambia, i risultati non
# sono più confrontabili con quelli in results/runs.csv e va detto.
SHA256 = "21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blocco in iter(lambda: fh.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=DESTINAZIONE)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--verifica", action="store_true", help="non scaricare, solo verificare")
    ap.add_argument("--registra", action="store_true", help="stampa il checksum del file presente")
    args = ap.parse_args(argv)

    if args.registra:
        if not os.path.exists(args.dest):
            print(f"{args.dest}: assente")
            return 1
        print(f"{args.dest}\nsha256 = {sha256(args.dest)}")
        return 0

    if not args.verifica and not os.path.exists(args.dest):
        os.makedirs(os.path.dirname(args.dest) or ".", exist_ok=True)
        print(f"scarico {args.url} -> {args.dest}")
        urllib.request.urlretrieve(args.url, args.dest)  # noqa: S310

    if not os.path.exists(args.dest):
        print(f"{args.dest}: assente")
        return 1

    somma = sha256(args.dest)
    dimensione = os.path.getsize(args.dest) / 1e6
    print(f"{args.dest}  {dimensione:.1f} MB\nsha256 = {somma}")
    if somma != SHA256:
        print(
            "ATTENZIONE: il checksum non coincide con quello usato per gli esperimenti.\n"
            "I risultati in results/runs.csv sono stati prodotti con un file diverso.\n"
            f"atteso  {SHA256}"
        )
        return 2
    print("checksum corretto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
