"""Riproduce l'intero progetto da `data/raw/` alla relazione, in un comando solo.

    python -m scripts.riproduci --controlla     # solo le precondizioni, non esegue
    python -m scripts.riproduci                 # tutto tranne E3 (LoFTR)
    python -m scripts.riproduci --con-loftr     # tutto, E3 compreso (~40 min in più)

Perché esiste. Il README elenca quindici comandi in ordine, e l'ordine conta:
i crop prima della rasterizzazione, la rasterizzazione prima di E2, E2 prima
delle tabelle, le tabelle prima della relazione. Alla consegna quella sequenza
va rieseguita su una macchina che ha i dati AdE — e `data/raw/` è fuori dal
versionamento (§5.8), quindi il repo da solo non basta mai. Un errore a metà
sequenza produce silenziosamente numeri parziali: vedi `--controlla`.

Il fallimento silenzioso che questo script previene
---------------------------------------------------
`m6_e1_completo` e `m9_e3_loftr` leggono il world file del foglio così:

    W_hist = read_jgw(args.jgw) if os.path.exists(args.jgw) else None

Se `data/raw/L675_004900.jgw` manca, `W_hist` resta None, `rmse_m` resta None
per ogni riga e `success` diventa False per tutte e quattrocento. L'esperimento
gira fino in fondo, scrive un CSV completo e conclude "0 riuscite". Sembra un
algoritmo che fallisce; è un file mancante. È esattamente la classe di falso
positivo di §5.5, e l'unica difesa è verificare le precondizioni **prima**.

Ogni comando viene stampato prima di essere eseguito: il log di una corsa è la
documentazione eseguibile che §12.10 chiede alla relazione.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Precondizioni: i file AdE. Non sono nel repo (§5.8), vanno rimessi a mano
# seguendo data/README.md. Senza questi non parte nulla di sensato.
GREZZI = {
    "data/raw/L675_004900.jpg": "scansione dell'Originale di Impianto (8489x5648)",
    "data/raw/L675_004900.jgw": "world file: è la ground truth (§5.3)",
    "data/raw/L675_004900.cxf": "vettoriale vigente, sistema 004900 e NON 00490Z (I1)",
}
GREZZI_OPZIONALI = {
    "data/raw/L675_004900_metadata.txt": "metadati: pavimento dell'RMSE (§5.3)",
}
PESI_LOFTR = "weights/loftr_outdoor.ckpt"

DIPENDENZE = ("cv2", "numpy", "pandas", "matplotlib", "scipy", "PIL")
DIPENDENZE_LOFTR = ("torch", "kornia")

CSV = "results/runs.csv"
FIGURE = "results/figures"


@dataclass(frozen=True)
class Fase:
    """Un passo della riproduzione: dei comandi, e gli artefatti che devono uscirne."""

    nome: str
    titolo: str
    comandi: tuple[tuple[str, ...], ...]
    prodotti: tuple[str, ...] = ()
    minuti: float = 0.1
    opzionale: bool = False
    nota: str = ""


def _crop_nomi() -> list[str]:
    """Nomi dei 5 crop di §5.6, letti dalla loro unica definizione.

    L'import sta qui e non in testa al file perché tira dentro cv2: senza, il
    controllo delle precondizioni non potrebbe nemmeno dire *quale* dipendenza
    manca, che è la sola cosa utile da dire in quel momento.
    """
    from src.prep.crop import CROPS

    return [c.nome for c in CROPS]


# Artefatti che `src.report` produce solo se il CSV contiene righe LoFTR: la
# figura E3 è dietro `df.matcher.nunique() > 2`. Senza la fase e3 la relazione
# resta senza il capitolo 10 illustrato, e va detto invece che scoperto dopo.
PRODOTTI_E3 = (f"{FIGURE}/m9_e3_confronto.png", f"{FIGURE}/m9_tabella.md")


def costruisci_fasi(con_loftr: bool = False) -> list[Fase]:
    nomi = _crop_nomi()

    rasterizza = tuple(
        ("src.prep.rasterize", "--crop", n, "--codici", codici)
        for n in nomi
        for codici in ("18", "18,12")
    )
    raster_prodotti = tuple(
        f"data/crops/{n}_{suff}{est}"
        for n in nomi
        for suff in ("vec", "vec1812")
        for est in (".png", ".jgw")
    )
    crop_prodotti = tuple(f"data/crops/{n}{est}" for n in nomi for est in (".png", ".jgw"))

    return [
        Fase(
            "crop",
            "M1 — ritagli dal foglio, con il world file affiancato",
            (("src.prep.crop",),),
            crop_prodotti + (f"{FIGURE}/m1_crops.png",),
            minuti=0.5,
        ),
        Fase(
            "cxf",
            "M2 — parser CXF: 871 BORDO, nflag, coordinate dentro l'estensione",
            (("src.io_geo",), ("experiments.m2_cxf_check",)),
            (f"{FIGURE}/m2_cxf_overview.png", f"{FIGURE}/m2_cornice.png"),
            minuti=0.5,
        ),
        Fase(
            "rasterize",
            "M7 — vettoriale → raster allineato, codice 18 e 18+12",
            rasterizza,
            raster_prodotti,
            minuti=1,
        ),
        Fase(
            "verifica-raster",
            "M7 — verifica a piena risoluzione (I5), non metriche indirette (§5.5)",
            (("experiments.m7_rasterize_check", "--crop", "ribba"),),
            (f"{FIGURE}/m7_ribba_vec.png",),
            minuti=0.5,
            nota="da guardare davvero: I5 vieta di concludere senza averla vista",
        ),
        Fase(
            "preprocess",
            "M5 — Otsu vs Sauvola vs CLAHE, morfologia, gradiente di illuminazione",
            (("experiments.m5_preprocess", "--crop", "tutti", "--dettaglio", "ribba"),),
            tuple(f"{FIGURE}/m5_preprocess_{n}.png" for n in nomi)
            + (f"{FIGURE}/m5_chiusura_ribba.png", f"{FIGURE}/m5_gradiente_ribba.png"),
            minuti=1,
        ),
        Fase(
            "e1",
            "M6 — E1 sintetico completo. Riparte il CSV da zero",
            (("experiments.m6_e1_completo", "--riparti"),),
            (CSV,),
            minuti=4,
            nota="`--riparti` cancella results/runs.csv: le fasi E2/E3 vanno dopo, non prima",
        ),
        Fase(
            "e2",
            "M8 — E2 cross-domain, griglia completa, in append al CSV",
            (("experiments.m8_e2_griglia",),),
            (CSV,),
            minuti=6,
        ),
        Fase(
            "e2-ratio",
            "M8 — diagnosi del ratio test di Lowe (esperimento E2-ratio)",
            (("experiments.m8_e2_griglia", "--diagnosi-ratio"),),
            (CSV,),
            minuti=1,
        ),
        Fase(
            "e3",
            "M9 — E3: LoFTR su E1 ed E2, stessa pipeline",
            (("experiments.m9_e3_loftr",),),
            (CSV,),
            minuti=40,
            opzionale=True,
            nota="richiede torch, kornia e i pesi in weights/ (python -m scripts.scarica_pesi)",
        ),
        Fase(
            "figura-e2",
            "La registrazione migliore di E2 (ORB + Sauvola + similarità) a piena risoluzione",
            (
                (
                    "src.main",
                    "--hist", "data/crops/ribba.png",
                    "--modern", "data/crops/ribba_vec.png",
                    "--matcher", "orb",
                    "--preprocess", "sauvola",
                    "--model", "similarity",
                    "--jgw-hist", "data/crops/ribba.jgw",
                    "--jgw-modern", "data/crops/ribba_vec.jgw",
                    "--soglia-m", "2.0",
                    "--esperimento", "E2-figura",
                    "--out-figure", f"{FIGURE}/m8_verifica_ribba.png",
                ),
            ),
            (f"{FIGURE}/m8_verifica_ribba.png",),
            minuti=0.5,
            nota="l'etichetta E2-figura tiene la riga fuori dalle aggregazioni, "
            "che filtrano per esperimento esatto",
        ),
        Fase(
            "figure",
            "M10 — figure e tabelle aggregate dal CSV (§7.4)",
            (("src.report", "--csv", CSV),),
            (
                f"{FIGURE}/m6_rmse_vs_degradazione.png",
                f"{FIGURE}/m6_rmse_vs_ampiezza.png",
                f"{FIGURE}/m6_preprocess.png",
                f"{FIGURE}/m6_tabella.md",
                f"{FIGURE}/m8_e1_vs_e2.png",
                f"{FIGURE}/m8_e2_dettaglio.png",
                f"{FIGURE}/m8_tabella.md",
            )
            + (PRODOTTI_E3 if con_loftr else ()),
            minuti=0.5,
        ),
        Fase(
            "didattiche",
            "M10 — le figure spiegate del capitolo 2 e delle tecniche",
            (
                ("experiments.m10_passo_passo",),
                ("experiments.m10_ratio_test",),
                ("experiments.m10_gradiente",),
                ("experiments.m10_ransac_iterazioni", "--csv", CSV),
            ),
            (
                f"{FIGURE}/m10_passo_passo.png",
                f"{FIGURE}/m10_ratio_test.png",
                f"{FIGURE}/m10_gradiente.png",
                f"{FIGURE}/m10_ransac_iterazioni.png",
            ),
            minuti=2,
        ),
        Fase(
            "relazione",
            "M10 — relazione con le tabelle iniettate dal CSV, più l'HTML da stampare",
            (("scripts.componi_relazione",), ("scripts.relazione_html",)),
            ("relazione/RELAZIONE.md", "relazione/RELAZIONE.html"),
            minuti=0.5,
        ),
    ]


# --------------------------------------------------------------------- controlli


def _manca(path: str) -> bool:
    intero = os.path.join(RADICE, path)
    return not os.path.exists(intero) or os.path.getsize(intero) == 0


def verifica_ambiente(con_loftr: bool) -> tuple[list[str], list[str]]:
    """Precondizioni, separate in (dipendenze, dati).

    La distinzione non è cosmetica: senza le dipendenze non si può fare nulla,
    nemmeno elencare le fasi (i nomi dei crop vivono nel codice); senza i dati
    si può ancora leggere che cosa lo script farebbe.
    """
    import importlib.util

    dipendenze, dati = [], []
    for modulo in DIPENDENZE:
        if importlib.util.find_spec(modulo) is None:
            dipendenze.append(f"{modulo} — pip install -r requirements.txt")
    for path, perche in GREZZI.items():
        if _manca(path):
            dati.append(f"{path} — {perche}. Vedi data/README.md")
    for path, perche in GREZZI_OPZIONALI.items():
        if _manca(path):
            print(f"  ⚠ {path} assente ({perche}): non blocca, ma il pavimento va citato a mano")

    if con_loftr:
        for modulo in DIPENDENZE_LOFTR:
            if importlib.util.find_spec(modulo) is None:
                dipendenze.append(f"{modulo} (fase e3)")
        if _manca(PESI_LOFTR):
            dati.append(
                f"{PESI_LOFTR} — python -m scripts.scarica_pesi "
                "(§3 vieta il download a runtime)"
            )
    return dipendenze, dati


RIFERIMENTO_FIGURA = re.compile(r"!\[[^\]]*\]\(\.\./(results/figures/[^)]+)\)")


def figure_richieste_dalla_relazione() -> list[str]:
    """Ogni figura che il testo della relazione incorpora.

    Il testo è la fonte: se cita una figura che nessuna fase produce, la
    riproduzione è incompleta anche se ogni comando è uscito con codice 0. È un
    controllo che serve — `m8_verifica_ribba.png` era citata e non la generava
    nessuno.
    """
    sorgente = os.path.join(RADICE, "relazione", "relazione_sorgente.md")
    if not os.path.exists(sorgente):
        return []
    with open(sorgente, encoding="utf-8") as fh:
        return sorted(set(RIFERIMENTO_FIGURA.findall(fh.read())))


def verifica_relazione(con_loftr: bool) -> tuple[list[str], list[str]]:
    """Controlli finali sul documento composto, non solo sui codici di uscita.

    Ritorna (problemi, avvisi). Saltare E3 non è un errore — è una scelta, e
    costa esattamente la figura del capitolo 10: dirlo così è più utile che
    segnalare un file mancante come se fosse un guasto.
    """
    problemi, avvisi = [], []
    for f in figure_richieste_dalla_relazione():
        if not _manca(f):
            continue
        if f in PRODOTTI_E3 and not con_loftr:
            avvisi.append(f"{f}: E3 non è stato eseguito (rilancia con --con-loftr)")
        else:
            problemi.append(f"figura citata dalla relazione ma mai generata: {f}")

    composta = os.path.join(RADICE, "relazione", "RELAZIONE.md")
    if os.path.exists(composta):
        with open(composta, encoding="utf-8") as fh:
            testo = fh.read()
        if "non disponibile: il CSV non contiene le righe necessarie" in testo:
            problemi.append(
                "la relazione contiene tabelle vuote: il CSV non ha tutte le righe "
                "(manca una fase sperimentale, oppure e1 con --riparti è stata rilanciata dopo e2)"
            )
        if "<!-- TABELLA:" in testo:
            problemi.append("segnaposto di tabella non sostituito in RELAZIONE.md")
    return problemi, avvisi


# ----------------------------------------------------------------- esecuzione


def esegui(fase: Fase, prova_secca: bool) -> bool:
    print(f"\n\033[1m── {fase.nome} · {fase.titolo}\033[0m")
    if fase.nota:
        print(f"   nota: {fase.nota}")
    for argv in fase.comandi:
        riga = " ".join(("python", "-m", *argv))
        print(f"   $ {riga}")
        if prova_secca:
            continue
        esito = subprocess.run([sys.executable, "-m", *argv], cwd=RADICE)
        if esito.returncode != 0:
            print(f"\n\033[31m✗ fallita: {riga} (codice {esito.returncode})\033[0m")
            return False

    mancanti = [p for p in fase.prodotti if _manca(p)]
    if mancanti and not prova_secca:
        print(f"\n\033[31m✗ {fase.nome}: comandi riusciti ma mancano gli artefatti:\033[0m")
        for p in mancanti[:10]:
            print(f"    {p}")
        if len(mancanti) > 10:
            print(f"    … e altri {len(mancanti) - 10}")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.riproduci",
        description="Riproduce il progetto da data/raw/ alla relazione",
    )
    ap.add_argument("--lista", action="store_true", help="elenca le fasi ed esce")
    ap.add_argument("--controlla", action="store_true", help="solo precondizioni e artefatti attesi")
    ap.add_argument("--solo", default=None, help="una sola fase, per nome")
    ap.add_argument("--da", default=None, help="riparte da questa fase")
    ap.add_argument("--fino-a", dest="fino_a", default=None, help="si ferma dopo questa fase")
    ap.add_argument("--con-loftr", action="store_true", help="include E3 (~40 min, serve torch)")
    args = ap.parse_args(argv)

    print("\033[1mhistreg — riproduzione completa\033[0m")
    print(f"radice: {RADICE}\n")

    print("precondizioni:")
    dipendenze, dati = verifica_ambiente(args.con_loftr)

    if dipendenze:
        print("\n\033[31mdipendenze mancanti:\033[0m")
        for p in dipendenze:
            print(f"  ✗ {p}")
        return 1

    if dati:
        # `--lista` è documentazione: risponde anche su una macchina senza i dati,
        # che è il caso di chi apre il repository per capire cosa fa.
        print(f"\n\033[{'33mdati mancanti (--lista non ne ha bisogno)' if args.lista else '31mnon si può procedere'}:\033[0m")
        for p in dati:
            print(f"  ✗ {p}")
        if not args.lista:
            print(
                "\nI dati AdE non stanno nel repository (§5.8): vanno riscaricati dal servizio "
                "\ndell'Agenzia delle Entrate seguendo data/README.md. Ricorda I1: il file "
                "\nutilizzabile è L675_004900, mai L675_00490Z."
            )
            return 1
    else:
        print("  ✓ dipendenze e dati grezzi a posto")

    fasi = costruisci_fasi(args.con_loftr)
    if args.solo:
        fasi = [f for f in fasi if f.nome == args.solo]
        if not fasi:
            print(f"fase sconosciuta: {args.solo}", file=sys.stderr)
            return 2
    else:
        if not args.con_loftr:
            fasi = [f for f in fasi if not f.opzionale]
        nomi = [f.nome for f in fasi]
        if args.da:
            if args.da not in nomi:
                print(f"fase sconosciuta: {args.da}", file=sys.stderr)
                return 2
            fasi = fasi[nomi.index(args.da) :]
        if args.fino_a:
            nomi = [f.nome for f in fasi]
            if args.fino_a not in nomi:
                print(f"fase sconosciuta: {args.fino_a}", file=sys.stderr)
                return 2
            fasi = fasi[: nomi.index(args.fino_a) + 1]

    if args.lista:
        print(f"\n{len(fasi)} fasi · circa {sum(f.minuti for f in fasi):.0f} minuti\n")
        for f in fasi:
            segno = "○" if f.opzionale else "●"
            print(f"  {segno} {f.nome:<16} {f.titolo}  (~{f.minuti:g} min)")
        return 0

    print(f"\n{len(fasi)} fasi · stima {sum(f.minuti for f in fasi):.0f} minuti")
    if args.controlla:
        print("(prova secca: stampo i comandi e verifico solo le precondizioni)")

    t0 = time.perf_counter()
    for fase in fasi:
        if not esegui(fase, args.controlla):
            print(f"\nfermato a `{fase.nome}`. Corretto il problema, riprendi con:")
            print(f"  python -m scripts.riproduci --da {fase.nome}")
            return 1

    if not args.controlla:
        print("\n\033[1m── verifica finale\033[0m")
        problemi, avvisi = verifica_relazione(args.con_loftr)
        for a in avvisi:
            print(f"  ⚠ {a}")
        if problemi:
            print("\033[31mla riproduzione è incompleta:\033[0m")
            for p in problemi:
                print(f"  ✗ {p}")
            return 1
        citate = figure_richieste_dalla_relazione()
        print(f"  ✓ {len(citate) - len(avvisi)}/{len(citate)} figure citate dalla relazione sono presenti")
        print("  ✓ nessun segnaposto di tabella rimasto")

    print(f"\n\033[32mfatto in {(time.perf_counter() - t0) / 60:.1f} minuti\033[0m")
    if not args.controlla:
        print("\nDa qui: relazione/RELAZIONE.html si apre nel browser e si stampa in PDF.")
        if not args.con_loftr:
            print("E3 (LoFTR) non è stato eseguito: rilancia con --con-loftr per la parte comparativa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
