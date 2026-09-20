"""Riproduce l'intero progetto da `data/raw/` alla relazione, in un comando solo.

    python -m scripts.riproduci --controlla     # solo le precondizioni, non esegue
    python -m scripts.riproduci                 # tutto tranne E3 (LoFTR)
    python -m scripts.riproduci --con-loftr     # tutto, E3 compreso (~40 min in più)
    python -m scripts.riproduci --da verifica-raster   # senza i dati AdE, dai soli crop

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

Se nessun world file è leggibile, `W_hist` resta None, `rmse_m` resta None per
ogni riga e `success` diventa False per tutte e quattrocento. L'esperimento gira
fino in fondo, scrive un CSV completo e conclude "0 riuscite". Sembra un
algoritmo che fallisce; è un file mancante. È esattamente la classe di falso
positivo di §5.5, e la difesa è duplice: i due esperimenti ripiegano sul world
file di un ritaglio quando manca quello del foglio, e comunque lo dicono invece
di lasciarlo dedurre dal CSV.

Quali dati servono, e a quali fasi
----------------------------------
Solo `crop`, `cxf` e `rasterize` aprono le scansioni AdE. Tutto il resto legge
`data/crops/`: i ritagli, i raster del vettoriale e i loro world file, cioè
artefatti che quelle tre fasi hanno già prodotto. Ogni `Fase` dichiara il
proprio fabbisogno nel campo `grezzi`, e il controllo somma quello delle sole
fasi selezionate.

Serve perché `data/crops/` è ridistribuibile e `data/raw/` no (§5.8): chi
riceve il progetto senza le scansioni può comunque rifare esperimenti, figure e
relazione. Prima il controllo pretendeva tutti e tre i file grezzi a ogni
invocazione, e rifiutava di partire anche a chi aveva chiesto `--da figura-e2`,
che non ne tocca nessuno.

Anche `e1` ed `e3` sono liberi: il world file del foglio serviva loro solo per
la risoluzione, e `io_geo.jgw_per_risoluzione` ripiega su quello di un ritaglio,
che ha gli stessi coefficienti lineari. La conversione in metri esce identica.

Ogni comando viene stampato prima di essere eseguito: il log di una corsa è la
documentazione eseguibile che §12.10 chiede alla relazione.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Precondizioni: i file AdE. Non sono nel repo (§5.8), vanno rimessi a mano
# seguendo data/README.md. Ogni fase dichiara quali le servono davvero (campo
# `grezzi`): la maggioranza non ne tocca nessuno, perché legge i ritagli e i
# raster vettoriali già prodotti in data/crops/.
GREZZO_JPG = "data/raw/L675_004900.jpg"
GREZZO_JGW = "data/raw/L675_004900.jgw"
GREZZO_CXF = "data/raw/L675_004900.cxf"
GREZZO_TXT = "data/raw/L675_004900_metadata.txt"

GREZZI = {
    GREZZO_JPG: "scansione dell'Originale di Impianto (8489x5648)",
    GREZZO_JGW: "world file: è la ground truth (§5.3)",
    GREZZO_CXF: "vettoriale vigente, sistema 004900 e NON 00490Z (I1)",
}
GREZZI_OPZIONALI = {
    GREZZO_TXT: "metadati: pavimento dell'RMSE (§5.3)",
}
PESI_LOFTR = "weights/loftr_outdoor.ckpt"

# `markdown` serve solo all'ultima fase, ma va controllato all'inizio: scoprirlo
# dopo un'ora di esperimenti, con il CSV già scritto e la relazione a un passo
# dalla fine, è il modo peggiore di venirlo a sapere.
DIPENDENZE = ("cv2", "numpy", "pandas", "matplotlib", "scipy", "PIL", "markdown", "latex2mathml")
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
    grezzi: tuple[str, ...] = ()
    grezzi_opzionali: tuple[str, ...] = ()
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
            grezzi=(GREZZO_JPG, GREZZO_JGW),
            minuti=0.5,
        ),
        Fase(
            "cxf",
            "M2 — parser CXF: 871 BORDO, nflag, coordinate dentro l'estensione",
            (("src.io_geo",), ("experiments.m2_cxf_check",)),
            (f"{FIGURE}/m2_cxf_overview.png", f"{FIGURE}/m2_cornice.png"),
            grezzi=(GREZZO_CXF, GREZZO_JGW, GREZZO_JPG),
            minuti=0.5,
        ),
        Fase(
            "rasterize",
            "M7 — vettoriale → raster allineato, codice 18 e 18+12",
            rasterizza,
            raster_prodotti,
            grezzi=(GREZZO_CXF,),
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
            (("experiments.m10_didattiche", "--csv", CSV),),
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
            ("relazione/relazione.md", "relazione/relazione.html"),
            grezzi_opzionali=(GREZZO_TXT,),
            minuti=0.5,
        ),
    ]


# --------------------------------------------------------------------- controlli


def _manca(path: str) -> bool:
    intero = os.path.join(RADICE, path)
    return not os.path.exists(intero) or os.path.getsize(intero) == 0


def verifica_dipendenze(con_loftr: bool) -> tuple[list[str], list[str]]:
    """Le precondizioni che non dipendono dalle fasi scelte: pacchetti e LoFTR.

    Va chiamata prima di costruire le fasi, e non per ordine estetico:
    `costruisci_fasi` importa `src.prep.crop` per leggere i nomi dei ritagli, e
    quell'import tira dentro cv2. Senza questo controllo, a chi non ha cv2
    installato arriverebbe un ImportError invece della riga che dice cosa fare.

    La distinzione fra i due gruppi non è cosmetica, perché la cura è diversa:
    le dipendenze si installano, il materiale di LoFTR si prende con
    `scarica_pesi` **oppure** si evita rilanciando senza `--con-loftr`. Tenerli
    insieme faceva stampare le istruzioni sbagliate a chi aveva solo il
    checkpoint mancante.
    """
    import importlib.util

    dipendenze, loftr = [], []
    for modulo in DIPENDENZE:
        if importlib.util.find_spec(modulo) is None:
            dipendenze.append(f"{modulo} — pip install -r requirements.txt")

    if con_loftr:
        for modulo in DIPENDENZE_LOFTR:
            if importlib.util.find_spec(modulo) is None:
                loftr.append(f"{modulo} non installato — pip install -r requirements.txt")
        if _manca(PESI_LOFTR):
            loftr.append(f"{PESI_LOFTR} assente — python -m scripts.scarica_pesi")
    return dipendenze, loftr


def verifica_dati(fasi: list[Fase]) -> list[str]:
    """I file AdE che servono **alle fasi selezionate**, non tutti e tre sempre.

    Pretendere l'intero corredo grezzo da chi ha lanciato `--da figura-e2` era
    un rifiuto immotivato: da `e2` in poi nessuna fase apre `data/raw/`, perché
    tutto quello che serve — ritagli, raster vettoriali e i rispettivi world
    file — sta già in `data/crops/`. Il caso non è raro né teorico: è quello di
    chiunque riceva il progetto senza le scansioni, che non sono ridistribuibili
    (§5.8), ed è anche quello di chi rifà solo la coda della sequenza dopo aver
    ritoccato una figura.
    """
    richiesti = {path for fase in fasi for path in fase.grezzi}
    for fase in fasi:
        for path in fase.grezzi_opzionali:
            if _manca(path):
                print(
                    f"  ⚠ {path} assente ({GREZZI_OPZIONALI[path]}): "
                    "non blocca, ma il pavimento va citato a mano"
                )
    return [
        f"{path} — {perche}. Vedi data/README.md"
        for path, perche in GREZZI.items()
        if path in richiesti and _manca(path)
    ]


def prima_fase_eseguibile(fasi: list[Fase]) -> str | None:
    """Da quale fase in poi si arriva in fondo anche senza i file mancanti.

    Si scorre dal fondo perché la risposta utile è la fase **più a monte** da
    cui in avanti nessuno tocca `data/raw/`: fermarsi alla prima che non chiede
    nulla darebbe un suggerimento che si schianta due fasi dopo. `None` significa
    che l'ultima fase stessa li richiede, e allora non c'è coda da salvare.
    """
    candidata = None
    for fase in reversed(fasi):
        if any(_manca(path) for path in fase.grezzi):
            break
        candidata = fase.nome
    return candidata


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

    composta = os.path.join(RADICE, "relazione", "relazione.md")
    if os.path.exists(composta):
        with open(composta, encoding="utf-8") as fh:
            testo = fh.read()
        if "non disponibile: il CSV non contiene le righe necessarie" in testo:
            problemi.append(
                "la relazione contiene tabelle vuote: il CSV non ha tutte le righe "
                "(manca una fase sperimentale, oppure e1 con --riparti è stata rilanciata dopo e2)"
            )
        if "<!-- TABELLA:" in testo:
            problemi.append("segnaposto di tabella non sostituito in relazione.md")
    return problemi, avvisi


# ----------------------------------------------------------------- esecuzione


def _csv_contiene_loftr() -> bool:
    """Se E3 sia stato eseguito lo dice il CSV, non il flag di questa invocazione.

    Riprendendo con `--da relazione` dopo una corsa `--con-loftr` completa, il
    flag è assente ma le righe LoFTR ci sono: annunciare "E3 non è stato
    eseguito" sarebbe falso proprio nel momento in cui l'utente ha finito.
    """
    percorso = os.path.join(RADICE, CSV)
    if not os.path.exists(percorso):
        return False
    with open(percorso, newline="", encoding="utf-8") as fh:
        lettore = csv.DictReader(fh)
        return any(riga.get("matcher") == "loftr" for riga in lettore)


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
    dipendenze, loftr = verifica_dipendenze(args.con_loftr)

    if dipendenze:
        print("\n\033[31mdipendenze mancanti:\033[0m")
        for p in dipendenze:
            print(f"  ✗ {p}")
        return 1

    if loftr and not args.lista:
        print("\n\033[31mmanca il materiale per E3 (LoFTR):\033[0m")
        for p in loftr:
            print(f"  ✗ {p}")
        print(
            "\nDue strade: procurarti quanto sopra, oppure rilanciare senza --con-loftr "
            "\ne fare tutto il resto (la relazione esce completa tranne la figura del "
            "\ncapitolo 10)."
        )
        return 1

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

    # I dati grezzi si controllano **dopo** aver scelto le fasi: quali servano
    # dipende da cosa si è chiesto di eseguire, e chiederli tutti a chi rifà solo
    # la coda della sequenza era un rifiuto senza motivo.
    dati = verifica_dati(fasi)
    if dati:
        # `--lista` è documentazione: risponde anche su una macchina senza i dati,
        # che è il caso di chi apre il repository per capire cosa fa.
        print(f"\n\033[{'33mdati mancanti (--lista non ne ha bisogno)' if args.lista else '31mnon si può procedere'}:\033[0m")
        for p in dati:
            print(f"  ✗ {p}")
        if not args.lista:
            bloccate = [f.nome for f in fasi if any(_manca(g) for g in f.grezzi)]
            print(
                "\nI dati AdE non stanno nel repository (§5.8): vanno riscaricati dal servizio "
                "\ndell'Agenzia delle Entrate seguendo data/README.md. Ricorda I1: il file "
                "\nutilizzabile è L675_004900, mai L675_00490Z."
            )
            print(f"\nLe chiedono solo queste fasi: {', '.join(bloccate)}.")
            ripresa = prima_fase_eseguibile(fasi)
            if ripresa:
                print(
                    f"Da `{ripresa}` in poi si legge soltanto data/crops/: se i ritagli e i raster "
                    f"\nvettoriali ci sono già, `--da {ripresa}` arriva in fondo lo stesso."
                )
            return 1
    elif not loftr:
        print("  ✓ dipendenze e dati a posto per le fasi richieste")

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
        print("\nDa qui: relazione/relazione.html si apre nel browser e si stampa in PDF.")
        if not _csv_contiene_loftr():
            print("E3 (LoFTR) non è nel CSV: rilancia con --con-loftr per la parte comparativa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
