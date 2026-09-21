"""Converte la relazione in un HTML autonomo, con le figure incorporate.

Serve a consegnare o stampare il documento: un file solo, che si apre in
qualunque browser e non ha bisogno della cartella `results/figures`. Da lì si
ottiene il PDF con la stampa del browser.

    python -m scripts.relazione_html                    # -> relazione/relazione.html
    python -m scripts.relazione_html --frammento        # senza <html>/<head>/<body>

Richiede `markdown` e `latex2mathml`, che **non** sono dipendenze della
pipeline: quella gira senza, e servono solo a questo ultimo passo. Entrambi
stanno in `requirements.txt` perché la traccia vuole i pacchetti esterni
presenti nella versione finale.

Le formule in TeX (`$...$` e `$$...$$`) diventano MathML qui, a tempo di
composizione. I browser lo rendono da soli, quindi il file consegnato resta
autosufficiente come lo è per le figure — nessun MathJax da scaricare
all'apertura. I blocchi di codice sono esclusi dalla conversione: la relazione
è piena di sessioni di shell dove ogni riga comincia con `$`.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re

LATO_MASSIMO = 1600  # px: le figure sono mostrate a ~900, il doppio basta per lo zoom
QUALITA = 85

IMMAGINE = re.compile(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>')

# Palette presa dal linguaggio delle figure stesse: il rosso è quello con cui
# viene sovrapposto il tratto storico, il verde quello degli inlier, l'inchiostro
# è il blu-nero del disegno a penna. La carta è chiara e appena calda, non crema.
CSS = """
:root {
  color-scheme: light dark;
  --inchiostro: #1b2430;
  --carta: #faf9f7;
  --carta-alt: #f1efea;
  --grigio: #5c6673;
  --linea: #ddd9d2;
  --rosso: #b23a2e;
  --verde: #2f6b4f;
  --misura: 68ch;
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --inchiostro: #e6e8ea; --carta: #12161b; --carta-alt: #1a1f26;
    --grigio: #98a2ae; --linea: #2a323c; --rosso: #e2705f; --verde: #6fb58c;
  }
}
:root[data-theme="dark"] {
  --inchiostro: #e6e8ea; --carta: #12161b; --carta-alt: #1a1f26;
  --grigio: #98a2ae; --linea: #2a323c; --rosso: #e2705f; --verde: #6fb58c;
}
:root[data-theme="light"] {
  --inchiostro: #1b2430; --carta: #faf9f7; --carta-alt: #f1efea;
  --grigio: #5c6673; --linea: #ddd9d2; --rosso: #b23a2e; --verde: #2f6b4f;
}

body {
  margin: 0;
  background: var(--carta);
  color: var(--inchiostro);
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.62;
  -webkit-text-size-adjust: 100%;
}
.foglio {
  max-width: 1180px;
  margin: 0 auto;
  padding: clamp(1.5rem, 4vw, 3.5rem) clamp(1rem, 4vw, 2.5rem) 6rem;
  display: grid;
  grid-template-columns: 1fr;
  gap: 2.5rem;
}
@media (min-width: 1040px) {
  .foglio { grid-template-columns: 210px minmax(0, 1fr); gap: 3.5rem; align-items: start; }
  .indice { position: sticky; top: 2rem; }
}

.indice {
  font-family: var(--sans);
  font-size: 0.8rem;
  line-height: 1.5;
  border-top: 2px solid var(--inchiostro);
  padding-top: 0.75rem;
}
.indice p {
  margin: 0 0 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  font-size: 0.66rem;
  color: var(--grigio);
}
.indice ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.42rem; }
.indice a { color: var(--grigio); text-decoration: none; display: block; }
.indice a:hover, .indice a:focus-visible { color: var(--rosso); }

.testo > * { max-width: var(--misura); }
.testo > h1, .testo > h2, .testo > figure, .testo > .tabella, .testo > pre { max-width: none; }

h1 {
  font-size: clamp(1.9rem, 4vw, 2.7rem);
  line-height: 1.15;
  margin: 0 0 0.6rem;
  text-wrap: balance;
  letter-spacing: -0.015em;
}
h1 + p { font-family: var(--sans); font-size: 0.95rem; color: var(--grigio); max-width: var(--misura); }
h2 {
  font-size: clamp(1.3rem, 2.4vw, 1.72rem);
  line-height: 1.25;
  margin: 3.6rem 0 1.1rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--linea);
  text-wrap: balance;
}
h3 {
  font-size: 1.12rem;
  margin: 2.2rem 0 0.7rem;
  color: var(--rosso);
  text-wrap: balance;
}
p, ul, ol { margin: 0 0 1.1rem; }
p, li { text-align: justify; text-justify: inter-word; }
li { margin-bottom: 0.35rem; }
strong { font-weight: 700; }
a { color: var(--rosso); }

blockquote {
  margin: 1.6rem 0;
  padding: 0.85rem 1.2rem;
  border-left: 3px solid var(--rosso);
  background: var(--carta-alt);
  font-family: var(--sans);
  font-size: 0.87rem;
  line-height: 1.55;
  color: var(--grigio);
}
blockquote p:last-child { margin-bottom: 0; }

hr { border: 0; border-top: 1px solid var(--linea); margin: 2.5rem 0; }

code {
  font-family: var(--mono);
  font-size: 0.85em;
  background: var(--carta-alt);
  padding: 0.12em 0.35em;
  border-radius: 3px;
}
pre {
  font-family: var(--mono);
  font-size: 0.78rem;
  line-height: 1.5;
  background: var(--carta-alt);
  border: 1px solid var(--linea);
  padding: 0.9rem 1.1rem;
  overflow-x: auto;
  margin: 0 0 1.4rem;
}
pre code { background: none; padding: 0; font-size: inherit; }

.tabella { overflow-x: auto; margin: 0 0 1.6rem; }
table {
  border-collapse: collapse;
  font-family: var(--sans);
  font-size: 0.79rem;
  font-variant-numeric: tabular-nums;
  min-width: 100%;
}
th, td { padding: 0.42rem 0.8rem; text-align: left; border-bottom: 1px solid var(--linea); white-space: nowrap; }
th {
  border-bottom: 2px solid var(--inchiostro);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.045em;
  font-size: 0.68rem;
  color: var(--grigio);
}
tbody tr:nth-child(even) { background: var(--carta-alt); }
td:first-child, th:first-child { white-space: normal; }

figure { margin: 2rem 0 2.2rem; }
figure img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid var(--linea);
  background: #fff;
}
figcaption {
  font-family: var(--sans);
  font-size: 0.78rem;
  color: var(--grigio);
  margin-top: 0.6rem;
  max-width: var(--misura);
}
figcaption::before { content: "Fig. "; }

/* MathML reso dal browser: nessun JavaScript, l'HTML resta autosufficiente. */
math { font-size: 1.02em; }
.formula {
  margin: 1.5rem 0;
  text-align: center;
  overflow-x: auto;
}
.formula math { font-size: 1.1em; }

@media print {
  .indice { display: none; }
  .foglio { display: block; max-width: none; padding: 0; }
  /* Documento breve (relazione condensata): un capitolo per pagina sprecherebbe
     spazio, quindi il testo scorre e usa tutta la larghezza del foglio. */
  .testo > * { max-width: none; }
  h2 { break-before: auto; break-after: avoid; margin: 1.6rem 0 0.8rem; padding-top: 0.6rem; }
  h3 { break-after: avoid; }
  figure, pre, .formula { break-inside: avoid; }
  figure { margin: 1rem 0 1.2rem; }
  /* Su schermo .tabella scorre in orizzontale (overflow-x: auto) quando una
     tabella con molte colonne non ci sta; in stampa non esiste uno scroll, e
     quel che sporge oltre il bordo pagina viene tagliato via in silenzio,
     senza errori. Va quindi forzato a stare nella larghezza del foglio, e le
     celle devono poter andare a capo invece di restare su una riga sola. */
  .tabella { overflow-x: visible; break-inside: auto; }
  table { width: 100%; table-layout: fixed; font-size: 0.72rem; }
  th, td { white-space: normal; word-break: break-word; padding: 0.32rem 0.5rem; }
  tr { break-inside: avoid; }
  /* Una figura quasi quadrata a larghezza piena occuperebbe da sola l'intera
     pagina: limitare l'altezza mantiene le proporzioni senza quello spreco. */
  figure img { width: auto; max-width: 100%; max-height: 115mm; margin: 0 auto; display: block; }
  body { font-size: 10.5pt; line-height: 1.4; background: #fff; color: #000; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def incorpora(html: str, base_dir: str) -> str:
    """Sostituisce i riferimenti alle figure con immagini incorporate."""
    from PIL import Image

    def sostituisci(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        percorso = os.path.normpath(os.path.join(base_dir, src))
        if not os.path.exists(percorso):
            return f'<p><em>figura mancante: {src} — rigenerala con i comandi del capitolo Modalità d\'uso</em></p>'
        with Image.open(percorso) as im:
            im = im.convert("RGB")
            if max(im.size) > LATO_MASSIMO:
                fattore = LATO_MASSIMO / max(im.size)
                im = im.resize((int(im.width * fattore), int(im.height * fattore)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=QUALITA, optimize=True)
        dati = base64.b64encode(buf.getvalue()).decode("ascii")
        return (
            f'<figure><img alt="{alt}" src="data:image/jpeg;base64,{dati}" />'
            f"<figcaption>{alt}</figcaption></figure>"
        )

    return IMMAGINE.sub(sostituisci, html)


def indice(html: str) -> str:
    """Costruisce l'indice dai titoli di capitolo, con le ancore."""
    voci = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', html, flags=re.S)
    righe = "".join(
        f'<li><a href="#{ancora}">{re.sub(r"<[^>]+>", "", titolo)}</a></li>' for ancora, titolo in voci
    )
    return f'<nav class="indice"><p>Indice</p><ol>{righe}</ol></nav>'


# --------------------------------------------------------------------- formule

# Markdown non conosce la matematica, e lasciata passare non resta semplicemente
# grezza: viene **corrotta**. In `\sum_{i=1}^{N}` i due underscore diventano una
# coppia di corsivi che si mangia mezza formula, e `attr_list` prende `{i=1}` per
# una lista di attributi e la appiccica al tag. Le formule vanno quindi estratte
# prima di markdown e reinserite dopo, già convertite in MathML.
#
# ⚠ I blocchi di codice vanno saltati, e non è un dettaglio: la relazione mostra
# sessioni di shell dove ogni comando comincia con `$`. Presi per delimitatori di
# formula, due prompt consecutivi diventerebbero una formula lunga un paragrafo.
CODICE = re.compile(r"```.*?```|``.*?``|`[^`\n]*`", re.DOTALL)
FORMULA_BLOCCO = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# Niente spazio subito dentro i delimitatori: "$ 5 e $ 7" non è una formula, e
# il prezzo scritto "$5" nemmeno, perché manca la chiusura sulla stessa riga.
FORMULA_INLINE = re.compile(r"(?<![\\$])\$(?!\s)([^\n$]+?)(?<!\s)\$(?!\$)")


def _segnaposto(i: int) -> str:
    """Un token che markdown lascia passare intatto: sole lettere e cifre."""
    return f"zzFORMULAzz{i}zzFINEzz"


def estrai_formule(testo: str) -> tuple[str, list[tuple[str, bool]]]:
    """Sostituisce le formule con segnaposto. Ritorna (testo, [(tex, è_blocco)])."""
    formule: list[tuple[str, bool]] = []

    def fuori_dal_codice(frammento: str) -> str:
        def blocco(m: re.Match) -> str:
            formule.append((m.group(1).strip(), True))
            return f"\n\n{_segnaposto(len(formule) - 1)}\n\n"

        def inline(m: re.Match) -> str:
            formule.append((m.group(1).strip(), False))
            return _segnaposto(len(formule) - 1)

        return FORMULA_INLINE.sub(inline, FORMULA_BLOCCO.sub(blocco, frammento))

    pezzi, pos = [], 0
    for m in CODICE.finditer(testo):
        pezzi.append(fuori_dal_codice(testo[pos : m.start()]))
        pezzi.append(m.group(0))  # il codice passa intatto
        pos = m.end()
    pezzi.append(fuori_dal_codice(testo[pos:]))
    return "".join(pezzi), formule


def reinserisci_formule(html: str, formule: list[tuple[str, bool]]) -> str:
    """Converte in MathML e rimette al posto dei segnaposto.

    MathML e non MathJax: i browser moderni lo rendono da soli, quindi l'HTML
    resta autosufficiente come lo sono le figure incorporate. Una formula che
    non si converte non sparisce — torna come codice, visibile e segnalata,
    perché perderla in silenzio sarebbe il modo peggiore di fallire.
    """
    if not formule:
        return html
    try:
        from latex2mathml.converter import convert
    except ImportError:
        print(
            "  ⚠ latex2mathml non installato: le formule restano in TeX grezzo\n"
            "    (pip install -r requirements.txt)"
        )
        return html

    for i, (tex, e_blocco) in enumerate(formule):
        try:
            mathml = convert(tex, display="block" if e_blocco else "inline")
        except Exception as exc:  # noqa: BLE001 — qualunque errore del parser TeX
            print(f"  ⚠ formula non convertita ({exc}): {tex[:60]}")
            mathml = f"<code>{tex}</code>"
        segno = _segnaposto(i)
        if e_blocco:
            html = html.replace(f"<p>{segno}</p>", f'<div class="formula">{mathml}</div>')
        html = html.replace(segno, mathml)
    return html


def costruisci(sorgente: str, frammento: bool = False) -> str:
    import markdown

    testo = open(sorgente, encoding="utf-8").read()
    testo, formule = estrai_formule(testo)
    corpo = markdown.markdown(
        testo, extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"]
    )
    corpo = reinserisci_formule(corpo, formule)
    corpo = incorpora(corpo, os.path.dirname(os.path.abspath(sorgente)))
    corpo = corpo.replace("<table>", '<div class="tabella"><table>').replace(
        "</table>", "</table></div>"
    )

    titolo = "Registrazione di mappe catastali storiche su cartografia moderna"
    pagina = (
        f"<title>{titolo}</title>\n<style>{CSS}</style>\n"
        f'<div class="foglio">{indice(corpo)}<article class="testo">{corpo}</article></div>'
    )
    if frammento:
        # per l'incorporamento in una pagina che fornisce già head e body
        return pagina
    return (
        '<!doctype html>\n<html lang="it">\n<head>\n<meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"<title>{titolo}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        f'<div class="foglio">{indice(corpo)}<article class="testo">{corpo}</article></div>\n'
        "</body>\n</html>"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sorgente", default=os.path.join("relazione", "relazione.md"))
    ap.add_argument("--out", default=os.path.join("relazione", "relazione.html"))
    ap.add_argument("--frammento", action="store_true", help="senza wrapper html/head/body")
    args = ap.parse_args(argv)

    html = costruisci(args.sorgente, frammento=args.frammento)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
