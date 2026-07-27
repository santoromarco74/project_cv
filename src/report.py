"""Figure e tabelle, generate dal CSV degli esperimenti.

Le tabelle della relazione sono aggregazioni di `results/runs.csv`, non numeri
ricopiati a mano (§7.4). Questo modulo è l'unico posto dove il CSV diventa
figura: se un numero della relazione non si ottiene da qui, non è riproducibile.

    python -m src.report --csv results/runs.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

# Pavimento del riferimento reale (§5.3): 0.56 m di scarto medio sulle 76 coppie
# omologhe del ricampionamento. Su E1 non si applica — lì la ground truth è
# esatta — ma va disegnato lo stesso, perché è il metro di paragone di E2.
PAVIMENTO_M = 0.56


def carica(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("rmse_px", "rmse_m", "inlier_ratio", "err_max_px"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def curva_degradazione(df: pd.DataFrame, out_path: str, preprocess: str = "none") -> None:
    """RMSE e tasso di successo al crescere della degradazione — la prima figura di E1.

    Due pannelli e non uno: l'RMSE da solo mente, perché è definito soltanto
    sulle prove riuscite. Dove il matching fallisce l'RMSE sparisce dalla media e
    la curva sembra migliorare. Il tasso di successo accanto racconta il resto.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df.esperimento == "E1") & (df.preprocess == preprocess) & (df.rot_deg == 15)]
    if d.empty:
        print(f"nessuna riga per preprocess={preprocess}: figura saltata")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for matcher, gruppo in d.groupby("matcher"):
        agg = gruppo.groupby("degrado").agg(
            mediana=("rmse_px", "median"),
            q1=("rmse_px", lambda s: s.quantile(0.25)),
            q3=("rmse_px", lambda s: s.quantile(0.75)),
            successo=("success", "mean"),
        )
        ax1.plot(agg.index, agg.mediana, marker="o", label=f"{matcher} (mediana)")
        ax1.fill_between(agg.index, agg.q1, agg.q3, alpha=0.15)
        ax2.plot(agg.index, agg.successo * 100, marker="o", label=matcher)

    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax1.text(0.02, 1.15, "soglia di successo (1 px)", fontsize=8, color="grey")
    ax1.set_yscale("log")
    ax1.set_xlabel("livello di degradazione")
    ax1.set_ylabel("RMSE (px, scala logaritmica)")
    # Mediana e non media: oltre la soglia di rottura RANSAC ritorna comunque
    # una H, ma sbagliata di migliaia di pixel. Una sola di quelle stime
    # trascina la media di tutto il gruppo e nasconde il comportamento tipico.
    ax1.set_title(
        f"E1 — RMSE vs degradazione (preprocess: {preprocess})\n"
        "mediana, banda interquartile; log perché gli errori grossolani sono 10⁴ volte i buoni"
    )
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    ax2.set_xlabel("livello di degradazione")
    ax2.set_ylabel("prove riuscite (%)")
    ax2.set_title("E1 — tasso di successo\n(RMSE sotto 1 px, cioè 0.25 m)")
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.3)
    ax2.legend()

    _salva(fig, out_path)


def curva_ampiezza(df: pd.DataFrame, out_path: str, preprocess: str = "none") -> None:
    """RMSE al crescere della sola ampiezza della trasformazione, senza degrado."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df.esperimento == "E1") & (df.preprocess == preprocess) & (df.degrado == 0) & (df.scala == 1.0)]
    if d.empty:
        print(f"nessuna riga di ampiezza per preprocess={preprocess}: figura saltata")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for matcher, gruppo in d.groupby("matcher"):
        agg = gruppo.groupby("rot_deg").rmse_px.agg(["mean", "max"])
        ax.plot(agg.index, agg["mean"], marker="o", label=f"{matcher} (media)")
        ax.fill_between(agg.index, agg["mean"], agg["max"], alpha=0.12)
    ax.set_xlabel("rotazione (gradi)")
    ax.set_ylabel("RMSE (px)")
    ax.set_title(f"E1 — RMSE vs ampiezza della rotazione (preprocess: {preprocess})")
    ax.grid(alpha=0.3)
    ax.legend()
    _salva(fig, out_path)


def confronto_preprocess(df: pd.DataFrame, out_path: str) -> None:
    """Il preprocessing aiuta o toglie? Risposta in RMSE, non in numero di keypoint.

    M5 aveva misurato che binarizzare costa il 21-39% dei keypoint. Questa è la
    domanda vera di §7.1, e la risposta è l'errore finale.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df.esperimento == "E1") & (df.rot_deg == 15) & (df.degrado <= 1.0)]
    if d.empty:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    d = d.copy()
    d["rmse_ok"] = d.rmse_px.where(d.success)
    d.groupby(["preprocess", "matcher"]).rmse_ok.median().unstack().plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("RMSE mediano sulle prove riuscite (px)")
    ax.set_title("E1 — preprocessing ed errore finale\n(degradazione ≤ 1.0)")
    ax.grid(alpha=0.3, axis="y")

    (d.groupby(["preprocess", "matcher"]).success.mean().unstack() * 100).plot(
        kind="bar", ax=ax2, rot=0
    )
    ax2.set_ylabel("prove riuscite (%)")
    ax2.set_title("E1 — preprocessing e tasso di successo")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3, axis="y")
    _salva(fig, out_path)


def tabella(df: pd.DataFrame, out_path: str | None = None) -> pd.DataFrame:
    """Aggregazione per matcher e preprocessing: la tabella della relazione."""
    d = df[df.esperimento == "E1"].copy()
    # RMSE solo sulle prove riuscite: dove la registrazione salta, RANSAC ritorna
    # comunque una H e l'errore è di migliaia di pixel. Mediarci sopra produce un
    # numero che non descrive né i casi buoni né i cattivi. Il tasso di successo,
    # nella colonna accanto, è ciò che descrive i cattivi.
    d["rmse_ok"] = d.rmse_px.where(d.success)
    agg = (
        d.groupby(["matcher", "preprocess"])
        .agg(
            prove=("success", "size"),
            successo_pct=("success", lambda s: round(100 * s.mean(), 1)),
            rmse_px_mediano_ok=("rmse_ok", lambda s: round(s.median(), 3)),
            rmse_px_max_ok=("rmse_ok", lambda s: round(s.max(), 3)),
            inlier_ratio=("inlier_ratio", lambda s: round(s.mean(), 3)),
            match_medi=("n_matches", lambda s: int(s.mean())),
            t_ms=("t_match_ms", lambda s: int(s.mean())),
        )
        .reset_index()
    )
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(_markdown(agg))
        print(f"tabella: {out_path}")
    return agg


def _markdown(df: pd.DataFrame) -> str:
    """Tabella Markdown senza `tabulate`.

    `DataFrame.to_markdown` esiste ma richiede una dipendenza in più, e lo stack
    di CLAUDE.md §3 è volutamente stretto: non vale una nuova riga in
    requirements.txt per formattare sei colonne.
    """
    colonne = list(df.columns)
    righe = [[str(v) for v in riga] for riga in df.itertuples(index=False)]
    larghezze = [
        max(len(str(c)), *(len(r[i]) for r in righe)) if righe else len(str(c))
        for i, c in enumerate(colonne)
    ]
    fmt = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, larghezze)) + " |"
    out = [fmt(colonne), "|" + "|".join("-" * (w + 2) for w in larghezze) + "|"]
    out += [fmt(r) for r in righe]
    return "\n".join(out) + "\n"


def _salva(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"figura: {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Figure e tabelle dal CSV (§7.4)")
    ap.add_argument("--csv", default="results/runs.csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args(argv)

    df = carica(args.csv)
    print(f"{len(df)} righe · esperimenti: {sorted(df.esperimento.unique())}\n")

    curva_degradazione(df, os.path.join(args.out, "m6_rmse_vs_degradazione.png"))
    curva_ampiezza(df, os.path.join(args.out, "m6_rmse_vs_ampiezza.png"))
    confronto_preprocess(df, os.path.join(args.out, "m6_preprocess.png"))
    agg = tabella(df, os.path.join(args.out, "m6_tabella.md"))

    print()
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
