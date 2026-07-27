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


def _mediana(s: pd.Series, cifre: int = 3) -> float:
    """Mediana che non protesta su un gruppo interamente vuoto.

    Una configurazione con zero prove riuscite ha una colonna "sulle riuscite"
    tutta NaN: è un'informazione, non un errore, e va riportata come tale invece
    di stampare un warning di numpy.
    """
    return float("nan") if s.isna().all() else round(s.median(), cifre)


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


def tabella_e2(df: pd.DataFrame, out_path: str | None = None) -> pd.DataFrame:
    """Aggregazione di E2. Riporta anche l'RMSE mediano su TUTTE le prove.

    Su E2 il tasso di successo può essere zero, e allora "RMSE sulle prove
    riuscite" non esiste. Serve comunque un numero che descriva quanto si
    sbaglia: la mediana su tutte le prove, in metri, che è grande di proposito.
    """
    d = df[df.esperimento == "E2"].copy()
    if d.empty:
        return d
    d["rmse_ok"] = d.rmse_px.where(d.success)
    agg = (
        d.groupby(["matcher", "preprocess", "modello"])
        .agg(
            prove=("success", "size"),
            successo_pct=("success", lambda s: round(100 * s.mean(), 1)),
            rmse_m_mediano=("rmse_m", lambda s: round(s.median(), 2)),
            rmse_m_minimo=("rmse_m", lambda s: round(s.min(), 3)),
            inlier_ratio=("inlier_ratio", lambda s: round(s.median(), 3)),
            match_mediani=("n_matches", lambda s: int(s.median())),
        )
        .reset_index()
    )
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(_markdown(agg))
        print(f"tabella: {out_path}")
    return agg


def figura_e1_vs_e2(df: pd.DataFrame, out_path: str) -> None:
    """Il confronto che spiega il progetto: stesso dominio contro cross-domain.

    L'inlier ratio è la misura più leggibile del divario. L'RMSE dice quanto si
    sbaglia quando si sbaglia; l'inlier ratio dice se il matcher stava
    guardando la stessa scena.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e1 = df[(df.esperimento == "E1") & (df.degrado == 0)]
    e2 = df[df.esperimento == "E2"]
    if e1.empty or e2.empty:
        print("servono sia E1 sia E2 per il confronto: figura saltata")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    dati, etichette = [], []
    for nome, gruppo in (("E1 (stesso dominio)", e1), ("E2 (cross-domain)", e2)):
        for matcher in sorted(gruppo.matcher.unique()):
            valori = gruppo[gruppo.matcher == matcher].inlier_ratio.dropna()
            if len(valori):
                dati.append(valori)
                etichette.append(f"{matcher}\n{nome.split()[0]}")
    ax1.boxplot(dati, tick_labels=etichette, showfliers=False)
    ax1.set_ylabel("inlier ratio")
    ax1.set_title("Inlier ratio: stesso dominio contro cross-domain\n(E1 senza degradazione)")
    ax1.grid(alpha=0.3, axis="y")

    successi = pd.DataFrame(
        {
            "E1": e1.groupby("matcher").success.mean() * 100,
            "E2": e2.groupby("matcher").success.mean() * 100,
        }
    )
    successi.plot(kind="bar", ax=ax2, rot=0)
    ax2.set_ylabel("prove riuscite (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Tasso di successo\n(E1: RMSE < 0.25 m · E2: RMSE < 2 m)")
    ax2.grid(alpha=0.3, axis="y")
    _salva(fig, out_path)


def figura_e2_dettaglio(df: pd.DataFrame, out_path: str) -> None:
    """Dove il cross-domain va meno peggio: preprocessing, modello, codici CXF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df.esperimento == "E2"]
    if d.empty:
        return
    fig, assi = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, chiave, titolo in zip(
        assi,
        ("preprocess", "modello", "codici"),
        ("preprocessing", "modello geometrico", "codici CXF rasterizzati"),
    ):
        if chiave not in d or d[chiave].isna().all():
            continue
        d.groupby([chiave, "matcher"]).inlier_ratio.median().unstack().plot(
            kind="bar", ax=ax, rot=0
        )
        ax.set_ylabel("inlier ratio mediano")
        ax.set_title(f"E2 — {titolo}")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("E2 cross-domain — nessuna combinazione salva la registrazione, ma non tutte sono uguali")
    _salva(fig, out_path)


def tabella_e3(df: pd.DataFrame, out_path: str | None = None) -> pd.DataFrame:
    """Confronto classico contro neurale: per ogni esperimento e matcher, la
    configurazione migliore.

    "Migliore" = più prove riuscite; a parità, RMSE mediano più basso. Scegliere
    la configurazione migliore per ciascun matcher è l'unico confronto onesto:
    un matcher penalizzato dal preprocessing sbagliato non dice nulla sul
    matcher.
    """
    d = df[df.esperimento.isin(("E1", "E2"))].copy()
    if d.empty:
        return d
    # La colonna deve misurare quello che il suo nome dice: l'errore sulle sole
    # prove riuscite. Calcolarla su tutte e chiamarla "_ok" metterebbe in
    # tabella un numero gonfiato dagli errori grossolani con l'etichetta
    # sbagliata — e nessuno se ne accorgerebbe leggendo la relazione.
    d["rmse_m_ok"] = d.rmse_m.where(d.success)
    d["config"] = d.preprocess.astype(str) + " / " + d.modello.astype(str)

    per_config = (
        d.groupby(["esperimento", "matcher", "config"])
        .agg(
            prove=("success", "size"),
            successo_pct=("success", lambda s: round(100 * s.mean(), 1)),
            rmse_m_mediano_ok=("rmse_m_ok", _mediana),
            inlier_ratio=("inlier_ratio", lambda s: round(s.median(), 3)),
            match_mediani=("n_matches", lambda s: int(s.median())),
            t_ms=("t_match_ms", lambda s: int(s.median())),
        )
        .reset_index()
    )
    migliori = (
        per_config.sort_values(
            ["esperimento", "matcher", "successo_pct", "rmse_m_mediano_ok"],
            ascending=[True, True, False, True],
        )
        .groupby(["esperimento", "matcher"])
        .head(1)
        .reset_index(drop=True)
    )
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(_markdown(migliori))
        print(f"tabella: {out_path}")
    return migliori


def figura_e3(df: pd.DataFrame, out_path: str) -> None:
    """Accuratezza e costo, affiancati. Il costo fa parte del risultato (§8)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df.esperimento.isin(("E1", "E2"))]
    if d.empty or d.matcher.nunique() < 2:
        return
    migliori = tabella_e3(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    migliori.pivot(index="matcher", columns="esperimento", values="successo_pct").plot(
        kind="bar", ax=ax1, rot=0
    )
    ax1.set_ylabel("prove riuscite (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Configurazione migliore per matcher\n(E1: RMSE < 0.25 m · E2: RMSE < 2 m)")
    ax1.grid(alpha=0.3, axis="y")

    tempi = d.groupby("matcher").t_match_ms.median().sort_values()
    ax2.bar(tempi.index, tempi.values, color="#7f7f7f")
    ax2.set_yscale("log")
    ax2.set_ylabel("tempo di matching (ms, scala logaritmica)")
    ax2.set_title("Costo per registrazione, mediana su tutte le prove\n(CPU; LoFTR a lato 640, gli altri a piena risoluzione)")
    ax2.grid(alpha=0.3, axis="y", which="both")
    for x, y in zip(tempi.index, tempi.values):
        ax2.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=9)
    _salva(fig, out_path)


def tabella_e2_per_crop(df: pd.DataFrame, out_path: str | None = None) -> pd.DataFrame:
    """E2 crop per crop, nella configurazione migliore (ORB / Sauvola / similarità).

    L'aggregato nasconde che i crop non sono equivalenti: uno di essi fallisce
    per una ragione precisa, ed è più istruttivo della media.
    """
    d = df[
        (df.esperimento == "E2")
        & (df.matcher == "orb")
        & (df.preprocess == "sauvola")
        & (df.modello == "similarity")
    ]
    if d.empty:
        return d
    agg = d[["crop", "codici", "n_matches", "inlier_ratio", "rmse_m", "success"]].copy()
    agg["rmse_m"] = agg.rmse_m.round(3)
    agg = agg.sort_values(["crop", "codici"]).reset_index(drop=True)
    return _forse_scrivi(agg, out_path)


def tabella_e2_fattori(df: pd.DataFrame, out_path: str | None = None) -> pd.DataFrame:
    """Quanto pesano modello geometrico e scelta dei codici CXF, su tutti i matcher."""
    d = df[df.esperimento == "E2"]
    if d.empty:
        return d
    righe = []
    for chiave, etichetta in (("modello", "modello geometrico"), ("codici", "codici CXF")):
        for valore, gruppo in d.groupby(chiave):
            righe.append(
                {
                    "fattore": etichetta,
                    "valore": valore,
                    "prove": len(gruppo),
                    "successo_pct": round(100 * gruppo.success.mean(), 1),
                    "rmse_m_mediano": round(gruppo.rmse_m.median(), 2),
                    "inlier_ratio": round(gruppo.inlier_ratio.median(), 3),
                }
            )
    return _forse_scrivi(pd.DataFrame(righe), out_path)


def tabella_diagnosi_ratio(df: pd.DataFrame, out_path: str | None = None) -> pd.DataFrame:
    """Sweep del ratio test di Lowe su E2 (esperimento E2-ratio)."""
    d = df[df.esperimento == "E2-ratio"]
    if d.empty:
        return d
    agg = (
        d.groupby("ratio")
        .agg(
            match_mediani=("n_matches", lambda s: int(s.median())),
            inlier_mediani=("n_inliers", lambda s: int(s.median())),
            inlier_ratio=("inlier_ratio", lambda s: round(s.median(), 4)),
            rmse_m_mediano=("rmse_m", lambda s: round(s.median(), 1)),
            riuscite=("success", lambda s: f"{int(s.sum())}/{len(s)}"),
        )
        .reset_index()
    )
    return _forse_scrivi(agg, out_path)


def tabella_crop(out_path: str | None = None) -> pd.DataFrame:
    """I 5 ritagli di §5.6, letti dalla definizione nel codice e non ricopiati."""
    from src.prep.crop import CROPS

    return _forse_scrivi(
        pd.DataFrame(
            [
                {
                    "crop": c.nome,
                    "x0": c.x0,
                    "y0": c.y0,
                    "larghezza": c.w,
                    "altezza": c.h,
                    "X (m)": f"{min(c.x_attesa)} … {max(c.x_attesa)}",
                    "Y (m)": f"{min(c.y_attesa)} … {max(c.y_attesa)}",
                }
                for c in CROPS
            ]
        ),
        out_path,
    )


def _forse_scrivi(df: pd.DataFrame, out_path: str | None) -> pd.DataFrame:
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(_markdown(df))
        print(f"tabella: {out_path}")
    return df


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
    print("E1 — sintetico, stesso dominio")
    print(agg.to_string(index=False))

    if (df.esperimento == "E2").any():
        figura_e1_vs_e2(df, os.path.join(args.out, "m8_e1_vs_e2.png"))
        figura_e2_dettaglio(df, os.path.join(args.out, "m8_e2_dettaglio.png"))
        agg2 = tabella_e2(df, os.path.join(args.out, "m8_tabella.md"))
        print()
        print("E2 — cross-domain reale")
        print(agg2.to_string(index=False))

    if df.matcher.nunique() > 2:
        figura_e3(df, os.path.join(args.out, "m9_e3_confronto.png"))
        agg3 = tabella_e3(df, os.path.join(args.out, "m9_tabella.md"))
        print()
        print("E3 — classico contro neurale, configurazione migliore per matcher")
        print(agg3.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
