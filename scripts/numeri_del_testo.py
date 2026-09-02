"""Stampa i numeri che il *testo* della relazione cita a mano, presi dal CSV.

    python -m scripts.numeri_del_testo

Le tabelle della relazione sono segnaposto sostituiti da `componi_relazione`, e
si aggiornano da sole a ogni rigenerazione. Le frasi intorno alle tabelle no:
"47 prove su 180", "0.41 m", "il 94% dei dati è sbagliato" sono scritte a mano.
Finché il CSV non cambia va bene; quando cambia, il testo comincia a
contraddire la tabella che ha sopra — ed è il tipo di incoerenza che un lettore
attento trova subito, perché guarda proprio lì.

Questo script non corregge niente: stampa, per ciascun paragrafo che cita
numeri a mano, i valori che il CSV dice **adesso**. Il confronto con quanto è
scritto resta un lavoro da fare a occhio, ma diventa un lavoro di dieci minuti
invece di una rilettura completa.
"""
from __future__ import annotations

import argparse

from src import report

SOGLIA_E2_M = 2.0


def _riga(etichetta: str, valore) -> None:
    print(f"  {etichetta:<44} {valore}")


def paragrafo_2(df) -> None:
    """§2.2 e §2.3 — la registrazione d'esempio del capitolo guidato.

    I numeri vengono da `experiments/m10_passo_passo.py`, che non scrive nel
    CSV: si rileggono dalla riga `E2-figura`, che è la stessa configurazione
    (ORB + Sauvola + similarità su ribba) usata per la figura di §9.2.
    """
    print("\n§2.2 e §2.3 — l'esempio del capitolo guidato")
    print("  ATTENZIONE: i numeri esatti li stampa `python -m experiments.m10_passo_passo`")
    print("  (match, inlier, RMSE). Qui sotto la riga E2-figura, per confronto.\n")
    d = df[df.esperimento == "E2-figura"]
    if d.empty:
        print("  (nessuna riga E2-figura nel CSV)")
        return
    r = d.iloc[-1]
    _riga("corrispondenze", int(r.n_matches))
    _riga("inlier", int(r.n_inliers))
    _riga("frazione corretta", f"{r.inlier_ratio:.3f}  ({100 * r.inlier_ratio:.1f}%)")
    _riga("quindi 'sbagliato'", f"{100 * (1 - r.inlier_ratio):.1f}%")
    _riga("RMSE", f"{r.rmse_m:.3f} m")


def paragrafo_8(df) -> None:
    """§8.1 e §8.2 — tetto di prestazione e soglia di rottura di E1."""
    print("\n§8.1 — il tetto di prestazione (E1, degradazione 0, senza preprocessing)")
    d = df[(df.esperimento == "E1") & (df.degrado == 0) & (df.preprocess == "none")]
    for matcher in sorted(d.matcher.unique()):
        m = d[(d.matcher == matcher) & d.success_stima]
        if m.empty or m.rmse_px.isna().all():
            continue
        peggiore = m.rmse_px.max()
        _riga(f"{matcher}: RMSE peggiore su {len(m)} combinazioni", f"{peggiore:.3f} px")

    print("\n§8.2 — la rottura è un precipizio (SIFT, preprocess none, rot 15)")
    d = df[
        (df.esperimento == "E1")
        & (df.matcher == "sift")
        & (df.preprocess == "none")
        & (df.rot_deg == 15)
    ]
    print(f"  {'degradazione':>12} {'RMSE mediano':>14} {'successo':>10} {'corrispondenze':>15}")
    for livello, g in d.groupby("degrado"):
        ok = g.rmse_px.where(g.success_stima)
        mediana = "—" if ok.isna().all() else f"{ok.median():.3f} px"
        print(
            f"  {livello:>12.2f} {mediana:>14} "
            f"{100 * g.success.mean():>9.0f}% {int(g.n_matches.median()):>15}"
        )


def paragrafo_9(df) -> None:
    """§9.2 — quante prove classiche riescono, e qual è la configurazione migliore."""
    print("\n§9.2 — E2, solo SIFT e ORB (le righe LoFTR sono E3)")
    d = df[(df.esperimento == "E2") & (df.matcher.isin(("sift", "orb")))]
    _riga("prove classiche", len(d))
    _riga(f"riuscite (RMSE < {SOGLIA_E2_M} m)", int(d.success.sum()))

    print("\n  configurazioni migliori di E2, per tasso di successo:")
    agg = (
        df[df.esperimento == "E2"]
        .assign(rmse_ok=lambda x: x.rmse_m.where(x.success))
        .groupby(["matcher", "preprocess", "modello"])
        .agg(
            successo=("success", lambda s: round(100 * s.mean(), 1)),
            rmse_mediano=("rmse_m", lambda s: round(s.median(), 3)),
            rmse_mediano_ok=("rmse_ok", lambda s: None if s.isna().all() else round(s.median(), 3)),
        )
        .reset_index()
        .sort_values(["successo", "rmse_mediano"], ascending=[False, True])
    )
    print(agg.head(6).to_string(index=False))


def paragrafo_10(df) -> None:
    """§10.2, §10.3 e §10.4 — il confronto classico contro neurale."""
    d = df[df.esperimento == "E2"]

    print("\n§10.2 — inlier ratio e corrispondenze, sulle sole righe con Sauvola")
    sauvola = d[d.preprocess.astype(str).str.startswith("sauvola")]
    for matcher in ("loftr", "sift", "orb"):
        m = sauvola[sauvola.matcher == matcher]
        if m.empty:
            continue
        per_config = m.groupby(["preprocess", "modello"]).inlier_ratio.median()
        match = m.groupby("preprocess").n_matches.median()
        _riga(
            f"{matcher}: inlier ratio",
            f"{per_config.min():.3f} – {per_config.max():.3f}   "
            f"corrispondenze {int(match.min())} – {int(match.max())}",
        )

    print("\n§10.3 — LoFTR e il preprocessing")
    loftr = d[d.matcher == "loftr"]
    for prep, g in loftr.groupby("preprocess"):
        ok = 100 * g.groupby("modello").success.mean()
        rmse = g.groupby("modello").rmse_m.median()
        _riga(
            str(prep),
            f"corrispondenze {int(g.n_matches.median())} · "
            f"successo {ok.min():.0f}–{ok.max():.0f}% · "
            f"RMSE {rmse.min():.2f}–{rmse.max():.2f} m",
        )

    print("\n§10.4 — fragilità alla degradazione (E1, preprocess none, rot 15)")
    e1 = df[(df.esperimento == "E1") & (df.preprocess == "none") & (df.rot_deg == 15)]
    livelli = sorted(e1.degrado.dropna().unique())
    matchers = [m for m in ("loftr", "orb", "sift") if m in set(e1.matcher)]
    print(f"  {'degradazione':>12} " + " ".join(f"{m:>8}" for m in matchers))
    for livello in livelli:
        riga = e1[e1.degrado == livello]
        celle = [f"{100 * riga[riga.matcher == m].success.mean():>7.0f}%" for m in matchers]
        print(f"  {livello:>12.2f} " + " ".join(celle))


def conclusioni(df) -> None:
    """§13 — i numeri citati nelle conclusioni."""
    print("\n§13 — conclusioni")
    e1 = df[(df.esperimento == "E1") & df.success]
    if not e1.rmse_m.isna().all():
        migliore = (
            e1.groupby(["matcher", "preprocess"]).rmse_m.median().sort_values().head(1)
        )
        for (matcher, prep), valore in migliore.items():
            _riga("punto 1 — RMSE mediano migliore su E1", f"{valore:.3f} m ({matcher} / {prep})")

    d = df[df.esperimento == "E2"]
    fattori = d.groupby("modello").agg(
        successo=("success", lambda s: round(100 * s.mean(), 1)),
        prove=("success", "size"),
    )
    print("  punto 3 — successo per modello geometrico:")
    print(fattori.to_string())

    orb_clahe = d[(d.matcher == "orb") & (d.preprocess == "clahe")]
    print("  punto 4 — ORB con CLAHE, per modello:")
    for modello, g in orb_clahe.groupby("modello"):
        _riga(f"    {modello}", f"{100 * g.success.mean():.0f}% di successo")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/runs.csv")
    args = ap.parse_args(argv)

    df = report.carica(args.csv)
    print(f"{len(df)} righe · esperimenti: {sorted(df.esperimento.unique())}")
    print("\nI numeri qui sotto sono quelli che il TESTO della relazione cita a mano.")
    print("Le tabelle si aggiornano da sole: queste no.")

    paragrafo_2(df)
    paragrafo_8(df)
    paragrafo_9(df)
    paragrafo_10(df)
    conclusioni(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
