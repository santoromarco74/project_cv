"""Ricalcola dal CSV i numeri citati a mano nella prosa di relazione_sorgente.md
e li stampa accanto al testo attuale, per un confronto a vista.

Non tocca nessun file: e' uno strumento di controllo, non di generazione. Le
tabelle marcate <!-- TABELLA: ... --> sono gia' rigenerate automaticamente da
componi_relazione.py -- qui interessano invece le frasi intorno a quelle
tabelle, scritte a mano, che raccontano un numero preciso (percentuale, metri,
iterazioni) e che nessun meccanismo tiene sincronizzato col CSV.

Alcuni paragrafi (§6.1 chiusura su un crop, §6.2 rapporto sui keypoint, §6.3
gradiente sintetico, §7.1 percentuali del ratio test) NON vengono dal CSV degli
esperimenti E1/E2/E3: sono output di script diagnostici a parte. Questo
controllo li salta -- vanno riverificati a mano contro l'output di quegli
script.

Uso:
    python -m scripts.verifica_numeri_relazione [--csv results/runs.csv]
"""
from __future__ import annotations

import argparse
import math

import pandas as pd


def _sez(titolo: str) -> None:
    print()
    print(f"--- {titolo} ---")


def _riga(testo_attuale: str, ricalcolato: str, filtro: str = "") -> None:
    marcatore = "OK" if testo_attuale.strip() == ricalcolato.strip() else "DA VERIFICARE"
    print(f"[{marcatore}]")
    print(f"  testo attuale : {testo_attuale}")
    print(f"  ricalcolato   : {ricalcolato}")
    if filtro:
        print(f"  filtro usato  : {filtro}")


def sezione_72_iterazioni(df: pd.DataFrame) -> None:
    """§7.2 -- k >= ln(1-p) / ln(1-w^s), p=0.995. w = inlier_ratio mediano del
    sottoinsieme descritto nel testo; s = campione minimo (2 similarita', 3
    affine, 4 omografia)."""
    _sez("§7.2 -- iterazioni RANSAC necessarie (formula, non tabella)")

    p = 0.995

    def iterazioni(w: float, s: int) -> float:
        if w <= 0 or w >= 1:
            return math.inf
        return math.log(1 - p) / math.log(1 - w**s)

    e1_sift = df[(df.esperimento == "E1") & (df.matcher == "sift") & (df.degrado == 0)]
    if not e1_sift.empty:
        w = e1_sift.inlier_ratio.median()
        _riga(
            "E1, SIFT senza degrado (w = 0.938): bastano 3 iterazioni",
            f"w = {w:.3f}: servono {iterazioni(w, 2):.0f} iterazioni (similarita', s=2)",
            "esperimento=E1, matcher=sift, degrado=0",
        )
    else:
        print("  (nessuna riga E1/sift/degrado=0 nel CSV: filtro da correggere)")

    e2_orb_sauvola = df[
        (df.esperimento == "E2") & (df.matcher == "orb") & (df.preprocess == "sauvola")
    ]
    if not e2_orb_sauvola.empty:
        w = e2_orb_sauvola.inlier_ratio.median()
        _riga(
            "E2, ORB+Sauvola, mediana (w = 0.054): similarita' 1805 iter., omografia oltre 600 000",
            f"w = {w:.3f}: similarita' {iterazioni(w, 2):.0f} iter., "
            f"affine {iterazioni(w, 3):.0f} iter., omografia {iterazioni(w, 4):.0f} iter.",
            "esperimento=E2, matcher=orb, preprocess=sauvola",
        )
    else:
        print("  (nessuna riga E2/orb/sauvola nel CSV: filtro da correggere)")

    aspera_18 = df[
        (df.esperimento == "E2") & (df.crop == "aspera") & (df.codici.astype(str) == "18")
    ]
    if not aspera_18.empty:
        w_mediano = aspera_18.inlier_ratio.median()
        w_minimo = aspera_18.inlier_ratio.min()
        _riga(
            "E2, crop aspera, solo codice 18 (w = 0.009): oltre 65 000 iterazioni (similarita')",
            f"mediana su tutte le config = {w_mediano:.3f} ({iterazioni(w_mediano, 2):.0f} iter.), "
            f"minima osservata = {w_minimo:.3f} ({iterazioni(w_minimo, 2):.0f} iter.)",
            "esperimento=E2, crop=aspera, codici=18 -- NON so quale singola riga citasse il testo,"
            " sotto il dettaglio per matcher/preprocess/modello",
        )
        print(
            aspera_18[["matcher", "preprocess", "modello", "inlier_ratio", "n_matches"]]
            .sort_values("inlier_ratio")
            .to_string(index=False)
        )
    else:
        print("  (nessuna riga E2/aspera/codici=18 nel CSV: filtro da correggere)")


def sezione_81_tetto_e1(df: pd.DataFrame) -> None:
    """§8.1 -- caso peggiore per matcher, degrado=0, su prove riuscite e su tutte."""
    _sez("§8.1 -- E1, caso peggiore per matcher a degrado 0")

    e1_deg0 = df[(df.esperimento == "E1") & (df.degrado == 0)]
    if e1_deg0.empty:
        print("  (nessuna riga E1/degrado=0 nel CSV)")
        return

    ok = e1_deg0[e1_deg0.success]
    peggiore_ok = ok.groupby("matcher").rmse_px.max()
    _riga(
        "sulle prove riuscite, caso peggiore: 0.502 px SIFT, 0.801 LoFTR, 0.930 ORB",
        ", ".join(f"{m}: {v:.3f} px" for m, v in peggiore_ok.items()),
    )

    peggiore_tutte = e1_deg0.groupby("matcher").rmse_px.max()
    _riga(
        "su tutte le prove: 0.502 px SIFT, 4.6 px ORB, 33128 px LoFTR",
        ", ".join(f"{m}: {v:.1f} px" for m, v in peggiore_tutte.items()),
    )


def sezione_81_rotazione(df: pd.DataFrame) -> None:
    """§8.1 -- tabella RMSE mediano per matcher x rot_deg, senza preprocessing."""
    _sez("§8.1 -- tabella per rotazione (preprocess=none)")
    d = df[(df.esperimento == "E1") & (df.preprocess == "none")]
    if d.empty:
        print("  (nessuna riga E1/preprocess=none nel CSV)")
        return
    tab = d.groupby(["matcher", "rot_deg"]).rmse_px.median().unstack("rot_deg").round(3)
    print(tab.to_string())
    print("  (confronta a mano con la tabella di riga 768-772 della relazione)")


def sezione_82_degradazione(df: pd.DataFrame) -> None:
    """§8.2 -- SIFT, senza preprocessing, per livello di degrado."""
    _sez("§8.2 -- SIFT senza preprocessing, per degradazione")
    d = df[(df.esperimento == "E1") & (df.matcher == "sift") & (df.preprocess == "none")]
    if d.empty:
        print("  (nessuna riga E1/sift/preprocess=none nel CSV)")
        return
    tab = d.groupby("degrado").agg(
        rmse_px_mediano=("rmse_px", "median"),
        successo_pct=("success", lambda s: round(100 * s.mean(), 1)),
        match_mediani=("n_matches", lambda s: int(s.median())),
    )
    print(tab.to_string())
    print("  (confronta a mano con la tabella di riga 799-805 della relazione)")


def sezione_94_modello_e_codici(df: pd.DataFrame) -> None:
    """§9.4 -- successo per modello geometrico, e per ablation dei codici CXF."""
    _sez("§9.4 -- successo per modello geometrico (E2)")
    d = df[df.esperimento == "E2"]
    if d.empty:
        print("  (nessuna riga E2 nel CSV)")
        return
    per_modello = d.groupby("modello").success.mean() * 100
    _riga(
        "similarita' 53%, affine 32%, omografia 19%",
        ", ".join(f"{m}: {v:.0f}%" for m, v in per_modello.items()),
    )

    per_codici = d.groupby(d.codici.astype(str)).success.mean() * 100
    _riga(
        "codici 18+12: 40%, soli 18: 27%",
        ", ".join(f"{c}: {v:.0f}%" for c, v in per_codici.items()),
    )


def sezione_95_per_crop(df: pd.DataFrame) -> None:
    """§9.5 -- aspera (18 vs 18+12) e vedra (inlier ratio minimo).

    tabella_e2_per_crop() in report.py NON fa una mediana su tutte le
    configurazioni: filtra a una sola, quella dichiarata "migliore" in §9.2
    (orb + sauvola + similarity), e mostra le righe grezze per crop e codici.
    Il controllo va fatto sullo stesso identico filtro, altrimenti si confronta
    la mela con la pera -- versione precedente di questa funzione mediava su
    TUTTE le configurazioni ed era per questo fuorviante.
    """
    _sez("§9.5 -- crop aspera e vedra (SOLO orb+sauvola+similarity, come la tabella vera)")
    d = df[
        (df.esperimento == "E2")
        & (df.matcher == "orb")
        & (df.preprocess == "sauvola")
        & (df.modello == "similarity")
    ]
    if d.empty:
        print("  (nessuna riga E2/orb/sauvola/similarity nel CSV: controlla i nomi esatti")
        print("   di preprocess/modello nel tuo CSV, es. 'similarity' vs 'similarita')")
        return

    print("  tutte le righe (5 crop x 2 codici), stessa configurazione:")
    print(
        d[["crop", "codici", "n_matches", "inlier_ratio", "rmse_m", "success"]]
        .sort_values(["crop", "codici"])
        .to_string(index=False)
    )

    riusciti = d[d.success]
    if not riusciti.empty:
        riga_min = riusciti.loc[riusciti.inlier_ratio.idxmin()]
        print()
        print(
            f"  fra le prove RIUSCITE, l'inlier ratio piu' basso e': crop={riga_min.crop}, "
            f"codici={riga_min.codici}, inlier_ratio={riga_min.inlier_ratio:.4f}, "
            f"rmse_m={riga_min.rmse_m:.3f}"
        )
        print("  -- e' questo il crop/codici da citare in \"il piu' basso, ma riesce comunque\",")
        print("     non necessariamente vedra: controllalo contro quanto scritto nel testo.")

    _riga(
        "aspera: 125.9 m (solo 18) -> 0.98 m (18+12); vedra: inlier ratio minimo 0.024, ma riesce comunque",
        "vedi la tabella e la riga segnalata sopra",
    )


def sezione_104_degradazione_matcher(df: pd.DataFrame) -> None:
    """§10.4 -- successo per matcher, per livello di degrado, senza preprocessing.

    §12.4 dice che ogni cella di questa curva aggrega 5 prove, una per
    ritaglio -- quindi la percentuale deve essere un multiplo di 20. Se non lo
    e', il filtro sta mescolando anche rotazione/scala diverse invece di
    isolare il solo asse del degrado (come fa §8.2 per SIFT). Qui si isola
    esplicitamente rot_deg=0, scala=1.0: solo degradazione radiometrica, come
    dovrebbe essere per una curva "senza preprocessing" pulita.
    """
    _sez("§10.4 -- successo per degradazione, tutti i matcher (preprocess=none)")

    d_largo = df[(df.preprocess == "none") & (df.esperimento.isin(["E1", "E3"]))]
    if d_largo.empty:
        print("  (nessuna riga trovata: puo' darsi che LoFTR sia taggato E2 nel tuo CSV --")
        print("   riprova sostituendo il filtro con esperimento.isin(['E1', 'E2', 'E3']))")
        return

    print("  filtro largo (preprocess=none, tutte le rot_deg/scala insieme):")
    tab_largo = (d_largo.groupby(["matcher", "degrado"]).success.mean() * 100).unstack("degrado").round(0)
    print(tab_largo.to_string())

    d_stretto = d_largo[(d_largo.rot_deg == 0) & (d_largo.scala == 1.0)]
    print()
    if d_stretto.empty:
        print("  filtro stretto (rot_deg=0, scala=1.0): nessuna riga -- controlla i nomi")
        print("  esatti delle colonne rot_deg/scala nel tuo CSV (es. potrebbero essere")
        print("  stringhe invece che numeri).")
        return
    print("  filtro stretto (preprocess=none, rot_deg=0, scala=1.0 -- solo degrado):")
    conteggio = d_stretto.groupby(["matcher", "degrado"]).size().unstack("degrado")
    print("  numero di prove per cella (deve essere 5 se una per ritaglio):")
    print(conteggio.to_string())
    tab_stretto = (d_stretto.groupby(["matcher", "degrado"]).success.mean() * 100).unstack("degrado").round(0)
    print(tab_stretto.to_string())
    print("  (confronta questa seconda tabella, non la prima, con la relazione a riga 1004-1010)")


def sezione_13_conclusioni(df: pd.DataFrame) -> None:
    """§13, punto 1 -- miglior configurazione E1 in metri."""
    _sez("§13 -- E1, RMSE mediano (sulle riuscite) della configurazione migliore, in metri")
    d = df[(df.esperimento == "E1") & (df.success)]
    if d.empty:
        print("  (nessuna riga E1 con success=True nel CSV)")
        return
    per_config = d.groupby(["matcher", "preprocess"]).rmse_m.median()
    migliore = per_config.idxmin()
    _riga(
        "0.046 m mediani nella configurazione migliore",
        f"{migliore[0]}+{migliore[1]}: {per_config.min():.3f} m",
    )


def diagnostica_tag_esperimento(df: pd.DataFrame) -> None:
    """Non e' un controllo di un numero: mostra come sono taggate le righe di
    LoFTR sul cross-domain, perche' la relazione (riga 862) dice che stanno
    sotto 'E3' ma la tabella 'migliori' di figura_e3 le mostra come 'E2' --
    prima di fidarsi dei filtri sopra conviene guardare qui.
    """
    _sez("Diagnostica -- come sono taggate le righe LoFTR sul cross-domain")
    tab = df[df.matcher == "loftr"].groupby("esperimento").size()
    print(tab.to_string())
    print("  (se qui vedi 'E2' per il cross-domain, correggi i filtri di §10.x")
    print("   in questo script sostituendo 'E3' con 'E2' dove serve)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/runs.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Caricato {args.csv}: {len(df)} righe")

    diagnostica_tag_esperimento(df)
    sezione_72_iterazioni(df)
    sezione_81_tetto_e1(df)
    sezione_81_rotazione(df)
    sezione_82_degradazione(df)
    sezione_94_modello_e_codici(df)
    sezione_95_per_crop(df)
    sezione_104_degradazione_matcher(df)
    sezione_13_conclusioni(df)

    print()
    print("Non coperti da questo script (non vengono dal CSV E1/E2/E3, servono")
    print("gli script diagnostici originali): §6.1 (chiusura su ribba), §6.2")
    print("(rapporto keypoint per preprocessing), §6.3 (gradiente sintetico),")
    print("§7.1 (percentuali del ratio test), §9.3 (range corrispondenze SIFT/ORB),")
    print("§10.2/10.3 (range inlier ratio e RMSE di LoFTR con CLAHE/Sauvola).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
