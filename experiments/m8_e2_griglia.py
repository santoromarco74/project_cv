"""M8 — E2 cross-domain: crop storico contro raster del CXF. La griglia di §8.

    {SIFT, ORB} x {CLAHE, Sauvola, Sauvola+morfologia} x
    {similarità, affine, omografia} x 5 crop x {codice 18, codici 18+12}

Ground truth dai due world file (§5.3), pavimento del riferimento ~0.5 m. Le
prove passano dalla STESSA pipeline della CLI: qui si cambiano solo i parametri
di §9, e a M9 basterà cambiare `--matcher` per far entrare LoFTR.

Qualunque sia il risultato, è il risultato (I7). Va misurato — inlier ratio,
distribuzione, RMSE — non constatato.

    python -m experiments.m8_e2_griglia --riparti
"""
from __future__ import annotations

import argparse
import os
import time

import cv2

from src.evaluate import append_csv, parametri_matcher, valuta
from src.groundtruth import h_true_from_jgw
from src.io_geo import read_jgw
from src.pipeline import Opzioni, registra
from src.prep.crop import CROPS

# §8: grayscale+CLAHE, Sauvola, Sauvola+morfologia.
PREPROCESS = (
    ("clahe", 0, 0),
    ("sauvola", 0, 0),
    ("sauvola", 0, 1),  # + chiusura: salda le interruzioni del tratto
)
MATCHER = ("sift", "orb")
MODELLI = ("similarity", "affine", "homography")
# Ablation di §5.4: solo particelle, oppure particelle + acque/strade.
VETTORIALI = (("vec", "18"), ("vec1812", "18+12"))

# Soglia di successo per E2. Il riferimento ha un pavimento di ~0.56 m di scarto
# medio (§5.3): chiedere meno misurerebbe il rumore del riferimento, non
# l'algoritmo. 2 m è una soglia onesta per dire "registrazione utilizzabile".
SOGLIA_M = 2.0


# Diagnosi del ratio test. Su E2 SIFT produce 74-121 corrispondenze contro le
# ~730 di ORB: il sospetto è che il ratio di Lowe a 0.75, tarato su immagini
# dello stesso dominio, stia scartando quasi tutto proprio dove i descrittori
# sono per forza meno simili. §7.2 ammette 0.7-0.8 da CLI; qui si guarda oltre
# per capire, non per scegliere un valore che faccia un bel numero (I7).
RATIO_DIAGNOSI = (0.75, 0.85, 0.95, 0.99)


def _valuta_e_scrivi(args, nome, chiave_vec, codici, hist, ris, extra: dict) -> dict:
    """Compone H_true dai world file e scrive la riga. I world file entrano solo qui (I3)."""
    crop_jgw = os.path.join(args.crops_dir, f"{nome}.jgw")
    H_true = h_true_from_jgw(crop_jgw, os.path.join(args.crops_dir, f"{chiave_vec}.jgw"))
    riga = valuta(
        ris.stima,
        H_true,
        hist.shape[1],
        hist.shape[0],
        W_hist=read_jgw(crop_jgw),
        soglia_m=SOGLIA_M,
    )
    riga |= {
        "crop": nome,
        "codici": codici,
        "seed": args.seed,
        "n_kp_a": ris.meta.get("n_kp_a"),
        "n_kp_b": ris.meta.get("n_kp_b"),
        "t_match_ms": ris.meta["t_match_ms"],
        "t_stima_ms": ris.meta["t_stima_ms"],
    } | parametri_matcher(ris.meta) | extra
    append_csv(args.out_csv, riga)
    return riga


def diagnosi_ratio(args) -> int:
    """Perché SIFT trova un ordine di grandezza meno corrispondenze di ORB?

    Non serve a scegliere un ratio migliore: serve a stabilire se il fallimento
    di SIFT su E2 sia dei descrittori o del filtro che li seleziona. Sono due
    conclusioni molto diverse da scrivere in relazione.
    """
    nomi = [c.nome for c in CROPS] if args.crop == "tutti" else [args.crop]
    print(f"diagnosi ratio · SIFT · sauvola · similarity · ratio {RATIO_DIAGNOSI}\n", flush=True)
    print(f"{'crop':<11} {'ratio':>6} {'match':>7} {'inlier':>7} {'ratio_in':>9} {'RMSE m':>10}")
    print("-" * 56)
    for nome in nomi:
        hist = cv2.imread(os.path.join(args.crops_dir, f"{nome}.png"), cv2.IMREAD_COLOR)
        modern = cv2.imread(os.path.join(args.crops_dir, f"{nome}_vec.png"), cv2.IMREAD_COLOR)
        if hist is None or modern is None:
            raise FileNotFoundError(f"{nome}: mancano crop o raster (M1/M7)")
        for ratio in RATIO_DIAGNOSI:
            ris = registra(
                hist,
                modern,
                Opzioni(
                    matcher="sift",
                    preprocess="sauvola",
                    model="similarity",
                    ratio=ratio,
                    seed=args.seed,
                ),
            )
            riga = _valuta_e_scrivi(
                args,
                nome,
                f"{nome}_vec",
                "18",
                hist,
                ris,
                {
                    "esperimento": "E2-ratio",
                    "matcher": "sift",
                    "preprocess": "sauvola",
                    "morph_open": 0,
                    "morph_close": 0,
                    "ratio": ratio,
                },
            )
            rmse = "—" if riga["rmse_m"] is None else f"{riga['rmse_m']:10.3f}"
            print(
                f"{nome:<11} {ratio:>6.2f} {riga['n_matches']:>7} {riga['n_inliers']:>7} "
                f"{riga['inlier_ratio']:>9.4f} {rmse}",
                flush=True,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="tutti")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--out-csv", default="results/runs.csv")
    ap.add_argument("--riparti", action="store_true", help="cancella il CSV prima di scrivere")
    ap.add_argument(
        "--diagnosi-ratio",
        action="store_true",
        help="sweep del ratio test di Lowe per SIFT (esperimento E2-ratio)",
    )
    args = ap.parse_args(argv)

    if args.diagnosi_ratio:
        return diagnosi_ratio(args)

    if args.riparti and os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    nomi = [c.nome for c in CROPS] if args.crop == "tutti" else [args.crop]
    prove = [
        (nome, suffisso, etichetta, matcher, prep, modello)
        for nome in nomi
        for suffisso, etichetta in VETTORIALI
        for matcher in MATCHER
        for prep in PREPROCESS
        for modello in MODELLI
    ]
    print(f"{len(prove)} prove · soglia di successo {SOGLIA_M} m · seed {args.seed}")
    print(f"CSV: {args.out_csv}\n", flush=True)

    cache: dict[str, object] = {}
    riusciti = 0
    t_inizio = time.perf_counter()

    for i, (nome, suffisso, codici, matcher, (prep, m_open, m_close), modello) in enumerate(
        prove, 1
    ):
        if nome not in cache:
            path = os.path.join(args.crops_dir, f"{nome}.png")
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"{path} — genera prima i crop (M1)")
            cache[nome] = img
        chiave_vec = f"{nome}_{suffisso}"
        if chiave_vec not in cache:
            path = os.path.join(args.crops_dir, f"{chiave_vec}.png")
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"{path} — genera prima i raster (M7)")
            cache[chiave_vec] = img

        hist, modern = cache[nome], cache[chiave_vec]
        opz = Opzioni(
            matcher=matcher,
            preprocess=prep,
            morph_open=m_open,
            morph_close=m_close,
            model=modello,
            seed=args.seed,
        )
        # La pipeline riceve solo pixel: i world file entrano dopo, in valuta (I3).
        ris = registra(hist, modern, opz)

        H_true = h_true_from_jgw(
            os.path.join(args.crops_dir, f"{nome}.jgw"),
            os.path.join(args.crops_dir, f"{chiave_vec}.jgw"),
        )
        riga = valuta(
            ris.stima,
            H_true,
            hist.shape[1],
            hist.shape[0],
            W_hist=read_jgw(os.path.join(args.crops_dir, f"{nome}.jgw")),
            soglia_m=SOGLIA_M,
        )
        riga |= {
            "esperimento": "E2",
            "crop": nome,
            "matcher": matcher,
            "preprocess": f"{prep}+close" if m_close else prep,
            "morph_open": m_open,
            "morph_close": m_close,
            "codici": codici,
            "seed": args.seed,
            "n_kp_a": ris.meta.get("n_kp_a"),
            "n_kp_b": ris.meta.get("n_kp_b"),
            "t_match_ms": ris.meta["t_match_ms"],
            "t_stima_ms": ris.meta["t_stima_ms"],
        } | parametri_matcher(ris.meta)
        append_csv(args.out_csv, riga)
        riusciti += bool(riga["success"])
        if i % 20 == 0 or i == len(prove):
            print(f"  {i}/{len(prove)} · {riusciti} sotto {SOGLIA_M} m", flush=True)

    print(f"\nfatto in {time.perf_counter() - t_inizio:.0f} s · {len(prove)} righe")
    print(f"riuscite: {riusciti}/{len(prove)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
