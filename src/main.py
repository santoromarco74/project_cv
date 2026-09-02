"""CLI. Contratto completo delle opzioni in CLAUDE.md §9.

    python -m src.main --hist <crop.png> --modern <raster.png> [opzioni]

Senza `--jgw-hist` / `--jgw-modern` la pipeline gira e produce H_est,
semplicemente non calcola l'RMSE. Non è una comodità: è la prova architetturale
di I3. La georeferenziazione entra solo qui, in coda, e va a `evaluate`; la
pipeline riceve due array di pixel e non sa dove siano nel mondo.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from src.pipeline import Opzioni, registra

MATCHER = ("sift", "orb", "loftr")
PREPROCESS = ("none", "clahe", "otsu", "sauvola")
MODELLI = ("similarity", "affine", "homography")


def costruisci_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Registrazione di una mappa storica su cartografia moderna (CLAUDE.md §9)",
    )
    ap.add_argument("--hist", required=True, help="immagine storica")
    ap.add_argument("--modern", required=True, help="immagine di riferimento")
    ap.add_argument("--matcher", default="sift", choices=MATCHER, help="loftr = modulo B")
    ap.add_argument("--preprocess", default="sauvola", choices=PREPROCESS)
    ap.add_argument("--morph-close", type=int, default=0)
    ap.add_argument("--morph-open", type=int, default=0)
    ap.add_argument("--model", default="homography", choices=MODELLI)
    ap.add_argument("--ratio", type=float, default=0.75, help="ratio test di Lowe")
    ap.add_argument("--ransac-thresh", type=float, default=3.0, help="px")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jgw-hist", default=None, help="world file storico (solo evaluate)")
    ap.add_argument("--jgw-modern", default=None, help="world file moderno (solo evaluate)")
    ap.add_argument("--out-csv", default="results/runs.csv", help="in append")
    ap.add_argument("--out-figure", default=None, help="overlay warp + match")
    ap.add_argument("--esperimento", default="cli", help="etichetta nella colonna esperimento")
    ap.add_argument("--crop", default=None, help="etichetta nella colonna crop")
    ap.add_argument("--soglia-m", type=float, default=None, help="soglia di success in metri")
    ap.add_argument("--verbose", action="store_true")
    return ap


def _leggi(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"immagine illeggibile: {path}")
    return img


def figura_overlay(hist, modern, ris, out_path: str) -> None:
    """Warp dello storico sul moderno, più le corrispondenze inlier."""
    h, w = modern.shape[:2]
    pannelli = []
    if ris.H is not None:
        warp = cv2.warpPerspective(hist, ris.H, (w, h), borderValue=(255, 255, 255))
        overlay = modern.copy()
        maschera = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) < 128
        overlay[maschera] = (0, 0, 220)
        pannelli.append(overlay)

    if ris.stima.inliers is not None and ris.stima.n_inliers > 0:
        idx = np.flatnonzero(ris.stima.inliers)
        rng = np.random.default_rng(ris.meta.get("seed", 42))
        idx = np.sort(rng.choice(idx, size=min(60, len(idx)), replace=False))
        kp_a = [cv2.KeyPoint(float(x), float(y), 4) for x, y in ris.pts_hist[idx]]
        kp_b = [cv2.KeyPoint(float(x), float(y), 4) for x, y in ris.pts_modern[idx]]
        match = [cv2.DMatch(i, i, 0) for i in range(len(idx))]
        pannelli.append(
            cv2.drawMatches(hist, kp_a, modern, kp_b, match, None, matchColor=(0, 180, 0), flags=2)
        )

    if not pannelli:
        print("nessuna figura: stima fallita e nessun inlier", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    altezza = max(p.shape[0] for p in pannelli)
    affiancati = [
        cv2.copyMakeBorder(p, 0, altezza - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for p in pannelli
    ]
    cv2.imwrite(out_path, np.hstack(affiancati))
    print(f"figura: {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = costruisci_parser().parse_args(argv)

    hist, modern = _leggi(args.hist), _leggi(args.modern)
    opz = Opzioni(
        matcher=args.matcher,
        preprocess=args.preprocess,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        model=args.model,
        ratio=args.ratio,
        ransac_thresh=args.ransac_thresh,
        seed=args.seed,
    )

    # --- la pipeline non vede i world file (I3) ---
    ris = registra(hist, modern, opz)

    np.set_printoptions(precision=6, suppress=True)
    if ris.success:
        print(f"H_est ({args.model}):")
        print(ris.H)
    else:
        print(f"stima fallita: {ris.stima.motivo}")
    print(
        f"match {ris.stima.n_matches} · inlier {ris.stima.n_inliers} "
        f"({ris.stima.inlier_ratio:.3f}) · {ris.meta['t_match_ms']:.0f} ms"
    )
    if ris.meta.get("morfologia_ignorata"):
        print(
            f"nota: --preprocess {args.preprocess} non produce un'immagine binaria, "
            "--morph-open/--morph-close sono stati ignorati (e il CSV registra 0)",
            file=sys.stderr,
        )
    if args.verbose:
        for chiave, valore in sorted(ris.meta.items()):
            print(f"  {chiave}: {valore}")

    # --- da qui in poi si valuta: è l'unico punto che tocca la georeferenziazione ---
    riga = {
        "esperimento": args.esperimento,
        "crop": args.crop or os.path.splitext(os.path.basename(args.hist))[0],
        "matcher": args.matcher,
    }
    if args.jgw_hist and args.jgw_modern:
        from src.evaluate import valuta
        from src.groundtruth import h_true_from_jgw
        from src.io_geo import read_jgw

        H_true = h_true_from_jgw(args.jgw_hist, args.jgw_modern)
        riga |= valuta(
            ris.stima,
            H_true,
            hist.shape[1],
            hist.shape[0],
            W_hist=read_jgw(args.jgw_hist),
            soglia_m=args.soglia_m,
        )
        if riga["rmse_m"] is not None:
            esito = (
                f" · success={riga['success']}"
                if args.soglia_m is not None
                else "  (nessuna --soglia-m: success non dichiarato)"
            )
            print(f"RMSE {riga['rmse_px']:.3f} px = {riga['rmse_m']:.3f} m{esito}")
    else:
        print("nessun world file: H_est prodotta, RMSE non calcolato (I3)")
        riga |= {
            "success_stima": ris.success,
            "modello": args.model,
            "n_matches": ris.stima.n_matches,
            "n_inliers": ris.stima.n_inliers,
            "inlier_ratio": round(ris.stima.inlier_ratio, 6),
            "motivo": ris.stima.motivo,
            # senza world file `valuta` non viene chiamato, ma la riga di CSV
            # esiste lo stesso e deve descrivere l'esperimento per intero
            "ransac_thresh": ris.stima.meta.get("ransac_thresh"),
        }

    riga |= {k: ris.meta.get(k) for k in ("preprocess", "morph_open", "morph_close", "seed")}
    riga |= {k: ris.meta[k] for k in ("t_init_ms", "t_match_ms", "t_stima_ms")}
    # I parametri del matcher: senza, `--ratio 0.95` e `--ratio 0.75` scrivono
    # due righe identiche. `ransac_thresh` è già nella riga, da `valuta` o dal
    # ramo senza world file.
    from src.evaluate import parametri_matcher

    riga |= parametri_matcher(ris.meta)

    if args.out_csv:
        from src.evaluate import append_csv

        append_csv(args.out_csv, riga)
        print(f"CSV: {args.out_csv}")

    if args.out_figure:
        figura_overlay(hist, modern, ris, args.out_figure)
    return 0 if ris.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
