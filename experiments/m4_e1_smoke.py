"""M4 — E1 sintetico: SIFT deve recuperare una H nota con RMSE ≈ 0.

È il controllo di correttezza della pipeline (§8): la ground truth è esatta per
costruzione, il domain gap non c'è. 

Non è ancora M6: niente CSV, niente curva, niente ORB su tutta la griglia. Qui
si risponde a una domanda sola, e la risposta è un numero.

    python -m experiments.m4_e1_smoke
    python -m experiments.m4_e1_smoke --matcher orb --modello affine
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from src.estimate import stima as stima_ransac
from src.evaluate import valuta
from src.io_geo import read_jgw
from src.matchers.classic import crea_matcher
from src.prep.crop import CROPS
from src.prep.synth import Trasformazione, genera_coppia, scala_degradazione

# Un ventaglio di trasformazioni, dalla più docile alla più cattiva. Servono a
# vedere *dove* cede, non solo se cede.
CASI = {
    "traslazione": Trasformazione(tx=40, ty=-25),
    "rotazione 10°": Trasformazione(rot_deg=10),
    "rotazione 45°": Trasformazione(rot_deg=45),
    "scala 1.25": Trasformazione(scala=1.25),
    "scala 0.75": Trasformazione(scala=0.75),
    "similarità": Trasformazione(rot_deg=15, scala=1.15, tx=30, ty=20),
    "omografia lieve": Trasformazione(rot_deg=8, scala=1.05, prospettiva=8e-5),
}


def una_prova(img, t, matcher, modello, degrado, seed, W_hist):
    b, H_true = genera_coppia(img, t, scala_degradazione(degrado) if degrado else None, seed)
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    pts_a, pts_b, meta = matcher.match(img, b)
    t_match = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    st = stima_ransac(pts_a, pts_b, modello=modello, seed=seed)
    t_stima = (time.perf_counter() - t0) * 1000

    riga = valuta(st, H_true, w, h, W_hist=W_hist)
    riga |= {"t_match_ms": round(t_match, 1), "t_stima_ms": round(t_stima, 1)}
    riga |= {k: meta[k] for k in ("n_kp_a", "n_kp_b") if k in meta}
    return riga, b, st, pts_a, pts_b


def figura(img_a, img_b, st, pts_a, pts_b, out_path, titolo, seed=42, quanti=80):
    """Le corrispondenze inlier, per guardare che cosa ha agganciato.

    Il campione è estratto a caso (con seed fisso, I9) e non preso in testa:
    SIFT restituisce i keypoint in ordine di scansione, quindi i primi 80 stanno
    tutti nella stessa fascia dell'immagine e la figura suggerirebbe un aggancio
    localizzato che non c'è.
    """
    if st.inliers is None or not st.success:
        return
    inlier = np.flatnonzero(st.inliers)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(inlier, size=min(quanti, len(inlier)), replace=False))
    kp_a = [cv2.KeyPoint(float(x), float(y), 4) for x, y in pts_a[idx]]
    kp_b = [cv2.KeyPoint(float(x), float(y), 4) for x, y in pts_b[idx]]
    match = [cv2.DMatch(i, i, 0) for i in range(len(idx))]
    vis = cv2.drawMatches(
        img_a, kp_a, img_b, kp_b, match, None,
        matchColor=(0, 180, 0), singlePointColor=None, flags=2,
    )
    # putText disegna solo ASCII: gli accenti e i separatori tipografici
    # diventerebbero '??' sull'immagine.
    ascii_titolo = titolo.encode("ascii", "replace").decode("ascii")
    cv2.putText(vis, ascii_titolo, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"figura: {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="ribba", help="crop di partenza, o 'tutti'")
    ap.add_argument("--matcher", default="sift", choices=("sift", "orb"))
    ap.add_argument("--modello", default="homography", choices=("similarity", "affine", "homography"))
    ap.add_argument("--degrado", type=float, default=0.0, help="livello in [0,1]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jgw", default="data/raw/L675_004900.jgw", help="solo per l'RMSE in metri")
    ap.add_argument("--out", default="results/figures/m4_e1_match.png")
    args = ap.parse_args(argv)

    W_hist = read_jgw(args.jgw) if os.path.exists(args.jgw) else None
    matcher = crea_matcher(args.matcher)
    nomi = [c.nome for c in CROPS] if args.crop == "tutti" else [args.crop]

    print(f"matcher {args.matcher} · modello {args.modello} · degrado {args.degrado} · seed {args.seed}\n")
    intestazione = f"{'crop':<11} {'caso':<17} {'match':>6} {'inlier':>7} {'ratio':>6} {'RMSE px':>9} {'RMSE m':>8} {'ms':>7}"
    print(intestazione)
    print("-" * len(intestazione))

    peggiore = 0.0
    for nome in nomi:
        img = cv2.imread(f"data/crops/{nome}.png", cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"data/crops/{nome}.png — genera prima i crop (M1)")
        for caso, t in CASI.items():
            riga, b, st, pa, pb = una_prova(
                img, t, matcher, args.modello, args.degrado, args.seed, W_hist
            )
            rmse_px = riga["rmse_px"]
            rmse_m = riga["rmse_m"]
            peggiore = max(peggiore, rmse_px if rmse_px is not None else float("inf"))
            col_px = "—".rjust(9) if rmse_px is None else f"{rmse_px:9.4f}"
            col_m = "—".rjust(8) if rmse_m is None else f"{rmse_m:8.4f}"
            print(
                f"{nome:<11} {caso:<17} {riga['n_matches']:>6} {riga['n_inliers']:>7} "
                f"{riga['inlier_ratio']:>6.3f} {col_px} {col_m} "
                f"{riga['t_match_ms'] + riga['t_stima_ms']:>7.0f}"
            )
            if nome == nomi[0] and caso == "similarità":
                figura(
                    img, b, st, pa, pb, args.out,
                    f"E1 {nome} - similarita - {args.matcher}", seed=args.seed,
                )

    print(f"\nRMSE peggiore sul lotto: {peggiore:.4f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
