"""M5 — Otsu vs Sauvola vs CLAHE, ed effetto della morfologia, sui crop reali.

Due uscite:

1. Una figura per crop: ogni variante affiancata, più una riga di dettagli a
   **1:1**. Su una binarizzazione il ridimensionamento inganna sempre — un
   tratto interrotto e uno saldato diventano identici a un quarto di scala
   (I5) — quindi il giudizio si dà sulla riga a piena risoluzione.

2. Una tabella: percentuale di inchiostro e numero di keypoint SIFT per
   variante. Serve alla domanda di §7.1: binarizzare distrugge la texture su
   cui SIFT lavora? Il conteggio dei keypoint non è la risposta finale — quella
   è l'RMSE di M6 — ma è il primo numero che la riguarda (I6).

    python -m experiments.m5_preprocess
    python -m experiments.m5_preprocess --crop tutti
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from src.preprocess import applica, percentuale_inchiostro, to_gray
from src.prep.crop import CROPS

# Le varianti confrontate. La chiave finisce nel titolo del pannello.
VARIANTI: dict[str, dict] = {
    "grayscale": {"modo": "none"},
    "CLAHE": {"modo": "clahe"},
    "Otsu": {"modo": "otsu"},
    "Sauvola": {"modo": "sauvola"},
    "Sauvola + chiusura": {"modo": "sauvola", "morph_close": 1},
    "Sauvola + chiusura + pulizia": {"modo": "sauvola", "morph_close": 1, "area_min": 12},
}

# Dettaglio a piena risoluzione: un riquadro dove c'è tratto, testo e carta.
DETTAGLIO = (420, 420, 256, 256)  # x, y, w, h


def _keypoint_sift(img: np.ndarray) -> int:
    return len(cv2.SIFT_create().detect(img, None))


def figura(nome_crop: str, img: np.ndarray, out_path: str) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, w, h = DETTAGLIO
    n = len(VARIANTI)
    fig, assi = plt.subplots(2, n, figsize=(3.1 * n, 6.8))

    righe = []
    for col, (etichetta, kwargs) in enumerate(VARIANTI.items()):
        out = applica(img, **kwargs)
        binaria = kwargs["modo"] in ("otsu", "sauvola")
        righe.append(
            {
                "crop": nome_crop,
                "variante": etichetta,
                "inchiostro_%": round(percentuale_inchiostro(out) * 100, 2) if binaria else None,
                "keypoint_sift": _keypoint_sift(out),
            }
        )

        assi[0, col].imshow(out, cmap="gray", vmin=0, vmax=255)
        assi[0, col].set_title(etichetta, fontsize=9)
        assi[1, col].imshow(out[y : y + h, x : x + w], cmap="gray", vmin=0, vmax=255)
        assi[1, col].set_title(f"dettaglio 1:1 · {righe[-1]['keypoint_sift']} kp", fontsize=8)
        for ax in (assi[0, col], assi[1, col]):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"M5 — preprocessing su {nome_crop} ({img.shape[1]}x{img.shape[0]} px). "
        f"Riga sotto: dettaglio {w}x{h} a piena risoluzione.",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"figura: {out_path}")
    return righe


def illumina(gray: np.ndarray, livello: float) -> np.ndarray:
    """Gradiente di illuminazione sintetico, che scurisce verso il basso a destra.

    §7.1 prevede che Otsu ceda "su carta ingiallita con gradiente di
    illuminazione". I crop di §5.6 sono illuminati in modo uniforme, quindi da
    soli non mettono alla prova quella previsione: qui il gradiente si aggiunge
    in modo controllato, che è l'unico modo di verificarla invece di darla per
    buona.
    """
    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rampa = (xx / max(w - 1, 1) + yy / max(h - 1, 1)) / 2.0
    return np.clip(gray.astype(np.float64) * (1.0 - livello * rampa), 0, 255).astype(np.uint8)


def figura_gradiente(nome_crop: str, img: np.ndarray, out_path: str, livelli=(0.0, 0.3, 0.6, 0.9)):
    """Otsu contro Sauvola al crescere del gradiente di illuminazione."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = to_gray(img)
    fig, assi = plt.subplots(3, len(livelli), figsize=(3.1 * len(livelli), 9.6))
    righe = []
    for col, livello in enumerate(livelli):
        g = illumina(gray, livello)
        otsu_out = applica(g, modo="otsu")
        sauvola_out = applica(g, modo="sauvola")
        p_otsu = percentuale_inchiostro(otsu_out) * 100
        p_sauvola = percentuale_inchiostro(sauvola_out) * 100
        righe.append(
            {"crop": nome_crop, "gradiente": livello, "otsu_%": p_otsu, "sauvola_%": p_sauvola}
        )

        for riga, (immagine, titolo) in enumerate(
            (
                (g, f"gradiente {livello:.1f}"),
                (otsu_out, f"Otsu · {p_otsu:.2f}% inchiostro"),
                (sauvola_out, f"Sauvola · {p_sauvola:.2f}% inchiostro"),
            )
        ):
            assi[riga, col].imshow(immagine, cmap="gray", vmin=0, vmax=255)
            assi[riga, col].set_title(titolo, fontsize=9)
            assi[riga, col].set_xticks([])
            assi[riga, col].set_yticks([])

    fig.suptitle(
        f"M5 — {nome_crop}: Otsu (soglia globale) contro Sauvola (locale) "
        "al crescere del gradiente di illuminazione",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"figura: {out_path}")
    return righe


def _riquadro_piu_attivo(maschera: np.ndarray, lato: int = 200, passo: int = 25):
    """Il riquadro dove la maschera ha più pixel accesi.

    Un riquadro a coordinate fisse finisce quasi sempre dove il tratto era già
    continuo, e la figura mostrerebbe due immagini identiche: non un effetto
    assente, ma un campione mal scelto. La ricerca è deterministica.
    """
    integrale = cv2.integral(maschera.astype(np.uint8))
    h, w = maschera.shape
    migliore, massimo = (0, 0), -1
    for y in range(0, h - lato + 1, passo):
        for x in range(0, w - lato + 1, passo):
            n = (
                integrale[y + lato, x + lato]
                - integrale[y, x + lato]
                - integrale[y + lato, x]
                + integrale[y, x]
            )
            if n > massimo:
                migliore, massimo = (x, y), n
    return (*migliore, lato, lato)


def figura_chiusura(nome_crop: str, img: np.ndarray, out_path: str, riquadro=None):
    """Che cosa aggiunge la chiusura, a piena risoluzione.

    La variante "sauvola+chiusura" delle tabelle è esattamente questo: la stessa
    binarizzazione, più un passaggio di morfologia che ricongiunge il tratto. Il
    quarto pannello isola i pixel aggiunti, che è l'unico modo di vederli: sono
    pochi e sottili, e su un'immagine intera non si notano.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sauvola_intera = applica(img, modo="sauvola")
    chiusa_intera = applica(img, modo="sauvola", morph_close=1)
    aggiunti_interi = (sauvola_intera >= 128) & (chiusa_intera < 128)

    x, y, w, h = riquadro or _riquadro_piu_attivo(aggiunti_interi)
    gray = to_gray(img)[y : y + h, x : x + w]
    sauvola = sauvola_intera[y : y + h, x : x + w]
    chiuso = chiusa_intera[y : y + h, x : x + w]
    aggiunti = aggiunti_interi[y : y + h, x : x + w]
    evidenza = cv2.cvtColor(chiuso, cv2.COLOR_GRAY2RGB)
    evidenza[aggiunti] = (200, 30, 30)

    fig, assi = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax, (immagine, titolo) in zip(
        assi,
        (
            (gray, "originale (grigio)"),
            (sauvola, "Sauvola"),
            (chiuso, "Sauvola + chiusura"),
            (evidenza, f"in rosso: i {int(aggiunti.sum())} pixel aggiunti"),
        ),
    ):
        ax.imshow(immagine, cmap=None if immagine.ndim == 3 else "gray", vmin=None if immagine.ndim == 3 else 0, vmax=None if immagine.ndim == 3 else 255)
        ax.set_title(titolo, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Che cosa fa la chiusura — {nome_crop}, dettaglio {w}x{h} px a piena risoluzione, "
        f"scelto dove l'effetto è massimo.\nSul ritaglio intero aggiunge "
        f"{int(aggiunti_interi.sum())} pixel su {aggiunti_interi.size} "
        f"({100 * aggiunti_interi.mean():.2f}%): salda le interruzioni senza ispessire il disegno.",
        fontsize=11,
    )
    _salva(fig, out_path)
    return int(aggiunti_interi.sum()), aggiunti_interi.size


def _salva(fig, out_path: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"figura: {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="ribba", help="nome del crop, o 'tutti'")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args(argv)

    nomi = [c.nome for c in CROPS] if args.crop == "tutti" else [args.crop]
    tutte: list[dict] = []
    for nome in nomi:
        path = f"data/crops/{nome}.png"
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"{path} — genera prima i crop (M1)")
        tutte += figura(nome, img, os.path.join(args.out, f"m5_preprocess_{nome}.png"))

    aggiunti, totali = figura_chiusura(
        nomi[0],
        cv2.imread(f"data/crops/{nomi[0]}.png", cv2.IMREAD_COLOR),
        os.path.join(args.out, f"m5_chiusura_{nomi[0]}.png"),
    )

    gradiente = figura_gradiente(
        nomi[0],
        cv2.imread(f"data/crops/{nomi[0]}.png", cv2.IMREAD_COLOR),
        os.path.join(args.out, f"m5_gradiente_{nomi[0]}.png"),
    )

    print()
    intestazione = f"{'crop':<11} {'variante':<30} {'inchiostro':>11} {'keypoint SIFT':>14}"
    print(intestazione)
    print("-" * len(intestazione))
    base = {}
    for r in tutte:
        inchiostro = "—" if r["inchiostro_%"] is None else f"{r['inchiostro_%']:.2f}%"
        if r["variante"] == "grayscale":
            base[r["crop"]] = r["keypoint_sift"]
        delta = r["keypoint_sift"] / base[r["crop"]] if base.get(r["crop"]) else float("nan")
        print(
            f"{r['crop']:<11} {r['variante']:<30} {inchiostro:>11} "
            f"{r['keypoint_sift']:>9} ({delta:4.2f}x)"
        )

    print(f"\ngradiente di illuminazione su {gradiente[0]['crop']} — % di inchiostro rilevato")
    print(f"{'gradiente':>10} {'Otsu':>9} {'Sauvola':>9}")
    print("-" * 30)
    for r in gradiente:
        print(f"{r['gradiente']:>10.1f} {r['otsu_%']:>8.2f}% {r['sauvola_%']:>8.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
