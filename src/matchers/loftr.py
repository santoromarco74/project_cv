"""LoftrMatcher (kornia). Import di torch/kornia LAZY e confinato qui (CLAUDE.md I4).

LoFTR è un matcher *detector-free*: non cerca keypoint ripetibili per poi
descriverli, ma mette in corrispondenza direttamente due griglie di feature
dense con attenzione incrociata. È esattamente ciò che dovrebbe aiutare dove
SIFT e ORB soffrono (§7.2): due disegni al tratto quasi privi di texture, dove
i rilevatori a blob e a corner hanno poco da agganciare.

⚠ I4 — torch e kornia si importano **dentro i metodi**, mai a livello di modulo.
La pipeline deve funzionare in un ambiente senza deep learning: chi non usa
`--matcher loftr` non paga né l'import né la dipendenza. Un test statico
verifica che qui non compaiano import in testa al file.

⚠ Pesi — §3 chiede che non vengano scaricati a runtime. `weights/loftr_outdoor.ckpt`
è la copia locale; se manca, il matcher lo dice e indica lo script invece di
scaricare di nascosto. Vedi weights/README.md.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# Percorso dei pesi vendorati. Non è un default di kornia: è una scelta di
# riproducibilità (§3), perché un peso scaricato a runtime può cambiare.
PESI_DEFAULT = os.path.join("weights", "loftr_outdoor.ckpt")

# LoFTR lavora su una griglia a 1/8 della risoluzione e il costo cresce con il
# quadrato del lato. A piena risoluzione su CPU un confronto richiede minuti e
# molti GB; 640 px è il compromesso usato in letteratura. I keypoint vengono
# riportati alle coordinate originali, quindi H_est resta nei pixel di partenza:
# il ridimensionamento è interno al matcher, non una misura su immagine ridotta
# (I5). Va però dichiarato nel confronto, perché è una differenza reale
# rispetto a SIFT e ORB, che girano a piena risoluzione.
MAX_LATO_DEFAULT = 640


class LoftrMatcher:
    """LoFTR via kornia, sulla stessa interfaccia di SiftMatcher e OrbMatcher."""

    nome = "loftr"

    def __init__(
        self,
        pesi: str = PESI_DEFAULT,
        max_lato: int = MAX_LATO_DEFAULT,
        soglia_conf: float = 0.5,
        device: str | None = None,
    ):
        self.pesi = pesi
        self.max_lato = max_lato
        self.soglia_conf = soglia_conf
        self.device = device
        self._modello = None

    # ------------------------------------------------------------------ interno

    def _carica(self):
        """Costruisce il modello al primo uso. Qui, e solo qui, entrano torch e kornia."""
        if self._modello is not None:
            return self._modello

        import kornia  # noqa: PLC0415  (lazy per I4)
        import torch  # noqa: PLC0415

        if not os.path.exists(self.pesi):
            raise FileNotFoundError(
                f"pesi LoFTR non trovati in {self.pesi}. "
                "§3 vieta di scaricarli a runtime: esegui `python -m scripts.scarica_pesi` "
                "una volta sola, poi il file resta locale. Vedi weights/README.md."
            )

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        modello = kornia.feature.LoFTR(pretrained=None)
        stato = torch.load(self.pesi, map_location="cpu")
        modello.load_state_dict(stato.get("state_dict", stato))
        self._modello = modello.to(self.device).eval()
        return self._modello

    def _prepara(self, img: np.ndarray):
        """Grayscale, ridimensionamento a multiplo di 8, tensore [1,1,H,W] in [0,1].

        Ritorna anche i fattori di scala, che servono a riportare i keypoint
        nelle coordinate dell'immagine originale.
        """
        import torch  # noqa: PLC0415

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h, w = gray.shape[:2]
        fattore = min(1.0, self.max_lato / max(h, w))
        nh, nw = max(8, int(round(h * fattore / 8)) * 8), max(8, int(round(w * fattore / 8)) * 8)
        if (nh, nw) != (h, w):
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        tensore = torch.from_numpy(gray).float()[None, None] / 255.0
        return tensore.to(self.device), (w / nw, h / nh)

    # ------------------------------------------------------------------ interfaccia

    def prepara(self) -> None:
        """Carica il modello adesso, invece che alla prima `match`.

        Metodo opzionale dell'interfaccia (matchers/base.py): la pipeline lo
        chiama, se c'è, fuori dal cronometro del matching. Senza, il
        caricamento del checkpoint finirebbe dentro il tempo di inferenza, che
        è uno dei risultati del confronto classico/neurale (§10.1).
        """
        self._carica()

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        import torch  # noqa: PLC0415

        modello = self._carica()
        ta, scala_a = self._prepara(img_a)
        tb, scala_b = self._prepara(img_b)

        with torch.inference_mode():
            uscita = modello({"image0": ta, "image1": tb})

        pts_a = uscita["keypoints0"].cpu().numpy().astype(np.float64)
        pts_b = uscita["keypoints1"].cpu().numpy().astype(np.float64)
        conf = uscita["confidence"].cpu().numpy()

        meta = {
            "matcher": self.nome,
            "device": self.device,
            "max_lato": self.max_lato,
            "soglia_conf": self.soglia_conf,
            "n_kp_a": int(len(pts_a)),
            "n_kp_b": int(len(pts_b)),
            "n_grezzi": int(len(conf)),
        }
        if len(conf):
            tieni = conf >= self.soglia_conf
            pts_a, pts_b, conf = pts_a[tieni], pts_b[tieni], conf[tieni]
            meta["conf_mediana"] = float(np.median(conf)) if len(conf) else 0.0

        # dalle coordinate dell'immagine ridimensionata a quelle originali
        pts_a = pts_a * np.array(scala_a, dtype=np.float64)
        pts_b = pts_b * np.array(scala_b, dtype=np.float64)
        meta["n_matches"] = int(len(pts_a))
        return pts_a.reshape(-1, 2), pts_b.reshape(-1, 2), meta
