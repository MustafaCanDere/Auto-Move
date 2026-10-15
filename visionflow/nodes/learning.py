"""Train-it-yourself blocks: sample collection, a quick classifier and anomaly detection.

None of these need a GPU or a deep learning framework: they learn in seconds from a folder
of example images, so a user can teach the program a new product without writing code.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import cv2
import numpy as np

from ..core import Node, Param, PortType, register

CAT = "3b · Öğrenme (Kendi Örneklerinle)"
IMG, NUM, BOOL, TEXT, DET = PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.TEXT, PortType.DETECTIONS
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _imread(path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _imwrite(path, img) -> None:
    ok, buf = cv2.imencode(pathlib.Path(path).suffix or ".png", img)
    if ok:
        buf.tofile(str(path))


def _images_in(folder: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT) if folder.is_dir() else []


def _folder_stamp(folder: pathlib.Path) -> tuple:
    """Changes when the class samples change (a class added/removed, an image added/changed), used to retrain only
    then. The review queue (_incelenecek) and other "_" folders do not count: collecting must not retrain."""
    if not folder.is_dir():
        return ()
    names, stamp, count = [], 0.0, 0
    for d in sorted(folder.iterdir()):
        if d.is_dir() and not d.name.startswith(("_", ".")):
            names.append(d.name)
            stamp = max(stamp, d.stat().st_mtime)
            for p in d.rglob("*"):
                stamp = max(stamp, p.stat().st_mtime)
                count += 1
    return tuple(names), stamp, count


def crop_box(img: np.ndarray, box, pad: int = 0) -> np.ndarray | None:
    """The box (x, y, w, h) plus a margin, cut to the image. None when the box is outside the image or broken
    (NaN), instead of Python's negative indexes silently cutting from the other side."""
    try:
        x, y, bw, bh = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x, y, bw, bh])):
        return None
    h, w = img.shape[:2]
    x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
    x1, y1 = min(w, int(x + bw) + pad), min(h, int(y + bh) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def features(img: np.ndarray) -> np.ndarray:
    """Colour + shape descriptor: HSV histogram, coarse gray layout, gradient orientations."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist /= hist.sum() + 1e-6
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    layout = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).flatten()
    layout = (layout - layout.mean()) / (layout.std() + 1e-6)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    cells = []
    for cy in range(4):          # 4x4 cells, 9 unsigned orientation bins each
        for cx in range(4):
            m = mag[cy * 24:(cy + 1) * 24, cx * 24:(cx + 1) * 24].flatten()
            a = ang[cy * 24:(cy + 1) * 24, cx * 24:(cx + 1) * 24].flatten() % 180
            h, _ = np.histogram(a, bins=9, range=(0, 180), weights=m)
            cells.append(h / (np.linalg.norm(h) + 1e-6))
    return np.concatenate([hist * 4, layout * 0.15, np.concatenate(cells) * 0.5]).astype(np.float32)


REVIEW_DIR = "_incelenecek"     # inside the data folder; folders starting with "_" are not classes


def class_folders(folder: pathlib.Path) -> list[pathlib.Path]:
    """Sub-folders that are classes: they hold images and do not start with "_" or "." (review queue, etc.)."""
    if not folder.is_dir():
        return []
    return sorted(d for d in folder.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")) and _images_in(d))


def augmented(img: np.ndarray) -> list[np.ndarray]:
    """A few believable variations of one sample (mirror, darker, brighter, slightly turned), so a class with
    5 photos is learned almost like one with 25."""
    h, w = img.shape[:2]
    out = [cv2.flip(img, 1), cv2.convertScaleAbs(img, alpha=0.8, beta=0), cv2.convertScaleAbs(img, alpha=1.2, beta=8)]
    for angle in (-8, 8):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out.append(cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    return out


class QuickModel:
    """k-nearest-neighbour classifier over standardised features."""

    def __init__(self, k: int = 3, augment: bool = False):
        self.k = k
        self.augment = augment
        self.x = np.zeros((0, 1), np.float32)
        self.y: list[int] = []
        self.classes: list[str] = []

    def fit(self, folder: pathlib.Path) -> "QuickModel":
        xs, ys, src = [], [], []
        self.classes = [d.name for d in class_folders(folder)]
        for ci, name in enumerate(self.classes):
            for p in _images_in(folder / name):
                img = _imread(p)
                if img is not None:
                    for variant in [img] + (augmented(img) if self.augment else []):
                        xs.append(features(variant))
                        ys.append(ci)
                        src.append(len(src) if not self.augment else p)
        if len(self.classes) < 2:
            raise ValueError("En az 2 sınıf klasörü ve içinde resim gerekli (ör. veri/saglam, veri/hatali)")
        x = np.stack(xs)
        sd = x.std(0)
        # the floor keeps near-constant features (empty histogram bins) from dominating; with few, very similar
        # samples more than half the features can be constant (median 0), so the median alone is not enough
        floor = max(float(np.median(sd)), 0.1 * float(sd.mean()), 1e-3)
        self.mu, self.sd = x.mean(0), sd + floor
        self.x, self.y = (x - self.mu) / self.sd, ys
        _, self.src = np.unique(np.array([str(s) for s in src]), return_inverse=True)
        return self

    def predict(self, img: np.ndarray) -> tuple[str, float, int]:
        return self.predict_full(img)[:3]

    def predict_full(self, img: np.ndarray) -> tuple[str, float, int, float]:
        """(label, vote share, class index, certainty). Certainty is 0 when the image is as close to another
        class as to its own and 1 when it is far from every other class: a well-known image of a clean dataset
        still gets a vote share of 1.0, certainty shows how close the call actually was."""
        f = (features(img) - self.mu) / self.sd
        d = np.linalg.norm(self.x - f, axis=1)
        # k nearest *photos*: the variants of one augmented photo vote once, with their closest distance, so
        # augmentation does not turn k=3 into "3 copies of the same photo" (which made every score 1.0)
        idx, seen = [], set()
        for i in np.argsort(d):
            if self.src[i] not in seen:
                seen.add(self.src[i])
                idx.append(i)
                if len(idx) == self.k:
                    break
        votes = np.zeros(len(self.classes))
        for i in idx:
            votes[self.y[i]] += 1.0 / (d[i] + 1e-3)
        ci = int(votes.argmax())
        y = np.asarray(self.y)
        own, other = d[y == ci].min(), d[y != ci].min()
        certainty = float(np.clip((other - own) / (other + own + 1e-6), 0.0, 1.0)) * float(votes[ci] / votes.sum())
        return self.classes[ci], float(votes[ci] / votes.sum()), ci, certainty


@register
class SampleCollector(Node):
    title = "Örnek Topla"
    category = CAT
    description = ("Eğitim için örnek resim kaydeder. Tetik geldiğinde görüntüyü klasör/sınıf_adı içine yazar. "
                   "Tespit bağlanırsa her nesneyi ayrı kırpıp kaydeder.")
    inputs = [("image", IMG), ("trigger", BOOL), ("detections", DET)]
    outputs = [("saved", NUM)]
    params = [Param("folder", "folder", "veri", "Veri seti klasörü"),
              Param("label", "str", "saglam", "Sınıf adı"),
              Param("every_n", "int", 0, "Tetik yoksa her N karede (0 = kapalı)", 0, 100000),
              Param("pad", "int", 10, "Kırpma payı (px)", 0, 500)]

    def setup(self):
        self.state.update(n=0, saved=0, prev=False)

    def _fire(self, trig) -> bool:
        if trig is not None:
            fire = bool(trig) and not self.state["prev"]
            self.state["prev"] = bool(trig)
            return fire
        n = self.v("every_n")
        return bool(n) and self.state["n"] % n == 0

    def process(self, inputs):
        img = inputs["image"]
        self.state["n"] += 1
        if img is None or not self._fire(inputs["trigger"]):
            return {"saved": self.state["saved"]}
        folder = pathlib.Path(self.v("folder")) / self.v("label")
        folder.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        crops = []
        if inputs["detections"] is not None:      # detector connected: only its boxes, none if it found nothing
            for d in inputs["detections"]:
                c = crop_box(img, d.get("box"), self.v("pad"))
                if c is not None:
                    crops.append(c)
        else:
            crops = [img]
        for i, c in enumerate(crops):
            if c.size:
                _imwrite(folder / f"{stamp}_{i}.png", c)
                self.state["saved"] += 1
        return {"saved": self.state["saved"]}


@register
class QuickClassifier(Node):
    title = "Hızlı Sınıflandırıcı (Eğit)"
    category = CAT
    description = ("Kendi örneklerinizle saniyeler içinde eğitilir, GPU gerekmez. Klasör yapısı: "
                   "veri/saglam/*.png, veri/hatali/*.png … Her alt klasör bir sınıftır. Klasöre yeni resim "
                   "eklenince model kendiliğinden yeniden eğitilir.")
    inputs = [("image", IMG), ("detections", DET)]
    outputs = [("label", TEXT), ("score", NUM), ("detections", DET), ("is_first", BOOL), ("certainty", NUM)]
    params = [Param("folder", "folder", "veri", "Veri seti klasörü"),
              Param("k", "int", 3, "Komşu sayısı (k)", 1, 25),
              Param("min_score", "float", 0.0, "Altındaysa 'bilinmiyor'", 0, 1, 0.01),
              Param("pad", "int", 10, "Kırpma payı (px, Örnek Topla ile aynı olmalı)", 0, 500),
              Param("augment", "bool", False, "Veri artırma (az örnekle daha iyi: ayna, ışık, açı)")]

    def setup(self):
        self.state.update(model=None, stamp=None, checked=0)

    def _model(self) -> QuickModel:
        folder = pathlib.Path(self.v("folder"))
        self.state["checked"] += 1
        # check the folder for new samples about once a second at 30 FPS
        if self.state["model"] is None or self.state["checked"] % 30 == 0:
            stamp = _folder_stamp(folder)
            if stamp != self.state["stamp"]:
                if not folder.is_dir():
                    raise ValueError(f"Klasör yok: {folder}")
                try:
                    self.state["model"] = QuickModel(self.v("k"), bool(self.v("augment"))).fit(folder)
                except Exception as exc:  # noqa: BLE001
                    if self.state["model"] is None:
                        raise
                    # e.g. a sample still being written: keep the previous model, retry next check
                    print(f"[{self.title}] yeniden eğitilemedi, eski model kullanılıyor: {exc}")
                    return self.state["model"]
                self.state["stamp"] = stamp
                print(f"[{self.title}] eğitildi: {', '.join(self.state['model'].classes)}")
        return self.state["model"]

    def _classify(self, model, img):
        label, score, ci, certainty = model.predict_full(img)
        if score < self.v("min_score"):
            return "bilinmiyor", score, -1, certainty
        return label, score, ci, certainty

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        model = self._model()
        dets = []
        if inputs["detections"] is not None:
            for d in inputs["detections"]:
                crop = crop_box(img, d.get("box"), self.v("pad"))
                if crop is None:
                    continue
                label, score, ci, cert = self._classify(model, crop)
                dets.append({**d, "label": label, "score": score, "class_id": ci, "certainty": round(cert, 3)})
            first = dets[0] if dets else {"label": "", "score": 0.0, "class_id": -1, "certainty": 0.0}
            label, score, ci, cert = first["label"], first["score"], first["class_id"], first["certainty"]
        else:
            label, score, ci, cert = self._classify(model, img)
        return {"label": label, "score": round(score, 3), "detections": dets, "is_first": ci == 0,
                "certainty": round(cert, 3)}


@register
class UncertainCollector(Node):
    title = "Emin Olmadıklarını Topla"
    category = CAT
    description = ("Kendi kendine öğrenme için: sınıflandırıcı veya dedektör bir görüntüden emin olmadığında (güven "
                   "iki sınır arasında) onu inceleme klasörüne kaydeder. Hızlı Sınıflandırıcı'nın tespitlerinde "
                   "'kesinlik' kullanılır; tek görüntüde 'certainty' çıkışını güven girişine bağlayın. Araçlar → Örnekleri Öğret ile doğru sınıfa "
                   "koyduğunuzda Hızlı Sınıflandırıcı kendiliğinden yeniden eğitilir.")
    inputs = [("image", IMG), ("score", NUM), ("label", TEXT), ("detections", DET)]
    outputs = [("saved", NUM), ("uncertain", BOOL)]
    params = [Param("folder", "folder", "veri", "Veri seti klasörü (inceleme: veri/_incelenecek)"),
              Param("low", "float", 0.0, "Güven en az", 0, 1, 0.01),
              Param("high", "float", 0.5, "Güven en çok (üstü: emin)", 0, 1, 0.01),
              Param("interval", "float", 2.0, "En sık kayıt (sn)", 0, 3600, 0.1),
              Param("max_files", "int", 500, "İnceleme klasöründe en çok", 1, 100000),
              Param("pad", "int", 10, "Kırpma payı (px)", 0, 500)]

    def setup(self):
        self.state.update(saved=0, last=-1e18)

    def _uncertain(self, score) -> bool:
        try:
            s = float(score)
        except (TypeError, ValueError):
            return False
        return s == s and self.v("low") <= s < self.v("high")          # NaN is not "uncertain", it is broken

    def process(self, inputs):
        import time
        img, dets = inputs["image"], inputs["detections"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        if dets is None and inputs["score"] is None:
            raise ValueError("Güven (score) veya tespit (detections) girişini bağlayın")
        if dets is not None:
            picks = []
            for d in dets:
                conf = d.get("certainty", d.get("score"))       # Hızlı Sınıflandırıcı's certainty when present
                if self._uncertain(conf):
                    crop = crop_box(img, d.get("box"), self.v("pad"))
                    if crop is not None:
                        picks.append((crop, str(d.get("label", "")), float(conf)))
        elif inputs["label"] == "":
            picks = []         # a classifier with a detector that found nothing: an empty frame, not a hard case
        else:
            picks = [(img, str(inputs["label"] or ""), float(inputs["score"]))] if self._uncertain(inputs["score"]) else []
        now = time.monotonic()
        if picks and now - self.state["last"] >= self.v("interval"):
            folder = pathlib.Path(self.v("folder")) / REVIEW_DIR
            folder.mkdir(parents=True, exist_ok=True)
            room = self.v("max_files") - len(_images_in(folder))
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for i, (crop, label, score) in enumerate(picks[:max(0, room)]):
                safe = safe_label(label)
                name = f"{stamp}_{self.id[:6]}_{i}"
                n = 0
                while (folder / f"{name}__{safe}__{score:.2f}.png").exists():    # coarse clock on Windows
                    n += 1
                    name = f"{stamp}_{self.id[:6]}_{i}-{n}"
                _imwrite(folder / f"{name}__{safe}__{score:.2f}.png", crop)
                self.state["saved"] += 1
            self.state["last"] = now
        return {"saved": self.state["saved"], "uncertain": bool(picks)}


def safe_label(label: str) -> str:
    """A label as a file-name part: letters, digits and '-' kept (Turkish too), anything else a single '_', never
    '__' (the separator) and never empty. "vida #12" → "vida_12"."""
    import re
    text = re.sub(r"[^\w-]+", "_", str(label), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")[:40]
    return text or "bilinmiyor"


def guess_from_name(name: str) -> tuple[str, float | None]:
    """The label and score UncertainCollector put into a file name ("..._0__hatali__0.42.png")."""
    parts = pathlib.Path(name).stem.split("__")
    if len(parts) >= 3:
        try:
            return parts[-2], float(parts[-1])
        except ValueError:
            pass
    return "", None


class AnomalyModel:
    """Per-pixel statistics of known-good images; deviations become an anomaly map."""

    def __init__(self, size: int, blur: int):
        self.size, self.blur = size, blur | 1

    def _prep(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        else:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32)
        return cv2.GaussianBlur(img, (self.blur, self.blur), 0)

    def fit(self, folder: pathlib.Path) -> "AnomalyModel":
        stack = [self._prep(img) for p in _images_in(folder) if (img := _imread(p)) is not None]
        if len(stack) < 3:
            raise ValueError("En az 3 sağlam örnek resim gerekli")
        arr = np.stack(stack)
        self.mean = arr.mean(0)
        # floor the std so flat regions don't turn sensor noise into "defects"
        self.std = np.maximum(arr.std(0), 4.0)
        self.n = len(stack)
        return self

    def score_map(self, img: np.ndarray) -> np.ndarray:
        z = np.abs(self._prep(img) - self.mean) / self.std
        return z.max(axis=2)


@register
class AnomalyDetector(Node):
    setup_params = ("blur", "folder", "size")   # read only when the block starts: changing them restarts it
    title = "Anomali Tespiti (Sağlam Örnekle)"
    category = CAT
    description = ("Sadece SAĞLAM ürün resimleriyle öğrenir; bunlardan farklı olan her şeyi (çizik, leke, eksik "
                   "parça) hata olarak işaretler. Hatalı örnek toplamak gerekmez. Ürün her karede aynı "
                   "konumda olmalıdır (gerekirse önce Kırp / ROI kullanın).")
    inputs = [("image", IMG)]
    outputs = [("heatmap", IMG), ("mask", IMG), ("detections", DET), ("ok", BOOL), ("score", NUM)]
    params = [Param("folder", "folder", "veri/saglam", "Sağlam örnek klasörü"),
              Param("threshold", "float", 6.0, "Hassasiyet (düşük = hassas)", 1, 50, 0.5),
              Param("min_area", "int", 30, "Min hata alanı (px)", 1, 1_000_000),
              Param("size", "int", 256, "Çalışma çözünürlüğü", 64, 1024, 32),
              Param("blur", "int", 5, "Yumuşatma", 1, 31)]

    def setup(self):
        self.state["model"] = AnomalyModel(self.v("size"), self.v("blur")).fit(pathlib.Path(self.v("folder")))
        print(f"[{self.title}] {self.state['model'].n} sağlam örnekle eğitildi")

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        h, w = img.shape[:2]
        z = cv2.resize(self.state["model"].score_map(img), (w, h))
        mask = ((z > self.v("threshold")) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cnts:
            if cv2.contourArea(c) >= self.v("min_area"):
                x, y, bw, bh = cv2.boundingRect(c)
                peak = float(z[y:y + bh, x:x + bw].max())
                dets.append({"box": (x, y, bw, bh), "score": min(1.0, peak / (self.v("threshold") * 3)),
                             "class_id": 0, "label": "anomali"})
        heat = cv2.applyColorMap(np.clip(z / (self.v("threshold") * 2) * 255, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        base = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        heat = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
        score = float(np.percentile(z, 99.9))
        return {"heatmap": heat, "mask": mask, "detections": dets, "ok": not dets, "score": round(score, 2)}


@register
class DatasetRecorder(Node):
    title = "YOLO Veri Seti Kaydet"
    category = CAT
    description = ("Kareleri ve tespitleri YOLO formatında (images/ + labels/) kaydeder. Mevcut bir modelin "
                   "tespitleriyle otomatik etiketleme yapıp, Etiketleme Aracı'nda düzeltip yeni model "
                   "eğitebilirsiniz.")
    inputs = [("image", IMG), ("detections", DET), ("trigger", BOOL)]
    outputs = [("saved", NUM)]
    params = [Param("folder", "folder", "yolo_veri", "Klasör"),
              Param("classes", "str", "", "Sınıf listesi (virgülle; boşsa etiketlerden)"),
              Param("every_n", "int", 15, "Tetik yoksa her N karede", 1, 100000)]

    def setup(self):
        self.state.update(n=0, saved=0, prev=False)

    def process(self, inputs):
        img, trig = inputs["image"], inputs["trigger"]
        self.state["n"] += 1
        if trig is not None:
            fire = bool(trig) and not self.state["prev"]
            self.state["prev"] = bool(trig)
        else:
            fire = self.state["n"] % self.v("every_n") == 0
        if img is None or not fire:
            return {"saved": self.state["saved"]}
        root = pathlib.Path(self.v("folder"))
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        classes = [c.strip() for c in self.v("classes").split(",") if c.strip()]
        cls_file = root / "classes.txt"
        if not classes and cls_file.exists():
            classes = [c for c in cls_file.read_text(encoding="utf-8").splitlines() if c]
        h, w = img.shape[:2]
        lines = []
        for d in inputs["detections"] or []:
            name = d["label"].split(" #")[0]
            if name not in classes:
                classes.append(name)
            x, y, bw, bh = d["box"]
            lines.append(f"{classes.index(name)} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
        stem = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _imwrite(root / "images" / f"{stem}.jpg", img)
        (root / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        cls_file.write_text("\n".join(classes), encoding="utf-8")
        write_data_yaml(root, classes)
        self.state["saved"] += 1
        return {"saved": self.state["saved"]}


def write_data_yaml(root: pathlib.Path, classes: list[str], images: str = "images") -> pathlib.Path:
    """data.yaml in the layout Ultralytics expects (train and val share the folder for small sets)."""
    # every value JSON-quoted (valid YAML): class names like "no", "null" or "kusur: çizik" and folders like
    # "proje #2" would otherwise be misread
    q = lambda v: json.dumps(str(v), ensure_ascii=False)  # noqa: E731
    names = "\n".join(f"  {i}: {q(n)}" for i, n in enumerate(classes))
    path = root / "data.yaml"
    path.write_text(f"path: {q(root.resolve().as_posix())}\ntrain: {q(images)}\nval: {q(images)}\nnames:\n{names}\n",
                    encoding="utf-8")
    return path
