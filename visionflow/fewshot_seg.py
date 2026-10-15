"""Few-shot segmentation ("Az Örnekle Segmentasyon"): the user paints a few strokes per class on a handful of images,
a small per-pixel classifier is trained on the CPU in a second or two and then labels every pixel of new pictures.

Per-pixel features: colour (BGR, Lab, HSV saturation/value), Lab blurred at three scales, gradient magnitude and
Laplacian-of-Gaussian at two scales, local standard deviation (texture) at two window sizes and, optionally, the
normalised position. The classifier is a small two-layer neural network written in plain numpy (cv2.ml is missing
from OpenCV 5 and scikit-learn is not a dependency), so a saved model gives exactly the same result everywhere.

Labels: a uint8 "paint" image of the same size as the picture where 0 = not painted (ignored), 1 = class 0
(the background, "arka plan"), 2 = class 1, … (paint value = class index + 1).

A model is saved as one ``.vfseg`` file (a numpy .npz archive: weights + a JSON header with class names, colours and
feature settings)."""
from __future__ import annotations

import io
import json
import pathlib

import cv2
import numpy as np

FORMAT = "visionflow-fewshot-seg"
VERSION = 1
EXT = ".vfseg"
MAX_CLASSES = 16
DEFAULT_COLORS = [(40, 40, 40), (0, 200, 0), (0, 0, 230), (230, 120, 0), (0, 200, 230), (200, 0, 200),
                  (230, 230, 0), (0, 128, 255), (128, 0, 255), (255, 0, 128), (128, 255, 0), (90, 160, 160),
                  (160, 90, 160), (160, 160, 90), (255, 255, 255), (60, 60, 180)]


def default_color(i: int) -> tuple[int, int, int]:
    return DEFAULT_COLORS[i % len(DEFAULT_COLORS)]


# --- images --------------------------------------------------------------------------------------------------------

def to_bgr8(img, what: str = "Görüntü") -> np.ndarray:
    """Any picture (gray, BGR, BGRA, 16-bit, float 0..1 / 0..255, bool) → contiguous uint8 BGR."""
    if img is None:
        raise ValueError(f"{what} girişi bağlı değil")
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{what} girişine resim bağlanmalı (gelen: {type(img).__name__})")
    if img.size == 0 or img.ndim not in (2, 3) or min(img.shape[:2]) < 1:
        raise ValueError(f"{what} boş veya geçersiz boyutta: {img.shape}")
    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    elif img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype.kind == "f":
        f = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if f.size and float(f.max()) <= 1.0:
            f = f * 255.0
        img = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 2:
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 2:
        return cv2.cvtColor(np.ascontiguousarray(img[:, :, 0]), cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(img[:, :, :3])


def work_scale(shape, work_max: int) -> float:
    h, w = shape[:2]
    return min(1.0, float(work_max) / max(h, w)) if work_max > 0 else 1.0


def _resize(img, scale: float, interp=cv2.INTER_AREA):
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=interp)


# --- features ------------------------------------------------------------------------------------------------------

def feature_names(use_position: bool = False) -> list[str]:
    names = ["B", "G", "R", "L", "a", "b", "S", "V"]
    for s in (1, 3, 6):
        names += [f"L_s{s}", f"a_s{s}", f"b_s{s}"]
    names += ["grad_s1", "grad_s3", "log_s1", "log_s3", "std_3", "std_7"]
    if use_position:
        names += ["x", "y"]
    return names


def features(bgr: np.ndarray, use_position: bool = False) -> np.ndarray:
    """(h, w, F) float32 per-pixel features of an 8-bit BGR image (values roughly 0..1)."""
    h, w = bgr.shape[:2]
    f32 = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f32, cv2.COLOR_BGR2Lab)               # L 0..100, a/b about -128..127
    lab = lab * np.float32(1 / 100.0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    chans = [f32[..., 0], f32[..., 1], f32[..., 2], lab[..., 0], lab[..., 1], lab[..., 2], hsv[..., 1], hsv[..., 2]]
    for s in (1, 3, 6):
        bl = cv2.GaussianBlur(lab, (0, 0), s)
        chans += [bl[..., 0], bl[..., 1], bl[..., 2]]
    for s in (1, 3):
        g = cv2.GaussianBlur(gray, (0, 0), s)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        chans.append(np.sqrt(gx * gx + gy * gy) * np.float32(s * 2))
    for s in (1, 3):
        g = cv2.GaussianBlur(gray, (0, 0), s)
        chans.append(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)) * np.float32(s * s * 4))
    for k in (3, 7):
        m = cv2.blur(gray, (k, k))
        m2 = cv2.blur(gray * gray, (k, k))
        chans.append(np.sqrt(np.maximum(m2 - m * m, 0)) * np.float32(4))
    if use_position:
        xs = np.broadcast_to(np.linspace(0, 1, w, dtype=np.float32)[None, :], (h, w))
        ys = np.broadcast_to(np.linspace(0, 1, h, dtype=np.float32)[:, None], (h, w))
        chans += [xs, ys]
    return np.ascontiguousarray(np.stack(chans, axis=-1), dtype=np.float32)


# --- the classifier (numpy MLP) ------------------------------------------------------------------------------------

def _forward(x, w1, b1, w2, b2):
    hid = np.maximum(x @ w1 + b1, 0)
    return hid, hid @ w2 + b2


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _train_mlp(x, y, n_out, hidden=48, epochs=60, lr=0.01, decay=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n_in = x.shape[1]
    params = [rng.normal(0, np.sqrt(2.0 / n_in), (n_in, hidden)).astype(np.float32),
              np.zeros(hidden, np.float32),
              rng.normal(0, np.sqrt(1.0 / hidden), (hidden, n_out)).astype(np.float32),
              np.zeros(n_out, np.float32)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    n = len(x)
    batch = 256
    onehot = np.eye(n_out, dtype=np.float32)[y]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = x[idx], onehot[idx]
            hid, z = _forward(xb, *params)
            dz = (_softmax(z) - yb) / len(idx)
            gw2 = hid.T @ dz + decay * params[2]
            gb2 = dz.sum(0)
            dh = (dz @ params[2].T) * (hid > 0)
            gw1 = xb.T @ dh + decay * params[0]
            gb1 = dh.sum(0)
            t += 1
            for i, g in enumerate((gw1, gb1, gw2, gb2)):
                m[i] = b1_ * m[i] + (1 - b1_) * g
                v[i] = b2_ * v[i] + (1 - b2_) * g * g
                mh = m[i] / (1 - b1_ ** t)
                vh = v[i] / (1 - b2_ ** t)
                params[i] = (params[i] - lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
    return params


# --- model ---------------------------------------------------------------------------------------------------------

class SegModel:
    """A trained few-shot segmentation model."""

    def __init__(self, classes, colors=None, use_position=False, work_max=400):
        self.classes = [str(c) for c in classes]
        cols = list(colors or [])
        self.colors = [tuple(int(v) for v in (cols[i] if i < len(cols) else default_color(i)))
                       for i in range(len(self.classes))]
        self.use_position = bool(use_position)
        self.work_max = int(work_max)
        self.mean = self.std = None
        self.weights: list[np.ndarray] | None = None
        self.trained_classes: list[int] = []      # class index of each network output
        self.info: dict = {}

    # training ------------------------------------------------------------------------------------------------
    @classmethod
    def train(cls, images, labels, classes, colors=None, use_position=False, work_max=400,
              samples_per_class=5000, hidden=48, epochs=60, seed=0) -> "SegModel":
        """Train on painted pictures. ``labels[i]`` is the paint image of ``images[i]`` (0 = not painted,
        k + 1 = class k). Raises ValueError with a Turkish message for unusable input."""
        classes = [str(c).strip() for c in (classes or [])]
        if len(classes) < 2:
            raise ValueError("En az iki sınıf gerekli (ör. 'arka plan' ve 'nesne')")
        if len(classes) > MAX_CLASSES:
            raise ValueError(f"En çok {MAX_CLASSES} sınıf olabilir")
        if not images:
            raise ValueError("Eğitim için hiç resim yok")
        if labels is None or len(labels) != len(images):
            raise ValueError("Her resim için bir etiket (boyama) gerekli")
        model = cls(classes, colors, use_position, work_max)
        rng = np.random.default_rng(seed)
        per_class: dict[int, list[np.ndarray]] = {}
        painted_images = 0
        for img, lab in zip(images, labels):
            if lab is None:
                continue
            lab = np.asarray(lab)
            if lab.ndim == 3:
                lab = lab[..., 0]
            if not lab.any():
                continue
            bgr = to_bgr8(img, "Eğitim resmi")
            if lab.shape[:2] != bgr.shape[:2]:
                raise ValueError(f"Etiket boyutu resimle aynı olmalı: resim {bgr.shape[1]}x{bgr.shape[0]}, "
                                 f"etiket {lab.shape[1]}x{lab.shape[0]}")
            if int(lab.max()) > len(classes):
                raise ValueError(f"Etikette tanımsız sınıf var (en büyük değer {int(lab.max())}, "
                                 f"{len(classes)} sınıf tanımlı)")
            scale = work_scale(bgr.shape, model.work_max)
            small = _resize(bgr, scale)
            lab_s = _resize(lab.astype(np.uint8), scale, cv2.INTER_NEAREST)
            feats = features(small, model.use_position)
            painted_images += 1
            for v in np.unique(lab_s):
                if v == 0:
                    continue
                per_class.setdefault(int(v) - 1, []).append(feats[lab_s == v])
        if not per_class:
            raise ValueError("Hiç boyanmış piksel yok: önce resimlerin üzerine sınıfları fırçayla boyayın")
        if len(per_class) < 2:
            only = classes[next(iter(per_class))]
            raise ValueError(f"Yalnızca '{only}' sınıfı boyanmış: en az iki sınıfı (ör. arka plan ve nesne) boyayın")
        xs, ys = [], []
        present = sorted(per_class)
        for out_i, ci in enumerate(present):
            x = np.concatenate(per_class[ci])
            take = rng.choice(len(x), samples_per_class, replace=len(x) < samples_per_class)  # balanced classes
            xs.append(x[take])
            ys.append(np.full(samples_per_class, out_i, np.int64))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        model.mean = x.mean(0).astype(np.float32)
        model.std = (x.std(0) + 1e-3).astype(np.float32)
        xn = (x - model.mean) / model.std
        model.weights = _train_mlp(xn, y, len(present), hidden=hidden, epochs=epochs, seed=seed)
        model.trained_classes = present
        pred = model._predict_feats(xn, normalised=True)
        model.info = {"images": painted_images,
                      "pixels": {classes[ci]: int(sum(len(a) for a in per_class[ci])) for ci in present},
                      "train_accuracy": round(float((pred == np.asarray(present)[y]).mean()), 4)}
        return model

    # prediction ----------------------------------------------------------------------------------------------
    def _check(self):
        if self.weights is None:
            raise ValueError("Model eğitilmemiş")

    def _predict_feats(self, x, normalised=False, proba=False):
        if not normalised:
            x = (x - self.mean) / self.std
        out = []
        for s in range(0, len(x), 65536):
            _, z = _forward(x[s:s + 65536], *self.weights)
            out.append(_softmax(z) if proba else np.argmax(z, axis=1))
        res = np.concatenate(out) if out else np.zeros((0,), np.int64)
        if proba:
            return res
        return np.asarray(self.trained_classes, np.int64)[res]

    def predict_proba_small(self, img) -> tuple[np.ndarray, tuple[int, int]]:
        """(h', w', n_classes) class probabilities at the working resolution and the full (w, h)."""
        self._check()
        bgr = to_bgr8(img)
        h, w = bgr.shape[:2]
        small = _resize(bgr, work_scale(bgr.shape, self.work_max))
        feats = features(small, self.use_position)
        sh, sw = feats.shape[:2]
        p = self._predict_feats(feats.reshape(-1, feats.shape[-1]), proba=True)
        full = np.zeros((sh * sw, len(self.classes)), np.float32)
        full[:, self.trained_classes] = p
        return full.reshape(sh, sw, len(self.classes)), (w, h)

    def predict(self, img, smooth: int = 0, min_area: int = 0, fill_class: int = 0) -> np.ndarray:
        """Class index of every pixel (uint8, full resolution). ``smooth``: blur radius of the class probabilities
        in working pixels (0 = off). ``min_area``: regions smaller than this (full-resolution pixels) of a
        non-background class are given to ``fill_class`` (the background)."""
        prob, (w, h) = self.predict_proba_small(img)
        if smooth and smooth > 0:
            k = int(smooth) * 2 + 1
            prob = np.stack([cv2.blur(prob[..., i], (k, k)) for i in range(prob.shape[-1])], axis=-1)
        sh, sw = prob.shape[:2]
        if (sw, sh) != (w, h):
            if w * h * prob.shape[-1] <= 60_000_000:      # smooth edges: upsample probabilities, then pick
                lab = np.argmax(np.stack([cv2.resize(prob[..., i], (w, h), interpolation=cv2.INTER_LINEAR)
                                          for i in range(prob.shape[-1])], axis=-1), axis=-1).astype(np.uint8)
            else:
                lab = cv2.resize(np.argmax(prob, -1).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            lab = np.argmax(prob, axis=-1).astype(np.uint8)
        if min_area and min_area > 0:
            lab = remove_small(lab, int(min_area), fill_class)
        return lab

    # persistence ---------------------------------------------------------------------------------------------
    def header(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "classes": self.classes,
                "colors": [list(c) for c in self.colors], "use_position": self.use_position,
                "work_max": self.work_max, "trained_classes": self.trained_classes,
                "features": feature_names(self.use_position), "info": self.info}

    def save(self, path) -> str:
        self._check()
        path = pathlib.Path(path)
        if path.suffix.lower() != EXT:
            path = path.with_name(path.name + EXT)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, header=np.frombuffer(json.dumps(self.header(), ensure_ascii=False).encode(), np.uint8),
                            mean=self.mean, std=self.std, w1=self.weights[0], b1=self.weights[1],
                            w2=self.weights[2], b2=self.weights[3])
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)                     # a running block never reads a half-written model
        return str(path)

    @classmethod
    def load(cls, path) -> "SegModel":
        path = pathlib.Path(path)
        if not path.is_file():
            raise ValueError(f"Model dosyası yok: {path}")
        try:
            with np.load(path, allow_pickle=False) as z:
                head = json.loads(bytes(z["header"]).decode()) if "header" in z.files else {}
                if not isinstance(head, dict) or head.get("format") != FORMAT:
                    raise TypeError("biçim")
                arrays = {k: z[k].astype(np.float32) for k in ("mean", "std", "w1", "b1", "w2", "b2")}
        except TypeError:
            raise ValueError(f"Bu dosya bir Az Örnekle Segmentasyon modeli değil: {path.name}") from None
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Model dosyası okunamadı ({path.name}): {exc}") from exc
        if int(head.get("version", 0)) > VERSION:
            raise ValueError("Model daha yeni bir sürümle kaydedilmiş; programı güncelleyin")
        m = cls(head["classes"], head.get("colors"), head.get("use_position", False), head.get("work_max", 400))
        m.mean, m.std = arrays["mean"], arrays["std"]
        m.weights = [arrays["w1"], arrays["b1"], arrays["w2"], arrays["b2"]]
        m.trained_classes = [int(c) for c in head["trained_classes"]]
        m.info = head.get("info", {})
        if m.mean.shape[0] != len(feature_names(m.use_position)) or m.weights[2].shape[1] != len(m.trained_classes) \
                or not all(0 <= c < len(m.classes) for c in m.trained_classes) or len(m.classes) < 2:
            raise ValueError(f"Model dosyası bozuk: {path.name}")
        return m


# --- post-processing and drawing -----------------------------------------------------------------------------------

def remove_small(lab: np.ndarray, min_area: int, fill_class: int = 0) -> np.ndarray:
    """Regions of any class other than ``fill_class`` smaller than ``min_area`` pixels become ``fill_class``."""
    out = lab.copy()
    for c in np.unique(lab):
        if c == fill_class:
            continue
        n, cc, stats, _ = cv2.connectedComponentsWithStats((lab == c).astype(np.uint8), connectivity=8)
        small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
        if len(small):
            out[np.isin(cc, small)] = fill_class
    return out


def colorize(lab: np.ndarray, colors) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(colors[:256]):
        lut[i] = c
    return lut[lab]


def overlay(img, lab: np.ndarray, colors, alpha: float = 0.45, skip_class: int | None = 0) -> np.ndarray:
    """The picture with each class tinted in its colour (``skip_class`` — the background — left untinted)."""
    bgr = to_bgr8(img).copy()
    col = colorize(lab, colors)
    m = np.ones(lab.shape, bool) if skip_class is None else lab != skip_class
    if m.any():
        bgr[m] = (bgr[m].astype(np.float32) * (1 - alpha) + col[m].astype(np.float32) * alpha + 0.5).astype(np.uint8)
    return bgr


def class_areas(lab: np.ndarray, n_classes: int) -> list[float]:
    """Area of each class in percent of the picture."""
    counts = np.bincount(lab.ravel(), minlength=n_classes)[:n_classes]
    return [round(100.0 * float(c) / max(1, lab.size), 3) for c in counts]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


# --- GrabCut helper ------------------------------------------------------------------------------------------------

def grabcut_rect(img, rect, iters: int = 4) -> np.ndarray:
    """GrabCut from a rectangle (x, y, w, h): bool mask of the probable foreground."""
    bgr = to_bgr8(img)
    h, w = bgr.shape[:2]
    x, y, rw, rh = (int(v) for v in rect)
    x, y = max(0, x), max(0, y)
    rw, rh = min(rw, w - x), min(rh, h - y)
    if rw < 3 or rh < 3:
        raise ValueError("GrabCut için daha büyük bir dikdörtgen çizin")
    if rw >= w - 1 and rh >= h - 1:
        raise ValueError("Dikdörtgen resmin tamamını kaplamamalı (dışı arka plan sayılır)")
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x, y, rw, rh), bgd, fgd, int(iters), cv2.GC_INIT_WITH_RECT)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
