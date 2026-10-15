"""PatchCore: anomaly detection learned from good parts only, with deep (CNN) features.

Every small patch of every good image is described by a feature vector (from a pretrained ResNet, or built-in
colour/texture features when no network is wanted). A representative subset of those vectors is kept as the
"memory bank". A patch of a new image that is far from everything in the bank is something never seen on a good
part: a scratch, a stain, a missing piece. The alarm level is set by itself from the good images.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile

import cv2
import numpy as np

from .. import accel
from ..core import Node, Param, PortType, register
from .ai_more import _bgr
from .learning import _images_in, _imread

CAT = "3 · Yapay Zeka"
BACKBONES = ["Yerleşik (hızlı, internetsiz)", "ResNet18 (derin, daha hassas)", "Kendi ONNX modelim"]
RESNET_FILE = "resnet18_patchcore.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# --- feature extractors ---------------------------------------------------------------------------------------------

class BuiltinFeatures:
    """Colour, edge and texture descriptors at two scales, pooled over 8x8 pixel patches. No network needed."""
    name = "builtin"
    device = "CPU (yerleşik)"

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(_bgr(img), (self.size, self.size), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        chans = [lab[..., 0], lab[..., 1], lab[..., 2]]
        g = lab[..., 0]
        for k in (1, 2):
            gs = cv2.GaussianBlur(g, (0, 0), k)
            gx, gy = cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1)
            mean = cv2.blur(gs, (7 * k, 7 * k))
            sq = cv2.blur(gs * gs, (7 * k, 7 * k))
            chans += [gx, gy, np.sqrt(gx * gx + gy * gy), cv2.Laplacian(gs, cv2.CV_32F),
                      np.sqrt(np.maximum(sq - mean * mean, 0))]
        f = np.stack(chans, -1)
        f = np.concatenate([cv2.blur(np.ascontiguousarray(f[..., i:i + 4]), (9, 9)).reshape(f.shape[0], f.shape[1], -1)
                            for i in range(0, f.shape[-1], 4)], -1)        # each patch also sees its neighbours
        return np.ascontiguousarray(f[4::8, 4::8])


class OnnxFeatures:
    """Feature maps of a CNN in ONNX form: the first two outputs (e.g. ResNet layer2 and layer3), the second
    resized onto the first and stacked, then averaged over 3x3 neighbours as in the PatchCore paper."""

    def __init__(self, path: str, size: int, node=None):
        self.sess = accel.node_session(node, path) if node is not None else accel.session(path)
        self.device = accel.device_of(self.sess)
        self.input = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        fixed = len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int)
        self.hw = (shape[2], shape[3]) if fixed else (size, size)
        self.size = size
        st = pathlib.Path(path).stat()          # a retrained model with the same name and size is a new model
        self.name = f"onnx:{pathlib.Path(path).name}:{st.st_size}:{st.st_mtime_ns}"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = self.hw
        rgb = cv2.resize(_bgr(img), (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
        blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
        outs = [o for o in self.sess.run(None, {self.input: np.ascontiguousarray(blob)}) if o.ndim == 4][:2]
        if not outs:
            raise ValueError("ONNX modeli öznitelik haritası (1, C, h, w) vermiyor")
        maps = [o[0].transpose(1, 2, 0) for o in outs]
        fh, fw = maps[0].shape[:2]
        maps = [maps[0]] + [_resize(m, fw, fh) for m in maps[1:]]
        f = np.concatenate(maps, -1).astype(np.float32)
        return np.ascontiguousarray(_pool3(f))


def _resize(m: np.ndarray, w: int, h: int) -> np.ndarray:
    # OpenCV handles only a few channels per image (and fewer in OpenCV 5): resize four at a time
    parts = [cv2.resize(np.ascontiguousarray(m[..., i:i + 4]), (w, h), interpolation=cv2.INTER_LINEAR)
             for i in range(0, m.shape[-1], 4)]
    return np.concatenate([p.reshape(h, w, -1) for p in parts], -1)


def _pool3(f: np.ndarray) -> np.ndarray:
    """3x3 average of every channel (edges repeat), in numpy: works for any number of channels."""
    p = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = f.shape[:2]
    return sum(p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9.0


def models_dir() -> pathlib.Path:
    from ..ai.config import home
    return home() / "modeller"


def ensure_resnet() -> str:
    """The pretrained ResNet18 feature extractor, made once from torchvision (ImageNet weights) and kept as ONNX,
    so the flow itself only needs onnxruntime."""
    path = models_dir() / RESNET_FILE
    if path.is_file():
        return str(path)
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError("Derin öznitelikler bir kez hazırlanmalı: 'pip install torch torchvision' kurun (sonra "
                           "gerekmez) veya 'Yerleşik' seçin") from exc
    print("ResNet18 öznitelik modeli hazırlanıyor (bir kez, ImageNet ağırlıkları indirilir)…")
    try:
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    except Exception as exc:  # noqa: BLE001  (no internet, proxy, ...)
        raise RuntimeError(f"ResNet18 ağırlıkları indirilemedi ({exc}). İnternet bağlantısını kontrol edin veya "
                           "'Yerleşik' seçin") from exc

    class Features(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            n = self.net
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            l2 = n.layer2(n.layer1(x))
            return l2, n.layer3(l2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".onnx", dir=path.parent)   # two programs exporting at once never collide
    os.close(fd)
    kw = dict(input_names=["image"], output_names=["layer2", "layer3"], opset_version=17,
              dynamic_axes={"image": {2: "h", 3: "w"}})
    try:
        with torch.no_grad():
            try:
                torch.onnx.export(Features(m), torch.zeros(1, 3, 224, 224), tmp, dynamo=False, **kw)
            except TypeError:                             # older torch without the dynamo switch
                torch.onnx.export(Features(m), torch.zeros(1, 3, 224, 224), tmp, **kw)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(path)


def make_features(backbone: str, onnx_path: str, size: int, node=None):
    """node: the block whose "Cihaz" setting (and flow) chooses the inference device."""
    if backbone == BACKBONES[0]:
        return BuiltinFeatures(size)
    if backbone == BACKBONES[1]:
        return OnnxFeatures(ensure_resnet(), size, node)
    if not onnx_path:
        raise ValueError("ONNX öznitelik modeli seçilmedi")
    return OnnxFeatures(onnx_path, size, node)


# --- the model -----------------------------------------------------------------------------------------------------

def nearest(x: np.ndarray, bank: np.ndarray, exclude: np.ndarray | None = None, bank_src=None) -> np.ndarray:
    """Distance from each row of x to its nearest row of bank (rows of the same source image may be excluded)."""
    bb = (bank * bank).sum(1)
    out = np.empty(len(x), np.float32)
    for i in range(0, len(x), 1024):
        a = x[i:i + 1024]
        d = (a * a).sum(1)[:, None] + bb[None] - 2 * a @ bank.T
        if exclude is not None:
            d[exclude[i:i + 1024][:, None] == bank_src[None]] = np.inf
        out[i:i + 1024] = np.sqrt(np.maximum(d.min(1), 0))
    return out


def coreset(x: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Greedy k-center selection (on a random projection for speed): k rows that cover all the others."""
    if k >= len(x):
        return np.arange(len(x))
    rng = np.random.default_rng(seed)
    dims = min(32, x.shape[1])
    proj = (x @ (rng.standard_normal((x.shape[1], dims)).astype(np.float32) / np.sqrt(dims))
            if x.shape[1] > dims else x).astype(np.float32)
    sq = (proj * proj).sum(1)
    chosen = np.empty(k, np.int64)
    chosen[0] = int(rng.integers(len(x)))
    dist = np.maximum(sq + sq[chosen[0]] - 2 * proj @ proj[chosen[0]], 0)
    new = np.empty_like(dist)
    for j in range(1, k):
        i = int(dist.argmax())
        chosen[j] = i
        np.dot(proj, proj[i], out=new)                    # |a-b|^2 = |a|^2 + |b|^2 - 2ab, all in place
        new *= -2
        new += sq
        new += sq[i]
        np.minimum(dist, new, out=dist)
    return chosen


class PatchCoreModel:
    def __init__(self, features, ratio: float = 0.1, max_bank: int = 5000, max_patches: int = 40000):
        self.features, self.ratio, self.max_bank, self.max_patches = features, ratio, max_bank, max_patches

    def _patches(self, img):
        f = self.features(img)
        self.grid = f.shape[:2]
        return f.reshape(-1, f.shape[-1])

    def fit(self, images: list[np.ndarray]) -> "PatchCoreModel":
        if len(images) < 2:
            raise ValueError("En az 2 sağlam örnek resmi gerekli (10–50 önerilir)")
        rng = np.random.default_rng(0)
        per_image = max(1, self.max_patches // len(images))
        feats, xs, srcs = [], [], []
        for i, im in enumerate(images):
            f = self._patches(im).astype(np.float32, copy=False)
            feats.append(f)
            # memory stays bounded however many and however big the images are: a sample of each one
            pick = np.sort(rng.choice(len(f), per_image, replace=False)) if len(f) > per_image else slice(None)
            xs.append(f[pick])
            srcs.append(np.full(len(xs[-1]), i))
        x, src = np.concatenate(xs), np.concatenate(srcs)
        del xs
        self.mu, sd = x.mean(0), x.std(0)
        self.sd = sd + max(float(np.median(sd)), 1e-6) * 0.1
        x -= self.mu
        x /= self.sd
        k = int(min(self.max_bank, max(64, len(x) * self.ratio)))
        idx = coreset(x, k)
        self.bank, self.bank_src = x[idx], src[idx]
        # alarm level: how far each good image is from the OTHER good images (its own patches excluded)
        own = []
        for i, f in enumerate(feats):
            z = (f - self.mu) / self.sd
            m = self._smooth(nearest(z, self.bank, np.full(len(z), i), self.bank_src))
            own.append(float(m.max()))
        self.good_scores = np.array(own, np.float32)
        self.threshold = float(self.good_scores.max())
        self.n = len(images)
        return self

    def _smooth(self, d: np.ndarray) -> np.ndarray:
        m = d.reshape(self.grid)
        return cv2.GaussianBlur(m, (0, 0), max(0.5, self.grid[0] / 64))

    def score_map(self, img: np.ndarray) -> np.ndarray:
        """Anomaly map at feature-grid size, in units of the alarm level (1.0 = the worst good image)."""
        z = (self._patches(img).astype(np.float32) - self.mu) / self.sd
        return self._smooth(nearest(z, self.bank)) / max(self.threshold, 1e-6)

    def save(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, mu=self.mu, sd=self.sd, bank=self.bank, bank_src=self.bank_src,
                            good=self.good_scores, threshold=self.threshold, grid=np.array(self.grid), n=self.n)
        tmp.replace(path)

    def load(self, path: pathlib.Path) -> "PatchCoreModel":
        with np.load(path) as z:
            self.mu, self.sd, self.bank, self.bank_src = z["mu"], z["sd"], z["bank"], z["bank_src"]
            self.good_scores, self.threshold = z["good"], float(z["threshold"])
            self.grid, self.n = tuple(int(v) for v in z["grid"]), int(z["n"])
        return self


def _cache_key(files: list[pathlib.Path], *parts) -> str:
    h = hashlib.sha1(json.dumps([str(p) for p in parts]).encode())
    for f in files:
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


@register
class PatchCore(Node):
    setup_params = ("folder", "backbone", "onnx", "size", "coreset", "device")
    title = "Derin Anomali (PatchCore)"
    category = CAT
    description = ("Sadece SAĞLAM ürün resimleriyle öğrenir (10–50 resim yeter), hiç görmediği her şeyi (çizik, "
                   "leke, çatlak, eksik parça) işaretler. Endüstride en iyi sonuç veren yöntemlerden PatchCore. "
                   "Alarm sınırını sağlam resimlerden kendisi belirler: skor 1.0'ı geçerse hatalı. ResNet18 "
                   "seçeneği daha hassastır (bir kez 'pip install torch torchvision' gerekir). Eğitim "
                   "klasörde saklanır, resimler değişmezse tekrar eğitilmez.")
    inputs = [("image", PortType.IMAGE)]
    outputs = [("heatmap", PortType.IMAGE), ("mask", PortType.IMAGE), ("detections", PortType.DETECTIONS),
               ("ok", PortType.BOOL), ("score", PortType.NUMBER), ("device", PortType.TEXT)]
    params = [Param("folder", "folder", "veri/saglam", "Sağlam örnek klasörü"),
              Param("backbone", "choice", BACKBONES[0], "Öznitelikler", choices=BACKBONES),
              Param("onnx", "path", "", "Kendi ONNX modeliniz"),
              Param("sensitivity", "float", 1.0, "Sınır çarpanı (düşük = hassas)", 0.3, 5.0, 0.05),
              Param("min_area", "int", 30, "Min hata alanı (px)", 1, 1_000_000),
              Param("size", "int", 256, "Çalışma çözünürlüğü", 64, 1024, 32),
              Param("coreset", "float", 10.0, "Hafıza oranı (%)", 1.0, 100.0, 1.0),
              Param("device", "choice", accel.GLOBAL, "Cihaz (ONNX için)", choices=accel.BLOCK_CHOICES)]

    def setup(self):
        from ..core.node import resolve_path
        folder = pathlib.Path(resolve_path(self.v("folder")))
        files = _images_in(folder)
        if len(files) < 2:
            raise ValueError(f"'{folder}' klasöründe en az 2 sağlam resim olmalı (şu an {len(files)})")
        feats = make_features(self.v("backbone"), self.v("onnx"), int(self.v("size")), self)
        model = PatchCoreModel(feats, ratio=float(self.v("coreset")) / 100)
        kind = hashlib.sha1(feats.name.encode()).hexdigest()[:6]      # one saved training per feature type
        cache = folder / "_patchcore" / f"{kind}_{_cache_key(files, feats.name, self.v('size'), self.v('coreset'))}.npz"
        try:
            model.load(cache)
            print(f"[{self.title}] kayıtlı eğitim kullanıldı ({model.n} sağlam örnek)")
        except Exception:  # noqa: BLE001  (missing, or cut off by a crash: train again)
            images = [im for im in (_imread(p) for p in files) if im is not None]
            print(f"[{self.title}] {len(images)} sağlam örnekle eğitiliyor…")
            model.fit(images)
            try:
                model.save(cache)
                # keep a few recent trainings (several blocks may use this folder with other settings)
                olds = sorted((f for f in cache.parent.glob(f"{kind}_*.npz") if not f.name.endswith(".tmp.npz")
                               and f != cache), key=lambda f: f.stat().st_mtime, reverse=True)
                for old in olds[3:]:
                    old.unlink()
            except OSError as exc:
                print(f"[{self.title}] eğitim kaydedilemedi: {exc}")
            print(f"[{self.title}] eğitildi: hafızada {len(model.bank)} parça, alarm sınırı sağlam resimlerden "
                  "belirlendi (skor 1.0)")
        self.state["model"] = model
        self.state["device"] = getattr(feats, "device", "CPU")

    def process(self, inputs):
        img = _bgr(inputs["image"])
        h, w = img.shape[:2]
        limit = float(self.v("sensitivity"))
        z = cv2.resize(self.state["model"].score_map(img), (w, h), interpolation=cv2.INTER_LINEAR)
        score = float(z.max())
        mask = ((z > limit) * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cnts:
            if cv2.contourArea(c) >= self.v("min_area"):
                x, y, bw, bh = cv2.boundingRect(c)
                peak = float(z[y:y + bh, x:x + bw].max())
                dets.append({"box": (x, y, bw, bh), "score": round(min(1.0, peak / (2 * limit)), 3), "class_id": 0,
                             "label": "anomali"})
        heat = cv2.applyColorMap(np.clip(z / (2 * limit) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
        return {"heatmap": heat, "mask": mask, "detections": dets, "ok": not dets, "score": round(score, 3),
                "device": self.state.get("device", "")}
