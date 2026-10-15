"""3D / depth vision blocks: depth cameras (Intel RealSense, Orbbec, files), a synthetic depth test scene, and
processing / measuring on depth maps (colourize, filter, ground plane, height, volume, 3D objects, flatness,
profile, point cloud saving, 3D view). The maths lives in visionflow/depth3d.py."""
from __future__ import annotations

import datetime as dt
import math
import os
import pathlib
import time

import cv2
import numpy as np

from .. import depth3d as D
from ..core import Node, Param, PortType, StopFlow, register
from ._text import put_text
from .outputs import _rising
from .vision_more import _nums, parse_roi

CAT = "2c · 3B ve Derinlik"
IMG, NUM, BOOL, TEXT, DET, ANY = (PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.TEXT, PortType.DETECTIONS,
                                  PortType.ANY)
VIEWER = ("🧊 3B Görüntüle…", "depth_viewer.DepthViewerDialog")
DEPTH_HELP = "Derinlik haritası: mm cinsinden, geçersiz pikseller NaN."


def _depth(inputs, name: str = "depth", what: str = "Derinlik") -> np.ndarray:
    return D.as_depth(inputs.get(name), what)


def _intr(inputs, depth) -> dict:
    return D.intrinsics(inputs.get("intrinsics"), depth.shape)


def _region(node, inputs, shape, mask_name: str = "mask") -> np.ndarray:
    """Bool mask of the block's 'roi' setting and its optional mask input (non-zero = use)."""
    x, y, w, h = parse_roi(node.v("roi"), shape)
    sel = np.zeros(shape[:2], bool)
    sel[y:y + h, x:x + w] = True
    m = inputs.get(mask_name)
    if m is not None:
        if not isinstance(m, np.ndarray) or m.ndim not in (2, 3):
            raise ValueError("Maske girişine resim bağlanmalı")
        if m.ndim == 3:
            m = m.max(axis=2)
        if m.shape[:2] != shape[:2]:
            m = cv2.resize(m.astype(np.uint8) if m.dtype == np.bool_ else m, (shape[1], shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        sel &= np.nan_to_num(m.astype(np.float32)) > 0
    return sel


def _rect_of(node, shape):
    return parse_roi(node.v("roi"), shape)


def _draw_roi(img, node):
    if str(node.v("roi") or "").strip():
        x, y, w, h = _rect_of(node, img.shape)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 220, 255), 1)


def _stat(values: np.ndarray, how: str, pct: float) -> float:
    if not values.size:
        return 0.0
    if how == "en yüksek":
        return float(values.max())
    if how == "ortalama":
        return float(values.mean())
    if how == "medyan":
        return float(np.median(values))
    return float(np.percentile(values, pct))


# ------------------------------------------------------------------------------------------------------- sources

@register
class DepthCamera(Node):
    title = "Derinlik Kamerası"
    category = "1 · Giriş"
    description = ("3B / derinlik kamerası: Intel RealSense (pyrealsense2), Orbbec (pyorbbecsdk) veya dosyadan "
                   "(16 bit PNG mm, .npy, .ply nokta bulutu; klasör seçilirse sırayla). Çıkışlar: derinlik (mm, "
                   "geçersiz = NaN), renkli resim (derinliğe hizalı; renk kamerası yoksa renklendirilmiş derinlik) ve "
                   "kamera iç parametreleri (fx, fy, cx, cy). Bloktaki düğmeyle nokta bulutu 3B görüntülenir.")
    outputs = [("depth", IMG), ("image", IMG), ("intrinsics", ANY), ("fps", NUM)]
    params = [Param("driver", "choice", D.DRIVERS[0], "Kaynak", choices=D.DRIVERS),
              Param("serial", "str", "", "Seri no (boş = ilk kamera)"),
              Param("width", "int", 640, "Genişlik", 160, 4096),
              Param("height", "int", 480, "Yükseklik", 120, 3072),
              Param("fps", "int", 30, "Kare hızı", 1, 300),
              Param("align", "bool", True, "Derinliği renkli resme hizala"),
              Param("path", "path", "", "Dosya veya klasör (Dosyadan)"),
              Param("unit", "choice", "mm", "Dosya birimi", choices=["mm", "m", "0.1 mm"]),
              Param("hfov", "float", 60.0, "Yatay görüş açısı ° (dosyada iç parametre yoksa)", 10, 170, 1),
              Param("loop", "bool", True, "Klasör bitince başa sar"),
              Param("timeout", "int", 1000, "Kare bekleme (ms)", 50, 60000)]
    setup_params = ("driver", "serial", "width", "height", "fps", "align", "path", "unit", "hfov")
    dialog = VIEWER
    RETRY_SECONDS = 2.0

    @property
    def is_source(self):
        if self.values.get("driver") != D.DRIVERS[2]:
            return True
        return os.path.isdir(self.v("path") or "")

    def setup(self):
        self.state.update(dev=None, files=None, i=0, t=time.perf_counter(), fps=0.0, retry_at=0.0, misses=0,
                          cache=None)
        if self.v("driver") == D.DRIVERS[2]:
            path = str(self.v("path") or "").strip()
            if not path:
                raise ValueError("Derinlik dosyası / klasörü seçilmedi")
            if os.path.isdir(path):
                files = sorted(p for p in pathlib.Path(path).iterdir()
                               if p.suffix.lower() in D.DEPTH_EXT and not p.stem.endswith(D.COLOR_SUFFIX))
                if not files:
                    raise ValueError(f"Klasörde derinlik dosyası yok (.png, .npy, .ply, .tif): {path}")
                self.state["files"] = files
            elif not os.path.isfile(path):
                raise ValueError(f"Dosya bulunamadı: {path}")
        else:
            self._connect()

    def _connect(self):
        self.state["dev"] = D.SharedDevice.acquire(self.v("driver"), self.v("serial"), self.v("width"),
                                                   self.v("height"), self.v("fps"), self.v("align"))
        print(f"Derinlik kamerası bağlandı: {self.state['dev'].name}")

    def _load(self, path: str):
        return D.load_depth_file(path, self.v("unit"), float(self.v("hfov")), (self.v("width"), self.v("height")))

    def _from_file(self):
        st = self.state
        if st["files"] is not None:
            if st["i"] >= len(st["files"]):
                if not self.v("loop"):
                    raise StopFlow()
                st["i"] = 0
            path = str(st["files"][st["i"]])
            st["i"] += 1
            return self._load(path)
        path = str(self.v("path")).strip()
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            raise ValueError(f"Dosya bulunamadı: {path}") from None
        if st["cache"] is None or st["cache"][0] != key:
            st["cache"] = (key, self._load(path))
        depth, color, intr = st["cache"][1]
        return depth.copy(), None if color is None else color.copy(), D.Intrinsics(intr)

    def _from_device(self):
        st = self.state
        if st["dev"] is None:                                # unplugged: reconnect every few seconds
            if time.monotonic() < st["retry_at"]:
                raise RuntimeError("Derinlik kamerası bağlantısı yok, tekrar deneniyor")
            st["retry_at"] = time.monotonic() + self.RETRY_SECONDS
            self._connect()
        try:
            got = st["dev"].grab(int(self.v("timeout")))
        except Exception:
            self._drop()
            raise
        if got is None:
            st["misses"] += 1
            if st["misses"] >= 3:
                self._drop()
            raise RuntimeError(f"Derinlik kamerasından {self.v('timeout')} ms içinde kare gelmedi")
        st["misses"] = 0
        return got

    def process(self, inputs):
        depth, color, intr = self._from_file() if self.v("driver") == D.DRIVERS[2] else self._from_device()
        if color is None:
            color, _, _ = D.colorize(depth, reverse=True)
        st = self.state
        now = time.perf_counter()
        dt_, st["t"] = now - st["t"], now
        st["fps"] = 0.9 * st["fps"] + 0.1 * (1.0 / dt_ if dt_ > 0 else 0)
        D.publish(self.id, depth, color, intr)
        return {"depth": depth, "image": color, "intrinsics": intr, "fps": round(st["fps"], 1)}

    def _drop(self):
        dev, self.state["dev"] = self.state.get("dev"), None
        self.state["retry_at"] = 0.0
        self.state["misses"] = 0
        if dev is not None:
            dev.release()

    def teardown(self):
        self._drop()


@register
class DepthTestScene(Node):
    title = "Derinlik Test Sahnesi"
    category = "1 · Giriş"
    description = ("Kamera olmadan denemek için yapay derinlik görüntüsü: zemin düzlemi üzerinde kutular, silindir "
                   "ve delik; gürültü, kayıp piksel ve eğik zemin eklenebilir. 'Her karede yeni parça' açıkken kutu "
                   "yükseklikleri her karede ± değişim kadar oynar (iyi / hatalı parça denemesi). 'truth' çıkışı "
                   "gerçek ölçüleri (yükseklik, hacim) verir.")
    outputs = [("depth", IMG), ("image", IMG), ("intrinsics", ANY), ("truth", ANY)]
    params = [Param("scene", "choice", D.SCENES[0], "Sahne", choices=D.SCENES),
              Param("width", "int", 640, "Genişlik", 64, 2048),
              Param("height", "int", 480, "Yükseklik", 48, 1536),
              Param("distance", "float", 800.0, "Kamera-zemin mesafesi (mm)", 100, 10000, 10),
              Param("tilt_x", "float", 0.0, "Zemin eğikliği X (°)", -45, 45, 0.5),
              Param("tilt_y", "float", 0.0, "Zemin eğikliği Y (°)", -45, 45, 0.5),
              Param("noise", "float", 0.3, "Gürültü σ (mm)", 0, 50, 0.1),
              Param("dropout", "float", 0.01, "Kayıp piksel oranı", 0, 0.9, 0.01),
              Param("vary", "float", 0.0, "Yükseklik değişimi ± (mm)", 0, 100, 0.5),
              Param("animate", "bool", False, "Her karede yeni parça"),
              Param("seed", "int", 0, "Rastgele tohum", 0, 1_000_000)]
    dialog = VIEWER

    @property
    def is_source(self):
        return bool(self.values.get("animate"))

    def setup(self):
        self.state.update(n=0, cache=None)

    def process(self, inputs):
        st = self.state
        key = tuple(self.v(p.name) for p in self.params)
        k = st["n"] if self.v("animate") else 0
        st["n"] += 1
        if st["cache"] is not None and st["cache"][0] == key and not self.v("animate"):
            sc = st["cache"][1]
        else:
            objs = D.scene_objects(self.v("scene"))
            vary = float(self.v("vary"))
            if vary > 0:
                rng = np.random.default_rng(int(self.v("seed")) * 7919 + k)
                for o in objs:
                    if o["kind"] != "hole":
                        o["h"] = max(1.0, o["h"] + float(rng.uniform(-vary, vary)))
            sc = D.synthetic_scene(self.v("width"), self.v("height"), self.v("distance"), self.v("tilt_x"),
                                   self.v("tilt_y"), self.v("noise"), self.v("dropout"), objs,
                                   seed=int(self.v("seed")) + k)
            st["cache"] = (key, sc)
        truth = [dict(o, volume=round(D.object_volume(o) / 1000.0, 3)) for o in sc["objects"]]
        D.publish(self.id, sc["depth"], sc["image"], sc["intrinsics"], sc["plane"])
        return {"depth": sc["depth"].copy(), "image": sc["image"].copy(), "intrinsics": D.Intrinsics(sc["intrinsics"]),
                "truth": truth}


# ------------------------------------------------------------------------------------------------ processing

@register
class ColorizeDepth(Node):
    title = "Derinliği Renklendir"
    category = CAT
    description = ("Derinlik veya yükseklik haritasını izlemek için renkli resme çevirir. Aralık 0-0 ise otomatik "
                   "(%2-%98). Geçersiz pikseller siyah. İsterseniz renk ölçeği ve değerler resmin altına yazılır.")
    inputs = [("depth", IMG)]
    outputs = [("image", IMG), ("min", NUM), ("max", NUM)]
    params = [Param("vmin", "float", 0.0, "En küçük (mm, 0-0 = otomatik)", -100000, 100000, 1),
              Param("vmax", "float", 0.0, "En büyük (mm)", -100000, 100000, 1),
              Param("cmap", "choice", "Turbo", "Renk ölçeği", choices=list(D.COLORMAPS)),
              Param("near_warm", "bool", True, "Yakın = sıcak renk (derinlik için)"),
              Param("legend", "bool", True, "Ölçek çubuğu göster")]

    def process(self, inputs):
        d = inputs.get("depth")
        if d is None:
            raise ValueError("Derinlik girişi bağlı değil")
        if not isinstance(d, np.ndarray) or d.ndim not in (2, 3) or (d.ndim == 3 and d.shape[2] != 1):
            raise ValueError("Derinlik girişine tek kanallı derinlik / yükseklik haritası bağlanmalı")
        v = d.reshape(d.shape[0], d.shape[1]).astype(np.float32)
        if d.dtype.kind in "ui":
            v[v <= 0] = np.nan                       # raw 16-bit depth: 0 = no measurement
        lo, hi = float(self.v("vmin")), float(self.v("vmax"))
        auto = lo == 0 and hi == 0
        img, lo, hi = D.colorize(v, None if auto else min(lo, hi), None if auto else max(lo, hi), self.v("cmap"),
                                 reverse=self.v("near_warm"))
        if self.v("legend") and img.shape[0] >= 60 and img.shape[1] >= 120:
            H, W = img.shape[:2]
            bar_w = min(W - 20, 256)
            ramp = np.linspace(0, 1, bar_w, dtype=np.float32)[None, :].repeat(10, 0)
            bar, _, _ = D.colorize(ramp * (hi - lo) + lo, lo, hi, self.v("cmap"), reverse=self.v("near_warm"))
            img[H - 18:H - 8, 10:10 + bar_w] = bar
            put_text(img, f"{lo:.0f}", (10, H - 22), 0.4, (255, 255, 255), 1)
            put_text(img, f"{hi:.0f} mm", (max(10, 10 + bar_w - 60), H - 22), 0.4, (255, 255, 255), 1)
        return {"image": img, "min": round(lo, 3), "max": round(hi, 3)}


def _nan_median(depth: np.ndarray, k: int) -> np.ndarray:
    """Median filter that ignores invalid pixels (0.1 mm steps through 16-bit OpenCV median; a pixel whose
    neighbourhood is mostly invalid becomes invalid)."""
    ok = np.isfinite(depth)
    if not ok.any():
        return depth.copy()
    top = float(np.nanmax(depth))
    scale = 10.0 if top < 6500 else 65000.0 / top if top > 65000 else 1.0
    u16 = np.where(ok, np.clip(np.round(depth * scale), 1, 65535), 0).astype(np.uint16)
    med = cv2.medianBlur(u16, k).astype(np.float32) / scale
    med[med <= 0] = np.nan
    return med


@register
class DepthFilter(Node):
    title = "Derinlik Filtresi"
    category = CAT
    description = ("Derinlik haritasını temizler: mesafe aralığı dışını geçersiz yapar (arka plan / masa altı), "
                   "medyan veya kenar koruyan (bilateral) yumuşatma, küçük delikleri (kayıp piksel) komşulardan "
                   "doldurur. " + DEPTH_HELP)
    inputs = [("depth", IMG)]
    outputs = [("depth", IMG), ("valid", NUM)]
    params = [Param("near", "float", 0.0, "En yakın (mm, 0 = sınırsız)", 0, 100000, 10),
              Param("far", "float", 0.0, "En uzak (mm, 0 = sınırsız)", 0, 100000, 10),
              Param("smooth", "choice", "medyan 5", "Yumuşatma",
                    choices=["yok", "medyan 3", "medyan 5", "bilateral"]),
              Param("sigma", "float", 15.0, "Bilateral: derinlik farkı (mm)", 0.1, 1000, 1),
              Param("fill", "int", 50, "Doldurulacak en büyük delik (piksel, 0 = kapalı)", 0, 100000)]

    def process(self, inputs):
        d = _depth(inputs)
        near, far = float(self.v("near")), float(self.v("far"))
        if near > 0:
            d[d < near] = np.nan
        if far > 0:
            d[d > far] = np.nan
        how = self.v("smooth")
        if how.startswith("medyan"):
            d = _nan_median(d, int(how[-1]))
        elif how == "bilateral":
            ok = np.isfinite(d)
            if ok.any():
                fill = float(np.nanmedian(d))
                sm = cv2.bilateralFilter(np.where(ok, d, fill).astype(np.float32), 5, float(self.v("sigma")), 3)
                d = np.where(ok, sm, np.nan).astype(np.float32)
        d = D.fill_holes(d, int(self.v("fill")))
        valid = 100.0 * float(np.isfinite(d).mean())
        return {"depth": d.astype(np.float32), "valid": round(valid, 2)}


# ------------------------------------------------------------------------------------------------ measurement

@register
class FindGroundPlane(Node):
    roi = ("rect", "roi")
    title = "Zemin Düzlemi Bul"
    category = CAT
    description = ("Bölgedeki en büyük düzlemi (bant, masa, palet zemini) RANSAC ile bulur. Yükseklik haritası "
                   "çıkışı her pikselin bu düzlemin ne kadar üstünde olduğunu verir (mm; Yükseklik Ölç, Hacim Ölç, "
                   "3B Nesne Bul için). Düzlem 'ilk karede sabitle' ile bir kez öğrenilir; zemin parçalarla "
                   "kapansa da kayma olmaz. Bölgeyi zeminin boş göründüğü yere çizmek en sağlamıdır.")
    inputs = [("depth", IMG), ("intrinsics", ANY), ("mask", IMG)]
    outputs = [("height", IMG), ("plane", ANY), ("tilt", NUM), ("inliers", NUM), ("rms", NUM), ("image", IMG)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("threshold", "float", 3.0, "Düzleme uzaklık eşiği (mm)", 0.1, 500, 0.5),
              Param("iterations", "int", 200, "Deneme sayısı (RANSAC)", 10, 10000),
              Param("mode", "choice", "her karede", "Düzlem", choices=["her karede", "ilk karede sabitle"])]

    def setup(self):
        self.state["plane"] = None

    def process(self, inputs):
        depth = _depth(inputs)
        intr = _intr(inputs, depth)
        plane = self.state.get("plane") if self.v("mode") == "ilk karede sabitle" else None
        if plane is None:
            sel = _region(self, inputs, depth.shape)
            pts, _ = D.depth_to_points(depth, intr, sel)
            if len(pts) < 50:
                raise ValueError(f"Bölgede düzlem için yeterli geçerli derinlik yok ({len(pts)} nokta)")
            n, d, inl = D.ransac_plane(pts, float(self.v("threshold")), int(self.v("iterations")))
            res = pts[inl].astype(np.float64) @ n + d
            plane = D.plane_dict(n, d, inliers=round(100.0 * float(inl.mean()), 2),
                                 rms=round(float(np.sqrt((res ** 2).mean())) if res.size else 0.0, 4))
            if self.v("mode") == "ilk karede sabitle":
                self.state["plane"] = plane
        h = D.height_map(depth, intr, plane)
        img, _, _ = D.colorize(h, cmap="Turbo")
        _draw_roi(img, self)
        put_text(img, f"eğim {plane['tilt']:.1f}°", (8, 22), 0.6, (255, 255, 255), 2)
        return {"height": h, "plane": plane, "tilt": round(plane["tilt"], 3), "inliers": plane.get("inliers", 0.0),
                "rms": plane.get("rms", 0.0), "image": img}


STATS = ["yüzdelik", "en yüksek", "ortalama", "medyan"]


@register
class MeasureHeight(Node):
    roi = ("rect", "roi")
    title = "Yükseklik Ölç"
    category = CAT
    description = ("Yükseklik haritasında (Zemin Düzlemi Bul çıkışı) bölgedeki cismin zeminden yüksekliğini mm "
                   "olarak ölçer. 'yüzdelik' (varsayılan %95) tek tük gürültülü pikselleri yok sayan sağlam bir "
                   "en yüksek değerdir. Eşikten alçak pikseller (zemin) hesaba girmez.")
    inputs = [("height", IMG), ("mask", IMG)]
    outputs = [("height", NUM), ("max", NUM), ("mean", NUM), ("pixels", NUM), ("found", BOOL)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("min_height", "float", 3.0, "Zeminden en az (mm)", -10000, 10000, 0.5),
              Param("stat", "choice", STATS[0], "Ölçü", choices=STATS),
              Param("percentile", "float", 95.0, "Yüzdelik (%)", 0, 100, 1),
              Param("min_pixels", "int", 20, "En az piksel", 1, 10_000_000)]

    def process(self, inputs):
        h = _height_input(inputs)
        sel = _region(self, inputs, h.shape) & np.isfinite(h) & (h > float(self.v("min_height")))
        vals = h[sel].astype(np.float64)
        if vals.size < int(self.v("min_pixels")):
            return {"height": 0.0, "max": 0.0, "mean": 0.0, "pixels": int(vals.size), "found": False}
        return {"height": round(_stat(vals, self.v("stat"), float(self.v("percentile"))), 3),
                "max": round(float(vals.max()), 3), "mean": round(float(vals.mean()), 3), "pixels": int(vals.size),
                "found": True}


def _height_input(inputs) -> np.ndarray:
    """A height map: float, one channel, may be negative (below the plane); NaN = unknown."""
    h = inputs.get("height")
    if h is None:
        raise ValueError("Yükseklik girişi bağlı değil: Zemin Düzlemi Bul bloğunun 'height' çıkışını bağlayın")
    if not isinstance(h, np.ndarray) or h.ndim not in (2, 3) or (h.ndim == 3 and h.shape[2] != 1) or h.size == 0:
        raise ValueError("Yükseklik girişine tek kanallı yükseklik haritası bağlanmalı")
    if h.dtype.kind not in "uif":
        raise ValueError(f"Yükseklik haritası sayı dizisi olmalı (gelen {h.dtype})")
    h = h.reshape(h.shape[0], h.shape[1]).astype(np.float32)
    h[~np.isfinite(h)] = np.nan
    return h


@register
class MeasureVolume(Node):
    roi = ("rect", "roi")
    title = "Hacim Ölç"
    category = CAT
    description = ("Zemin düzleminin üstündeki malzemenin hacmini cm³ olarak ölçer (yığın, dolum, kek, granül, "
                   "parça). Bölge ve isteğe bağlı maske ile sınırlanır. Kutu duvarları gibi yukarıdan "
                   "görünmeyen yüzeyler iki kez sayılmaz. Taban alanı (mm²) ve en yüksek nokta da verilir.")
    inputs = [("depth", IMG), ("intrinsics", ANY), ("plane", ANY), ("mask", IMG)]
    outputs = [("volume", NUM), ("area", NUM), ("max_height", NUM), ("found", BOOL)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("min_height", "float", 2.0, "Zeminden en az (mm; altı sayılmaz)", 0, 10000, 0.5),
              Param("fill", "int", 100, "Kayıp pikselleri doldur (en büyük delik, px)", 0, 100000)]

    def process(self, inputs):
        depth = D.fill_holes(_depth(inputs), int(self.v("fill")))
        intr = _intr(inputs, depth)
        plane = D.as_plane(inputs.get("plane"))
        sel = _region(self, inputs, depth.shape)
        r = D.volume_above(depth, intr, plane, sel, float(self.v("min_height")))
        return {"volume": round(r["volume"] / 1000.0, 3), "area": round(r["area"], 2),
                "max_height": round(r["max_height"], 3), "found": r["pixels"] > 0}


@register
class Find3DObjects(Node):
    roi = ("rect", "roi")
    title = "3B Nesne Bul"
    category = CAT
    description = ("Zemin düzleminin üstünde yükselen nesneleri ayırır (yükseklik eşiği) ve tespit listesi verir: "
                   "her nesnenin kutusu, yüksekliği (height, mm), hacmi (volume, cm³) ve taban alanı (area_mm2). "
                   "Tespitleri Çiz, Tespit Filtrele, sayma blokları doğrudan kullanır. Birbirine değen nesneler "
                   "tek nesne sayılır.")
    inputs = [("depth", IMG), ("intrinsics", ANY), ("plane", ANY), ("image", IMG)]
    outputs = [("detections", DET), ("count", NUM), ("image", IMG), ("mask", IMG), ("max_height", NUM),
               ("total_volume", NUM)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("min_height", "float", 5.0, "Zeminden en az (mm)", 0, 10000, 0.5),
              Param("max_height", "float", 0.0, "Zeminden en çok (mm, 0 = sınırsız)", 0, 100000, 1),
              Param("min_area", "float", 200.0, "En küçük taban alanı (mm²)", 0, 1e9, 10),
              Param("percentile", "float", 95.0, "Yükseklik yüzdeliği (%)", 50, 100, 1),
              Param("label", "str", "nesne", "Etiket"),
              Param("clean", "int", 1, "Gürültü temizliği (piksel)", 0, 20),
              Param("fill", "int", 100, "Kayıp pikselleri doldur (en büyük delik, px)", 0, 100000)]

    def process(self, inputs):
        depth = D.fill_holes(_depth(inputs), int(self.v("fill")))
        intr = _intr(inputs, depth)
        plane = D.as_plane(inputs.get("plane"))
        h = D.height_map(depth, intr, plane)
        area = D.pixel_areas(depth, intr, plane)
        sel = _region(self, inputs, depth.shape)
        fg = sel & np.isfinite(h) & (h > float(self.v("min_height")))
        if float(self.v("max_height")) > 0:
            fg &= h <= float(self.v("max_height"))
        m = fg.astype(np.uint8)
        k = int(self.v("clean"))
        if k > 0:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # bridge dropped pixels
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        dets, mask = [], np.zeros(depth.shape, np.uint8)
        label = str(self.v("label") or "nesne")
        pct = float(self.v("percentile"))
        for i in range(1, n):
            comp = lab == i
            meas = comp & fg                                 # pixels of the object that really are above
            hv = h[meas]
            av = area[meas]
            if not hv.size:
                continue
            footprint = float(np.nansum(av))
            if footprint < float(self.v("min_area")):
                continue
            x, y, w, hh = (int(v) for v in stats[i, :4])
            vol = float(np.nansum(hv.astype(np.float64) * av))
            pts, _ = D.depth_to_points(depth, intr, meas)
            centre = [round(float(c), 2) for c in pts.mean(0)] if len(pts) else [0.0, 0.0, 0.0]
            dets.append({"box": (x, y, w, hh), "score": 1.0, "class_id": 0, "label": label,
                         "height": round(float(np.percentile(hv, pct)), 3),
                         "mean_height": round(float(hv.mean()), 3), "volume": round(vol / 1000.0, 3),
                         "area_mm2": round(footprint, 2), "area": int(comp.sum()), "center": centre})
            mask[comp] = 255
        dets.sort(key=lambda d: (d["box"][1] // 40, d["box"][0]))           # reading order
        base = inputs.get("image")
        if isinstance(base, np.ndarray) and base.ndim in (2, 3) and base.shape[:2] == depth.shape:
            img = base.copy() if base.ndim == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if img.dtype != np.uint8:
                img, _, _ = D.colorize(h, cmap="Turbo")
            img = np.ascontiguousarray(img[:, :, :3])
        else:
            img, _, _ = D.colorize(h, cmap="Turbo")
        _draw_roi(img, self)
        for i, d in enumerate(dets, 1):
            x, y, w, hh = d["box"]
            cv2.rectangle(img, (x, y), (x + w, y + hh), (0, 255, 0), 2)
            put_text(img, f"{i}: {d['height']:.1f} mm  {d['volume']:.1f} cm³", (x, max(14, y - 6)), 0.5,
                     (255, 255, 255), 1)
        return {"detections": dets, "count": len(dets), "image": img, "mask": mask,
                "max_height": max((d["height"] for d in dets), default=0.0),
                "total_volume": round(sum(d["volume"] for d in dets), 3)}


@register
class Flatness(Node):
    roi = ("rect", "roi")
    title = "Düzlemsellik / Eğiklik"
    category = CAT
    description = ("Bölgedeki yüzeye düzlem oturtur: düzlemsellik (en yüksek - en alçak sapma, mm), en büyük "
                   "sapma, RMS ve eğiklik açısı (°). Referans düzlem bağlıysa (Zemin Düzlemi Bul) eğiklik ona "
                   "göre, değilse kamera eksenine göre ölçülür. Uç değer kırpma gürültüdeki tek tük pikselleri "
                   "yok sayar. Sapma haritası renkli resim olarak verilir.")
    inputs = [("depth", IMG), ("intrinsics", ANY), ("plane", ANY)]
    outputs = [("flatness", NUM), ("max_dev", NUM), ("rms", NUM), ("tilt", NUM), ("image", IMG)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("trim", "float", 0.5, "Uç değer kırpma (%)", 0, 20, 0.1),
              Param("outlier", "float", 0.0, "Aykırı nokta eşiği (mm, 0 = hepsi)", 0, 1000, 0.5)]

    def process(self, inputs):
        depth = _depth(inputs)
        intr = _intr(inputs, depth)
        sel = _region(self, inputs, depth.shape)
        pts, (vs, us) = D.depth_to_points(depth, intr, sel)
        if len(pts) < 10:
            raise ValueError(f"Bölgede yeterli geçerli derinlik yok ({len(pts)} nokta)")
        if float(self.v("outlier")) > 0:
            n, d, inl = D.ransac_plane(pts, float(self.v("outlier")), 200)
            if inl.sum() >= 3:
                n, d = D.fit_plane_lsq(pts[inl])
        else:
            n, d = D.fit_plane_lsq(pts)
        res = pts.astype(np.float64) @ n + d
        q = float(self.v("trim"))
        lo, hi = (np.percentile(res, [q, 100 - q]) if q > 0 else (res.min(), res.max()))
        flat = float(hi - lo)
        max_dev = float(max(abs(lo), abs(hi)))
        rms = float(np.sqrt((res ** 2).mean()))
        ref = inputs.get("plane")
        if ref is not None:
            rn, _ = D.as_plane(ref)
            tilt = math.degrees(math.acos(min(1.0, abs(float(n @ rn)))))
        else:
            tilt = D.tilt_deg(n)
        dev = np.full(depth.shape, np.nan, np.float32)
        dev[vs, us] = res
        lim = max(max_dev, 1e-3)
        img, _, _ = D.colorize(dev, -lim, lim, "Turbo")
        _draw_roi(img, self)
        put_text(img, f"düzlemsellik {flat:.2f} mm  eğim {tilt:.2f}°", (8, 22), 0.55, (255, 255, 255), 2)
        return {"flatness": round(flat, 4), "max_dev": round(max_dev, 4), "rms": round(rms, 4),
                "tilt": round(tilt, 4), "image": img}


@register
class DepthProfile(Node):
    roi = ("line", "line")
    title = "Derinlik Profili"
    category = CAT
    description = ("Çizdiğiniz çizgi boyunca derinlik (veya yükseklik haritası bağlanırsa yükseklik) değerlerinin "
                   "grafiğini çizer; en küçük, en büyük, ortalama ve çizginin gerçek uzunluğunu (mm; iç parametreler "
                   "bağlıysa) verir. Kesit, kaynak dikişi, oluk derinliği kontrolü için.")
    inputs = [("depth", IMG), ("intrinsics", ANY)]
    outputs = [("image", IMG), ("min", NUM), ("max", NUM), ("mean", NUM), ("length", NUM), ("values", ANY)]
    params = [Param("line", "str", "40,240,600,240", "Çizgi x1,y1,x2,y2"),
              Param("kind", "choice", "derinlik", "Değer", choices=["derinlik", "yükseklik"]),
              Param("chart_w", "int", 640, "Grafik genişliği", 160, 4096),
              Param("chart_h", "int", 240, "Grafik yüksekliği", 100, 2048)]

    def process(self, inputs):
        raw = inputs.get("depth")
        if self.v("kind") == "yükseklik":
            m = _height_input({"height": raw})
        else:
            m = D.as_depth(raw)
        H, W = m.shape
        x1, y1, x2, y2 = _nums(self.v("line"), 4, "Çizgi (x1,y1,x2,y2)")
        if not all(-1 <= x <= W and -1 <= y <= H for x, y in ((x1, y1), (x2, y2))):
            raise ValueError(f"Çizgi ({self.v('line')}) resmin dışında (resim {W}x{H})")
        n = int(max(2, round(math.hypot(x2 - x1, y2 - y1))) + 1)
        xs = np.clip(np.round(np.linspace(x1, x2, n)).astype(int), 0, W - 1)
        ys = np.clip(np.round(np.linspace(y1, y2, n)).astype(int), 0, H - 1)
        vals = m[ys, xs]
        ok = np.isfinite(vals)
        length = math.hypot(x2 - x1, y2 - y1)
        unit = "px"
        if inputs.get("intrinsics") is not None and self.v("kind") == "derinlik":
            intr = D.intrinsics(inputs["intrinsics"], m.shape)
            ends = []
            for u, v, z in ((x1, y1, vals[0]), (x2, y2, vals[-1])):
                if np.isfinite(z):
                    ends.append(np.array([(u - intr["cx"]) / intr["fx"] * z, (v - intr["cy"]) / intr["fy"] * z, z]))
            if len(ends) == 2:
                length, unit = float(np.linalg.norm(ends[1] - ends[0])), "mm"
        cw, ch = int(self.v("chart_w")), int(self.v("chart_h"))
        img = np.full((ch, cw, 3), 30, np.uint8)
        stats = {"min": 0.0, "max": 0.0, "mean": 0.0}
        if ok.any():
            v = vals[ok]
            stats = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}
            lo, hi = stats["min"], stats["max"]
            pad = max((hi - lo) * 0.1, 1.0)
            lo, hi = lo - pad, hi + pad
            px = np.linspace(40, cw - 10, n)
            # depth: far is low on the chart (like a cross-section seen from the side); height: high is high
            frac = (vals - lo) / (hi - lo)
            py = (ch - 25) - frac * (ch - 45) if self.v("kind") == "yükseklik" else 20 + frac * (ch - 45)
            for k in range(n - 1):
                if ok[k] and ok[k + 1]:
                    cv2.line(img, (int(px[k]), int(py[k])), (int(px[k + 1]), int(py[k + 1])), (80, 220, 255), 2,
                             cv2.LINE_AA)
            put_text(img, f"min {stats['min']:.1f}  max {stats['max']:.1f}  ort {stats['mean']:.1f} mm",
                     (44, 16), 0.45, (230, 230, 230), 1)
        else:
            put_text(img, "çizgide geçerli değer yok", (44, ch // 2), 0.6, (80, 80, 255), 1)
        put_text(img, f"uzunluk {length:.1f} {unit}", (44, ch - 6), 0.45, (200, 200, 200), 1)
        return {"image": img, "min": round(stats["min"], 3), "max": round(stats["max"], 3),
                "mean": round(stats["mean"], 3), "length": round(length, 3),
                "values": [None if not np.isfinite(x) else round(float(x), 3) for x in vals]}


SAVE_FORMATS = ["PLY (ikili)", "PLY (metin)", "PNG 16 bit + JSON"]


@register
class SavePointCloud(Node):
    title = "Nokta Bulutu Kaydet"
    category = CAT
    description = ("Derinlik haritasını nokta bulutu olarak kaydeder (.ply; renkli resim bağlıysa renkli). "
                   "CloudCompare, MeshLab, Open3D açar; Derinlik Kamerası 'Dosyadan' geri okur. Tetik gelince veya "
                   "her N karede bir zaman damgalı dosya yazar. PNG seçeneği 16 bit mm derinlik + iç parametre "
                   "JSON'u yazar.")
    inputs = [("depth", IMG), ("image", IMG), ("intrinsics", ANY), ("trigger", BOOL)]
    outputs = [("path", TEXT)]
    params = [Param("folder", "str", "nokta_bulutlari", "Klasör"),
              Param("prefix", "str", "bulut", "Dosya ön eki"),
              Param("format", "choice", SAVE_FORMATS[0], "Biçim", choices=SAVE_FORMATS),
              Param("every_n", "int", 0, "Tetik yoksa her N karede (0 = kapalı)", 0, 100000)]

    def setup(self):
        self.state["n"] = 0

    def process(self, inputs):
        self.state["n"] += 1
        trig = inputs.get("trigger")
        if trig is not None:
            fire = _rising(self, trig)
        else:
            n = self.v("every_n")
            fire = bool(n) and self.state["n"] % n == 0
        if not fire or inputs.get("depth") is None:
            return {"path": ""}
        depth = _depth(inputs)
        intr = _intr(inputs, depth)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = os.path.join(self.v("folder"), f"{self.v('prefix')}_{stamp}")
        fmt = self.v("format")
        if fmt == SAVE_FORMATS[2]:
            return {"path": D.save_depth_png(base + ".png", depth, intr)}
        pts, (vs, us) = D.depth_to_points(depth, intr)
        img = inputs.get("image")
        cols = None
        if isinstance(img, np.ndarray) and img.shape[:2] == depth.shape:
            c = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if c.dtype == np.uint8:
                cols = c[vs, us, :3]
        path = D.write_ply(base + ".ply", pts, cols, binary=fmt == SAVE_FORMATS[0], intr=intr,
                           size=(depth.shape[1], depth.shape[0]))
        return {"path": path}


VIEW_COLORS = ["yükseklik", "kamera rengi", "düz"]


@register
class PointCloudView(Node):
    title = "3B Görünüm"
    category = CAT
    description = ("Derinlik haritasını nokta bulutu olarak verilen açıdan çizer (resim çıkışı, panelde canlı "
                   "izlenir). 'Zemin düzlemi' bağlıysa renkler zeminden yüksekliği gösterir. Bloktaki düğme fareyle "
                   "döndürülüp yakınlaştırılan 3B pencereyi açar.")
    inputs = [("depth", IMG), ("image", IMG), ("intrinsics", ANY), ("plane", ANY)]
    outputs = [("image", IMG)]
    params = [Param("yaw", "float", -30.0, "Yatay açı (°)", -180, 180, 5),
              Param("pitch", "float", 35.0, "Dikey açı (°)", -89, 89, 5),
              Param("zoom", "float", 1.0, "Yakınlaştırma", 0.1, 20, 0.1),
              Param("color", "choice", VIEW_COLORS[0], "Renk", choices=VIEW_COLORS),
              Param("width", "int", 640, "Genişlik", 64, 4096),
              Param("height", "int", 480, "Yükseklik", 64, 4096),
              Param("point_size", "int", 2, "Nokta boyu (px)", 1, 8),
              Param("max_points", "int", 60000, "En çok nokta", 1000, 2_000_000)]
    dialog = VIEWER

    def process(self, inputs):
        depth = _depth(inputs)
        intr = _intr(inputs, depth)
        plane = inputs.get("plane")
        if plane is not None:
            D.as_plane(plane)                                    # a clear error for a wrong connection
        color = inputs.get("image")
        D.publish(self.id, depth, color, intr, plane)
        pts, cols = D.cloud_for_view(depth, intr, color, plane, self.v("color"), int(self.v("max_points")))
        img = D.render_cloud(pts, cols, (self.v("width"), self.v("height")), float(self.v("yaw")),
                             float(self.v("pitch")), float(self.v("zoom")), point_size=int(self.v("point_size")))
        return {"image": img}
