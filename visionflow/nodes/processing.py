"""Classic image processing blocks (OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core import Node, Param, PortType, register

CAT = "2 · Görüntü İşleme"
IMG = PortType.IMAGE


def _odd(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 else k + 1


def _bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


class _ImageOp(Node):
    """Single image in, single image out."""
    category = CAT
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        return {"image": self.apply(img)}

    def apply(self, img):
        raise NotImplementedError


@register
class Grayscale(_ImageOp):
    title = "Gri Tonlama"
    description = "Renkli görüntüyü griye çevirir."

    def apply(self, img):
        return _gray(img)


@register
class Blur(_ImageOp):
    title = "Bulanıklaştır"
    description = "Gauss, medyan veya bilateral filtre ile gürültü azaltır."
    params = [
        Param("mode", "choice", "gauss", "Yöntem", choices=["gauss", "median", "bilateral"]),
        Param("ksize", "int", 5, "Çekirdek", 1, 99, 2),
    ]

    def apply(self, img):
        k = _odd(self.v("ksize"))
        mode = self.v("mode")
        if mode == "median":
            return cv2.medianBlur(img, k)
        if mode == "bilateral":
            return cv2.bilateralFilter(img, k, 75, 75)
        return cv2.GaussianBlur(img, (k, k), 0)


@register
class Canny(_ImageOp):
    title = "Kenar Bul (Canny)"
    description = "Canny kenar algılama."
    params = [Param("t1", "int", 80, "Alt eşik", 0, 500), Param("t2", "int", 160, "Üst eşik", 0, 500)]

    def apply(self, img):
        return cv2.Canny(_gray(img), self.v("t1"), self.v("t2"))


@register
class Threshold(_ImageOp):
    title = "Eşikle"
    description = "Görüntüyü siyah/beyaz maskeye çevirir."
    params = [
        Param("mode", "choice", "binary", "Yöntem", choices=["binary", "binary_inv", "otsu", "adaptive"]),
        Param("value", "int", 127, "Eşik", 0, 255),
    ]

    def apply(self, img):
        g, mode = _gray(img), self.v("mode")
        if mode == "otsu":
            return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if mode == "adaptive":
            return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        flag = cv2.THRESH_BINARY_INV if mode == "binary_inv" else cv2.THRESH_BINARY
        return cv2.threshold(g, self.v("value"), 255, flag)[1]


@register
class ColorRange(Node):
    title = "Renk Filtresi (HSV)"
    category = CAT
    description = ("HSV aralığındaki pikselleri maske olarak seçer. Ton min > max ise aralık 180'den "
                   "başa sarar (kırmızı için ör. 170 → 10).")
    inputs = [("image", IMG)]
    outputs = [("mask", IMG), ("ratio", PortType.NUMBER)]
    params = [
        Param("h_min", "int", 0, "Ton min", 0, 179), Param("h_max", "int", 10, "Ton max", 0, 179),
        Param("s_min", "int", 120, "Doygunluk min", 0, 255), Param("s_max", "int", 255, "Doygunluk max", 0, 255),
        Param("v_min", "int", 70, "Parlaklık min", 0, 255), Param("v_max", "int", 255, "Parlaklık max", 0, 255),
    ]

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        hsv = cv2.cvtColor(_bgr(img), cv2.COLOR_BGR2HSV)
        h1, h2 = self.v("h_min"), self.v("h_max")
        lo_sv = [self.v("s_min"), self.v("v_min")]
        hi_sv = [self.v("s_max"), self.v("v_max")]
        if h1 <= h2:
            mask = cv2.inRange(hsv, np.array([h1, *lo_sv]), np.array([h2, *hi_sv]))
        else:   # wraps around 180, e.g. red = 170..10
            mask = cv2.inRange(hsv, np.array([h1, *lo_sv]), np.array([179, *hi_sv])) | \
                cv2.inRange(hsv, np.array([0, *lo_sv]), np.array([h2, *hi_sv]))
        return {"mask": mask, "ratio": round(float(np.count_nonzero(mask)) / mask.size * 100, 2)}


@register
class Morphology(_ImageOp):
    title = "Morfoloji"
    description = "Aşındır, genişlet, aç, kapat."
    params = [
        Param("op", "choice", "open", "İşlem", choices=["erode", "dilate", "open", "close"]),
        Param("ksize", "int", 5, "Çekirdek", 1, 51),
        Param("iterations", "int", 1, "Tekrar", 1, 20),
    ]

    def apply(self, img):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.v("ksize"),) * 2)
        ops = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
        return cv2.morphologyEx(img, ops[self.v("op")], k, iterations=self.v("iterations"))


@register
class Resize(_ImageOp):
    title = "Boyutlandır"
    description = "Ölçek veya sabit boyut ile yeniden boyutlandırır."
    params = [
        Param("scale", "float", 0.5, "Ölçek", 0.01, 10, 0.05),
        Param("width", "int", 0, "Genişlik (0 = ölçek kullan)", 0, 8000),
        Param("height", "int", 0, "Yükseklik", 0, 8000),
    ]

    def apply(self, img):
        if self.v("width") and self.v("height"):
            return cv2.resize(img, (self.v("width"), self.v("height")))
        s = self.v("scale")
        size = (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s)))   # never 0 px on tiny images
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


@register
class Crop(_ImageOp):
    roi = ("rect", ("x", "y", "w", "h"))
    title = "Kırp (ROI)"
    description = "İlgilenilen bölgeyi keser."
    params = [
        Param("x", "int", 0, "X", 0, 10000), Param("y", "int", 0, "Y", 0, 10000),
        Param("w", "int", 200, "Genişlik", 1, 10000), Param("h", "int", 200, "Yükseklik", 1, 10000),
    ]

    def apply(self, img):
        x, y, w, h = (self.v(k) for k in ("x", "y", "w", "h"))
        out = img[y:y + h, x:x + w].copy()
        if out.size == 0:
            raise ValueError(f"Kırpma bölgesi resmin dışında (resim {img.shape[1]}x{img.shape[0]})")
        return out


@register
class FlipRotate(_ImageOp):
    title = "Çevir / Döndür"
    params = [Param("op", "choice", "rot90", "İşlem", choices=["rot90", "rot180", "rot270", "flip_h", "flip_v"])]

    def apply(self, img):
        op = self.v("op")
        return {
            "rot90": lambda: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            "rot180": lambda: cv2.rotate(img, cv2.ROTATE_180),
            "rot270": lambda: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            "flip_h": lambda: cv2.flip(img, 1),
            "flip_v": lambda: cv2.flip(img, 0),
        }[op]()


@register
class ApplyMask(Node):
    title = "Maske Uygula"
    category = CAT
    description = "Maskenin beyaz olduğu yerleri bırakır, gerisini siler."
    inputs = [("image", IMG), ("mask", IMG)]
    outputs = [("image", IMG)]

    def process(self, inputs):
        img, mask = inputs["image"], inputs["mask"]
        if img is None or mask is None:
            raise ValueError("Görüntü ve maske gerekli")
        return {"image": cv2.bitwise_and(img, img, mask=_gray(mask))}


@register
class FindContours(Node):
    title = "Nesne Bul (Kontur)"
    category = CAT
    description = "Maskedeki nesneleri bulur, alana göre filtreler ve sayar."
    inputs = [("mask", IMG)]
    outputs = [("contours", PortType.CONTOURS), ("detections", PortType.DETECTIONS), ("count", PortType.NUMBER)]
    params = [
        Param("min_area", "int", 200, "Min alan", 0, 10_000_000),
        Param("max_area", "int", 0, "Max alan (0 = sınırsız)", 0, 10_000_000),
    ]

    def process(self, inputs):
        mask = inputs["mask"]
        if mask is None:
            raise ValueError("Maske girişi bağlı değil")
        cnts, _ = cv2.findContours(_gray(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lo, hi = self.v("min_area"), self.v("max_area") or float("inf")
        cnts = [c for c in cnts if lo <= cv2.contourArea(c) <= hi]
        dets = [{"box": tuple(int(v) for v in cv2.boundingRect(c)), "score": 1.0, "class_id": 0,
                 "label": f"nesne {int(cv2.contourArea(c))}px"} for c in cnts]
        return {"contours": cnts, "detections": dets, "count": len(cnts)}


@register
class DrawContours(Node):
    title = "Kontur Çiz"
    category = CAT
    inputs = [("image", IMG), ("contours", PortType.CONTOURS)]
    outputs = [("image", IMG)]
    params = [Param("thickness", "int", 2, "Kalınlık", 1, 20)]

    def process(self, inputs):
        if inputs["image"] is None:
            raise ValueError("Görüntü girişi bağlı değil")
        img = _bgr(inputs["image"]).copy()
        cv2.drawContours(img, inputs["contours"] or [], -1, (0, 255, 255), self.v("thickness"))
        return {"image": img}


@register
class TemplateMatch(Node):
    title = "Şablon Eşleştir"
    category = CAT
    description = "Bir şablon resmini görüntüde arar (logo, buton, parça konumu)."
    inputs = [("image", IMG)]
    outputs = [("detections", PortType.DETECTIONS), ("found", PortType.BOOL), ("score", PortType.NUMBER)]
    params = [Param("template", "path", "", "Şablon resmi"), Param("threshold", "float", 0.8, "Eşik", 0, 1, 0.01)]

    def process(self, inputs):
        from .inputs import _read_image

        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        if self.state.get("tpath") != self.v("template"):
            self.state["tpl"] = _gray(_read_image(self.v("template")))
            self.state["tpath"] = self.v("template")
        tpl = self.state["tpl"]
        res = cv2.matchTemplate(_gray(img), tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        found = score >= self.v("threshold")
        dets = [{"box": (loc[0], loc[1], tpl.shape[1], tpl.shape[0]), "score": float(score),
                 "class_id": 0, "label": "şablon"}] if found else []
        return {"detections": dets, "found": bool(found), "score": round(float(score), 3)}
