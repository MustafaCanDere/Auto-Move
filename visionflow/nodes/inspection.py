"""Quality inspection and measurement blocks: codes, markers, motion, reference comparison, measuring."""
from __future__ import annotations

import cv2
import numpy as np

from ._text import put_text
from ..core import Node, Param, PortType, register
from .processing import _bgr, _gray

CAT_IMG = "2 · Görüntü İşleme"
CAT = "2b · Kalite Kontrol ve Ölçüm"
IMG, NUM, BOOL, DET = PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.DETECTIONS


def _need(img):
    if img is None:
        raise ValueError("Görüntü girişi bağlı değil")
    return img


# --- image adjustments ---------------------------------------------------------------------

@register
class BrightnessContrast(Node):
    title = "Parlaklık / Kontrast"
    category = CAT_IMG
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [Param("contrast", "float", 1.0, "Kontrast", 0, 3, 0.05),
              Param("brightness", "int", 0, "Parlaklık", -255, 255),
              Param("gamma", "float", 1.0, "Gama", 0.1, 3, 0.05)]

    def process(self, inputs):
        img = cv2.convertScaleAbs(_need(inputs["image"]), alpha=self.v("contrast"), beta=self.v("brightness"))
        g = self.v("gamma")
        if abs(g - 1.0) > 1e-3:
            lut = np.clip(((np.arange(256) / 255.0) ** (1 / g)) * 255, 0, 255).astype(np.uint8)
            img = cv2.LUT(img, lut)
        return {"image": img}


@register
class Equalize(Node):
    title = "Histogram Eşitle (CLAHE)"
    category = CAT_IMG
    description = "Karanlık / düşük kontrastlı görüntülerde detayı ortaya çıkarır."
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [Param("clip", "float", 2.0, "Kırpma sınırı", 0.1, 10, 0.1), Param("tile", "int", 8, "Karo", 1, 32)]

    def process(self, inputs):
        img = _need(inputs["image"])
        clahe = cv2.createCLAHE(self.v("clip"), (self.v("tile"),) * 2)
        if img.ndim == 2:
            return {"image": clahe.apply(img)}
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return {"image": cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)}


@register
class Combine(Node):
    title = "Yan Yana Birleştir"
    category = CAT_IMG
    description = "İki görüntüyü yan yana veya alt alta koyar (karşılaştırma için)."
    inputs = [("a", IMG), ("b", IMG)]
    outputs = [("image", IMG)]
    params = [Param("mode", "choice", "yatay", "Yön", choices=["yatay", "dikey", "üst üste (%50)"])]

    def process(self, inputs):
        a, b = _bgr(_need(inputs["a"])), _bgr(_need(inputs["b"]))
        mode = self.v("mode")
        if mode == "yatay":
            b = cv2.resize(b, (int(b.shape[1] * a.shape[0] / b.shape[0]), a.shape[0]))
            return {"image": np.hstack([a, b])}
        b = cv2.resize(b, (a.shape[1], int(b.shape[0] * a.shape[1] / b.shape[1])) if mode == "dikey" else
                       (a.shape[1], a.shape[0]))
        if mode == "dikey":
            return {"image": np.vstack([a, b])}
        return {"image": cv2.addWeighted(a, 0.5, b, 0.5, 0)}


@register
class DrawText(Node):
    title = "Yazı Yaz"
    category = CAT_IMG
    description = "Görüntüye yazı ekler. {a} ve {b} yerine girişteki değerler yazılır."
    inputs = [("image", IMG), ("a", PortType.ANY), ("b", PortType.ANY)]
    outputs = [("image", IMG)]
    params = [Param("text", "str", "Adet: {a}", "Metin"),
              Param("x", "int", 10, "X", 0, 10000), Param("y", "int", 30, "Y", 0, 10000),
              Param("scale", "float", 0.9, "Boyut", 0.2, 5, 0.1),
              Param("color", "choice", "beyaz", "Renk", choices=["beyaz", "yeşil", "kırmızı", "sarı", "siyah"])]
    COLORS = {"beyaz": (255, 255, 255), "yeşil": (0, 220, 0), "kırmızı": (0, 0, 255), "sarı": (0, 230, 255),
              "siyah": (0, 0, 0)}

    def process(self, inputs):
        img = _bgr(_need(inputs["image"])).copy()
        text = self.v("text").replace("{a}", str(inputs["a"])).replace("{b}", str(inputs["b"]))
        org, sc = (self.v("x"), self.v("y")), self.v("scale")
        put_text(img, text, org, sc, (0, 0, 0), max(1, int(sc * 5)))
        put_text(img, text, org, sc, self.COLORS[self.v("color")], max(1, int(sc * 2)))
        return {"image": img}


@register
class DrawShape(Node):
    title = "Bölge / Çizgi Çiz"
    category = CAT_IMG
    description = "Görüntüye dikdörtgen bölge veya çizgi çizer (sayım çizgisi, kontrol alanı göstermek için)."
    inputs = [("image", IMG), ("active", BOOL)]
    outputs = [("image", IMG)]
    params = [Param("shape", "choice", "dikdörtgen", "Şekil", choices=["dikdörtgen", "çizgi"]),
              Param("coords", "str", "100,100,300,200", "x1,y1,x2,y2"),
              Param("thickness", "int", 2, "Kalınlık", 1, 20)]

    def process(self, inputs):
        img = _bgr(_need(inputs["image"])).copy()
        x1, y1, x2, y2 = (int(v) for v in self.v("coords").split(","))
        color = (0, 0, 255) if inputs["active"] else (0, 220, 255)
        if self.v("shape") == "çizgi":
            cv2.line(img, (x1, y1), (x2, y2), color, self.v("thickness"))
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, self.v("thickness"))
        return {"image": img}


# --- inspection ---------------------------------------------------------------------------------

@register
class CodeReader(Node):
    title = "QR / Barkod Oku"
    category = CAT
    description = "QR kod ve 1B barkodları (EAN, UPC, Code128…) okur."
    inputs = [("image", IMG)]
    outputs = [("text", PortType.TEXT), ("detections", DET), ("found", BOOL), ("count", NUM)]
    params = [Param("kind", "choice", "ikisi", "Tür", choices=["ikisi", "QR", "barkod"])]

    def setup(self):
        self.state["qr"] = cv2.QRCodeDetector()
        self.state["bar"] = cv2.barcode.BarcodeDetector() if hasattr(cv2, "barcode") else None

    def process(self, inputs):
        img = _need(inputs["image"])
        dets: list[dict] = []
        kind = self.v("kind")
        if kind in ("ikisi", "QR"):
            ok, texts, pts, _ = self.state["qr"].detectAndDecodeMulti(img)
            if ok and pts is not None:
                for t, p in zip(texts, pts):
                    if t:
                        dets.append({"box": tuple(int(v) for v in cv2.boundingRect(p.astype(np.int32))),
                                     "score": 1.0, "class_id": 0, "label": t})
        if kind in ("ikisi", "barkod") and self.state["bar"] is not None:
            res = self.state["bar"].detectAndDecodeWithType(img)
            ok, texts, pts = res[0], res[1], res[-1]
            if ok and pts is not None:
                for t, p in zip(texts, pts):
                    if t:
                        dets.append({"box": tuple(int(v) for v in cv2.boundingRect(p.astype(np.int32))),
                                     "score": 1.0, "class_id": 1, "label": t})
        return {"text": dets[0]["label"] if dets else "", "detections": dets, "found": bool(dets),
                "count": len(dets)}


@register
class ArucoDetector(Node):
    setup_params = ("dict",)
    title = "ArUco İşaretçi Bul"
    category = CAT
    description = "ArUco işaretçilerini bulur; robot konumlama, kalibrasyon ve parça takibi için."
    inputs = [("image", IMG)]
    outputs = [("detections", DET), ("count", NUM)]
    params = [Param("dict", "choice", "4X4_50", "Sözlük", choices=["4X4_50", "5X5_100", "6X6_250", "7X7_1000"])]

    def setup(self):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{self.v('dict')}"))
        self.state["det"] = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    def process(self, inputs):
        corners, ids, _ = self.state["det"].detectMarkers(_need(inputs["image"]))
        dets = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                dets.append({"box": tuple(int(v) for v in cv2.boundingRect(c.astype(np.int32))), "score": 1.0,
                             "class_id": int(i), "label": f"id {int(i)}"})
        return {"detections": dets, "count": len(dets)}


@register
class MotionDetector(Node):
    setup_params = ("history", "sensitivity")
    title = "Hareket Algıla"
    category = CAT
    description = "Arka planı öğrenir, hareket eden bölgeleri maske olarak verir."
    inputs = [("image", IMG)]
    outputs = [("mask", IMG), ("motion", BOOL), ("ratio", NUM)]
    params = [Param("sensitivity", "int", 25, "Hassasiyet (düşük = hassas)", 1, 200),
              Param("min_ratio", "float", 0.5, "Hareket için min alan %", 0, 100, 0.1),
              Param("history", "int", 300, "Öğrenme süresi (kare)", 10, 5000)]

    def setup(self):
        self.state["bg"] = cv2.createBackgroundSubtractorMOG2(self.v("history"), self.v("sensitivity"), False)

    def process(self, inputs):
        img = _need(inputs["image"])
        mask = self.state["bg"].apply(img)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ratio = float(np.count_nonzero(mask)) / mask.size * 100
        return {"mask": mask, "motion": ratio >= self.v("min_ratio"), "ratio": round(ratio, 2)}


@register
class ReferenceCompare(Node):
    title = "Referansla Karşılaştır"
    category = CAT
    description = ("Görüntüyü hatasız bir referans resimle karşılaştırır; farklı bölgeleri maske ve "
                   "tespit olarak verir (eksik parça, çizik, leke kontrolü).")
    inputs = [("image", IMG)]
    outputs = [("mask", IMG), ("detections", DET), ("ok", BOOL), ("diff", NUM)]
    params = [Param("reference", "path", "", "Referans resim"),
              Param("threshold", "int", 40, "Fark eşiği", 1, 255),
              Param("min_area", "int", 50, "Min hata alanı (px)", 1, 1_000_000),
              Param("blur", "int", 5, "Yumuşatma", 1, 31)]

    def setup(self):
        from .inputs import _read_image
        path = self.v("reference")
        self.state["ref"] = _gray(_read_image(path)) if path else None

    def process(self, inputs):
        img = _gray(_need(inputs["image"]))
        ref = self.state.get("ref")
        if ref is None:
            raise ValueError("Referans resim seçilmedi")
        if ref.shape != img.shape:
            ref = cv2.resize(ref, (img.shape[1], img.shape[0]))
        k = self.v("blur") | 1
        diff = cv2.absdiff(cv2.GaussianBlur(img, (k, k), 0), cv2.GaussianBlur(ref, (k, k), 0))
        mask = cv2.threshold(diff, self.v("threshold"), 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = [{"box": tuple(int(v) for v in cv2.boundingRect(c)), "score": 1.0, "class_id": 0,
                 "label": "hata"} for c in cnts if cv2.contourArea(c) >= self.v("min_area")]
        return {"mask": mask, "detections": dets, "ok": not dets,
                "diff": round(float(np.count_nonzero(mask)) / mask.size * 100, 3)}


@register
class Sharpness(Node):
    title = "Netlik Ölç"
    category = CAT
    description = "Laplace varyansı ile görüntü netliğini ölçer; bulanık / odak dışı kareleri ayıklar."
    inputs = [("image", IMG)]
    outputs = [("score", NUM), ("sharp", BOOL)]
    params = [Param("min_score", "float", 100.0, "Net sayılması için min", 0, 100000)]

    def process(self, inputs):
        score = float(cv2.Laplacian(_gray(_need(inputs["image"])), cv2.CV_64F).var())
        return {"score": round(score, 1), "sharp": score >= self.v("min_score")}


NAMED_COLORS = {"kırmızı": 0, "turuncu": 15, "sarı": 30, "yeşil": 60, "camgöbeği": 90, "mavi": 120,
                "mor": 145, "pembe": 165}


@register
class AverageColor(Node):
    roi = ("rect", "roi")
    title = "Renk Tanı"
    category = CAT
    description = "Bölgenin ortalama rengini ölçer ve adını söyler (kırmızı, mavi, siyah…). Renk ayıklama için."
    inputs = [("image", IMG)]
    outputs = [("color", PortType.TEXT), ("hue", NUM), ("bgr", PortType.ANY)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)")]

    def process(self, inputs):
        img = _bgr(_need(inputs["image"]))
        if self.v("roi").strip():
            x, y, w, h = (int(v) for v in self.v("roi").split(","))
            img = img[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
            if img.size == 0:
                raise ValueError(f"Bölge ({self.v('roi')}) resmin dışında")
        b, g, r = (float(v) for v in np.nan_to_num(img.reshape(-1, 3).astype(np.float64)).mean(0))
        h, s, v = (int(x) for x in cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0])
        if v < 50:
            name = "siyah"
        elif s < 40:
            name = "beyaz" if v > 190 else "gri"
        else:
            name = min(NAMED_COLORS, key=lambda n: min(abs(h - NAMED_COLORS[n]), 180 - abs(h - NAMED_COLORS[n])))
        return {"color": name, "hue": h, "bgr": (round(b), round(g), round(r))}


@register
class MeasureObjects(Node):
    title = "Nesne Ölç"
    category = CAT
    description = ("Konturların genişlik, yükseklik ve açısını ölçer. 'mm başına piksel' girilirse (veya Ölçek Bul "
                   "bloğundan bağlanırsa) sonuç milimetre olur. Ölçüleri görüntüye de çizer.")
    inputs = [("image", IMG), ("contours", PortType.CONTOURS), ("px_per_mm", NUM)]
    outputs = [("image", IMG), ("detections", DET), ("width", NUM), ("height", NUM), ("angle", NUM)]
    params = [Param("px_per_mm", "float", 0.0, "mm başına piksel (0 = piksel)", 0, 10000, 0.01)]

    def process(self, inputs):
        img = _bgr(_need(inputs["image"])).copy()
        ppm = inputs.get("px_per_mm") or self.v("px_per_mm")     # a connected scale wins over the setting
        scale = ppm or 1.0
        unit = "mm" if ppm else "px"
        dets, first = [], None
        for c in inputs["contours"] or []:
            (cx, cy), (w, h), ang = cv2.minAreaRect(c)
            if w < h:
                w, h, ang = h, w, ang + 90
            w_u, h_u = w / scale, h / scale
            box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            label = f"{w_u:.1f}x{h_u:.1f}{unit} {ang:.0f}°"
            put_text(img, label, (int(cx) - 40, int(cy)), 0.5, (0, 255, 0), 1)
            dets.append({"box": tuple(int(v) for v in cv2.boundingRect(c)), "score": 1.0, "class_id": 0,
                         "label": label, "width": w_u, "height": h_u, "angle": ang})
            first = first or (w_u, h_u, ang)
        w_u, h_u, ang = first or (0.0, 0.0, 0.0)
        return {"image": img, "detections": dets, "width": round(w_u, 2), "height": round(h_u, 2),
                "angle": round(ang, 1)}


@register
class OCR(Node):
    title = "Yazı Oku (OCR)"
    category = CAT
    description = ("Görüntüdeki yazıyı okur: etiket, seri no, SKT tarihi, plaka. Tesseract kurulu olmalı "
                   "(Windows: UB-Mannheim kurulumu, Linux: apt install tesseract-ocr tesseract-ocr-tur).")
    inputs = [("image", IMG)]
    outputs = [("text", PortType.TEXT), ("detections", DET), ("found", BOOL), ("number", NUM)]
    params = [Param("lang", "choice", "tur+eng", "Dil", choices=["tur+eng", "eng", "tur"]),
              Param("mode", "choice", "blok", "Düzen", choices=["blok", "tek satır", "tek kelime", "seyrek"]),
              Param("whitelist", "str", "", "Sadece bu karakterler (ör. 0123456789)"),
              Param("prep", "choice", "otsu", "Ön işleme", choices=["yok", "otsu", "ters otsu"]),
              Param("scale", "float", 2.0, "Büyütme", 1, 6, 0.5),
              Param("min_conf", "int", 50, "Min güven (%)", 0, 100),
              Param("tesseract", "path", "", "tesseract.exe yolu (PATH'te değilse)")]
    PSM = {"blok": 6, "tek satır": 7, "tek kelime": 8, "seyrek": 11}

    def setup(self):
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("OCR için 'pip install pytesseract' ve Tesseract kurulumu gerekli") from exc
        exe = self.v("tesseract") or _default_tesseract()
        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe
        self.state["tess"] = pytesseract

    def process(self, inputs):
        img = _gray(_need(inputs["image"]))
        s = self.v("scale")
        if s != 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        prep = self.v("prep")
        if prep != "yok":
            flag = cv2.THRESH_BINARY_INV if prep == "ters otsu" else cv2.THRESH_BINARY
            img = cv2.threshold(img, 0, 255, flag + cv2.THRESH_OTSU)[1]
        cfg = f"--psm {self.PSM[self.v('mode')]}"
        if self.v("whitelist"):
            cfg += f" -c tessedit_char_whitelist={self.v('whitelist')}"
        tess = self.state["tess"]
        data = tess.image_to_data(img, lang=self.v("lang"), config=cfg, output_type=tess.Output.DICT)
        words, dets = [], []
        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if word.strip() and conf >= self.v("min_conf"):
                words.append(word.strip())
                x, y, w, h = (int(data[k][i] / s) for k in ("left", "top", "width", "height"))
                dets.append({"box": (x, y, w, h), "score": conf / 100, "class_id": 0, "label": word.strip()})
        text = " ".join(words)
        digits = "".join(ch for ch in text if ch.isdigit())
        return {"text": text, "detections": dets, "found": bool(text), "number": int(digits) if digits else 0}


def _default_tesseract() -> str:
    import os
    import shutil
    if shutil.which("tesseract"):
        return ""
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(p):
            return p
    return ""
