"""More machine-vision blocks: feature matching, Hough lines, caliper, blob defects, fill level, colour area,
histogram comparison, auto alignment and the tray (grid) check."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ._text import put_text
from ..core import Node, Param, PortType, register

CAT = "2b · Kalite Kontrol ve Ölçüm"
IMG, NUM, BOOL, TEXT, DET, ANY = (PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.TEXT, PortType.DETECTIONS,
                                  PortType.ANY)
GREEN, RED, YELLOW, CYAN = (0, 220, 0), (0, 0, 255), (0, 230, 255), (255, 255, 0)


# --- helpers ------------------------------------------------------------------------------------------------------

def to_bgr8(img, what: str = "Görüntü") -> np.ndarray:
    """Any picture (gray, BGR, BGRA, 16-bit, float 0..1 or 0..255, bool) -> contiguous 8-bit 3-channel BGR.
    Levels keep their meaning (no min-max stretch), so thresholds behave the same for 8- and 16-bit cameras."""
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


def to_gray8(img, what: str = "Görüntü") -> np.ndarray:
    return cv2.cvtColor(to_bgr8(img, what), cv2.COLOR_BGR2GRAY)


def _nums(text: str, n: int, what: str) -> list[float]:
    parts = [p for p in str(text).replace(";", ",").split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"{what} {n} sayı olmalı (virgülle), girilen: '{text}'")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"{what} sayı olmalı, girilen: '{text}'") from None
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{what} geçersiz sayı içeriyor: '{text}'")
    return vals


def parse_roi(text: str, shape) -> tuple[int, int, int, int]:
    """'x,y,w,h' (empty = the whole picture) clipped to the picture; error when nothing of it is inside."""
    H, W = shape[:2]
    if not str(text or "").strip():
        return 0, 0, W, H
    x, y, w, h = (int(round(v)) for v in _nums(text, 4, "Bölge (x,y,w,h)"))
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"Bölge ({text}) resmin dışında veya çok küçük (resim {W}x{H})")
    return x0, y0, x1 - x0, y1 - y0


def _ppm(inputs, node) -> float:
    v = inputs.get("px_per_mm")
    try:
        v = float(v) if v is not None and not isinstance(v, bool) else 0.0
    except (TypeError, ValueError):
        v = 0.0
    if not math.isfinite(v) or v <= 0:
        v = float(node.v("px_per_mm") or 0.0)
    return v if v > 0 else 0.0


def _ref_image(node, inputs) -> np.ndarray:
    """The reference picture: the 'reference' input when connected, else the file setting (read once)."""
    ref = inputs.get("reference")
    if ref is not None:
        return to_bgr8(ref, "Referans")
    path = str(node.v("reference") or "").strip()
    if not path:
        raise ValueError("Referans resim yok: dosya seçin veya 'reference' girişine resim bağlayın")
    cache = node.state.get("ref_file")
    if cache is None or cache[0] != path:
        from .inputs import _read_image
        cache = node.state["ref_file"] = (path, _read_image(path))
    return cache[1]


def _hsv_mask(bgr, h1, h2, s1, s2, v1, v2):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo_sv, hi_sv = [min(s1, s2), min(v1, v2)], [max(s1, s2), max(v1, v2)]
    if h1 <= h2:
        return cv2.inRange(hsv, np.array([h1, *lo_sv]), np.array([h2, *hi_sv]))
    return cv2.inRange(hsv, np.array([h1, *lo_sv]), np.array([179, *hi_sv])) | \
        cv2.inRange(hsv, np.array([0, *lo_sv]), np.array([h2, *hi_sv]))


def hsv_params() -> list[Param]:
    return [Param("h_min", "int", 35, "Ton min", 0, 179), Param("h_max", "int", 85, "Ton max", 0, 179),
            Param("s_min", "int", 60, "Doygunluk min", 0, 255), Param("s_max", "int", 255, "Doygunluk max", 0, 255),
            Param("v_min", "int", 40, "Parlaklık min", 0, 255), Param("v_max", "int", 255, "Parlaklık max", 0, 255)]


def _hsv_of(node, bgr):
    return _hsv_mask(bgr, *(int(node.v(k)) for k in ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max")))


def _ccw_angle(dx: float, dy: float) -> float:
    """Angle of an image-space vector in degrees, counter-clockwise positive as seen on screen (y points down)."""
    return math.degrees(math.atan2(-dy, dx))


def _wrap180(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _features(gray, method: str, n: int):
    """Keypoints and descriptors. SIFT gives float descriptors (L2), ORB binary ones (Hamming)."""
    if method.startswith("SIFT"):
        make = getattr(cv2, "SIFT_create", None)
        if make is None:
            raise ValueError("Bu OpenCV sürümünde SIFT yok: yöntem olarak ORB seçin")
        det = make(nfeatures=int(n))
    else:
        det = cv2.ORB_create(nfeatures=int(n), scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=15,
                             fastThreshold=10)
    kp, des = det.detectAndCompute(gray, None)
    return kp, des


def _ratio_matches(des_ref, des_img, ratio: float):
    if des_ref is None or des_img is None or len(des_ref) < 2 or len(des_img) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2 if des_ref.dtype == np.float32 else cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(des_ref, des_img, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


# --- 1. feature matching --------------------------------------------------------------------------------------------

@register
class FeatureMatch(Node):
    title = "Özellik Eşleştir (Nesne Bul)"
    category = CAT
    description = ("Referans resimdeki nesneyi (etiket, logo, parça) döndürülmüş, büyütülmüş/küçültülmüş veya "
                   "kısmen kapanmış olsa da bulur: ORB/SIFT özellik noktaları + oran testi + RANSAC homografi. "
                   "Şablon Eşleştir'den farkı dönmeye ve ölçeğe dayanıklı olmasıdır. Referans desenli/dokulu olmalı; "
                   "düz renkli parçalarda Nesne Bul (Kontur) kullanın. Çıkış: kutu, merkez, açı (°, saat yönü tersi "
                   "+), ölçek ve skor (eşleşmelerin RANSAC'a uyan oranı).")
    inputs = [("image", IMG), ("reference", IMG)]
    outputs = [("image", IMG), ("detections", DET), ("found", BOOL), ("count", NUM), ("angle", NUM),
               ("scale", NUM), ("x", NUM), ("y", NUM), ("score", NUM)]
    params = [Param("reference", "path", "", "Referans resim (giriş bağlıysa o)"),
              Param("method", "choice", "ORB (hızlı)", "Yöntem", choices=["ORB (hızlı)", "SIFT (daha sağlam, yavaş)"]),
              Param("features", "int", 2000, "En çok özellik noktası", 100, 20000, 100),
              Param("ratio", "float", 0.75, "Oran testi (küçük = seçici)", 0.4, 0.95, 0.05),
              Param("min_matches", "int", 12, "En az eşleşme (RANSAC'a uyan)", 4, 1000),
              Param("max_objects", "int", 1, "En çok kaç nesne", 1, 20),
              Param("label", "str", "nesne", "Etiket")]

    def setup(self):
        self.state["ref_feat"] = None

    def _ref_features(self, ref):
        key = (ref.shape, self.v("method"), self.v("features"), hash(ref[::5, ::5].tobytes()))
        cached = self.state.get("ref_feat")
        if cached is None or cached[0] != key:
            kp, des = _features(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), self.v("method"), self.v("features"))
            if des is None or len(kp) < 4:
                raise ValueError("Referans resimde yeterli özellik noktası yok: daha desenli/dokulu bir bölge seçin")
            cached = self.state["ref_feat"] = (key, kp, des)
        return cached[1], cached[2]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        ref = _ref_image(self, inputs)
        kp_r, des_r = self._ref_features(ref)
        kp_i, des_i = _features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.v("method"),
                                max(self.v("features"), 500) * 2)
        good = _ratio_matches(des_r, des_i, float(self.v("ratio")))
        h, w = ref.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        out = img.copy()
        dets = []
        min_m = int(self.v("min_matches"))
        remaining = good
        H_img, W_img = img.shape[:2]
        while remaining and len(dets) < int(self.v("max_objects")) and len(remaining) >= max(4, min_m):
            src = np.float32([kp_r[m.queryIdx].pt for m in remaining]).reshape(-1, 1, 2)
            dst = np.float32([kp_i[m.trainIdx].pt for m in remaining]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None or inl is None:
                break
            inl = inl.ravel().astype(bool)
            poly = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            if int(inl.sum()) < min_m or not self._plausible(poly, w, h, W_img, H_img):
                break
            top = poly[1] - poly[0]
            angle = _wrap180(_ccw_angle(float(top[0]), float(top[1])))
            scale = math.sqrt(abs(cv2.contourArea(poly.astype(np.float32))) / float(w * h))
            center = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H).reshape(2)
            score = float(inl.sum()) / len(remaining)
            bx, by, bw, bh = cv2.boundingRect(poly.astype(np.float32))
            label = f"{self.v('label')} {angle:.0f}° x{scale:.2f}"
            dets.append({"box": (int(bx), int(by), int(bw), int(bh)), "score": round(score, 3), "class_id": 0,
                         "label": label, "center": (float(center[0]), float(center[1])),
                         "angle": round(angle, 2), "scale": round(scale, 4), "matches": int(inl.sum()),
                         "poly": [(float(px), float(py)) for px, py in poly]})
            cv2.polylines(out, [np.round(poly).astype(np.int32)], True, GREEN, 2)
            cv2.circle(out, (int(center[0]), int(center[1])), 4, RED, -1)
            put_text(out, label, (int(bx), max(15, int(by) - 6)), 0.6, GREEN, 2)
            # forget the matches of this object, look for the next one in the rest
            contour = poly.astype(np.float32).reshape(-1, 1, 2)
            remaining = [m for m, ok in zip(remaining, inl) if not ok
                         and cv2.pointPolygonTest(contour, kp_i[m.trainIdx].pt, False) < 0]
        if not dets:
            put_text(out, f"Bulunamadı ({len(good)} eşleşme)", (10, 30), 0.7, RED, 2)
        best = dets[0] if dets else None
        return {"image": out, "detections": dets, "found": bool(dets), "count": len(dets),
                "angle": best["angle"] if best else 0.0, "scale": best["scale"] if best else 0.0,
                "x": round(best["center"][0], 2) if best else 0.0, "y": round(best["center"][1], 2) if best else 0.0,
                "score": best["score"] if best else 0.0}

    @staticmethod
    def _plausible(poly, w, h, W, H) -> bool:
        """A homography from bad matches folds the box, shrinks it to a dot or blows it up: reject those."""
        if not np.isfinite(poly).all() or not cv2.isContourConvex(np.round(poly * 8).astype(np.int32)):
            return False
        signed = cv2.contourArea(poly.astype(np.float32), oriented=True)
        ref_sign = cv2.contourArea(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), oriented=True)
        if signed * ref_sign <= 0:                       # mirrored: a real part cannot turn inside out
            return False
        area = abs(signed)
        if area < 0.0025 * w * h or area > 100.0 * w * h or area < 64:
            return False
        sides = np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1)
        if sides.min() < 1e-3 or sides.max() / sides.min() > 6 * max(w, h) / max(1, min(w, h)):
            return False
        cx, cy = poly.mean(0)
        return -W <= cx <= 2 * W and -H <= cy <= 2 * H


# --- 2. Hough lines -------------------------------------------------------------------------------------------------

@register
class HoughLines(Node):
    roi = ("rect", "roi")
    title = "Çizgi Bul (Hough)"
    category = CAT
    description = ("Görüntüdeki düz çizgileri (kenar, çizik, kablo, etiket kenarı) bulur. En uzun çizginin açısını "
                   "(°; yatay = 0, saat yönü tersi +) ve uzunluğunu verir: eğik takılmış parça, etiket açısı, bant "
                   "kayması kontrolü. Açı aralığıyla sadece istenen yöndeki çizgiler sayılır.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("detections", DET), ("count", NUM), ("angle", NUM), ("length", NUM),
               ("found", BOOL)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("t1", "int", 50, "Kenar eşiği alt (Canny)", 0, 1000),
              Param("t2", "int", 150, "Kenar eşiği üst (Canny)", 0, 1000),
              Param("votes", "int", 50, "Oy eşiği (yüksek = sadece belirgin)", 1, 5000),
              Param("min_length", "int", 40, "En kısa çizgi (px)", 1, 20000),
              Param("max_gap", "int", 10, "Birleştirilecek boşluk (px)", 0, 1000),
              Param("angle_min", "float", -90.0, "Açı min (°)", -90, 90, 1),
              Param("angle_max", "float", 90.0, "Açı max (°)", -90, 90, 1)]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        gray = cv2.cvtColor(img[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.v("t1"), self.v("t2"))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, int(self.v("votes")), minLineLength=int(self.v("min_length")),
                                maxLineGap=int(self.v("max_gap")))
        lo, hi = sorted((float(self.v("angle_min")), float(self.v("angle_max"))))
        out = img.copy()
        if self.v("roi").strip():
            cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), YELLOW, 1)
        dets = []
        for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):
            x1, y1, x2, y2 = int(x1) + x0, int(y1) + y0, int(x2) + x0, int(y2) + y0
            if x2 < x1 or (x2 == x1 and y2 < y1):           # left point first: the angle is then -90..90
                x1, y1, x2, y2 = x2, y2, x1, y1
            ang = _ccw_angle(x2 - x1, y2 - y1)
            if ang <= -90.0:
                ang += 180.0
            if not lo <= ang <= hi:
                continue
            length = math.hypot(x2 - x1, y2 - y1)
            dets.append({"box": (min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1), "score": 1.0,
                         "class_id": 0, "label": f"{ang:.1f}°", "line": (x1, y1, x2, y2), "angle": round(ang, 2),
                         "length": round(length, 2), "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0)})
        dets.sort(key=lambda d: -d["length"])
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["line"]
            cv2.line(out, (x1, y1), (x2, y2), GREEN if i else RED, 2)
        if dets:
            x1, y1, _, _ = dets[0]["line"]
            put_text(out, f"{dets[0]['angle']:.1f}°  {dets[0]['length']:.0f}px", (x1, max(15, y1 - 8)), 0.6, RED, 2)
        best = dets[0] if dets else None
        return {"image": out, "detections": dets, "count": len(dets), "angle": best["angle"] if best else 0.0,
                "length": best["length"] if best else 0.0, "found": bool(dets)}


# --- 3. caliper -----------------------------------------------------------------------------------------------------

POLARITY = ["herhangi", "koyudan açığa", "açıktan koyuya"]
PAIRING = ["ilk iki kenar", "en dış kenarlar"]


def edge_profile(gray: np.ndarray, p0, p1, width: int, sigma: float):
    """Brightness along p0→p1 sampled every pixel (averaged over `width` parallel lines) and its smoothed
    derivative. Returns (positions along the line in px, profile, gradient, unit vector)."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    length = float(np.linalg.norm(p1 - p0))
    if length < 3:
        raise ValueError("Ölçüm çizgisi çok kısa (en az 3 piksel)")
    u = (p1 - p0) / length
    nrm = np.array([-u[1], u[0]])
    n = int(math.floor(length)) + 1
    t = np.arange(n, dtype=np.float64)
    offs = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    xs = (p0[0] + t[None, :] * u[0] + offs[:, None] * nrm[0]).astype(np.float32)
    ys = (p0[1] + t[None, :] * u[1] + offs[:, None] * nrm[1]).astype(np.float32)
    samples = cv2.remap(gray.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    prof = samples.mean(axis=0).astype(np.float64)
    if sigma > 0:
        r = max(1, int(math.ceil(3 * sigma)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        prof = np.convolve(np.pad(prof, r, mode="edge"), k, mode="valid")
    grad = np.gradient(prof)
    return t, prof, grad, u


def find_edges(grad: np.ndarray, min_contrast: float, polarity: str) -> list[tuple[float, float]]:
    """Local gradient peaks at least `min_contrast` grey levels per pixel strong, as (sub-pixel index, gradient)."""
    a = np.abs(grad)
    edges = []
    for i in range(1, len(a) - 1):
        if a[i] < min_contrast or not (a[i] >= a[i - 1] and a[i] > a[i + 1]):
            continue
        if polarity == POLARITY[1] and grad[i] <= 0 or polarity == POLARITY[2] and grad[i] >= 0:
            continue
        den = a[i - 1] - 2 * a[i] + a[i + 1]
        off = 0.5 * (a[i - 1] - a[i + 1]) / den if den != 0 else 0.0
        edges.append((i + max(-0.5, min(0.5, off)), float(grad[i])))
    return edges


@register
class Caliper(Node):
    roi = ("line", "line")
    title = "Kenar Ölç (Kumpas)"
    category = CAT
    description = ("Dijital kumpas: çizdiğiniz ölçüm çizgisi boyunca parlaklık geçişlerini (kenarları) piksel altı "
                   "hassasiyetle bulur ve iki kenar arasındaki mesafeyi verir (px; mm başına piksel verilirse mm). "
                   "Parça genişliği, delik çapı, boşluk, kapak yüksekliği ölçümü için. Çizgiyi önizlemede fareyle "
                   "çizin; 'kenar yönü' ile koyudan açığa / açıktan koyuya geçiş seçilir.")
    inputs = [("image", IMG), ("px_per_mm", NUM)]
    outputs = [("image", IMG), ("distance", NUM), ("distance_px", NUM), ("found", BOOL), ("edge1", NUM),
               ("edge2", NUM), ("edges", NUM), ("points", ANY)]
    params = [Param("line", "str", "50,240,590,240", "Ölçüm çizgisi x1,y1,x2,y2"),
              Param("width", "int", 5, "Çizgi kalınlığı (ortalama, px)", 1, 201),
              Param("edge1", "choice", POLARITY[0], "1. kenar yönü", choices=POLARITY),
              Param("edge2", "choice", POLARITY[0], "2. kenar yönü", choices=POLARITY),
              Param("pairing", "choice", PAIRING[0], "Hangi kenarlar", choices=PAIRING),
              Param("contrast", "float", 15.0, "En az kenar kontrastı (gri/px)", 0.5, 255, 1),
              Param("sigma", "float", 1.0, "Yumuşatma (σ)", 0.0, 20.0, 0.5),
              Param("px_per_mm", "float", 0.0, "mm başına piksel (0 = piksel; giriş bağlıysa o)", 0, 10000, 0.01)]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = _nums(self.v("line"), 4, "Ölçüm çizgisi (x1,y1,x2,y2)")
        H, W = gray.shape
        if not all(-1 <= x <= W and -1 <= y <= H for x, y in ((x1, y1), (x2, y2))):
            raise ValueError(f"Ölçüm çizgisi ({self.v('line')}) resmin dışında (resim {W}x{H})")
        t, _, grad, u = edge_profile(gray, (x1, y1), (x2, y2), int(self.v("width")), float(self.v("sigma")))
        thr = float(self.v("contrast"))
        e1s = find_edges(grad, thr, self.v("edge1"))
        e2s = find_edges(grad, thr, self.v("edge2"))
        all_edges = find_edges(grad, thr, POLARITY[0])
        first = second = None
        if e1s:
            first = e1s[0][0]
            later = [e for e, _ in e2s if e > first + 0.5]
            if later:
                second = later[-1] if self.v("pairing") == PAIRING[1] else later[0]
        out = img.copy()
        p0 = np.array([x1, y1])
        cv2.line(out, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), YELLOW, 1)
        ppm = _ppm(inputs, self)
        pts = []
        for e in (first, second):
            if e is not None:
                q = p0 + e * u
                pts.append((round(float(q[0]), 3), round(float(q[1]), 3)))
                nrm = np.array([-u[1], u[0]]) * (int(self.v("width")) / 2 + 6)
                a, b = q - nrm, q + nrm
                cv2.line(out, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), RED, 2)
        if first is None or second is None:
            put_text(out, "Kenar bulunamadı" if first is None else "2. kenar bulunamadı", (10, 30), 0.7, RED, 2)
            return {"image": out, "distance": 0.0, "distance_px": 0.0, "found": False,
                    "edge1": round(first, 3) if first is not None else 0.0, "edge2": 0.0,
                    "edges": len(all_edges), "points": pts}
        dist_px = float(second - first)
        dist = dist_px / ppm if ppm else dist_px
        unit = "mm" if ppm else "px"
        mid = p0 + (first + second) / 2 * u
        put_text(out, f"{dist:.2f} {unit}", (int(mid[0]) + 6, max(15, int(mid[1]) - 12)), 0.7, GREEN, 2)
        return {"image": out, "distance": round(dist, 4), "distance_px": round(dist_px, 3), "found": True,
                "edge1": round(first, 3), "edge2": round(second, 3), "edges": len(all_edges), "points": pts}


# --- 4. blob / defect -----------------------------------------------------------------------------------------------

BLOB_POL = ["koyu leke", "açık leke", "ikisi"]
BLOB_BG = ["yerel arka plan (medyan)", "tüm yüzeyin ortalaması"]


@register
class BlobDefects(Node):
    roi = ("rect", "roi")
    title = "Leke / Kusur Bul (Blob)"
    category = CAT
    description = ("Düz/tek renkli yüzeyde (sac, plastik, kumaş, cam, kağıt) çevresinden koyu veya açık lekeleri, "
                   "çizik ve delikleri bulur. Yerel arka plan yöntemi ışık eşit değilse de çalışır: arka plan boyutu "
                   "en büyük lekenin en az iki katı olmalı. Alan ve yuvarlaklıkla süzülür; hiç leke yoksa ok doğru.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("mask", IMG), ("detections", DET), ("count", NUM), ("total_area", NUM),
               ("largest_area", NUM), ("ok", BOOL)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("polarity", "choice", BLOB_POL[0], "Leke türü", choices=BLOB_POL),
              Param("background", "choice", BLOB_BG[0], "Arka plan", choices=BLOB_BG),
              Param("bg_size", "int", 31, "Arka plan boyutu (px, tek sayı)", 3, 255, 2),
              Param("threshold", "int", 30, "Fark eşiği (gri seviye)", 1, 255),
              Param("min_area", "int", 10, "En küçük leke alanı (px)", 1, 10_000_000),
              Param("max_area", "int", 1_000_000, "En büyük leke alanı (px)", 1, 100_000_000),
              Param("min_circularity", "float", 0.0, "En az yuvarlaklık (0 = hepsi, 1 = daire)", 0, 1, 0.05)]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        gray = cv2.cvtColor(img[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.int16)
        if self.v("background") == BLOB_BG[0]:
            k = int(self.v("bg_size")) | 1
            k = min(k, (min(w, h) - 1) | 1) if min(w, h) > 3 else 3
            small = 1
            src = gray
            if k > 61:                               # a big median is slow: take it on a smaller copy
                small = int(math.ceil(k / 61))
                src = cv2.resize(gray, (max(1, w // small), max(1, h // small)), interpolation=cv2.INTER_AREA)
                k = max(3, (k // small) | 1)
            bg = cv2.medianBlur(src, k)
            if small > 1:
                bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
            bg = bg.astype(np.int16)
        else:
            bg = np.full_like(g, int(round(float(np.median(gray)))))
        diff = g - bg
        thr = int(self.v("threshold"))
        pol = self.v("polarity")
        if pol == BLOB_POL[0]:
            mask = diff <= -thr
        elif pol == BLOB_POL[1]:
            mask = diff >= thr
        else:
            mask = np.abs(diff) >= thr
        mask = mask.astype(np.uint8) * 255
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        lo, hi = sorted((int(self.v("min_area")), int(self.v("max_area"))))
        min_circ = float(self.v("min_circularity"))
        out = img.copy()
        if self.v("roi").strip():
            cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), YELLOW, 1)
        full_mask = np.zeros(img.shape[:2], np.uint8)
        dets = []
        for i in range(1, n):
            bx, by, bw, bh, area = (int(v) for v in stats[i])
            if not lo <= area <= hi:
                continue
            blob = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(np.pad(blob, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = [c + np.array([bx - 1, by - 1], np.int32) for c in cnts]
            per = cv2.arcLength(max(cnts, key=len), True) if cnts else 0.0
            circ = min(1.0, 4 * math.pi * area / (per * per)) if per > 0 else 1.0
            if circ < min_circ:
                continue
            full_mask[y0 + by:y0 + by + bh, x0 + bx:x0 + bx + bw] |= blob
            cv2.drawContours(out[y0:y0 + h, x0:x0 + w], cnts, -1, RED, 2)
            dets.append({"box": (bx + x0, by + y0, bw, bh), "score": 1.0, "class_id": 0,
                         "label": f"leke {area}px", "area": float(area), "circularity": round(circ, 3),
                         "center": (float(cents[i][0]) + x0, float(cents[i][1]) + y0)})
        dets.sort(key=lambda d: -d["area"])
        total = sum(d["area"] for d in dets)
        put_text(out, f"Leke: {len(dets)}  alan: {int(total)}px", (10, 30), 0.7, RED if dets else GREEN, 2)
        return {"image": out, "mask": full_mask, "detections": dets, "count": len(dets),
                "total_area": total, "largest_area": dets[0]["area"] if dets else 0.0, "ok": not dets}


# --- 5. fill level --------------------------------------------------------------------------------------------------

FILL_METHOD = ["en güçlü yatay geçiş", "parlaklık eşiği"]
LIQUID = ["fark etmez", "sıvı daha koyu", "sıvı daha açık"]


@register
class FillLevel(Node):
    roi = ("rect", "roi")
    title = "Doluluk Seviyesi"
    category = CAT
    description = ("Şişe, bardak veya kaptaki sıvı seviyesini bulur: bölgeyi kabın içine çizin. Satır satır "
                   "parlaklık profilinden sıvı çizgisini (en güçlü yatay geçiş veya parlaklık eşiği) bulur; seviyeyi "
                   "bölgenin altından yüzde ve piksel olarak verir. Min-max yüzde arasındaysa ok doğru (eksik / "
                   "fazla dolum kontrolü).")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("level", NUM), ("level_px", NUM), ("line_y", NUM), ("ok", BOOL), ("found", BOOL)]
    params = [Param("roi", "str", "", "Kap bölgesi x,y,w,h (boş = tümü)"),
              Param("method", "choice", FILL_METHOD[0], "Yöntem", choices=FILL_METHOD),
              Param("liquid", "choice", LIQUID[1], "Sıvı", choices=LIQUID),
              Param("threshold", "int", 100, "Parlaklık eşiği (eşik yöntemi)", 0, 255),
              Param("contrast", "float", 10.0, "En az geçiş kontrastı (gri/px)", 0.5, 255, 1),
              Param("min_level", "float", 70.0, "En az doluluk (%)", 0, 100, 1),
              Param("max_level", "float", 95.0, "En çok doluluk (%)", 0, 100, 1)]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        if h < 4:
            raise ValueError("Kap bölgesi çok kısa (en az 4 piksel yükseklik)")
        gray = cv2.cvtColor(img[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY).astype(np.float64)
        prof = gray.mean(axis=1)                     # one value per row, top to bottom
        liquid = self.v("liquid")
        line = None                                  # sub-pixel row (in the ROI) of the liquid surface
        if self.v("method") == FILL_METHOD[0]:
            r = 2
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / 1.0) ** 2)
            sm = np.convolve(np.pad(prof, r, mode="edge"), k / k.sum(), mode="valid")
            grad = np.gradient(sm)                   # > 0: gets brighter going down
            if liquid == LIQUID[1]:
                score = -grad                        # dark liquid below a bright empty part
            elif liquid == LIQUID[2]:
                score = grad
            else:
                score = np.abs(grad)
            i = int(np.argmax(score))
            if score[i] >= float(self.v("contrast")):
                off = 0.0
                if 0 < i < len(score) - 1:
                    den = score[i - 1] - 2 * score[i] + score[i + 1]
                    off = 0.5 * (score[i - 1] - score[i + 1]) / den if den != 0 else 0.0
                line = i + 0.5 + max(-0.5, min(0.5, off))    # gradient at row i sits between rows; +0.5 = edge
        else:
            thr = float(self.v("threshold"))
            is_liquid = prof > thr if liquid == LIQUID[2] else prof < thr
            if is_liquid[-1]:                        # walk up from the bottom while the rows are liquid
                top = h - 1
                while top > 0 and is_liquid[top - 1]:
                    top -= 1
                line = float(top)
        out = img.copy()
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), YELLOW, 1)
        if line is None:
            put_text(out, "Sıvı seviyesi bulunamadı", (10, 30), 0.7, RED, 2)
            return {"image": out, "level": 0.0, "level_px": 0.0, "line_y": 0.0, "ok": False, "found": False}
        level_px = h - line
        level = 100.0 * level_px / h
        lo, hi = sorted((float(self.v("min_level")), float(self.v("max_level"))))
        ok = lo <= level <= hi
        yy = int(round(y0 + line))
        cv2.line(out, (x0, yy), (x0 + w - 1, yy), GREEN if ok else RED, 2)
        put_text(out, f"%{level:.1f}", (x0 + 4, max(15, yy - 6)), 0.7, GREEN if ok else RED, 2)
        return {"image": out, "level": round(level, 2), "level_px": round(level_px, 2), "line_y": round(y0 + line, 2),
                "ok": ok, "found": True}


# --- 6. colour area -------------------------------------------------------------------------------------------------

@register
class ColorArea(Node):
    roi = ("rect", "roi")
    title = "Renk Alanı %"
    category = CAT
    description = ("Bölgedeki piksellerin yüzde kaçının seçilen HSV renk aralığında olduğunu ölçer: tutkal/boya "
                   "kaplaması, pas, yanık, olgunluk, etiket varlığı. Ton min > max ise aralık 180'den başa sarar "
                   "(kırmızı için ör. 170 → 10). Yüzde min-max arasındaysa ok doğru.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("mask", IMG), ("percent", NUM), ("pixels", NUM), ("ok", BOOL)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"), *hsv_params(),
              Param("min_pct", "float", 50.0, "En az (%)", 0, 100, 0.5),
              Param("max_pct", "float", 100.0, "En çok (%)", 0, 100, 0.5)]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        m = _hsv_of(self, img[y0:y0 + h, x0:x0 + w])
        n = int(cv2.countNonZero(m))
        pct = 100.0 * n / (w * h)
        lo, hi = sorted((float(self.v("min_pct")), float(self.v("max_pct"))))
        ok = lo <= pct <= hi
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y0:y0 + h, x0:x0 + w] = m
        out = img.copy()
        sub = out[y0:y0 + h, x0:x0 + w]
        sub[m > 0] = (0.5 * sub[m > 0] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), GREEN if ok else RED, 2)
        put_text(out, f"%{pct:.1f}", (x0 + 4, max(15, y0 - 6) if y0 > 20 else y0 + 22), 0.7, GREEN if ok else RED, 2)
        return {"image": out, "mask": mask, "percent": round(pct, 3), "pixels": n, "ok": ok}


# --- 7. histogram compare -------------------------------------------------------------------------------------------

HIST_METHOD = ["korelasyon", "Bhattacharyya", "kesişim"]
HIST_SPACE = ["renk (HSV)", "gri"]


@register
class HistogramCompare(Node):
    roi = ("rect", "roi")
    title = "Histogram Karşılaştır"
    category = CAT
    description = ("Görüntünün (veya bölgesinin) renk/parlaklık dağılımını referans resimle karşılaştırır; 0..1 "
                   "benzerlik verir (1 = aynı). Konumdan bağımsızdır: doğru ürün/renk/etiket mi, yanlış parti mi "
                   "kontrolü. Referans resim görüntüyle aynı boyutta ise bölge ona da uygulanır, değilse referansın "
                   "tamamı kullanılır.")
    inputs = [("image", IMG), ("reference", IMG)]
    outputs = [("similarity", NUM), ("ok", BOOL), ("image", IMG)]
    params = [Param("reference", "path", "", "Referans resim (giriş bağlıysa o)"),
              Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("method", "choice", HIST_METHOD[1], "Yöntem", choices=HIST_METHOD),
              Param("space", "choice", HIST_SPACE[0], "Renk uzayı", choices=HIST_SPACE),
              Param("bins", "int", 32, "Kutu sayısı", 4, 256),
              Param("min_similarity", "float", 0.8, "Uygun için en az benzerlik", 0, 1, 0.01)]

    def _hist(self, bgr):
        bins = int(self.v("bins"))
        if self.v("space") == HIST_SPACE[1]:
            hist = cv2.calcHist([cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)], [0], None, [bins], [0, 256])
        else:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [min(bins, 180), bins], [0, 180, 0, 256])
        hist = hist.astype(np.float32)
        s = float(hist.sum())
        return hist / s if s > 0 else hist

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        ref = _ref_image(self, inputs)
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        part = img[y0:y0 + h, x0:x0 + w]
        if ref.shape[:2] == img.shape[:2]:
            ref = ref[y0:y0 + h, x0:x0 + w]
        a, b = self._hist(part), self._hist(ref)
        method = self.v("method")
        if method == HIST_METHOD[0]:
            sim = max(0.0, float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL)))
        elif method == HIST_METHOD[1]:
            sim = 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
        else:
            sim = float(cv2.compareHist(a, b, cv2.HISTCMP_INTERSECT))     # both sum to 1
        sim = min(1.0, max(0.0, sim if math.isfinite(sim) else 0.0))
        ok = sim >= float(self.v("min_similarity"))
        out = img.copy()
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), GREEN if ok else RED, 2)
        put_text(out, f"Benzerlik {sim:.3f}", (10, 30), 0.7, GREEN if ok else RED, 2)
        return {"similarity": round(sim, 4), "ok": ok, "image": out}


# --- 8. auto align --------------------------------------------------------------------------------------------------

ALIGN_METHOD = ["ECC (küçük kayma, hassas)", "özellik (ORB, büyük kayma/dönme)"]
ALIGN_MOTION = ["kaydırma + dönme", "sadece kaydırma", "afin (eğme/ölçek dahil)"]


@register
class AutoAlign(Node):
    title = "Otomatik Hizala"
    category = CAT
    description = ("Görüntüyü referans resme hizalar: parça biraz kaysa veya dönse de çıkıştaki görüntüde hep aynı "
                   "yerde durur, böylece sonraki bloklardaki sabit bölgeler (ROI) doğru yere düşer. Kayma (dx, dy, "
                   "piksel) ve dönme açısını (°, saat yönü tersi +) da verir. ECC küçük kaymalarda çok hassastır; "
                   "büyük kayma veya dönmede 'özellik' yöntemini seçin.")
    inputs = [("image", IMG), ("reference", IMG)]
    outputs = [("image", IMG), ("dx", NUM), ("dy", NUM), ("angle", NUM), ("ok", BOOL), ("score", NUM)]
    params = [Param("reference", "path", "", "Referans resim (giriş bağlıysa o)"),
              Param("method", "choice", ALIGN_METHOD[0], "Yöntem", choices=ALIGN_METHOD),
              Param("motion", "choice", ALIGN_MOTION[0], "Hareket", choices=ALIGN_MOTION),
              Param("iterations", "int", 100, "En çok adım (ECC)", 5, 5000),
              Param("work_scale", "float", 0.5, "Hız için küçültme (1 = tam boy)", 0.1, 1.0, 0.05),
              Param("min_score", "float", 0.5, "En az uyum (ECC: 0..1, özellik: eşleşme oranı)", 0, 1, 0.05),
              Param("max_shift", "float", 0.0, "En çok kayma (px, 0 = sınırsız)", 0, 100000, 1)]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        ref = _ref_image(self, inputs)
        hr, wr = ref.shape[:2]
        ecc = self.v("method") == ALIGN_METHOD[0]
        if ecc and img.shape[:2] != (hr, wr):           # ECC compares pixel by pixel: same size needed
            img = cv2.resize(img, (wr, hr), interpolation=cv2.INTER_AREA)
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        try:
            if ecc:
                W, score = self._ecc(g_ref, g_img)
            else:
                W, score = self._feature(g_ref, g_img)
        except cv2.error:
            W, score = None, 0.0
        if W is None:
            return {"image": cv2.resize(img, (wr, hr)) if img.shape[:2] != (hr, wr) else img,
                    "dx": 0.0, "dy": 0.0, "angle": 0.0, "ok": False, "score": round(float(score), 4)}
        aligned = cv2.warpAffine(img, W, (wr, hr), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
        c = np.array([wr / 2.0, hr / 2.0, 1.0])
        moved = W @ c                                 # where the reference centre is in the image
        dx, dy = float(moved[0] - c[0]), float(moved[1] - c[1])
        angle = _wrap180(math.degrees(math.atan2(-W[1, 0], W[0, 0])))
        ok = score >= float(self.v("min_score"))
        ms = float(self.v("max_shift"))
        if ms > 0 and math.hypot(dx, dy) > ms:
            ok = False
        return {"image": aligned, "dx": round(dx, 3), "dy": round(dy, 3), "angle": round(angle, 3), "ok": ok,
                "score": round(float(score), 4)}

    def _ecc(self, g_ref, g_img):
        motion = self.v("motion")
        mtype = {ALIGN_MOTION[0]: cv2.MOTION_EUCLIDEAN, ALIGN_MOTION[1]: cv2.MOTION_TRANSLATION,
                 ALIGN_MOTION[2]: cv2.MOTION_AFFINE}[motion]
        s = float(self.v("work_scale"))
        if min(g_ref.shape) * s < 32:
            s = min(1.0, 32.0 / min(g_ref.shape))
        fr = g_ref.astype(np.float32)
        fi = g_img.astype(np.float32)
        # a large shift is out of ECC's reach: start from the phase-correlation shift
        win = cv2.createHanningWindow((fr.shape[1], fr.shape[0]), cv2.CV_32F)
        (sx, sy), _ = cv2.phaseCorrelate(fr, fi, win)
        W = np.array([[1, 0, sx], [0, 1, sy]], np.float32)
        its = int(self.v("iterations"))
        score = 0.0
        for scale in ((s, 1.0) if s < 0.999 else (1.0,)):
            a = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else fr
            b = cv2.resize(fi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else fi
            Ws = W.copy()
            Ws[:, 2] *= scale
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, its if scale < 1 else max(5, its // 2), 1e-5)
            score, Ws = cv2.findTransformECC(a, b, Ws, mtype, crit, None, 5)
            W = Ws.copy()
            W[:, 2] /= scale
        return W.astype(np.float64), float(score)

    def _feature(self, g_ref, g_img):
        kp_r, des_r = _features(g_ref, "ORB", 3000)
        kp_i, des_i = _features(g_img, "ORB", 3000)
        good = _ratio_matches(des_r, des_i, 0.75)
        if len(good) < 6:
            return None, 0.0
        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_i[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        motion = self.v("motion")
        if motion == ALIGN_MOTION[2]:
            M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        else:
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inl is None or int(inl.sum()) < 6:
            return None, 0.0
        if motion == ALIGN_MOTION[0]:                # rigid: drop the scale the partial affine may have picked up
            sc = math.hypot(M[0, 0], M[1, 0])
            if sc > 0:
                M[:, :2] /= sc
        elif motion == ALIGN_MOTION[1]:
            d = (dst - src).reshape(-1, 2)[inl.ravel().astype(bool)]
            M = np.array([[1, 0, np.median(d[:, 0])], [0, 1, np.median(d[:, 1])]], np.float64)
        return M.astype(np.float64), float(inl.sum()) / len(good)


# --- 9. tray / grid check -------------------------------------------------------------------------------------------

GRID_METHOD = ["dolu = koyu", "dolu = açık", "dolu = renk aralığında (HSV)"]


@register
class GridInspect(Node):
    roi = ("rect", "roi")
    title = "Sayma Izgarası (Tepsi Kontrol)"
    category = CAT
    description = ("Bölgeyi satır × sütun hücrelere böler ve her hücrenin dolu mu boş mu olduğuna bakar: tepsi, "
                   "blister, yumurta kolisi, kasa, şişe kasası kontrolü. Boş hücre numaralarını (soldan sağa, "
                   "yukarıdan aşağı, 1'den başlar) metin olarak verir; hepsi doluysa ok doğru. Hücre kenar payı "
                   "bölmelerin duvarlarını yok saymak içindir.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("ok", BOOL), ("empty_count", NUM), ("filled_count", NUM), ("empty_cells", TEXT),
               ("cells", ANY), ("detections", DET)]
    params = [Param("roi", "str", "", "Tepsi bölgesi x,y,w,h (boş = tümü)"),
              Param("rows", "int", 3, "Satır", 1, 200), Param("cols", "int", 4, "Sütun", 1, 200),
              Param("method", "choice", GRID_METHOD[0], "Dolu hücre", choices=GRID_METHOD),
              Param("threshold", "int", 128, "Parlaklık eşiği (hücre ortalaması)", 0, 255),
              Param("fill_pct", "float", 30.0, "Renkli hücre için en az renk (%)", 0, 100, 1),
              Param("margin", "float", 15.0, "Hücre kenar payı (%)", 0, 45, 1), *hsv_params()]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        rows, cols = int(self.v("rows")), int(self.v("cols"))
        if w < cols or h < rows:
            raise ValueError(f"Bölge ({w}x{h}) {rows}x{cols} hücre için çok küçük")
        method = self.v("method")
        region = img[y0:y0 + h, x0:x0 + w]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        cmask = _hsv_of(self, region) if method == GRID_METHOD[2] else None
        thr, need = float(self.v("threshold")), float(self.v("fill_pct"))
        mg = float(self.v("margin")) / 100.0
        out = img.copy()
        cells, empty, dets = [], [], []
        for r in range(rows):
            for c in range(cols):
                cx0, cx1 = int(round(c * w / cols)), int(round((c + 1) * w / cols))
                cy0, cy1 = int(round(r * h / rows)), int(round((r + 1) * h / rows))
                mx, my = int((cx1 - cx0) * mg), int((cy1 - cy0) * mg)
                ix0, ix1, iy0, iy1 = cx0 + mx, max(cx0 + mx + 1, cx1 - mx), cy0 + my, max(cy0 + my + 1, cy1 - my)
                if cmask is not None:
                    value = 100.0 * float(np.count_nonzero(cmask[iy0:iy1, ix0:ix1])) / ((ix1 - ix0) * (iy1 - iy0))
                    filled = value >= need
                else:
                    value = float(gray[iy0:iy1, ix0:ix1].mean())
                    filled = value < thr if method == GRID_METHOD[0] else value > thr
                n = r * cols + c + 1
                cells.append(bool(filled))
                color = GREEN if filled else RED
                p0, p1 = (x0 + cx0, y0 + cy0), (x0 + cx1 - 1, y0 + cy1 - 1)
                cv2.rectangle(out, p0, p1, color, 2 if filled else 3)
                if not filled:
                    empty.append(n)
                    dets.append({"box": (x0 + cx0, y0 + cy0, cx1 - cx0, cy1 - cy0), "score": 1.0, "class_id": 0,
                                 "label": f"boş {n}", "cell": n, "row": r + 1, "col": c + 1,
                                 "value": round(value, 2)})
                put_text(out, str(n), (p0[0] + 4, p0[1] + 18), 0.5, color, 1)
        ok = not empty
        put_text(out, "TAMAM" if ok else f"Boş: {len(empty)}", (10, 30), 0.8, GREEN if ok else RED, 2)
        return {"image": out, "ok": ok, "empty_count": len(empty), "filled_count": len(cells) - len(empty),
                "empty_cells": ", ".join(str(n) for n in empty), "cells": cells, "detections": dets}
