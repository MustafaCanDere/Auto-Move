"""Measuring in millimetres: lens correction, scale from a reference, bird's-eye plane view, distances."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ._text import put_text
from .. import calibration as cal
from ..core import Node, Param, PortType, register

CAT = "2b · Kalite Kontrol ve Ölçüm"
IMG, NUM, BOOL, DET = PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.DETECTIONS
ARUCO_DICTS = ["4X4_50", "5X5_100", "6X6_250", "7X7_1000"]


def _bgr(img):
    if img is None:
        raise ValueError("Görüntü girişi bağlı değil")
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(img[:, :, :3]) if img.shape[2] >= 3 else cv2.cvtColor(img[:, :, 0],
                                                                                      cv2.COLOR_GRAY2BGR)


@register
class LensUndistort(Node):
    setup_params = ("file",)
    title = "Lens Bozulmasını Düzelt"
    category = CAT
    description = ("Kamera kalibrasyon dosyasıyla (Araçlar → Kamera Kalibrasyonu) lensin bükmesini düzeltir: düz "
                   "kenarlar düz, ölçüler resmin kenarında da doğru olur. Ölçümden önce kullanın.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [Param("file", "path", "", "Kalibrasyon dosyası (boş = varsayılan)"),
              Param("keep_all", "float", 0.0, "Kenarları koru (0 = kırp, 1 = hepsi)", 0.0, 1.0, 0.1)]

    def setup(self):
        path = self.v("file").strip() or cal.default_path()
        self.state.update(cal=cal.Calibration.load(path), maps={})

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        h, w = img.shape[:2]
        key = (w, h, float(self.v("keep_all")))
        maps = self.state["maps"]
        if key not in maps:                      # maps are made once per picture size: remapping is then fast
            c = self.state["cal"]
            cw, ch = c.image_size
            if abs(w / h - cw / ch) > 0.01:      # a cropped mode (16:9 of a 4:3 sensor) has another lens geometry
                raise ValueError(f"Görüntü {w}x{h}, kalibrasyon {cw}x{ch}: en-boy oranı farklı. Kamerayı bu "
                                 "çözünürlükte yeniden kalibre edin")
            K = c.K_for(w, h)
            newK, _ = cv2.getOptimalNewCameraMatrix(K, c.D, (w, h), key[2], (w, h))
            maps.clear()
            maps[key] = cv2.initUndistortRectifyMap(K, c.D, None, newK, (w, h), cv2.CV_16SC2)
        m1, m2 = maps[key]
        return {"image": cv2.remap(img, m1, m2, cv2.INTER_LINEAR)}


class _Locked(Node):
    """Finds a reference once and keeps using it ("kilitle"), so the part can cover the reference afterwards.
    Only the found points are kept: sizes are worked out again every frame, so a changed size setting counts at
    once, and a change of the settings the points depend on (board size, marker) makes it search again."""
    ref_params: tuple = ()

    def setup(self):
        self.state["found"] = None

    def _reference(self, img):
        raise NotImplementedError

    def reference(self, img):
        """(points, seen_now): seen_now is False when locked points are reused."""
        sig = tuple(self.v(p) for p in self.ref_params)
        found = self.state.get("found")
        if found is not None and found[0] != sig:
            found = self.state["found"] = None
        if self.v("lock") and found is not None:
            return found[1], False
        ref = self._reference(img)
        if ref is not None:
            self.state["found"] = (sig, ref)
            return ref, True
        return None, False                       # not locked: no reference in this picture means no scale


@register
class ScaleFromReference(_Locked):
    title = "Ölçek Bul (mm)"
    category = CAT
    description = ("Görüntüdeki satranç tahtası karesinden veya ArUco işaretçisinden 'mm başına piksel' değerini "
                   "bulur. Çıkışı Nesne Ölç / Mesafe Ölç bloklarının px_per_mm girişine bağlayın. 'Kilitle' açıksa "
                   "bir kez bulduktan sonra referans kaldırılabilir (kamera ve masa yerinden oynamamalı).")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("px_per_mm", NUM), ("found", BOOL)]
    ref_params = ("method", "cols", "rows", "dict", "marker_id")
    params = [Param("method", "choice", "Satranç tahtası", "Referans", choices=["Satranç tahtası", "ArUco işaretçi"]),
              Param("cols", "int", 9, "İç köşe (yatay)", 2, 50), Param("rows", "int", 6, "İç köşe (dikey)", 2, 50),
              Param("square_mm", "float", 25.0, "Kare / işaretçi kenarı (mm)", 0.01, 10000, 0.5),
              Param("dict", "choice", "4X4_50", "ArUco sözlüğü", choices=ARUCO_DICTS),
              Param("marker_id", "int", -1, "ArUco no (-1 = herhangi)", -1, 999),
              Param("lock", "bool", True, "Bulunca kilitle")]

    def _reference(self, img):
        if self.v("method").startswith("ArUco"):
            _, q = cal.aruco_px_per_mm(img, 1.0, self.v("dict"), self.v("marker_id"))
            return q
        return cal.find_board(img, self.v("cols"), self.v("rows"))

    def _ppm(self, pts) -> float:
        if self.v("method").startswith("ArUco"):
            q = np.asarray(pts).reshape(4, 2)
            return float(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1).mean() / self.v("square_mm"))
        return cal.board_px_per_mm(pts, self.v("cols"), self.v("rows"), self.v("square_mm"))

    def process(self, inputs):
        img = _bgr(inputs["image"])
        pts, seen = self.reference(img)
        out = img.copy()
        if pts is None:
            put_text(out, "Referans bulunamadı", (10, 30), 0.8, (0, 0, 255), 2)
            return {"image": out, "px_per_mm": 0.0, "found": False}
        ppm = self._ppm(pts)
        if seen:
            for x, y in np.asarray(pts).reshape(-1, 2):
                cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
        put_text(out, f"{ppm:.3f} piksel/mm" + ("" if seen else " (kilitli)"), (10, 30), 0.8, (0, 255, 0), 2)
        return {"image": out, "px_per_mm": round(ppm, 5), "found": True}


@register
class PlaneRectify(_Locked):
    title = "Kuşbakışı Düzelt (mm)"
    category = CAT
    description = ("Masadaki satranç tahtasıyla eğik bakan kameranın görüntüsünü tam üstten bakılmış hale getirir; "
                   "çıkışta her yerde 1 mm aynı sayıda pikseldir (ör. 10 piksel/mm). Kamera açılı olsa da doğru "
                   "ölçüm için. Tahtayı bir kez gösterin, 'Kilitle' ile sonra kaldırabilirsiniz.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("px_per_mm", NUM), ("found", BOOL)]
    ref_params = ("cols", "rows")
    params = [Param("cols", "int", 9, "İç köşe (yatay)", 2, 50), Param("rows", "int", 6, "İç köşe (dikey)", 2, 50),
              Param("square_mm", "float", 25.0, "Kare kenarı (mm)", 0.01, 10000, 0.5),
              Param("out_ppm", "float", 4.0, "Çıkış çözünürlüğü (piksel/mm)", 0.1, 100, 0.5),
              Param("width_mm", "float", 400.0, "Çıkış genişliği (mm)", 1, 100000, 10),
              Param("height_mm", "float", 300.0, "Çıkış yüksekliği (mm)", 1, 100000, 10),
              Param("margin_mm", "float", 50.0, "Tahtanın sol-üst boşluğu (mm)", 0, 100000, 5),
              Param("lock", "bool", True, "Bulunca kilitle")]

    def _reference(self, img):
        return cal.find_board(img, self.v("cols"), self.v("rows"))

    def process(self, inputs):
        img = _bgr(inputs["image"])
        corners, seen = self.reference(img)
        ppm = float(self.v("out_ppm"))
        if corners is None:
            out = img.copy()
            put_text(out, "Satranç tahtası bulunamadı", (10, 30), 0.8, (0, 0, 255), 2)
            return {"image": out, "px_per_mm": 0.0, "found": False}
        H = cal.plane_homography(corners, self.v("cols"), self.v("rows"), self.v("square_mm"), ppm,
                                 self.v("margin_mm"))
        w, h = int(round(self.v("width_mm") * ppm)), int(round(self.v("height_mm") * ppm))
        if w * h > 64_000_000:
            raise ValueError("Çıkış çok büyük: genişlik × yükseklik × çözünürlüğü küçültün")
        return {"image": cv2.warpPerspective(img, H, (w, h)), "px_per_mm": ppm, "found": True}


def _center(d):
    if d.get("center"):
        return float(d["center"][0]), float(d["center"][1])
    x, y, w, h = d["box"]
    return x + w / 2.0, y + h / 2.0


@register
class DistanceMeasure(Node):
    title = "Mesafe Ölç"
    category = CAT
    description = ("İki tespitin (ör. iki delik, iki işaretçi) merkezleri arasındaki mesafeyi ölçer; px_per_mm "
                   "verilirse milimetre. Tespit sırası soldan sağa (veya en büyük iki) seçilebilir. Sonucu "
                   "Tolerans Kontrolü'ne bağlayarak uygun / değil kararı verin.")
    inputs = [("image", IMG), ("detections", DET), ("px_per_mm", NUM)]
    outputs = [("image", IMG), ("distance", NUM), ("dx", NUM), ("dy", NUM), ("found", BOOL)]
    params = [Param("pick", "choice", "Soldan ilk iki", "Hangi iki tespit",
                    choices=["Soldan ilk iki", "Yukarıdan ilk iki", "En büyük iki", "En emin iki"]),
              Param("px_per_mm", "float", 0.0, "mm başına piksel (0 = piksel; giriş bağlıysa o)", 0, 10000, 0.01)]

    def process(self, inputs):
        dets = list(inputs["detections"] or [])
        ppm = inputs.get("px_per_mm") or self.v("px_per_mm") or 0.0
        img = inputs["image"]
        out = _bgr(img).copy() if img is not None else None
        if len(dets) < 2:
            return {"image": out, "distance": 0.0, "dx": 0.0, "dy": 0.0, "found": False}
        pick = self.v("pick")
        if pick.startswith("Soldan"):
            dets.sort(key=lambda d: _center(d)[0])
        elif pick.startswith("Yukarıdan"):
            dets.sort(key=lambda d: _center(d)[1])
        elif pick.startswith("En büyük"):
            dets.sort(key=lambda d: -d["box"][2] * d["box"][3])
        else:
            dets.sort(key=lambda d: -d.get("score", 0))
        (x1, y1), (x2, y2) = _center(dets[0]), _center(dets[1])
        scale = ppm if ppm > 0 else 1.0
        unit = "mm" if ppm > 0 else "px"
        dist, dx, dy = math.hypot(x2 - x1, y2 - y1) / scale, abs(x2 - x1) / scale, abs(y2 - y1) / scale
        if out is not None:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.line(out, p1, p2, (0, 255, 255), 2)
            for p in (p1, p2):
                cv2.circle(out, p, 4, (0, 255, 255), -1)
            put_text(out, f"{dist:.2f} {unit}", ((p1[0] + p2[0]) // 2 + 6, (p1[1] + p2[1]) // 2 - 6), 0.6,
                     (0, 255, 255), 2)
        return {"image": out, "distance": round(dist, 3), "dx": round(dx, 3), "dy": round(dy, 3), "found": True}
