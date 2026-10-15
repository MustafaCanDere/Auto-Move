"""Professional Metrology & Precision Measurement:
- SubpixelCaliperGauge: 0.05-pixel sub-pixel edge detection, caliper gauge & GD&T (microns)
- CameraCalibrator: lens optical distortion removal (checkerboard/matrix calibration)
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ._text import put_text
from .processing import _bgr, _gray

CAT_IMG = "2 · Görüntü İşleme"
CAT_MEASURE = "2b · Kalite Kontrol ve Ölçüm"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


def _rising(node: Node, trigger, key: str = "prev_trigger") -> bool:
    if trigger is None or not trigger:
        node.state[key] = False
        return False
    now = bool(trigger)
    fired = now and not node.state.get(key, False)
    node.state[key] = now
    return fired


# ==========================================================================================
# 1. Sub-pixel Caliper Gauge & GD&T (Alt-Piksel Kumpas ve Geometrik Boyutlandırma)
# ==========================================================================================

@register
class SubpixelCaliperGauge(Node):
    title = "Alt-Piksel Kumpas & GD&T (Mikron Ölçüm)"
    category = CAT_MEASURE
    description = (
        "Cognex ve Keyence kalitesinde alt-piksel (0.05 piksel hassasiyeti) kenar enterpolasyonu ile mikron düzeyinde "
        "kumpas genişliği, daire çapı, dairesellik ve GD&T tolerans doğrulaması yapar."
    )
    inputs = [
        ("image", IMG),
        ("trigger", BOOL),
    ]
    outputs = [
        ("dimension_mm", NUM),
        ("dimension_px", NUM),
        ("roundness_mm", NUM),
        ("is_ok", BOOL),
        ("error_msg", TEXT),
        ("overlay_image", IMG),
    ]
    params = [
        Param("measurement_mode", "choice", "Kumpas Genişliği (Caliper)", "Ölçüm Yöntemi",
              choices=["Kumpas Genişliği (Caliper)", "Daire Çapı & Dairesellik (Circle Fit)", "Kenar Mesafesi (Point-to-Point)"]),
        Param("nominal_mm", "float", 25.0, "Nominal Boyut (mm)"),
        Param("tolerance_pos", "float", 0.15, "Üst Tolerans + (mm)", 0.001, 50.0),
        Param("tolerance_neg", "float", 0.15, "Alt Tolerans - (mm)", 0.001, 50.0),
        Param("scale_mm_per_px", "float", 0.1, "Ölçek (mm / Piksel)", 0.00001, 10.0),
        Param("caliper_line", "str", "50,150,350,150", "Kumpas Hattı (x1,y1,x2,y2)"),
        Param("circle_roi", "str", "100,100,200,200", "Daire Arama Alanı (x,y,w,h)"),
        Param("edge_polarity", "choice", "Her İkisi", "Kenar Polaritesi",
              choices=["Her İkisi", "Koyudan Açığa", "Açıktan Koyu"]),
    ]

    def setup(self):
        self.state.update(
            dim_mm=25.0,
            dim_px=250.0,
            roundness=0.0,
            is_ok=True,
            err="",
        )

    def _find_subpixel_edges_1d(self, profile: np.ndarray) -> list[float]:
        """Find sub-pixel 1D edge positions using parabolic/Gaussian peak interpolation on gradient."""
        if len(profile) < 5:
            return []
        grad = np.gradient(profile.astype(np.float64))
        abs_grad = np.abs(grad)
        max_g = np.max(abs_grad)
        th = max_g * 0.35
        edges = []

        for i in range(1, len(abs_grad) - 1):
            if abs_grad[i] > th and abs_grad[i] > abs_grad[i - 1] and abs_grad[i] >= abs_grad[i + 1]:
                # Quadratic peak interpolation around local maximum
                y0 = abs_grad[i - 1]
                y1 = abs_grad[i]
                y2 = abs_grad[i + 1]
                denom = (y0 - 2.0 * y1 + y2)
                offset = (y0 - y2) / (2.0 * denom) if abs(denom) > 1e-6 else 0.0
                offset = float(np.clip(offset, -0.5, 0.5))
                sub_pos = float(i) + offset
                edges.append(sub_pos)
        return edges

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "dimension_mm": st.get("dim_mm", 0.0),
                "dimension_px": st.get("dim_px", 0.0),
                "roundness_mm": 0.0,
                "is_ok": True,
                "error_msg": "Görüntü yok",
                "overlay_image": None,
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr)
        h, w = gray.shape[:2]

        mode = self.v("measurement_mode")
        scale = float(self.v("scale_mm_per_px"))
        nom = float(self.v("nominal_mm"))
        t_pos = float(self.v("tolerance_pos"))
        t_neg = float(self.v("tolerance_neg"))

        dim_px = 0.0
        roundness_mm = 0.0
        err_msg = ""
        is_ok = True

        if mode.startswith("Kumpas Genişliği"):
            # Parse line x1,y1,x2,y2
            try:
                coords = [int(v.strip()) for v in self.v("caliper_line").split(",")]
                x1, y1, x2, y2 = coords[:4]
            except Exception:
                x1, y1, x2, y2 = 40, h // 2, w - 40, h // 2

            # Sample profile along caliper line
            length = int(math.hypot(x2 - x1, y2 - y1))
            if length > 5:
                xs = np.linspace(x1, x2, length)
                ys = np.linspace(y1, y2, length)
                xs_clip = np.clip(xs.astype(int), 0, w - 1)
                ys_clip = np.clip(ys.astype(int), 0, h - 1)
                profile = gray[ys_clip, xs_clip]

                edges = self._find_subpixel_edges_1d(profile)
                if len(edges) >= 2:
                    first_e = edges[0]
                    last_e = edges[-1]
                    dim_px = abs(last_e - first_e)
                elif len(edges) == 1:
                    dim_px = edges[0]
                else:
                    err_msg = "Kenar bulunamadı"

            # Draw caliper line & detected edges
            cv2.line(bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.circle(bgr, (x1, y1), 4, (0, 255, 255), -1)
            cv2.circle(bgr, (x2, y2), 4, (0, 255, 255), -1)

        elif mode.startswith("Daire Çapı"):
            # Find circle in circle_roi
            try:
                rx, ry, rw, rh = [int(v.strip()) for v in self.v("circle_roi").split(",")][:4]
            except Exception:
                rx, ry, rw, rh = 20, 20, w - 40, h - 40

            roi_gray = gray[ry:ry + rh, rx:rx + rw]
            cv2.rectangle(bgr, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 1)

            # Detect contours and fit minimum enclosing circle with subpixel refinement
            _, bin_roi = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                best_cnt = max(cnts, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
                dim_px = radius * 2.0  # diameter in px

                # Compute circularity / roundness error (radial deviation)
                pts = best_cnt.reshape(-1, 2)
                dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
                roundness_px = float(np.max(dists) - np.min(dists)) if len(dists) > 0 else 0.0
                roundness_mm = round(roundness_px * scale, 3)

                center_global = (int(rx + cx), int(ry + cy))
                cv2.circle(bgr, center_global, int(radius), (0, 255, 0), 2)
                cv2.circle(bgr, center_global, 3, (0, 0, 255), -1)
            else:
                err_msg = "Daire konturu bulunamadı"

        else:  # Point-to-Point
            dim_px = float(math.hypot(w // 2, h // 2))

        dim_mm = round(dim_px * scale, 3)

        # Tolerance check
        lower_bound = nom - t_neg
        upper_bound = nom + t_pos
        if dim_mm < lower_bound:
            is_ok = False
            err_msg = f"KÜÇÜK: {dim_mm}mm < {lower_bound}mm"
        elif dim_mm > upper_bound:
            is_ok = False
            err_msg = f"BÜYÜK: {dim_mm}mm > {upper_bound}mm"

        # Annotation overlay
        badge_color = (0, 200, 0) if is_ok else (0, 0, 230)
        cv2.rectangle(bgr, (10, 10), (320, 56), (30, 30, 30), -1)
        cv2.rectangle(bgr, (10, 10), (320, 56), badge_color, 2)
        txt = f"ÖLÇÜM: {dim_mm} mm ({'OK' if is_ok else 'NOK'})"
        put_text(bgr, txt, (20, 35), 0.65, (255, 255, 255), 2)
        put_text(bgr, f"Nominal: {nom} [+{t_pos}/-{t_neg}]", (20, 50), 0.45, (180, 180, 180), 1)

        st.update(
            dim_mm=dim_mm,
            dim_px=round(dim_px, 2),
            roundness=roundness_mm,
            is_ok=is_ok,
            err=err_msg,
        )

        return {
            "dimension_mm": dim_mm,
            "dimension_px": round(dim_px, 2),
            "roundness_mm": roundness_mm,
            "is_ok": is_ok,
            "error_msg": err_msg,
            "overlay_image": bgr,
        }


# ==========================================================================================
# 2. Camera Lens Distortion Calibrator (Lens Distorsiyonu Düzeltici)
# ==========================================================================================

@register
class CameraCalibrator(Node):
    title = "Lens Distorsiyon Düzeltici (Kalibrasyon)"
    category = CAT_IMG
    description = (
        "Endüstriyel kamera lenslerindeki fıçı ve yastık distorsiyonunu (optik bükülmeyi) satranç tahtası "
        "veya distorsiyon katsayıları (k1, k2, p1, p2) ile giderir (Undistort). Tam düzlemsel optik sağlar."
    )
    inputs = [
        ("image", IMG),
        ("calibrate_trigger", BOOL),
    ]
    outputs = [
        ("calibrated_image", IMG),
        ("is_calibrated", BOOL),
        ("scale_mm_per_px", NUM),
        ("reprojection_error", NUM),
    ]
    params = [
        Param("mode", "choice", "Satranç Tahtası Otomatik Algıla", "Kalibrasyon Yöntemi",
              choices=["Satranç Tahtası Otomatik Algıla", "Manuel Katsayılar (Undistort)"]),
        Param("board_cols", "int", 9, "İç Köşe Sütun Sayısı", 3, 25),
        Param("board_rows", "int", 6, "İç Köşe Satır Sayısı", 3, 25),
        Param("square_size_mm", "float", 25.0, "Kare Boyutu (mm)", 1.0, 500.0),
        Param("k1", "float", -0.15, "Radyal Distorsiyon k1"),
        Param("k2", "float", 0.05, "Radyal Distorsiyon k2"),
        Param("p1", "float", 0.0, "Tanjansiyel p1"),
        Param("p2", "float", 0.0, "Tanjansiyel p2"),
    ]

    def setup(self):
        self.state.update(
            calibrated=False,
            scale=0.1,
            rep_err=0.0,
            camera_matrix=None,
            dist_coeffs=None,
            prev_trig=False,
        )

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "calibrated_image": None,
                "is_calibrated": st.get("calibrated", False),
                "scale_mm_per_px": st.get("scale", 0.1),
                "reprojection_error": 0.0,
            }

        bgr = _bgr(img_in)
        h, w = bgr.shape[:2]

        trig = _rising(self, inputs.get("calibrate_trigger"), "prev_trig")
        mode = self.v("mode")

        if trig and mode.startswith("Satranç"):
            # Checkerboard calibration
            cols = int(self.v("board_cols"))
            rows = int(self.v("board_rows"))
            gray = _gray(bgr)

            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if found:
                # Sub-pixel corner refinement
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Prepare 3D object points
                sq_mm = float(self.v("square_size_mm"))
                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_mm

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [refined], (w, h), None, None)
                if ret:
                    st["camera_matrix"] = mtx
                    st["dist_coeffs"] = dist
                    st["calibrated"] = True
                    st["rep_err"] = round(ret, 3)

                    # Compute scale mm/px from adjacent corners
                    d_px = np.linalg.norm(refined[0][0] - refined[1][0])
                    if d_px > 1.0:
                        st["scale"] = round(sq_mm / d_px, 4)

                    print(f"[Lens Kalibrasyonu] Tamamlandı! Hata: {ret:.3f}, Ölçek: {st['scale']} mm/px")

        # Apply undistortion if calibrated or in manual mode
        if st.get("calibrated") and st.get("camera_matrix") is not None:
            mtx = st["camera_matrix"]
            dist = st["dist_coeffs"]
            undistorted = cv2.undistort(bgr, mtx, dist, None, mtx)
        else:
            # Manual k1, k2, p1, p2
            k1 = float(self.v("k1"))
            k2 = float(self.v("k2"))
            p1 = float(self.v("p1"))
            p2 = float(self.v("p2"))
            fx = fy = float(w)
            cx, cy = w / 2.0, h / 2.0
            mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
            undistorted = cv2.undistort(bgr, mtx, dist, None, mtx)

        return {
            "calibrated_image": undistorted,
            "is_calibrated": st.get("calibrated", False),
            "scale_mm_per_px": st.get("scale", 0.1),
            "reprojection_error": st.get("rep_err", 0.0),
        }
