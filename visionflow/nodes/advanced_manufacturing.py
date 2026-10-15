"""Advanced Manufacturing & Metrology blocks:
- LaserLine3DProfiler: 3D laser line optical triangulation & height profilometry
- GlueBeadDispenseInspector: Sealant, silicone & adhesive dispense bead continuity and width analyzer
- DeltaESpectrophotometer: ISO/CIE Delta-E 2000 (CIEDE2000) spectrophotometric color difference analyzer
"""
from __future__ import annotations

import json
import math
import time
import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ..core import node as _node
from ._text import put_text
from .processing import _bgr, _gray

CAT_MEASURE = "2b · Kalite Kontrol ve Ölçüm"
CAT_IMG = "2 · Görüntü İşleme"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


# ==========================================================================================
# 1. Laser Line 3D Profiler (3D Lazer Çizgi Triangülasyon & Yükseklik Profili)
# ==========================================================================================

@register
class LaserLine3DProfiler(Node):
    title = "3D Lazer Profilometre (Yükseklik & Düzlemsellik)"
    category = CAT_MEASURE
    description = (
        "Lazer çizgi triangülasyonu ile parçanın yükseklik profilini (Z ekseni) mikron düzeyinde çıkarır. "
        "Basamak yüksekliği, tepe noktası, düzlemsellik sapması ve 3D derinlik haritası hesaplar."
    )
    inputs = [
        ("image", IMG),
        ("trigger", BOOL),
    ]
    outputs = [
        ("profile_height_mm", NUM),
        ("flatness_deviation_mm", NUM),
        ("heightmap_image", IMG),
        ("profile_points_json", TEXT),
        ("is_in_spec", BOOL),
    ]
    params = [
        Param("triangulation_angle_deg", "float", 45.0, "Triangülasyon Açısı (°)", 10.0, 85.0),
        Param("scale_z_mm_per_px", "float", 0.025, "Z Ölçeği (mm / Piksel)", 0.0001, 1.0),
        Param("scale_x_mm_per_px", "float", 0.05, "X Ölçeği (mm / Piksel)", 0.0001, 1.0),
        Param("target_height_mm", "float", 5.0, "Hedef Basamak Yüksekliği (mm)", 0.0, 100.0),
        Param("height_tolerance_mm", "float", 0.4, "Yükseklik Toleransı ± (mm)", 0.01, 10.0),
        Param("peak_detection_method", "choice", "Ağırlık Merkezi (Center of Gravity)", "Tepe Bulma Yöntemi",
              choices=["Ağırlık Merkezi (Center of Gravity)", "Maksimum Yoğunluk (Max Peak)"]),
    ]

    def setup(self):
        self.state.update(
            height=5.0,
            flatness=0.08,
            in_spec=True,
            pts_json="[]",
        )

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "profile_height_mm": st.get("height", 5.0),
                "flatness_deviation_mm": st.get("flatness", 0.08),
                "heightmap_image": None,
                "profile_points_json": st.get("pts_json", "[]"),
                "is_in_spec": st.get("in_spec", True),
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr)
        h, w = gray.shape[:2]

        sz = float(self.v("scale_z_mm_per_px"))
        sx = float(self.v("scale_x_mm_per_px"))
        target_h = float(self.v("target_height_mm"))
        tol_h = float(self.v("height_tolerance_mm"))
        method = self.v("peak_detection_method")

        # Extract laser line peak along each column
        # For industrial laser lines, line is roughly horizontal across the frame with vertical displacement
        profile_z_px = np.zeros(w, dtype=np.float32)
        valid_cols = []

        # Threshold to isolate bright laser line
        laser_thresh = 60
        for col in range(w):
            col_vals = gray[:, col].astype(np.float32)
            mask = col_vals > laser_thresh
            if np.any(mask):
                if "Ağırlık" in method:
                    # Center of Gravity (sub-pixel)
                    y_indices = np.where(mask)[0].astype(np.float32)
                    weights = col_vals[mask]
                    total_w = np.sum(weights)
                    if total_w > 0:
                        cog_y = float(np.sum(y_indices * weights) / total_w)
                        profile_z_px[col] = cog_y
                        valid_cols.append(col)
                else:
                    max_y = float(np.argmax(col_vals))
                    profile_z_px[col] = max_y
                    valid_cols.append(col)

        # Baseline: median of first and last 10% columns
        edge_n = max(5, int(w * 0.1))
        baseline_left = np.median(profile_z_px[:edge_n]) if edge_n > 0 else 0.0
        baseline_right = np.median(profile_z_px[-edge_n:]) if edge_n > 0 else 0.0
        baseline_y = (baseline_left + baseline_right) / 2.0

        # Calculate height profile in mm relative to baseline
        # Inverted Y (smaller Y = higher height in camera)
        profile_z_mm = (baseline_y - profile_z_px) * sz

        # Filter active step height (peak height in central 50%)
        mid_start, mid_end = int(w * 0.25), int(w * 0.75)
        central_z = profile_z_mm[mid_start:mid_end]
        measured_height_mm = float(np.percentile(central_z, 90)) if len(central_z) > 0 else target_h
        measured_height_mm = round(measured_height_mm, 3)

        # Flatness deviation: peak-to-valley range in central plateau
        flatness_dev_mm = round(float(np.ptp(central_z)), 3) if len(central_z) > 0 else 0.05

        is_ok = abs(measured_height_mm - target_h) <= tol_h

        # Render 3D Heightmap Visualization
        # Create false-color height overlay on BGR
        hud_img = bgr.copy()

        # Draw laser detection track
        pts_to_draw = []
        for c in range(0, w, 2):
            y_pt = int(profile_z_px[c])
            if 0 < y_pt < h:
                pts_to_draw.append((c, y_pt))

        for i in range(len(pts_to_draw) - 1):
            cv2.line(hud_img, pts_to_draw[i], pts_to_draw[i + 1], (0, 255, 255), 2)

        # Render mini profile elevation curve at bottom
        curve_h = 90
        cv2.rectangle(hud_img, (10, h - curve_h - 10), (w - 10, h - 10), (15, 20, 25), -1)
        border_col = (0, 220, 0) if is_ok else (0, 0, 240)
        cv2.rectangle(hud_img, (10, h - curve_h - 10), (w - 10, h - 10), border_col, 2)

        # Draw zero line
        zero_y = h - 25
        cv2.line(hud_img, (20, zero_y), (w - 20, zero_y), (80, 80, 80), 1)

        # Plot profile curve
        curve_pts = []
        max_graph_h = max(1.0, target_h * 1.5)
        for c in range(20, w - 20, 4):
            z_val = profile_z_mm[c]
            graph_y = int(zero_y - (z_val / max_graph_h) * (curve_h - 35))
            graph_y = max(h - curve_h, min(zero_y, graph_y))
            curve_pts.append((c, graph_y))

        for i in range(len(curve_pts) - 1):
            cv2.line(hud_img, curve_pts[i], curve_pts[i + 1], (0, 200, 255), 2)

        # Put metrics
        put_text(hud_img, f"3D LAZER PROFiLOMETRE: {measured_height_mm} mm (Hedef: {target_h} mm)",
                 (20, 30), 0.6, (255, 255, 255), 2)
        put_text(hud_img, f"Düzlemsellik Sapması: {flatness_dev_mm} mm | Durum: {'UYGUN (OK)' if is_ok else 'REDDEDiLDi (NOK)'}",
                 (20, 52), 0.45, border_col, 1)

        profile_sample = [round(float(z), 3) for z in profile_z_mm[::10]]

        st.update(
            height=measured_height_mm,
            flatness=flatness_dev_mm,
            in_spec=is_ok,
            pts_json=json.dumps(profile_sample),
        )

        return {
            "profile_height_mm": measured_height_mm,
            "flatness_deviation_mm": flatness_dev_mm,
            "heightmap_image": hud_img,
            "profile_points_json": st["pts_json"],
            "is_in_spec": is_ok,
        }


# ==========================================================================================
# 2. Continuous Glue / Adhesive Bead Dispense Inspector (Conta & Yapıştırıcı Hat Denetimi)
# ==========================================================================================

@register
class GlueBeadDispenseInspector(Node):
    title = "Conta & Yapıştırıcı Hat Denetimi (Glue Bead)"
    category = CAT_MEASURE
    description = (
        "Otomotiv camı, batarya paketi ve motor kapağı sızdırmazlık contası (silikon/mastik) hatlarını denetler. "
        "Yapıştırıcı hattında kopma (gap/break), aşırı taşma, kalınlık incelmesi ve hacim sapmalarını bulur."
    )
    inputs = [
        ("image", IMG),
        ("trigger", BOOL),
    ]
    outputs = [
        ("bead_continuity_pass", BOOL),
        ("avg_width_mm", NUM),
        ("gap_count", NUM),
        ("overflow_count", NUM),
        ("overlay_image", IMG),
        ("inspection_summary", TEXT),
    ]
    params = [
        Param("nominal_width_mm", "float", 4.0, "Nominal Hat Genişliği (mm)", 0.5, 30.0),
        Param("min_width_mm", "float", 2.8, "Minimum İzin Verilen Genişlik (mm)", 0.1, 25.0),
        Param("max_width_mm", "float", 5.5, "Maksimum İzin Verilen Genişlik (mm)", 0.5, 40.0),
        Param("scale_mm_per_px", "float", 0.05, "Ölçek (mm / Piksel)", 0.0001, 2.0),
        Param("min_gap_length_px", "int", 15, "Kopma Kabul Edilecek Min Boşluk (px)", 3, 200),
    ]

    def setup(self):
        self.state.update(
            pass_flag=True,
            avg_w=4.0,
            gaps=0,
            overflows=0,
            summary="Conta hattı sürekliliği kusursuz",
        )

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "bead_continuity_pass": st.get("pass_flag", True),
                "avg_width_mm": st.get("avg_w", 4.0),
                "gap_count": st.get("gaps", 0),
                "overflow_count": st.get("overflows", 0),
                "overlay_image": None,
                "inspection_summary": st.get("summary", "Görüntü yok"),
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr)
        h, w = gray.shape[:2]

        scale = float(self.v("scale_mm_per_px"))
        nom_w = float(self.v("nominal_width_mm"))
        min_w = float(self.v("min_width_mm"))
        max_w = float(self.v("max_width_mm"))
        min_gap_px = int(self.v("min_gap_length_px"))

        # Segment glue bead (usually dark on light or bright on dark)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Invert if background is white
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # Find connected components of glue bead
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_bead_cnts = [c for c in cnts if cv2.contourArea(c) > 150]

        gap_count = 0
        overflow_count = 0
        measured_widths = []

        if not valid_bead_cnts:
            gap_count = 1
            avg_width_mm = 0.0
        else:
            # If the bead is split into multiple pieces where there should be one continuous track:
            if len(valid_bead_cnts) > 1:
                gap_count = len(valid_bead_cnts) - 1

            for cnt in valid_bead_cnts:
                # Estimate local widths using distance transform inside bead
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

                # Distance transform gives half-width (radius) in pixels
                max_radius_px = float(np.max(dist))
                width_mm = max_radius_px * 2.0 * scale
                measured_widths.append(width_mm)

                # Check overflow
                if width_mm > max_w:
                    overflow_count += 1
                    # Draw overflow spot in red
                    cv2.drawContours(bgr, [cnt], -1, (0, 0, 255), 3)
                elif width_mm < min_w:
                    # Draw thin spot in orange
                    cv2.drawContours(bgr, [cnt], -1, (0, 165, 255), 2)
                else:
                    # Draw normal bead contour in neon green
                    cv2.drawContours(bgr, [cnt], -1, (0, 255, 100), 2)

            avg_width_mm = round(float(np.mean(measured_widths)), 2)

        is_pass = (gap_count == 0) and (overflow_count == 0) and (min_w <= avg_width_mm <= max_w)

        reasons = []
        if gap_count > 0:
            reasons.append(f"Kopma/Boşluk Tespiti ({gap_count} adet)")
        if overflow_count > 0:
            reasons.append(f"Aşırı Taşma ({overflow_count} bölge)")
        if avg_width_mm < min_w:
            reasons.append(f"Yetersiz Mastik ({avg_width_mm}mm < {min_w}mm)")
        if avg_width_mm > max_w:
            reasons.append(f"Genişlik Aşımı ({avg_width_mm}mm > {max_w}mm)")

        summary_text = "Conta Sürekliliği ONAYLANDI (OK)" if is_pass else " | ".join(reasons)

        # HUD Overlay
        banner_col = (0, 200, 0) if is_pass else (0, 0, 230)
        cv2.rectangle(bgr, (12, 12), (390, 70), (20, 25, 30), -1)
        cv2.rectangle(bgr, (12, 12), (390, 70), banner_col, 2)
        put_text(bgr, f"YAPIŞTIRICI & CONTA HATTI: {'OK' if is_pass else 'NOK'}", (22, 32), 0.55, (255, 255, 255), 2)
        put_text(bgr, f"Ort. Genişlik: {avg_width_mm} mm (Nom: {nom_w} mm)", (22, 48), 0.42, (200, 230, 255), 1)
        put_text(bgr, f"Kopma: {gap_count} | Taşma: {overflow_count} | Sınır: [{min_w}-{max_w}]mm", (22, 64), 0.42, banner_col, 1)

        st.update(
            pass_flag=is_pass,
            avg_w=avg_width_mm,
            gaps=gap_count,
            overflows=overflow_count,
            summary=summary_text,
        )

        return {
            "bead_continuity_pass": is_pass,
            "avg_width_mm": avg_width_mm,
            "gap_count": gap_count,
            "overflow_count": overflow_count,
            "overlay_image": bgr,
            "inspection_summary": summary_text,
        }


# ==========================================================================================
# 3. ISO/CIE Delta-E 2000 Spectrophotometer (Delta-E 2000 Renk Farkı Analizörü)
# ==========================================================================================

def _rgb_to_lab(bgr_pixel: np.ndarray) -> tuple[float, float, float]:
    """Convert a single BGR uint8 pixel into CIELAB L*, a*, b*."""
    pix = np.uint8([[bgr_pixel]])
    lab = cv2.cvtColor(pix, cv2.COLOR_BGR2Lab)[0, 0]
    # Standard OpenCV scaling: L* [0, 255] -> [0, 100], a* [0, 255] -> [-128, 127], b* [0, 255] -> [-128, 127]
    l_star = float(lab[0]) * 100.0 / 255.0
    a_star = float(lab[1]) - 128.0
    b_star = float(lab[2]) - 128.0
    return l_star, a_star, b_star


def _ciede2000(l1: float, a1: float, b1: float, l2: float, a2: float, b2: float) -> float:
    """Rigorous implementation of CIEDE2000 Delta-E color difference formula."""
    # 1. Calculate C_i, h_i
    c1 = math.sqrt(a1 ** 2 + b1 ** 2)
    c2 = math.sqrt(a2 ** 2 + b2 ** 2)
    c_bar = (c1 + c2) / 2.0

    g = 0.5 * (1.0 - math.sqrt((c_bar ** 7) / (c_bar ** 7 + 25.0 ** 7 + 1e-9)))
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2

    c1_prime = math.sqrt(a1_prime ** 2 + b1 ** 2)
    c2_prime = math.sqrt(a2_prime ** 2 + b2 ** 2)

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360.0
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360.0

    # 2. Calculate Delta L', Delta C', Delta H'
    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime

    if c1_prime * c2_prime == 0:
        delta_h_prime = 0.0
    elif abs(h2_prime - h1_prime) <= 180.0:
        delta_h_prime = h2_prime - h1_prime
    elif h2_prime - h1_prime > 180.0:
        delta_h_prime = (h2_prime - h1_prime) - 360.0
    else:
        delta_h_prime = (h2_prime - h1_prime) + 360.0

    delta_h_cap = 2.0 * math.sqrt(c1_prime * c2_prime) * math.sin(math.radians(delta_h_prime / 2.0))

    # 3. Calculate CIEDE2000 weighting factors
    l_bar_prime = (l1 + l2) / 2.0
    c_bar_prime = (c1_prime + c2_prime) / 2.0

    if c1_prime * c2_prime == 0:
        h_bar_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180.0:
        h_bar_prime = (h1_prime + h2_prime) / 2.0
    elif h1_prime + h2_prime < 360.0:
        h_bar_prime = (h1_prime + h2_prime + 360.0) / 2.0
    else:
        h_bar_prime = (h1_prime + h2_prime - 360.0) / 2.0

    t = (1.0 - 0.17 * math.cos(math.radians(h_bar_prime - 30.0))
         + 0.24 * math.cos(math.radians(2.0 * h_bar_prime))
         + 0.32 * math.cos(math.radians(3.0 * h_bar_prime + 6.0))
         - 0.20 * math.cos(math.radians(4.0 * h_bar_prime - 63.0)))

    delta_theta = 30.0 * math.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))
    r_c = 2.0 * math.sqrt((c_bar_prime ** 7) / (c_bar_prime ** 7 + 25.0 ** 7 + 1e-9))
    s_l = 1.0 + ((0.015 * ((l_bar_prime - 50.0) ** 2)) / math.sqrt(20.0 + ((l_bar_prime - 50.0) ** 2)))
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -math.sin(math.radians(2.0 * delta_theta)) * r_c

    # 4. Final Delta E 00
    de00 = math.sqrt(
        (delta_l_prime / s_l) ** 2 +
        (delta_c_prime / s_c) ** 2 +
        (delta_h_cap / s_h) ** 2 +
        r_t * (delta_c_prime / s_c) * (delta_h_cap / s_h)
    )
    return float(de00)


@register
class DeltaESpectrophotometer(Node):
    title = "Delta-E 2000 Renk Spektrofotometresi"
    category = CAT_MEASURE
    description = (
        "Otomotiv boyası, plastik ve tekstil ürünlerinde ISO/CIE Delta-E 2000 standardına göre "
        "referans renk ile canlı numune arasındaki algısal renk farkını (CIEDE2000) ve L*a*b* değerlerini hesaplar."
    )
    inputs = [
        ("image", IMG),
        ("target_color_image", IMG),
    ]
    outputs = [
        ("delta_e_00", NUM),
        ("l_val", NUM),
        ("a_val", NUM),
        ("b_val", NUM),
        ("is_color_ok", BOOL),
        ("match_grade", TEXT),
        ("color_comparison_image", IMG),
    ]
    params = [
        Param("max_acceptable_delta_e", "float", 2.0, "Maks İzin Verilen Delta-E (00)", 0.1, 15.0),
        Param("roi_center_x", "int", 320, "Ölçüm Bölgesi X (Piksel)", 10, 4000),
        Param("roi_center_y", "int", 240, "Ölçüm Bölgesi Y (Piksel)", 10, 4000),
        Param("roi_radius_px", "int", 30, "Örnekleme Yarıçapı (Piksel)", 5, 200),
        Param("target_hex_color", "str", "#2E5B88", "Referans Renk HEX (Görüntü Yoksa)"),
    ]

    def setup(self):
        self.state.update(
            delta_e=0.8,
            l=50.0,
            a=0.0,
            b=0.0,
            is_ok=True,
            grade="Mükemmel Eşleşme (Gözle Ayırt Edilemez)",
        )

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "delta_e_00": st.get("delta_e", 0.8),
                "l_val": st.get("l", 50.0),
                "a_val": st.get("a", 0.0),
                "b_val": st.get("b", 0.0),
                "is_color_ok": st.get("is_ok", True),
                "match_grade": st.get("grade", "Normal"),
                "color_comparison_image": None,
            }

        bgr = _bgr(img_in).copy()
        h, w = bgr.shape[:2]

        cx = min(w - 1, max(0, int(self.v("roi_center_x"))))
        cy = min(h - 1, max(0, int(self.v("roi_center_y"))))
        rad = int(self.v("roi_radius_px"))
        max_de = float(self.v("max_acceptable_delta_e"))

        # Sample live BGR inside circle ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), rad, 255, -1)
        mean_bgr = cv2.mean(bgr, mask=mask)[:3]
        sample_bgr = np.array(mean_bgr, dtype=np.uint8)

        l1, a1, b1 = _rgb_to_lab(sample_bgr)

        # Target Reference Color
        target_img = inputs.get("target_color_image")
        if target_img is not None:
            ref_bgr_img = _bgr(target_img)
            ref_mean = cv2.mean(ref_bgr_img)[:3]
            ref_bgr = np.array(ref_mean, dtype=np.uint8)
        else:
            hex_str = str(self.v("target_hex_color") or "#2E5B88").strip().lstrip("#")
            if len(hex_str) == 6:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                ref_bgr = np.array([b, g, r], dtype=np.uint8)
            else:
                ref_bgr = np.array([128, 128, 128], dtype=np.uint8)

        l2, a2, b2 = _rgb_to_lab(ref_bgr)

        # Calculate exact CIEDE2000
        de00 = round(_ciede2000(l1, a1, b1, l2, a2, b2), 2)
        is_ok = (de00 <= max_de)

        if de00 < 1.0:
            grade_str = "Mükemmel Eşleşme (Fark Algılanamaz)"
        elif de00 <= 2.0:
            grade_str = "Endüstriyel Kabul Sınırında (Çok Az Fark)"
        elif de00 <= 3.5:
            grade_str = "Fark Edilebilir Renk Sapması (Dikkat)"
        else:
            grade_str = "Hatalı Renk Tonu (Red / NOK)"

        # Render Comparison HUD
        hud = bgr.copy()
        # Draw sampling circle
        cv2.circle(hud, (cx, cy), rad, (0, 255, 255), 2)
        cv2.drawMarker(hud, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

        # Draw Side-by-side Color Swatch on HUD
        swatch_w, swatch_h = 160, 60
        sw_x, sw_y = w - swatch_w - 20, 20
        # Reference swatch (left)
        cv2.rectangle(hud, (sw_x, sw_y), (sw_x + 80, sw_y + swatch_h), [int(c) for c in ref_bgr], -1)
        # Sample swatch (right)
        cv2.rectangle(hud, (sw_x + 80, sw_y), (sw_x + 160, sw_y + swatch_h), [int(c) for c in sample_bgr], -1)
        cv2.rectangle(hud, (sw_x, sw_y), (sw_x + 160, sw_y + swatch_h), (255, 255, 255), 2)
        put_text(hud, "REF", (sw_x + 15, sw_y - 6), 0.4, (200, 200, 200), 1)
        put_text(hud, "NUMUNE", (sw_x + 90, sw_y - 6), 0.4, (200, 200, 200), 1)

        # Banner
        banner_col = (0, 200, 0) if is_ok else (0, 0, 240)
        cv2.rectangle(hud, (12, 12), (360, 72), (20, 25, 30), -1)
        cv2.rectangle(hud, (12, 12), (360, 72), banner_col, 2)
        put_text(hud, f"DELTA-E 00: {de00} (Maks: {max_de})", (22, 32), 0.6, banner_col, 2)
        put_text(hud, f"L*:{l1:.1f} a*:{a1:.1f} b*:{b1:.1f} | Durum: {'OK' if is_ok else 'NOK'}", (22, 48), 0.42, (220, 220, 220), 1)
        put_text(hud, grade_str, (22, 64), 0.4, (180, 220, 255), 1)

        st.update(
            delta_e=de00,
            l=round(l1, 1),
            a=round(a1, 1),
            b=round(b1, 1),
            is_ok=is_ok,
            grade=grade_str,
        )

        return {
            "delta_e_00": de00,
            "l_val": round(l1, 1),
            "a_val": round(a1, 1),
            "b_val": round(b1, 1),
            "is_color_ok": is_ok,
            "match_grade": grade_str,
            "color_comparison_image": hud,
        }
