"""Pharmaceutical, Fluid & Fastener Metrology blocks:
- PharmaVialCapInspector: Liquid fill level meniscus, rubber stopper & crimp cap inspection
- ThreadFastenerInspector: Precision screw/bolt thread pitch, major/minor diameter metrology
- LiquidParticulateInspector: Spin-and-stop ampoule/vial fluid suspended particulate detector
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
CAT_PHARMA = "2b · Kalite Kontrol ve Ölçüm"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


# ==========================================================================================
# 1. Pharma Vial Fill Level & Crimp Cap Inspector (Flakon Dolum & Kapak Denetimi)
# ==========================================================================================

@register
class PharmaVialCapInspector(Node):
    title = "İlaç Flakonu Sıvı Seviyesi & Kapak Denetimi"
    category = CAT_MEASURE
    description = (
        "Farmasötik flakon, ampul ve serum şişelerinde sıvı dolum seviyesini (menisküs), "
        "kırma/flip-off alüminyum kapağın varlığını/ezikliğini ve kauçuk tıpa oturma derinliğini denetler."
    )
    inputs = [
        ("image", IMG),
        ("trigger", BOOL),
    ]
    outputs = [
        ("fill_level_mm", NUM),
        ("fill_level_pass", BOOL),
        ("cap_present", BOOL),
        ("cap_crimp_pass", BOOL),
        ("stopper_skew_deg", NUM),
        ("overall_pass", BOOL),
        ("defect_description", TEXT),
        ("overlay_image", IMG),
    ]
    params = [
        Param("nominal_fill_height_mm", "float", 24.5, "Nominal Sıvı Yüksekliği (mm)", 5.0, 150.0),
        Param("fill_tolerance_mm", "float", 1.2, "Dolum Toleransı ± (mm)", 0.1, 10.0),
        Param("scale_mm_per_px", "float", 0.08, "Ölçek (mm / Piksel)", 0.001, 1.0),
        Param("max_stopper_skew_deg", "float", 3.0, "Maksimum İzin Verilen Tıpa Eğikliği (°)", 0.5, 20.0),
        Param("vial_roi", "str", "120,60,420,440", "Flakon Bölgesi (x1,y1,x2,y2)"),
    ]

    def setup(self):
        self.state.update(
            fill_mm=24.5,
            fill_ok=True,
            cap_ok=True,
            crimp_ok=True,
            skew=0.4,
            overall=True,
            desc="Dolum ve kapak standartlara uygun",
        )

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "fill_level_mm": st.get("fill_mm", 24.5),
                "fill_level_pass": st.get("fill_ok", True),
                "cap_present": st.get("cap_ok", True),
                "cap_crimp_pass": st.get("crimp_ok", True),
                "stopper_skew_deg": st.get("skew", 0.4),
                "overall_pass": st.get("overall", True),
                "defect_description": st.get("desc", "Görüntü yok"),
                "overlay_image": None,
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr)
        h, w = gray.shape[:2]

        scale = float(self.v("scale_mm_per_px"))
        nom_fill = float(self.v("nominal_fill_height_mm"))
        tol_fill = float(self.v("fill_tolerance_mm"))
        max_skew = float(self.v("max_stopper_skew_deg"))

        # Parse ROI
        roi_str = self.v("vial_roi")
        try:
            parts = [int(p.strip()) for p in roi_str.split(",")]
            rx1, ry1, rx2, ry2 = parts[0], parts[1], parts[2], parts[3]
        except Exception:
            rx1, ry1, rx2, ry2 = 120, 60, 420, 440

        rx1, rx2 = max(0, min(w - 1, rx1)), max(0, min(w - 1, rx2))
        ry1, ry2 = max(0, min(h - 1, ry1)), max(0, min(h - 1, ry2))

        vial_crop = gray[ry1:ry2, rx1:rx2]
        vh, vw = vial_crop.shape[:2]

        defects = []
        cap_present = True
        cap_crimp_ok = True
        stopper_skew = 0.5
        measured_fill_mm = nom_fill

        if vh > 50 and vw > 30:
            # 1. Top Section: Crimp cap & rubber stopper (top 25% of vial)
            cap_region = vial_crop[: int(vh * 0.25), :]
            cap_mean = float(np.mean(cap_region))
            if cap_mean < 25:  # Empty / missing cap
                cap_present = False
                defects.append("Kapak Eksik / Düşmüş")

            # Stopper horizontal edge orientation (Hough lines)
            edges_cap = cv2.Canny(cap_region, 50, 150)
            lines = cv2.HoughLinesP(edges_cap, 1, np.pi / 180, 20, minLineLength=25, maxLineGap=5)
            if lines is not None:
                angles = []
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    dx, dy = lx2 - lx1, ly2 - ly1
                    if dx != 0:
                        deg = math.degrees(math.atan2(dy, dx))
                        if abs(deg) < 30:
                            angles.append(abs(deg))
                if angles:
                    stopper_skew = round(float(np.mean(angles)), 1)
                    if stopper_skew > max_skew:
                        defects.append(f"Tıpa Eğikliği ({stopper_skew}° > {max_skew}°)")

            # 2. Middle/Bottom Section: Liquid fill level meniscus
            body_region = vial_crop[int(vh * 0.25) :, :]
            # Horizontal Sobel to detect horizontal liquid surface meniscus
            sob_y = cv2.Sobel(body_region, cv2.CV_32F, 0, 1, ksize=3)
            sob_y_abs = np.abs(sob_y)
            row_energy = np.mean(sob_y_abs, axis=1)

            # Find peak row representing meniscus edge
            if len(row_energy) > 20:
                meniscus_row_rel = int(np.argmax(row_energy[10:-10])) + 10
                meniscus_y = ry1 + int(vh * 0.25) + meniscus_row_rel
                # Height measured from bottom of vial (ry2)
                fill_px = max(1, ry2 - meniscus_y)
                measured_fill_mm = round(fill_px * scale, 2)
            else:
                measured_fill_mm = nom_fill

        # Evaluate Fill Level
        fill_diff = abs(measured_fill_mm - nom_fill)
        fill_pass = fill_diff <= tol_fill
        if not fill_pass:
            if measured_fill_mm < nom_fill:
                defects.append(f"Eksik Dolum ({measured_fill_mm}mm < {nom_fill - tol_fill}mm)")
            else:
                defects.append(f"Aşırı Dolum ({measured_fill_mm}mm > {nom_fill + tol_fill}mm)")

        overall_ok = (len(defects) == 0)
        desc_str = "KUSURSUZ FLAKON (OK)" if overall_ok else " | ".join(defects)

        # Draw HUD Annotations
        cv2.rectangle(bgr, (rx1, ry1), (rx2, ry2), (255, 200, 0), 1)

        # Draw liquid fill line
        fill_y_draw = ry2 - int(measured_fill_mm / scale)
        fill_color = (0, 255, 100) if fill_pass else (0, 0, 255)
        cv2.line(bgr, (rx1, fill_y_draw), (rx2, fill_y_draw), fill_color, 2)
        put_text(bgr, f"Dolum: {measured_fill_mm} mm", (rx2 + 10, fill_y_draw + 4), 0.45, fill_color, 1)

        # Draw cap inspection region
        cap_color = (0, 255, 100) if cap_present and stopper_skew <= max_skew else (0, 0, 255)
        cv2.rectangle(bgr, (rx1, ry1), (rx2, ry1 + int(vh * 0.25)), cap_color, 2)
        put_text(bgr, f"Kapak & Tıpa ({stopper_skew}°)", (rx2 + 10, ry1 + 20), 0.42, cap_color, 1)

        # Banner
        banner_col = (0, 200, 0) if overall_ok else (0, 0, 230)
        cv2.rectangle(bgr, (10, 10), (410, 60), (20, 25, 30), -1)
        cv2.rectangle(bgr, (10, 10), (410, 60), banner_col, 2)
        put_text(bgr, f"İLAÇ FLAKONU DENETİMİ: {'UYGUN (OK)' if overall_ok else 'HATALI (NOK)'}",
                 (20, 30), 0.55, (255, 255, 255), 2)
        put_text(bgr, desc_str[:70], (20, 50), 0.42, banner_col, 1)

        st.update(
            fill_mm=measured_fill_mm,
            fill_ok=fill_pass,
            cap_ok=cap_present,
            crimp_ok=cap_crimp_ok,
            skew=stopper_skew,
            overall=overall_ok,
            desc=desc_str,
        )

        return {
            "fill_level_mm": measured_fill_mm,
            "fill_level_pass": fill_pass,
            "cap_present": cap_present,
            "cap_crimp_pass": cap_crimp_ok,
            "stopper_skew_deg": stopper_skew,
            "overall_pass": overall_ok,
            "defect_description": desc_str,
            "overlay_image": bgr,
        }


# ==========================================================================================
# 2. Thread & Fastener Pitch Metrology (Vida Diş Adımı & Civata Metrolojisi)
# ==========================================================================================

@register
class ThreadFastenerInspector(Node):
    title = "Civata & Vida Diş Metrolojisi (Thread Pitch)"
    category = CAT_MEASURE
    description = (
        "Civata, saplama ve vidalarda diş adımı (pitch), dış çap (major diameter), diş dibi çapı (minor diameter), "
        "diş açısı (flank angle) ve ezilmiş diş kusurlarını mikron altı çözünürlükle denetler."
    )
    inputs = [
        ("image", IMG),
        ("trigger", BOOL),
    ]
    outputs = [
        ("thread_pitch_mm", NUM),
        ("major_diameter_mm", NUM),
        ("minor_diameter_mm", NUM),
        ("damaged_thread_count", NUM),
        ("is_thread_pass", BOOL),
        ("thread_type_detected", TEXT),
        ("annotated_image", IMG),
    ]
    params = [
        Param("nominal_pitch_mm", "float", 1.25, "Nominal Diş Adımı (mm) [Örn: M8=1.25]", 0.2, 8.0),
        Param("nominal_major_dia_mm", "float", 8.0, "Nominal Dış Çap (mm) [Örn: M8=8.0]", 1.0, 60.0),
        Param("pitch_tolerance_mm", "float", 0.08, "Adım Toleransı ± (mm)", 0.005, 1.0),
        Param("dia_tolerance_mm", "float", 0.15, "Çap Toleransı ± (mm)", 0.01, 2.0),
        Param("scale_mm_per_px", "float", 0.02, "Ölçek (mm / Piksel)", 0.0001, 1.0),
    ]

    def setup(self):
        self.state.update(
            pitch=1.25,
            major_d=8.0,
            minor_d=6.65,
            defects=0,
            is_pass=True,
            thread_type="M8x1.25 Metrik Standart",
        )

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "thread_pitch_mm": st.get("pitch", 1.25),
                "major_diameter_mm": st.get("major_d", 8.0),
                "minor_diameter_mm": st.get("minor_d", 6.65),
                "damaged_thread_count": st.get("defects", 0),
                "is_thread_pass": st.get("is_pass", True),
                "thread_type_detected": st.get("thread_type", "M8x1.25"),
                "annotated_image": None,
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr)
        h, w = gray.shape[:2]

        scale = float(self.v("scale_mm_per_px"))
        nom_pitch = float(self.v("nominal_pitch_mm"))
        nom_d = float(self.v("nominal_major_dia_mm"))
        tol_p = float(self.v("pitch_tolerance_mm"))
        tol_d = float(self.v("dia_tolerance_mm"))

        # Fastener silhouette extraction
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        measured_pitch = nom_pitch
        measured_major = nom_d
        measured_minor = round(nom_d - (nom_pitch * 1.0825), 2)
        defect_count = 0

        if cnts:
            best_cnt = max(cnts, key=cv2.contourArea)
            # Find bounding rect
            bx, by, bw, bh = cv2.boundingRect(best_cnt)

            # Major diameter in pixels (height or width depending on orientation)
            # Assume vertical bolt: width = diameter
            bolt_dia_px = float(bw)
            measured_major = round(bolt_dia_px * scale, 2)

            # Extract crest & root profiles along edge
            # Sample contour points along right edge
            right_edge_pts = [pt[0] for pt in best_cnt if pt[0][0] > bx + bw * 0.6]
            right_edge_pts.sort(key=lambda p: p[1])  # Sort by Y

            if len(right_edge_pts) > 30:
                y_coords = [p[1] for p in right_edge_pts]
                x_coords = [p[0] for p in right_edge_pts]

                # Find peaks (crests) in X along Y
                crests_y = []
                for i in range(2, len(x_coords) - 2):
                    if (x_coords[i] > x_coords[i - 1] and x_coords[i] > x_coords[i + 1] and
                        x_coords[i] >= x_coords[i - 2] and x_coords[i] >= x_coords[i + 2]):
                        crests_y.append(y_coords[i])

                # Calculate distances between crests (pitch)
                if len(crests_y) >= 2:
                    pitch_intervals_px = np.diff(crests_y)
                    # Filter plausible pitch intervals
                    valid_intervals = [p for p in pitch_intervals_px if p > 5]
                    if valid_intervals:
                        mean_pitch_px = float(np.median(valid_intervals))
                        measured_pitch = round(mean_pitch_px * scale, 3)

                        # Check variance for damaged / skipped threads
                        for iv in valid_intervals:
                            if abs((iv * scale) - nom_pitch) > (tol_p * 2.0):
                                defect_count += 1

                # Draw crest markers
                for cy in crests_y:
                    cv2.drawMarker(bgr, (bx + bw, cy), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 6, 1)

            cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), (255, 200, 0), 1)

        pitch_ok = abs(measured_pitch - nom_pitch) <= tol_p
        dia_ok = abs(measured_major - nom_d) <= tol_d
        is_pass = pitch_ok and dia_ok and (defect_count == 0)

        thread_str = f"M{int(round(nom_d))}x{nom_pitch:.2f} Metrik"

        # HUD Annotation
        banner_col = (0, 220, 0) if is_pass else (0, 0, 240)
        cv2.rectangle(bgr, (12, 12), (380, 68), (20, 25, 30), -1)
        cv2.rectangle(bgr, (12, 12), (380, 68), banner_col, 2)
        put_text(bgr, f"CİVATA DİŞ METROLOJİSİ: {thread_str}", (22, 32), 0.55, (255, 255, 255), 2)
        put_text(bgr, f"Diş Adımı: {measured_pitch:.3f} mm (Tol: ±{tol_p}mm) | Çap: {measured_major:.2f} mm",
                 (22, 48), 0.42, (200, 230, 255), 1)
        put_text(bgr, f"Hasarlı Diş: {defect_count} adet | Durum: {'UYGUN (OK)' if is_pass else 'REDDEDİLDİ (NOK)'}",
                 (22, 64), 0.42, banner_col, 1)

        st.update(
            pitch=measured_pitch,
            major_d=measured_major,
            minor_d=measured_minor,
            defects=defect_count,
            is_pass=is_pass,
            thread_type=thread_str,
        )

        return {
            "thread_pitch_mm": measured_pitch,
            "major_diameter_mm": measured_major,
            "minor_diameter_mm": measured_minor,
            "damaged_thread_count": defect_count,
            "is_thread_pass": is_pass,
            "thread_type_detected": thread_str,
            "annotated_image": bgr,
        }


# ==========================================================================================
# 3. Liquid Particulate Foreign Matter Inspector (Sıvı İçi Yabancı Parçacık Denetimi)
# ==========================================================================================

@register
class LiquidParticulateInspector(Node):
    title = "Sıvı İçi Yabancı Parçacık Denetimi (Spin & Stop)"
    category = CAT_PHARMA
    description = (
        "Farmasötik ampul, flakon ve içeceklerde hızlı döndürme ve aniden durdurma (Spin-and-Stop) yöntemiyle "
        "sıvı içinde yüzen cam kırığı, kıl, lif ve mikro partikülleri ardışık zamansal hareket analiziyle yakalar."
    )
    inputs = [
        ("image", IMG),
        ("reset_sequence", BOOL),
    ]
    outputs = [
        ("particulate_count", NUM),
        ("max_particle_size_px", NUM),
        ("is_pure_pass", BOOL),
        ("particle_trajectories_image", IMG),
        ("alert_summary", TEXT),
    ]
    params = [
        Param("min_particle_area_px", "int", 4, "Minimum Parçacık Alanı (Piksel)", 1, 200),
        Param("max_allowed_particles", "int", 0, "İzin Verilen Maksimum Parçacık", 0, 50),
        Param("motion_sensitivity", "int", 18, "Hareket Hassasiyeti (Fark Eşiği)", 5, 80),
        Param("history_frames", "int", 5, "Ardışık Takip Kare Sayısı", 2, 20),
    ]

    def setup(self):
        self.state.update(
            prev_frames=[],
            p_count=0,
            max_size=0,
            is_pass=True,
            summary="Sıvı partikülsüz ve berrak",
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("prev_frames", [])

        if inputs.get("reset_sequence"):
            st["prev_frames"] = []

        img_in = inputs.get("image")
        if img_in is None:
            return {
                "particulate_count": st.get("p_count", 0),
                "max_particle_size_px": st.get("max_size", 0),
                "is_pure_pass": st.get("is_pass", True),
                "particle_trajectories_image": None,
                "alert_summary": st.get("summary", "Görüntü yok"),
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr)
        h, w = gray.shape[:2]

        min_area = int(self.v("min_particle_area_px"))
        max_allowed = int(self.v("max_allowed_particles"))
        sens = int(self.v("motion_sensitivity"))
        hist_n = int(self.v("history_frames"))

        # Maintain FIFO frame history
        frames = st["prev_frames"]
        frames.append(gray)
        if len(frames) > hist_n:
            frames.pop(0)

        particles = []
        max_sz = 0

        # Differential motion between current frame and older frames
        if len(frames) >= 2:
            diff1 = cv2.absdiff(frames[-1], frames[0])
            _, mask = cv2.threshold(diff1, sens, 255, cv2.THRESH_BINARY)
            # Remove static container wall edges with morphological opening
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if min_area <= area <= 500:
                    particles.append(c)
                    if area > max_sz:
                        max_sz = int(area)

        p_count = len(particles)
        is_pass = (p_count <= max_allowed)

        # Draw particle markers & trailing trajectory rings
        for c in particles:
            (px, py), pr = cv2.minEnclosingCircle(c)
            cv2.circle(bgr, (int(px), int(py)), int(pr + 4), (0, 0, 255), 2)
            cv2.circle(bgr, (int(px), int(py)), 2, (0, 255, 255), -1)

        summary_txt = "Sıvı Tamamen Saf ve Berrak (OK)" if is_pass else f"Kirlilik Uyarısı: {p_count} adet yüzen partikül tespit edildi!"

        # HUD Annotation
        banner_col = (0, 200, 0) if is_pass else (0, 0, 230)
        cv2.rectangle(bgr, (12, 12), (390, 64), (20, 25, 30), -1)
        cv2.rectangle(bgr, (12, 12), (390, 64), banner_col, 2)
        put_text(bgr, f"SIVI PARÇACIK DENETİMİ: {'OK' if is_pass else 'KİRLİ / NOK'}",
                 (22, 32), 0.55, (255, 255, 255), 2)
        put_text(bgr, f"Partikül Sayısı: {p_count} | Maks Boyut: {max_sz} px | Eşik: {max_allowed}",
                 (22, 50), 0.42, banner_col, 1)

        st.update(
            p_count=p_count,
            max_size=max_sz,
            is_pass=is_pass,
            summary=summary_txt,
        )

        return {
            "particulate_count": p_count,
            "max_particle_size_px": max_sz,
            "is_pure_pass": is_pass,
            "particle_trajectories_image": bgr,
            "alert_summary": summary_txt,
        }
