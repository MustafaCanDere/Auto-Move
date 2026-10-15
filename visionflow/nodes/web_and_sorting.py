"""Web Roll, High-Speed Sorting & Micro-Crack Inspection blocks:
- ConveyorBeltSorter: Encoder-synchronized high-speed conveyor part tracking & pneumatic ejection pulse timing
- MicroCrackSurfaceFilter: Multiscale Hessian matrix ridge & Frangi-based semiconductor/wafer micro-crack filter
- WebRollDefectMapper: Continuous sheet, foil & roll material defect mapping, density analysis and roll grading
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

CAT_OUT = "5 · Çıkış ve Otomasyon"
CAT_IMG = "2 · Görüntü İşleme"
CAT_MEASURE = "2b · Kalite Kontrol ve Ölçüm"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


# ==========================================================================================
# 4. Conveyor Belt Tracking & High-Speed Ejection Sorter (Konveyör Takip & Pnömatik Fırlatıcı)
# ==========================================================================================

@register
class ConveyorBeltSorter(Node):
    title = "Konveyör Takip & Pnömatik Ayıklayıcı (Sorter)"
    category = CAT_OUT
    description = (
        "Yüksek hızlı konveyör bantlarında enkoder veya bant hızına göre hatalı parçanın nozüle varış anını "
        "milisaniye hassasiyetle hesaplar. Pnömatik hava üfleyici veya iticiye (ejector) tam zamanında tetik sinyali üretir."
    )
    inputs = [
        ("part_detected", BOOL),
        ("is_defective", BOOL),
        ("encoder_pulse", NUM),
        ("image", IMG),
    ]
    outputs = [
        ("ejector_blow_signal", BOOL),
        ("active_queue_count", NUM),
        ("total_inspected", NUM),
        ("total_rejected", NUM),
        ("reject_rate_pct", NUM),
        ("conveyor_monitor_image", IMG),
    ]
    params = [
        Param("conveyor_speed_m_per_min", "float", 30.0, "Konveyör Hızı (metre / dakika)", 1.0, 300.0),
        Param("sensor_to_ejector_dist_mm", "float", 450.0, "Kamera - Fırlatıcı Arası Mesafe (mm)", 50.0, 5000.0),
        Param("ejection_pulse_ms", "float", 60.0, "Üfleme / İtme Pals Süresi (ms)", 10.0, 500.0),
        Param("lead_time_compensation_ms", "float", 8.0, "Valf Açılma Gecikme Telafisi (ms)", 0.0, 100.0),
    ]

    def setup(self):
        self.state.update(
            queue=[],  # List of target ejection timestamps
            total_in=0,
            total_rej=0,
            ejector_active=False,
            prev_detect=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("queue", [])
        st.setdefault("total_in", 0)
        st.setdefault("total_rej", 0)

        now = time.time()
        speed_m_min = float(self.v("conveyor_speed_m_per_min"))
        dist_mm = float(self.v("sensor_to_ejector_dist_mm"))
        pulse_duration_sec = float(self.v("ejection_pulse_ms")) / 1000.0
        lead_comp_sec = float(self.v("lead_time_compensation_ms")) / 1000.0

        # Calculate transit time from camera to pneumatic nozzle
        # speed in mm/sec = (speed_m_min * 1000.0) / 60.0
        speed_mm_s = max(1.0, (speed_m_min * 1000.0) / 60.0)
        transit_time_sec = (dist_mm / speed_mm_s) - lead_comp_sec

        # Rising edge of part detection
        part_trig = bool(inputs.get("part_detected"))
        is_defect = bool(inputs.get("is_defective"))
        fired_detect = part_trig and not st.get("prev_detect", False)
        st["prev_detect"] = part_trig

        if fired_detect:
            st["total_in"] += 1
            if is_defect:
                st["total_rej"] += 1
                target_eject_time = now + transit_time_sec
                st["queue"].append(target_eject_time)

        # Check queue for active ejection pulse
        # Purge past events and check if current time is within [eject_time, eject_time + pulse_duration]
        eject_signal = False
        remaining_queue = []
        for eject_time in st["queue"]:
            if eject_time <= now <= (eject_time + pulse_duration_sec):
                eject_signal = True
                remaining_queue.append(eject_time)
            elif eject_time > now:
                remaining_queue.append(eject_time)

        st["queue"] = remaining_queue

        tot_in = st["total_in"]
        tot_rej = st["total_rej"]
        rej_rate = round((tot_rej / tot_in * 100.0) if tot_in > 0 else 0.0, 2)

        # Render Conveyor Queue Visualizer HUD
        img_in = inputs.get("image")
        if img_in is not None:
            mon = _bgr(img_in).copy()
        else:
            mon = np.zeros((360, 520, 3), dtype=np.uint8)
        mh, mw = mon.shape[:2]

        # Draw Conveyor Belt Graphic at bottom
        conv_y = mh - 70
        cv2.rectangle(mon, (20, conv_y), (mw - 20, conv_y + 40), (40, 45, 55), -1)
        cv2.line(mon, (20, conv_y), (mw - 20, conv_y), (100, 110, 130), 2)
        cv2.line(mon, (20, conv_y + 40), (mw - 20, conv_y + 40), (100, 110, 130), 2)

        # Draw Camera Station (Left) & Nozzle Station (Right)
        cv2.rectangle(mon, (40, conv_y - 25), (70, conv_y), (0, 200, 255), -1)
        put_text(mon, "KAMERA", (30, conv_y - 30), 0.38, (0, 200, 255), 1)

        nozzle_color = (0, 0, 255) if eject_signal else (180, 180, 180)
        cv2.rectangle(mon, (mw - 70, conv_y - 25), (mw - 40, conv_y), nozzle_color, -1)
        put_text(mon, "FIRLATICI", (mw - 85, conv_y - 30), 0.38, nozzle_color, 1)

        # Draw parts travelling on conveyor
        for eject_time in st["queue"]:
            time_left = eject_time - now
            progress = 1.0 - max(0.0, min(1.0, time_left / max(1e-3, transit_time_sec)))
            part_x = int(55 + progress * (mw - 110))
            cv2.circle(mon, (part_x, conv_y + 20), 10, (0, 0, 255), -1)
            cv2.circle(mon, (part_x, conv_y + 20), 12, (255, 255, 255), 1)

        # Top Banner
        status_banner_col = (0, 0, 240) if eject_signal else (0, 200, 0)
        cv2.rectangle(mon, (12, 12), (390, 68), (20, 25, 30), -1)
        cv2.rectangle(mon, (12, 12), (390, 68), status_banner_col, 2)
        put_text(mon, f"KONVEYÖR FIRLATICI: {'VALF AKTİF (ÜFLEME)!' if eject_signal else 'BEKLEMEDE'}",
                 (22, 32), 0.55, status_banner_col, 2)
        put_text(mon, f"Hız: {speed_m_min:.1f} m/dk | Mesafe: {dist_mm:.0f} mm | Süre: {transit_time_sec*1000:.0f} ms",
                 (22, 48), 0.42, (220, 220, 220), 1)
        put_text(mon, f"Toplam: {tot_in} | Fire (NOK): {tot_rej} (%{rej_rate}) | Kuyruk: {len(st['queue'])}",
                 (22, 64), 0.42, (180, 220, 255), 1)

        return {
            "ejector_blow_signal": eject_signal,
            "active_queue_count": len(st["queue"]),
            "total_inspected": tot_in,
            "total_rejected": tot_rej,
            "reject_rate_pct": rej_rate,
            "conveyor_monitor_image": mon,
        }


# ==========================================================================================
# 5. Surface Scratch & Micro-Crack Ridge Filter (Mikro Kılcal Çatlak & Çizik Filtresi)
# ==========================================================================================

@register
class MicroCrackSurfaceFilter(Node):
    title = "Mikro Kılcal Çatlak & Çizik Filtresi (Hessian Ridge)"
    category = CAT_IMG
    description = (
        "Yarı iletken gofret (wafer), güneş paneli, seramik ve krank millerinde standart kenar bulucuların kaçırdığı "
        "kılcal mikro çatlak ve çizikleri çok ölçekli Hessian özdeğer ridge (Frangi) analiziyle belirginleştirir."
    )
    inputs = [("image", IMG)]
    outputs = [
        ("crack_detected", BOOL),
        ("max_crack_length_px", NUM),
        ("crack_ridge_image", IMG),
        ("overlay_image", IMG),
        ("defect_severity", TEXT),
    ]
    params = [
        Param("ridge_scale_sigma", "float", 1.8, "Çatlak Genişlik Ölçeği (Sigma)", 0.5, 6.0),
        Param("crack_threshold", "float", 35.0, "Çatlak Hassasiyet Eşiği", 5.0, 150.0),
        Param("min_crack_length_px", "int", 25, "Minimum Çatlak Uzunluğu (Piksel)", 5, 500),
        Param("defect_mode", "choice", "Karanlık Çatlaklar (Dark Fissures)", "Çatlak Tipi",
              choices=["Karanlık Çatlaklar (Dark Fissures)", "Aydınlık Çizikler (Bright Scratches)"]),
    ]

    def setup(self):
        self.state.update(
            detected=False,
            length=0,
            severity="Kusursuz Yüzey",
        )

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "crack_detected": False,
                "max_crack_length_px": 0,
                "crack_ridge_image": None,
                "overlay_image": None,
                "defect_severity": "Görüntü yok",
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr).astype(np.float32)
        h, w = gray.shape[:2]

        sigma = float(self.v("ridge_scale_sigma"))
        thresh = float(self.v("crack_threshold"))
        min_len = int(self.v("min_crack_length_px"))
        is_dark = (self.v("defect_mode") == "Karanlık Çatlaklar (Dark Fissures)")

        if is_dark:
            work_gray = 255.0 - gray
        else:
            work_gray = gray

        # Compute Gaussian smoothed second derivatives (Hessian matrix components)
        k_sz = max(3, int(sigma * 4) | 1)
        smoothed = cv2.GaussianBlur(work_gray, (k_sz, k_sz), sigma)

        dxx = cv2.Sobel(cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3), cv2.CV_32F, 1, 0, ksize=3)
        dyy = cv2.Sobel(cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3), cv2.CV_32F, 0, 1, ksize=3)
        dxy = cv2.Sobel(cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3), cv2.CV_32F, 0, 1, ksize=3)

        # Eigenvalues of 2x2 Hessian:
        # lambda = 0.5 * (dxx + dyy +/- sqrt((dxx - dyy)^2 + 4*dxy^2))
        trace = dxx + dyy
        disc = np.sqrt(np.maximum(0.0, (dxx - dyy) ** 2 + 4.0 * (dxy ** 2)))
        lambda1 = 0.5 * (trace + disc)
        lambda2 = 0.5 * (trace - disc)

        # Ridge response (Frangi-like principal curvature)
        # For dark fissures (inverted to bright), negative curvature in transverse direction -> large positive lambda
        ridge_response = np.maximum(0.0, lambda1)
        max_resp = float(np.max(ridge_response))
        if max_resp > 0:
            ridge_u8 = np.clip((ridge_response / max_resp) * 255.0, 0, 255).astype(np.uint8)
        else:
            ridge_u8 = np.zeros((h, w), dtype=np.uint8)

        # Threshold to binary crack mask
        _, crack_bin = cv2.threshold(ridge_u8, int(thresh), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        crack_bin = cv2.morphologyEx(crack_bin, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(crack_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crack_detected = False
        max_crack_len = 0

        for c in cnts:
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            length = max(rw, rh)
            if length >= min_len:
                crack_detected = True
                if length > max_crack_len:
                    max_crack_len = int(length)
                # Highlight crack in bright red
                cv2.drawContours(bgr, [c], -1, (0, 0, 255), 2)

        if not crack_detected:
            sev_str = "KUSURSUZ YÜZEY (Çatlak Yok)"
        elif max_crack_len > min_len * 2:
            sev_str = f"KRİTİK HATA: Derin Yapısal Çatlak ({max_crack_len}px)"
        else:
            sev_str = f"DİKKAT: Yüzeysel Mikro Kılcal Çizik ({max_crack_len}px)"

        # HUD Annotation
        banner_col = (0, 220, 0) if not crack_detected else (0, 0, 240)
        cv2.rectangle(bgr, (12, 12), (380, 62), (20, 25, 30), -1)
        cv2.rectangle(bgr, (12, 12), (380, 62), banner_col, 2)
        put_text(bgr, f"MİKRO ÇATLAK FİLTRESİ: {'OK' if not crack_detected else 'ÇATLAK VAR / NOK'}",
                 (22, 32), 0.55, (255, 255, 255), 2)
        put_text(bgr, sev_str, (22, 50), 0.42, banner_col, 1)

        ridge_bgr = cv2.applyColorMap(ridge_u8, cv2.COLORMAP_MAGMA)

        st.update(
            detected=crack_detected,
            length=max_crack_len,
            severity=sev_str,
        )

        return {
            "crack_detected": crack_detected,
            "max_crack_length_px": max_crack_len,
            "crack_ridge_image": ridge_bgr,
            "overlay_image": bgr,
            "defect_severity": sev_str,
        }


# ==========================================================================================
# 6. Web & Roll Continuous Material Defect Mapper (Rulo & Bobin Malzeme Hata Haritalayıcı)
# ==========================================================================================

@register
class WebRollDefectMapper(Node):
    title = "Rulo & Bobin Malzeme Hata Haritalayıcı (Web Inspection)"
    category = CAT_MEASURE
    description = (
        "Saç metal rulo, lityum batarya elektrot folyosu, film ve kumaş bobinlerinde sürekli malzeme boyunca "
        "kusurların konumunu (metre & en mm) kaydeder. Bobin kalite sınıfını (Grade A/B/Scrap) ve 2D hata haritasını çıkarır."
    )
    inputs = [
        ("defect_detected", BOOL),
        ("defect_width_pos_mm", NUM),
        ("current_roll_length_m", NUM),
        ("reset_new_roll", BOOL),
        ("image", IMG),
    ]
    outputs = [
        ("roll_quality_grade", TEXT),
        ("total_roll_defects", NUM),
        ("defects_per_100m", NUM),
        ("is_roll_acceptable", BOOL),
        ("roll_defect_map_image", IMG),
        ("defect_positions_json", TEXT),
    ]
    params = [
        Param("nominal_roll_length_m", "float", 500.0, "Toplam Bobin Uzunluğu (Metre)", 10.0, 10000.0),
        Param("roll_width_mm", "float", 1200.0, "Bobin / Şerit Genişliği (mm)", 50.0, 5000.0),
        Param("max_allowed_defects_per_100m", "float", 2.0, "Maks İzin Verilen Hata / 100m", 0.1, 50.0),
        Param("grade_a_threshold", "float", 0.5, "Grade A Hata Eşiği / 100m", 0.0, 10.0),
    ]

    def setup(self):
        self.state.update(
            defects=[],  # [(pos_x_m, pos_y_mm)]
            grade="Grade A (Kusursuz Kalite)",
            acc=True,
            prev_trig=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("defects", [])

        if inputs.get("reset_new_roll"):
            st["defects"] = []

        total_len_m = float(self.v("nominal_roll_length_m"))
        roll_w_mm = float(self.v("roll_width_mm"))
        max_rate = float(self.v("max_allowed_defects_per_100m"))
        grade_a_rate = float(self.v("grade_a_threshold"))

        cur_len_m = float(inputs.get("current_roll_length_m") if inputs.get("current_roll_length_m") is not None else 100.0)

        # Detect new defect event
        trig = bool(inputs.get("defect_detected"))
        fired = trig and not st.get("prev_trig", False)
        st["prev_trig"] = trig

        if fired:
            pos_y = float(inputs.get("defect_width_pos_mm") if inputs.get("defect_width_pos_mm") is not None else (roll_w_mm / 2.0))
            st["defects"].append((cur_len_m, pos_y))

        defect_count = len(st["defects"])
        analyzed_len = max(1.0, cur_len_m)
        defects_per_100 = round((defect_count / analyzed_len) * 100.0, 2)

        # Grade calculation
        if defects_per_100 <= grade_a_rate:
            grade_str = "Grade A (Üstün Kalite)"
            is_acceptable = True
        elif defects_per_100 <= max_rate:
            grade_str = "Grade B (Kabul Edilebilir Standart)"
            is_acceptable = True
        else:
            grade_str = "HURDA / RED (Aşırı Hata Oranı)"
            is_acceptable = False

        # Generate 2D Roll Defect Strip Map Image
        map_w, map_h = 500, 180
        strip_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        strip_img[:] = (25, 30, 38)

        # Draw Roll Boundaries
        cv2.rectangle(strip_img, (30, 30), (map_w - 30, map_h - 40), (60, 70, 85), -1)
        cv2.rectangle(strip_img, (30, 30), (map_w - 30, map_h - 40), (120, 130, 150), 1)

        # Draw defects on the 2D map
        plot_w = map_w - 60
        plot_h = map_h - 70
        for (dx_m, dy_mm) in st["defects"]:
            px = int(30 + (dx_m / max(1.0, total_len_m)) * plot_w)
            py = int(30 + (dy_mm / max(1.0, roll_w_mm)) * plot_h)
            px = max(30, min(map_w - 30, px))
            py = max(30, min(map_h - 40, py))
            cv2.circle(strip_img, (px, py), 4, (0, 0, 255), -1)
            cv2.circle(strip_img, (px, py), 6, (0, 255, 255), 1)

        # Axis labels
        put_text(strip_img, f"0m", (25, map_h - 22), 0.38, (180, 180, 180), 1)
        put_text(strip_img, f"{total_len_m:.0f}m", (map_w - 65, map_h - 22), 0.38, (180, 180, 180), 1)
        put_text(strip_img, f"En: {roll_w_mm:.0f}mm", (30, 22), 0.38, (180, 180, 180), 1)

        # Status text
        banner_col = (0, 220, 0) if is_acceptable else (0, 0, 240)
        put_text(strip_img, f"{grade_str}  |  {defects_per_100}/100m ({defect_count} adet)",
                 (map_w - 330, 22), 0.42, banner_col, 1)

        defects_json = json.dumps([{"meter": round(x, 2), "width_mm": round(y, 2)} for x, y in st["defects"]])

        st.update(
            grade=grade_str,
            acc=is_acceptable,
        )

        return {
            "roll_quality_grade": grade_str,
            "total_roll_defects": defect_count,
            "defects_per_100m": defects_per_100,
            "is_roll_acceptable": is_acceptable,
            "roll_defect_map_image": strip_img,
            "defect_positions_json": defects_json,
        }
