"""Industrial Vision & Advanced Factory Automation blocks:
- LabelComplianceChecker: Pharma & Food expiry date (SKT) and lot number cross-validation
- MultiCameraStitcher: 2-4 camera panoramic image fusion
- ThermalInspector: Hotspot detection, thermal thresholding & overheating alarm
- AutoLightingController: Closed-loop illumination feedback & brightness controller
- VirtualPLCSimulator: Desk testing conveyor sensor & PLC signal generator
"""
from __future__ import annotations

import datetime as dt
import math
import random
import re
import time

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ._text import put_text
from .processing import _bgr, _gray

CAT_IMG = "2 · Görüntü İşleme"
CAT_MEASURE = "2b · Kalite Kontrol ve Ölçüm"
CAT_INPUT = "1 · Giriş"
CAT_OUT = "5 · Çıkış ve Otomasyon"

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
# 1. Label Compliance Checker (SKT, Parti No & Barkod Çapraz Doğrulayıcı)
# ==========================================================================================

@register
class LabelComplianceChecker(Node):
    title = "Etiket & SKT / Parti No Denetçisi"
    category = CAT_MEASURE
    description = (
        "Gıda, ilaç ve kozmetik ambalajlarındaki Son Tüketim Tarihi (SKT), Parti/Lot No ve barkod bilgilerini "
        "çapraz doğrular. Tarihi geçmiş veya barkod ile parti numarası uyuşmayan ürünleri anında tespit eder."
    )
    inputs = [
        ("image", IMG),
        ("ocr_text", TEXT),
        ("barcode_data", TEXT),
        ("trigger", BOOL),
    ]
    outputs = [
        ("is_compliant", BOOL),
        ("is_expired", BOOL),
        ("lot_match", BOOL),
        ("expiry_date", TEXT),
        ("lot_number", TEXT),
        ("status_text", TEXT),
        ("overlay_image", IMG),
    ]
    params = [
        Param("min_shelf_days", "int", 0, "Minimum Kalan Raf Ömrü (Gün)", 0, 3650),
        Param("check_barcode_lot", "bool", True, "Barkod ile Parti No Eşleştir"),
    ]

    def setup(self):
        self.state.update(
            compliant=True,
            expired=False,
            lot_match=True,
            expiry_str="",
            lot_str="",
            status="Bekleniyor",
        )

    def _parse_dates(self, text: str) -> list[dt.date]:
        dates = []
        # Match DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD, MM/YY, MM.YYYY
        patterns = [
            r"\b(\d{2})[./-](\d{2})[./-](\d{4})\b",
            r"\b(\d{4})[./-](\d{2})[./-](\d{2})\b",
            r"\b(\d{2})[./](\d{4})\b",
            r"\b(\d{2})[./](\d{2})\b",
        ]
        for p in patterns:
            for m in re.finditer(p, text):
                g = m.groups()
                try:
                    if len(g) == 3:
                        if len(g[0]) == 4:  # YYYY-MM-DD
                            d = dt.date(int(g[0]), int(g[1]), int(g[2]))
                        else:  # DD.MM.YYYY
                            d = dt.date(int(g[2]), int(g[1]), int(g[0]))
                        dates.append(d)
                    elif len(g) == 2:
                        y = int(g[1])
                        if y < 100:
                            y += 2000
                        d = dt.date(y, int(g[0]), 28)
                        dates.append(d)
                except Exception:
                    pass
        return dates

    def _extract_lot(self, text: str) -> str:
        # Match LOT: 1234, PARTİ: A998, L: 881
        m = re.search(r"(?:LOT|PARTI|PARTİ|BATCH|BN|SERI|SERİ)[\s:#_-]*([A-Z0-9_-]{3,15})", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: look for alphanumeric tokens like L1234, B998
        m2 = re.search(r"\b(L[0-9]{4,8})\b", text, re.IGNORECASE)
        return m2.group(1).strip() if m2 else ""

    def process(self, inputs):
        st = self.state
        raw_text = str(inputs.get("ocr_text") or "")
        barcode = str(inputs.get("barcode_data") or "").strip()

        extracted_dates = self._parse_dates(raw_text)
        lot_num = self._extract_lot(raw_text)

        today = dt.date.today()
        min_days = int(self.v("min_shelf_days"))
        is_expired = False
        expiry_display = ""

        if extracted_dates:
            # Furthest date is usually expiry date
            exp_date = max(extracted_dates)
            expiry_display = exp_date.strftime("%d.%m.%Y")
            delta_days = (exp_date - today).days
            if delta_days < min_days:
                is_expired = True

        # Cross-check lot with barcode
        lot_matches = True
        if bool(self.v("check_barcode_lot")) and barcode and lot_num:
            lot_matches = (lot_num.upper() in barcode.upper()) or (barcode.upper() in lot_num.upper())

        is_compliant = (not is_expired) and lot_matches

        if is_expired:
            status_msg = f"HATALI: SKT Geçmiş ({expiry_display})"
        elif not lot_matches:
            status_msg = f"HATALI: Parti No Barkodla Uyuşmuyor ({lot_num} != {barcode})"
        else:
            status_msg = f"ONAYLI: SKT {expiry_display or 'Belirsiz'}, Lot {lot_num or 'OK'}"

        st.update(
            compliant=is_compliant,
            expired=is_expired,
            lot_match=lot_matches,
            expiry_str=expiry_display,
            lot_str=lot_num,
            status=status_msg,
        )

        # Draw overlay if image provided
        img_in = inputs.get("image")
        out_img = None
        if img_in is not None:
            out_img = _bgr(img_in).copy()
            badge_color = (0, 200, 0) if is_compliant else (0, 0, 220)
            cv2.rectangle(out_img, (10, 10), (out_img.shape[1] - 10, 48), badge_color, -1)
            put_text(out_img, status_msg, (20, 36), 0.7, (255, 255, 255), 2)

        return {
            "is_compliant": is_compliant,
            "is_expired": is_expired,
            "lot_match": lot_matches,
            "expiry_date": expiry_display,
            "lot_number": lot_num,
            "status_text": status_msg,
            "overlay_image": out_img,
        }


# ==========================================================================================
# 2. Multi-Camera Stitcher (Çoklu Kamera Panoramik Birleştirici)
# ==========================================================================================

@register
class MultiCameraStitcher(Node):
    title = "Çoklu Kamera Birleştirici (Stitcher)"
    category = CAT_IMG
    description = (
        "Tek bir kameranın açısına sığmayan uzun veya geniş parçalar için 2, 3 veya 4 kameranın "
        "görüntüsünü yan yana ya da alt alta tek bir panoramik görüntüde birleştirir."
    )
    inputs = [
        ("image_1", IMG),
        ("image_2", IMG),
        ("image_3", IMG),
        ("image_4", IMG),
    ]
    outputs = [
        ("stitched_image", IMG),
        ("success", BOOL),
        ("stitch_width", NUM),
        ("stitch_height", NUM),
    ]
    params = [
        Param("layout", "choice", "Yatay Yan Yana", "Birleştirme Yönü",
              choices=["Yatay Yan Yana", "Dikey Alt Alta", "2x2 Izgara Matrisi"]),
        Param("overlap_px", "int", 0, "Örtüşme Kırpma (Piksel)", 0, 500),
        Param("separator_line", "bool", True, "Ayrım Çizgisi Göster"),
    ]

    def process(self, inputs):
        raw_imgs = [inputs.get(f"image_{i}") for i in range(1, 5)]
        valid_imgs = [_bgr(im) for im in raw_imgs if im is not None]

        if not valid_imgs:
            return {"stitched_image": None, "success": False, "stitch_width": 0, "stitch_height": 0}

        if len(valid_imgs) == 1:
            h, w = valid_imgs[0].shape[:2]
            return {"stitched_image": valid_imgs[0], "success": True, "stitch_width": w, "stitch_height": h}

        layout = self.v("layout")
        overlap = int(self.v("overlap_px"))
        show_line = bool(self.v("separator_line"))

        if layout == "Yatay Yan Yana":
            # Match heights to the minimum height
            min_h = min(im.shape[0] for im in valid_imgs)
            resized = []
            for im in valid_imgs:
                h, w = im.shape[:2]
                if h != min_h:
                    scale = min_h / float(h)
                    im = cv2.resize(im, (int(w * scale), min_h))
                resized.append(im)

            if overlap > 0:
                trimmed = [resized[0]]
                for im in resized[1:]:
                    trimmed.append(im[:, overlap:])
                stitched = np.hstack(trimmed)
            else:
                stitched = np.hstack(resized)

            if show_line:
                curr_x = 0
                for im in resized[:-1]:
                    curr_x += im.shape[1] - overlap
                    cv2.line(stitched, (curr_x, 0), (curr_x, min_h), (0, 255, 255), 2)

        elif layout == "Dikey Alt Alta":
            min_w = min(im.shape[1] for im in valid_imgs)
            resized = []
            for im in valid_imgs:
                h, w = im.shape[:2]
                if w != min_w:
                    scale = min_w / float(w)
                    im = cv2.resize(im, (min_w, int(h * scale)))
                resized.append(im)

            if overlap > 0:
                trimmed = [resized[0]]
                for im in resized[1:]:
                    trimmed.append(im[overlap:, :])
                stitched = np.vstack(trimmed)
            else:
                stitched = np.vstack(resized)

        else:  # 2x2 Grid
            target_w = 480
            target_h = 360
            tiles = [cv2.resize(im, (target_w, target_h)) for im in valid_imgs[:4]]
            while len(tiles) < 4:
                tiles.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
            top_row = np.hstack([tiles[0], tiles[1]])
            bot_row = np.hstack([tiles[2], tiles[3]])
            stitched = np.vstack([top_row, bot_row])

        sh, sw = stitched.shape[:2]
        return {
            "stitched_image": stitched,
            "success": True,
            "stitch_width": sw,
            "stitch_height": sh,
        }


# ==========================================================================================
# 3. Thermal Inspector (Termal Kamera & Sıcaklık / Aşırı Isınma Analitiği)
# ==========================================================================================

@register
class ThermalInspector(Node):
    title = "Termal Kamera / Isı Analizörü"
    category = CAT_MEASURE
    description = (
        "Termal kamera görüntüsü üzerindeki sıcak noktaları (Hotspot), soğuk bölgeleri ve ortalama ısıyı "
        "tespit eder. Motor, elektrik panosu veya lehim fırınındaki aşırı ısınma durumunda anında alarm üretir."
    )
    inputs = [
        ("image", IMG),
        ("temperature_data", NUM),
    ]
    outputs = [
        ("max_temp", NUM),
        ("min_temp", NUM),
        ("avg_temp", NUM),
        ("hotspot_x", NUM),
        ("hotspot_y", NUM),
        ("alarm_active", BOOL),
        ("overlay_image", IMG),
    ]
    params = [
        Param("alarm_threshold_temp", "float", 75.0, "Alarm Sıcaklık Eşiği (°C)", -50.0, 1500.0),
        Param("temp_min_scale", "float", 20.0, "Görüntü Min Sıcaklık (°C)", -50.0, 1500.0),
        Param("temp_max_scale", "float", 120.0, "Görüntü Max Sıcaklık (°C)", -50.0, 1500.0),
        Param("draw_crosshair", "bool", True, "En Sıcak Noktayı İşaretle"),
    ]

    def process(self, inputs):
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "max_temp": 0.0,
                "min_temp": 0.0,
                "avg_temp": 0.0,
                "hotspot_x": 0,
                "hotspot_y": 0,
                "alarm_active": False,
                "overlay_image": None,
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr)

        t_min = float(self.v("temp_min_scale"))
        t_max = float(self.v("temp_max_scale"))
        alarm_th = float(self.v("alarm_threshold_temp"))

        # Find min and max pixel brightness and locations
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        avg_val = float(np.mean(gray))

        # Map brightness (0..255) to temperature range (°C)
        t_range = t_max - t_min
        calc_max_temp = round(t_min + (max_val / 255.0) * t_range, 1)
        calc_min_temp = round(t_min + (min_val / 255.0) * t_range, 1)
        calc_avg_temp = round(t_min + (avg_val / 255.0) * t_range, 1)

        alarm = calc_max_temp >= alarm_th

        # Overlay hotspot
        if bool(self.v("draw_crosshair")):
            hx, hy = max_loc
            # Crosshair on maximum temperature
            color = (0, 0, 255) if alarm else (0, 255, 255)
            cv2.drawMarker(bgr, (hx, hy), color, cv2.MARKER_CROSS, 24, 2)
            cv2.circle(bgr, (hx, hy), 14, color, 2)
            put_text(bgr, f"MAX: {calc_max_temp}C", (hx + 18, hy - 8), 0.6, color, 2)

            # Banner at top
            stat_color = (0, 0, 200) if alarm else (40, 40, 40)
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 32), stat_color, -1)
            msg = f"{'⚠️ ASIRI ISINMA ALARMI!' if alarm else 'TERMAL NORMAL'} | Max: {calc_max_temp}C | Ort: {calc_avg_temp}C"
            put_text(bgr, msg, (12, 22), 0.55, (255, 255, 255), 1)

        return {
            "max_temp": calc_max_temp,
            "min_temp": calc_min_temp,
            "avg_temp": calc_avg_temp,
            "hotspot_x": max_loc[0],
            "hotspot_y": max_loc[1],
            "alarm_active": alarm,
            "overlay_image": bgr,
        }


# ==========================================================================================
# 4. Auto Lighting Controller (Kapalı Çevrim Akıllı Aydınlatma Kontrolcüsü)
# ==========================================================================================

@register
class AutoLightingController(Node):
    title = "Akıllı Aydınlatma Kontrolcüsü"
    category = CAT_OUT
    description = (
        "Görüntünün ortalama aydınlık düzeyini ölçerek kamera ışığını kapalı çevrim (feedback) ile dengeler. "
        "Güneş ışığı veya ortam değişimlerinde görüntünün kararmasını/patlamasını engelleyerek PWM veya Serial komut üretir."
    )
    inputs = [
        ("image", IMG),
        ("target_override", NUM),
    ]
    outputs = [
        ("current_brightness", NUM),
        ("target_brightness", NUM),
        ("adjustment_needed", NUM),
        ("pwm_output", NUM),
        ("serial_command", TEXT),
    ]
    params = [
        Param("target_luminance", "float", 128.0, "Hedef Parlaklık (0-255)", 10.0, 245.0),
        Param("tolerance", "float", 8.0, "İzin Verilen Tolerans (±)", 1.0, 50.0),
        Param("step_size", "float", 4.0, "Düzeltme Adımı", 0.5, 30.0),
        Param("command_prefix", "str", "LIGHT:SET:", "Seri Port Komut Ön Eki"),
    ]

    def setup(self):
        self.state.update(current_pwm=128.0)

    def process(self, inputs):
        st = self.state
        st.setdefault("current_pwm", 128.0)

        img_in = inputs.get("image")
        if img_in is None:
            return {
                "current_brightness": 0.0,
                "target_brightness": 128.0,
                "adjustment_needed": 0.0,
                "pwm_output": 128,
                "serial_command": "",
            }

        gray = _gray(_bgr(img_in))
        curr_lum = float(np.mean(gray))

        override = inputs.get("target_override")
        target_lum = float(override) if override is not None else float(self.v("target_luminance"))
        tol = float(self.v("tolerance"))
        step = float(self.v("step_size"))

        diff = target_lum - curr_lum
        adj = 0.0

        if abs(diff) > tol:
            adj = step if diff > 0 else -step
            st["current_pwm"] = float(np.clip(st["current_pwm"] + adj, 0.0, 255.0))

        pwm_val = int(round(st["current_pwm"]))
        prefix = str(self.v("command_prefix"))
        cmd = f"{prefix}{pwm_val}\n"

        return {
            "current_brightness": round(curr_lum, 1),
            "target_brightness": round(target_lum, 1),
            "adjustment_needed": round(adj, 1),
            "pwm_output": pwm_val,
            "serial_command": cmd,
        }


# ==========================================================================================
# 5. Virtual PLC & Conveyor Simulator (Sanal PLC & Konveyör Simülatörü)
# ==========================================================================================

@register
class VirtualPLCSimulator(Node):
    title = "Sanal PLC & Konveyör Simülatörü"
    category = CAT_INPUT
    description = (
        "Gerçek bir PLC veya konveyör bağlı olmadığında akışları masabaşında test etmek için sanal parça sensörü (tetik), "
        "parça numarası, ölçüm sapması ve ret (reject) pistonu mantığı simüle eder."
    )
    inputs = [
        ("reject_cmd", BOOL),
    ]
    outputs = [
        ("part_sensor", BOOL),
        ("part_id", TEXT),
        ("sim_measurement", NUM),
        ("is_defect", BOOL),
        ("conveyor_running", BOOL),
        ("total_parts", NUM),
        ("rejected_parts", NUM),
    ]
    params = [
        Param("parts_per_minute", "float", 30.0, "Dakikadaki Parça Hızı (PPM)", 1.0, 300.0),
        Param("defect_probability_pct", "float", 8.0, "Kusur Olasılığı %", 0.0, 100.0),
        Param("nominal_measurement", "float", 25.0, "Nominal Ölçüm Değeri"),
        Param("tolerance", "float", 0.25, "Kabul Edilen Tolerans (±)"),
    ]

    def setup(self):
        self.state.update(
            last_part_time=time.monotonic(),
            total_parts=0,
            rejected_parts=0,
            part_id="SIM-100",
            meas=25.0,
            defect=False,
            prev_reject=False,
        )

    def process(self, inputs):
        st = self.state
        now = time.monotonic()
        ppm = max(1.0, float(self.v("parts_per_minute")))
        interval_sec = 60.0 / ppm

        # Check reject feedback command from flow
        if _rising(self, inputs.get("reject_cmd"), "prev_reject"):
            st["rejected_parts"] += 1

        part_arrived = False
        if now - st["last_part_time"] >= interval_sec:
            st["last_part_time"] = now
            st["total_parts"] += 1
            part_arrived = True

            st["part_id"] = f"SIM-{1000 + st['total_parts']}"
            defect_prob = float(self.v("defect_probability_pct"))
            is_bad = (random.random() * 100.0) < defect_prob
            st["defect"] = is_bad

            nom = float(self.v("nominal_measurement"))
            tol = float(self.v("tolerance"))
            if is_bad:
                dev = random.choice([-1.0, 1.0]) * random.uniform(tol + 0.1, tol + 0.8)
            else:
                dev = random.uniform(-tol * 0.7, tol * 0.7)

            st["meas"] = round(nom + dev, 3)

        return {
            "part_sensor": part_arrived,
            "part_id": st["part_id"],
            "sim_measurement": st["meas"],
            "is_defect": st["defect"],
            "conveyor_running": True,
            "total_parts": st["total_parts"],
            "rejected_parts": st["rejected_parts"],
        }
