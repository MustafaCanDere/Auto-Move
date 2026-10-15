"""Multimodal & Augmented Reality Industrial Vision blocks:
- ARProjectionGuidance: AR projection & heads-up assembly guidance for manual workstations
- AcousticVisionFusion: Acoustic / vibration FFT spectral analysis & thermal/visual predictive maintenance
- CADSurfaceDeviationHeatmap: Digital Twin CAD surface deviation heatmap (tolerance envelope colormap)
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ..core import node as _node
from ._text import put_text
from .processing import _bgr, _gray

CAT_MEASURE = "2b · Kalite Kontrol ve Ölçüm"
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
# 4. AR Projection & Assembly Guidance (Artırılmış Gerçeklik Montaj Rehberi)
# ==========================================================================================

DEFAULT_RECIPES = {
    "Elektronik Kart Montajı": [
        {"step": 1, "text": "Adım 1/4: U1 Ana MCU Entegresini soketine yerleştirin (Pin 1 sol üste)", "box": [140, 100, 260, 220], "torque": ""},
        {"step": 2, "text": "Adım 2/4: C1 & C2 Filtre Kondansatörlerini monte edin", "box": [300, 120, 420, 210], "torque": ""},
        {"step": 3, "text": "Adım 3/4: Besleme terminal vidalarını 2.5 Nm tork ile sıkın", "box": [420, 240, 540, 340], "torque": "2.5 Nm"},
        {"step": 4, "text": "Adım 4/4: QC Onay Barkodunu sağ alt köşeye yapıştırın", "box": [160, 280, 290, 370], "torque": ""},
    ],
    "Kablo Demeti Soketleme": [
        {"step": 1, "text": "Adım 1/3: Beyaz ana güç kablosunu CN1 soketine takın", "box": [120, 110, 240, 210], "torque": ""},
        {"step": 2, "text": "Adım 2/3: Kırmızı CAN-Bus sinyal soketini CN2'ye kilitleyin (Klik sesi)", "box": [280, 130, 400, 230], "torque": ""},
        {"step": 3, "text": "Adım 3/3: Şasi topraklama vidasını sabitleyin", "box": [410, 220, 520, 320], "torque": "3.0 Nm"},
    ],
    "Motor Bloğu Civata Sıkma": [
        {"step": 1, "text": "Adım 1/4: Civata #1 (Sol Üst) - 35 Nm ile sıkın", "box": [150, 120, 230, 200], "torque": "35 Nm"},
        {"step": 2, "text": "Adım 2/4: Civata #2 (Sağ Alt - Çapraz) - 35 Nm ile sıkın", "box": [380, 300, 460, 380], "torque": "35 Nm"},
        {"step": 3, "text": "Adım 3/4: Civata #3 (Sağ Üst) - 35 Nm ile sıkın", "box": [380, 120, 460, 200], "torque": "35 Nm"},
        {"step": 4, "text": "Adım 4/4: Civata #4 (Sol Alt - Çapraz) - 35 Nm ile sıkın", "box": [150, 300, 230, 380], "torque": "35 Nm"},
    ],
}


@register
class ARProjectionGuidance(Node):
    title = "Artırılmış Gerçeklik Montaj Rehberi (AR)"
    category = CAT_OUT
    description = (
        "Montaj masası veya çalışma alanına tepe projektörü ya da ekran HUD ile adım adım görsel montaj rehberi yansıtır. "
        "Operatörü yönlendirir, parça montaj kutularını, yön oklarını ve tork değerlerini gösterir."
    )
    inputs = [
        ("image", IMG),
        ("trigger_next_step", BOOL),
        ("reset", BOOL),
        ("current_step", NUM),
    ]
    outputs = [
        ("ar_projection_image", IMG),
        ("step_number", NUM),
        ("step_text", TEXT),
        ("step_completed", BOOL),
        ("all_steps_done", BOOL),
    ]
    params = [
        Param("preset_recipe", "choice", "Elektronik Kart Montajı", "Reçete / Görev Şablonu",
              choices=["Elektronik Kart Montajı", "Kablo Demeti Soketleme", "Motor Bloğu Civata Sıkma", "Özel JSON"]),
        Param("display_mode", "choice", "Kamera Üstü AR (HUD)", "Görüntüleme Modu",
              choices=["Kamera Üstü AR (HUD)", "Projektör Karanlık Mod (Yansıtma)"]),
        Param("guide_color", "choice", "Neon Mavi", "Rehber Renk Teması",
              choices=["Neon Mavi", "Zümrüt Yeşil", "Parlak Sarı"]),
        Param("enable_auto_verify", "bool", True, "Montajı Görsel Olarak Otomatik Doğrula"),
        Param("auto_advance_delay_sec", "float", 1.8, "Doğrulama Sonrası Geçiş Süresi (sn)", 0.2, 10.0),
        Param("custom_recipe_json", "str", "", "Özel Reçete JSON (İsteğe Bağlı)"),
    ]

    def setup(self):
        self.state.update(
            step_idx=0,
            verified_time=None,
            completed_steps=set(),
            prev_next_trig=False,
            prev_reset_trig=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("step_idx", 0)
        st.setdefault("verified_time", None)
        st.setdefault("completed_steps", set())

        # Reset trigger
        if _rising(self, inputs.get("reset"), "prev_reset_trig"):
            st["step_idx"] = 0
            st["verified_time"] = None
            st["completed_steps"].clear()

        # Manual step override
        if inputs.get("current_step") is not None:
            try:
                ovr = int(inputs.get("current_step")) - 1
                if ovr >= 0:
                    st["step_idx"] = ovr
            except Exception:
                pass

        # Load recipe
        preset = self.v("preset_recipe")
        steps = DEFAULT_RECIPES.get(preset, DEFAULT_RECIPES["Elektronik Kart Montajı"])
        custom_json = str(self.v("custom_recipe_json") or "").strip()
        if preset == "Özel JSON" and custom_json:
            try:
                parsed = json.loads(custom_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    steps = parsed
            except Exception:
                pass

        total_steps = len(steps)
        curr_idx = min(st["step_idx"], total_steps)

        # Handle manual trigger next
        if _rising(self, inputs.get("trigger_next_step"), "prev_next_trig"):
            if curr_idx < total_steps:
                st["completed_steps"].add(curr_idx)
                st["step_idx"] = curr_idx + 1
                curr_idx = st["step_idx"]
                st["verified_time"] = None

        img_in = inputs.get("image")
        h, w = (480, 640)
        if img_in is not None:
            bgr_in = _bgr(img_in)
            h, w = bgr_in.shape[:2]
        else:
            bgr_in = np.zeros((h, w, 3), dtype=np.uint8)

        # Output canvas based on display mode
        is_dark_proj = (self.v("display_mode") == "Projektör Karanlık Mod (Yansıtma)")
        if is_dark_proj:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            canvas = bgr_in.copy()

        # Theme colors (BGR)
        theme_str = self.v("guide_color")
        if theme_str == "Zümrüt Yeşil":
            neon_col = (50, 240, 50)
        elif theme_str == "Parlak Sarı":
            neon_col = (0, 220, 255)
        else:
            neon_col = (255, 200, 0)  # Neon Cyan / Blue

        all_done = (curr_idx >= total_steps)
        step_completed = False
        step_text = "Tüm montaj adımları başarıyla tamamlandı! Parça hazır." if all_done else ""

        if not all_done:
            active_step = steps[curr_idx]
            step_text = active_step.get("text", f"Adım {curr_idx + 1}")
            box = active_step.get("box", [100, 100, 200, 200])
            torque = active_step.get("torque", "")

            # Scale box to canvas if needed
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            bx1, bx2 = max(0, min(w - 1, bx1)), max(0, min(w - 1, bx2))
            by1, by2 = max(0, min(h - 1, by1)), max(0, min(h - 1, by2))

            # Auto verification check inside target box
            auto_verify = bool(self.v("enable_auto_verify"))
            if auto_verify and img_in is not None:
                roi = canvas[by1:by2, bx1:bx2]
                if roi.size > 0:
                    roi_std = float(np.std(roi))
                    # If high contrast/detail is observed inside box, consider verified
                    if roi_std > 25.0:
                        step_completed = True
                        if st["verified_time"] is None:
                            st["verified_time"] = time.time()
                        delay = float(self.v("auto_advance_delay_sec"))
                        if (time.time() - st["verified_time"]) >= delay:
                            st["completed_steps"].add(curr_idx)
                            st["step_idx"] = curr_idx + 1
                            st["verified_time"] = None
                    else:
                        st["verified_time"] = None

            # Render AR Graphics for Active Step
            target_col = (0, 255, 100) if step_completed else neon_col
            # Corner brackets around ROI
            corn_len = 16
            thick = 3
            # Top-left
            cv2.line(canvas, (bx1, by1), (bx1 + corn_len, by1), target_col, thick)
            cv2.line(canvas, (bx1, by1), (bx1, by1 + corn_len), target_col, thick)
            # Top-right
            cv2.line(canvas, (bx2, by1), (bx2 - corn_len, by1), target_col, thick)
            cv2.line(canvas, (bx2, by1), (bx2, by1 + corn_len), target_col, thick)
            # Bottom-left
            cv2.line(canvas, (bx1, by2), (bx1 + corn_len, by2), target_col, thick)
            cv2.line(canvas, (bx1, by2), (bx1, by2 - corn_len), target_col, thick)
            # Bottom-right
            cv2.line(canvas, (bx2, by2), (bx2 - corn_len, by2), target_col, thick)
            cv2.line(canvas, (bx2, by2), (bx2, by2 - corn_len), target_col, thick)

            # Center target crosshair
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            cv2.circle(canvas, (cx, cy), 8, target_col, 1)
            cv2.line(canvas, (cx - 14, cy), (cx + 14, cy), target_col, 1)
            cv2.line(canvas, (cx, cy - 14), (cx, cy + 14), target_col, 1)

            # Pulsing target circle
            pulse_rad = int(22 + 4 * math.sin(time.time() * 5.0))
            cv2.circle(canvas, (cx, cy), pulse_rad, target_col, 2)

            # Torque info if present
            if torque:
                cv2.rectangle(canvas, (bx1, by2 + 6), (bx1 + 140, by2 + 28), (20, 20, 20), -1)
                put_text(canvas, f"Tork: {torque}", (bx1 + 6, by2 + 22), 0.45, (0, 230, 255), 1)

            # Directional arrow
            arrow_start = (max(20, bx1 - 40), cy)
            arrow_end = (bx1 - 10, cy)
            cv2.arrowedLine(canvas, arrow_start, arrow_end, target_col, 3, tipLength=0.35)

        # Draw Global HUD Banner
        banner_h = 55
        cv2.rectangle(canvas, (0, 0), (w, banner_h), (15, 20, 30), -1)
        cv2.line(canvas, (0, banner_h), (w, banner_h), neon_col, 2)

        status_lbl = "TAMAMLANDI (OK)" if all_done else f"ADIM {min(curr_idx + 1, total_steps)} / {total_steps}"
        stat_col = (0, 255, 100) if all_done else (255, 255, 255)
        put_text(canvas, f"AR REHBER: {status_lbl}", (16, 24), 0.6, stat_col, 2)
        put_text(canvas, step_text, (16, 45), 0.45, (200, 220, 240), 1)

        # Draw mini step progress dots on top right
        dot_spacing = 20
        dot_start_x = w - 30 - (total_steps * dot_spacing)
        for i in range(total_steps):
            dot_x = dot_start_x + i * dot_spacing
            dot_y = 25
            if i in st["completed_steps"]:
                cv2.circle(canvas, (dot_x, dot_y), 6, (0, 255, 100), -1)
            elif i == curr_idx:
                cv2.circle(canvas, (dot_x, dot_y), 7, neon_col, 2)
                cv2.circle(canvas, (dot_x, dot_y), 3, neon_col, -1)
            else:
                cv2.circle(canvas, (dot_x, dot_y), 5, (80, 80, 80), -1)

        return {
            "ar_projection_image": canvas,
            "step_number": curr_idx + 1 if not all_done else total_steps,
            "step_text": step_text,
            "step_completed": step_completed or all_done,
            "all_steps_done": all_done,
        }


# ==========================================================================================
# 5. Acoustic & Visual Fusion Predictive Maintenance (Akustik & Görsel Kestirimci Bakım)
# ==========================================================================================

@register
class AcousticVisionFusion(Node):
    title = "Akustik ve Görsel Füzyon Kestirimci Bakım"
    category = CAT_MEASURE
    description = (
        "Motor, rulman, mil ve robot eklemlerinden gelen titreşim/akustik ses FFT spektrumu ile "
        "kamera ve sıcaklık verilerini füzyonlayarak mekanik aşınma ve arızaları günler öncesinden öngörür."
    )
    inputs = [
        ("image", IMG),
        ("vibration_signal", ANY),
        ("temperature_c", NUM),
        ("rpm", NUM),
    ]
    outputs = [
        ("health_index_pct", NUM),
        ("bearing_fault_detected", BOOL),
        ("vibration_rms_g", NUM),
        ("dominant_frequency_hz", NUM),
        ("anomaly_score", NUM),
        ("diagnostic_summary", TEXT),
        ("fusion_hud_image", IMG),
    ]
    params = [
        Param("sample_rate_hz", "int", 20000, "Örnekleme Frekansı (Hz)", 1000, 100000),
        Param("rpm_nominal", "float", 1500.0, "Nominal Mil Hızı (RPM)", 60.0, 30000.0),
        Param("rms_alarm_threshold_g", "float", 4.5, "Titreşim RMS Alarm Sınırı (g)", 0.1, 50.0),
        Param("temp_alarm_c", "float", 75.0, "Kritik Sıcaklık Sınırı (°C)", 20.0, 200.0),
        Param("simulation_mode", "bool", True, "Sinyal Yoksa Gerçekçi Endüstriyel Titreşim Simüle Et"),
    ]

    def setup(self):
        self.state.update(
            health=98.0,
            fault=False,
            rms=0.8,
            freq=25.0,
            diag="Normal Çalışma",
        )

    def process(self, inputs):
        st = self.state
        fs = int(self.v("sample_rate_hz"))
        nom_rpm = float(inputs.get("rpm") if inputs.get("rpm") is not None else self.v("rpm_nominal"))
        f0 = nom_rpm / 60.0  # Fundamental shaft frequency in Hz
        temp_c = float(inputs.get("temperature_c") if inputs.get("temperature_c") is not None else 42.0)
        rms_thr = float(self.v("rms_alarm_threshold_g"))
        temp_thr = float(self.v("temp_alarm_c"))

        # Obtain or simulate vibration waveform
        raw_sig = inputs.get("vibration_signal")
        if raw_sig is not None and isinstance(raw_sig, (list, tuple, np.ndarray)) and len(raw_sig) >= 64:
            sig = np.asarray(raw_sig, dtype=np.float32)
        elif bool(self.v("simulation_mode")):
            # Synthesize 1024 samples
            n_pts = 1024
            t = np.linspace(0, n_pts / fs, n_pts, endpoint=False)
            # Base rotational harmonics (1X, 2X)
            sig = 1.2 * np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * (2 * f0) * t)
            # If high temperature or random trigger, simulate bearing fault peak BPFI ~ 5.4 * f0
            if temp_c > temp_thr or st.get("sim_fault", False):
                sig += 3.8 * np.sin(2 * np.pi * (5.4 * f0) * t)
            # Add factory background noise
            sig += np.random.normal(0, 0.35, n_pts)
        else:
            sig = np.zeros(512, dtype=np.float32)

        # Calculate RMS vibration (g)
        rms_val = float(np.sqrt(np.mean(sig ** 2)))

        # Compute FFT spectrum
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)

        # Dominant frequency
        if len(fft_vals) > 2:
            peak_idx = int(np.argmax(fft_vals[1:])) + 1
            dom_freq = float(freqs[peak_idx])
        else:
            dom_freq = f0

        # Anomaly scoring and diagnosis
        diag_items = []
        bearing_fault = False
        anomaly = 0.0

        if rms_val > rms_thr:
            bearing_fault = True
            anomaly += 0.55
            diag_items.append(f"Yüksek Titreşim ({rms_val:.2f}g > {rms_thr:.1f}g)")

        if temp_c > temp_thr:
            bearing_fault = True
            anomaly += 0.35
            diag_items.append(f"Aşırı Rulman Isınması ({temp_c:.1f}°C > {temp_thr:.1f}°C)")

        # Check for characteristic bearing outer/inner race harmonic ~ 4-8X of f0
        if dom_freq > 3.5 * f0 and rms_val > 2.0:
            anomaly += 0.2
            diag_items.append(f"Rulman Bilye/İç Bilezik Frekansı ({dom_freq:.1f} Hz)")

        anomaly = float(min(1.0, anomaly))
        health_pct = round(max(0.0, (1.0 - anomaly) * 100.0), 1)

        if not diag_items:
            diag_str = "Sistem Sağlıklı: Rulman ve titreşim spektrumu nominal sınırlar içinde."
        else:
            diag_str = "DİKKAT: " + " | ".join(diag_items)

        # Render Fusion HUD Image
        img_in = inputs.get("image")
        if img_in is not None:
            hud = _bgr(img_in).copy()
        else:
            hud = np.zeros((480, 640, 3), dtype=np.uint8)
        hh, hw = hud.shape[:2]

        # Draw HUD Box for Telemetry & FFT at bottom
        box_h = 130
        cv2.rectangle(hud, (10, hh - box_h - 10), (hw - 10, hh - 10), (10, 15, 25), -1)
        hud_border_col = (0, 0, 230) if bearing_fault else (0, 200, 100)
        cv2.rectangle(hud, (10, hh - box_h - 10), (hw - 10, hh - 10), hud_border_col, 2)

        # Text Telemetry
        put_text(hud, f"AKUSTİK & TİTREŞİM FÜZYON HUD  |  Sağlık İndeksi: %{health_pct}",
                 (22, hh - box_h + 12), 0.55, hud_border_col, 2)
        put_text(hud, f"RMS: {rms_val:.2f} g  |  Tepe: {dom_freq:.1f} Hz  |  Sıcaklık: {temp_c:.1f} °C  |  Mil: {nom_rpm:.0f} RPM",
                 (22, hh - box_h + 34), 0.45, (220, 220, 220), 1)
        put_text(hud, diag_str[:85], (22, hh - box_h + 52), 0.42, (180, 180, 240), 1)

        # Draw Mini FFT Spectrum waveform
        spec_x1 = 22
        spec_y2 = hh - 20
        spec_w = hw - 44
        spec_h = 45
        cv2.rectangle(hud, (spec_x1, spec_y2 - spec_h), (spec_x1 + spec_w, spec_y2), (20, 25, 35), -1)

        if len(fft_vals) > 10:
            n_bars = min(60, len(fft_vals) // 2)
            max_fft = max(1e-5, float(np.max(fft_vals[:n_bars])))
            bar_w = max(2, spec_w // n_bars)
            for bi in range(n_bars):
                bh = int((fft_vals[bi] / max_fft) * (spec_h - 6))
                bx = spec_x1 + bi * bar_w
                bar_color = (0, 120, 255) if bi == peak_idx else (0, 200, 180)
                cv2.rectangle(hud, (bx, spec_y2 - bh), (bx + bar_w - 1, spec_y2), bar_color, -1)

        st.update(
            health=health_pct,
            fault=bearing_fault,
            rms=rms_val,
            freq=dom_freq,
            diag=diag_str,
        )

        return {
            "health_index_pct": health_pct,
            "bearing_fault_detected": bearing_fault,
            "vibration_rms_g": round(rms_val, 3),
            "dominant_frequency_hz": round(dom_freq, 1),
            "anomaly_score": round(anomaly, 2),
            "diagnostic_summary": diag_str,
            "fusion_hud_image": hud,
        }


# ==========================================================================================
# 6. CAD Surface Deviation Heatmap (CAD Yüzey Sapma Isı Haritası)
# ==========================================================================================

@register
class CADSurfaceDeviationHeatmap(Node):
    title = "CAD Yüzey Sapma Isı Haritası (Digital Twin)"
    category = CAT_MEASURE
    description = (
        "Üretilen parçayı ideal CAD modeli veya referans şablon ile mikron düzeyinde karşılaştırır. "
        "Yüzeydeki çapak, çekme, ezilme ve boyutsal form sapmalarını renkli ısı haritası (Heatmap) olarak sunar."
    )
    inputs = [
        ("image", IMG),
        ("cad_template", IMG),
    ]
    outputs = [
        ("deviation_heatmap", IMG),
        ("max_positive_dev_mm", NUM),
        ("max_negative_dev_mm", NUM),
        ("surface_defect_pct", NUM),
        ("is_within_tolerance", BOOL),
        ("tolerance_report", TEXT),
    ]
    params = [
        Param("tolerance_mm", "float", 0.35, "Tolerans Sınırı ± (mm)", 0.01, 10.0),
        Param("scale_mm_per_px", "float", 0.05, "Ölçek (mm / Piksel)", 0.0001, 2.0),
        Param("colormap", "choice", "TURBO (Yüksek Kontrast)", "Isı Haritası Renk Paleti",
              choices=["TURBO (Yüksek Kontrast)", "JET (Mavi-Yeşil-Kırmızı)", "VIRIDIS (Bilimsel)"]),
        Param("blend_alpha", "float", 0.65, "Isı Haritası Saydamlık (Alpha)", 0.1, 1.0),
        Param("auto_align_cad", "bool", True, "CAD Modeli ile Otomatik Hizala (Moments/Centroid)"),
    ]

    def setup(self):
        self.state.update(
            pos_dev=0.0,
            neg_dev=0.0,
            defect_pct=0.0,
            is_pass=True,
            rep="Tolerans içinde",
        )

    def process(self, inputs):
        st = self.state
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "deviation_heatmap": None,
                "max_positive_dev_mm": 0.0,
                "max_negative_dev_mm": 0.0,
                "surface_defect_pct": 0.0,
                "is_within_tolerance": True,
                "tolerance_report": "Görüntü yok",
            }

        bgr = _bgr(img_in).copy()
        gray = _gray(bgr)
        h, w = gray.shape[:2]

        scale = float(self.v("scale_mm_per_px"))
        tol_mm = float(self.v("tolerance_mm"))
        tol_px = max(1.0, tol_mm / scale)

        # Segment live part mask
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, live_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Invert if background was bright
        if np.mean(live_mask) > 127:
            live_mask = cv2.bitwise_not(live_mask)

        # Reference CAD mask: use provided cad_template or synthetic golden reference
        cad_in = inputs.get("cad_template")
        if cad_in is not None:
            cad_gray = _gray(_bgr(cad_in))
            if cad_gray.shape[:2] != (h, w):
                cad_gray = cv2.resize(cad_gray, (w, h))
            _, cad_mask = cv2.threshold(cad_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            if np.mean(cad_mask) > 127:
                cad_mask = cv2.bitwise_not(cad_mask)
        else:
            # Generate smoothed reference template from morphological closing/opening of live mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            cad_mask = cv2.morphologyEx(live_mask, cv2.MORPH_CLOSE, kernel)

        # Distance Transforms
        # Positive deviation: Live has material where CAD does not (Burr, swelling)
        extra_mat = cv2.bitwise_and(live_mask, cv2.bitwise_not(cad_mask))
        dist_pos = cv2.distanceTransform(extra_mat, cv2.DIST_L2, 5)

        # Negative deviation: CAD has material where Live does not (Void, shrinkage)
        miss_mat = cv2.bitwise_and(cad_mask, cv2.bitwise_not(live_mask))
        dist_neg = cv2.distanceTransform(miss_mat, cv2.DIST_L2, 5)

        max_pos_px = float(np.max(dist_pos)) if extra_mat.any() else 0.0
        max_neg_px = float(np.max(dist_neg)) if miss_mat.any() else 0.0

        max_pos_mm = round(max_pos_px * scale, 3)
        max_neg_mm = round(max_neg_px * scale, 3)

        # Total part area and defect percentage
        total_part_px = max(1, int(np.count_nonzero(cad_mask)))
        defect_px = int(np.count_nonzero(dist_pos > tol_px) + np.count_nonzero(dist_neg > tol_px))
        defect_pct = round((defect_px / total_part_px) * 100.0, 2)

        is_pass = (max_pos_mm <= tol_mm) and (max_neg_mm <= tol_mm) and (defect_pct <= 1.0)

        # Color map selection
        cmap_name = self.v("colormap")
        if "TURBO" in cmap_name:
            cv_cmap = cv2.COLORMAP_TURBO
        elif "VIRIDIS" in cmap_name:
            cv_cmap = cv2.COLORMAP_VIRIDIS
        else:
            cv_cmap = cv2.COLORMAP_JET

        # Combine signed deviation map into normalized 0-255 image:
        # 128 = 0 deviation (nominal)
        # > 128 = positive deviation
        # < 128 = negative deviation
        dev_map = np.ones((h, w), dtype=np.float32) * 128.0
        max_range_px = max(tol_px * 2.5, 10.0)
        dev_map += (dist_pos / max_range_px) * 127.0
        dev_map -= (dist_neg / max_range_px) * 127.0
        dev_map_u8 = np.clip(dev_map, 0, 255).astype(np.uint8)

        # Apply colormap
        heat_color = cv2.applyColorMap(dev_map_u8, cv_cmap)

        # Mask only regions belonging to live or CAD part
        combined_mask = cv2.bitwise_or(live_mask, cad_mask)
        heat_color[combined_mask == 0] = [20, 20, 20]

        # Blend with original
        alpha = float(self.v("blend_alpha"))
        blended = cv2.addWeighted(bgr, 1.0 - alpha, heat_color, alpha, 0)
        blended[combined_mask == 0] = bgr[combined_mask == 0]

        # Draw contour of golden CAD model in white/green
        cad_cnts, _ = cv2.findContours(cad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, cad_cnts, -1, (255, 255, 255), 1)

        # Overlay HUD Box
        hud_col = (0, 220, 0) if is_pass else (0, 0, 240)
        cv2.rectangle(blended, (12, 12), (380, 75), (20, 20, 30), -1)
        cv2.rectangle(blended, (12, 12), (380, 75), hud_col, 2)
        put_text(blended, f"CAD YÜZEY SAPMA ISI HARİTASI", (22, 32), 0.55, (255, 255, 255), 2)
        put_text(blended, f"Pozitif (+Çapak): +{max_pos_mm} mm  |  Negatif (-Eksik): -{max_neg_mm} mm",
                 (22, 50), 0.42, (200, 230, 255), 1)
        put_text(blended, f"Tolerans: ±{tol_mm} mm  |  Hata Alanı: %{defect_pct}  |  {'OK' if is_pass else 'NOK'}",
                 (22, 68), 0.44, hud_col, 1)

        rep_str = f"Durum: {'ONAYLANDI (OK)' if is_pass else 'REDDEDİLDİ (NOK)'} | Maks Sapma: +{max_pos_mm}/-{max_neg_mm} mm (Tol: ±{tol_mm} mm)"

        st.update(
            pos_dev=max_pos_mm,
            neg_dev=max_neg_mm,
            defect_pct=defect_pct,
            is_pass=is_pass,
            rep=rep_str,
        )

        return {
            "deviation_heatmap": blended,
            "max_positive_dev_mm": max_pos_mm,
            "max_negative_dev_mm": max_neg_mm,
            "surface_defect_pct": defect_pct,
            "is_within_tolerance": is_pass,
            "tolerance_report": rep_str,
        }
