"""Robotics Guidance, Statistical Process Control (SPC / Cpk), and Fast Template Alignment blocks."""
from __future__ import annotations

import collections
import math
import socket
import threading
import time

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ._text import put_text
from .processing import _bgr, _gray

CAT_IMG = "2 · Görüntü İşleme"
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
# 1. Robot Guidance & Pick-and-Place (Robot Rehberliği)
# ==========================================================================================

@register
class RobotGuidance(Node):
    title = "Robot Rehberliği (Pick & Place)"
    category = CAT_OUT
    description = (
        "Görüntüdeki parça konumunu (piksel ve açı) kalibre edilmiş robot çalışma koordinatlarına (X mm, Y mm, Rz derece) "
        "dönüştürür. Universal Robots (URScript), standart TCP veya PLC için alma (Pick) komutları üretir."
    )
    inputs = [
        ("image", IMG),
        ("x_px", NUM),
        ("y_px", NUM),
        ("angle_deg", NUM),
        ("trigger", BOOL),
    ]
    outputs = [
        ("x_mm", NUM),
        ("y_mm", NUM),
        ("rz_deg", NUM),
        ("pick_ready", BOOL),
        ("command_str", TEXT),
        ("overlay_image", IMG),
    ]
    params = [
        Param("scale_mm_per_px", "float", 0.25, "Ölçek (mm / Piksel)", 0.0001, 100.0),
        Param("origin_x_px", "float", 0.0, "Orijin Sıfır Noktası X (px)"),
        Param("origin_y_px", "float", 0.0, "Orijin Sıfır Noktası Y (px)"),
        Param("offset_z_mm", "float", 120.0, "Yaklaşma Yüksekliği Z (mm)"),
        Param("robot_type", "choice", "Universal Robots (URScript)", "Robot Protokolü",
              choices=["Universal Robots (URScript)", "Standart TCP Komutu", "Sadece Koordinat"]),
        Param("robot_ip", "str", "192.168.1.50", "Robot IP Adresi"),
        Param("robot_port", "int", 30002, "Robot Portu", 1024, 65535),
        Param("send_network", "bool", False, "Ağ Üzerinden Robota Gönder"),
        Param("cooldown", "float", 1.0, "En Az Bekleme (sn)", 0.0, 60.0),
    ]

    def setup(self):
        self.state.update(
            last_sent=0.0,
            count=0,
            command="",
            prev_trig=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("last_sent", 0.0)
        st.setdefault("count", 0)
        st.setdefault("command", "")

        x_px = float(inputs.get("x_px") or 0.0)
        y_px = float(inputs.get("y_px") or 0.0)
        rz = float(inputs.get("angle_deg") or 0.0)

        # Coordinate transformation
        scale = float(self.v("scale_mm_per_px"))
        ox = float(self.v("origin_x_px"))
        oy = float(self.v("origin_y_px"))
        z_mm = float(self.v("offset_z_mm"))

        x_mm = round((x_px - ox) * scale, 2)
        y_mm = round((y_px - oy) * scale, 2)
        rz_deg = round(rz % 360.0, 2)

        # Generate command format
        rob_proto = self.v("robot_type")
        if rob_proto.startswith("Universal Robots"):
            # URScript movel pose [x, y, z, rx, ry, rz] in meters and radians
            x_m = x_mm / 1000.0
            y_m = y_mm / 1000.0
            z_m = z_mm / 1000.0
            rz_rad = math.radians(rz_deg)
            cmd = f"def pick():\n  movel(p[{x_m:.4f}, {y_m:.4f}, {z_m:.4f}, 0.0, 3.1415, {rz_rad:.4f}], a=1.2, v=0.25)\nend\npick()\n"
        elif rob_proto.startswith("Standart TCP"):
            cmd = f"PICK;X={x_mm:.2f};Y={y_mm:.2f};Z={z_mm:.2f};RZ={rz_deg:.2f};\n"
        else:
            cmd = f"X:{x_mm:.2f}, Y:{y_mm:.2f}, Rz:{rz_deg:.2f}"

        st["command"] = cmd

        # Trigger check
        trig = _rising(self, inputs.get("trigger"), "prev_trig")
        now = time.monotonic()
        cooldown = float(self.v("cooldown"))

        ready = False
        if trig and (now - st["last_sent"] >= cooldown):
            ready = True
            st["last_sent"] = now
            st["count"] += 1

            if bool(self.v("send_network")):
                ip = str(self.v("robot_ip")).strip()
                port = int(self.v("robot_port"))

                def do_send():
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.settimeout(2.0)
                            s.connect((ip, port))
                            s.sendall(cmd.encode("utf-8"))
                        print(f"[Robot Rehberliği] Komut iletildi ({ip}:{port})")
                    except Exception as exc:
                        print(f"[Robot Rehberliği] Ağ Hatası ({ip}): {exc}")

                threading.Thread(target=do_send, daemon=True).start()

        # Visualization overlay
        img_in = inputs.get("image")
        out_img = None
        if img_in is not None:
            out_img = _bgr(img_in).copy()
            ix, iy = int(x_px), int(y_px)
            cv2.drawMarker(out_img, (ix, iy), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(out_img, (ix, iy), 18, (0, 255, 255), 2)
            # Arrow showing grasp rotation
            rad = math.radians(rz_deg)
            ax = int(ix + 35 * math.cos(rad))
            ay = int(iy + 35 * math.sin(rad))
            cv2.arrowedLine(out_img, (ix, iy), (ax, ay), (0, 0, 255), 2, tipLength=0.3)
            put_text(out_img, f"({x_mm}mm, {y_mm}mm, {rz_deg}deg)", (ix + 15, iy - 10), 0.55, (255, 255, 255), 1)

        return {
            "x_mm": x_mm,
            "y_mm": y_mm,
            "rz_deg": rz_deg,
            "pick_ready": ready,
            "command_str": cmd,
            "overlay_image": out_img,
        }


# ==========================================================================================
# 2. Statistical Process Control (SPC / Cpk / Altı Sigma Analizörü)
# ==========================================================================================

@register
class SPCAnalyzer(Node):
    title = "İstatistiksel Proses Kontrolü (SPC / Cpk)"
    category = CAT_MEASURE
    description = (
        "Otomotiv ve havacılık standartlarında (IATF 16949 / Altı Sigma) zorunlu olan "
        "Cp, Cpk yeterlilik indislerini, 3-Sigma kontrol limitlerini (UCL, LCL) ve proses kayma trendlerini canlı hesaplar."
    )
    inputs = [
        ("measurement", NUM),
        ("sample_trigger", BOOL),
        ("reset", BOOL),
    ]
    outputs = [
        ("mean", NUM),
        ("std_dev", NUM),
        ("cp", NUM),
        ("cpk", NUM),
        ("ucl", NUM),
        ("lcl", NUM),
        ("in_control", BOOL),
        ("trend_warning", BOOL),
        ("sample_count", NUM),
        ("summary", TEXT),
    ]
    params = [
        Param("usl", "float", 25.5, "Üst Tolerans Sınırı (USL)"),
        Param("lsl", "float", 24.5, "Alt Tolerans Sınırı (LSL)"),
        Param("window_size", "int", 50, "Örnek Pencere Boyutu (N)", 5, 1000),
        Param("sigma_level", "float", 3.0, "Sigma Düzeyi (Varsayılan: 3)", 1.0, 6.0),
    ]

    def setup(self):
        self.state.update(
            samples=collections.deque(maxlen=int(self.v("window_size"))),
            mean=0.0,
            std_dev=0.0,
            cp=0.0,
            cpk=0.0,
            ucl=0.0,
            lcl=0.0,
            in_control=True,
            trend_warning=False,
            prev_sample=False,
            prev_reset=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("samples", collections.deque(maxlen=int(self.v("window_size"))))

        # Reset trigger
        if _rising(self, inputs.get("reset"), "prev_reset"):
            st["samples"].clear()
            self.setup()

        # Sample acquisition on rising edge
        trig = _rising(self, inputs.get("sample_trigger"), "prev_sample")
        meas = inputs.get("measurement")
        if trig and meas is not None:
            st["samples"].append(float(meas))

        samples = list(st["samples"])
        n = len(samples)

        usl = float(self.v("usl"))
        lsl = float(self.v("lsl"))
        sigma_mult = float(self.v("sigma_level"))

        if n < 3:
            return {
                "mean": round(samples[-1], 3) if n > 0 else 0.0,
                "std_dev": 0.0,
                "cp": 0.0,
                "cpk": 0.0,
                "ucl": usl,
                "lcl": lsl,
                "in_control": True,
                "trend_warning": False,
                "sample_count": n,
                "summary": f"Yetersiz örnek ({n}/3+)",
            }

        mean_val = float(np.mean(samples))
        std_val = float(np.std(samples, ddof=1))  # sample standard deviation

        # Control limits
        ucl = mean_val + (sigma_mult * std_val)
        lcl = mean_val - (sigma_mult * std_val)

        # Cp & Cpk
        tol_range = usl - lsl
        if std_val > 1e-6 and tol_range > 0:
            cp = tol_range / (6.0 * std_val)
            cpu = (usl - mean_val) / (3.0 * std_val)
            cpl = (mean_val - lsl) / (3.0 * std_val)
            cpk = min(cpu, cpl)
        else:
            cp = 0.0
            cpk = 0.0

        # In-control verification
        last_val = samples[-1]
        in_control = (lsl <= last_val <= usl) and (lcl <= last_val <= ucl)

        # Western Electric Rule: Trend of 7 consecutive points on one side of mean
        trend_warning = False
        if n >= 7:
            last_7 = samples[-7:]
            all_above = all(x > mean_val for x in last_7)
            all_below = all(x < mean_val for x in last_7)
            trend_warning = all_above or all_below

        st.update(
            mean=round(mean_val, 3),
            std_dev=round(std_val, 4),
            cp=round(cp, 2),
            cpk=round(cpk, 2),
            ucl=round(ucl, 3),
            lcl=round(lcl, 3),
            in_control=in_control,
            trend_warning=trend_warning,
        )

        stat_summary = (
            f"SPC: Cpk={round(cpk, 2)}, Cp={round(cp, 2)} | "
            f"Ort={round(mean_val, 3)}, σ={round(std_val, 4)} | "
            f"Limitler: [{round(lcl, 3)} .. {round(ucl, 3)}] (N={n})"
        )

        return {
            "mean": round(mean_val, 3),
            "std_dev": round(std_val, 4),
            "cp": round(cp, 2),
            "cpk": round(cpk, 2),
            "ucl": round(ucl, 3),
            "lcl": round(lcl, 3),
            "in_control": in_control,
            "trend_warning": trend_warning,
            "sample_count": n,
            "summary": stat_summary,
        }


# ==========================================================================================
# 3. Fast Template & Orientation Aligner (Şablon Doğrultucu / Hiza Bulucu)
# ==========================================================================================

@register
class FastTemplateAligner(Node):
    title = "Hızlı Parça Doğrultucu / Hiza Bulucu"
    category = CAT_IMG
    description = (
        "Konveyörden geçen parçanın eğim açısını ve merkezini PCA / Moment analizi ile tespit eder. "
        "Görüntüyü otomatik olarak dik konuma (0 derece) çevirip sonraki denetim bloklarına doğrultulmuş olarak aktarır."
    )
    inputs = [
        ("image", IMG),
        ("trigger", BOOL),
    ]
    outputs = [
        ("aligned_image", IMG),
        ("angle_deg", NUM),
        ("offset_x", NUM),
        ("offset_y", NUM),
        ("found", BOOL),
    ]
    params = [
        Param("min_contour_area", "float", 400.0, "Minimum Parça Alanı (px)", 10.0, 1000000.0),
        Param("target_angle", "float", 0.0, "Hedef Açı (derece)", -180.0, 180.0),
        Param("invert_binary", "bool", False, "Koyu Parça / Açık Zemin"),
    ]

    def setup(self):
        self.state.update(
            angle=0.0,
            ox=0.0,
            oy=0.0,
            found=False,
        )

    def process(self, inputs):
        img_in = inputs.get("image")
        if img_in is None:
            return {
                "aligned_image": None,
                "angle_deg": 0.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "found": False,
            }

        bgr = _bgr(img_in)
        gray = _gray(bgr)
        h, w = gray.shape[:2]

        thresh_type = cv2.THRESH_BINARY_INV if bool(self.v("invert_binary")) else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, thresh_type | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = float(self.v("min_contour_area"))

        valid_cnts = [c for c in contours if cv2.contourArea(c) >= min_a]
        if not valid_cnts:
            return {
                "aligned_image": bgr,
                "angle_deg": 0.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "found": False,
            }

        # Largest contour is the part
        best_cnt = max(valid_cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(best_cnt)
        (cx, cy), (rw, rh), raw_angle = rect

        # Normalize angle
        angle = raw_angle
        if rw < rh:
            angle = raw_angle + 90.0

        target_ang = float(self.v("target_angle"))
        rotation_needed = angle - target_ang

        # Rotate around center
        m_rot = cv2.getRotationMatrix2D((cx, cy), rotation_needed, 1.0)
        aligned = cv2.warpAffine(bgr, m_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        offset_x = round(cx - (w / 2.0), 1)
        offset_y = round(cy - (h / 2.0), 1)

        return {
            "aligned_image": aligned,
            "angle_deg": round(angle, 2),
            "offset_x": offset_x,
            "offset_y": offset_y,
            "found": True,
        }
