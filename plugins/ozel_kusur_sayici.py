"""Özel Kusur Sayıcı ve Kalite Kapısı Eklentisi:
Görüntüdeki eşiklenmiş kusurları (leke, çapak, delik) sayar,
maksimum tolerans sınırını aşarsa HATALI (NG) alarmı ve görsel uyarı üretir.
"""
from __future__ import annotations

import cv2
import numpy as np

from visionflow.core import Node, Param, PortType, register

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT


@register
class OzelKusurSayici(Node):
    title = "Özel Kusur Sayıcı & Kalite Kapısı"
    category = "6 · Eklentiler"
    description = "Kusur adedini sayarak izin verilen tolerans sınırına göre anında OK/NG kalite kararı verir."
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("defect_count", NUM), ("is_ok", BOOL), ("result_text", TEXT)]
    params = [
        Param("threshold", "int", 130, "Piksel Eşik Seviyesi", 0, 255),
        Param("max_allowed_defects", "int", 0, "İzin Verilen Maksimum Kusur", 0, 50),
        Param("min_defect_area", "int", 30, "Minimum Kusur Alanı (px)", 1, 10000),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        vis = img.copy()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY) if vis.ndim == 3 else vis

        th = int(self.v("threshold"))
        _, bin_img = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = int(self.v("min_defect_area"))
        max_allowed = int(self.v("max_allowed_defects"))

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_a:
                count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(vis, f"K:{count}", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        is_ok = (count <= max_allowed)
        status = "KALITE: ONAYLANDI (OK)" if is_ok else f"KALITE: HATALI (NG) - {count} kusur > {max_allowed}"

        col = (0, 180, 0) if is_ok else (0, 0, 220)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(vis, status, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        return {
            "image": vis,
            "defect_count": count,
            "is_ok": is_ok,
            "result_text": status,
        }
