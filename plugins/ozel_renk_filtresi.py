"""Özel HSL Renk Filtresi Eklentisi:
Kullanıcının belirlediği HSV renk aralığını ayıklar, gürültüyü morfolojik olarak süzer
ve nesne sayısını ve alanını anlık hesaplar.
"""
from __future__ import annotations

import cv2
import numpy as np

from visionflow.core import Node, Param, PortType, register

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL


@register
class OzelRenkFiltresi(Node):
    title = "Özel Renk Filtresi (HSL/HSV)"
    category = "6 · Eklentiler"
    description = "Seçilen renk tonunu (Hue) maskeler, konturları bulur ve piksel sayısını hesaplar."
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("mask", IMG), ("pixel_count", NUM), ("detected", BOOL)]
    params = [
        Param("lower_hue", "int", 35, "Alt Ton (Hue 0-180)", 0, 180),
        Param("upper_hue", "int", 85, "Üst Ton (Hue 0-180)", 0, 180),
        Param("min_sat", "int", 60, "Minimum Doygunluk (Sat)", 0, 255),
        Param("min_val", "int", 50, "Minimum Parlaklık (Val)", 0, 255),
        Param("min_area", "int", 100, "Minimum Piksel Alanı (px)", 1, 100000),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        vis = img.copy()
        hsv = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV) if vis.ndim == 3 else vis

        lh = int(self.v("lower_hue"))
        uh = int(self.v("upper_hue"))
        sat = int(self.v("min_sat"))
        val = int(self.v("min_val"))

        lower = np.array([lh, sat, val], dtype=np.uint8)
        upper = np.array([uh, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morfolojik açma (gürültü temizleme)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Kontur tespiti ve çerçeveleme
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = int(self.v("min_area"))

        found_count = 0
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a >= min_a:
                found_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, f"Hedef A:{int(a)}", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        pixel_count = int(cv2.countNonZero(mask_clean))
        detected = (found_count > 0)

        # Üst bilgi bandı
        banner_col = (0, 180, 0) if detected else (0, 0, 180)
        status_txt = f"Renk Tespit Edildi ({found_count} parça, {pixel_count} px)" if detected else "Hedef Renk Bulunamadı"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(vis, status_txt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, banner_col, 1, cv2.LINE_AA)

        return {
            "image": vis,
            "mask": mask_clean,
            "pixel_count": pixel_count,
            "detected": detected,
        }
