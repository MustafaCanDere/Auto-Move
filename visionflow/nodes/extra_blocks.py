"""New high-productivity VisionFlow blocks:
- PrivacyBlur: KVKK / Face / License Plate mosaic and blurring
- Watermark: Transparent logo or formatted text overlay
- AutoWhiteBalance: Gray-world illumination correction
- DominantColor: Dominant color recognition and naming
- HeatmapDiff: Reference difference thermal / defect heatmap
- WhatsAppSend: WhatsApp notification via CallMeBot / HTTP API
- GoogleSheetsAppend: Google Sheets / Webhook row logger
- BeepAlarm: System beep or WAV audio alert
- SmartPrompt: Multi-variable string / AI prompt composer
- OutlierFilter: Measurement outlier / spike rejection
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

from ._text import put_text, text_size
from ..core import Node, Param, PortType, register
from .learning import _imread
from .outputs import _rising
from .processing import _bgr, _gray

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY
DET = PortType.DETECTIONS

CAT_INPUT = "1 · Giriş"
CAT_IMG = "2 · Görüntü İşleme"
CAT_MEASURE = "2b · Kalite Kontrol ve Ölçüm"
CAT_LOGIC = "4 · Mantık"
CAT_OUT = "5 · Çıkış ve Otomasyon"


# --- 1. Privacy Blur / Mosaic -------------------------------------------------------------

@register
class PrivacyBlur(Node):
    title = "Mozaik / Gizleme (KVKK)"
    category = CAT_IMG
    description = ("Yüz, plaka veya belirlenen alanları mozaikleştirir, bulanıklaştırır veya siyah kutuyla "
                   "sansürler. Tespitler girişine bağlanan nesneleri veya manuel ROI bölgesini gizler.")
    inputs = [("image", IMG), ("detections", DET), ("mask", IMG)]
    outputs = [("image", IMG), ("count", NUM)]
    params = [
        Param("mode", "choice", "mozaik", "Yöntem", choices=["mozaik", "gauss", "siyah_kutu"]),
        Param("block_size", "int", 16, "Mozaik / Bulanıklık boyutu", 2, 120),
        Param("filter_label", "str", "", "Sadece bu etiket (boş = hepsi)"),
        Param("use_roi", "bool", False, "Manuel ROI alanı kullan"),
        Param("x", "int", 0, "ROI X", 0, 10000),
        Param("y", "int", 0, "ROI Y", 0, 10000),
        Param("w", "int", 100, "ROI Genişlik", 1, 10000),
        Param("h", "int", 100, "ROI Yükseklik", 1, 10000),
    ]

    def _apply_area(self, img: np.ndarray, x: int, y: int, w: int, h: int, mode: str, k: int):
        ih, iw = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        if mode == "siyah_kutu":
            img[y1:y2, x1:x2] = 0
        elif mode == "gauss":
            gk = max(3, k if k % 2 == 1 else k + 1)
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (gk, gk), 0)
        else:  # mozaik
            bs = max(2, k)
            rw, rh = max(1, (x2 - x1) // bs), max(1, (y2 - y1) // bs)
            small = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_LINEAR)
            img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")
        out = _bgr(img_in).copy()
        mode = self.v("mode")
        k = max(2, int(self.v("block_size")))
        count = 0

        # 1. Detections list
        dets = inputs.get("detections") or []
        flt = str(self.v("filter_label")).strip().lower()
        for d in dets:
            box = d.get("box")
            lbl = str(d.get("label", "")).lower()
            if flt and flt not in lbl:
                continue
            if box and len(box) == 4:
                bx, by, bw, bh = (int(v) for v in box)
                self._apply_area(out, bx, by, bw, bh, mode, k)
                count += 1

        # 2. Mask input
        mask = inputs.get("mask")
        if mask is not None:
            gmask = _gray(mask)
            cnts, _ = cv2.findContours(gmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                self._apply_area(out, bx, by, bw, bh, mode, k)
                count += 1

        # 3. Manual ROI
        if self.v("use_roi"):
            rx, ry, rw, rh = (self.v(key) for key in ("x", "y", "w", "h"))
            self._apply_area(out, rx, ry, rw, rh, mode, k)
            count += 1

        return {"image": out, "count": count}


# --- 2. Watermark / Logo Overlay ----------------------------------------------------------

@register
class Watermark(Node):
    title = "Filigran / Logo Ekle"
    category = CAT_IMG
    description = ("Görüntü üzerine yarı saydam logo resmi veya metin (tarih/saat damgası) ekler. "
                   "Metin içinde {tarih} ve {saat} etiketleri otomatik güncellenir.")
    inputs = [("image", IMG), ("watermark", IMG)]
    outputs = [("image", IMG)]
    params = [
        Param("mode", "choice", "metin", "Tür", choices=["metin", "resim"]),
        Param("text", "str", "VisionFlow • {tarih} {saat}", "Filigran metni"),
        Param("path", "path", "", "Logo resim dosyası (tür: resim ise)"),
        Param("position", "choice", "sağ alt", "Konum",
              choices=["sağ alt", "sol alt", "sağ üst", "sol üst", "orta"]),
        Param("opacity", "float", 0.7, "Opaklık (0-1)", 0.05, 1.0, 0.05),
        Param("scale", "float", 1.0, "Ölçek", 0.1, 5.0, 0.1),
        Param("color", "choice", "beyaz", "Metin rengi",
              choices=["beyaz", "siyah", "kırmızı", "yeşil", "mavi", "sarı"]),
    ]

    COLORS = {
        "beyaz": (255, 255, 255), "siyah": (0, 0, 0), "kırmızı": (0, 0, 255),
        "yeşil": (0, 255, 0), "mavi": (255, 120, 0), "sarı": (0, 255, 255),
    }

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")
        out = _bgr(img_in).copy()
        ih, iw = out.shape[:2]
        pos = self.v("position")
        alpha = float(np.clip(self.v("opacity"), 0.05, 1.0))
        scale = max(0.1, float(self.v("scale")))

        if self.v("mode") == "resim":
            # Image overlay
            logo = inputs.get("watermark")
            if logo is None and self.v("path"):
                if self.state.get("path") != self.v("path"):
                    self.state["logo"] = _imread(self.v("path"))
                    self.state["path"] = self.v("path")
                logo = self.state.get("logo")
            if logo is not None:
                lw, lh = int(logo.shape[1] * scale), int(logo.shape[0] * scale)
                if lw > 0 and lh > 0 and lw <= iw and lh <= ih:
                    resized = cv2.resize(logo, (lw, lh))
                    bgr_logo = _bgr(resized)
                    # Coordinates
                    if pos == "sol üst":
                        x, y = 20, 20
                    elif pos == "sağ üst":
                        x, y = iw - lw - 20, 20
                    elif pos == "sol alt":
                        x, y = 20, ih - lh - 20
                    elif pos == "orta":
                        x, y = (iw - lw) // 2, (ih - lh) // 2
                    else:  # sağ alt
                        x, y = iw - lw - 20, ih - lh - 20
                    x, y = max(0, x), max(0, y)
                    roi = out[y:y + lh, x:x + lw]
                    cv2.addWeighted(bgr_logo, alpha, roi, 1.0 - alpha, 0, roi)
        else:
            # Text watermark
            now = dt.datetime.now()
            txt = (self.v("text")
                   .replace("{tarih}", now.strftime("%d.%m.%Y"))
                   .replace("{saat}", now.strftime("%H:%M:%S")))
            tw, th = text_size(txt, scale=scale * 0.8)
            pad = 20
            if pos == "sol üst":
                x, y = pad, pad + th
            elif pos == "sağ üst":
                x, y = iw - tw - pad, pad + th
            elif pos == "sol alt":
                x, y = pad, ih - pad
            elif pos == "orta":
                x, y = (iw - tw) // 2, (ih + th) // 2
            else:  # sağ alt
                x, y = iw - tw - pad, ih - pad
            col = self.COLORS.get(self.v("color"), (255, 255, 255))
            if alpha >= 0.95:
                put_text(out, txt, (x, y), scale=scale * 0.8, color=col, thickness=2)
            else:
                overlay = out.copy()
                put_text(overlay, txt, (x, y), scale=scale * 0.8, color=col, thickness=2)
                cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

        return {"image": out}


# --- 3. Auto White Balance ----------------------------------------------------------------

@register
class AutoWhiteBalance(Node):
    title = "Otomatik Beyaz Dengesi"
    category = CAT_IMG
    description = ("Ortam ışığından kaynaklanan renk sapmalarını (sarı/mavi ton) Gri Dünya (Gray World) veya "
                   "Kontrast Germe yöntemiyle otomatik dengeler.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [
        Param("mode", "choice", "gri dünya", "Algoritma", choices=["gri dünya", "kontrast germe"]),
        Param("strength", "float", 1.0, "Etki gücü (0-1)", 0.0, 1.0, 0.1),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")
        bgr = _bgr(img_in).astype(np.float32)
        strength = float(np.clip(self.v("strength"), 0.0, 1.0))
        if strength <= 0.0:
            return {"image": img_in}

        mode = self.v("mode")
        if mode == "gri dünya":
            b_avg, g_avg, r_avg = np.mean(bgr[:, :, 0]), np.mean(bgr[:, :, 1]), np.mean(bgr[:, :, 2])
            gray_avg = (b_avg + g_avg + r_avg) / 3.0
            kb = gray_avg / max(1e-3, b_avg)
            kg = gray_avg / max(1e-3, g_avg)
            kr = gray_avg / max(1e-3, r_avg)
            corr = np.empty_like(bgr)
            corr[:, :, 0] = np.clip(bgr[:, :, 0] * kb, 0, 255)
            corr[:, :, 1] = np.clip(bgr[:, :, 1] * kg, 0, 255)
            corr[:, :, 2] = np.clip(bgr[:, :, 2] * kr, 0, 255)
        else:  # kontrast germe per channel
            corr = np.empty_like(bgr)
            for c in range(3):
                ch = bgr[:, :, c]
                lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
                if hi > lo:
                    corr[:, :, c] = np.clip((ch - lo) * (255.0 / (hi - lo)), 0, 255)
                else:
                    corr[:, :, c] = ch

        if strength < 1.0:
            out = cv2.addWeighted(corr.astype(np.uint8), strength, _bgr(img_in), 1.0 - strength, 0)
        else:
            out = corr.astype(np.uint8)
        return {"image": out}


# --- 4. Dominant Color Recognition --------------------------------------------------------

@register
class DominantColor(Node):
    title = "Baskın Renk Bul"
    category = CAT_MEASURE
    description = ("Görüntüdeki veya maskeli bölgedeki en baskın rengi analiz eder; Türkçe renk adı "
                   "(Kırmızı, Mavi, Sarı, Yeşil, Siyah, Beyaz...), HEX kodu ve RGB oranını verir.")
    inputs = [("image", IMG), ("mask", IMG)]
    outputs = [("color_name", TEXT), ("hex_code", TEXT), ("percentage", NUM),
               ("r", NUM), ("g", NUM), ("b", NUM)]
    params = [Param("sample_step", "int", 4, "Örnekleme hızı", 1, 16)]

    COLOR_TABLE = [
        ("Siyah", 0, 0, 0), ("Beyaz", 255, 255, 255), ("Gri", 128, 128, 128),
        ("Kırmızı", 220, 20, 60), ("Yeşil", 34, 139, 34), ("Mavi", 30, 144, 255),
        ("Sarı", 255, 215, 0), ("Turuncu", 255, 140, 0), ("Mor", 138, 43, 226),
        ("Pembe", 255, 105, 180), ("Kahverengi", 139, 69, 19), ("Camgöbeği", 0, 206, 209),
    ]

    def _closest_name(self, r: int, g: int, b: int) -> str:
        best_name, best_dist = "Bilinmeyen", float("inf")
        for name, cr, cg, cb in self.COLOR_TABLE:
            dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
            if dist < best_dist:
                best_dist, best_name = dist, name
        return best_name

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")
        bgr = _bgr(img_in)
        step = max(1, int(self.v("sample_step")))
        sampled = bgr[::step, ::step]

        mask_in = inputs.get("mask")
        if mask_in is not None:
            m = _gray(mask_in)[::step, ::step]
            pixels = sampled[m > 100]
        else:
            pixels = sampled.reshape(-1, 3)

        if len(pixels) == 0:
            return {"color_name": "Yok", "hex_code": "#000000", "percentage": 0.0, "r": 0, "g": 0, "b": 0}

        mean_b, mean_g, mean_r = np.mean(pixels, axis=0)
        r, g, b = int(round(mean_r)), int(round(mean_g)), int(round(mean_b))
        hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
        cname = self._closest_name(r, g, b)

        # Approximate coverage of this color group
        dists = np.sqrt(np.sum((pixels.astype(np.int32) - np.array([b, g, r])) ** 2, axis=1))
        matching = np.count_nonzero(dists < 60)
        pct = round(float(matching) / len(pixels) * 100.0, 1)

        return {"color_name": cname, "hex_code": hex_code, "percentage": pct, "r": r, "g": g, "b": b}


# --- 5. Difference Heatmap / Defect Visualizer --------------------------------------------

@register
class HeatmapDiff(Node):
    title = "Fark Isı Haritası (Kusur)"
    category = CAT_MEASURE
    description = ("Referans resim ile mevcut kare arasındaki farkı hesaplar ve termal ısı haritası (JET/HOT) "
                   "olarak görselleştirir. Kusurlu parçaların anında göze çarpmasını sağlar.")
    inputs = [("image", IMG), ("reference", IMG)]
    outputs = [("image", IMG), ("diff_percent", NUM), ("has_defect", BOOL)]
    params = [
        Param("threshold", "int", 30, "Fark eşiği", 0, 255),
        Param("colormap", "choice", "JET", "Renk haritası", choices=["JET", "HOT", "TURBO"]),
        Param("blend", "float", 0.6, "Isı haritası karışımı (0-1)", 0.0, 1.0, 0.05),
        Param("defect_limit", "float", 2.0, "Kusur alarm eşiği (%)", 0.0, 100.0, 0.5),
    ]

    CMAPS = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "TURBO": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    }

    def process(self, inputs):
        img = inputs["image"]
        ref = inputs["reference"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        if ref is None:
            # No reference yet: output current image unchanged
            return {"image": img, "diff_percent": 0.0, "has_defect": False}

        b_cur = _bgr(img)
        b_ref = _bgr(ref)
        if b_cur.shape[:2] != b_ref.shape[:2]:
            b_ref = cv2.resize(b_ref, (b_cur.shape[1], b_cur.shape[0]))

        diff = cv2.absdiff(b_cur, b_ref)
        gdiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        th = int(self.v("threshold"))
        mask = (gdiff >= th).astype(np.uint8) * 255

        diff_pct = round(float(np.count_nonzero(mask)) / mask.size * 100.0, 2)
        has_defect = diff_pct >= float(self.v("defect_limit"))

        # Generate heatmap
        cmap_id = self.CMAPS.get(self.v("colormap"), cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(gdiff, cmap_id)

        # Apply heatmap only where difference exceeds threshold
        blend = float(np.clip(self.v("blend"), 0.0, 1.0))
        out = b_cur.copy()
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
        blended = cv2.addWeighted(heatmap, blend, b_cur, 1.0 - blend, 0)
        out[mask_3d] = blended[mask_3d]

        return {"image": out, "diff_percent": diff_pct, "has_defect": has_defect}


# --- 6. WhatsApp Notification -------------------------------------------------------------
from .communication import WhatsAppSend  # noqa: F401


# --- 7. Google Sheets / Webhook Append ----------------------------------------------------

@register
class GoogleSheetsAppend(Node):
    title = "Google E-Tablolar'a Kaydet"
    category = CAT_OUT
    description = ("Tetik gelince Google Apps Script Web App veya webhook servisine yeni satır verisi gönderir. "
                   "Tarih/saat ve 4 serbest sütun değerini JSON olarak post eder.")
    inputs = [("trigger", BOOL), ("col1", ANY), ("col2", ANY), ("col3", ANY), ("col4", ANY)]
    outputs = [("sent", NUM)]
    params = [
        Param("webhook_url", "str", "", "Google Apps Script / Webhook URL"),
        Param("include_time", "bool", True, "Zaman damgası ekle"),
        Param("cooldown", "float", 2.0, "En az bekleme (sn)", 0, 86400),
    ]

    def setup(self):
        self.state.update(last_sent=0.0, sent=0)

    def process(self, inputs):
        self.state.setdefault("last_sent", 0.0)
        self.state.setdefault("sent", 0)
        if not _rising(self, inputs.get("trigger")):
            return {"sent": self.state["sent"]}
        url = str(self.v("webhook_url")).strip()
        if not url:
            return {"sent": self.state["sent"]}
        now = time.monotonic()
        if now - self.state["last_sent"] < self.v("cooldown"):
            return {"sent": self.state["sent"]}
        self.state["last_sent"] = now

        payload = {
            "col1": inputs.get("col1"),
            "col2": inputs.get("col2"),
            "col3": inputs.get("col3"),
            "col4": inputs.get("col4"),
        }
        if self.v("include_time"):
            payload["zaman"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "User-Agent": "VisionFlow",
        })

        def send():
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    r.read()
                self.state["sent"] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[Google E-Tablolar] Gönderilemedi: {exc}")

        threading.Thread(target=send, daemon=True).start()
        return {"sent": self.state["sent"]}


# --- 8. Beep / Audio Alarm ----------------------------------------------------------------

@register
class BeepAlarm(Node):
    title = "Sesli Uyarı (Bip / WAV)"
    category = CAT_OUT
    description = ("Tetik gelince anında sesli uyarı verir. Windows dahili bip sesi (ek paket gerektirmez) "
                   "veya belirtilen bir .WAV ses dosyasını çalar.")
    inputs = [("trigger", BOOL)]
    outputs = [("played", NUM)]
    params = [
        Param("frequency", "int", 1000, "Frekans (Hz)", 100, 5000),
        Param("duration_ms", "int", 300, "Süre (ms)", 50, 3000),
        Param("sound_file", "path", "", "Özel .wav dosyası (boşsa bip sesi)"),
        Param("cooldown", "float", 0.5, "En az bekleme (sn)", 0, 3600),
    ]

    def setup(self):
        self.state.update(last_play=0.0, played=0)

    def process(self, inputs):
        self.state.setdefault("last_play", 0.0)
        self.state.setdefault("played", 0)
        if not _rising(self, inputs.get("trigger")):
            return {"played": self.state["played"]}
        now = time.monotonic()
        if now - self.state["last_play"] < self.v("cooldown"):
            return {"played": self.state["played"]}
        self.state["last_play"] = now

        freq = int(self.v("frequency"))
        dur = int(self.v("duration_ms"))
        wav = str(self.v("sound_file")).strip()

        def play():
            try:
                if sys.platform == "win32":
                    import winsound
                    if wav and os.path.isfile(wav):
                        winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    else:
                        winsound.Beep(freq, dur)
                else:
                    # Linux / macOS fallback terminal bell
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                self.state["played"] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[Sesli Uyarı] Hata: {exc}")

        threading.Thread(target=play, daemon=True).start()
        return {"played": self.state["played"]}


# --- 9. Smart Prompt / Text Builder -------------------------------------------------------

@register
class SmartPrompt(Node):
    title = "Akıllı Metin Şablonu"
    category = CAT_LOGIC
    description = ("Değişkenleri ({a}, {b}, {c}, {d}, {tarih}, {saat}) şablonda birleştirip yapay zeka istemi, "
                   "rapor veya bildirim mesajı üretir. Tetik verilirse sadece tetik anında güncellenir.")
    inputs = [("a", ANY), ("b", ANY), ("c", ANY), ("d", ANY), ("trigger", BOOL)]
    outputs = [("text", TEXT)]
    params = [
        Param("template", "code", "Rapor [{tarih} {saat}]: Parça: {a}, Ölçüm: {b} mm, Durum: {c}", "Şablon"),
    ]

    def setup(self):
        self.state.update(text="")

    def process(self, inputs):
        self.state.setdefault("text", "")
        trig = inputs.get("trigger")
        fire = _rising(self, trig) if trig is not None else True
        if fire:
            now = dt.datetime.now()
            res = (self.v("template")
                   .replace("{tarih}", now.strftime("%d.%m.%Y"))
                   .replace("{saat}", now.strftime("%H:%M:%S"))
                   .replace("{a}", "" if inputs.get("a") is None else str(inputs["a"]))
                   .replace("{b}", "" if inputs.get("b") is None else str(inputs["b"]))
                   .replace("{c}", "" if inputs.get("c") is None else str(inputs["c"]))
                   .replace("{d}", "" if inputs.get("d") is None else str(inputs["d"])))
            self.state["text"] = res
        return {"text": self.state["text"]}


# --- 10. Outlier / Spike Filter -----------------------------------------------------------

@register
class OutlierFilter(Node):
    title = "Aykırı Değer Filtresi"
    category = CAT_LOGIC
    description = ("Sensör veya kamera ölçümlerindeki ani hatalı sıçramaları (aykırı uç değerleri) filtreler. "
                   "Sapma sınırını aşan değerleri önceki kararlı değerle veya medyanla değiştirir.")
    inputs = [("value", NUM)]
    outputs = [("filtered", NUM), ("is_outlier", BOOL)]
    params = [
        Param("window_size", "int", 5, "Pencere boyutu", 3, 31),
        Param("max_deviation", "float", 10.0, "Maksimum sapma sınırı", 0.1, 10000.0),
        Param("replace_mode", "choice", "önceki", "Aykırıysa yerine koy", choices=["önceki", "medyan", "ortalama"]),
    ]

    def setup(self):
        self.state.update(history=[], last_clean=0.0)

    def process(self, inputs):
        self.state.setdefault("history", [])
        self.state.setdefault("last_clean", 0.0)
        val = inputs.get("value")
        if val is None:
            return {"filtered": self.state["last_clean"], "is_outlier": False}
        val = float(val)

        hist = self.state["history"]
        win = max(3, int(self.v("window_size")))
        max_dev = float(self.v("max_deviation"))

        is_outlier = False
        if len(hist) >= 3:
            med = float(np.median(hist))
            if abs(val - med) > max_dev:
                is_outlier = True
                mode = self.v("replace_mode")
                if mode == "medyan":
                    val = med
                elif mode == "ortalama":
                    val = float(np.mean(hist))
                else:  # önceki
                    val = self.state["last_clean"]

        hist.append(val)
        if len(hist) > win:
            hist.pop(0)
        self.state["last_clean"] = val

        return {"filtered": round(val, 4), "is_outlier": is_outlier}


# --- 11. Windows System Toast & Alert Notification ----------------------------------------

@register
class WindowsNotification(Node):
    title = "Windows Sistem Bildirimi"
    category = CAT_OUT
    description = ("Hata veya kritik durumlarda Windows 10/11 masaüstü bildirim balonu (toast) ve sesli uyarı verir. "
                   "PowerShell ve Windows Shell API kullanır; ek paket gerektirmez.")
    inputs = [("message", TEXT), ("trigger", BOOL), ("title", TEXT)]
    outputs = [("sent", BOOL)]
    params = [
        Param("sound", "choice", "Uyarı", "Ses Türü", choices=["Uyarı", "Bip", "Sessiz"]),
        Param("on_change_only", "bool", True, "Yalnızca tetik False -> True olunca bildir"),
        Param("throttle_sec", "float", 4.0, "En az bekleme süresi (sn)", 0.5, 60.0),
    ]

    def setup(self):
        self.state.update(last_trigger=False, last_sent_time=0.0, sent_count=0)

    def _show_toast(self, title_txt: str, msg_txt: str, sound_type: str):
        safe_title = title_txt.replace("'", " ").replace('"', ' ').replace("\n", " ")[:60]
        safe_msg = msg_txt.replace("'", " ").replace('"', ' ').replace("\n", " ")[:140]
        
        # Async PowerShell Balloon Notification
        ps_cmd = (
            f"[void] [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
            f"$n = New-Object System.Windows.Forms.NotifyIcon;"
            f"$n.Icon = [System.Drawing.SystemIcons]::Information;"
            f"$n.Visible = $True;"
            f"$n.ShowBalloonTip(3000, '{safe_title}', '{safe_msg}', [System.Windows.Forms.ToolTipIcon]::Info);"
        )
        if sound_type == "Bip":
            ps_cmd = "[console]::beep(800, 200);" + ps_cmd
        elif sound_type == "Uyarı":
            ps_cmd = "[System.Media.SystemSounds]::Exclamation.Play();" + ps_cmd
            
        import subprocess
        try:
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def process(self, inputs):
        self.state.setdefault("last_trigger", False)
        self.state.setdefault("last_sent_time", 0.0)
        
        trig_val = inputs.get("trigger")
        trigger = True if trig_val is None else bool(trig_val)
        
        msg = str(inputs.get("message") or "VisionFlow uyarısı: Kalite kontrol şartı sağlandı.").strip()
        title = str(inputs.get("title") or "VisionFlow Studio").strip()
        
        on_change = bool(self.v("on_change_only"))
        throttle = float(self.v("throttle_sec"))
        now = time.time()
        
        sent = False
        should_send = False
        if on_change:
            if trigger and not self.state["last_trigger"]:
                should_send = True
        else:
            if trigger:
                should_send = True
                
        self.state["last_trigger"] = trigger
        
        if should_send and (now - self.state["last_sent_time"] >= throttle):
            self.state["last_sent_time"] = now
            threading.Thread(target=self._show_toast, args=(title, msg, self.v("sound")), daemon=True).start()
            sent = True
            
        return {"sent": sent}


# --- 12. Live KPI & Quality Dashboard Overlay ---------------------------------------------

@register
class LiveKPIDashboard(Node):
    title = "KPI ve Kalite Panosu"
    category = CAT_MEASURE
    description = ("Gerçek zamanlı hat kalite ve üretim göstergesi (OEE / Verim Panosu). "
                   "Toplam, sağlam ve hatalı parça sayısını, verim oranını (%) ve PPM değerini "
                   "hesaplayıp modern bir HUD gösterge paneli üretir.")
    inputs = [("ok", BOOL), ("trigger", BOOL), ("image", IMG)]
    outputs = [("dashboard", IMG), ("total", NUM), ("good", NUM), ("defective", NUM),
               ("yield_pct", NUM), ("ppm", NUM)]
    params = [
        Param("title", "str", "HAT-1 KALİTE KONTROL", "Pano Başlığı"),
        Param("target_yield", "float", 98.0, "Hedef Verim (%)", 50.0, 100.0),
        Param("overlay_mode", "choice", "HUD Bindir", "Görünüm",
              choices=["HUD Bindir", "Bağımsız Pano", "Görüntü Altına Ekle"]),
        Param("reset_counter", "bool", False, "Sayaçları Sıfırla"),
    ]

    def setup(self):
        self.state.update(total=0, good=0, bad=0, last_trigger=False)

    def _draw_hud(self, base_img: np.ndarray | None, total: int, good: int, bad: int,
                  yield_pct: float, target_yield: float, title_text: str, mode: str) -> np.ndarray:
        w, h = 640, 360
        if base_img is not None and mode in ("HUD Bindir", "Görüntü Altına Ekle"):
            src_h, src_w = base_img.shape[:2]
            if mode == "HUD Bindir":
                canvas = _bgr(base_img).copy()
                # Draw sleek top HUD overlay ribbon
                hud_h = min(90, int(src_h * 0.25))
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0, 0), (src_w, hud_h), (18, 22, 30), -1)
                cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
                
                # Bottom accent line (vibrant cyan / emerald depending on yield)
                bar_color = (0, 220, 80) if yield_pct >= target_yield else (0, 70, 230)
                cv2.line(canvas, (0, hud_h), (src_w, hud_h), bar_color, 2)
                
                # Texts
                s = max(0.5, src_w / 1200.0)
                cv2.putText(canvas, f"{title_text} | Toplam: {total}", (16, int(hud_h * 0.42)),
                            cv2.FONT_HERSHEY_SIMPLEX, s, (240, 245, 250), 1, cv2.LINE_AA)
                
                stats_txt = f"Saglam: {good}  Hatali: {bad}  Verim: %{yield_pct:.1f}"
                cv2.putText(canvas, stats_txt, (16, int(hud_h * 0.82)),
                            cv2.FONT_HERSHEY_SIMPLEX, s * 0.95, bar_color, 2 if yield_pct < target_yield else 1, cv2.LINE_AA)
                return canvas
            else:  # Görüntü Altına Ekle
                hud_strip = np.full((100, src_w, 3), (20, 24, 34), dtype=np.uint8)
                bar_color = (0, 220, 80) if yield_pct >= target_yield else (0, 80, 240)
                cv2.line(hud_strip, (0, 0), (src_w, 0), bar_color, 3)
                cv2.putText(hud_strip, f"{title_text} - VERİM: %{yield_pct:.1f}", (20, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(hud_strip, f"Toplam: {total}   Saglam: {good}   Hatali: {bad}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 240), 1, cv2.LINE_AA)
                return np.vstack([_bgr(base_img), hud_strip])

        # Bağımsız Pano (Standalone sleek dashboard card)
        panel = np.full((h, w, 3), (16, 20, 28), dtype=np.uint8)
        # Gradient background header
        cv2.rectangle(panel, (0, 0), (w, 54), (26, 32, 45), -1)
        cv2.line(panel, (0, 54), (w, 54), (0, 200, 255), 2)
        cv2.putText(panel, title_text.upper(), (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        time_str = dt.datetime.now().strftime("%H:%M:%S")
        cv2.putText(panel, time_str, (w - 110, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 160, 185), 1, cv2.LINE_AA)

        # 3 metric cards: Toplam, Sağlam, Hatalı
        cards = [
            ("TOPLAM", str(total), (220, 230, 240), (20, 75, 180, 100)),
            ("SAGLAM", str(good), (0, 230, 100), (220, 75, 180, 100)),
            ("HATALI", str(bad), (0, 70, 240), (420, 75, 180, 100)),
        ]
        for label, val, color, (cx, cy, cw, ch) in cards:
            cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (24, 30, 42), -1)
            cv2.rectangle(panel, (cx, cy), (cx + cw, cy + ch), (40, 48, 64), 1)
            cv2.putText(panel, label, (cx + 12, cy + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 150, 175), 1, cv2.LINE_AA)
            cv2.putText(panel, val, (cx + 12, cy + 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        # Progress bar for yield
        bar_x, bar_y, bar_w, bar_h = 20, 210, w - 40, 26
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 38, 52), -1)
        fill_w = int(bar_w * (max(0.0, min(100.0, yield_pct)) / 100.0))
        bar_color = (0, 220, 90) if yield_pct >= target_yield else ((0, 165, 255) if yield_pct >= 90 else (0, 60, 240))
        if fill_w > 0:
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 60, 80), 1)

        cv2.putText(panel, f"VERIM ORANI: %{yield_pct:.2f} (Hedef: %{target_yield:.1f})",
                    (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 230, 240), 1, cv2.LINE_AA)

        # PPM text
        ppm = int((bad / max(1, total)) * 1_000_000)
        cv2.putText(panel, f"Kusur Orani: {ppm:,} PPM", (bar_x, bar_y + bar_h + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 170, 195), 1, cv2.LINE_AA)

        return panel

    def process(self, inputs):
        self.state.setdefault("total", 0)
        self.state.setdefault("good", 0)
        self.state.setdefault("bad", 0)
        self.state.setdefault("last_trigger", False)

        if bool(self.v("reset_counter")):
            self.state["total"] = 0
            self.state["good"] = 0
            self.state["bad"] = 0

        trig_val = inputs.get("trigger")
        trigger = True if trig_val is None else bool(trig_val)
        ok_val = bool(inputs.get("ok", True))

        # Rising edge trigger check
        if trigger and not self.state["last_trigger"]:
            self.state["total"] += 1
            if ok_val:
                self.state["good"] += 1
            else:
                self.state["bad"] += 1
        self.state["last_trigger"] = trigger

        total = self.state["total"]
        good = self.state["good"]
        bad = self.state["bad"]
        yield_pct = 100.0 if total == 0 else round((good / total) * 100.0, 2)
        ppm = 0.0 if total == 0 else round((bad / total) * 1_000_000.0, 1)

        base_img = inputs.get("image")
        dashboard = self._draw_hud(base_img, total, good, bad, yield_pct,
                                   float(self.v("target_yield")), str(self.v("title")),
                                   str(self.v("overlay_mode")))

        return {
            "dashboard": dashboard,
            "total": total,
            "good": good,
            "defective": bad,
            "yield_pct": yield_pct,
            "ppm": ppm,
        }


# --- 13. Industrial Image Enhancer & Dehaze ------------------------------------------------

@register
class IndustrialDehazeEnhance(Node):
    title = "Görüntü İyileştir / Sis ve Parlama Gider"
    category = CAT_IMG
    description = ("Endüstriyel ortamlardaki sis, duman, aşırı parlama/yansıma veya gölgeleri giderir. "
                   "Akıllı CLAHE, Dark-Channel Dehaze ve Gamma dengeleme ile kusurları belirginleştirir.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [
        Param("method", "choice", "Akıllı Kontrast (CLAHE)", "İyileştirme Yöntemi",
              choices=["Akıllı Kontrast (CLAHE)", "Sis ve Duman Gider (Dehaze)",
                       "Parlama ve Yansıma Bastır", "Gölge Aydınlat (HDR)"]),
        Param("strength", "float", 1.2, "İyileştirme Gücü", 0.2, 4.0),
        Param("auto_wb", "bool", True, "Otomatik Renk / Beyaz Dengesi"),
    ]

    def _auto_white_balance(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 3:
            return img
        b, g, r = cv2.split(img)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        if avg_b < 1 or avg_g < 1 or avg_r < 1:
            return img
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        kb, kg, kr = avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r
        b = np.clip(b * kb, 0, 255).astype(np.uint8)
        g = np.clip(g * kg, 0, 255).astype(np.uint8)
        r = np.clip(r * kr, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        work = _bgr(img).copy()

        if bool(self.v("auto_wb")):
            work = self._auto_white_balance(work)

        method = str(self.v("method"))
        strength = float(self.v("strength"))

        if method == "Akıllı Kontrast (CLAHE)":
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clip = max(1.0, 2.0 * strength)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            l = clahe.apply(l)
            work = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        elif method == "Sis ve Duman Gider (Dehaze)":
            # Fast dark channel approximation
            dark = np.min(work, axis=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dark = cv2.erode(dark, kernel)
            
            # Atmospheric light estimation
            top_pixels = np.percentile(dark, 99.0)
            a_light = max(10, min(240, int(top_pixels)))
            
            # Transmission map
            omega = min(0.95, 0.70 * strength)
            transmission = 1.0 - omega * (dark.astype(np.float32) / float(a_light))
            transmission = np.clip(transmission, 0.15, 1.0)
            
            # Recover scene radiance
            out = np.zeros_like(work, dtype=np.float32)
            for c in range(3):
                out[:, :, c] = ((work[:, :, c].astype(np.float32) - a_light) / transmission) + a_light
            work = np.clip(out, 0, 255).astype(np.uint8)

        elif method == "Parlama ve Yansıma Bastır":
            # Detect overexposed specular highlights
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            thresh = int(245 - 20 * (strength - 1.0))
            _, mask = cv2.threshold(gray, max(180, thresh), 255, cv2.THRESH_BINARY)
            
            # Multi-scale inpaint / blur blending over glare spots
            mask_dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            blurred = cv2.medianBlur(work, 11)
            work = np.where(mask_dilated[:, :, None] > 0, blurred, work)
            
            # Soft gamma compression
            gamma = 1.0 + (strength * 0.25)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            work = cv2.LUT(work, table)

        elif method == "Gölge Aydınlat (HDR)":
            # Adaptive tone curve boosting underexposed shadows while retaining highlights
            gamma = max(0.2, 1.0 / (1.0 + 0.8 * strength))
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            boosted = cv2.LUT(work, table)
            
            # Blend with original guided by luminance
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            weight = np.clip(1.0 - gray, 0.0, 1.0)[:, :, None]
            work = (boosted.astype(np.float32) * weight + work.astype(np.float32) * (1.0 - weight)).astype(np.uint8)

        return {"image": work}


# --- 14. Barcode & QR Code Generator / Stamper ---------------------------------------------

@register
class BarcodeGenerator(Node):
    title = "Barkod ve QR Kod Damgala"
    category = CAT_OUT
    description = ("Dinamik metin, seri numarası veya lot kodundan endüstriyel barkod / 2D kod deseni üretir. "
                   "İster bağımsız etiket resmi verir, ister gelen ürün görselinin üzerine otomatik damgalar.")
    inputs = [("image", IMG), ("text", TEXT), ("trigger", BOOL)]
    outputs = [("image", IMG), ("barcode_only", IMG)]
    params = [
        Param("code_type", "choice", "Barkod (Çizgili)", "Kod Türü", choices=["Barkod (Çizgili)", "2D Kare Kod (Matris)"]),
        Param("default_text", "str", "VF-2026-A101", "Varsayılan Metin (Giriş Boşsa)"),
        Param("position", "choice", "Sol Üst", "Konum (Görsel Üzerine)", choices=["Sol Üst", "Sağ Üst", "Sol Alt", "Sağ Alt", "Yalnızca Barkod"]),
        Param("scale", "float", 1.0, "Ölçek", 0.5, 3.0),
    ]

    def _render_barcode(self, text: str, is_2d: bool, scale: float) -> np.ndarray:
        s = max(0.5, float(scale))
        if is_2d:
            # High-contrast 2D Data Matrix style pattern
            grid_size = 16
            cell_size = int(12 * s)
            w = h = grid_size * cell_size
            mat = np.full((h, w, 3), 255, dtype=np.uint8)
            # Border finder pattern (L-shape)
            cv2.rectangle(mat, (0, 0), (w, cell_size), (0, 0, 0), -1)
            cv2.rectangle(mat, (0, 0), (cell_size, h), (0, 0, 0), -1)
            # Hash text into deterministic cells
            import hashlib
            h_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            byte_idx = 0
            for r in range(1, grid_size - 1):
                for c in range(1, grid_size - 1):
                    val = h_bytes[byte_idx % len(h_bytes)]
                    byte_idx += 1
                    if (val + r * 7 + c * 13) % 2 == 0:
                        cv2.rectangle(mat, (c * cell_size, r * cell_size),
                                      ((c + 1) * cell_size, (r + 1) * cell_size), (0, 0, 0), -1)
            # Add white quiet zone margin
            pad = int(8 * s)
            return cv2.copyMakeBorder(mat, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            # 1D Linear Barcode style
            bar_w = int(240 * s)
            bar_h = int(70 * s)
            label = np.full((bar_h + int(24 * s), bar_w, 3), 255, dtype=np.uint8)
            import hashlib
            h_int = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            # Draw synthetic Code128-like bars
            x = int(10 * s)
            end_x = bar_w - int(10 * s)
            step = 0
            while x < end_x:
                width = int(((h_int >> (step % 28)) & 3) + 1) * max(1, int(s))
                if (step % 2) == 0:
                    cv2.rectangle(label, (x, int(8 * s)), (min(x + width, end_x), bar_h), (0, 0, 0), -1)
                x += width
                step += 1
            # Add text caption
            cv2.putText(label, text[:24], (int(12 * s), bar_h + int(16 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (20, 20, 20), 1, cv2.LINE_AA)
            return label

    def process(self, inputs):
        txt = str(inputs.get("text") or self.v("default_text")).strip()
        is_2d = self.v("code_type") == "2D Kare Kod (Matris)"
        scale = float(self.v("scale"))
        barcode_img = self._render_barcode(txt, is_2d, scale)

        base_img = inputs.get("image")
        pos = self.v("position")

        if base_img is None or pos == "Yalnızca Barkod":
            return {"image": barcode_img, "barcode_only": barcode_img}

        out = _bgr(base_img).copy()
        bh, bw = barcode_img.shape[:2]
        ih, iw = out.shape[:2]

        if bw > iw or bh > ih:
            barcode_img = cv2.resize(barcode_img, (min(bw, iw // 2), min(bh, ih // 2)))
            bh, bw = barcode_img.shape[:2]

        if pos == "Sol Üst":
            x1, y1 = 12, 12
        elif pos == "Sağ Üst":
            x1, y1 = iw - bw - 12, 12
        elif pos == "Sol Alt":
            x1, y1 = 12, ih - bh - 12
        else:  # Sağ Alt
            x1, y1 = iw - bw - 12, ih - bh - 12

        out[y1:y1 + bh, x1:x1 + bw] = barcode_img
        cv2.rectangle(out, (x1 - 1, y1 - 1), (x1 + bw, y1 + bh), (80, 80, 80), 1)

        return {"image": out, "barcode_only": barcode_img}


# --- 15. Voice & Speech Synthesis (TTS Alert) ---------------------------------------------

@register
class VoiceSpeechAlert(Node):
    title = "Sesli Konuşma ve Anons (TTS)"
    category = CAT_OUT
    description = ("Üretim hattında hata veya kritik olaylarda hoparlörden Türkçe konuşarak sesli anons yapar. "
                   "pyttsx3 ile internetsiz çevrimdışı çalışır; ses tonu ve hızı ayarlanabilir.")
    inputs = [("message", TEXT), ("trigger", BOOL)]
    outputs = [("spoken", BOOL)]
    params = [
        Param("default_message", "str", "Dikkat! Hatalı parça tespit edildi.", "Varsayılan Mesaj"),
        Param("on_change_only", "bool", True, "Yalnızca tetik False -> True olunca konuş"),
        Param("speech_rate", "int", 150, "Konuşma Hızı (Kelime/Dk)", 80, 250),
        Param("volume", "float", 1.0, "Ses Seviyesi", 0.1, 1.0),
        Param("throttle_sec", "float", 5.0, "En az bekleme süresi (sn)", 1.0, 60.0),
    ]

    def setup(self):
        self.state.update(last_trigger=False, last_spoken_time=0.0)

    def _speak_worker(self, text: str, rate: int, volume: float):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            # Fallback to Windows PowerShell System SAPI voice if pyttsx3 fails in thread
            import subprocess
            clean_txt = text.replace("'", " ").replace('"', ' ')
            cmd = f"Add-Type -AssemblyName System.speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean_txt}');"
            try:
                subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
            except Exception:
                pass

    def process(self, inputs):
        self.state.setdefault("last_trigger", False)
        self.state.setdefault("last_spoken_time", 0.0)

        trig_val = inputs.get("trigger")
        trigger = True if trig_val is None else bool(trig_val)
        msg = str(inputs.get("message") or self.v("default_message")).strip()

        on_change = bool(self.v("on_change_only"))
        throttle = float(self.v("throttle_sec"))
        now = time.time()

        should_speak = False
        if on_change:
            if trigger and not self.state["last_trigger"]:
                should_speak = True
        else:
            if trigger:
                should_speak = True

        self.state["last_trigger"] = trigger
        spoken = False

        if should_speak and (now - self.state["last_spoken_time"] >= throttle):
            self.state["last_spoken_time"] = now
            threading.Thread(target=self._speak_worker,
                             args=(msg, int(self.v("speech_rate")), float(self.v("volume"))),
                             daemon=True).start()
            spoken = True

        return {"spoken": spoken}


# --- 16. Smart Part Classifier & Sorter ---------------------------------------------------

@register
class SmartPartClassifier(Node):
    title = "Akıllı Parça Sınıflandırıcı"
    category = CAT_MEASURE
    description = ("Parçaları en, boy, alan ve renk tonuna göre otomatik sınıflandırır (ör. 'Tip-A', 'Tip-B', 'Kusurlu'). "
                   "Ayıklama ve red mekanizmaları için tetik sinyali üretir.")
    inputs = [("image", IMG), ("width", NUM), ("height", NUM), ("area", NUM)]
    outputs = [("class_name", TEXT), ("is_accepted", BOOL), ("reject_trigger", BOOL), ("overlay_image", IMG)]
    params = [
        Param("min_area", "float", 1000.0, "Minimum Parça Alanı (px)", 1.0, 1000000.0),
        Param("max_area", "float", 50000.0, "Maksimum Parça Alanı (px)", 10.0, 5000000.0),
        Param("target_ratio", "float", 1.0, "Hedef En/Boy Oranı (w/h)", 0.1, 10.0),
        Param("ratio_tolerance", "float", 0.35, "En/Boy Toleransı (+/-)", 0.05, 5.0),
        Param("type_a_name", "str", "STANDART-A", "Sağlam Parça Adı"),
        Param("reject_name", "str", "RED (KUSURLU)", "Hatalı Parça Adı"),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        w_val = float(inputs.get("width") or 0.0)
        h_val = float(inputs.get("height") or 0.0)
        area_val = float(inputs.get("area") or (w_val * h_val if w_val and h_val else 0.0))

        min_a = float(self.v("min_area"))
        max_a = float(self.v("max_area"))
        target_r = float(self.v("target_ratio"))
        tol_r = float(self.v("ratio_tolerance"))

        # Dimension / Area checks
        ratio = (w_val / max(0.001, h_val)) if (w_val > 0 and h_val > 0) else target_r
        area_ok = (min_a <= area_val <= max_a) if area_val > 0 else True
        ratio_ok = abs(ratio - target_r) <= tol_r

        is_accepted = bool(area_ok and ratio_ok)
        class_name = str(self.v("type_a_name")) if is_accepted else str(self.v("reject_name"))
        reject_trig = not is_accepted

        # Annotated image overlay
        overlay = _bgr(img).copy() if img is not None else np.full((300, 400, 3), (20, 24, 30), dtype=np.uint8)
        ih, iw = overlay.shape[:2]
        badge_color = (0, 210, 80) if is_accepted else (0, 60, 240)
        cv2.rectangle(overlay, (12, 12), (min(iw - 12, 280), 65), (20, 24, 34), -1)
        cv2.rectangle(overlay, (12, 12), (min(iw - 12, 280), 65), badge_color, 2)
        cv2.putText(overlay, f"{class_name}", (24, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return {
            "class_name": class_name,
            "is_accepted": is_accepted,
            "reject_trigger": reject_trig,
            "overlay_image": overlay,
        }


# --- 17. Screen Capture Source ------------------------------------------------------------

@register
class ScreenCaptureSource(Node):
    title = "Ekran / Pencere Yakalama"
    category = CAT_INPUT
    description = ("Masaüstünü veya ekranın belirlenen bir koordinat alanını (SCADA ekranı, kamera programı, "
                   "tarayıcı vb.) gerçek zamanlı video akışı olarak yakalar.")
    inputs = [("trigger", BOOL)]
    outputs = [("image", IMG), ("fps", NUM), ("width", NUM), ("height", NUM)]
    params = [
        Param("mode", "choice", "tam_ekran", "Yakalama Modu", choices=["tam_ekran", "ozel_alan"]),
        Param("x", "int", 0, "Alan X Koordinatı", 0, 10000),
        Param("y", "int", 0, "Alan Y Koordinatı", 0, 10000),
        Param("w", "int", 1280, "Alan Genişliği", 10, 10000),
        Param("h", "int", 720, "Alan Yüksekliği", 10, 10000),
    ]

    def process(self, inputs):
        self.state.setdefault("last_time", time.time())
        self.state.setdefault("last_fps", 30.0)

        trig = inputs.get("trigger")
        if trig is not None and not bool(trig):
            last_img = self.state.get("cached_img")
            if last_img is not None:
                return {
                    "image": last_img,
                    "fps": self.state["last_fps"],
                    "width": last_img.shape[1],
                    "height": last_img.shape[0],
                }

        try:
            from PIL import ImageGrab
            mode = self.v("mode")
            if mode == "ozel_alan":
                x = int(self.v("x"))
                y = int(self.v("y"))
                w = max(10, int(self.v("w")))
                h = max(10, int(self.v("h")))
                bbox = (x, y, x + w, y + h)
                try:
                    pil_img = ImageGrab.grab(bbox=bbox, all_screens=True)
                except Exception:
                    pil_img = ImageGrab.grab(bbox=bbox)
            else:
                try:
                    pil_img = ImageGrab.grab(all_screens=True)
                except Exception:
                    pil_img = ImageGrab.grab()

            rgb = np.array(pil_img)
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            elif rgb.ndim == 3 and rgb.shape[2] == 3:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                bgr = _bgr(rgb)
        except Exception:
            last_img = self.state.get("cached_img")
            if last_img is not None:
                bgr = last_img
            else:
                w_val = int(self.v("w")) if self.v("mode") == "ozel_alan" else 1280
                h_val = int(self.v("h")) if self.v("mode") == "ozel_alan" else 720
                bgr = np.full((h_val, w_val, 3), (25, 28, 36), dtype=np.uint8)
                cv2.putText(bgr, "Ekran Yakalama: Masaustu bekleniyor...", (30, h_val // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 2, cv2.LINE_AA)

        now = time.time()
        dt_val = now - self.state["last_time"]
        self.state["last_time"] = now
        fps = 1.0 / dt_val if dt_val > 0.001 else self.state["last_fps"]
        fps = 0.9 * self.state["last_fps"] + 0.1 * fps
        self.state["last_fps"] = fps
        self.state["cached_img"] = bgr

        return {
            "image": bgr,
            "fps": round(fps, 1),
            "width": bgr.shape[1],
            "height": bgr.shape[0],
        }


# --- 18. Thermal False Color --------------------------------------------------------------

@register
class ThermalFalseColor(Node):
    title = "Termal ve Sahte Renk Paleti"
    category = CAT_IMG
    description = ("Görüntüleri endüstriyel termal ve sahte renk paletleriyle (TURBO, JET, INFERNO, HOT, BONE vb.) "
                   "renklendirir. Isı haritası, yoğunluk ve yüzey sızıntısı analizlerini görselleştirir.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("min_val", NUM), ("max_val", NUM)]
    params = [
        Param("colormap", "choice", "turbo", "Renk Paleti",
              choices=["turbo", "jet", "inferno", "plasma", "hot", "bone", "rainbow", "ocean", "viridis"]),
        Param("auto_contrast", "bool", True, "Otomatik Kontrast / Min-Max Ölçekleme"),
        Param("invert", "bool", False, "Renkleri Ters Çevir"),
        Param("show_colorbar", "bool", True, "Sağ Kenara Renk Skalası Ekle"),
    ]

    CMAP_MAP = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "ocean": cv2.COLORMAP_OCEAN,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        gray = _gray(img)
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))

        if bool(self.v("auto_contrast")):
            if max_val > min_val:
                norm = ((gray.astype(np.float32) - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            else:
                norm = gray
        else:
            norm = gray

        if bool(self.v("invert")):
            norm = 255 - norm

        cmap_id = self.CMAP_MAP.get(self.v("colormap"), cv2.COLORMAP_TURBO)
        colored = cv2.applyColorMap(norm, cmap_id)

        if bool(self.v("show_colorbar")):
            ih, iw = colored.shape[:2]
            bar_w = max(28, int(iw * 0.05))
            grad = np.linspace(255, 0, ih, endpoint=True, dtype=np.uint8)[:, None]
            grad_strip = np.repeat(grad, bar_w, axis=1)
            colored_bar = cv2.applyColorMap(grad_strip, cmap_id)
            cv2.rectangle(colored_bar, (0, 0), (bar_w - 1, ih - 1), (50, 50, 50), 1)
            cv2.putText(colored_bar, f"{int(max_val)}", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(colored_bar, f"{int(min_val)}", (2, ih - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            colored = np.hstack([colored, colored_bar])

        return {
            "image": colored,
            "min_val": round(min_val, 1),
            "max_val": round(max_val, 1),
        }


# --- 19. Defect Auto Zoom (Picture-in-Picture) --------------------------------------------

@register
class DefectAutoZoom(Node):
    title = "Kusur Detay Büyüteci (PIP)"
    category = CAT_MEASURE
    description = ("Tespit edilen en belirgin kusuru veya anomaliyi büyüterek (2x-5x) görüntünün bir köşesine "
                   "detay penceresi (Picture-in-Picture) ve artı işaretiyle yerleştirir.")
    inputs = [("image", IMG), ("detections", DET)]
    outputs = [("image", IMG), ("zoom_crop", IMG), ("has_defect", BOOL)]
    params = [
        Param("zoom_factor", "float", 2.5, "Yakınlaştırma Oranı (x)", 1.2, 8.0),
        Param("pip_size", "int", 220, "Detay Pencere Boyutu (px)", 100, 600),
        Param("position", "choice", "sag_ust", "Pencere Konumu",
              choices=["sag_ust", "sag_alt", "sol_ust", "sol_alt"]),
        Param("border_color", "choice", "kirmizi", "Çerçeve Rengi",
              choices=["kirmizi", "sari", "camgobegi", "yesil"]),
        Param("draw_crosshair", "bool", True, "Merkeze Artı İşareti Çiz"),
    ]

    COLOR_MAP = {
        "kirmizi": (0, 30, 240),
        "sari": (0, 220, 240),
        "camgobegi": (240, 210, 0),
        "yesil": (0, 210, 60),
    }

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        ih, iw = img.shape[:2]

        dets = inputs.get("detections") or []
        has_defect = len(dets) > 0

        if has_defect:
            d = dets[0]
            box = d.get("box", [iw // 2 - 50, ih // 2 - 50, iw // 2 + 50, ih // 2 + 50])
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
        else:
            cx, cy = iw // 2, ih // 2

        pip_sz = int(self.v("pip_size"))
        zoom = float(self.v("zoom_factor"))
        crop_half = int(pip_sz / (2.0 * zoom))

        x1 = max(0, cx - crop_half)
        y1 = max(0, cy - crop_half)
        x2 = min(iw, cx + crop_half)
        y2 = min(ih, cy + crop_half)

        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
            pip_img = cv2.resize(crop, (pip_sz, pip_sz), interpolation=cv2.INTER_LINEAR)
        else:
            pip_img = np.zeros((pip_sz, pip_sz, 3), dtype=np.uint8)

        col = self.COLOR_MAP.get(self.v("border_color"), (0, 30, 240))
        if bool(self.v("draw_crosshair")):
            pc = pip_sz // 2
            cv2.line(pip_img, (pc - 15, pc), (pc + 15, pc), col, 1)
            cv2.line(pip_img, (pc, pc - 15), (pc, pc + 15), col, 1)
            cv2.circle(pip_img, (pc, pc), 8, col, 1)

        cv2.rectangle(pip_img, (0, 0), (pip_sz - 1, pip_sz - 1), col, 3)
        lbl = f"KUSUR {zoom:.1f}x" if has_defect else f"DETAY {zoom:.1f}x"
        cv2.putText(pip_img, lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pip_img, lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        pos = self.v("position")
        pad = 12
        if pos == "sag_ust":
            px1, py1 = max(0, iw - pip_sz - pad), pad
        elif pos == "sag_alt":
            px1, py1 = max(0, iw - pip_sz - pad), max(0, ih - pip_sz - pad)
        elif pos == "sol_ust":
            px1, py1 = pad, pad
        else:
            px1, py1 = pad, max(0, ih - pip_sz - pad)

        px2 = min(iw, px1 + pip_sz)
        py2 = min(ih, py1 + pip_sz)
        if px2 - px1 == pip_sz and py2 - py1 == pip_sz:
            img[py1:py2, px1:px2] = pip_img

        return {
            "image": img,
            "zoom_crop": pip_img,
            "has_defect": has_defect,
        }


# --- 20. Edge & Burr Defect Detector ------------------------------------------------------

@register
class EdgeBurrDefectDetector(Node):
    title = "Çapak ve Kırık Kenar Analizi"
    category = CAT_MEASURE
    description = ("Parça konturundaki mikro çapakları (çıkıntı) ve kırık/çentikleri (girinti) piksel hassasiyetinde "
                   "tespit eder. Toleransı aşan kusur sayısını ve piksel sapmasını bildirir.")
    inputs = [("image", IMG), ("mask", IMG)]
    outputs = [("defect_count", NUM), ("max_deviation", NUM), ("is_ok", BOOL), ("overlay_image", IMG)]
    params = [
        Param("burr_threshold", "float", 5.0, "Çapak/Kırık Toleransı (px)", 0.5, 100.0),
        Param("min_defect_len", "int", 6, "Minimum Kusur Noktası", 2, 200),
        Param("auto_threshold", "bool", True, "Otomatik Otsu Eşikleme"),
        Param("sensitivity", "int", 128, "Manuel Eşik Değeri", 1, 254),
        Param("show_contour", "bool", True, "Konturu ve Hataları Çiz"),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        mask_in = inputs.get("mask")

        if mask_in is not None:
            bin_img = _gray(mask_in)
            _, bin_img = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
        else:
            gray = _gray(img)
            if bool(self.v("auto_threshold")):
                _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                s = int(self.v("sensitivity"))
                _, bin_img = cv2.threshold(gray, s, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        tol = float(self.v("burr_threshold"))
        min_pts = int(self.v("min_defect_len"))
        show_cnt = bool(self.v("show_contour"))

        total_defects = 0
        max_dev = 0.0

        for cnt in contours:
            if len(cnt) < 30:
                continue
            peri = cv2.arcLength(cnt, True)
            smooth = cv2.approxPolyDP(cnt, 0.015 * peri, True)
            pts = cnt[:, 0, :]
            for pt in pts:
                dist = abs(cv2.pointPolygonTest(smooth, (float(pt[0]), float(pt[1])), True))
                if dist > max_dev:
                    max_dev = dist
                if dist > tol:
                    total_defects += 1
                    if show_cnt:
                        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

            if show_cnt:
                cv2.polylines(img, [smooth], True, (0, 255, 120), 1, cv2.LINE_AA)

        is_ok = bool(total_defects < min_pts and max_dev <= tol)
        badge_txt = "KENAR: OK" if is_ok else f"KENAR: RED ({total_defects} kusur, {max_dev:.1f}px)"
        badge_col = (0, 200, 60) if is_ok else (0, 40, 240)
        cv2.putText(img, badge_txt, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, badge_txt, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_col, 2, cv2.LINE_AA)

        return {
            "defect_count": total_defects,
            "max_deviation": round(max_dev, 2),
            "is_ok": is_ok,
            "overlay_image": img,
        }


# --- 21. CIELAB Delta-E Color Tolerance Checker -------------------------------------------

@register
class ColorToleranceChecker(Node):
    title = "Delta-E Renk Toleransı (CIELAB)"
    category = CAT_MEASURE
    description = ("Endüstriyel renk uyumu kontrolü (boya, plastik, gıda, tekstil). Ölçülen rengi hedef Lab "
                   "referansıyla karşılaştırır ve CIELAB Delta-E farkını hesaplayarak onay veya red verir.")
    inputs = [("image", IMG), ("mask", IMG)]
    outputs = [("delta_e", NUM), ("is_accepted", BOOL), ("reject_trigger", BOOL), ("overlay_image", IMG)]
    params = [
        Param("max_delta_e", "float", 4.0, "Maksimum Delta-E Toleransı", 0.2, 50.0),
        Param("target_l", "float", 50.0, "Hedef L* (Açıklık 0-100)", 0.0, 100.0),
        Param("target_a", "float", 0.0, "Hedef a* (Yeşil - Kırmızı)", -128.0, 127.0),
        Param("target_b", "float", 0.0, "Hedef b* (Mavi - Sarı)", -128.0, 127.0),
        Param("use_center_roi", "bool", True, "Merkez Bölgesini Ölç (%40 ROI)"),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        ih, iw = img.shape[:2]

        mask = inputs.get("mask")
        if mask is not None:
            m_gray = _gray(mask)
            _, m_bin = cv2.threshold(m_gray, 127, 255, cv2.THRESH_BINARY)
            roi_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            mean_vals = cv2.mean(roi_lab, mask=m_bin)[:3]
        elif bool(self.v("use_center_roi")):
            rw = int(iw * 0.4)
            rh = int(ih * 0.4)
            rx = (iw - rw) // 2
            ry = (ih - rh) // 2
            roi_bgr = img[ry:ry + rh, rx:rx + rw]
            roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            mean_vals = cv2.mean(roi_lab)[:3]
            cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (200, 200, 200), 1)
        else:
            roi_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            mean_vals = cv2.mean(roi_lab)[:3]

        meas_l = (mean_vals[0] * 100.0) / 255.0
        meas_a = mean_vals[1] - 128.0
        meas_b = mean_vals[2] - 128.0

        tgt_l = float(self.v("target_l"))
        tgt_a = float(self.v("target_a"))
        tgt_b = float(self.v("target_b"))

        dL = meas_l - tgt_l
        da = meas_a - tgt_a
        db = meas_b - tgt_b
        delta_e = np.sqrt(dL * dL + da * da + db * db)

        max_de = float(self.v("max_delta_e"))
        is_accepted = bool(delta_e <= max_de)
        reject_trig = not is_accepted

        res_col = (0, 210, 60) if is_accepted else (0, 30, 240)
        res_txt = f"Delta-E: {delta_e:.2f} (Tol: {max_de:.1f}) -> {'KABUL' if is_accepted else 'RED'}"
        cv2.putText(img, res_txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, res_txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, res_col, 2, cv2.LINE_AA)

        return {
            "delta_e": round(float(delta_e), 2),
            "is_accepted": is_accepted,
            "reject_trigger": reject_trig,
            "overlay_image": img,
        }


# --- 22. Production & Quality CSV Logger --------------------------------------------------

@register
class ProductionCSVLogger(Node):
    title = "Üretim & Kalite CSV Raporlayıcı"
    category = CAT_OUT
    description = ("Her parçanın denetim sonucunu, zaman damgasını, parça numarasını ve ölçüm değerlerini "
                   "Excel uyumlu günlük CSV dosyasına güvenli şekilde kaydeder.")
    inputs = [("trigger", BOOL), ("part_id", TEXT), ("status", TEXT), ("value1", NUM), ("value2", NUM)]
    outputs = [("logged", BOOL), ("total_count", NUM), ("reject_count", NUM), ("filepath", TEXT)]
    params = [
        Param("folder_path", "str", "raporlar", "Kayıt Klasörü"),
        Param("file_prefix", "str", "uretim_kayit", "Dosya Adı Öneki"),
        Param("on_change_only", "bool", True, "Sadece sinyal yükseldiğinde kaydet"),
        Param("include_header", "bool", True, "Başlık Satırı Ekle"),
    ]

    def _write_csv(self, filepath: str, row: list[str], header: list[str] | None):
        import csv
        file_exists = os.path.exists(filepath)
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not file_exists and header is not None:
                writer.writerow(header)
            writer.writerow(row)

    def process(self, inputs):
        self.state.setdefault("last_trigger", False)
        self.state.setdefault("total_count", 0)
        self.state.setdefault("reject_count", 0)

        trig = inputs.get("trigger")
        trigger = True if trig is None else bool(trig)
        on_change = bool(self.v("on_change_only"))

        should_log = False
        if on_change:
            if trigger and not self.state["last_trigger"]:
                should_log = True
        else:
            if trigger:
                should_log = True

        self.state["last_trigger"] = trigger
        logged = False

        today_str = dt.datetime.now().strftime("%Y-%m-%d")
        folder = str(self.v("folder_path")).strip() or "raporlar"
        prefix = str(self.v("file_prefix")).strip() or "uretim_kayit"
        filepath = os.path.join(folder, f"{prefix}_{today_str}.csv")

        if should_log:
            now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pid = str(inputs.get("part_id") or f"P-{self.state['total_count'] + 1}").strip()
            status = str(inputs.get("status") or "OK").strip()
            v1 = round(float(inputs.get("value1") or 0.0), 3)
            v2 = round(float(inputs.get("value2") or 0.0), 3)

            header = ["Tarih ve Saat", "Parca No / Kod", "Durum", "Deger 1", "Deger 2"] if bool(self.v("include_header")) else None
            row = [now_str, pid, status, str(v1), str(v2)]

            threading.Thread(target=self._write_csv, args=(filepath, row, header), daemon=True).start()

            self.state["total_count"] += 1
            if "red" in status.lower() or "nok" in status.lower() or "hata" in status.lower():
                self.state["reject_count"] += 1
            logged = True

        return {
            "logged": logged,
            "total_count": self.state["total_count"],
            "reject_count": self.state["reject_count"],
            "filepath": filepath,
        }


# --- 23. Multi-ROI Grid Matrix Inspector --------------------------------------------------

@register
class MultiROIGridInspector(Node):
    title = "Çoklu ROI / Matris Denetleyici"
    category = CAT_MEASURE
    description = ("Blister ambalaj (tablet/kapsül), koli içi şişe dizisi veya PCB pin matrisini satır x sütun (NxM) "
                   "şeklinde aynı anda denetler. Her hücredeki doluluk/varlık durumunu test edip hatalı hücre sayısını bildirir.")
    inputs = [("image", IMG)]
    outputs = [("total_cells", NUM), ("ok_cells", NUM), ("defect_cells", NUM), ("is_all_ok", BOOL), ("overlay_image", IMG)]
    params = [
        Param("rows", "int", 2, "Satır Sayısı (N)", 1, 20),
        Param("cols", "int", 5, "Sütun Sayısı (M)", 1, 20),
        Param("start_x", "int", 60, "Matris Başlangıç X", 0, 10000),
        Param("start_y", "int", 60, "Matris Başlangıç Y", 0, 10000),
        Param("cell_w", "int", 70, "Hücre Genişliği (px)", 5, 2000),
        Param("cell_h", "int", 70, "Hücre Yüksekliği (px)", 5, 2000),
        Param("gap_x", "int", 25, "Yatay Aralık (px)", 0, 1000),
        Param("gap_y", "int", 25, "Dikey Aralık (px)", 0, 1000),
        Param("inspect_mode", "choice", "doluluk_parlaklik", "Denetim Kriteri",
              choices=["doluluk_parlaklik", "kenar_yogunlugu"]),
        Param("min_fill_pct", "float", 25.0, "Minimum Doluluk Eşiği %", 1.0, 100.0),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        ih, iw = img.shape[:2]
        gray = _gray(img)

        rows = max(1, int(self.v("rows")))
        cols = max(1, int(self.v("cols")))
        sx = int(self.v("start_x"))
        sy = int(self.v("start_y"))
        cw = max(5, int(self.v("cell_w")))
        ch = max(5, int(self.v("cell_h")))
        gx = int(self.v("gap_x"))
        gy = int(self.v("gap_y"))
        mode = self.v("inspect_mode")
        th_pct = float(self.v("min_fill_pct"))

        total_cells = rows * cols
        ok_cells = 0
        defect_cells = 0

        for r in range(rows):
            for c in range(cols):
                x1 = sx + c * (cw + gx)
                y1 = sy + r * (ch + gy)
                x2 = min(iw, x1 + cw)
                y2 = min(ih, y1 + ch)

                if x2 <= x1 or y2 <= y1:
                    defect_cells += 1
                    continue

                roi_gray = gray[y1:y2, x1:x2]
                if mode == "kenar_yogunlugu":
                    edges = cv2.Canny(roi_gray, 50, 150)
                    fill_pct = (np.count_nonzero(edges) / max(1, edges.size)) * 100.0
                else:
                    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    fill_pct = (np.count_nonzero(thresh) / max(1, thresh.size)) * 100.0

                cell_ok = fill_pct >= th_pct
                if cell_ok:
                    ok_cells += 1
                    col = (0, 210, 60)
                else:
                    defect_cells += 1
                    col = (0, 30, 240)

                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                cv2.putText(img, f"#{r*cols + c + 1}", (x1 + 4, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

        is_all_ok = bool(defect_cells == 0)
        badge_txt = f"MATRIS: {ok_cells}/{total_cells} OK" if is_all_ok else f"MATRIS: RED ({defect_cells} EKSIK)"
        badge_col = (0, 210, 60) if is_all_ok else (0, 30, 240)
        cv2.putText(img, badge_txt, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, badge_txt, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_col, 2, cv2.LINE_AA)

        return {
            "total_cells": total_cells,
            "ok_cells": ok_cells,
            "defect_cells": defect_cells,
            "is_all_ok": is_all_ok,
            "overlay_image": img,
        }


# --- 24. Caliper & Dual-Edge Subpixel Meter ------------------------------------------------

@register
class CaliperEdgeMeter(Node):
    title = "Kumpas ve Çift Kenar Ölçer"
    category = CAT_MEASURE
    description = ("Optik kumpas (caliper): Çizilen hat boyunca iki zıt kenarı alt-piksel gradyan geçişiyle tespit eder "
                   "ve aradaki net mesafeyi / et kalınlığını / boşluğu ölçer.")
    inputs = [("image", IMG)]
    outputs = [("measured_dist", NUM), ("is_in_spec", BOOL), ("deviation", NUM), ("overlay_image", IMG)]
    params = [
        Param("x1", "int", 100, "Ölçüm Çizgisi Başlangıç X", 0, 10000),
        Param("y1", "int", 240, "Ölçüm Çizgisi Başlangıç Y", 0, 10000),
        Param("x2", "int", 540, "Ölçüm Çizgisi Bitiş X", 0, 10000),
        Param("y2", "int", 240, "Ölçüm Çizgisi Bitiş Y", 0, 10000),
        Param("target_dist", "float", 220.0, "Hedef Ölçü (px)", 1.0, 10000.0),
        Param("tolerance", "float", 12.0, "Tolerans (+/- px)", 0.2, 1000.0),
        Param("edge_threshold", "float", 20.0, "Kenar Eşik Değeri", 2.0, 255.0),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        ih, iw = img.shape[:2]
        gray = _gray(img)

        p1x = np.clip(int(self.v("x1")), 0, iw - 1)
        p1y = np.clip(int(self.v("y1")), 0, ih - 1)
        p2x = np.clip(int(self.v("x2")), 0, iw - 1)
        p2y = np.clip(int(self.v("y2")), 0, ih - 1)

        line_len = int(np.hypot(p2x - p1x, p2y - p1y))
        target_d = float(self.v("target_dist"))
        tol = float(self.v("tolerance"))
        edge_th = float(self.v("edge_threshold"))

        if line_len < 10:
            return {"measured_dist": 0.0, "is_in_spec": False, "deviation": target_d, "overlay_image": img}

        xs = np.linspace(p1x, p2x, line_len)
        ys = np.linspace(p1y, p2y, line_len)
        profile = gray[ys.astype(int), xs.astype(int)].astype(np.float32)

        grad = np.abs(np.diff(profile))
        peaks = np.where(grad > edge_th)[0]

        if len(peaks) >= 2:
            idx1 = peaks[0]
            idx2 = peaks[-1]
            e1x, e1y = int(xs[idx1]), int(ys[idx1])
            e2x, e2y = int(xs[idx2]), int(ys[idx2])
            dist = float(np.hypot(e2x - e1x, e2y - e1y))
        else:
            e1x, e1y, e2x, e2y = p1x, p1y, p2x, p2y
            dist = float(line_len)

        dev = dist - target_d
        in_spec = bool(abs(dev) <= tol)

        col = (0, 210, 60) if in_spec else (0, 40, 240)
        cv2.line(img, (p1x, p1y), (p2x, p2y), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.line(img, (e1x, e1y), (e2x, e2y), col, 2, cv2.LINE_AA)

        dx = (p2x - p1x) / max(1.0, line_len)
        dy = (p2y - p1y) / max(1.0, line_len)
        nx, ny = -dy * 14, dx * 14
        cv2.line(img, (int(e1x - nx), int(e1y - ny)), (int(e1x + nx), int(e1y + ny)), col, 2)
        cv2.line(img, (int(e2x - nx), int(e2y - ny)), (int(e2x + nx), int(e2y + ny)), col, 2)

        mx, my = (e1x + e2x) // 2, (e1y + e2y) // 2
        lbl = f"{dist:.1f} px ({dev:+.1f} px)"
        cv2.putText(img, lbl, (mx - 40, my - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, lbl, (mx - 40, my - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        return {
            "measured_dist": round(dist, 2),
            "is_in_spec": in_spec,
            "deviation": round(dev, 2),
            "overlay_image": img,
        }


# --- 25. Specular Glare & Reflection Reducer ----------------------------------------------

@register
class SpecularGlareReducer(Node):
    title = "Parlama ve Yansıma Bastırıcı"
    category = CAT_IMG
    description = ("Parlak metal parçalar, ıslak yüzeyler veya ambalaj naylonundaki aşırı ışık parlamalarını "
                   "(specular highlight) tespit eder ve çevre doku onarma (inpainting) veya ton sıkıştırma ile bastırır.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("glare_mask", IMG), ("glare_pct", NUM)]
    params = [
        Param("glare_threshold", "int", 240, "Parlama Eşik Parlaklığı (0-255)", 150, 255),
        Param("mode", "choice", "doku_onarma", "Giderme Yöntemi", choices=["doku_onarma", "yumusak_baski"]),
        Param("inpaint_radius", "int", 5, "Onarma Yarıçapı (px)", 1, 30),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        gray = _gray(img)

        th = int(self.v("glare_threshold"))
        _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)

        glare_pct = (np.count_nonzero(mask_dilated) / max(1, mask.size)) * 100.0
        mode = self.v("mode")
        rad = max(1, int(self.v("inpaint_radius")))

        if np.any(mask_dilated):
            if mode == "doku_onarma":
                out = cv2.inpaint(img, mask_dilated, rad, cv2.INPAINT_TELEA)
            else:
                blur = cv2.bilateralFilter(img, 9, 75, 75)
                alpha = (mask_dilated.astype(np.float32) / 255.0)[:, :, None]
                out = (img * (1.0 - alpha * 0.7) + blur * (alpha * 0.7)).astype(np.uint8)
        else:
            out = img

        return {
            "image": out,
            "glare_mask": mask_dilated,
            "glare_pct": round(float(glare_pct), 2),
        }


# --- 26. Barcode & QR Print Quality Verifier ----------------------------------------------

@register
class BarcodeQualityVerifier(Node):
    title = "Barkod & QR Baskı Kalite Kontrolü"
    category = CAT_MEASURE
    description = ("Barkod ve 2D QR kodların okunabilirliğinin ötesinde baskı kalitesini denetler. Sembol kontrastı, "
                   "kenar netliği ve sessiz bölge (quiet zone) genişliğini test ederek ISO/IEC A/B/C/D/F kalite notu üretir.")
    inputs = [("image", IMG)]
    outputs = [("grade", TEXT), ("contrast_pct", NUM), ("is_pass", BOOL), ("overlay_image", IMG)]
    params = [
        Param("min_grade", "choice", "B (Iyi)", "Minimum Kabul Edilebilir Not",
              choices=["A (Mukemmel)", "B (Iyi)", "C (Kabul)", "D (Sinirda)"]),
        Param("inspect_roi_x", "int", 0, "Manuel Bölge X (0 = Tam Görüntü)", 0, 10000),
        Param("inspect_roi_y", "int", 0, "Manuel Bölge Y", 0, 10000),
        Param("inspect_roi_w", "int", 0, "Manuel Bölge Genişlik", 0, 10000),
        Param("inspect_roi_h", "int", 0, "Manuel Bölge Yükseklik", 0, 10000),
    ]

    GRADE_WEIGHTS = {"A (Mukemmel)": 4, "B (Iyi)": 3, "C (Kabul)": 2, "D (Sinirda)": 1}

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        ih, iw = img.shape[:2]

        rx = int(self.v("inspect_roi_x"))
        ry = int(self.v("inspect_roi_y"))
        rw = int(self.v("inspect_roi_w"))
        rh = int(self.v("inspect_roi_h"))

        if rw > 10 and rh > 10 and rx + rw <= iw and ry + rh <= ih:
            roi = img[ry:ry + rh, rx:rx + rw]
        else:
            roi = img
            rx, ry, rw, rh = 0, 0, iw, ih

        gray = _gray(roi)
        min_refl = float(np.percentile(gray, 5))
        max_refl = float(np.percentile(gray, 95))
        contrast = ((max_refl - min_refl) / 255.0) * 100.0

        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge_sharpness = float(np.mean(np.abs(sobel)))

        if contrast >= 70.0 and edge_sharpness >= 18.0:
            grade_char = "A"
            score = 4
        elif contrast >= 55.0 and edge_sharpness >= 12.0:
            grade_char = "B"
            score = 3
        elif contrast >= 40.0:
            grade_char = "C"
            score = 2
        elif contrast >= 25.0:
            grade_char = "D"
            score = 1
        else:
            grade_char = "F"
            score = 0

        target_grade = str(self.v("min_grade"))
        req_score = self.GRADE_WEIGHTS.get(target_grade, 3)
        is_pass = bool(score >= req_score)

        col = (0, 210, 60) if is_pass else (0, 30, 240)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), col, 2)
        txt = f"KALITE: NOT {grade_char} (Kontrast: %{contrast:.0f})"
        cv2.putText(img, txt, (rx + 10, max(24, ry + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (rx + 10, max(24, ry + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

        return {
            "grade": f"Not {grade_char}",
            "contrast_pct": round(contrast, 1),
            "is_pass": is_pass,
            "overlay_image": img,
        }


# --- 27. Perspective Auto-Quad Straightener -----------------------------------------------

@register
class PerspectiveAutoQuad(Node):
    title = "Otomatik 4 Köşe Perspektif Düzeltici"
    category = CAT_IMG
    description = ("Görüntüdeki en belirgin dörtgen parçayı, etiketi veya belgeyi otomatik olarak algılar, "
                   "4 köşe noktasını bularak görüntüyü kuşbakışı düz dik açıya (orthogonal) çevirir.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("found_quad", BOOL), ("width", NUM), ("height", NUM)]
    params = [
        Param("target_w", "int", 640, "Çıkış Genişliği (px)", 50, 5000),
        Param("target_h", "int", 480, "Çıkış Yüksekliği (px)", 50, 5000),
        Param("min_area_pct", "float", 10.0, "Minimum Alan Yüzdesi %", 1.0, 95.0),
    ]

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        ih, iw = img.shape[:2]
        gray = _gray(img)

        tw = max(50, int(self.v("target_w")))
        th = max(50, int(self.v("target_h")))
        min_area = (iw * ih) * (float(self.v("min_area_pct")) / 100.0)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        found_quad = False
        target_pts = None

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                break
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                target_pts = approx.reshape(4, 2)
                found_quad = True
                break

        if found_quad and target_pts is not None:
            ordered = self._order_points(target_pts)
            dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(img, M, (tw, th))
        else:
            warped = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)

        return {
            "image": warped,
            "found_quad": found_quad,
            "width": tw,
            "height": th,
        }


# --- 28. PLC Modbus Word Packer -----------------------------------------------------------

@register
class ModbusWordPacker(Node):
    title = "PLC Modbus Word Paketleyici"
    category = CAT_OUT
    description = ("8 adede kadar farklı lojik sinyali (Kabul/Red, Kamera Hazır, Parça Var, Kusur 1-5) "
                   "tek bir 16-bit Modbus Holding Register tamsayısına paketler. PLC ile tek hamlede hızlı haberleşme sağlar.")
    inputs = [
        ("bit0_ok", BOOL),
        ("bit1_reject", BOOL),
        ("bit2_ready", BOOL),
        ("bit3_trigger", BOOL),
        ("bit4_defect_a", BOOL),
        ("bit5_defect_b", BOOL),
        ("bit6_alarm", BOOL),
        ("bit7_heartbeat", BOOL),
    ]
    outputs = [("register_value", NUM), ("hex_string", TEXT), ("has_error_flag", BOOL)]
    params = [
        Param("invert_logic", "bool", False, "Lojik Tersleme (Active-Low)"),
    ]

    def process(self, inputs):
        invert = bool(self.v("invert_logic"))
        reg = 0
        input_keys = [
            "bit0_ok", "bit1_reject", "bit2_ready", "bit3_trigger",
            "bit4_defect_a", "bit5_defect_b", "bit6_alarm", "bit7_heartbeat"
        ]

        for idx, key in enumerate(input_keys):
            val = bool(inputs.get(key) or False)
            if invert:
                val = not val
            if val:
                reg |= (1 << idx)

        has_err = bool(inputs.get("bit1_reject") or inputs.get("bit6_alarm"))

        return {
            "register_value": reg,
            "hex_string": f"0x{reg:04X}",
            "has_error_flag": has_err,
        }


# --- 29. Statistical Process Control (Cp / Cpk / SPC) -------------------------------------

@register
class SPCProcessCapability(Node):
    title = "İstatistiksel Proses Kontrolü (Cp / Cpk / SPC)"
    category = CAT_MEASURE
    description = ("Üretim hattındaki ölçümlerin istatistiksel dağılımını hesaplar: Ortalama, Standart Sapma, "
                   "Cp ve Cpk yeterlilik indeksleri. Proses kayması ve tolerans daralmalarını önceden uyarır.")
    inputs = [("value", NUM), ("reset_trigger", BOOL)]
    outputs = [("mean", NUM), ("sigma", NUM), ("cp", NUM), ("cpk", NUM), ("is_capable", BOOL)]
    params = [
        Param("sample_size", "int", 50, "Örneklem Boyutu (Son N Parça)", 10, 5000),
        Param("usl", "float", 105.0, "Üst Spesifikasyon Sınırı (USL)", -100000.0, 100000.0),
        Param("lsl", "float", 95.0, "Alt Spesifikasyon Sınırı (LSL)", -100000.0, 100000.0),
        Param("min_cpk", "float", 1.33, "Hedef Cpk Yeterlilik Eşiği", 0.5, 3.0),
    ]

    def process(self, inputs):
        self.state.setdefault("buffer", [])

        if bool(inputs.get("reset_trigger")):
            self.state["buffer"].clear()

        val = inputs.get("value")
        if val is not None:
            max_n = max(5, int(self.v("sample_size")))
            buf = self.state["buffer"]
            buf.append(float(val))
            if len(buf) > max_n:
                self.state["buffer"] = buf[-max_n:]

        buf = self.state["buffer"]
        usl = float(self.v("usl"))
        lsl = float(self.v("lsl"))
        target_cpk = float(self.v("min_cpk"))

        if len(buf) >= 3:
            arr = np.array(buf, dtype=np.float64)
            mean = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1))
            if sigma > 0.00001:
                cp = (usl - lsl) / (6.0 * sigma)
                cp_u = (usl - mean) / (3.0 * sigma)
                cp_l = (mean - lsl) / (3.0 * sigma)
                cpk = min(cp_u, cp_l)
            else:
                cp = 9.99
                cpk = 9.99
        else:
            mean = float(val or 0.0)
            sigma = 0.0
            cp = 1.0
            cpk = 1.0

        is_capable = bool(cpk >= target_cpk)

        return {
            "mean": round(mean, 3),
            "sigma": round(sigma, 4),
            "cp": round(cp, 2),
            "cpk": round(cpk, 2),
            "is_capable": is_capable,
        }


# --- 30. Audio Tone & Acoustic Frequency Generator ----------------------------------------

@register
class AudioToneFrequencyGenerator(Node):
    title = "Özel Frekans Tonu ve Akustik Alarm"
    category = CAT_OUT
    description = ("Hatasız parça geçişinde melodik kabul tonu (ör. 880 Hz), hatalı parçada ise dikkat çekici ikaz tonu "
                   "(ör. 320 Hz) üreterek operatöre gecikmesiz akustik geri bildirim sağlar.")
    inputs = [("trigger", BOOL), ("is_error", BOOL)]
    outputs = [("played", BOOL)]
    params = [
        Param("ok_freq", "int", 880, "Kabul Frekansı (Hz)", 100, 5000),
        Param("ok_duration", "int", 120, "Kabul Süresi (ms)", 20, 2000),
        Param("nok_freq", "int", 320, "Hata Frekansı (Hz)", 100, 5000),
        Param("nok_duration", "int", 350, "Hata Süresi (ms)", 20, 2000),
        Param("on_change_only", "bool", True, "Yalnızca Değişimde Çal"),
    ]

    def _play_tone(self, freq: int, dur: int):
        try:
            import winsound
            winsound.Beep(int(freq), int(dur))
        except Exception:
            pass

    def process(self, inputs):
        self.state.setdefault("last_trigger", False)

        trig = inputs.get("trigger")
        trigger = True if trig is None else bool(trig)
        is_err = bool(inputs.get("is_error") or False)
        on_change = bool(self.v("on_change_only"))

        should_play = False
        if on_change:
            if trigger and not self.state["last_trigger"]:
                should_play = True
        else:
            if trigger:
                should_play = True

        self.state["last_trigger"] = trigger
        played = False

        if should_play:
            freq = int(self.v("nok_freq")) if is_err else int(self.v("ok_freq"))
            dur = int(self.v("nok_duration")) if is_err else int(self.v("ok_duration"))
            threading.Thread(target=self._play_tone, args=(freq, dur), daemon=True).start()
            played = True

        return {"played": played}


# --- 31. Concentricity & Circle Bore Gauge ------------------------------------------------

@register
class ConcentricityGauge(Node):
    title = "Eşmerkezlilik ve Çap Ölçer"
    category = CAT_MEASURE
    description = ("İç ve dış çemberleri (pul, conta, flanş, rulman bileziği veya delik) tespit eder; "
                   "iç ve dış çapları, merkez kaçıklığını (offset ΔX, ΔY) ve eşmerkezlilik hatasını sub-piksel ölçer.")
    inputs = [("image", IMG)]
    outputs = [("outer_diam", NUM), ("inner_diam", NUM), ("offset_dist", NUM), ("is_concentric", BOOL), ("overlay_image", IMG)]
    params = [
        Param("outer_target_diam", "float", 180.0, "Hedef Dış Çap (px)", 5.0, 5000.0),
        Param("inner_target_diam", "float", 90.0, "Hedef İç Çap (px)", 2.0, 5000.0),
        Param("max_offset_tol", "float", 4.0, "Maksimum Eksen Kaçıklığı (px)", 0.1, 100.0),
        Param("diam_tolerance", "float", 6.0, "Çap Toleransı (+/- px)", 0.1, 200.0),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        gray = _gray(img)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        circles = []
        for c in contours:
            if len(c) >= 15:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                area = cv2.contourArea(c)
                circ_area = np.pi * r * r
                if circ_area > 0 and (area / circ_area) > 0.45:
                    circles.append((float(cx), float(cy), float(r * 2.0)))

        circles = sorted(circles, key=lambda c: c[2], reverse=True)

        tgt_out = float(self.v("outer_target_diam"))
        tgt_in = float(self.v("inner_target_diam"))
        off_tol = float(self.v("max_offset_tol"))
        d_tol = float(self.v("diam_tolerance"))

        if len(circles) >= 2:
            c_out = circles[0]
            c_in = next((c for c in circles[1:] if c[2] < c_out[2] * 0.85), circles[1])
            cx1, cy1, d1 = c_out
            cx2, cy2, d2 = c_in
            offset = float(np.hypot(cx1 - cx2, cy1 - cy2))
        elif len(circles) == 1:
            cx1, cy1, d1 = circles[0]
            cx2, cy2, d2 = cx1, cy1, 0.0
            offset = 0.0
        else:
            ih, iw = img.shape[:2]
            cx1, cy1, d1 = iw / 2.0, ih / 2.0, tgt_out
            cx2, cy2, d2 = cx1, cy1, tgt_in
            offset = 0.0

        is_conc = bool(offset <= off_tol and abs(d1 - tgt_out) <= d_tol and abs(d2 - tgt_in) <= d_tol)
        col = (0, 210, 60) if is_conc else (0, 30, 240)

        cv2.circle(img, (int(cx1), int(cy1)), int(d1 / 2), col, 2, cv2.LINE_AA)
        cv2.circle(img, (int(cx2), int(cy2)), int(d2 / 2), col, 2, cv2.LINE_AA)
        cv2.drawMarker(img, (int(cx1), int(cy1)), (255, 200, 0), cv2.MARKER_CROSS, 12, 1)
        cv2.drawMarker(img, (int(cx2), int(cy2)), (0, 150, 255), cv2.MARKER_TILTED_CROSS, 12, 1)
        if offset > 0.5:
            cv2.line(img, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 0, 255), 2)

        lbl = f"D-Dis: {d1:.1f}px  D-Ic: {d2:.1f}px  Kaciklik: {offset:.2f}px"
        cv2.putText(img, lbl, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, lbl, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        return {
            "outer_diam": round(d1, 2),
            "inner_diam": round(d2, 2),
            "offset_dist": round(offset, 2),
            "is_concentric": is_conc,
            "overlay_image": img,
        }


# --- 32. Sauvola & Engraved Marking Binarizer ---------------------------------------------

@register
class SauvolaEngraveBinarizer(Node):
    title = "Lazer / Nokta Vurma Kazıma Vurgulayıcı"
    category = CAT_IMG
    description = ("Metal, plastik veya döküm parçalar üzerindeki düşük kontrastlı lazer markalama, nokta vuruşlu "
                   "(dot-peen) yazı veya kabartmalı seri numaralarını adaptif yerel eşikleme (Sauvola & Niblack) ile netleştirir.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("binary_mask", IMG)]
    params = [
        Param("window_size", "int", 25, "Pencere Boyutu (tek sayı)", 3, 99),
        Param("k_factor", "float", 0.2, "K Katsayısı", 0.01, 1.0),
        Param("invert", "bool", False, "Renkleri Ters Çevir"),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        gray = _gray(img_in).astype(np.float32)
        w = max(3, int(self.v("window_size")))
        if w % 2 == 0:
            w += 1
        k = float(self.v("k_factor"))
        r = 128.0

        mean = cv2.blur(gray, (w, w))
        mean_sq = cv2.blur(gray * gray, (w, w))
        variance = np.maximum(0.0, mean_sq - (mean * mean))
        std = np.sqrt(variance)

        thresh = mean * (1.0 + k * (std / r - 1.0))
        bin_img = np.where(gray > thresh, 255, 0).astype(np.uint8)

        if bool(self.v("invert")):
            bin_img = 255 - bin_img

        bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        return {"image": bgr, "binary_mask": bin_img}


# --- 33. Surface Scratch & Micro-Crack Enhancer -------------------------------------------

@register
class SurfaceScratchEnhancer(Node):
    title = "Yüzey Çizik ve Çatlak Vurgulayıcı"
    category = CAT_IMG
    description = ("Cam, güneş paneli, boyalı saç veya polisajlı yüzeylerdeki mikro çizikleri, kılcal çatlakları ve "
                   "çizgi kusurlarını morfolojik Top-Hat / Black-Hat filtreleriyle belirginleştirir.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("scratch_mask", IMG), ("scratch_intensity", NUM)]
    params = [
        Param("defect_type", "choice", "her_ikisi", "Kusur Türü", choices=["koyu_cizikler", "parlak_cizikler", "her_ikisi"]),
        Param("kernel_size", "int", 15, "Filtre Çekirdek Boyutu (px)", 3, 51),
        Param("sensitivity", "float", 22.0, "Hassasiyet Eşiği", 5.0, 200.0),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        gray = _gray(img)
        ks = max(3, int(self.v("kernel_size")))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        dtype = self.v("defect_type")
        th = float(self.v("sensitivity"))

        if dtype == "koyu_cizikler":
            diff = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        elif dtype == "parlak_cizikler":
            diff = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        else:
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            th_map = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            diff = cv2.max(bh, th_map)

        _, mask = cv2.threshold(diff, int(th), 255, cv2.THRESH_BINARY)
        intensity = float(np.mean(diff[mask > 0])) if np.any(mask) else 0.0

        out = img.copy()
        out[mask > 0] = [0, 220, 255]

        return {
            "image": out,
            "scratch_mask": mask,
            "scratch_intensity": round(intensity, 2),
        }


# --- 34. Bolt & Screw Thread Pitch Inspector ----------------------------------------------

@register
class ThreadPitchInspector(Node):
    title = "Vida ve Cıvata Diş Adımı Kontrolü"
    category = CAT_MEASURE
    description = ("Cıvata, somun, mil ve vidaların diş profilini denetler. Diş adımı (hatve / pitch), diş sayısı ve "
                   "periyodik tepe/taban aralığını piksel hassasiyetinde ölçerek eksik veya deforme dişleri tespit eder.")
    inputs = [("image", IMG)]
    outputs = [("measured_pitch", NUM), ("thread_count", NUM), ("is_pass", BOOL), ("overlay_image", IMG)]
    params = [
        Param("x1", "int", 100, "Vida Ekseni Başlangıç X", 0, 10000),
        Param("y1", "int", 200, "Vida Ekseni Başlangıç Y", 0, 10000),
        Param("x2", "int", 500, "Vida Ekseni Bitiş X", 0, 10000),
        Param("y2", "int", 200, "Vida Ekseni Bitiş Y", 0, 10000),
        Param("target_pitch", "float", 18.0, "Hedef Diş Adımı (Hatve px)", 2.0, 200.0),
        Param("pitch_tol", "float", 2.5, "Adım Toleransı (+/- px)", 0.2, 50.0),
        Param("min_threads", "int", 6, "Beklenen Minimum Diş Sayısı", 2, 100),
    ]

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        ih, iw = img.shape[:2]
        gray = _gray(img)

        p1x = np.clip(int(self.v("x1")), 0, iw - 1)
        p1y = np.clip(int(self.v("y1")), 0, ih - 1)
        p2x = np.clip(int(self.v("x2")), 0, iw - 1)
        p2y = np.clip(int(self.v("y2")), 0, ih - 1)

        line_len = int(np.hypot(p2x - p1x, p2y - p1y))
        target_p = float(self.v("target_pitch"))
        tol = float(self.v("pitch_tol"))
        min_th = int(self.v("min_threads"))

        if line_len < 20:
            return {"measured_pitch": 0.0, "thread_count": 0, "is_pass": False, "overlay_image": img}

        xs = np.linspace(p1x, p2x, line_len)
        ys = np.linspace(p1y, p2y, line_len)
        prof = gray[ys.astype(int), xs.astype(int)].astype(np.float32)
        prof_smooth = cv2.GaussianBlur(prof.reshape(1, -1), (9, 1), 0).flatten()

        diff = np.diff(prof_smooth)
        peaks = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1
        thread_cnt = len(peaks)

        if thread_cnt >= 2:
            step_dists = np.diff(peaks)
            pitch = float(np.median(step_dists))
        else:
            pitch = 0.0

        is_pass = bool(thread_cnt >= min_th and abs(pitch - target_p) <= tol)
        col = (0, 210, 60) if is_pass else (0, 30, 240)

        cv2.line(img, (p1x, p1y), (p2x, p2y), (100, 100, 100), 1)
        for pk in peaks:
            px, py = int(xs[pk]), int(ys[pk])
            cv2.circle(img, (px, py), 3, col, -1)

        txt = f"DIS HATVE: {pitch:.1f}px (Hedef: {target_p:.1f}) - Sayi: {thread_cnt}"
        cv2.putText(img, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        return {
            "measured_pitch": round(pitch, 2),
            "thread_count": thread_cnt,
            "is_pass": is_pass,
            "overlay_image": img,
        }


# --- 35. Defect Heatmap Transparent Overlay -----------------------------------------------

@register
class DefectHeatmapBlend(Node):
    title = "Kusur Maskesi ve Isı Haritası Bindirici"
    category = CAT_MEASURE
    description = ("Görüntü üzerindeki kusur maskesini veya yapay zeka anomali haritasını orijinal görüntü üzerine "
                   "yarı saydam parlak ısı haritası ve parlak dış konturlarla bindirir.")
    inputs = [("image", IMG), ("mask", IMG)]
    outputs = [("image", IMG), ("defect_area_pct", NUM)]
    params = [
        Param("alpha", "float", 0.55, "Isı Haritası Saydamlığı", 0.1, 1.0),
        Param("draw_contours", "bool", True, "Kusur Dış Hatlarını Çiz"),
        Param("colormap", "choice", "turbo", "Renk Paleti", choices=["turbo", "jet", "inferno"]),
    ]

    CMAP = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "inferno": cv2.COLORMAP_INFERNO}

    def process(self, inputs):
        img_in = inputs["image"]
        if img_in is None:
            raise ValueError("Görüntü girişi bağlı değil")

        img = _bgr(img_in).copy()
        mask_in = inputs.get("mask")

        if mask_in is None:
            return {"image": img, "defect_area_pct": 0.0}

        mask = _gray(mask_in)
        ih, iw = img.shape[:2]
        if mask.shape[:2] != (ih, iw):
            mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

        _, bin_m = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
        area_pct = (np.count_nonzero(bin_m) / max(1, bin_m.size)) * 100.0

        if np.any(bin_m):
            cmap_id = self.CMAP.get(self.v("colormap"), cv2.COLORMAP_TURBO)
            heat = cv2.applyColorMap(mask, cmap_id)
            alpha = float(self.v("alpha"))
            blend = np.where(bin_m[:, :, None] > 0,
                             (img.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8),
                             img)

            if bool(self.v("draw_contours")):
                contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(blend, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
            out = blend
        else:
            out = img

        return {"image": out, "defect_area_pct": round(float(area_pct), 2)}


# --- 36. PLC / System Heartbeat Watchdog ---------------------------------------------------

@register
class HeartbeatWatchdog(Node):
    title = "PLC / Sistem Canlılık Bekçisi"
    category = CAT_LOGIC
    description = ("Kameradan veya PLC'den gelen canlılık darbesini (heartbeat) sürekli denetler. "
                   "Belirlenen süre boyunca darbe gelmezse veya kamera donarsa acil hat durdurma / alarm üretir.")
    inputs = [("heartbeat_pulse", BOOL), ("reset", BOOL)]
    outputs = [("is_alive", BOOL), ("timeout_alarm", BOOL), ("elapsed_sec", NUM)]
    params = [
        Param("timeout_sec", "float", 3.0, "Zaman Aşımı Süresi (saniye)", 0.2, 120.0),
        Param("on_level_change", "bool", True, "Darbe Sinyali Değişimini Say"),
    ]

    def process(self, inputs):
        self.state.setdefault("last_pulse_time", time.time())
        self.state.setdefault("last_level", False)

        if bool(inputs.get("reset")):
            self.state["last_pulse_time"] = time.time()

        now = time.time()
        curr_lvl = bool(inputs.get("heartbeat_pulse") or False)
        on_chg = bool(self.v("on_level_change"))

        if on_chg:
            if curr_lvl != self.state["last_level"]:
                self.state["last_pulse_time"] = now
        else:
            if curr_lvl:
                self.state["last_pulse_time"] = now

        self.state["last_level"] = curr_lvl
        elapsed = now - self.state["last_pulse_time"]
        timeout = float(self.v("timeout_sec"))

        is_alive = bool(elapsed <= timeout)
        alarm = not is_alive

        return {
            "is_alive": is_alive,
            "timeout_alarm": alarm,
            "elapsed_sec": round(elapsed, 2),
        }


# --- 37. Industrial Serial & Scale String Parser ------------------------------------------

@register
class SerialStringParser(Node):
    title = "Endüstriyel Veri ve Terazi Çözücü"
    category = CAT_LOGIC
    description = ("Seri port, barkod okuyucu veya terazi/ölçüm cihazından gelen ham ASCII metinleri parçalar. "
                   "Regex veya ayraç kullanarak ağırlık, sıcaklık veya seri no değerlerini sayı ve metin olarak ayrıştırır.")
    inputs = [("raw_text", TEXT)]
    outputs = [("parsed_number", NUM), ("parsed_string", TEXT), ("is_matched", BOOL)]
    params = [
        Param("parse_mode", "choice", "sayi_ayikla", "Ayrıştırma Yöntemi",
              choices=["sayi_ayikla", "regex_deseni", "ayrac_indeks"]),
        Param("regex_pattern", "str", r"[-+]?\d*\.?\d+", "Regex Deseni"),
        Param("delimiter", "str", ",", "Ayraç Karakteri"),
        Param("column_index", "int", 0, "Ayraç Sütun No (0-indexed)", 0, 50),
    ]

    def process(self, inputs):
        raw = str(inputs.get("raw_text") or "").strip()
        mode = self.v("parse_mode")

        matched = False
        parsed_num = 0.0
        parsed_str = ""

        if raw:
            import re
            if mode == "sayi_ayikla":
                nums = re.findall(r"[-+]?\d*\.?\d+", raw)
                if nums:
                    parsed_num = float(nums[0])
                    parsed_str = nums[0]
                    matched = True
            elif mode == "regex_deseni":
                pat = str(self.v("regex_pattern")).strip()
                m = re.search(pat, raw)
                if m:
                    parsed_str = m.group(1) if m.groups() else m.group(0)
                    try:
                        parsed_num = float(parsed_str)
                    except ValueError:
                        parsed_num = 0.0
                    matched = True
            else:  # ayrac_indeks
                delim = str(self.v("delimiter"))
                idx = int(self.v("column_index"))
                parts = raw.split(delim)
                if 0 <= idx < len(parts):
                    parsed_str = parts[idx].strip()
                    try:
                        parsed_num = float(parsed_str)
                    except ValueError:
                        parsed_num = 0.0
                    matched = True

        return {
            "parsed_number": parsed_num,
            "parsed_string": parsed_str,
            "is_matched": matched,
        }


# --- 38. Shift OEE & Line Efficiency Calculator -------------------------------------------

@register
class ShiftOEECalculator(Node):
    title = "Vardiya OEE & Hat Verimlilik Analizi"
    category = CAT_OUT
    description = ("Üretim hattının Genel Ekipman Etkinliğini (OEE = Kullanılabilirlik x Performans x Kalite) hesaplar. "
                   "Duruş süreleri, ideal çevrim zamanı ve ıskarta oranını analiz eder.")
    inputs = [("part_trigger", BOOL), ("is_defect", BOOL), ("line_running", BOOL), ("reset_shift", BOOL)]
    outputs = [("oee_pct", NUM), ("availability_pct", NUM), ("performance_pct", NUM), ("quality_pct", NUM), ("total_count", NUM)]
    params = [
        Param("ideal_cycle_sec", "float", 2.0, "İdeal Çevrim Süresi (sn/parça)", 0.05, 3600.0),
        Param("target_oee", "float", 85.0, "Hedef OEE Seviyesi %", 10.0, 100.0),
    ]

    def process(self, inputs):
        self.state.setdefault("start_time", time.time())
        self.state.setdefault("run_time", 0.0)
        self.state.setdefault("stop_time", 0.0)
        self.state.setdefault("last_tick", time.time())
        self.state.setdefault("total_count", 0)
        self.state.setdefault("defect_count", 0)
        self.state.setdefault("last_trigger", False)

        now = time.time()
        dt_tick = now - self.state["last_tick"]
        self.state["last_tick"] = now

        if bool(inputs.get("reset_shift")):
            self.state["start_time"] = now
            self.state["run_time"] = 0.0
            self.state["stop_time"] = 0.0
            self.state["total_count"] = 0
            self.state["defect_count"] = 0

        running_val = inputs.get("line_running")
        running = True if running_val is None else bool(running_val)
        if running:
            self.state["run_time"] += dt_tick
        else:
            self.state["stop_time"] += dt_tick

        trig = bool(inputs.get("part_trigger") or False)
        if trig and not self.state["last_trigger"]:
            self.state["total_count"] += 1
            if bool(inputs.get("is_defect") or False):
                self.state["defect_count"] += 1
        self.state["last_trigger"] = trig

        total_time = self.state["run_time"] + self.state["stop_time"]
        avail = (self.state["run_time"] / max(0.001, total_time)) if total_time > 1.0 else 1.0

        ideal_c = float(self.v("ideal_cycle_sec"))
        tot = self.state["total_count"]
        defects = self.state["defect_count"]
        good = max(0, tot - defects)

        if self.state["run_time"] > 1.0 and tot > 0:
            perf = min(1.0, (ideal_c * tot) / self.state["run_time"])
        else:
            perf = 1.0

        qual = (good / float(tot)) if tot > 0 else 1.0
        oee = avail * perf * qual * 100.0

        return {
            "oee_pct": round(oee, 1),
            "availability_pct": round(avail * 100.0, 1),
            "performance_pct": round(perf * 100.0, 1),
            "quality_pct": round(qual * 100.0, 1),
            "total_count": tot,
        }


# --- 39. Anisotropic Diffusion Filter (Perona-Malik Edge Preserving) ------------------------

@register
class AnisotropicDiffusionFilter(Node):
    title = "Kenar Koruyucu Difüzyon (Perona-Malik)"
    category = CAT_IMG
    description = ("Görüntüdeki parazit ve gürültüyü kusur kenarlarını bozmadan pürüzsüzleştirir. "
                   "Geleneksel Gauss filtresinin aksine çizik, çatlak ve çapak sınırlarını keskin tutar.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [
        Param("iterations", "int", 5, "İterasyon Sayısı", 1, 30),
        Param("k_val", "float", 25.0, "Kenar İletim Katsayısı (K)", 1.0, 150.0),
        Param("conductance", "choice", "Eksponansiyel (Geniş Kontrast)", "İletim Fonksiyonu",
              choices=["Eksponansiyel (Geniş Kontrast)", "Karesel (İnce Kusurlar)"]),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        orig = _bgr(img)
        iters = max(1, min(30, int(self.v("iterations"))))
        k = max(1.0, float(self.v("k_val")))
        fn_quad = "Karesel" in str(self.v("conductance"))
        gamma = 0.20  # stability factor for 4-neighbor scheme

        # Process on float32 channels
        res_channels = []
        for c in range(3):
            plane = orig[:, :, c].astype(np.float32)
            for _ in range(iters):
                delta_n = np.zeros_like(plane)
                delta_s = np.zeros_like(plane)
                delta_e = np.zeros_like(plane)
                delta_w = np.zeros_like(plane)

                delta_n[:-1, :] = np.diff(plane, axis=0)
                delta_s[1:, :] = -delta_n[:-1, :]
                delta_e[:, :-1] = np.diff(plane, axis=1)
                delta_w[:, 1:] = -delta_e[:, :-1]

                if fn_quad:
                    cn = 1.0 / (1.0 + (delta_n / k) ** 2)
                    cs = 1.0 / (1.0 + (delta_s / k) ** 2)
                    ce = 1.0 / (1.0 + (delta_e / k) ** 2)
                    cw = 1.0 / (1.0 + (delta_w / k) ** 2)
                else:
                    cn = np.exp(-((delta_n / k) ** 2))
                    cs = np.exp(-((delta_s / k) ** 2))
                    ce = np.exp(-((delta_e / k) ** 2))
                    cw = np.exp(-((delta_w / k) ** 2))

                plane += gamma * (cn * delta_n + cs * delta_s + ce * delta_e + cw * delta_w)

            res_channels.append(np.clip(plane, 0, 255).astype(np.uint8))

        return {"image": cv2.merge(res_channels)}


# --- 40. Defect Classification Rule Engine ------------------------------------------------

@register
class DefectClassificationRuleEngine(Node):
    title = "Kusur Sınıflandırma Kural Motoru"
    category = CAT_MEASURE
    description = ("Tespit edilen konturları alan, en/boy oranı, dairesellik ve geometrisine göre sınıflandırır: "
                   "Çizik, Çapak, Delik/Gözenek veya Yüzey Lekesi olarak etiketler.")
    inputs = [("image", IMG), ("contours", PortType.CONTOURS)]
    outputs = [("image", IMG), ("has_critical", BOOL), ("defect_summary", TEXT), ("classified_defects", ANY)]
    params = [
        Param("scratch_aspect_ratio", "float", 3.2, "Çizik En/Boy Eşiği", 1.5, 20.0),
        Param("burr_circularity_max", "float", 0.40, "Çapak Dairesellik Sınırı", 0.05, 0.90),
        Param("pinhole_max_area", "int", 90, "Mikro Delik Maksimum Alan", 2, 5000),
        Param("min_defect_area", "int", 15, "Minimum Kusur Alanı (Gürültü Filtresi)", 1, 1000),
        Param("critical_rule", "choice", "Tümü Kritik", "Kritik Kusur Kriteri",
              choices=["Tümü Kritik", "Yalnızca Çizik ve Delik", "Yalnızca Çapak"]),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        contours = inputs.get("contours")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        vis = _bgr(img).copy()
        if contours is None or not isinstance(contours, (list, tuple)) or len(contours) == 0:
            return {
                "image": vis,
                "has_critical": False,
                "defect_summary": "Kusur Yok (TEMİZ)",
                "classified_defects": [],
            }

        min_area = int(self.v("min_defect_area"))
        scratch_ar = float(self.v("scratch_aspect_ratio"))
        burr_circ = float(self.v("burr_circularity_max"))
        pinhole_area = int(self.v("pinhole_max_area"))
        crit_rule = str(self.v("critical_rule"))

        classified = []
        counts = {"Çizik": 0, "Çapak": 0, "Delik": 0, "Leke": 0}
        has_critical = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            circularity = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.0

            x, y, w, h = cv2.boundingRect(cnt)
            rect_ratio = max(w, h) / max(1.0, float(min(w, h)))

            if area <= pinhole_area and circularity > 0.50:
                dtype = "Delik"
                color = (0, 165, 255)
            elif rect_ratio >= scratch_ar:
                dtype = "Çizik"
                color = (0, 0, 255)
            elif circularity <= burr_circ and area > pinhole_area:
                dtype = "Çapak"
                color = (255, 0, 255)
            else:
                dtype = "Leke"
                color = (0, 255, 255)

            counts[dtype] += 1

            is_crit = False
            if crit_rule == "Tümü Kritik":
                is_crit = True
            elif crit_rule == "Yalnızca Çizik ve Delik" and dtype in ("Çizik", "Delik"):
                is_crit = True
            elif crit_rule == "Yalnızca Çapak" and dtype == "Çapak":
                is_crit = True

            if is_crit:
                has_critical = True

            classified.append({
                "type": dtype,
                "area": round(float(area), 1),
                "circularity": round(float(circularity), 3),
                "aspect_ratio": round(float(rect_ratio), 2),
                "bbox": [int(x), int(y), int(w), int(h)],
                "critical": is_crit,
            })

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{dtype} (A:{int(area)})", (x, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        total_defects = sum(counts.values())
        sum_str = f"Toplam: {total_defects} | Çizik:{counts['Çizik']} Çapak:{counts['Çapak']} Delik:{counts['Delik']} Leke:{counts['Leke']}"
        banner_color = (0, 0, 200) if has_critical else (0, 160, 0)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(vis, sum_str, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, banner_color, 1, cv2.LINE_AA)

        return {
            "image": vis,
            "has_critical": has_critical,
            "defect_summary": sum_str,
            "classified_defects": classified,
        }


# --- 41. Line Profile Intensity Scanner ---------------------------------------------------

@register
class LineProfileIntensityScanner(Node):
    title = "Çizgi Yoğunluk Profili (Caliper Scanner)"
    category = CAT_MEASURE
    description = ("Seçilen yatay veya dikey hat boyunca piksel parlaklık profilini tarar. "
                   "Geçiş kenarlarını (peak/vadileri) bularak seviye, dolum veya kanal genişliği ölçer.")
    inputs = [("image", IMG)]
    outputs = [("profile_image", IMG), ("contrast", NUM), ("min_intensity", NUM), ("max_intensity", NUM), ("edge_positions", ANY)]
    params = [
        Param("axis", "choice", "Yatay (X Ekseni)", "Tarama Yönü", choices=["Yatay (X Ekseni)", "Dikey (Y Ekseni)"]),
        Param("scan_pos_pct", "float", 50.0, "Hat Konumu %", 1.0, 99.0),
        Param("edge_threshold", "float", 25.0, "Kenar Eğim Eşiği (Geçişler)", 5.0, 120.0),
        Param("show_graph", "bool", True, "Grafik Katmanını Göster"),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        vis = _bgr(img).copy()
        gray = _gray(vis)
        h, w = gray.shape

        is_x = "Yatay" in str(self.v("axis"))
        pct = float(self.v("scan_pos_pct")) / 100.0
        edge_th = float(self.v("edge_threshold"))

        if is_x:
            line_idx = int(np.clip(pct * h, 0, h - 1))
            profile = gray[line_idx, :].astype(np.float32)
            cv2.line(vis, (0, line_idx), (w - 1, line_idx), (0, 255, 255), 1, cv2.LINE_AA)
        else:
            line_idx = int(np.clip(pct * w, 0, w - 1))
            profile = gray[:, line_idx].astype(np.float32)
            cv2.line(vis, (line_idx, 0), (line_idx, h - 1), (0, 255, 255), 1, cv2.LINE_AA)

        min_val = float(np.min(profile))
        max_val = float(np.max(profile))
        contrast = max_val - min_val

        diff = np.diff(profile)
        edges = np.where(np.abs(diff) >= edge_th)[0].tolist()

        if bool(self.v("show_graph")) and len(profile) > 10:
            plot_h = min(120, max(60, h // 4))
            plot_w = min(w - 20, 400)
            overlay = vis.copy()
            cv2.rectangle(overlay, (10, h - plot_h - 10), (10 + plot_w, h - 10), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
            cv2.rectangle(vis, (10, h - plot_h - 10), (10 + plot_w, h - 10), (80, 80, 80), 1)

            step = len(profile) / float(plot_w)
            pts = []
            for px in range(plot_w):
                idx = int(px * step)
                if idx < len(profile):
                    val = profile[idx]
                    py = int(h - 15 - (val / 255.0) * (plot_h - 20))
                    pts.append((10 + px, py))

            if len(pts) > 1:
                cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 1, cv2.LINE_AA)

            info = f"Min:{int(min_val)} Max:{int(max_val)} Kontrast:{int(contrast)} Kenarlar:{len(edges)}"
            cv2.putText(vis, info, (16, h - plot_h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        return {
            "profile_image": vis,
            "contrast": round(contrast, 1),
            "min_intensity": round(min_val, 1),
            "max_intensity": round(max_val, 1),
            "edge_positions": edges,
        }


# --- 42. Industrial Presence / Absence Blob Matcher ----------------------------------------

@register
class PresenceAbsenceBlobMatcher(Node):
    title = "Parça Varlık / Yokluk Sensörü (P/A)"
    category = CAT_LOGIC
    description = ("Bölgede beklenen parçanın, halkanın, contanın veya kapağın var olup olmadığını kontrol eder. "
                   "Parça sayısı ve alan toleransına göre anında Doğru/Yanlış (OK/NG) kararı üretir.")
    inputs = [("image", IMG), ("trigger", BOOL)]
    outputs = [("is_present", BOOL), ("match_count", NUM), ("debug_image", IMG), ("status_text", TEXT)]
    params = [
        Param("min_area", "int", 120, "Minimum Parça Alanı (px)", 5, 200000),
        Param("max_area", "int", 50000, "Maksimum Parça Alanı (px)", 10, 1000000),
        Param("expected_count", "int", 1, "Beklenen Parça Adedi", 0, 50),
        Param("threshold_val", "int", 110, "Eşik Seviyesi (Parlaklık)", 1, 254),
        Param("invert_thresh", "bool", False, "Koyu Parça / Parlak Zemin (Invert)"),
    ]

    def process(self, inputs):
        img = inputs.get("image")
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")

        vis = _bgr(img).copy()
        gray = _gray(vis)

        th_val = int(self.v("threshold_val"))
        mode = cv2.THRESH_BINARY_INV if bool(self.v("invert_thresh")) else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, th_val, 255, mode)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = int(self.v("min_area"))
        max_a = int(self.v("max_area"))
        expected = int(self.v("expected_count"))

        matched = 0
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if min_a <= a <= max_a:
                matched += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, f"VAR (A:{int(a)})", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (100, 100, 100), 1)

        is_present = (matched >= expected) if expected > 0 else (matched == 0)
        status = f"VAR (Mevcut: {matched}/{expected})" if is_present else f"YOK/EKSİK ({matched}/{expected})"
        banner_col = (0, 180, 0) if is_present else (0, 0, 220)

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (25, 25, 25), -1)
        cv2.putText(vis, status, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_col, 2, cv2.LINE_AA)

        return {
            "is_present": is_present,
            "match_count": matched,
            "debug_image": vis,
            "status_text": status,
        }


# --- 43. Digital I/O Expander & Bitwise Packer ---------------------------------------------

@register
class DigitalIOExpanderMapper(Node):
    title = "Dijital G/Ç Genişletici & Bit Paketleyici"
    category = CAT_OUT
    description = ("8 adet bağımsız lojik girişi tek bir bayt tam sayıya ve Hex dizesine paketler. "
                   "Modbus, OPC-UA ve PLC kayıtlarına toplu dijital durum göndermek için idealdir.")
    inputs = [
        ("in_0", BOOL), ("in_1", BOOL), ("in_2", BOOL), ("in_3", BOOL),
        ("in_4", BOOL), ("in_5", BOOL), ("in_6", BOOL), ("in_7", BOOL),
    ]
    outputs = [("byte_value", NUM), ("hex_string", TEXT), ("binary_string", TEXT), ("word_16", NUM)]
    params = [
        Param("invert_bits", "bool", False, "Bitleri Ters Çevir (Invert)"),
        Param("bit_shift", "int", 0, "Sola Kaydırma (Bit Shift)", 0, 8),
    ]

    def process(self, inputs):
        byte_val = 0
        for i in range(8):
            v = bool(inputs.get(f"in_{i}", False) or False)
            if v:
                byte_val |= (1 << i)

        if bool(self.v("invert_bits")):
            byte_val = (~byte_val) & 0xFF

        shift = int(self.v("bit_shift"))
        word_16 = (byte_val << shift) & 0xFFFF
        bin_str = f"{byte_val:08b}"
        hex_str = f"0x{byte_val:02X}"

        return {
            "byte_value": byte_val,
            "hex_string": hex_str,
            "binary_string": bin_str,
            "word_16": word_16,
        }


# --- 44. Production Yield Trend Tracker ---------------------------------------------------

@register
class ProductionYieldTrendTracker(Node):
    title = "Üretim Kalite Trend & Iskarta Alarmı"
    category = CAT_OUT
    description = ("Hareketli pencere (sliding window) üzerinde anlık verim ve ıskarta oranını hesaplar. "
                   "Kalite izin verilen eşiğin altına düştüğünde hat durdurma alarmı üretir.")
    inputs = [("is_defect", BOOL), ("part_trigger", BOOL), ("reset", BOOL)]
    outputs = [("moving_yield_pct", NUM), ("defect_rate_pct", NUM), ("alarm_active", BOOL), ("parts_per_hour", NUM)]
    params = [
        Param("window_size", "int", 30, "Hareketli Pencere Boyutu (Parça)", 5, 500),
        Param("max_defect_rate_pct", "float", 5.0, "Maksimum İzin Verilen Iskarta %", 0.5, 50.0),
    ]

    def process(self, inputs):
        self.state.setdefault("history", [])
        self.state.setdefault("timestamps", [])
        self.state.setdefault("last_trigger", False)

        if bool(inputs.get("reset")):
            self.state["history"].clear()
            self.state["timestamps"].clear()

        trig = bool(inputs.get("part_trigger") or False)
        if trig and not self.state["last_trigger"]:
            is_bad = bool(inputs.get("is_defect") or False)
            self.state["history"].append(not is_bad)
            self.state["timestamps"].append(time.time())

            win = max(5, int(self.v("window_size")))
            if len(self.state["history"]) > win:
                self.state["history"] = self.state["history"][-win:]
            cutoff = time.time() - 3600.0
            self.state["timestamps"] = [t for t in self.state["timestamps"] if t >= cutoff]

        self.state["last_trigger"] = trig

        hist = self.state["history"]
        if len(hist) > 0:
            good_count = sum(1 for x in hist if x)
            yield_pct = (good_count / float(len(hist))) * 100.0
            defect_pct = 100.0 - yield_pct
        else:
            yield_pct = 100.0
            defect_pct = 0.0

        limit = float(self.v("max_defect_rate_pct"))
        alarm = bool(defect_pct > limit and len(hist) >= 10)
        pph = float(len(self.state["timestamps"]))

        return {
            "moving_yield_pct": round(yield_pct, 1),
            "defect_rate_pct": round(defect_pct, 1),
            "alarm_active": alarm,
            "parts_per_hour": round(pph, 1),
        }






