"""Practical inspection blocks: label presence, date code check, colour sorting, edge defects (chips / burrs),
surface defects (scratch / stain), alignment check and kit completeness (expected vs counted per class)."""
from __future__ import annotations

import calendar
import datetime as dt
import math
import re

import cv2
import numpy as np

from ._text import put_text
from ..core import Node, Param, PortType, register
from .vision_more import _ccw_angle, _hsv_mask, _ref_image, hsv_params, parse_roi, to_bgr8

CAT = "2b · Kalite Kontrol ve Ölçüm"
IMG, NUM, BOOL, TEXT, DET, ANY = (PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.TEXT, PortType.DETECTIONS,
                                  PortType.ANY)
GREEN, RED, YELLOW, CYAN = (0, 220, 0), (0, 0, 255), (0, 230, 255), (255, 255, 0)


def _fold90(a: float) -> float:
    """An angle of an undirected line folded into (-90, 90]."""
    a = (a + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _rect_angle(box_pts) -> float:
    """Angle of the long side of a rotated rectangle (4 corner points), counter-clockwise positive, in (-90, 90]."""
    p = np.asarray(box_pts, np.float64).reshape(4, 2)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    e = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    return _fold90(_ccw_angle(float(e[0]), float(e[1])))


# --- 1. label presence ----------------------------------------------------------------------------------------------

LABEL_METHOD = ["açık renkli etiket", "koyu etiket", "renk aralığı (HSV)", "şablon (referans resim)"]


@register
class LabelPresence(Node):
    roi = ("rect", "roi")
    title = "Etiket Var mı"
    category = CAT
    description = ("Bölgede etiket var mı, yerinde mi, eğik mi bakar. Etiketi rengiyle (açık/koyu/HSV) veya bir "
                   "referans resimle (şablon) bulur. Çıkışlar: ok (var, kayma ve eğiklik sınırda), offset (beklenen "
                   "merkezden uzaklık, px), angle (eğiklik, °). Beklenen merkez -1 ise bölgenin ortası alınır.")
    inputs = [("image", IMG), ("reference", IMG)]
    outputs = [("image", IMG), ("ok", BOOL), ("found", BOOL), ("offset", NUM), ("dx", NUM), ("dy", NUM),
               ("angle", NUM), ("score", NUM)]
    params = [Param("roi", "str", "", "Arama bölgesi x,y,w,h (boş = tümü)"),
              Param("method", "choice", LABEL_METHOD[0], "Etiketi nasıl bul", choices=LABEL_METHOD),
              Param("threshold", "int", 170, "Parlaklık eşiği (açık/koyu)", 0, 255),
              *hsv_params(),
              Param("reference", "path", "", "Şablon resmi (şablon yöntemi; giriş bağlıysa o)"),
              Param("min_score", "float", 0.6, "Şablon için en az benzerlik", 0.05, 1.0, 0.05),
              Param("search_angle", "float", 10.0, "Şablon: aranacak eğiklik ± (°)", 0, 45, 1),
              Param("min_area_pct", "float", 5.0, "Renk: etiket en az bölgenin %'si", 0.1, 100, 0.5),
              Param("expected_x", "float", -1.0, "Beklenen merkez X (-1 = bölge ortası)", -1, 100000, 1),
              Param("expected_y", "float", -1.0, "Beklenen merkez Y (-1 = bölge ortası)", -1, 100000, 1),
              Param("max_offset", "float", 20.0, "En çok kayma (px)", 0, 100000, 1),
              Param("max_angle", "float", 5.0, "En çok eğiklik (°)", 0, 90, 0.5)]

    def _by_colour(self, roi_img):
        method = self.v("method")
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        if method == LABEL_METHOD[0]:
            mask = (gray >= int(self.v("threshold"))).astype(np.uint8) * 255
        elif method == LABEL_METHOD[1]:
            mask = (gray < int(self.v("threshold"))).astype(np.uint8) * 255
        else:
            mask = _hsv_mask(roi_img, *(int(self.v(k)) for k in ("h_min", "h_max", "s_min", "s_max", "v_min",
                                                                  "v_max")))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        pct = 100.0 * area / float(roi_img.shape[0] * roi_img.shape[1])
        if pct < float(self.v("min_area_pct")):
            return None
        rect = cv2.minAreaRect(c)
        pts = cv2.boxPoints(rect)
        return {"center": rect[0], "angle": _rect_angle(pts), "pts": pts, "score": round(min(1.0, pct / 100.0), 4)}

    def _by_template(self, roi_img, inputs):
        ref = cv2.cvtColor(_ref_image(self, inputs), cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        th, tw = ref.shape[:2]
        if th > gray.shape[0] or tw > gray.shape[1]:
            raise ValueError(f"Şablon ({tw}x{th}) arama bölgesinden ({gray.shape[1]}x{gray.shape[0]}) büyük")
        rng = float(self.v("search_angle"))
        steps = [0.0] if rng <= 0 else [0.0] + [float(v) for v in np.arange(-rng, rng + 1e-6, max(0.5, rng / 10.0))
                                                if abs(v) > 1e-9]
        H, W = gray.shape[:2]
        centre = (W / 2.0, H / 2.0)
        best = None
        for a in steps:
            # a label turned by +a (counter-clockwise) is straight in the picture turned back by -a
            M = cv2.getRotationMatrix2D(centre, -a, 1.0)
            view = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if a \
                else gray
            res = np.nan_to_num(cv2.matchTemplate(view, ref, cv2.TM_CCOEFF_NORMED), nan=-1.0)
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or score > best[0] + 1e-6:
                c = np.array([loc[0] + tw / 2.0, loc[1] + th / 2.0, 1.0])
                back = cv2.invertAffineTransform(M) @ c
                best = (float(score), (float(back[0]), float(back[1])), a)
        if best is None or best[0] < float(self.v("min_score")):
            return None
        score, c, a = best
        pts = cv2.boxPoints((c, (float(tw), float(th)), -a))
        return {"center": c, "angle": _fold90(a), "pts": pts, "score": round(score, 4)}

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        roi_img = img[y0:y0 + h, x0:x0 + w]
        if self.v("method") == LABEL_METHOD[3]:
            hit = self._by_template(roi_img, inputs)
        else:
            hit = self._by_colour(roi_img)
        out = img.copy()
        if self.v("roi").strip():
            cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), YELLOW, 1)
        ex = float(self.v("expected_x"))
        ey = float(self.v("expected_y"))
        ex = x0 + w / 2.0 if ex < 0 else ex
        ey = y0 + h / 2.0 if ey < 0 else ey
        if hit is None:
            put_text(out, "ETİKET YOK", (10, 30), 0.8, RED, 2)
            return {"image": out, "ok": False, "found": False, "offset": 0.0, "dx": 0.0, "dy": 0.0, "angle": 0.0,
                    "score": 0.0}
        cx, cy = hit["center"][0] + x0, hit["center"][1] + y0
        dx, dy = cx - ex, cy - ey
        off = math.hypot(dx, dy)
        ang = hit["angle"]
        ok = off <= float(self.v("max_offset")) and abs(ang) <= float(self.v("max_angle"))
        pts = (np.asarray(hit["pts"]) + (x0, y0)).astype(np.int32)
        col = GREEN if ok else RED
        cv2.polylines(out, [pts], True, col, 2)
        cv2.drawMarker(out, (int(round(ex)), int(round(ey))), CYAN, cv2.MARKER_CROSS, 16, 1)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 3, col, -1)
        put_text(out, f"{'ETİKET TAMAM' if ok else 'ETİKET HATALI'}  kayma {off:.1f}px  açı {ang:.1f}°", (10, 30),
                 0.7, col, 2)
        return {"image": out, "ok": ok, "found": True, "offset": round(off, 2), "dx": round(dx, 2),
                "dy": round(dy, 2), "angle": round(ang, 2), "score": hit["score"]}


# --- 2. date code check ---------------------------------------------------------------------------------------------

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11,
    "DEC": 12,
    "OCA": 1, "ŞUB": 2, "SUB": 2, "NİS": 4, "NIS": 4, "HAZ": 6, "TEM": 7, "AĞU": 8, "AGU": 8, "EYL": 9, "EKİ": 10,
    "EKI": 10, "EKM": 10, "KAS": 11, "ARA": 12,
    "OCAK": 1, "ŞUBAT": 2, "SUBAT": 2, "MART": 3, "NİSAN": 4, "NISAN": 4, "MAYIS": 5, "HAZİRAN": 6, "HAZIRAN": 6,
    "TEMMUZ": 7, "AĞUSTOS": 8, "AGUSTOS": 8, "EYLÜL": 9, "EYLUL": 9, "EKİM": 10, "EKIM": 10, "KASIM": 11,
    "ARALIK": 12,
}
DATE_FORMATS = ["otomatik", "GG.AA.YYYY", "YYYY-AA-GG", "GG AAA YY (12OCT26)", "AA/YYYY"]
DATE_CHECKS = ["geçmemiş olmalı (SKT)", "bugün olmalı (üretim)", "bugün + N gün olmalı", "sadece okunsun"]
DATE_PICK = ["en geç tarih", "ilk tarih"]
_SEP = r"[./\-]"
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))
_PATTERNS = {
    "dmy": re.compile(rf"(?<!\d)(\d{{1,2}})\s*{_SEP}\s*(\d{{1,2}})\s*{_SEP}\s*(\d{{4}}|\d{{2}})(?!\d)"),
    "ymd": re.compile(rf"(?<!\d)(\d{{4}})\s*{_SEP}\s*(\d{{1,2}})\s*{_SEP}\s*(\d{{1,2}})(?!\d)"),
    "dmony": re.compile(rf"(?<!\d)(\d{{1,2}})\s*[./\-]?\s*({_MON})\s*[./\-]?\s*(\d{{4}}|\d{{2}})(?!\d)"),
    "my": re.compile(r"(?<![\d./\-])(\d{1,2})\s*/\s*(\d{4})(?![\d./\-])"),
}
_FMT_KINDS = {DATE_FORMATS[0]: ("ymd", "dmy", "dmony", "my"), DATE_FORMATS[1]: ("dmy",),
              DATE_FORMATS[2]: ("ymd",), DATE_FORMATS[3]: ("dmony",), DATE_FORMATS[4]: ("my",)}


def _year(s: str) -> int:
    y = int(s)
    return 2000 + y if len(s) == 2 else y


def _ocr_fix(text: str) -> str:
    """Letters OCR often reads instead of digits (O→0, I/l/|→1, S→5, B→8) next to digits or separators, and
    digits it reads inside a month name (0CT → OCT)."""
    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})
    back = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})

    def month(m):                         # "14 0CT 27": the month name read with a digit in it
        mid = m.group(2).upper().translate(back)
        return f"{m.group(1)}{mid}{m.group(3)}" if mid in MONTHS else m.group(0)
    text = re.sub(r"(?<=[\d./\-])[OoIl|SB]|[OoIl|SB](?=[\d./\-])", lambda m: m.group(0).translate(table), text)
    # then a month name that got (or was read with) a digit in it goes back to letters: "140CT27" → "14OCT27"
    return re.sub(r"(?<!\d)(\d{1,2}\s*)([A-Za-zÇĞİÖŞÜçğıöşü0158]{3,7}?)(\s*\d{2}(?:\d{2})?)(?!\d)",
                  lambda m: month(m) if re.search(r"[^\d]", m.group(2)) else m.group(0), text)


def find_dates(text: str, fmt: str = "otomatik") -> list[tuple[int, dt.date | None, str]]:
    """Every date in the text as (position, date or None when impossible like 31.02, matched text)."""
    up = text.upper()
    found: list[tuple[int, int, dt.date | None, str]] = []
    for kind in _FMT_KINDS.get(fmt, _FMT_KINDS["otomatik"]):
        for m in _PATTERNS[kind].finditer(up):
            if any(s < m.end() and m.start() < e for s, e, _, _ in found):
                continue                       # already part of another (more specific) date
            try:
                if kind == "dmy":
                    d = dt.date(_year(m.group(3)), int(m.group(2)), int(m.group(1)))
                elif kind == "ymd":
                    d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                elif kind == "dmony":
                    d = dt.date(_year(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
                else:
                    y, mo = int(m.group(2)), int(m.group(1))
                    d = dt.date(y, mo, calendar.monthrange(y, mo)[1])     # "best before 10/2026" = end of month
            except (ValueError, KeyError):
                d = None
            found.append((m.start(), m.end(), d, m.group(0).strip()))
    found.sort()
    return [(s, d, t) for s, _, d, t in found]


@register
class DateCodeCheck(Node):
    title = "Tarih Kodu Doğrula"
    category = CAT
    description = ("Yazı Oku (OCR) bloğunun okuduğu metinde tarihi bulur (12.10.2026, 2026-10-12, 12OCT26, "
                   "12 EKİ 2026, 10/2026) ve kontrol eder: son kullanma geçmemiş mi, bugünün (vardiyanın) tarihi mi, "
                   "bugünden N gün sonrası mı. 'Anahtar kelime' (ör. SKT) verilirse ondan sonraki tarih alınır. "
                   "Çıkışlar: ok, date (YYYY-AA-GG), days_left (bugünden kaç gün sonra), reason (neden).")
    inputs = [("text", TEXT)]
    outputs = [("ok", BOOL), ("found", BOOL), ("date", TEXT), ("days_left", NUM), ("reason", TEXT)]
    params = [Param("format", "choice", DATE_FORMATS[0], "Tarih biçimi", choices=DATE_FORMATS),
              Param("check", "choice", DATE_CHECKS[0], "Kontrol", choices=DATE_CHECKS),
              Param("keyword", "str", "", "Anahtar kelime (ör. SKT; boş = yok)"),
              Param("pick", "choice", DATE_PICK[0], "Birden çok tarih varsa", choices=DATE_PICK),
              Param("min_days", "int", 0, "SKT: en az kaç gün kalmalı", -36500, 36500),
              Param("offset_days", "int", 365, "Bugün + N gün: N", -36500, 36500),
              Param("tolerance_days", "int", 0, "Gün toleransı (±)", 0, 3650),
              Param("shift_start", "str", "00:00", "Gün başlangıcı (gece vardiyası için ör. 06:00)"),
              Param("fix_ocr", "bool", True, "OCR karışıklıklarını düzelt (O→0, I→1)")]

    def now(self) -> dt.datetime:                     # tests replace the clock
        return dt.datetime.now()

    def _day(self) -> dt.date:
        """The production day: before the shift start time it still counts as the day before (night shift)."""
        text = str(self.v("shift_start") or "").strip() or "00:00"
        m = re.fullmatch(r"(\d{1,2})[:.](\d{2})", text)
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
            raise ValueError(f"Gün başlangıcı SS:DD biçiminde olmalı (ör. 06:00), girilen: '{text}'")
        now = self.now()
        start = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        return (now - dt.timedelta(days=1)).date() if now < start else now.date()

    def process(self, inputs):
        day = self._day()
        text = inputs["text"]
        empty = {"ok": False, "found": False, "date": "", "days_left": 0}
        if text is None or (isinstance(text, str) and not text.strip()):
            return {**empty, "reason": "metin yok (OCR bir şey okumadı)"}
        if not isinstance(text, str):
            text = str(text)
        kw = str(self.v("keyword") or "").strip()
        if kw:
            m = re.search(re.escape(kw), text, re.IGNORECASE)
            if m is None:
                return {**empty, "reason": f"'{kw}' yazısı bulunamadı"}
            text = text[m.end():]
        dates = find_dates(text, self.v("format"))
        if not dates and self.v("fix_ocr"):
            dates = find_dates(_ocr_fix(text), self.v("format"))
        if not dates:
            return {**empty, "reason": "tarih bulunamadı"}
        good = [d for _, d, _ in dates if d is not None]
        if not good:
            return {**empty, "reason": f"geçersiz tarih: {dates[0][2]}"}
        if kw or self.v("pick") == DATE_PICK[1]:
            d = good[0]
        else:
            d = max(good)
        left = (d - day).days
        tol = int(self.v("tolerance_days"))
        check = self.v("check")
        if check == DATE_CHECKS[0]:
            need = int(self.v("min_days"))
            ok = left >= need
            reason = "tamam" if ok else ("süresi geçmiş" if left < 0 else f"{left} gün kalmış (en az {need} olmalı)")
        elif check == DATE_CHECKS[1]:
            ok = abs(left) <= tol
            reason = "tamam" if ok else f"bugünün tarihi değil ({day:%d.%m.%Y} olmalı)"
        elif check == DATE_CHECKS[2]:
            want = day + dt.timedelta(days=int(self.v("offset_days")))
            ok = abs((d - want).days) <= tol
            reason = "tamam" if ok else f"{want:%d.%m.%Y} olmalı ({(d - want).days:+d} gün fark)"
        else:
            ok, reason = True, "okundu"
        return {"ok": ok, "found": True, "date": d.isoformat(), "days_left": left, "reason": reason}


# --- 3. colour sorting ----------------------------------------------------------------------------------------------

COLOR_STAT = ["baskın renk", "ortalama", "medyan"]
COLOR_BG = ["yok sayma", "koyu zemini yok say", "açık zemini yok say"]


def _bgr_to_lab(bgr) -> np.ndarray:
    """CIELAB in real units (L 0..100, a/b about -128..127) of BGR uint8 pixels shaped (N, 3)."""
    px = np.asarray(bgr, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3)


def parse_color_classes(text: str) -> list[tuple[str, np.ndarray]]:
    """'kırmızı=#d02020; yeşil=60,-50,40' (#rrggbb or L,a,b) → [(name, Lab)]."""
    out = []
    for part in re.split(r"[;\n]", str(text or "")):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Renk sınıfı 'ad=#rrggbb' veya 'ad=L,a,b' olmalı, girilen: '{part.strip()}'")
        name, val = (s.strip() for s in part.split("=", 1))
        if not name:
            raise ValueError(f"Renk sınıfının adı boş: '{part.strip()}'")
        if re.fullmatch(r"#?[0-9a-fA-F]{6}", val):
            h = val.lstrip("#")
            rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
            lab = _bgr_to_lab([rgb[::-1]])[0]
        else:
            try:
                nums = [float(v) for v in val.replace(" ", "").split(",")]
            except ValueError:
                nums = []
            if len(nums) != 3 or not all(math.isfinite(v) for v in nums):
                raise ValueError(f"'{name}' rengi #rrggbb veya L,a,b (3 sayı) olmalı, girilen: '{val}'")
            lab = np.array(nums, np.float32)
        out.append((name, np.asarray(lab, np.float32)))
    if not out:
        raise ValueError("Renk sınıfı tanımlı değil (ör. kırmızı=#d02020; mavi=#2040d0)")
    return out


@register
class ColorSort(Node):
    roi = ("rect", "roi")
    title = "Renk ile Ayıkla"
    category = CAT
    description = ("Bölgedeki baskın rengi adlandırdığınız renklerden en yakınına atar (Lab renk uzayında, ΔE "
                   "uzaklığı). Sınıflar 'ad=#rrggbb' veya 'ad=L,a,b' biçiminde, noktalı virgülle. En yakın renk de "
                   "toleranstan uzaksa sınıf 'bilinmiyor' olur. 'index' çıkışı (1, 2, 3…; 0 = bilinmiyor) PLC'ye "
                   "yazılarak ayıklama kapısı sürülebilir.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("class", TEXT), ("index", NUM), ("distance", NUM), ("ok", BOOL), ("lab", ANY)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("classes", "str", "kırmızı=#d02020; yeşil=#20a040; mavi=#2050d0; sarı=#e0d020",
                    "Renkler (ad=#rrggbb veya ad=L,a,b; ...)"),
              Param("tolerance", "float", 30.0, "Tolerans (ΔE)", 1, 300, 1),
              Param("stat", "choice", COLOR_STAT[0], "Bölgenin rengi", choices=COLOR_STAT),
              Param("background", "choice", COLOR_BG[0], "Zemin", choices=COLOR_BG),
              Param("bg_level", "int", 50, "Zemin eşiği (parlaklık)", 0, 255),
              Param("unknown", "str", "bilinmiyor", "Eşleşmeyince sınıf adı")]

    def _colour(self, roi_img) -> np.ndarray | None:
        px = roi_img.reshape(-1, 3)
        if len(px) > 20000:                          # a sample is enough for the colour, keeps it fast
            px = px[:: int(math.ceil(len(px) / 20000))]
        bg = self.v("background")
        if bg != COLOR_BG[0]:
            lum = px.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)
            keep = lum > self.v("bg_level") if bg == COLOR_BG[1] else lum < self.v("bg_level")
            px = px[keep]
        if len(px) == 0:
            return None
        lab = _bgr_to_lab(px)
        stat = self.v("stat")
        if stat == COLOR_STAT[1]:
            return lab.mean(axis=0)
        if stat == COLOR_STAT[2] or len(lab) < 8:
            return np.median(lab, axis=0)
        k = 3
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        cv2.setRNGSeed(1)
        _, labels, centers = cv2.kmeans(lab.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=k)
        return centers[int(np.argmax(counts))]

    def process(self, inputs):
        classes = parse_color_classes(self.v("classes"))
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        lab = self._colour(img[y0:y0 + h, x0:x0 + w])
        out = img.copy()
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), YELLOW, 1)
        if lab is None:
            put_text(out, "Renk yok (hepsi zemin)", (10, 30), 0.7, RED, 2)
            return {"image": out, "class": self.v("unknown"), "index": 0, "distance": 0.0, "ok": False, "lab": None}
        dists = [float(np.linalg.norm(lab - ref)) for _, ref in classes]
        i = int(np.argmin(dists))
        ok = dists[i] <= float(self.v("tolerance"))
        name = classes[i][0] if ok else self.v("unknown")
        put_text(out, f"{name}  ΔE {dists[i]:.1f}", (x0 + 4, max(20, y0 - 8)), 0.7, GREEN if ok else RED, 2)
        return {"image": out, "class": name, "index": i + 1 if ok else 0, "distance": round(dists[i], 2), "ok": ok,
                "lab": [round(float(v), 2) for v in lab]}


# --- 4. edge defects ------------------------------------------------------------------------------------------------

PART_POL = ["otomatik (Otsu)", "açık parça / koyu zemin", "koyu parça / açık zemin"]
EDGE_MODE = ["dışbükey zarf (çentik, kırık)", "çentik + çapak (dışa taşan)"]


def part_mask(gray: np.ndarray, polarity: str, threshold: int) -> np.ndarray:
    """Binary mask of the part (255) on its background."""
    if polarity == PART_POL[0] or threshold <= 0:
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if polarity == PART_POL[2] or (polarity == PART_POL[0] and
                                        np.count_nonzero(m[0, :]) + np.count_nonzero(m[-1, :]) +
                                        np.count_nonzero(m[:, 0]) + np.count_nonzero(m[:, -1]) >
                                        (m.shape[0] + m.shape[1])):
            m = 255 - m                  # the border is mostly "on": that is the background
        return m
    if polarity == PART_POL[1]:
        return (gray > threshold).astype(np.uint8) * 255
    return (gray <= threshold).astype(np.uint8) * 255


@register
class EdgeDefect(Node):
    roi = ("rect", "roi")
    title = "Kenar Kusuru Bul"
    category = CAT
    description = ("Parçanın dış hattındaki kırık, çentik ve çapakları bulur. 'Dışbükey zarf' yöntemi düz/dışbükey "
                   "parçalarda (kare, daire, kapak, pul) içe doğru kusurları bulur; 'çentik + çapak' dışa taşan çapakları "
                   "da bulur: yumuşatma boyutu çapak genişliğinden büyük olmalı; köşeler çapak sanılmasın diye "
                   "yumuşatma boyutunun ~0,2 katından kısa çapaklar sayılmaz. Çıkışlar: kusur "
                   "sayısı, en derin kusur (px) ve işaretli resim.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("ok", BOOL), ("count", NUM), ("max_depth", NUM), ("detections", DET),
               ("found", BOOL)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("polarity", "choice", PART_POL[0], "Parça", choices=PART_POL),
              Param("threshold", "int", 0, "Eşik (0 = otomatik)", 0, 255),
              Param("mode", "choice", EDGE_MODE[0], "Yöntem", choices=EDGE_MODE),
              Param("smooth", "int", 15, "Yumuşatma boyutu (px)", 3, 501, 2),
              Param("min_depth", "float", 4.0, "En az kusur derinliği (px)", 0.5, 10000, 0.5),
              Param("min_part_area", "int", 500, "En küçük parça alanı (px)", 1, 100_000_000)]

    def _hull(self, c, x0, y0):
        hull = cv2.convexHull(c, returnPoints=False)
        if hull is None or len(hull) < 3:
            return []
        try:
            defects = cv2.convexityDefects(c, hull)
        except cv2.error:
            # some contours give a hull whose index order OpenCV then rejects: a sorted copy is accepted
            try:
                defects = cv2.convexityDefects(c, np.ascontiguousarray(np.sort(hull.ravel())[::-1].reshape(-1, 1)))
            except cv2.error:
                return []
        out = []
        for s, e, f, d in (defects.reshape(-1, 4) if defects is not None else []):
            depth = d / 256.0
            if depth < float(self.v("min_depth")):
                continue
            p_s, p_e, p_f = c[s][0], c[e][0], c[f][0]
            xs, ys = [p_s[0], p_e[0], p_f[0]], [p_s[1], p_e[1], p_f[1]]
            out.append({"box": (int(min(xs)) + x0, int(min(ys)) + y0, int(max(xs) - min(xs)) + 1,
                                int(max(ys) - min(ys)) + 1), "score": 1.0, "class_id": 0,
                        "label": f"çentik {depth:.1f}px", "depth": round(depth, 2), "kind": "çentik",
                        "center": (float(p_f[0]) + x0, float(p_f[1]) + y0),
                        "points": [(int(p[0]) + x0, int(p[1]) + y0) for p in (p_s, p_f, p_e)]})
        return out

    def _burrs(self, mask, x0, y0):
        """Burrs: what an opening with a disc of the smoothing size takes off the part. Returns (burrs, the part
        without them)."""
        k = int(self.v("smooth")) | 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        outside = cv2.distanceTransform(255 - opened, cv2.DIST_L2, 5)
        # the opening also rounds the part's own sharp corners (a 90° corner by about 0.21 × the size): anything
        # that shallow is not reported, so corners are never taken for burrs
        least = max(float(self.v("min_depth")), 0.21 * k + 1.0)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(((mask > 0) & (opened == 0)).astype(np.uint8),
                                                               connectivity=8)
        out = []
        for i in range(1, n):
            comp = labels == i
            depth = float(outside[comp].max()) + 0.5         # +0.5: the pixel's own half width
            if depth < least:
                continue
            bx, by, bw, bh, _area = (int(v) for v in stats[i])
            ys, xs = np.nonzero(comp & (outside >= depth - 1.0))
            cx, cy = (float(xs.mean()), float(ys.mean())) if len(xs) else (bx + bw / 2.0, by + bh / 2.0)
            out.append({"box": (bx + x0, by + y0, bw, bh), "score": 1.0, "class_id": 1,
                        "label": f"çapak {depth:.1f}px", "depth": round(depth, 2), "kind": "çapak",
                        "center": (cx + x0, cy + y0)})
        return out, opened

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        gray = cv2.cvtColor(img[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY)
        mask = part_mask(cv2.GaussianBlur(gray, (3, 3), 0), self.v("polarity"), int(self.v("threshold")))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= int(self.v("min_part_area"))]
        out = img.copy()
        if self.v("roi").strip():
            cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), YELLOW, 1)
        if not cnts:
            put_text(out, "Parça bulunamadı", (10, 30), 0.7, RED, 2)
            return {"image": out, "ok": False, "count": 0, "max_depth": 0.0, "detections": [], "found": False}
        part = max(cnts, key=cv2.contourArea)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [part], -1, 255, -1)        # holes inside the part are not edge defects
        if self.v("mode") == EDGE_MODE[0]:
            dets = self._hull(part, x0, y0)
        else:
            # burrs first; the notches are then looked for on the part without its burrs, whose hull a burr
            # would otherwise lift off the straight edge next to it
            dets, opened = self._burrs(filled, x0, y0)
            rest, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if rest:
                dets += self._hull(max(rest, key=cv2.contourArea), x0, y0)
        dets.sort(key=lambda d: -d["depth"])
        cv2.drawContours(out[y0:y0 + h, x0:x0 + w], [part], -1, GREEN if not dets else YELLOW, 1)
        for d in dets:
            bx, by, bw, bh = d["box"]
            cv2.rectangle(out, (bx - 3, by - 3), (bx + bw + 2, by + bh + 2), RED, 2)
            cx, cy = d["center"]
            cv2.circle(out, (int(cx), int(cy)), 3, RED, -1)
        top = dets[0]["depth"] if dets else 0.0
        put_text(out, f"Kenar kusuru: {len(dets)}" + (f"  en derin {top:.1f}px" if dets else ""), (10, 30), 0.7,
                 RED if dets else GREEN, 2)
        return {"image": out, "ok": not dets, "count": len(dets), "max_depth": top, "detections": dets,
                "found": True}


# --- 5. surface defects ---------------------------------------------------------------------------------------------

SURF_METHOD = ["hepsi", "çizik (Laplace)", "leke (arka plan farkı)", "doku (yerel varyans)"]


def _robust_z(f: np.ndarray, floor: float) -> np.ndarray:
    """How unusual each value is compared with the whole region: (f - median) / spread, the spread being the
    median absolute deviation, but never below `floor` (a perfectly flat surface must not see noise as defects)."""
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med))) * 1.4826
    return (f - med) / max(mad, floor)


@register
class SurfaceDefect(Node):
    roi = ("rect", "roi")
    title = "Doku / Yüzey Kusuru"
    category = CAT
    description = ("Bölgedeki yüzeyde çevresinden farklı görünen yerleri bulur: çizik (ince keskin çizgiler, "
                   "Laplace), leke (yerel arka plandan koyu/açık alanlar) ve doku bozukluğu (yerel varyans). "
                   "Hassasiyet yüksekse daha küçük farklar da kusur sayılır. Hiç kusur yoksa ok doğru.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG), ("mask", IMG), ("ok", BOOL), ("count", NUM), ("area", NUM), ("detections", DET)]
    params = [Param("roi", "str", "", "Bölge x,y,w,h (boş = tümü)"),
              Param("method", "choice", SURF_METHOD[0], "Ne aransın", choices=SURF_METHOD),
              Param("sensitivity", "int", 50, "Hassasiyet (1-100)", 1, 100),
              Param("window", "int", 15, "Pencere boyutu (px)", 3, 255, 2),
              Param("min_area", "int", 20, "En küçük kusur alanı (px)", 1, 10_000_000)]

    def process(self, inputs):
        img = to_bgr8(inputs["image"])
        x0, y0, w, h = parse_roi(self.v("roi"), img.shape)
        gray = cv2.cvtColor(img[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        win = int(self.v("window")) | 1
        win = max(3, min(win, (min(w, h) - 1) | 1)) if min(w, h) > 3 else 3
        # sensitivity 1 → 14 spreads, 50 → ~7.6, 100 → 1.5 above normal
        k = 1.5 + (100 - int(self.v("sensitivity"))) * 0.125
        method = self.v("method")
        mask = np.zeros(gray.shape, bool)
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        if method in (SURF_METHOD[0], SURF_METHOD[1]):
            lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
            lap = cv2.blur(lap, (3, 3))
            mask |= _robust_z(lap, 4.0) > k
        if method in (SURF_METHOD[0], SURF_METHOD[2]):
            big = min(255, max(3, (win * 3) | 1))      # the background: larger than any stain to be found
            bg = cv2.medianBlur(np.clip(g + 0.5, 0, 255).astype(np.uint8), big).astype(np.float32)
            mask |= np.abs(_robust_z(g - bg, 3.0)) > k
        if method in (SURF_METHOD[0], SURF_METHOD[3]):
            mean = cv2.blur(g, (win, win))
            var = np.maximum(cv2.blur(g * g, (win, win)) - mean * mean, 0)
            mask |= _robust_z(np.sqrt(var), 2.0) > k
        m8 = mask.astype(np.uint8) * 255
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m8, connectivity=8)
        out = img.copy()
        if self.v("roi").strip():
            cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), YELLOW, 1)
        full = np.zeros(img.shape[:2], np.uint8)
        dets = []
        for i in range(1, n):
            bx, by, bw, bh, area = (int(v) for v in stats[i])
            if area < int(self.v("min_area")):
                continue
            comp = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8) * 255
            full[y0 + by:y0 + by + bh, x0 + bx:x0 + bx + bw] |= comp
            ys, xs = np.nonzero(comp)
            (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            elong = max(rw, rh) / max(1.0, min(rw, rh))      # a thin line at any angle is a scratch
            kind = "çizik" if elong >= 4 else "leke"
            dets.append({"box": (bx + x0, by + y0, bw, bh), "score": 1.0, "class_id": 0 if kind == "çizik" else 1,
                         "label": f"{kind} {area}px", "area": float(area), "kind": kind,
                         "center": (float(cents[i][0]) + x0, float(cents[i][1]) + y0)})
        dets.sort(key=lambda d: -d["area"])
        for d in dets:
            bx, by, bw, bh = d["box"]
            cv2.rectangle(out, (bx - 2, by - 2), (bx + bw + 1, by + bh + 1), RED, 2)
        total = sum(d["area"] for d in dets)
        put_text(out, f"Yüzey kusuru: {len(dets)}  alan {int(total)}px", (10, 30), 0.7, RED if dets else GREEN, 2)
        return {"image": out, "mask": full, "ok": not dets, "count": len(dets), "area": total, "detections": dets}


# --- 6. alignment check ---------------------------------------------------------------------------------------------

ALIGN_MODE = ["iki tespitin merkezleri", "en uzun çizgi"]
ALIGN_PICK = ["soldan ilk iki", "yukarıdan ilk iki", "en büyük iki", "en emin iki"]


def _center(d):
    if d.get("center") is not None:
        return float(d["center"][0]), float(d["center"][1])
    x, y, w, h = d["box"]
    return x + w / 2.0, y + h / 2.0


@register
class AlignmentCheck(Node):
    title = "Hizalama Kontrolü"
    category = CAT
    description = ("İki özellikten (iki delik, iki işaretçi; ya da Çizgi Bul'un çizgisi) parçanın açısını ve "
                   "konumunu bulur, beklenen açı ve merkezle karşılaştırır. Açı sapması ve kayma toleranstaysa ok "
                   "doğru. px_per_mm verilirse kayma mm olur. Beklenen X/Y -1 ise konum denetlenmez.")
    inputs = [("detections", DET), ("image", IMG), ("px_per_mm", NUM)]
    outputs = [("image", IMG), ("ok", BOOL), ("found", BOOL), ("angle", NUM), ("angle_error", NUM),
               ("offset", NUM), ("dx", NUM), ("dy", NUM)]
    params = [Param("mode", "choice", ALIGN_MODE[0], "Ne ölçülsün", choices=ALIGN_MODE),
              Param("pick", "choice", ALIGN_PICK[0], "Hangi iki tespit", choices=ALIGN_PICK),
              Param("label", "str", "", "Sadece bu sınıf (boş = hepsi)"),
              Param("nominal_angle", "float", 0.0, "Beklenen açı (°)", -180, 180, 0.5),
              Param("angle_tol", "float", 2.0, "Açı toleransı ± (°)", 0, 180, 0.1),
              Param("nominal_x", "float", -1.0, "Beklenen merkez X (px, -1 = bakma)", -1, 100000, 1),
              Param("nominal_y", "float", -1.0, "Beklenen merkez Y (px, -1 = bakma)", -1, 100000, 1),
              Param("offset_tol", "float", 5.0, "Kayma toleransı (px veya mm)", 0, 100000, 0.1),
              Param("px_per_mm", "float", 0.0, "mm başına piksel (0 = piksel; giriş bağlıysa o)", 0, 10000, 0.01)]

    def process(self, inputs):
        dets = inputs["detections"]
        if dets is not None and not isinstance(dets, (list, tuple)):
            raise ValueError("detections girişine tespit listesi bağlanmalı")
        dets = [d for d in (dets or []) if isinstance(d, dict) and ("box" in d or "center" in d)]
        want = str(self.v("label") or "").strip().casefold()
        if want:
            dets = [d for d in dets if str(d.get("label", "")).split(" #")[0].strip().casefold() == want]
        img = inputs["image"]
        out = to_bgr8(img).copy() if img is not None else None
        ppm = _num(inputs["px_per_mm"])
        if ppm is None or ppm <= 0:
            ppm = float(self.v("px_per_mm") or 0.0)
        scale = ppm if ppm > 0 else 1.0
        none = {"image": out, "ok": False, "found": False, "angle": 0.0, "angle_error": 0.0, "offset": 0.0,
                "dx": 0.0, "dy": 0.0}
        if self.v("mode") == ALIGN_MODE[1]:
            lines = [d for d in dets if d.get("line") is not None and len(d["line"]) == 4]
            if not lines:
                if out is not None:
                    put_text(out, "Çizgi yok", (10, 30), 0.7, RED, 2)
                return none
            x1, y1, x2, y2 = (float(v) for v in max(lines, key=lambda d: math.hypot(d["line"][2] - d["line"][0],
                                                                                    d["line"][3] - d["line"][1]))["line"])
        else:
            if len(dets) < 2:
                if out is not None:
                    put_text(out, f"İki özellik gerekli ({len(dets)} bulundu)", (10, 30), 0.7, RED, 2)
                return none
            pick = self.v("pick")
            if pick == ALIGN_PICK[0]:
                two = sorted(dets, key=lambda d: _center(d)[0])[:2]
            elif pick == ALIGN_PICK[1]:
                two = sorted(dets, key=lambda d: _center(d)[1])[:2]
            elif pick == ALIGN_PICK[2]:
                two = sorted(dets, key=lambda d: -(d.get("box", (0, 0, 0, 0))[2] * d.get("box", (0, 0, 0, 0))[3]))[:2]
                two.sort(key=lambda d: _center(d)[0])
            else:
                two = sorted(dets, key=lambda d: -float(d.get("score", 0)))[:2]
                two.sort(key=lambda d: _center(d)[0])
            (x1, y1), (x2, y2) = _center(two[0]), _center(two[1])
        if math.hypot(x2 - x1, y2 - y1) < 1e-6:
            raise ValueError("İki özellik aynı yerde: açı ölçülemez")
        ang = _ccw_angle(x2 - x1, y2 - y1)
        nominal = float(self.v("nominal_angle"))
        err = (ang - nominal + 180.0) % 360.0 - 180.0
        if self.v("mode") == ALIGN_MODE[1] or abs(nominal) <= 90:
            err = _fold90(err)               # a line has no direction: 179° is 1° off from 0°
            ang = _fold90(ang)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        nx, ny = float(self.v("nominal_x")), float(self.v("nominal_y"))
        dx = (mx - nx) / scale if nx >= 0 else 0.0
        dy = (my - ny) / scale if ny >= 0 else 0.0
        off = math.hypot(dx, dy)
        ok = abs(err) <= float(self.v("angle_tol")) and off <= float(self.v("offset_tol"))
        if out is not None:
            col = GREEN if ok else RED
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            for p in ((x1, y1), (x2, y2)):
                cv2.circle(out, (int(p[0]), int(p[1])), 4, col, -1)
            if nx >= 0 or ny >= 0:
                cv2.drawMarker(out, (int(nx if nx >= 0 else mx), int(ny if ny >= 0 else my)), CYAN,
                               cv2.MARKER_CROSS, 16, 1)
            unit = "mm" if ppm > 0 else "px"
            put_text(out, f"açı {ang:.2f}° (sapma {err:+.2f}°)  kayma {off:.2f}{unit}", (10, 30), 0.7, col, 2)
        return {"image": out, "ok": ok, "found": True, "angle": round(ang, 3), "angle_error": round(err, 3),
                "offset": round(off, 3), "dx": round(dx, 3), "dy": round(dy, 3)}


# --- 7. counts against a spec ---------------------------------------------------------------------------------------

COUNT_MODE = ["tam eşit olmalı", "en az bu kadar"]


def _norm_label(s) -> str:
    """'vida #3 0.91' / 'Vida' → 'vida' (tracking number and score dropped, case folded)."""
    s = str(s or "").split(" #")[0].strip()
    s = re.sub(r"\s+\d+(?:[.,]\d+)?%?$", "", s)
    return s.casefold()


def parse_count_spec(text: str) -> dict[str, int]:
    """'3 vida, 2 somun' / 'vida=3; somun: 2' / 'vida x3' → {'vida': 3, 'somun': 2}."""
    spec: dict[str, int] = {}
    for part in re.split(r"[,;\n]", str(text or "")):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*(?:x|×|adet)?\s+(.+)", part, re.I) or \
            re.fullmatch(r"(.+?)\s*(?:=|:|\bx|×)\s*(\d+)(?:\s*adet)?", part, re.I)
        if not m:
            raise ValueError(f"Beklenen adet anlaşılmadı: '{part}' (ör. '3 vida, 2 somun' veya 'vida=3')")
        a, b = m.groups()
        n, name = (int(a), b) if a.isdigit() else (int(b), a)
        key = _norm_label(name)
        if not key:
            raise ValueError(f"Sınıf adı boş: '{part}'")
        spec[key] = spec.get(key, 0) + n
    if not spec:
        raise ValueError("Beklenen adetler boş (ör. '3 vida, 2 somun')")
    return spec


@register
class CountCompare(Node):
    title = "Sayaç Karşılaştır"
    category = CAT
    description = ("Kutu/kit eksiksiz mi: bulunan nesneleri sınıfına göre sayar ve beklenenle karşılaştırır "
                   "(ör. '3 vida, 2 somun'). 'detections' (YOLO, Şekil Tanı…) veya 'counts' (Sınıfa Göre Say'ın "
                   "counts çıkışı) bağlanabilir. Çıkışlar: ok, missing (ör. '1 vida eksik'), eksik ve fazla adet.")
    inputs = [("detections", DET), ("counts", ANY)]
    outputs = [("ok", BOOL), ("missing", TEXT), ("missing_count", NUM), ("extra_count", NUM), ("summary", TEXT),
               ("counts", ANY)]
    params = [Param("spec", "str", "3 vida, 2 somun", "Beklenen (ör. 3 vida, 2 somun)"),
              Param("mode", "choice", COUNT_MODE[0], "Kural", choices=COUNT_MODE),
              Param("others", "bool", False, "Listede olmayan sınıflar da hata")]

    def process(self, inputs):
        spec = parse_count_spec(self.v("spec"))
        counts: dict[str, int] = {}
        dets, given = inputs["detections"], inputs["counts"]
        if dets is not None:
            if not isinstance(dets, (list, tuple)):
                raise ValueError("detections girişine tespit listesi bağlanmalı")
            for d in dets:
                if isinstance(d, dict):
                    k = _norm_label(d.get("label", ""))
                    counts[k] = counts.get(k, 0) + 1
        if given is not None:
            if isinstance(given, str):
                given = parse_count_spec(given) if given.strip() else {}
            if not isinstance(given, dict):
                raise ValueError("counts girişine {sınıf: adet} sözlüğü veya '3 vida, 2 somun' metni bağlanmalı")
            for k, v in given.items():
                n = _num(v)
                if n is not None:
                    key = _norm_label(k)
                    counts[key] = counts.get(key, 0) + int(n)
        exact = self.v("mode") == COUNT_MODE[0]
        problems, missing, extra = [], 0, 0
        for name, want in spec.items():
            have = counts.get(name, 0)
            if have < want:
                problems.append(f"{want - have} {name} eksik")
                missing += want - have
            elif exact and have > want:
                problems.append(f"{have - want} {name} fazla")
                extra += have - want
        if self.v("others"):
            for name, have in sorted(counts.items()):
                if name not in spec and have:
                    problems.append(f"{have} {name or '(adsız)'} fazla")
                    extra += have
        summary = ", ".join(f"{counts.get(k, 0)}/{v} {k}" for k, v in spec.items())
        return {"ok": not problems, "missing": ", ".join(problems) if problems else "tamam",
                "missing_count": missing, "extra_count": extra, "summary": summary, "counts": counts}
