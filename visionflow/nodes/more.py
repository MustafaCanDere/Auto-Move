"""0.4 blocks: batch image input, geometry, shape analysis, charts, database, e-mail, Telegram."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import smtplib
import sqlite3
import threading
import time
import urllib.request
import uuid
from email.message import EmailMessage

import cv2
import numpy as np

from ._text import put_text, text_size
from ..core import Node, Param, PortType, StopFlow, register
from .learning import IMAGE_EXT, _imread
from .outputs import _rising
from .processing import _bgr, _gray

IMG, NUM, BOOL, TEXT, DET = PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.TEXT, PortType.DETECTIONS


def _need(img):
    if img is None:
        raise ValueError("Görüntü girişi bağlı değil")
    return img


# --- input ---------------------------------------------------------------------------------------

@register
class ImageFolder(Node):
    setup_params = ("folder", "recursive")   # read only when the block starts: changing them restarts it
    title = "Klasördeki Resimler"
    category = "1 · Giriş"
    description = ("Bir klasördeki resimleri sırayla verir: toplu kalite kontrol, model deneme, arşiv tarama. "
                   "Son resimden sonra durur veya başa sarar.")
    outputs = [("image", IMG), ("name", TEXT), ("index", NUM), ("total", NUM)]
    params = [Param("folder", "folder", "resimler", "Klasör"),
              Param("loop", "bool", False, "Başa sar"),
              Param("delay", "float", 0.5, "Resim başına süre (sn)", 0, 60, 0.1),
              Param("recursive", "bool", False, "Alt klasörler dahil")]
    is_source = True

    def setup(self):
        root = pathlib.Path(self.v("folder"))
        if not root.is_dir():
            raise ValueError(f"Klasör yok: {root}")
        it = root.rglob("*") if self.v("recursive") else root.iterdir()
        files = sorted(p for p in it if p.suffix.lower() in IMAGE_EXT)
        if not files:
            raise ValueError(f"Klasörde resim yok: {root}")
        self.state.update(files=files, i=0, last=0.0)

    def process(self, inputs):
        st = self.state
        # keep showing the current picture until its time is up (no sleeping: live tuning stays responsive)
        if "out" in st and time.perf_counter() - st["last"] < self.v("delay"):
            return st["out"]
        if st["i"] >= len(st["files"]):
            if not self.v("loop"):
                raise StopFlow()
            st["i"] = 0
        st["last"] = time.perf_counter()
        path = st["files"][st["i"]]
        st["i"] += 1
        img = _imread(path)
        if img is None:
            raise ValueError(f"Okunamadı: {path.name}")
        st["out"] = {"image": img, "name": path.name, "index": st["i"], "total": len(st["files"])}
        return st["out"]


# --- geometry & shapes -----------------------------------------------------------------------------

@register
class PerspectiveWarp(Node):
    roi = ("quad", "points")
    title = "Perspektif Düzelt"
    category = "2 · Görüntü İşleme"
    description = ("Eğik açıdan çekilen bir yüzeyi (etiket, bant, belge) dört köşesinden düz, kuşbakışı hale "
                   "getirir. Köşeler: sol-üst, sağ-üst, sağ-alt, sol-alt.")
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [Param("points", "str", "100,100,540,80,560,400,80,420", "x1,y1,…,x4,y4"),
              Param("width", "int", 600, "Çıkış genişliği", 10, 8000),
              Param("height", "int", 400, "Çıkış yüksekliği", 10, 8000)]

    def process(self, inputs):
        img = _need(inputs["image"])
        pts = [float(v) for v in self.v("points").split(",")]
        if len(pts) != 8:
            raise ValueError("8 sayı gerekli (4 köşe)")
        w, h = self.v("width"), self.v("height")
        src = np.float32(pts).reshape(4, 2)
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        return {"image": cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))}


SHAPES = ["hepsi", "üçgen", "kare", "dikdörtgen", "beşgen", "altıgen", "daire", "elips"]


def classify_shape(c) -> str:
    peri = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    n = len(approx)
    if n == 3:
        return "üçgen"
    if n == 4:
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        return "kare" if 0.88 <= w / max(h, 1e-6) <= 1.12 else "dikdörtgen"
    if n == 5:
        return "beşgen"
    if n == 6:
        return "altıgen"
    circularity = 4 * np.pi * area / max(peri * peri, 1e-6)
    return "daire" if circularity > 0.85 else "elips"


@register
class ShapeDetector(Node):
    title = "Şekil Tanı"
    category = "2b · Kalite Kontrol ve Ölçüm"
    description = "Konturları üçgen, kare, dikdörtgen, beşgen, altıgen, daire veya elips olarak sınıflandırır."
    inputs = [("contours", PortType.CONTOURS)]
    outputs = [("detections", DET), ("count", NUM)]
    params = [Param("only", "choice", "hepsi", "Sadece bu şekil", choices=SHAPES)]

    def process(self, inputs):
        dets = []
        for c in inputs["contours"] or []:
            name = classify_shape(c)
            if self.v("only") not in ("hepsi", name):
                continue
            dets.append({"box": tuple(int(v) for v in cv2.boundingRect(c)), "score": 1.0,
                         "class_id": SHAPES.index(name), "label": name})
        return {"detections": dets, "count": len(dets)}


@register
class HoughCircles(Node):
    title = "Daire Bul (Hough)"
    category = "2b · Kalite Kontrol ve Ölçüm"
    description = "Görüntüde daireleri bulur: delik, vida başı, kapak, tablet sayma ve çap ölçme."
    inputs = [("image", IMG)]
    outputs = [("detections", DET), ("count", NUM), ("image", IMG)]
    params = [Param("min_r", "int", 10, "Min yarıçap", 1, 2000), Param("max_r", "int", 80, "Max yarıçap", 1, 4000),
              Param("min_dist", "int", 20, "Merkezler arası min", 1, 4000),
              Param("roundness", "float", 0.8, "Yuvarlaklık (yüksek = sadece düzgün daireler)", 0.3, 0.99, 0.05),
              Param("edge", "int", 200, "Kenar eşiği", 10, 1000)]

    def process(self, inputs):
        img = _need(inputs["image"])
        g = cv2.GaussianBlur(_gray(img), (5, 5), 1.5)
        circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, 1.5, self.v("min_dist"), param1=self.v("edge"),
                                   param2=min(self.v("roundness"), 0.99),   # OpenCV needs < 1
                                    minRadius=self.v("min_r"), maxRadius=self.v("max_r"))
        kept: list[tuple[int, int, int]] = []
        if circles is not None:   # ALT does not enforce min_dist itself; keep the first (strongest) of each cluster
            for x, y, r in np.round(circles[0]).astype(int):
                if all((x - kx) ** 2 + (y - ky) ** 2 >= self.v("min_dist") ** 2 for kx, ky, _ in kept):
                    kept.append((int(x), int(y), int(r)))
        out = _bgr(img).copy()
        dets = []
        for x, y, r in kept:
            cv2.circle(out, (x, y), r, (0, 255, 0), 2)
            cv2.circle(out, (x, y), 2, (0, 0, 255), 3)
            dets.append({"box": (x - r, y - r, 2 * r, 2 * r), "score": 1.0,
                         "class_id": 0, "label": f"Ø{2 * r}px", "radius": r})
        return {"detections": dets, "count": len(dets), "image": out}


@register
class CountByLabel(Node):
    title = "Sınıfa Göre Say"
    category = "3 · Yapay Zeka"
    description = "Tespitleri sınıf adına göre sayar; 3 sınıfın adedini ayrı çıkışlara, hepsini metin olarak verir."
    inputs = [("detections", DET)]
    outputs = [("a", NUM), ("b", NUM), ("c", NUM), ("summary", TEXT), ("counts", PortType.ANY)]
    params = [Param("labels", "str", "person,car,truck", "a,b,c sınıfları")]

    def process(self, inputs):
        counts: dict[str, int] = {}
        for d in inputs["detections"] or []:
            name = d["label"].split(" #")[0]
            counts[name] = counts.get(name, 0) + 1
        names = [s.strip() for s in self.v("labels").split(",")] + ["", "", ""]
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "yok"
        return {"a": counts.get(names[0], 0), "b": counts.get(names[1], 0), "c": counts.get(names[2], 0),
                "summary": summary, "counts": counts}


@register
class ZoneCounter(Node):
    roi = ("poly", "zone")
    title = "Bölgedeki Nesneler"
    category = "3 · Yapay Zeka"
    description = ("Çokgen bir bölgenin (park yeri, tehlikeli alan, bant üstü) içindeki tespitleri sayar. "
                   "Kutunun alt-orta noktası bölgedeyse içeride sayılır.")
    inputs = [("detections", DET), ("image", IMG)]
    outputs = [("inside", DET), ("count", NUM), ("occupied", BOOL), ("image", IMG)]
    params = [Param("zone", "str", "100,100,540,100,540,380,100,380", "Bölge köşeleri x1,y1,x2,y2,…"),
              Param("point", "choice", "alt-orta", "Referans nokta", choices=["alt-orta", "merkez"]),
              Param("min_count", "int", 1, "Dolu sayılması için", 1, 1000)]

    def process(self, inputs):
        vals = [float(v) for v in self.v("zone").split(",")]
        if len(vals) < 6 or len(vals) % 2:
            raise ValueError("En az 3 köşe (6 sayı) gerekli")
        poly = np.float32(vals).reshape(-1, 2)
        inside = []
        for d in inputs["detections"] or []:
            x, y, w, h = d["box"]
            pt = (x + w / 2, y + h) if self.v("point") == "alt-orta" else (x + w / 2, y + h / 2)
            if cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0:
                inside.append(d)
        occupied = len(inside) >= self.v("min_count")
        out = None
        if inputs["image"] is not None:
            out = _bgr(inputs["image"]).copy()
            color = (0, 0, 255) if occupied else (0, 200, 0)
            overlay = out.copy()
            cv2.fillPoly(overlay, [np.int32(poly)], color)
            out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
            cv2.polylines(out, [np.int32(poly)], True, color, 2)
            for d in inside:
                x, y, w, h = d["box"]
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            put_text(out, f"Bölge: {len(inside)}", (int(poly[0][0]) + 4, int(poly[0][1]) + 20), 0.6, color, 2)
        return {"inside": inside, "count": len(inside), "occupied": occupied, "image": out}


# --- enhancement & background ----------------------------------------------------------------------

@register
class Enhance(Node):
    title = "Keskinleştir / Gürültü Azalt"
    category = "2 · Görüntü İşleme"
    description = "Görüntüyü keskinleştirir veya kenarları koruyarak gürültüyü azaltır (OCR, ölçüm öncesi)."
    inputs = [("image", IMG)]
    outputs = [("image", IMG)]
    params = [Param("mode", "choice", "keskinleştir", "İşlem",
                    choices=["keskinleştir", "gürültü azalt", "kenar koruyan yumuşatma", "detay artır"]),
              Param("strength", "float", 1.0, "Güç", 0.1, 5.0, 0.1)]

    def process(self, inputs):
        img = _need(inputs["image"])
        k, mode = self.v("strength"), self.v("mode")
        if mode == "keskinleştir":
            blur = cv2.GaussianBlur(img, (0, 0), 2)
            return {"image": cv2.addWeighted(img, 1 + k, blur, -k, 0)}
        if mode == "gürültü azalt":
            h = 3 + 7 * k
            if img.ndim == 2:
                return {"image": cv2.fastNlMeansDenoising(img, None, h)}
            return {"image": cv2.fastNlMeansDenoisingColored(img, None, h, h)}
        if mode == "kenar koruyan yumuşatma":
            return {"image": cv2.bilateralFilter(img, 9, 25 * k, 25 * k)}
        return {"image": cv2.detailEnhance(_bgr(img), sigma_s=10, sigma_r=min(0.15 * k, 1.0))}


@register
class BackgroundSubtract(Node):
    setup_params = ("history", "threshold")   # read only when the block starts: changing them restarts it
    title = "Arka Plan Çıkar"
    category = "2 · Görüntü İşleme"
    description = ("Sabit kamerada sahneyi öğrenir, hareket eden / yeni gelen nesnelerin maskesini verir "
                   "(bant üstündeki ürün, giren kişi). Gölgeler yok sayılır.")
    inputs = [("image", IMG), ("reset", BOOL)]
    outputs = [("mask", IMG), ("foreground", IMG), ("ratio", NUM)]
    params = [Param("history", "int", 300, "Öğrenme süresi (kare)", 10, 10000),
              Param("threshold", "float", 25.0, "Hassasiyet (düşük = hassas)", 1, 400, 1),
              Param("learn", "float", -1.0, "Öğrenme hızı (-1 = otomatik, 0 = dondur)", -1, 1, 0.001)]

    def setup(self):
        self.state["bg"] = cv2.createBackgroundSubtractorMOG2(self.v("history"), self.v("threshold"), True)

    def process(self, inputs):
        img = _need(inputs["image"])
        if inputs["reset"]:
            self.setup()
        mask = self.state["bg"].apply(img, learningRate=self.v("learn"))
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]   # drop shadows (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.bitwise_and(img, img, mask=mask)
        return {"mask": mask, "foreground": fg, "ratio": round(100.0 * cv2.countNonZero(mask) / mask.size, 2)}


# --- charts --------------------------------------------------------------------------------------

@register
class Chart(Node):
    title = "Grafik"
    category = "5 · Çıkış ve Otomasyon"
    description = "Bir veya iki sayının zaman içindeki değişimini çizgi grafik olarak çizer (önizleme, web panel)."
    inputs = [("a", NUM), ("b", NUM), ("reset", BOOL)]
    outputs = [("image", IMG)]
    params = [Param("points", "int", 200, "Gösterilecek nokta", 10, 5000),
              Param("width", "int", 480, "Genişlik", 120, 2000), Param("height", "int", 220, "Yükseklik", 80, 1500),
              Param("title", "str", "", "Başlık"),
              Param("names", "str", "a,b", "Seri adları")]

    def setup(self):
        self.state.update(a=[], b=[])

    def process(self, inputs):
        st, n = self.state, self.v("points")
        if inputs["reset"]:
            st["a"].clear()
            st["b"].clear()
        for k in ("a", "b"):
            if inputs[k] is not None:
                st[k].append(float(inputs[k]))
                del st[k][:-n]
        w, h = self.v("width"), self.v("height")
        img = np.full((h, w, 3), 30, np.uint8)
        pad_l, pad_r, pad_t, pad_b = 46, 10, 26, 18
        series = [(st["a"], (247, 195, 79)), (st["b"], (129, 213, 174))]
        vals = [v for s, _ in series for v in s]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        if hi - lo < 1e-9:
            lo, hi = lo - 1, hi + 1
        for i in range(5):   # grid + y labels
            y = int(pad_t + (h - pad_t - pad_b) * i / 4)
            cv2.line(img, (pad_l, y), (w - pad_r, y), (55, 55, 55), 1)
            cv2.putText(img, f"{hi - (hi - lo) * i / 4:.4g}", (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (150, 150, 150), 1, cv2.LINE_AA)
        names = [s.strip() for s in self.v("names").split(",")] + ["a", "b"]
        for si, (s, color) in enumerate(series):
            if len(s) >= 2:
                xs = np.linspace(pad_l, w - pad_r, n)[-len(s):]
                ys = pad_t + (h - pad_t - pad_b) * (1 - (np.array(s) - lo) / (hi - lo))
                cv2.polylines(img, [np.int32(np.column_stack([xs, ys]))], False, color, 2, cv2.LINE_AA)
            if s:
                put_text(img, f"{names[si]}: {s[-1]:.4g}", (pad_l + si * 150, 16), 0.45, color, 1)
        if self.v("title"):
            put_text(img, self.v("title"), (w - 10 - text_size(self.v("title"), 0.45)[0], 16), 0.45,
                     (220, 220, 220), 1)
        return {"image": img}


# --- database --------------------------------------------------------------------------------------

@register
class SQLiteLogger(Node):
    setup_params = ("path", "table", "names")   # read only when the block starts: changing them restarts it
    title = "Veritabanı Kaydı (SQLite)"
    category = "5 · Çıkış ve Otomasyon"
    description = ("Değerleri zaman damgasıyla SQLite veritabanına yazar (tek dosya, kurulum gerekmez; Excel, "
                   "Power BI, DB Browser ile açılır). Tetik bağlı değilse değer değişince yazar.")
    inputs = [("a", PortType.ANY), ("b", PortType.ANY), ("c", PortType.ANY), ("trigger", BOOL)]
    outputs = [("rows", NUM)]
    params = [Param("path", "str", "kayitlar.db", "Veritabanı dosyası"),
              Param("table", "str", "olaylar", "Tablo"),
              Param("names", "str", "a,b,c", "Sütun adları")]

    def _cols(self) -> list[str]:
        names = ([n.strip() for n in self.v("names").split(",")] + ["", "", ""])[:3]
        clean, used = [], {"id", "zaman"}          # reserved by the table itself
        for i, n in enumerate(names):
            c = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in n) or "abc"[i]
            base, k = c, 2
            while c.lower() in used:               # "x,x,y" → x, x_2, y
                c, k = f"{base}_{k}", k + 1
            used.add(c.lower())
            clean.append(c)
        return clean

    def setup(self):
        table = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in self.v("table")) or "olaylar"
        con = sqlite3.connect(self.v("path"), check_same_thread=False)
        cols = self._cols()
        con.execute(f'CREATE TABLE IF NOT EXISTS "{table}" (id INTEGER PRIMARY KEY AUTOINCREMENT, '
                    f'zaman TEXT, "{cols[0]}", "{cols[1]}", "{cols[2]}")')
        # an existing table made with other column names gets the new columns added (old rows are kept)
        have = {r[1].lower() for r in con.execute(f'PRAGMA table_info("{table}")')}
        for c in cols:
            if c.lower() not in have:
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}"')
        con.commit()
        self.state.update(con=con, table=table, last=object(), rows=0, cols=cols)

    @staticmethod
    def _plain(v):
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, np.generic):
            return v.item()
        if v is None or isinstance(v, (int, float, str)):
            return v
        if isinstance(v, list):
            return json.dumps([d.get("label", str(d)) if isinstance(d, dict) else str(d) for d in v], ensure_ascii=False)
        return str(v)

    def process(self, inputs):
        vals = tuple(self._plain(inputs[k]) for k in ("a", "b", "c"))
        if inputs["trigger"] is not None:
            fire = _rising(self, inputs["trigger"])
        else:
            fire = vals != self.state["last"]
        if fire:
            st = self.state
            c = st["cols"]
            st["con"].execute(f'INSERT INTO "{st["table"]}" (zaman, "{c[0]}", "{c[1]}", "{c[2]}") VALUES (?, ?, ?, ?)',
                              (dt.datetime.now().isoformat(timespec="milliseconds"), *vals))
            st["con"].commit()
            st["last"], st["rows"] = vals, st["rows"] + 1
        return {"rows": self.state["rows"]}

    def teardown(self):
        if self.state.get("con"):
            self.state["con"].close()


# --- notifications -----------------------------------------------------------------------------------

def _jpeg(img, max_w=1280) -> bytes:
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


class _Notifier(Node):
    """Shared logic: send on a rising trigger, at most once per cooldown, in a background thread."""

    def setup(self):
        self.state.update(last_sent=0.0, sent=0)

    def _should_send(self, trigger) -> bool:
        if trigger is None or not _rising(self, trigger):
            return False
        if time.monotonic() - self.state["last_sent"] < self.v("cooldown"):
            return False
        self.state["last_sent"] = time.monotonic()
        return True

    def _fire(self, fn, *args):
        def run():
            try:
                fn(*args)
                self.state["sent"] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[{self.title}] gönderilemedi: {exc}")
        threading.Thread(target=run, daemon=True).start()


@register
class EmailSend(_Notifier):
    title = "E-posta Gönder"
    category = "5 · Çıkış ve Otomasyon"
    description = ("Tetik gelince e-posta gönderir, isterseniz görüntüyü ekler. Gmail için 'uygulama şifresi' "
                   "kullanın (smtp.gmail.com, 587). {v} yerine değer yazılır.")
    inputs = [("trigger", BOOL), ("image", IMG), ("value", PortType.ANY)]
    outputs = [("sent", NUM)]
    params = [Param("smtp", "str", "smtp.gmail.com", "SMTP sunucu"), Param("port", "int", 587, "Port", 1, 65535),
              Param("user", "str", "", "Kullanıcı (e-posta)"), Param("password", "str", "", "Şifre"),
              Param("to", "str", "", "Alıcı(lar), virgülle"),
              Param("subject", "str", "VisionFlow uyarısı", "Konu"),
              Param("body", "str", "Olay algılandı. Değer: {v}", "Mesaj"),
              Param("cooldown", "float", 60.0, "En az bekleme (sn)", 0, 86400)]

    def build(self, value, img) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = self.v("subject")
        msg["From"] = self.v("user")
        msg["To"] = self.v("to")
        msg.set_content(self.v("body").replace("{v}", str(value)) + f"\n\n{dt.datetime.now():%Y-%m-%d %H:%M:%S}")
        if img is not None:
            msg.add_attachment(_jpeg(img), maintype="image", subtype="jpeg", filename="goruntu.jpg")
        return msg

    def _send(self, msg):
        port = self.v("port")
        cls = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
        with cls(self.v("smtp"), port, timeout=15) as s:
            if port != 465:
                try:
                    s.starttls()
                except smtplib.SMTPNotSupportedError:
                    if self.v("user") and self.v("smtp").strip().lower() not in ("localhost", "127.0.0.1", "::1"):
                        raise RuntimeError(f"{self.v('smtp')} şifreli bağlantı (STARTTLS) sunmuyor; şifre açık "
                                           "gönderilmez. Port 465 (SSL) veya 587 deneyin.") from None
            if self.v("user"):
                s.login(self.v("user"), self.v("password"))
            s.send_message(msg)

    def process(self, inputs):
        if self._should_send(inputs["trigger"]):
            if not self.v("to"):
                raise ValueError("Alıcı adresi boş")
            self._fire(self._send, self.build(inputs["value"], inputs["image"]))
        return {"sent": self.state["sent"]}


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (name, data, ctype) in files.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


@register
class TelegramSend(_Notifier):
    title = "Telegram Gönder"
    category = "5 · Çıkış ve Otomasyon"
    description = ("Tetik gelince Telegram'a mesaj ve fotoğraf gönderir. @BotFather'dan bot token alın; chat ID "
                   "için bota mesaj atın ('Telegram Mesaj Al' bloğu ID'yi günlüğe yazar). Token boşsa AI Merkezi "
                   "ayarındaki bot kullanılır; 'chat_id' girişi bağlıysa cevap o sohbete gider.")
    inputs = [("trigger", BOOL), ("image", IMG), ("value", PortType.ANY), ("chat_id", TEXT)]
    outputs = [("sent", NUM)]
    params = [Param("token", "str", "", "Bot token (boş = AI Merkezi)"), Param("chat_id", "str", "", "Chat ID"),
              Param("text", "str", "⚠️ Olay algılandı: {v}", "Mesaj"),
              Param("cooldown", "float", 10.0, "En az bekleme (sn)", 0, 86400),
              Param("api", "str", "https://api.telegram.org", "API adresi")]

    def _token(self) -> str:
        if self.v("token"):
            return self.v("token")
        from ..ai.config import AIConfig
        return AIConfig.load().telegram_token

    def request(self, value, img, chat_id: str | None = None) -> urllib.request.Request:
        chat = chat_id or self.v("chat_id")
        text = self.v("text").replace("{v}", str(value))
        base = f"{self.v('api').rstrip('/')}/bot{self._token()}"
        if img is not None:
            cap = text if len(text) <= 1024 else text[:1020] + "…"     # Telegram caption limit
            body, ctype = _multipart({"chat_id": chat, "caption": cap},
                                     {"photo": ("goruntu.jpg", _jpeg(img), "image/jpeg")})
            return urllib.request.Request(f"{base}/sendPhoto", body, {"Content-Type": ctype})
        msg = text if len(text) <= 4096 else text[:4090] + "…"         # Telegram message limit
        body = json.dumps({"chat_id": chat, "text": msg}).encode()
        return urllib.request.Request(f"{base}/sendMessage", body, {"Content-Type": "application/json"})

    def process(self, inputs):
        if self._should_send(inputs["trigger"]):
            chat = str(inputs.get("chat_id") or "") or self.v("chat_id")
            if not self._token() or not chat:
                raise ValueError("Bot token ve chat ID gerekli")
            req = self.request(inputs["value"], inputs["image"], chat)
            self._fire(lambda r: urllib.request.urlopen(r, timeout=15).close(), req)
        return {"sent": self.state["sent"]}
