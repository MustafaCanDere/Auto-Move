"""Blocks for the production line: tolerance check with yield, crop a detection, smooth a value, production rate,
shift report."""
from __future__ import annotations

import collections
import datetime as dt
import math
import statistics
import time

import numpy as np

from ..core import Node, Param, PortType, register
from .ai_hub import Scheduler
from .outputs import _rising

IMG, NUM, BOOL, TEXT, DET, ANY = (PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.TEXT, PortType.DETECTIONS,
                                  PortType.ANY)
QC, LOGIC = "2b · Kalite Kontrol ve Ölçüm", "4 · Mantık"


def _num(v):
    """A usable number or None (None, text, NaN and inf are not measurements)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@register
class ToleranceCheck(Node):
    title = "Tolerans Kontrolü"
    category = QC
    description = ("Ölçülen değer alt-üst sınır arasında mı bakar; sağlam/hatalı sayar ve verim yüzdesini verir. "
                   "'trigger' bağlıysa her yükselişte bir parça sayılır, bağlı değilse değer değiştikçe. Hatalı "
                   "çıkışını ayıklama silindirine (PLC Yaz) veya alarma bağlayın.")
    inputs = [("value", NUM), ("trigger", BOOL), ("reset", BOOL)]
    outputs = [("ok", BOOL), ("nok", BOOL), ("ok_count", NUM), ("nok_count", NUM), ("yield", NUM),
               ("deviation", NUM), ("text", TEXT)]
    params = [Param("low", "float", 9.9, "Alt sınır", -1e9, 1e9, 0.01),
              Param("high", "float", 10.1, "Üst sınır", -1e9, 1e9, 0.01),
              Param("nominal", "float", 10.0, "Nominal (sapma için)", -1e9, 1e9, 0.01),
              Param("missing", "choice", "hatalı say", "Ölçüm yoksa", choices=["hatalı say", "sayma"])]

    def setup(self):
        self.state.update(ok_count=0, nok_count=0, last=None, prev_reset=False)

    def process(self, inputs):
        st = self.state
        reset = bool(inputs["reset"])
        if reset and not st["prev_reset"]:
            st.update(ok_count=0, nok_count=0, last=None)
        st["prev_reset"] = reset
        lo, hi = sorted((self.v("low"), self.v("high")))
        v = _num(inputs["value"])
        ok = v is not None and lo <= v <= hi
        if inputs["trigger"] is not None:
            count = _rising(self, inputs["trigger"])
        else:
            # without a trigger a new part = a new valid measurement (NaN != NaN must not count every frame)
            count = v is not None and v != st["last"]
            if v is not None:
                st["last"] = v
        if count and (v is not None or self.v("missing") == "hatalı say"):
            st["ok_count" if ok else "nok_count"] += 1
        total = st["ok_count"] + st["nok_count"]
        dev = round(v - self.v("nominal"), 6) if v is not None else 0.0
        if v is None:
            text = "ölçüm yok"
        else:
            text = "SAĞLAM" if ok else ("ALT SINIR ALTINDA" if v < lo else "ÜST SINIR ÜSTÜNDE")
        return {"ok": ok, "nok": not ok and (v is not None or self.v("missing") == "hatalı say"),
                "ok_count": st["ok_count"], "nok_count": st["nok_count"],
                "yield": round(100.0 * st["ok_count"] / total, 2) if total else 100.0,
                "deviation": dev, "text": text}


@register
class CropDetection(Node):
    title = "Tespiti Kırp"
    category = QC
    description = ("Bulunan nesnelerden birini (en emin olunan, en büyük veya sıradaki) resimden keser. Kesilen "
                   "parçayı Yazı Oku (OCR), Görüntü Sınıflandır veya Renk Tanı bloğuna verin: ör. plakayı bul → "
                   "kırp → oku.")
    inputs = [("image", IMG), ("detections", DET)]
    outputs = [("image", IMG), ("label", TEXT), ("score", NUM), ("found", BOOL), ("count", NUM)]
    params = [Param("pick", "choice", "en yüksek güven", "Hangisi",
                    choices=["en yüksek güven", "en büyük", "sıra no"]),
              Param("index", "int", 0, "Sıra no (0 = ilk)", 0, 999),
              Param("label", "str", "", "Sadece bu sınıf (boş = hepsi)"),
              Param("margin", "int", 0, "Kenar payı (piksel)", 0, 500)]

    def process(self, inputs):
        img, dets = inputs["image"], inputs["detections"] or []
        want = self.v("label").strip().casefold()
        if want:
            dets = [d for d in dets if str(d.get("label", "")).split(" #")[0].strip().casefold() == want]
        empty = {"image": None, "label": "", "score": 0.0, "found": False, "count": len(dets)}
        if img is None or not dets:
            return empty
        pick = self.v("pick")
        if pick == "en büyük":
            d = max(dets, key=lambda d: d["box"][2] * d["box"][3])
        elif pick == "sıra no":
            if self.v("index") >= len(dets):
                return empty
            d = dets[self.v("index")]
        else:
            d = max(dets, key=lambda d: d.get("score", 0))
        x, y, w, h = (int(round(v)) for v in d["box"])
        m = self.v("margin")
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
        if x1 <= x0 or y1 <= y0:                     # box entirely outside the picture
            return empty
        return {"image": np.ascontiguousarray(img[y0:y1, x0:x1]), "label": str(d.get("label", "")),
                "score": float(d.get("score", 0.0)), "found": True, "count": len(dets)}


@register
class SmoothValue(Node):
    title = "Değeri Yumuşat"
    category = LOGIC
    description = ("Titreyen ölçümü sakinleştirir: son N değerin ortalaması veya medyanı. Medyan tek tük hatalı "
                   "ölçümleri (ör. bir karede yanlış okunan çap) tamamen yok sayar. Boş ve geçersiz değerler atlanır.")
    inputs = [("value", NUM), ("reset", BOOL)]
    outputs = [("value", NUM), ("min", NUM), ("max", NUM), ("ready", BOOL)]
    params = [Param("mode", "choice", "medyan", "Yöntem", choices=["medyan", "ortalama"]),
              Param("window", "int", 5, "Kaç değer", 1, 1000)]

    def setup(self):
        self.state["buf"] = collections.deque(maxlen=self.v("window"))

    def process(self, inputs):
        buf = self.state["buf"]
        if buf.maxlen != self.v("window"):           # window changed while running: keep the newest values
            buf = self.state["buf"] = collections.deque(buf, maxlen=self.v("window"))
        if inputs["reset"]:
            buf.clear()
        v = _num(inputs["value"])
        if v is not None:
            buf.append(v)
        if not buf:
            return {"value": 0.0, "min": 0.0, "max": 0.0, "ready": False}
        out = statistics.median(buf) if self.v("mode") == "medyan" else sum(buf) / len(buf)
        return {"value": round(out, 6), "min": min(buf), "max": max(buf), "ready": len(buf) == buf.maxlen}


@register
class ProductionRate(Node):
    title = "Üretim Hızı"
    category = QC
    description = ("Bir sayacın (Sayaç, Çizgi Geçiş Sayacı, Tolerans Kontrolü) ne kadar hızlı arttığını ölçer: "
                   "dakikada ve saatte kaç parça. Hız hedefin altına düşünce 'slow' doğru olur (hat durdu mu?).")
    inputs = [("count", NUM)]
    outputs = [("per_minute", NUM), ("per_hour", NUM), ("slow", BOOL), ("idle_seconds", NUM)]
    params = [Param("window", "int", 60, "Ölçüm penceresi (sn)", 5, 3600),
              Param("target", "float", 0.0, "Hedef (adet/dk, 0 = yok)", 0, 1e6, 0.1)]

    now = staticmethod(time.monotonic)               # tests replace the clock

    def setup(self):
        self.state.update(hist=collections.deque(), last=None, last_change=self.now(), start=self.now())

    def process(self, inputs):
        st, t = self.state, self.now()
        c = _num(inputs["count"])
        if c is not None:
            if st["last"] is not None and c < st["last"]:      # counter was reset: start measuring again
                st["hist"].clear()
            if c != st["last"]:
                st["last_change"] = t
            st["last"] = c
            st["hist"].append((t, c))
        win = self.v("window")
        hist = st["hist"]
        while len(hist) > 2 and t - hist[1][0] >= win:          # keep one sample older than the window
            hist.popleft()
        if len(hist) >= 2 and t - hist[0][0] > 0:
            per_min = max(0.0, (hist[-1][1] - hist[0][1]) / (t - hist[0][0]) * 60.0)
        else:
            per_min = 0.0
        warm = t - st["start"] >= win                           # no "slow" alarm before one full window
        target = self.v("target")
        return {"per_minute": round(per_min, 2), "per_hour": round(per_min * 60.0, 1),
                "slow": bool(target and warm and per_min < target),
                "idle_seconds": round(t - st["last_change"], 1)}


@register
class ShiftReport(Node):
    setup_params = ("times",)
    title = "Vardiya Raporu"
    category = QC
    description = ("Vardiya sonlarında (ör. 08:00, 16:00, 00:00) o vardiyada üretilen sağlam/hatalı parça sayısını, "
                   "verimi ve isterseniz ek bir değeri metin olarak verir; 'report' o an bir kez doğru olur. Metni "
                   "Telegram Gönder, E-posta Gönder veya Metni Dosyaya Kaydet bloğuna bağlayın. Girişlere Tolerans "
                   "Kontrolü'nün ok_count / nok_count çıkışlarını verin; sayaç sıfırlansa da doğru sayar. 'now' "
                   "girişi elle rapor almak içindir.")
    inputs = [("ok", NUM), ("nok", NUM), ("extra", ANY), ("now", BOOL)]
    outputs = [("text", TEXT), ("report", BOOL), ("shift_ok", NUM), ("shift_nok", NUM), ("yield", NUM)]
    params = [Param("times", "str", "08:00, 16:00, 00:00", "Vardiya bitişleri (virgülle)"),
              Param("title", "str", "Vardiya raporu", "Başlık"),
              Param("extra_name", "str", "", "Ek değerin adı (ör. Ortalama çap)")]
    is_source = True                     # keeps a flow without a camera running so the report time is noticed

    _times = Scheduler._times            # same "08:00, 16:00" parsing and errors

    def now(self) -> dt.datetime:
        return dt.datetime.now()

    def setup(self):
        now = self.now()
        times = self._times()
        self.state.update(fired={(h, m): now.date() for h, m in times if (h, m) < (now.hour, now.minute)},
                          start=now, last={"ok": None, "nok": None}, acc={"ok": 0.0, "nok": 0.0},
                          base={"ok": 0.0, "nok": 0.0}, prev_now=False, text="")

    def _count(self, key, value):
        """Parts made in this shift from a running counter. The counter's value when the flow starts is not this
        shift's work, and a counter that is reset to 0 mid-shift keeps adding up."""
        st, v = self.state, _num(value)
        last = st["last"][key]
        if v is not None:
            if last is not None:
                st["acc"][key] += v - last if v >= last else v
            st["last"][key] = v
        return st["acc"][key] - st["base"][key]

    def process(self, inputs):
        st, now = self.state, self.now()
        ok, nok = self._count("ok", inputs["ok"]), self._count("nok", inputs["nok"])
        due = False
        for h, m in self._times():
            if (now.hour, now.minute) >= (h, m) and st["fired"].get((h, m)) != now.date():
                st["fired"][(h, m)] = now.date()
                due = True
        manual = bool(inputs["now"]) and not st["prev_now"]
        st["prev_now"] = bool(inputs["now"])
        total = ok + nok
        yld = round(100.0 * ok / total, 1) if total else 100.0
        report = due or manual
        if report:
            lines = [f"{self.v('title')} · {st['start']:%d.%m.%Y %H:%M} – {now:%H:%M}",
                     f"Sağlam: {int(ok)}", f"Hatalı: {int(nok)}", f"Toplam: {int(total)}", f"Verim: %{yld:g}"]
            if inputs["extra"] is not None:
                ex = inputs["extra"]
                ex = round(ex, 3) if isinstance(ex, float) else ex
                lines.append(f"{self.v('extra_name') or 'Ek değer'}: {ex}")
            st["text"] = "\n".join(lines)
            if due:                                  # a new shift starts; a manual report does not reset
                st.update(start=now, base=dict(st["acc"]))
        return {"text": st["text"], "report": report, "shift_ok": int(ok), "shift_nok": int(nok), "yield": yld}
