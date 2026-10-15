"""Factory-floor logic blocks: reject gate delay (shift register), SPC chart with Western Electric rules, OEE, lot
tracking, alarm / downtime history, multi-camera verdict and a signal simulator to try them without a conveyor.

Every block reads time through a ``now`` method so tests can replace the clock."""
from __future__ import annotations

import collections
import csv
import datetime as dt
import json
import math
import os
import random
import re
import time

from ..ai.config import home
from ..core import Node, Param, PortType, register

NUM, BOOL, TEXT = PortType.NUMBER, PortType.BOOL, PortType.TEXT
QC, LOGIC, SRC = "2b · Kalite Kontrol ve Ölçüm", "4 · Mantık", "1 · Giriş"


def _num(v):
    """A usable number or None (None, bools, text, NaN and inf are not measurements)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _edge(node: Node, key: str, value) -> bool:
    """True once per False→True change of ``value`` (state kept under ``key``)."""
    now = bool(value)
    fired = now and not node.state.get(key, False)
    node.state[key] = now
    return fired


def _times(text: str) -> list[tuple[int, int]]:
    """'08:00, 16:00' → [(8, 0), (16, 0)]; empty text → []."""
    out = []
    for t in str(text or "").split(","):
        if not t.strip():
            continue
        m = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})\s*", t)
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
            raise ValueError(f"Saat anlaşılmadı: '{t.strip()}' (ör. 08:00)")
        out.append((int(m.group(1)), int(m.group(2))))
    return sorted(set(out))


def _last_boundary(now: dt.datetime, times: list[tuple[int, int]]) -> dt.datetime | None:
    """The latest of the given clock times that is not after ``now`` (today or yesterday)."""
    best = None
    for days in (0, 1):
        d = now.date() - dt.timedelta(days=days)
        for h, m in times:
            b = dt.datetime.combine(d, dt.time(h, m))
            if b <= now and (best is None or b > best):
                best = b
        if best is not None:
            return best
    return best


def _safe_name(text: str, fallback: str) -> str:
    return re.sub(r"[^\w-]+", "_", str(text or "")).strip("_") or fallback


# --------------------------------------------------------------------------------------------------------------------
@register
class RejectDelay(Node):
    setup_params = ("mode",)                         # a queue in one unit cannot be read in another
    title = "Ret Kapısı Gecikmesi (Kaydırma Yazmacı)"
    category = LOGIC
    description = ("Kamera hatalı parçayı görür ama ret üfleyicisi/iticisi bantta daha ileridedir. Bu blok her hatalı "
                   "parçayı sıraya (FIFO) koyar ve parça üfleyiciye vardığında 'reject' çıkışını ayarlanan süre "
                   "kadar doğru yapar. Mesafe üç şekilde verilebilir: parça sayısı (N parça sonra), enkoder darbesi "
                   "(N darbe sonra) veya süre (N ms sonra). Parçalar sık gelse de her biri ayrı izlenir. 'bad' "
                   "değeri 'trigger' yükseldiği karede okunur. 'pending' bekleyen ret sayısıdır.")
    inputs = [("trigger", BOOL), ("bad", BOOL), ("encoder", NUM), ("reset", BOOL)]
    outputs = [("reject", BOOL), ("pending", NUM), ("rejected", NUM), ("parts", NUM)]
    MODES = ["parça sayısı", "enkoder", "süre (ms)"]
    params = [Param("mode", "choice", MODES[0], "Mesafe türü", choices=MODES),
              Param("distance", "float", 3, "Mesafe (parça / darbe / ms)", 0, 1e9, 1),
              Param("pulse_ms", "float", 100, "Ret darbesi süresi (ms)", 0, 60000, 10),
              Param("encoder_max", "float", 0, "Enkoder taşma değeri (0 = taşmaz)", 0, 1e12, 1)]

    now = staticmethod(time.monotonic)               # tests replace the clock
    MAX_PENDING = 100000

    def setup(self):
        self.state.update(queue=collections.deque(), parts=0, rejected=0, until=-math.inf, pos=None, last_enc=None)

    def _position(self, enc) -> float | None:
        """Unwrapped encoder position: a counter that rolls over at 'encoder_max' (or is reset to a lower value)
        keeps adding up, so targets set before the roll-over are still reached."""
        st, e = self.state, _num(enc)
        if e is None:
            return st["pos"]
        last = st["last_enc"]
        if last is None:
            st["pos"] = e
        else:
            if e >= last:
                delta = e - last
            else:
                mx = self.v("encoder_max")
                delta = (mx - last) + e if mx > 0 else e
            st["pos"] += delta
        st["last_enc"] = e
        return st["pos"]

    def process(self, inputs):
        st, mode, now = self.state, self.v("mode"), self.now()
        q = st["queue"]
        if inputs["reset"]:
            q.clear()
            st.update(until=-math.inf)
        if inputs["trigger"] is None:
            raise ValueError("'trigger' (parça tetiği) girişi bağlı değil")
        pos = None
        if mode == "enkoder":
            if inputs["encoder"] is None:
                raise ValueError("Enkoder modunda 'encoder' girişi bağlanmalı")
            pos = self._position(inputs["encoder"])
            if pos is None:
                raise ValueError(f"Enkoder değeri sayı değil: {inputs['encoder']!r}")
        dist = self.v("distance")
        if _edge(self, "prev_trigger", inputs["trigger"]):
            st["parts"] += 1
            if inputs["bad"]:
                if len(q) >= self.MAX_PENDING:
                    raise ValueError("Çok fazla bekleyen ret var; mesafe ayarını kontrol edin")
                if mode == "parça sayısı":
                    q.append(st["parts"] + int(round(dist)))
                elif mode == "enkoder":
                    q.append(pos + dist)
                else:
                    q.append(now + dist / 1000.0)
        at = {"parça sayısı": st["parts"], "enkoder": pos, "süre (ms)": now}[mode]
        fired = 0
        while q and q[0] <= at + 1e-9:
            q.popleft()
            fired += 1
        if fired:
            st["rejected"] += fired
            st["until"] = max(st["until"], now + self.v("pulse_ms") / 1000.0)
        reject = bool(fired) or now < st["until"] - 1e-9
        return {"reject": reject, "pending": len(q), "rejected": st["rejected"], "parts": st["parts"]}


# --------------------------------------------------------------------------------------------------------------------
# d2 constants: expected range of n normal samples in sigmas (n = 2…25)
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173,
      12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472, 16: 3.532, 17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735, 21: 3.778,
      22: 3.819, 23: 3.858, 24: 3.895, 25: 3.931}


def spc_stats(points: list[tuple[float, float]], n: int) -> tuple[float, float] | None:
    """(grand mean, within-subgroup sigma) of X-bar points [(mean, range)]: sigma = R̄/d2(n); for n = 1
    (individuals) sigma = moving range average / 1.128. None when there is not enough data."""
    if n == 1:
        if len(points) < 2:
            return None
        xs = [p[0] for p in points]
        mr = [abs(b - a) for a, b in zip(xs, xs[1:])]
        return sum(xs) / len(xs), (sum(mr) / len(mr)) / D2[2]
    if not points:
        return None
    return sum(p[0] for p in points) / len(points), (sum(p[1] for p in points) / len(points)) / D2[n]


@register
class SPCChart(Node):
    title = "SPC (İstatistiksel Süreç Kontrol)"
    category = QC
    description = ("Ölçümleri alt gruplara (n adet) ayırıp X-bar kontrol kartı tutar: ortalama, sigma, Cp, Cpk, "
                   "UCL/LCL. Western Electric kurallarıyla alarm verir: 1) nokta 3σ dışında, 2) 9 nokta ortalamanın "
                   "aynı tarafında, 3) 6 nokta sürekli artıyor/azalıyor (isteğe bağlı 4: 3'te 2 nokta 2σ dışında, "
                   "5: 5'te 4 nokta 1σ dışında). 'Öğrenme' > 0 ise ilk N nokta kontrol sınırlarını belirler ve sınırlar "
                   "sabitlenir; 0 ise sınırlar son noktalardan (kayan pencere) hesaplanır. 'trigger' bağlıysa her "
                   "yükselişte bir ölçüm alınır, bağlı değilse her karede. Alt/üst tolerans ikisi de 0 ise Cp/Cpk "
                   "hesaplanmaz.")
    inputs = [("value", NUM), ("trigger", BOOL), ("reset", BOOL)]
    outputs = [("alarm", BOOL), ("alarm_text", TEXT), ("point", NUM), ("center", NUM), ("ucl", NUM), ("lcl", NUM),
               ("mean", NUM), ("sigma", NUM), ("cp", NUM), ("cpk", NUM), ("learning", BOOL), ("points", NUM)]
    params = [Param("lsl", "float", 0.0, "Alt tolerans (LSL)", -1e12, 1e12, 0.01),
              Param("usl", "float", 0.0, "Üst tolerans (USL)", -1e12, 1e12, 0.01),
              Param("subgroup", "int", 5, "Alt grup büyüklüğü (n)", 1, 25),
              Param("baseline", "int", 25, "Öğrenme: ilk N nokta sınırları belirler (0 = kayan)", 0, 10000),
              Param("window", "int", 100, "Ortalama/sigma/Cpk penceresi (nokta)", 2, 100000),
              Param("extra_rules", "bool", False, "Kural 4 ve 5'i de kullan")]

    ROLLING_MIN = 5                     # rolling limits need this many earlier points before alarms are given
    setup_params = ("subgroup", "baseline", "window")

    def setup(self):
        if self.v("subgroup") == 1 and self.v("baseline") == 1:
            raise ValueError("Alt grup 1 iken öğrenme için en az 2 nokta gerekli")
        self.state.update(buf=[], points=collections.deque(maxlen=max(self.v("window"), 15)), base=[], limits=None,
                          out=None)

    def _reset(self):
        self.setup()

    @staticmethod
    def _limits(stats, n):
        cl, sigma = stats
        sx = sigma / math.sqrt(n) if n > 1 else sigma
        return cl, sx

    def _rules(self, xs: list[float], cl: float, sx: float) -> list[str]:
        out = []
        x = xs[-1]
        if abs(x - cl) > 3 * sx + 1e-12:
            side = f"> UCL {cl + 3 * sx:.4g}" if x > cl else f"< LCL {cl - 3 * sx:.4g}"
            out.append(f"Kural 1: nokta 3σ dışında ({x:.4g} {side})")
        last9 = xs[-9:]
        if len(last9) == 9 and (all(v > cl for v in last9) or all(v < cl for v in last9)):
            out.append(f"Kural 2: 9 nokta ortalamanın {'üstünde' if x > cl else 'altında'}")
        last6 = xs[-6:]
        if len(last6) == 6:
            steps = [b - a for a, b in zip(last6, last6[1:])]
            if all(s > 0 for s in steps):
                out.append("Kural 3: 6 nokta sürekli artıyor")
            elif all(s < 0 for s in steps):
                out.append("Kural 3: 6 nokta sürekli azalıyor")
        if self.v("extra_rules") and sx > 0:
            for k, of, lim, rule in ((2, 3, 2, 4), (4, 5, 1, 5)):
                tail = xs[-of:]
                if len(tail) == of and (sum(v > cl + lim * sx for v in tail) >= k
                                        or sum(v < cl - lim * sx for v in tail) >= k):
                    out.append(f"Kural {rule}: {of} noktanın {k}'{'si' if k == 2 else 'ü'} {lim}σ dışında")
        return out

    def _point(self, mean: float, rng: float):
        st, n = self.state, self.v("subgroup")
        pts = st["points"]
        base_n = self.v("baseline")
        limits, alarms = None, []
        if base_n > 0:
            if st["limits"] is None:
                st["base"].append((mean, rng))
                pts.append((mean, rng))
                if len(st["base"]) >= base_n:
                    stats = spc_stats(st["base"], n)
                    if stats is None:
                        raise ValueError("Öğrenme için en az 2 nokta gerekli")
                    st["limits"] = self._limits(stats, n)
                    st["base"] = []
                limits = st["limits"]
            else:
                pts.append((mean, rng))
                limits = st["limits"]
                alarms = self._rules([p[0] for p in pts], *limits)
        else:
            prev = list(pts)[-self.v("window"):]
            if len(prev) >= self.ROLLING_MIN:
                stats = spc_stats(prev, n)
                limits = self._limits(stats, n)
            pts.append((mean, rng))
            if limits is not None:
                alarms = self._rules([p[0] for p in pts], *limits)
        recent = list(pts)[-self.v("window"):]
        stats = spc_stats(recent, n)
        mu, sigma = stats if stats else (mean, None)
        lsl, usl = self.v("lsl"), self.v("usl")
        cp = cpk = None
        if usl > lsl and sigma:
            cp = (usl - lsl) / (6 * sigma)
            cpk = min(usl - mu, mu - lsl) / (3 * sigma)
        cl, sx = limits if limits else (None, None)
        st["out"] = {"alarm": bool(alarms), "alarm_text": "; ".join(alarms), "point": mean, "center": cl,
                     "ucl": None if cl is None else cl + 3 * sx, "lcl": None if cl is None else cl - 3 * sx,
                     "mean": mu, "sigma": sigma, "cp": cp, "cpk": cpk}

    def process(self, inputs):
        st, n = self.state, self.v("subgroup")
        if self.v("lsl") > self.v("usl"):
            raise ValueError("Alt tolerans (LSL) üst toleranstan (USL) büyük olamaz")
        take = inputs["trigger"] is None or _edge(self, "prev_trigger", inputs["trigger"])
        if inputs["reset"]:
            self._reset()
            take = False                                 # the reset frame's value belongs to nothing
        v = _num(inputs["value"])
        if take and v is not None:
            st["buf"].append(v)
            if len(st["buf"]) >= n:
                buf, st["buf"] = st["buf"], []
                self._point(sum(buf) / len(buf), max(buf) - min(buf))
        out = dict(st["out"] or {"alarm": False, "alarm_text": "", "point": None, "center": None, "ucl": None,
                                 "lcl": None, "mean": None, "sigma": None, "cp": None, "cpk": None})
        for k in ("point", "center", "ucl", "lcl", "mean", "sigma", "cp", "cpk"):
            if out[k] is not None:
                out[k] = round(out[k], 6)
        out["learning"] = self.v("baseline") > 0 and st["limits"] is None
        out["points"] = len(st["points"])
        return out


# --------------------------------------------------------------------------------------------------------------------
@register
class OEECalc(Node):
    title = "OEE Hesapla"
    category = QC
    description = ("Ekipman etkinliği: Kullanılabilirlik (çalışma süresi / planlanan süre) × Performans (ideal çevrim "
                   "süresi × parça / çalışma süresi) × Kalite (sağlam / toplam). 'running' makine çalışıyor mu, "
                   "'good'/'bad' sağlam ve hatalı parça (sayaç değeri ya da her yükselişte bir parça). Planlanan "
                   "süre akışın başladığı ya da son vardiya başlangıcından bu yana geçen süredir; vardiya saatlerinde "
                   "her şey sıfırlanır. Performans en fazla %100 gösterilir.")
    inputs = [("running", BOOL), ("good", NUM), ("bad", NUM), ("reset", BOOL)]
    outputs = [("oee", NUM), ("availability", NUM), ("performance", NUM), ("quality", NUM), ("downtime_min", NUM),
               ("run_min", NUM), ("good_count", NUM), ("bad_count", NUM)]
    COUNT_MODES = ["sayaç değeri", "darbe (yükselen kenar)"]
    params = [Param("ideal_cycle", "float", 1.0, "İdeal çevrim süresi (sn/parça)", 0.001, 1e6, 0.1),
              Param("count_mode", "choice", COUNT_MODES[0], "Parça girişleri", choices=COUNT_MODES),
              Param("shift_times", "str", "", "Vardiya başlangıçları (ör. 08:00, 16:00; boş = yok)")]

    def now(self) -> dt.datetime:
        return dt.datetime.now()

    def setup(self):
        _times(self.v("shift_times"))                       # bad text: error when the flow starts
        self._start(self.now())

    def _start(self, t: dt.datetime):
        last = self.state.get("last") or {"good": None, "bad": None}
        self.state.update(start=t, last_t=t, run=0.0, good=0, bad=0, last=last, prev_running=None)

    def _count(self, key, value) -> None:
        st = self.state
        if self.v("count_mode") == self.COUNT_MODES[1] or isinstance(value, bool):   # a bool can only be pulses
            if _edge(self, "edge_" + key, value):
                st[key] += 1
            return
        v = _num(value)
        if v is None:
            return
        last = st["last"][key]
        if last is not None:
            st[key] += int(round(v - last if v >= last else v))   # a counter reset to 0 keeps adding up
        st["last"][key] = v

    def process(self, inputs):
        st, now = self.state, self.now()
        if inputs["reset"]:
            self._start(now)
        times = _times(self.v("shift_times"))
        if times:
            b = _last_boundary(now, times)
            if b is not None and b > st["start"]:
                # time before the shift start belongs to the old shift
                was = st["prev_running"]
                self._start(b)
                st["prev_running"] = was
        span = max(0.0, (now - st["last_t"]).total_seconds())
        if st["prev_running"]:
            st["run"] += span                                # the state of the last frame held until now
        st["last_t"] = now
        st["prev_running"] = bool(inputs["running"])
        self._count("good", inputs["good"])
        self._count("bad", inputs["bad"])
        planned = max(0.0, (now - st["start"]).total_seconds())
        run, good, bad = st["run"], st["good"], st["bad"]
        total = good + bad
        a = run / planned if planned > 0 else 0.0
        p = min(1.0, self.v("ideal_cycle") * total / run) if run > 0 else 0.0
        q = good / total if total else 1.0
        return {"oee": round(100 * a * p * q, 2), "availability": round(100 * a, 2), "performance": round(100 * p, 2),
                "quality": round(100 * q, 2), "downtime_min": round((planned - run) / 60.0, 2),
                "run_min": round(run / 60.0, 2), "good_count": good, "bad_count": bad}


# --------------------------------------------------------------------------------------------------------------------
@register
class LotTracker(Node):
    setup_params = ("name", "persist")
    title = "Parti / Lot Takibi"
    category = QC
    description = ("Parçaları partilere (lot) ayırır. 'lot' girişine barkod/metin verilirse o lot numarası kullanılır, "
                   "boşsa otomatik numara üretilir (ön ek-YYYYAAGG-sıra, sıra her gün 1'den başlar). Lot dolunca "
                   "'done' bir kare doğru olur; sonraki parça yeni lotu açar (elle verilen numaraya -2, -3 eklenir). "
                   "'new_lot' yükselince lot erkenden kapatılır. İsterseniz durum kaydedilir, program yeniden "
                   "açılınca kaldığı yerden devam eder.")
    inputs = [("trigger", BOOL), ("lot", TEXT), ("new_lot", BOOL)]
    outputs = [("lot_id", TEXT), ("count", NUM), ("remaining", NUM), ("done", BOOL), ("last_lot", TEXT),
               ("lots", NUM)]
    params = [Param("size", "int", 100, "Lot başına parça (0 = sınırsız)", 0, 10 ** 9),
              Param("prefix", "str", "LOT", "Otomatik numara ön eki"),
              Param("persist", "bool", True, "Durumu kaydet (kapatıp açınca devam)"),
              Param("name", "str", "lot", "Kayıt adı")]

    def now(self) -> dt.datetime:
        return dt.datetime.now()

    def _file(self):
        return home() / "lotlar" / f"{_safe_name(self.v('name'), 'lot')}.json"

    def setup(self):
        data = {}
        if self.v("persist"):
            try:
                data = json.loads(self._file().read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except (OSError, ValueError):
                data = {}
        try:
            self.state["d"] = {"lot": str(data.get("lot", "")), "count": int(data.get("count", 0)),
                               "seq": int(data.get("seq", 0)), "seq_day": str(data.get("seq_day", "")),
                               "manual": str(data.get("manual", "")), "suffix": int(data.get("suffix", 1)),
                               "full": bool(data.get("full", False)), "last_lot": str(data.get("last_lot", "")),
                               "lots": int(data.get("lots", 0))}
        except (TypeError, ValueError):
            self.state["d"] = {"lot": "", "count": 0, "seq": 0, "seq_day": "", "manual": "", "suffix": 1,
                               "full": False, "last_lot": "", "lots": 0}

    def _save(self):
        if not self.v("persist"):
            return
        f = self._file()
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state["d"], ensure_ascii=False), encoding="utf-8")
            tmp.replace(f)
        except OSError as exc:
            raise ValueError(f"Lot durumu kaydedilemedi: {exc}") from None

    def _auto_id(self) -> str:
        d, now = self.state["d"], self.now()
        day = now.strftime("%Y%m%d")
        if d["seq_day"] != day:
            d.update(seq_day=day, seq=0)
        d["seq"] += 1
        prefix = self.v("prefix").strip()
        return f"{prefix}-{day}-{d['seq']:03d}" if prefix else f"{day}-{d['seq']:03d}"

    def _open(self, manual: str):
        """Start the next lot: the manual number (with -2, -3 when it was used up) or an automatic one."""
        d = self.state["d"]
        if manual:
            if manual == d["manual"]:
                d["suffix"] += 1
                d["lot"] = f"{manual}-{d['suffix']}"
            else:
                d.update(manual=manual, suffix=1, lot=manual)
        else:
            d.update(manual="", suffix=1, lot=self._auto_id())
        d.update(count=0, full=False)

    def _close(self):
        d = self.state["d"]
        d.update(last_lot=d["lot"], lots=d["lots"] + 1, full=True)

    def process(self, inputs):
        d, changed, done = self.state["d"], False, False
        manual = inputs["lot"].strip() if isinstance(inputs["lot"], str) else ""
        if inputs["lot"] is not None and not isinstance(inputs["lot"], str):
            manual = str(inputs["lot"]).strip()
        if manual and manual != d["manual"]:
            if d["lot"] and d["count"] and not d["full"]:
                self._close()
                done = True
            self._open(manual)
            changed = True
        if _edge(self, "prev_new", inputs["new_lot"]):
            if d["lot"] and d["count"] and not d["full"]:
                self._close()
                done = True
            self._open(manual)
            changed = True
        if _edge(self, "prev_trigger", inputs["trigger"]):
            if not d["lot"] or d["full"]:
                self._open(manual)
            d["count"] += 1
            size = self.v("size")
            if size and d["count"] >= size:
                self._close()
                done = True
            changed = True
        if changed:
            self._save()
        size = self.v("size")
        return {"lot_id": d["lot"], "count": d["count"], "remaining": max(0, size - d["count"]) if size else 0,
                "done": done, "last_lot": d["last_lot"], "lots": d["lots"]}


# --------------------------------------------------------------------------------------------------------------------
@register
class AlarmHistory(Node):
    setup_params = ("path", "keep")
    title = "Duruş Sebebi Kaydı / Alarm Geçmişi"
    category = "5 · Çıkış ve Otomasyon"
    description = ("Bir alarm/duruş koşulunun her başlangıç-bitişini ve süresini sebebiyle birlikte CSV dosyasına "
                   "yazar (Excel'de açılır) ve son N kaydı bellekte tutar. Kısa titreşimleri yok saymak için "
                   "'en kısa süre': koşul bu kadar sürmezse alarm sayılmaz, bu kadar kısa kesintiler alarmı "
                   "bitirmez. Sebep metni alarm başladığında okunur.")
    inputs = [("condition", BOOL), ("text", TEXT)]
    outputs = [("active", BOOL), ("count_today", NUM), ("last_text", TEXT), ("duration", NUM), ("history", TEXT)]
    params = [Param("path", "str", "alarm_gecmisi.csv", "CSV dosyası (boş = yazma)"),
              Param("keep", "int", 50, "Bellekte tutulacak kayıt", 1, 100000),
              Param("min_seconds", "float", 0.0, "En kısa süre (sn, titreşim filtresi)", 0, 86400, 0.5)]

    RETRY_EVERY = 3.0
    HEADER = ["Başlangıç", "Bitiş", "Süre (sn)", "Sebep"]

    def now(self) -> dt.datetime:
        return dt.datetime.now()

    def setup(self):
        self.state.update(active=False, start=None, text="", cand=None, cand_since=None, history=collections.deque(
            maxlen=self.v("keep")), day=None, today=0, last_text="", last_dur=0.0, pending=[], next_try=None,
            write_error="")

    def _flush(self, now: dt.datetime):
        st, path = self.state, self.v("path").strip()
        if not path or not st["pending"]:
            st["pending"] = [] if not path else st["pending"]
            return
        if st["next_try"] is not None and now < st["next_try"]:
            return
        try:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                if new:
                    w.writerow(self.HEADER)
                w.writerows(st["pending"])
            st.update(pending=[], next_try=None, write_error="")
        except OSError as exc:                     # file open in Excel: keep the rows and try again later
            st["next_try"] = now + dt.timedelta(seconds=self.RETRY_EVERY)
            st["write_error"] = f"CSV yazılamadı, bekletiliyor: {exc}"

    def _end(self, end: dt.datetime, note: str = ""):
        st = self.state
        dur = max(0.0, (end - st["start"]).total_seconds())
        st["last_dur"] = dur
        text = st["text"] + (f" {note}" if note else "")
        st["history"].appendleft((st["start"], end, dur, text))
        st["pending"].append([f"{st['start']:%Y-%m-%d %H:%M:%S}", f"{end:%Y-%m-%d %H:%M:%S}",
                              f"{dur:.1f}".replace(".", ","), text])
        st.update(active=False, start=None)

    def process(self, inputs):
        st, now = self.state, self.now()
        if st["day"] != now.date():
            st.update(day=now.date(), today=0)
        cond = bool(inputs["condition"])
        text = inputs["text"] if isinstance(inputs["text"], str) else ("" if inputs["text"] is None
                                                                        else str(inputs["text"]))
        if cond != st["cand"]:
            st.update(cand=cond, cand_since=now)
            if cond:
                st["cand_text"] = text
        if cond and not st.get("cand_text") and text:
            st["cand_text"] = text
        held = (now - st["cand_since"]).total_seconds() >= self.v("min_seconds") - 1e-9
        if cond and not st["active"] and held:
            st.update(active=True, start=st["cand_since"], text=st.get("cand_text") or text or "(sebep yok)")
            st["today"] += 1
            st["last_text"] = st["text"]
        elif not cond and st["active"] and held:
            self._end(st["cand_since"])
        elif cond and st["active"] and st["text"] == "(sebep yok)" and text:
            st["text"] = st["last_text"] = text
        self._flush(now)
        dur = (now - st["start"]).total_seconds() if st["active"] else st["last_dur"]
        lines = [f"{s:%d.%m %H:%M:%S} – {e:%H:%M:%S} ({d:.0f} sn) {t}" for s, e, d, t in st["history"]]
        if st["write_error"]:
            lines.insert(0, st["write_error"])
        return {"active": st["active"], "count_today": st["today"], "last_text": st["last_text"],
                "duration": round(dur, 1), "history": "\n".join(lines)}

    def teardown(self):
        st = self.state
        if st.get("active"):
            self._end(self.now(), "(akış durdu)")
        if st.get("pending"):
            st["next_try"] = None
            self._flush(self.now())


# --------------------------------------------------------------------------------------------------------------------
@register
class MultiStationVerdict(Node):
    title = "Çoklu Kamera Kararı (Hepsi Sağlam mı)"
    category = LOGIC
    description = ("Aynı parçayı farklı istasyonlarda gören en fazla 4 kameranın kararını birleştirir. Her istasyon "
                   "için 'tN' (karar hazır tetiği) ve 'okN' (sağlam mı) bağlayın. Parçalar geliş sırasına göre "
                   "eşleştirilir: her istasyondan bir karar gelince 'done' bir kare doğru olur, 'ok' hepsi sağlamsa "
                   "doğrudur, 'failed' hatalı istasyonları yazar. Bir istasyon süre içinde bildirmezse parça hatalı "
                   "sayılır ve 'timeout' doğru olur.")
    inputs = [("t1", BOOL), ("ok1", BOOL), ("t2", BOOL), ("ok2", BOOL), ("t3", BOOL), ("ok3", BOOL),
              ("t4", BOOL), ("ok4", BOOL)]
    outputs = [("done", BOOL), ("ok", BOOL), ("bad", BOOL), ("failed", TEXT), ("timeout", BOOL), ("parts", NUM),
               ("timeouts", NUM)]
    params = [Param("stations", "int", 2, "İstasyon sayısı", 1, 4),
              Param("timeout", "float", 2.0, "Bekleme süresi (sn)", 0.01, 3600, 0.1),
              Param("names", "str", "İstasyon 1, İstasyon 2, İstasyon 3, İstasyon 4", "İstasyon adları (virgülle)")]

    now = staticmethod(time.monotonic)
    MAX_QUEUE = 1000

    def setup(self):
        self.state.update(queues=[collections.deque() for _ in range(4)], ok=True, failed="", parts=0, timeouts=0)

    def _names(self, k):
        given = [s.strip() for s in self.v("names").split(",")]
        return [given[i] if i < len(given) and given[i] else f"İstasyon {i + 1}" for i in range(k)]

    def process(self, inputs):
        st, now, k = self.state, self.now(), self.v("stations")
        names = self._names(k)
        for i in range(k):
            if inputs[f"t{i + 1}"] is None:
                raise ValueError(f"{names[i]}: 't{i + 1}' tetik girişi bağlı değil")
            if _edge(self, f"prev_t{i}", inputs[f"t{i + 1}"]):
                q = st["queues"][i]
                if len(q) >= self.MAX_QUEUE:
                    raise ValueError(f"{names[i]} çok fazla eşleşmemiş karar biriktirdi; diğer istasyonlar "
                                     f"bildirmiyor mu?")
                if inputs[f"ok{i + 1}"] is None:
                    raise ValueError(f"{names[i]}: 'ok{i + 1}' (sağlam mı) girişi bağlı değil")
                q.append((now, bool(inputs[f"ok{i + 1}"])))
        qs = st["queues"][:k]
        done = timeout = False
        if all(qs):
            results = [q.popleft()[1] for q in qs]
            done = True
            st["ok"] = all(results)
            st["failed"] = ", ".join(n for n, r in zip(names, results) if not r)
        elif any(qs):
            oldest = min(q[0][0] for q in qs if q)
            if now - oldest >= self.v("timeout") - 1e-9:
                missing = [n for n, q in zip(names, qs) if not q]
                failed = []
                for n, q in zip(names, qs):
                    if q and not q.popleft()[1]:
                        failed.append(n)
                done = timeout = True
                st["ok"] = False
                st["timeouts"] += 1
                st["failed"] = ", ".join(failed + [f"{n} bildirmedi" for n in missing])
        if done:
            st["parts"] += 1
        return {"done": done, "ok": st["ok"], "bad": done and not st["ok"], "failed": st["failed"],
                "timeout": timeout, "parts": st["parts"], "timeouts": st["timeouts"]}


# --------------------------------------------------------------------------------------------------------------------
@register
class SignalSimulator(Node):
    setup_params = ("seed", "rate", "pulses_per_part", "stop_every", "stop_seconds")
    title = "Sinyal Üreteci / Simülatör"
    category = SRC
    description = ("Donanım olmadan denemek için sanal bir konveyör: dakikada N parça tetiği ('trigger' bir kare "
                   "doğru), ayarlanan oranda rastgele hatalı parça ('bad', tohumlu: hep aynı sıra), enkoder sayısı, "
                   "parça başına bir ölçüm değeri ve isterseniz düzenli duruşlar ('running' yanlış olur). Ret Kapısı "
                   "Gecikmesi, SPC, OEE, Lot Takibi ve Alarm Geçmişi bloklarını bantsız deneyin. Tetik, akış hızının "
                   "yarısından sık parça üretemez (30 kare/sn → en fazla 15 parça/sn).")
    outputs = [("trigger", BOOL), ("bad", BOOL), ("encoder", NUM), ("running", BOOL), ("measurement", NUM),
               ("parts", NUM), ("good_count", NUM), ("bad_count", NUM)]
    params = [Param("rate", "float", 60.0, "Hız (parça/dk)", 0.1, 60000, 1),
              Param("bad_ratio", "float", 10.0, "Hatalı oranı (%)", 0, 100, 1),
              Param("seed", "int", 1, "Rastgele tohumu", 0, 2 ** 31 - 1),
              Param("pulses_per_part", "float", 100.0, "Parça arası enkoder darbesi", 1, 1e9, 1),
              Param("mean", "float", 10.0, "Ölçüm ortalaması", -1e9, 1e9, 0.1),
              Param("sd", "float", 0.1, "Ölçüm sapması (σ)", 0, 1e9, 0.01),
              Param("drift", "float", 0.0, "Parça başına kayma", -1e6, 1e6, 0.001),
              Param("stop_every", "float", 0.0, "Her N sn çalışınca dur (0 = durma)", 0, 1e7, 1),
              Param("stop_seconds", "float", 0.0, "Duruş süresi (sn)", 0, 1e7, 1)]
    is_source = True

    now = staticmethod(time.monotonic)
    MAX_BACKLOG = 5                   # parts that may wait for a free frame; beyond this they are dropped

    def setup(self):
        self.state.update(t0=self.now(), rng=random.Random(self.v("seed")), next_part=1, last_trig=False, parts=0,
                          good=0, bad_n=0, bad=False, meas=None)

    def _run_time(self, t: float) -> float:
        every, stop = self.v("stop_every"), self.v("stop_seconds")
        if every <= 0 or stop <= 0:
            return t
        cycle = every + stop
        full, rest = divmod(t, cycle)
        return full * every + min(rest, every)

    def _running(self, t: float) -> bool:
        every, stop = self.v("stop_every"), self.v("stop_seconds")
        return every <= 0 or stop <= 0 or t % (every + stop) < every

    def process(self, inputs):
        st = self.state
        t = max(0.0, self.now() - st["t0"])
        ppp = self.v("pulses_per_part")
        pos = self._run_time(t) * self.v("rate") / 60.0 * ppp
        trig = False
        due = int(math.floor(pos / ppp + 1e-9))                 # parts that should have arrived by now
        if due - st["next_part"] + 1 > self.MAX_BACKLOG:          # flow too slow: skip what cannot be shown
            st["next_part"] = due - self.MAX_BACKLOG + 1
        if due >= st["next_part"] and not st["last_trig"]:
            trig = True
            st["next_part"] += 1
            st["parts"] += 1
            rng = st["rng"]
            st["bad"] = rng.random() * 100.0 < self.v("bad_ratio")
            if st["bad"]:
                st["bad_n"] += 1
            else:
                st["good"] += 1
            st["meas"] = rng.gauss(self.v("mean"), self.v("sd")) + self.v("drift") * (st["parts"] - 1)
        st["last_trig"] = trig
        return {"trigger": trig, "bad": st["bad"], "encoder": int(math.floor(pos + 1e-9)),
                "running": self._running(t), "measurement": st["meas"], "parts": st["parts"],
                "good_count": st["good"], "bad_count": st["bad_n"]}
