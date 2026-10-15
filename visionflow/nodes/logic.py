"""Logic, math and scripting blocks."""
from __future__ import annotations

import math
import operator
import time

from ..core import Node, Param, PortType, register

CAT = "4 · Mantık"
NUM, BOOL = PortType.NUMBER, PortType.BOOL

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne}


@register
class Number(Node):
    title = "Sayı"
    category = CAT
    outputs = [("value", NUM)]
    params = [Param("value", "float", 0.0, "Değer", -1e9, 1e9)]

    def process(self, inputs):
        return {"value": self.v("value")}


@register
class Compare(Node):
    title = "Karşılaştır"
    category = CAT
    description = "a ile b'yi (ya da sabit değeri) karşılaştırır."
    inputs = [("a", NUM), ("b", NUM)]
    outputs = [("result", BOOL)]
    params = [Param("op", "choice", ">", "İşlem", choices=list(OPS)), Param("b", "float", 0.0, "b (bağlı değilse)")]

    def process(self, inputs):
        a = inputs["a"] if inputs["a"] is not None else 0
        b = inputs["b"] if inputs["b"] is not None else self.v("b")
        return {"result": bool(OPS[self.v("op")](a, b))}


@register
class Math(Node):
    title = "Matematik"
    category = CAT
    inputs = [("a", NUM), ("b", NUM)]
    outputs = [("result", NUM)]
    params = [Param("op", "choice", "+", "İşlem", choices=["+", "-", "*", "/", "min", "max"])]

    def process(self, inputs):
        a, b = inputs["a"] or 0, inputs["b"] or 0
        f = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
             "/": lambda: a / b if b else 0, "min": lambda: min(a, b), "max": lambda: max(a, b)}
        return {"result": f[self.v("op")]()}


@register
class Logic(Node):
    title = "Mantık Kapısı"
    category = CAT
    inputs = [("a", BOOL), ("b", BOOL)]
    outputs = [("result", BOOL)]
    params = [Param("op", "choice", "and", "Kapı", choices=["and", "or", "xor", "not a"])]

    def process(self, inputs):
        a, b = bool(inputs["a"]), bool(inputs["b"])
        return {"result": {"and": a and b, "or": a or b, "xor": a != b, "not a": not a}[self.v("op")]}


@register
class EdgeCounter(Node):
    title = "Sayaç"
    category = CAT
    description = "Giriş yanlış→doğru olduğunda bir artar (bant üzerinde parça sayma)."
    inputs = [("trigger", BOOL), ("reset", BOOL)]
    outputs = [("count", NUM), ("rising", BOOL)]

    def setup(self):
        self.state.update(count=0, prev=False)

    def process(self, inputs):
        if inputs["reset"]:
            self.state["count"] = 0
        now = bool(inputs["trigger"])
        rising = now and not self.state["prev"]
        self.state["prev"] = now
        self.state["count"] += int(rising)
        return {"count": self.state["count"], "rising": rising}


@register
class Debounce(Node):
    title = "Kararlılık (Debounce)"
    category = CAT
    description = "Sinyal belirtilen süre boyunca aynı kalırsa çıkışı değiştirir. Titreşimli tespitleri yumuşatır."
    inputs = [("signal", BOOL)]
    outputs = [("stable", BOOL)]
    params = [Param("seconds", "float", 0.5, "Süre (sn)", 0, 3600, 0.1)]

    def setup(self):
        self.state.update(stable=False, cand=False, since=time.monotonic())

    def process(self, inputs):
        s, now, st = bool(inputs["signal"]), time.monotonic(), self.state
        if s != st["cand"]:
            st["cand"], st["since"] = s, now
        if now - st["since"] >= self.v("seconds"):
            st["stable"] = st["cand"]
        return {"stable": st["stable"]}


@register
class PythonScript(Node):
    title = "Python Kodu"
    category = CAT
    description = ("Kendi Python kodunuzu yazın. Girişler: a, b (her tip), img. "
                   "Sonucu 'out' ve 'out_img' değişkenlerine atayın. cv2 ve np hazır gelir.")
    inputs = [("a", PortType.ANY), ("b", PortType.ANY), ("img", PortType.IMAGE)]
    outputs = [("out", PortType.ANY), ("out_img", PortType.IMAGE)]
    params = [Param("code", "code", "out = a\nout_img = img", "Kod")]

    def setup(self):
        self.state["code"] = None

    def process(self, inputs):
        import cv2
        import numpy as np

        if self.state.get("src") != self.v("code"):
            self.state["code"] = compile(self.v("code"), f"<{self.title} {self.id}>", "exec")
            self.state["src"] = self.v("code")
        ns = {"cv2": cv2, "np": np, "state": self.state.setdefault("user", {}), "out": None, "out_img": None, **inputs}
        exec(self.state["code"], ns)  # noqa: S102 - user's own local script
        return {"out": ns["out"], "out_img": ns["out_img"]}


@register
class IntervalTimer(Node):
    title = "Zamanlayıcı"
    category = CAT
    description = ("Her N saniyede bir kısa süreli 'doğru' sinyali üretir (periyodik kayıt, rapor, günlük video). "
                   "Kamerasız akışları da sürekli çalıştırır.")
    outputs = [("tick", BOOL), ("elapsed", NUM)]
    params = [Param("seconds", "float", 5.0, "Aralık (sn)", 0.05, 86400, 0.5),
              Param("at_start", "bool", False, "Başlarken hemen bir kez tetikle")]
    is_source = True

    def setup(self):
        now = time.monotonic()
        self.state.update(start=now, last=now - self.v("seconds") if self.v("at_start") else now)

    def process(self, inputs):
        now = time.monotonic()
        tick = now - self.state["last"] >= self.v("seconds")
        if tick:
            self.state["last"] = now
        return {"tick": tick, "elapsed": round(now - self.state["start"], 1)}


@register
class HoldPulse(Node):
    title = "Sinyali Uzat"
    category = CAT
    description = "Giriş bir an bile doğru olursa çıkışı N saniye boyunca doğru tutar (lamba, röle, kayıt)."
    inputs = [("signal", BOOL)]
    outputs = [("out", BOOL)]
    params = [Param("seconds", "float", 2.0, "Süre (sn)", 0, 3600, 0.1)]

    def setup(self):
        self.state["until"] = 0.0

    def process(self, inputs):
        now = time.monotonic()
        if inputs["signal"]:
            self.state["until"] = now + self.v("seconds")
        return {"out": now < self.state["until"]}


@register
class Latch(Node):
    title = "Mandal (SR)"
    category = CAT
    description = "'set' gelince doğru olur ve 'reset' gelene kadar öyle kalır (alarm kilitleme)."
    inputs = [("set", BOOL), ("reset", BOOL)]
    outputs = [("out", BOOL)]

    def setup(self):
        self.state["q"] = False

    def process(self, inputs):
        if inputs["reset"]:
            self.state["q"] = False
        elif inputs["set"]:
            self.state["q"] = True
        return {"out": self.state["q"]}


@register
class Switch(Node):
    title = "Seçici"
    category = CAT
    description = "Koşul doğruysa 'a', yanlışsa 'b' girişini geçirir."
    inputs = [("cond", BOOL), ("a", PortType.ANY), ("b", PortType.ANY)]
    outputs = [("out", PortType.ANY)]

    def process(self, inputs):
        return {"out": inputs["a"] if inputs["cond"] else inputs["b"]}


@register
class TextFormat(Node):
    title = "Metin Oluştur"
    category = CAT
    description = "Değerlerden metin oluşturur: {a}, {b}, {c} ve {zaman} yer tutucuları."
    inputs = [("a", PortType.ANY), ("b", PortType.ANY), ("c", PortType.ANY)]
    outputs = [("text", PortType.TEXT)]
    params = [Param("template", "str", "Adet={a}", "Şablon")]

    def process(self, inputs):
        text = self.v("template").replace("{zaman}", time.strftime("%Y-%m-%d %H:%M:%S"))
        for k in ("a", "b", "c"):
            text = text.replace("{" + k + "}", str(inputs[k]))
        return {"text": text}


@register
class Statistics(Node):
    title = "İstatistik"
    category = CAT
    description = "Bir sayının son N değerinin ortalamasını, min/max'ını verir; gürültülü ölçümleri yumuşatır."
    inputs = [("value", NUM), ("reset", BOOL)]
    outputs = [("mean", NUM), ("min", NUM), ("max", NUM)]
    params = [Param("window", "int", 30, "Pencere (değer)", 1, 100000)]

    def setup(self):
        self.state["vals"] = []

    def process(self, inputs):
        vals = self.state["vals"]
        if inputs["reset"]:
            vals.clear()
        if inputs["value"] is not None:
            v = float(inputs["value"])
            if math.isfinite(v):                    # one NaN/inf reading must not spoil the whole window
                vals.append(v)
            del vals[:-self.v("window")]
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {"mean": round(sum(vals) / len(vals), 3), "min": min(vals), "max": max(vals)}


@register
class Note(Node):
    title = "Not"
    category = "7 · Düzen"
    description = "Tuvale açıklama notu koyar. Akışı etkilemez."
    params = [Param("text", "code", "Bu akış ne yapıyor?\nBuraya not yazın.", "Metin")]

    def process(self, inputs):
        return {}
