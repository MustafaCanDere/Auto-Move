"""Output / automation blocks: preview, saving, logging, webhooks, CSV, sound."""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import threading
import time
import urllib.request

import cv2
import numpy as np

from ._text import put_text
from ..core import Node, Param, PortType, register

CAT = "5 · Çıkış ve Otomasyon"


@register
class Preview(Node):
    title = "Önizleme"
    category = CAT
    description = "Görüntüyü blok içinde ve önizleme panelinde gösterir."
    inputs = [("image", PortType.IMAGE)]
    outputs = [("image", PortType.IMAGE)]   # pass-through so it can sit mid-chain
    params = [Param("overlay", "str", "", "Üst yazı")]

    def process(self, inputs):
        img = inputs["image"]
        if img is not None and self.v("overlay"):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
            put_text(img, self.v("overlay"), (10, 30), 0.9, (255, 255, 255), 2)
        return {"image": img}


@register
class ShowValue(Node):
    title = "Değer Göster"
    category = CAT
    description = "Herhangi bir değeri blok üstünde ve günlükte gösterir."
    inputs = [("value", PortType.ANY)]
    outputs = [("text", PortType.TEXT)]
    params = [Param("prefix", "str", "", "Ön ek"), Param("log", "bool", False, "Her karede günlüğe yaz")]

    def process(self, inputs):
        val = inputs["value"]
        if isinstance(val, list):
            text = f"{len(val)} öğe"
        elif isinstance(val, np.ndarray):
            text = f"dizi {val.shape}"
        else:
            text = str(val)
        text = f"{self.v('prefix')}{text}"
        if self.v("log"):
            print(f"[{self.title}] {text}")
        return {"text": text}


def _rising(node: Node, trigger) -> bool:
    """True once per False->True transition. Unconnected trigger = always fire."""
    if trigger is None:
        return True
    now = bool(trigger)
    fired = now and not node.state.get("prev_trigger", False)
    node.state["prev_trigger"] = now
    return fired


@register
class SaveImage(Node):
    title = "Resim Kaydet"
    category = CAT
    description = "Tetik geldiğinde (veya her N karede) görüntüyü zaman damgalı dosyaya kaydeder."
    inputs = [("image", PortType.IMAGE), ("trigger", PortType.BOOL)]
    outputs = [("path", PortType.TEXT)]
    params = [
        Param("folder", "str", "kayitlar", "Klasör"),
        Param("prefix", "str", "kare", "Dosya ön eki"),
        Param("every_n", "int", 0, "Tetik yoksa her N karede (0 = kapalı)", 0, 100000),
    ]

    def setup(self):
        self.state["n"] = 0

    def process(self, inputs):
        img, trig = inputs["image"], inputs["trigger"]
        self.state["n"] += 1
        if trig is not None:
            fire = _rising(self, trig)
        else:
            n = self.v("every_n")
            fire = bool(n) and self.state["n"] % n == 0
        if img is None or not fire:
            return {"path": ""}
        os.makedirs(self.v("folder"), exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.v("folder"), f"{self.v('prefix')}_{stamp}.jpg")
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            buf.tofile(path)   # unicode-safe on Windows
        return {"path": path}


@register
class Webhook(Node):
    title = "Webhook / HTTP"
    category = CAT
    description = ("Tetik yükseldiğinde bir URL'ye JSON POST eder: Telegram, Slack, Teams, "
                   "Node-RED, kendi sunucunuz, PLC ağ geçidi...")
    inputs = [("trigger", PortType.BOOL), ("value", PortType.ANY)]
    outputs = [("sent", PortType.BOOL)]
    params = [Param("url", "str", "", "URL"), Param("message", "str", "Olay algılandı", "Mesaj")]

    def process(self, inputs):
        if inputs["trigger"] is None or not _rising(self, inputs["trigger"]) or not self.v("url"):
            return {"sent": False}
        val = inputs["value"]
        body = json.dumps({"text": self.v("message"), "message": self.v("message"),
                           "value": val if isinstance(val, (int, float, str, bool)) else str(val),
                           "time": dt.datetime.now().isoformat()}).encode()
        req = urllib.request.Request(self.v("url"), body, {"Content-Type": "application/json"})

        def send():
            try:
                urllib.request.urlopen(req, timeout=5).close()
            except Exception as exc:  # noqa: BLE001
                print(f"[Webhook] gönderilemedi: {exc}")
        # fire-and-forget so a slow server never stalls the vision loop
        threading.Thread(target=send, daemon=True).start()
        return {"sent": True}


@register
class CsvLogger(Node):
    title = "CSV Kaydı"
    category = CAT
    description = "Değerleri zaman damgasıyla CSV dosyasına yazar (Excel ile açılır)."
    inputs = [("a", PortType.ANY), ("b", PortType.ANY), ("trigger", PortType.BOOL)]
    params = [Param("path", "str", "kayit.csv", "Dosya"), Param("every_n", "int", 1, "Tetik yoksa her N karede", 1, 100000)]

    RETRY_EVERY = 2.0       # seconds between attempts while Excel has the file open

    def setup(self):
        self.state.update(n=0, pending=[], retry_at=0.0)

    def _flush(self):
        path = self.v("path")
        new = not os.path.exists(path)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            if new:
                w.writerow(["zaman", "a", "b"])
            w.writerows(self.state["pending"])
        self.state["pending"] = []

    def process(self, inputs):
        st = self.state
        st["n"] += 1
        fire = _rising(self, inputs["trigger"]) if inputs["trigger"] is not None else st["n"] % self.v("every_n") == 0
        if fire:
            st["pending"].append([dt.datetime.now().isoformat(timespec="milliseconds"), inputs["a"], inputs["b"]])
        if st["pending"] and time.monotonic() >= st["retry_at"]:
            try:
                self._flush()
            except PermissionError:
                # Excel locks an open CSV: keep the rows and write them all once the file is closed
                st["retry_at"] = time.monotonic() + self.RETRY_EVERY
        if st["pending"]:
            raise RuntimeError(f"Dosya başka bir programda (Excel) açık, {len(st['pending'])} satır bekletiliyor; "
                               "dosya kapanınca hepsi yazılır")
        return {}

    def teardown(self):
        if self.state.get("pending"):
            try:
                self._flush()
            except OSError:
                pass


@register
class Beep(Node):
    title = "Sesli Uyarı"
    category = CAT
    description = "Tetik yükseldiğinde bilgisayar hoparlöründen bip sesi çalar."
    inputs = [("trigger", PortType.BOOL)]
    params = [Param("freq", "int", 1000, "Frekans (Hz)", 37, 32767), Param("ms", "int", 300, "Süre (ms)", 10, 5000)]

    def process(self, inputs):
        if inputs["trigger"] is not None and _rising(self, inputs["trigger"]):
            try:
                import winsound
                threading.Thread(target=winsound.Beep, args=(self.v("freq"), self.v("ms")), daemon=True).start()
            except ImportError:
                print("\a", end="", flush=True)
        return {}
