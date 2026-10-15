"""Source blocks: image files, cameras, videos, screen and a synthetic test pattern."""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

from ..core import Node, Param, PortType, StopFlow, register

CAT = "1 · Giriş"


def _read_image(path: str):
    # np.fromfile handles non-ASCII (Turkish) Windows paths, cv2.imread does not
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Resim okunamadı: {path}")
    return img


@register
class ImageFile(Node):
    title = "Resim Dosyası"
    category = CAT
    description = "Diskten bir resim okur."
    outputs = [("image", PortType.IMAGE)]
    params = [Param("path", "path", "", "Dosya")]

    def setup(self):
        self.state["img"] = None
        self.state["path"] = None

    def process(self, inputs):
        path = self.v("path")
        if not path:
            raise ValueError("Dosya seçilmedi")
        if self.state.get("path") != path:
            self.state["img"], self.state["path"] = _read_image(path), path
        return {"image": self.state["img"].copy()}


@register
class Camera(Node):
    title = "Kamera"
    category = CAT
    description = "USB/dahili kamera veya RTSP/HTTP IP kamera akışı."
    outputs = [("image", PortType.IMAGE), ("fps", PortType.NUMBER)]
    params = [
        Param("source", "str", "0", "Kaynak (indeks veya rtsp://...)"),
        Param("width", "int", 1280, "Genişlik", 0, 7680),
        Param("height", "int", 720, "Yükseklik", 0, 4320),
    ]
    is_source = True
    setup_params = ("source", "width", "height")

    RETRY_SECONDS = 2.0

    def setup(self):
        self.state.update(cap=self._open(), t=time.perf_counter(), fps=0.0, fails=0, retry_at=0.0)

    def _open(self):
        src = self.v("source").strip()
        if src.isdigit():
            # DirectShow opens USB cameras much faster than MSMF on Windows
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(int(src))
        else:
            cap = cv2.VideoCapture(src)
        if self.v("width"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.v("width"))
        if self.v("height"):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.v("height"))
        return cap

    def process(self, inputs):
        st = self.state
        if st["cap"] is None or not st["cap"].isOpened():
            # a camera that dropped out (cable, Wi-Fi, RTSP restart) is reopened, the flow keeps running
            if time.monotonic() >= st["retry_at"]:
                st["retry_at"] = time.monotonic() + self.RETRY_SECONDS
                if st["cap"] is not None:
                    st["cap"].release()
                st["cap"] = self._open()
            if not st["cap"].isOpened():
                raise RuntimeError(f"Kamera açılamadı: {self.v('source')} (tekrar deneniyor)")
        ok, frame = st["cap"].read()
        if not ok:
            if os.path.isfile(self.v("source").strip()):
                raise StopFlow()                 # a video file given as source: it simply ended
            st["fails"] += 1
            if st["fails"] >= 5:                 # several bad reads in a row: reconnect
                st["cap"].release()
                st["fails"], st["retry_at"] = 0, 0.0
            raise RuntimeError("Kameradan görüntü alınamadı (bağlantı koptu mu?), tekrar deneniyor")
        st["fails"] = 0
        now = time.perf_counter()
        dt, self.state["t"] = now - self.state["t"], now
        self.state["fps"] = 0.9 * self.state["fps"] + 0.1 * (1.0 / dt if dt > 0 else 0)
        return {"image": frame, "fps": round(self.state["fps"], 1)}

    def teardown(self):
        if self.state.get("cap") is not None:
            self.state["cap"].release()


@register
class VideoFile(Node):
    title = "Video Dosyası"
    category = CAT
    description = "Bir video dosyasını kare kare oynatır."
    outputs = [("image", PortType.IMAGE), ("frame", PortType.NUMBER)]
    params = [
        Param("path", "path", "", "Dosya"),
        Param("loop", "bool", True, "Başa sar"),
        Param("realtime", "bool", True, "Gerçek hızda oynat"),
    ]
    is_source = True

    def setup(self):
        if not self.v("path"):
            raise ValueError("Video dosyası seçilmedi")
        cap = cv2.VideoCapture(self.v("path"))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Video açılamadı: {self.v('path')}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.state.update(cap=cap, n=0, delay=1.0 / fps, last=0.0)

    def process(self, inputs):
        cap = self.state["cap"]
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {self.v('path')}")
        ok, frame = cap.read()
        if not ok:
            if not self.v("loop"):
                raise StopFlow()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                raise StopFlow()
        if self.v("realtime"):
            wait = self.state["delay"] - (time.perf_counter() - self.state["last"])
            if wait > 0:
                time.sleep(wait)
            self.state["last"] = time.perf_counter()
        self.state["n"] += 1
        return {"image": frame, "frame": self.state["n"]}

    def teardown(self):
        if self.state.get("cap") is not None:
            self.state["cap"].release()


@register
class ScreenCapture(Node):
    title = "Ekran Görüntüsü"
    category = CAT
    description = "Masaüstünün bir bölgesini yakalar (RPA / ekran otomasyonu için)."
    outputs = [("image", PortType.IMAGE)]
    params = [
        Param("x", "int", 0, "X", 0, 10000), Param("y", "int", 0, "Y", 0, 10000),
        Param("w", "int", 0, "Genişlik (0 = tüm ekran)", 0, 10000), Param("h", "int", 0, "Yükseklik", 0, 10000),
    ]
    is_source = True

    def process(self, inputs):
        from PIL import ImageGrab  # optional dependency

        x, y, w, h = (self.v(k) for k in ("x", "y", "w", "h"))
        bbox = (x, y, x + w, y + h) if w and h else None
        shot = ImageGrab.grab(bbox=bbox)
        return {"image": cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)}


@register
class TestPattern(Node):
    title = "Test Görüntüsü"
    category = CAT
    description = "Kamera olmadan denemek için hareketli renkli şekiller üretir."
    outputs = [("image", PortType.IMAGE)]
    params = [
        Param("width", "int", 640, "Genişlik", 64, 4096),
        Param("height", "int", 480, "Yükseklik", 64, 4096),
        Param("animate", "bool", True, "Hareketli (sürekli akış)"),
        Param("motion", "choice", "salınım", "Hareket", choices=["salınım", "bant"]),
    ]

    @property
    def is_source(self):  # only drive the loop when animated
        return bool(self.values.get("animate"))

    def setup(self):
        self.state["t"] = 0

    def process(self, inputs):
        w, h, t = self.v("width"), self.v("height"), self.state["t"]
        img = np.full((h, w, 3), 40, np.uint8)
        r = h // 8
        if self.v("motion") == "bant":   # conveyor: balls enter left, leave right, a new one follows
            cx = int((t * 8) % (w + 2 * r) - r)
        else:
            cx = int(w * 0.5 + w * 0.3 * np.sin(t / 15))
        cv2.circle(img, (cx, h // 3), r, (0, 0, 255), -1)                   # red ball
        cv2.rectangle(img, (w // 10, h // 2), (w // 10 + w // 5, h // 2 + h // 4), (0, 200, 0), -1)  # green box
        cv2.rectangle(img, (w // 2, h // 2 + 20), (w // 2 + w // 6, h // 2 + 20 + h // 6), (255, 128, 0), -1)
        cv2.putText(img, "VisionFlow", (w // 10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        self.state["t"] = t + 1
        return {"image": img}
