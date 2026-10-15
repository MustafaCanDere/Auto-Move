"""Web panel: watch the live image and values from any browser or phone on the network."""
from __future__ import annotations

import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from ..core import Node, Param, PortType, register

PAGE = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
body{{margin:0;font-family:system-ui,sans-serif;background:#1e1f22;color:#dfe1e5}}
header{{padding:12px 16px;background:#2b2d30;font-weight:600}}
main{{display:flex;flex-wrap:wrap;gap:16px;padding:16px}}
img{{max-width:100%;border-radius:8px;background:#000}}
.card{{background:#2b2d30;border-radius:8px;padding:12px 16px;min-width:140px}}
.k{{color:#8b8e94;font-size:13px}}.v{{font-size:28px;font-weight:600;margin-top:4px}}
#vals{{display:flex;flex-direction:column;gap:12px}}
</style></head><body><header>{title} <span id="st" class="k"></span></header>
<main><div style="flex:3 1 480px"><img src="stream.mjpg" alt="canlı görüntü"></div><div id="vals"></div></main>
<script>
async function tick(){{try{{const r=await fetch('api');const d=await r.json();
const box=document.getElementById('vals');box.replaceChildren(...Object.entries(d.values).map(([k,v])=>{{
const c=document.createElement('div');c.className='card';const a=document.createElement('div');a.className='k';
a.textContent=k;const b=document.createElement('div');b.className='v';b.textContent=v;c.append(a,b);return c}}));
document.getElementById('st').textContent=' · '+d.time}}catch(e){{document.getElementById('st').textContent=' · bağlantı yok'}}}}
setInterval(tick,500);tick();
</script></body></html>"""


class _Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.values: dict = {}
        self.seq = 0
        self.title = "VisionFlow"
        self.closed = False


def _handler(shared: _Shared):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):   # keep the editor log clean
            pass

        def _send(self, code, ctype, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.format(title=shared.title).encode())
            elif path == "/api":
                with shared.lock:
                    values = {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
                              for k, v in shared.values.items()}     # NaN is not valid JSON for the browser
                    body = json.dumps({"values": values, "time": time.strftime("%H:%M:%S")},
                                      ensure_ascii=False, default=str).encode()
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/snapshot.jpg":
                with shared.lock:
                    jpg = shared.jpeg
                self._send(200 if jpg else 503, "image/jpeg", jpg or b"")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = -1
                try:
                    while not shared.closed:
                        with shared.lock:
                            jpg, seq = shared.jpeg, shared.seq
                        if jpg and seq != last:
                            last = seq
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            else:
                self._send(404, "text/plain", b"yok")
    return H


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


@register
class WebPanel(Node):
    setup_params = ("port", "title")   # read only when the block starts: changing them restarts it
    title = "Web Panel"
    category = "5 · Çıkış ve Otomasyon"
    description = ("Canlı görüntüyü ve değerleri tarayıcıda yayınlar. Aynı ağdaki telefon veya bilgisayardan "
                   "http://<bu-bilgisayarın-ip>:8080 adresini açın. /api JSON, /snapshot.jpg tek kare verir.")
    inputs = [("image", PortType.IMAGE), ("a", PortType.ANY), ("b", PortType.ANY), ("c", PortType.ANY),
              ("d", PortType.ANY)]
    outputs = [("url", PortType.TEXT)]
    params = [Param("port", "int", 8080, "Port", 1024, 65535),
              Param("title", "str", "VisionFlow Canlı", "Sayfa başlığı"),
              Param("names", "str", "Adet,Durum,Değer,Not", "a,b,c,d isimleri"),
              Param("quality", "int", 75, "JPEG kalitesi", 10, 100),
              Param("max_width", "int", 960, "Yayın genişliği (px)", 160, 4096)]

    def setup(self):
        shared = _Shared()
        shared.title = self.v("title")
        server = ThreadingHTTPServer(("0.0.0.0", self.v("port")), _handler(shared))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://{local_ip()}:{self.v('port')}"
        print(f"[Web Panel] yayında: {url}")
        self.state.update(shared=shared, server=server, url=url)

    def process(self, inputs):
        sh: _Shared = self.state["shared"]
        names = [n.strip() for n in self.v("names").split(",")] + ["a", "b", "c", "d"]
        values = {}
        for i, k in enumerate(("a", "b", "c", "d")):
            v = inputs[k]
            if v is None:
                continue
            if isinstance(v, bool):
                v = "✔" if v else "✘"
            elif isinstance(v, float):
                v = round(v, 3)
            elif isinstance(v, list):
                v = len(v)
            elif isinstance(v, np.ndarray):
                continue
            values[names[i] or k] = v
        jpg = None
        img = inputs["image"]
        if img is not None:
            h, w = img.shape[:2]
            mw = self.v("max_width")
            if w > mw:
                img = cv2.resize(img, (mw, int(h * mw / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.v("quality")])
            jpg = buf.tobytes() if ok else None
        with sh.lock:
            sh.values = values
            if jpg:
                sh.jpeg, sh.seq = jpg, sh.seq + 1
        return {"url": self.state["url"]}

    def teardown(self):
        srv = self.state.get("server")
        if self.state.get("shared"):
            self.state["shared"].closed = True   # ends open MJPEG streams
        if srv is not None:
            srv.shutdown()
            srv.server_close()
