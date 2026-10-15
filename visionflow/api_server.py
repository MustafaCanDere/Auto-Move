"""Remote control REST API: lets MES / ERP / SCADA or any other program watch and control VisionFlow over HTTP(S).

Everything lives under /api/v1 and speaks JSON (errors in Turkish: {"ok": false, "hata": "..."}):

  GET  /api/v1/durum              running?, FPS, frame count, errors, recipe, version
  POST /api/v1/baslat | durdur    start / stop the flow
  GET  /api/v1/degerler           latest numbers / yes-no / text outputs of every block (no images)
  GET  /api/v1/goruntu/<blok id>  latest picture of a block as JPEG
  POST /api/v1/tetik/<ad>         fires the 'API Tetik' block named <ad>; the JSON body becomes its 'data' output
  GET  /api/v1/recete             recipes; POST /api/v1/recete {"ad": ...} switches to one
  GET  /api/v1/degisken[/<ad>]    shared variables of 'Değişken Yaz / Oku'; POST {"ad", "deger", "kalici"} sets one
  GET  /api/v1/bloklar            blocks of the flow
  GET  /api/v1/openapi.json       machine-readable description (no key needed)

Security: an API key (X-API-Key header or "Authorization: Bearer <key>") is required by default, the server listens
on 127.0.0.1 unless the network is chosen explicitly, HTTPS with the user's certificate is optional and every client
address is rate limited. Only standard library modules are used (http.server).

The flow runs on another thread (the editor's worker or the headless loop): requests only read a snapshot that
the flow thread publishes after every frame (FlowState), and start / stop / recipe changes go through a controller
that does them on the right thread.
"""
from __future__ import annotations

import collections
import hmac
import json
import math
import secrets
import ssl
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

PREFIX = "/api/v1"
DEFAULT_PORT = 8790
MAX_BODY = 1_000_000
MAX_TEXT = 10_000
QUEUE_MAX = 100

DEFAULTS = {"enabled": False, "port": DEFAULT_PORT, "host": "127.0.0.1", "key": "", "require_key": True,
            "https": False, "cert": "", "keyfile": "", "rate": 20}
HOSTS = ("127.0.0.1", "0.0.0.0")


# =============================================================================================== settings
def settings_path():
    from .ai.config import home
    return home() / "uzaktan_api.json"


def new_key() -> str:
    return secrets.token_urlsafe(24)


def clean_settings(raw) -> dict:
    s = dict(DEFAULTS)
    if isinstance(raw, dict):
        for k, default in DEFAULTS.items():
            v = raw.get(k, default)
            if isinstance(default, bool):
                s[k] = v if isinstance(v, bool) else default
            elif isinstance(default, int):
                s[k] = v if isinstance(v, int) and not isinstance(v, bool) else default
            else:
                s[k] = v if isinstance(v, str) else default
    if s["host"] not in HOSTS:
        s["host"] = "127.0.0.1"
    s["port"] = min(max(s["port"], 0), 65535)
    s["rate"] = min(s["rate"], 10000) if s["rate"] >= 0 else DEFAULTS["rate"]     # 0 = no limit
    return s


def load_settings() -> dict:
    """The API settings of this computer; a key is made (and saved) the first time."""
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    s = clean_settings(raw)
    if not s["key"]:
        s["key"] = new_key()
        save_settings(s)
    return s


def save_settings(s: dict) -> None:
    s = clean_settings(s)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)                  # holds the key: only this user
    except OSError:
        pass
    tmp.replace(path)


def make_self_signed(cert_path, key_path, host: str = "localhost") -> None:
    """A self-signed HTTPS certificate (PEM) for testing / a closed network. Needs the 'cryptography' package."""
    try:
        import datetime as dt
        import ipaddress

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise RuntimeError("Sertifika oluşturmak için 'cryptography' paketi gerekli: pip install cryptography") \
            from exc
    import pathlib
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VisionFlow")])
    alt = [x509.DNSName(host), x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        alt.append(x509.IPAddress(ipaddress.ip_address(host)))
    except ValueError:
        pass
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = pathlib.Path(cert_path), pathlib.Path(key_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    try:
        key_path.chmod(0o600)
    except OSError:
        pass


# =============================================================================================== API Tetik queues
class TriggerHub:
    """Requests for the running 'API Tetik' blocks, by name. Blocks register while they run."""

    def __init__(self):
        self.lock = threading.Lock()
        self.queues: dict[str, collections.deque] = {}
        self.users: dict[str, int] = {}

    def register(self, name: str) -> None:
        with self.lock:
            self.users[name] = self.users.get(name, 0) + 1
            self.queues.setdefault(name, collections.deque())

    def unregister(self, name: str) -> None:
        with self.lock:
            n = self.users.get(name, 0) - 1
            if n > 0:
                self.users[name] = n
            else:
                self.users.pop(name, None)
                self.queues.pop(name, None)

    def push(self, name: str, payload) -> int:
        """Queue a request; returns how many are waiting. KeyError: no such block running; OverflowError: full."""
        with self.lock:
            q = self.queues.get(name)
            if q is None:
                raise KeyError(name)
            if len(q) >= QUEUE_MAX:
                raise OverflowError(name)
            q.append(payload)
            return len(q)

    def pop(self, name: str):
        """(True, payload) for the oldest request, (False, None) when there is none."""
        with self.lock:
            q = self.queues.get(name)
            if not q:
                return False, None
            return True, q.popleft()

    def waiting(self, name: str) -> int:
        with self.lock:
            return len(self.queues.get(name) or ())

    def names(self) -> list[str]:
        with self.lock:
            return sorted(self.users)


TRIGGERS = TriggerHub()


# =============================================================================================== flow snapshot
def plain(v):
    """A block output as JSON: numbers, yes/no, text (and short lists of them). None = leave it out."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, str):
        return v[:MAX_TEXT]
    if isinstance(v, (list, tuple, dict)):          # e.g. the JSON an 'API Tetik' got, a list of numbers
        def conv(x):
            if isinstance(x, (np.generic,)):
                return x.item()
            raise TypeError
        try:
            text = json.dumps(v, ensure_ascii=False, allow_nan=False, default=conv)
        except (TypeError, ValueError, RecursionError):
            return None                          # pictures, detections, NaN ...: not a plain value
        return json.loads(text) if len(text) <= 100_000 else None
    return None


def _image(v):
    return v if isinstance(v, np.ndarray) and v.ndim in (2, 3) and v.size > 0 and v.dtype != object else None


class FlowState:
    """What the API reports, written by the flow thread after every frame and read by the request threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.fps = 0.0
        self.frames = 0
        self.started = 0.0
        self.recipe = ""
        self.errors: dict[str, dict] = {}
        self.error_count = 0
        self.last_errors: collections.deque = collections.deque(maxlen=20)
        self.values: dict[str, dict] = {}
        self.images: dict[str, np.ndarray] = {}
        self._t0 = 0.0
        self._n0 = 0

    def attach(self, recipe: str = "") -> None:
        """A run starts: forget the previous one's values. ``recipe`` is the recipe it starts with ("" = none):
        never the one a previous run (of maybe another flow) ended with."""
        with self.lock:
            self.recipe = str(recipe or "")
            self.running, self.fps, self.frames = True, 0.0, 0
            self.started = self._t0 = time.time()
            self._n0 = 0
            self.errors, self.values, self.images = {}, {}, {}

    def detach(self) -> None:
        with self.lock:
            self.running, self.fps = False, 0.0

    def note_error(self, where: str, message: str) -> None:
        """A problem outside a block (the flow could not start, ...)."""
        with self.lock:
            self.error_count += 1
            self.last_errors.append({"zaman": time.strftime("%Y-%m-%d %H:%M:%S"), "blok": where, "hata": message})

    def set_recipe(self, name: str) -> None:
        with self.lock:
            self.recipe = str(name)

    def publish(self, graph, flat: dict, fps: float | None = None) -> None:
        """Called on the flow thread with the frame's results (flatten()ed: inner blocks as "sub/inner")."""
        from .core.graph import inner_error, node_at
        values, images, errors = {}, {}, {}
        for nid, r in flat.items():
            if r.waiting:
                continue
            node = node_at(graph, nid)
            if node is None:
                continue
            if r.error:
                if r.error != "Önceki blok hata verdi" and not inner_error(node, r.error):
                    errors[nid] = {"blok": nid, "baslik": node.title, "hata": str(r.error)}
                continue
            outs = {}
            for port, val in (r.outputs or {}).items():
                img = _image(val)
                if img is not None:
                    images.setdefault(nid, img)
                    continue
                p = plain(val)
                if p is not None:
                    outs[port] = p
            values[nid] = {"baslik": node.title, "tip": node.type_name, "cikislar": outs}
        now = time.time()
        with self.lock:
            self.frames += 1
            if fps is not None:
                self.fps = float(fps)
            elif now - self._t0 >= 1.0:
                self.fps = (self.frames - self._n0) / (now - self._t0)
                self._t0, self._n0 = now, self.frames
            self.values.update(values)
            self.images.update(images)
            for nid in values:
                self.errors.pop(nid, None)
            for nid, e in errors.items():
                if self.errors.get(nid, {}).get("hata") != e["hata"]:      # a new error, not the same every frame
                    self.error_count += 1
                    self.last_errors.append({"zaman": time.strftime("%Y-%m-%d %H:%M:%S"), **e})
                self.errors[nid] = e
                self.values.pop(nid, None)

    def status(self) -> dict:
        with self.lock:
            return {"calisiyor": self.running, "fps": round(self.fps, 2), "kare": self.frames,
                    "sure_sn": round(time.time() - self.started, 1) if self.running else 0.0,
                    "recete": self.recipe, "hatalar": list(self.errors.values()), "hata_sayisi": self.error_count,
                    "son_hatalar": list(self.last_errors)}

    def block_values(self) -> dict:
        with self.lock:
            return {k: {**v, "cikislar": dict(v["cikislar"])} for k, v in self.values.items()}

    def image(self, nid: str):
        with self.lock:
            return self.images.get(nid)


def stretch_u8(img: np.ndarray) -> np.ndarray:
    """A number picture (16-bit, float, depth in mm) stretched to 0…255 over its FINITE values: NaN / inf (no
    measurement, e.g. the holes of a depth map) are black and do not squeeze the contrast of the rest."""
    import cv2
    f = img.astype(np.float32)
    ok = np.isfinite(f)
    if ok.all():
        return cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = np.zeros(f.shape, np.uint8)
    if ok.any():
        v = f[ok]
        lo, hi = float(v.min()), float(v.max())
        if hi > lo:
            out[ok] = np.clip((v - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return out


def to_jpeg(img: np.ndarray, quality: int = 85, max_w: int = 0) -> bytes:
    """Any picture a block may output (gray, BGR, BGRA, bool mask, 16-bit, float) as JPEG bytes."""
    import cv2
    if img.ndim == 3:
        c = img.shape[2]
        img = img[:, :, 0] if c in (1, 2) else img[:, :, :3] if c >= 4 else img
    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        img = stretch_u8(img)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if max_w and w > max_w:
        img = cv2.resize(img, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("Görüntü JPEG'e çevrilemedi")
    return buf.tobytes()


def blocks_of(graph) -> list[dict]:
    from .core.graph import walk_nodes
    out = []
    for key, label, node in walk_nodes(graph):
        out.append({"id": key, "baslik": node.title, "etiket": label, "tip": node.type_name,
                    "kategori": node.category, "etkin": bool(node.enabled),
                    "cikislar": [{"ad": n, "tip": t} for n, t in node.outputs],
                    "goruntu": any(t in ("image", "any") for _, t in node.outputs)})
        if node.type_name == "ApiTrigger":
            out[-1]["tetik"] = str(node.values.get("name", ""))
    return out


# =============================================================================================== variables
def variables() -> dict:
    from .nodes.connect import VARS

    def show(v):
        if isinstance(v, np.ndarray):
            return f"<görüntü {'x'.join(str(s) for s in v.shape)}>"
        p = plain(v)
        return p if p is not None or v is None else str(v)[:MAX_TEXT]
    return {str(k): show(v) for k, v in list(VARS.items())}


def set_variable(name: str, value, persist: bool = False) -> None:
    from .nodes import connect
    connect.VARS[name] = value                 # one assignment: atomic for the flow thread reading it
    if persist:
        connect._persist(name, value)


# =============================================================================================== rate limit
class RateLimiter:
    """Token bucket per client address: ``rate`` requests a second on average, bursts up to ``rate`` too."""

    def __init__(self, rate: float):
        self.rate = float(rate)
        self.lock = threading.Lock()
        self.buckets: dict[str, list[float]] = {}

    def allow(self, who: str) -> bool:
        if self.rate <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(who, (self.rate, now))
            tokens = min(self.rate, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self.buckets[who] = [tokens, now]
                return False
            self.buckets[who] = [tokens - 1.0, now]
            if len(self.buckets) > 10000:      # many addresses: forget the full (idle) ones
                for k in [k for k, (t, _) in self.buckets.items() if t >= self.rate - 1][:5000]:
                    self.buckets.pop(k, None)
            return True


# =============================================================================================== HTTP
class ApiError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"
    ssl_context: ssl.SSLContext | None = None

    def get_request(self):
        sock, addr = super().get_request()
        if self.ssl_context is not None:
            # the TLS handshake happens on the request's own thread (first read), not in the accept loop
            sock = self.ssl_context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False)
        return sock, addr

    def handle_error(self, request, client_address):
        pass                                   # a client that hangs up / speaks plain HTTP to HTTPS: not our error


class ApiServer:
    """The HTTP server. ``controller`` does start / stop / recipe / blocks (see HeadlessController and the editor's
    one in ui/api_dialog.py); ``state`` is the FlowState the flow thread publishes to."""

    def __init__(self, controller, state: FlowState, settings: dict | None = None):
        self.controller, self.state = controller, state
        self.settings = clean_settings(settings if settings is not None else load_settings())
        self.limiter = RateLimiter(self.settings["rate"])
        self.server: _Server | None = None
        self.thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.server is not None

    @property
    def port(self) -> int:
        return self.server.server_address[1] if self.server else self.settings["port"]

    @property
    def url(self) -> str:
        s = self.settings
        host = "127.0.0.1"
        if s["host"] == "0.0.0.0":
            from .nodes.web import local_ip
            host = local_ip()
        return f"{'https' if s['https'] else 'http'}://{host}:{self.port}{PREFIX}"

    def start(self) -> str:
        s = self.settings
        if s["require_key"] and len(s["key"]) < 16:
            raise ValueError("API anahtarı çok kısa (en az 16 karakter): yeni bir anahtar oluşturun")
        if not s["require_key"] and s["host"] != "127.0.0.1":
            raise ValueError("Ağdan erişim için API anahtarı zorunlu: anahtarı açın ya da sadece bu bilgisayarı seçin")
        ctx = None
        if s["https"]:
            if not s["cert"] or not s["keyfile"]:
                raise ValueError("HTTPS için sertifika (.pem/.crt) ve anahtar (.key/.pem) dosyası seçin")
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            try:
                ctx.load_cert_chain(s["cert"], s["keyfile"])
            except (OSError, ssl.SSLError) as exc:
                raise ValueError(f"HTTPS sertifikası yüklenemedi ({s['cert']}): {exc}") from None
        try:
            srv = _Server((s["host"], s["port"]), _handler(self))
        except OSError as exc:
            raise RuntimeError(f"API için {s['port']} portu açılamadı (başka bir program kullanıyor olabilir): "
                               f"{exc}") from None
        srv.ssl_context = ctx
        self.server = srv
        self.thread = threading.Thread(target=srv.serve_forever, name="VisionFlow API", daemon=True)
        self.thread.start()
        return self.url

    def stop(self) -> None:
        srv, self.server = self.server, None
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        if self.thread is not None:
            self.thread.join(5)
            self.thread = None

    # request handling (on the request's thread) -------------------------------------------------------------
    def check_key(self, headers) -> None:
        if not self.settings["require_key"]:
            return
        given = headers.get("X-API-Key") or ""
        auth = headers.get("Authorization") or ""
        if not given and auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        if not given:
            raise ApiError(401, "API anahtarı eksik: X-API-Key başlığında gönderin")
        if not hmac.compare_digest(given.encode(), self.settings["key"].encode()):
            raise ApiError(401, "API anahtarı yanlış")

    def route(self, method: str, path: str, query: dict, body):
        """(status, JSON object) or (status, bytes, content type) for one request."""
        rest = path[len(PREFIX):].strip("/") if path.startswith(PREFIX) else None
        if rest is None:
            raise ApiError(404, f"Böyle bir adres yok: {path} (API adresleri {PREFIX}/ ile başlar)")
        head, _, tail = rest.partition("/")
        tail = urllib.parse.unquote(tail)
        c = self.controller
        if head == "openapi.json" and method == "GET":
            return 200, openapi(self.url)
        if head == "durum" and not tail:
            self._method(method, "GET")
            from . import __version__
            st = self.state.status()
            st["recete"] = self._recipe(st["calisiyor"], st["recete"])
            return 200, {"ok": True, "surum": __version__, "api": "v1", "akis": c.flow_name(), **st,
                         "tetikler": TRIGGERS.names()}
        if head in ("baslat", "durdur") and not tail:
            self._method(method, "POST")
            msg = c.start() if head == "baslat" else c.stop()
            return 200, {"ok": True, "mesaj": msg, "calisiyor": self.state.running}
        if head == "degerler" and not tail:
            self._method(method, "GET")
            vals = self.state.block_values()
            want = query.get("blok")
            if want:
                if want not in vals:
                    raise ApiError(404, f"'{want}' bloğunun değeri yok (blok yok, görüntü dışında çıkışı yok ya da "
                                        "akış henüz çalışmadı)")
                vals = {want: vals[want]}
            return 200, {"ok": True, "calisiyor": self.state.running, "kare": self.state.frames, "bloklar": vals}
        if head == "goruntu":
            self._method(method, "GET")
            if not tail:
                raise ApiError(400, f"Blok kimliği eksik: {PREFIX}/goruntu/<blok id>")
            img = self.state.image(tail)
            if img is None:
                raise ApiError(404, f"'{tail}' bloğunun görüntüsü yok (blok yok, görüntü çıkışı yok ya da akış henüz "
                                    "çalışmadı)")
            quality = _int(query.get("kalite"), 85, 10, 100, "kalite")
            width = _int(query.get("genislik"), 0, 0, 10000, "genislik")
            return 200, to_jpeg(img, quality, width), "image/jpeg"
        if head == "tetik":
            self._method(method, "POST")
            if not tail:
                raise ApiError(400, f"Tetik adı eksik: {PREFIX}/tetik/<ad>")
            try:
                n = TRIGGERS.push(tail, body)
            except KeyError:
                running = TRIGGERS.names()
                raise ApiError(404, f"Çalışan akışta '{tail}' adlı 'API Tetik' bloğu yok"
                                    + (f" (olanlar: {', '.join(running)})" if running else
                                       " (akış çalışıyor mu?)")) from None
            except OverflowError:
                raise ApiError(503, f"'{tail}' tetiğinin kuyruğu dolu ({QUEUE_MAX}); akış yetişemiyor") from None
            return 200, {"ok": True, "tetik": tail, "kuyrukta": n}
        if head == "recete" and not tail:
            if method == "GET":
                return 200, {"ok": True, "recete": self._recipe(self.state.running, self.state.recipe),
                             "receteler": c.recipes()}
            self._method(method, "POST")
            name = body.get("ad") if isinstance(body, dict) else None
            if isinstance(name, bool) or not isinstance(name, (str, int)) or (isinstance(name, str)
                                                                               and not name.strip()):
                raise ApiError(400, 'Reçete adı eksik: {"ad": "A ürünü"} gönderin (ya da numarası: {"ad": 1})')
            applied, warnings = c.set_recipe(name.strip() if isinstance(name, str) else name)
            self.state.set_recipe(applied)
            return 200, {"ok": True, "recete": applied, "uyarilar": warnings}
        if head == "degisken":
            if method == "GET":
                allv = variables()
                if tail:
                    if tail not in allv:
                        raise ApiError(404, f"'{tail}' adlı değişken yok")
                    return 200, {"ok": True, "ad": tail, "deger": allv[tail]}
                return 200, {"ok": True, "degiskenler": allv}
            self._method(method, "POST")
            if not isinstance(body, dict):
                raise ApiError(400, 'JSON nesnesi gönderin: {"ad": "sayac", "deger": 5}')
            name = str(body.get("ad") or tail or "").strip()
            if not name:
                raise ApiError(400, 'Değişken adı eksik: {"ad": "sayac", "deger": 5}')
            if "deger" not in body:
                raise ApiError(400, "'deger' eksik")
            value = body["deger"]
            if value is not None and plain(value) is None:
                raise ApiError(400, "Değer sayı, doğru/yanlış, metin ya da bunların listesi olmalı")
            set_variable(name, value, bool(body.get("kalici")))
            return 200, {"ok": True, "ad": name, "deger": value}
        if head == "bloklar" and not tail:
            self._method(method, "GET")
            return 200, {"ok": True, "bloklar": c.blocks()}
        raise ApiError(404, f"Böyle bir adres yok: {path}")

    def _recipe(self, running: bool, reported: str) -> str:
        """The recipe to report: the running flow's, else the one the flow that is open now would start with
        (the controller knows it; a stopped run's recipe may belong to a flow that is no longer open)."""
        fn = getattr(self.controller, "current_recipe", None)
        if running or fn is None:
            return reported
        try:
            return str(fn() or "")
        except Exception:  # noqa: BLE001
            return reported

    @staticmethod
    def _method(method: str, wanted: str) -> None:
        if method != wanted:
            raise ApiError(405, f"Bu adres {wanted} ister ({method} değil)")


def _no_nan(name):
    raise ValueError(f"JSON'da {name} olamaz")


def _int(text, default: int, lo: int, hi: int, what: str) -> int:
    if text in (None, ""):
        return default
    try:
        v = int(text)
    except ValueError:
        raise ApiError(400, f"'{what}' bir tamsayı olmalı") from None
    return min(max(v, lo), hi)


def _handler(api: ApiServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VisionFlowAPI/1"
        sys_version = ""
        timeout = 30                       # a client that never sends its request does not hold a thread forever

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj, extra: dict | None = None):
            body = json.dumps(obj, ensure_ascii=False, default=str, allow_nan=False).encode()
            self._send(code, body, "application/json; charset=utf-8", extra)

        def _handle(self, method: str):
            try:
                if not api.limiter.allow(self.client_address[0]):
                    raise ApiError(429, "Çok fazla istek: biraz yavaşlayın")
                url = urllib.parse.urlsplit(self.path)
                query = dict(urllib.parse.parse_qsl(url.query))
                path = url.path.rstrip("/") or "/"
                public = path == f"{PREFIX}/openapi.json"
                if not public:
                    api.check_key(self.headers)       # before reading the body: strangers send nothing big
                body = self._body(method)
                if not public and not api.settings["require_key"]:
                    self._local_only(method)
                result = api.route(method, path, query, body)
            except ApiError as exc:
                extra = {"Retry-After": "1"} if exc.code == 429 else \
                    {"WWW-Authenticate": 'ApiKey header="X-API-Key"'} if exc.code == 401 else None
                return self._json(exc.code, {"ok": False, "hata": str(exc)}, extra)
            except ValueError as exc:            # the controller refused (unknown recipe, flow cannot start ...)
                return self._json(400, {"ok": False, "hata": str(exc)})
            except TimeoutError as exc:
                return self._json(503, {"ok": False, "hata": str(exc) or "Program zamanında cevap vermedi"})
            except Exception as exc:  # noqa: BLE001
                return self._json(500, {"ok": False, "hata": f"Beklenmeyen hata: {type(exc).__name__}: {exc}"})
            if len(result) == 3:
                code, data, ctype = result
                return self._send(code, data, ctype)
            self._json(*result)

        def _body(self, method: str):
            if method != "POST":
                return None
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0:
                raise ApiError(400, "Content-Length geçersiz")
            if n > MAX_BODY:
                raise ApiError(413, f"İstek çok büyük (en fazla {MAX_BODY // 1000} kB)")
            raw = self.rfile.read(n) if n else b""
            self._ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if not raw.strip():
                return None
            try:
                return json.loads(raw.decode("utf-8"), parse_constant=_no_nan)
            except (UnicodeDecodeError, ValueError):
                raise ApiError(400, "Gövde geçerli JSON değil") from None

        def _local_only(self, method: str):
            """Without a key (this computer only): refuse what a web page open in a browser could send — a
            foreign Host (DNS rebinding) or a POST that is not JSON (a form)."""
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host not in ("127.0.0.1", "localhost", "::1"):
                raise ApiError(403, "Anahtarsız kullanımda sadece bu bilgisayardan (127.0.0.1) erişilebilir")
            if method == "POST" and getattr(self, "_ctype", "") != "application/json" \
                    and not self.headers.get("X-VisionFlow"):
                raise ApiError(403, "Anahtarsız kullanımda POST istekleri Content-Type: application/json olmalı")

        def do_GET(self):
            self._handle("GET")

        def do_HEAD(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_PUT(self):
            self._json(405, {"ok": False, "hata": "PUT desteklenmiyor: GET veya POST kullanın"})

        do_DELETE = do_PATCH = do_PUT

        def do_OPTIONS(self):
            # no CORS: a web page in a browser must not be able to call the API with the user's key
            self._json(405, {"ok": False, "hata": "Tarayıcıdan başka sitelerin erişimine izin verilmiyor"})

    return Handler


# =============================================================================================== OpenAPI
def openapi(url: str = "") -> dict:
    from . import __version__
    err = {"description": "Hata", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Hata"}}}}
    ok = {"description": "Tamam", "content": {"application/json": {"schema": {"type": "object"}}}}

    def op(summary, method="get", body=None, params=None, responses=None):
        o = {"summary": summary, "responses": responses or {"200": ok, "400": err, "401": err, "429": err}}
        if params:
            o["parameters"] = params
        if body:
            o["requestBody"] = {"required": False, "content": {"application/json": {"schema": body}}}
        return {method: o}

    def path_param(name, desc):
        return {"name": name, "in": "path", "required": True, "schema": {"type": "string"}, "description": desc}

    var_body = {"type": "object", "required": ["ad", "deger"],
                "properties": {"ad": {"type": "string"}, "deger": {}, "kalici": {"type": "boolean"}}}
    return {
        "openapi": "3.0.3",
        "info": {"title": "VisionFlow Uzaktan Kontrol API", "version": __version__,
                 "description": "VisionFlow'u MES/ERP/SCADA gibi programlardan izleme ve kontrol etme."},
        "servers": [{"url": url or PREFIX}],
        "security": [{"ApiKey": []}],
        "components": {
            "securitySchemes": {"ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}},
            "schemas": {"Hata": {"type": "object", "properties": {"ok": {"type": "boolean"},
                                                                  "hata": {"type": "string"}}}}},
        "paths": {
            "/durum": op("Çalışıyor mu, FPS, kare sayısı, hatalar, reçete, sürüm"),
            "/baslat": op("Akışı başlat", "post"),
            "/durdur": op("Akışı durdur", "post"),
            "/degerler": op("Blokların son sayı / doğru-yanlış / metin çıkışları (görüntüler hariç)",
                            params=[{"name": "blok", "in": "query", "required": False, "schema": {"type": "string"}}]),
            "/goruntu/{blok}": op("Bir bloğun son görüntüsü (JPEG)", params=[
                path_param("blok", "blok kimliği (alt akışta: altakis/blok)"),
                {"name": "kalite", "in": "query", "required": False, "schema": {"type": "integer"}},
                {"name": "genislik", "in": "query", "required": False, "schema": {"type": "integer"}}],
                responses={"200": {"description": "JPEG", "content": {"image/jpeg": {}}}, "404": err}),
            "/tetik/{ad}": op("'API Tetik' bloğunu tetikle; JSON gövde bloğun 'data' çıkışı olur", "post",
                              body={}, params=[path_param("ad", "API Tetik bloğundaki ad")]),
            "/recete": {**op("Reçeteler ve etkin reçete"),
                        **op("Reçete değiştir", "post", body={"type": "object", "required": ["ad"],
                                                              "properties": {"ad": {"type": "string"}}})},
            "/degisken": {**op("Değişken Yaz/Oku değişkenleri"), **op("Değişken yaz", "post", body=var_body)},
            "/degisken/{ad}": op("Bir değişkenin değeri", params=[path_param("ad", "değişken adı")]),
            "/bloklar": op("Akıştaki bloklar"),
        },
    }


# =============================================================================================== headless
class HeadlessController:
    """Start / stop / recipe for `python -m visionflow.headless akis.json --api PORT`: the flow runs on the thread
    that calls serve(); API requests only set flags that thread acts on between runs or between frames."""

    def __init__(self, flow: str, state: FlowState, *, fps: float = 30, plugins: str = "plugins",
                 recete: str | None = None, frames: int = 0, stats: float = 5.0, autostart: bool = True):
        self.flow, self.state = flow, state
        self.fps, self.plugins, self.recete, self.frames, self.stats = fps, plugins, recete, frames, stats
        self.graph = None                       # the running flow, while it runs
        self._want = autostart
        self._req = 0                           # 'baslat' requests so far (one that arrives as a run ends counts)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._defs = None
        self._runs = 0
        self.restart_on_crash = False
        self.max_restarts = 0                   # --en-fazla-yeniden (0 = forever)
        self.restarts = 0                       # restarts after crashes since the last 'baslat'
        self.log = None                         # logging.Logger of --yeniden-baslat (kayitlar/headless.log)

    # hooks called by headless.run_flow on the flow thread ----------------------------------------------------
    def attach(self, graph) -> None:
        self.graph = graph
        self._runs += 1
        self.state.attach(self.recete or "")

    def publish(self, graph, flat, fps=None) -> None:
        self.state.publish(graph, flat, fps)

    def recipe_applied(self, name: str) -> None:
        self.recete = name
        self.state.set_recipe(name)

    def detach(self) -> None:
        self.graph = None
        self.state.detach()

    # the flow thread ------------------------------------------------------------------------------------------
    def serve(self, should_stop=lambda: False) -> int:
        """Runs the flow whenever it is wanted. A crash with ``restart_on_crash`` starts it again after 1, 2, 4 … 60 s
        (like headless.watchdog: a run that lasted STABLE_SECONDS resets the wait), at most ``max_restarts`` times
        (0 = forever); then it waits for a 'baslat' request."""
        from . import headless
        code = 0
        delay = headless.BACKOFF_START
        while not should_stop():
            if not self._want:
                self._wake.wait(0.2)
                self._wake.clear()
                continue
            req = self._req
            started = headless._clock()
            if not self.restarts:               # a fresh start (not a restart after a crash): the shortest wait
                delay = headless.BACKOFF_START
            try:
                code = headless.run_flow(self.flow, self.frames, self.fps, self.stats, self.plugins,
                                         should_stop=lambda: should_stop() or not self._want,
                                         recete=self.recete, api=self)
                if code:
                    self.state.note_error("akış", f"Akış başlatılamadı (çıkış kodu {code})")
            except Exception as exc:  # noqa: BLE001
                code = 1
                self.state.note_error("akış", f"Akış çöktü: {type(exc).__name__}: {exc}")
                print(f"[HATA] Akış çöktü: {type(exc).__name__}: {exc}", flush=True)
                if self.log is not None:
                    import traceback
                    self.log.error("Akış çöktü: %s: %s\n%s", type(exc).__name__, exc, traceback.format_exc())
                if self.restart_on_crash and self._want and not should_stop():
                    if self.max_restarts and self.restarts >= self.max_restarts:
                        msg = f"En fazla yeniden başlatma sayısına ({self.max_restarts}) ulaşıldı"
                        print(f"{msg}; 'baslat' isteği bekleniyor.", flush=True)
                        self.state.note_error("akış", msg)
                        if self.log is not None:
                            self.log.error(msg)
                    else:
                        if headless._clock() - started >= headless.STABLE_SECONDS:
                            delay = headless.BACKOFF_START
                        print(f"{delay:g} sn sonra yeniden başlatılacak...", flush=True)
                        if self.log is not None:
                            self.log.warning("%g sn sonra yeniden başlatılacak", delay)
                        waited = 0.0
                        while waited < delay and not should_stop() and self._want:
                            step = min(0.1, delay - waited)
                            headless._sleep(step)
                            waited += step
                        self.restarts += 1
                        delay = min(delay * 2, headless.BACKOFF_MAX)
                        continue
            finally:
                self.graph = None
                self.state.detach()
            with self._lock:                    # ended (stopped, video over, frame limit): wait for 'baslat'
                if self._req == req:
                    self._want = False
        return code

    # API side (request threads) --------------------------------------------------------------------------------
    def flow_name(self) -> str:
        import pathlib
        return pathlib.Path(self.flow).name

    def start(self) -> str:
        if self.state.running:
            return "Akış zaten çalışıyor"
        before = self._runs
        with self._lock:
            self.restarts = 0                   # a new start gets the full number of restarts again
            self._req += 1
            self._want = True
        self._wake.set()
        end = time.monotonic() + 10
        while time.monotonic() < end and not self.state.running and self._want and self._runs == before:
            time.sleep(0.02)
        if self.state.running or self._runs != before:
            return "Akış başlatıldı"
        if not self._want:
            last = list(self.state.last_errors)
            raise ValueError("Akış başlatılamadı" + (f": {last[-1]['hata']}" if last else ""))
        return "Akış başlatılıyor"

    def stop(self) -> str:
        if not self.state.running and not self._want:
            return "Akış zaten durmuş"
        with self._lock:
            self._want = False
        end = time.monotonic() + 10
        while time.monotonic() < end and self.state.running:
            time.sleep(0.02)
        if self.state.running:
            raise TimeoutError("Akış 10 sn içinde durmadı (bir cihaz yanıt vermiyor olabilir)")
        return "Akış durduruldu"

    def current_recipe(self) -> str:
        """The recipe the next start begins with (for /durum while the flow is stopped)."""
        return self.recete or ""

    def _definitions(self):
        """The flow file's blocks and recipes (for when it is not running)."""
        if self._defs is None:
            from .core import Graph, load_builtin
            load_builtin()
            self._defs = Graph.load(self.flow)
        return self._defs

    def set_recipe(self, wanted):
        from .core import recipes
        g = self.graph
        if g is not None:
            r = g.request_recipe(wanted)          # applied by the flow thread before its next frame
        else:
            r = recipes.find(self._definitions(), wanted)
        self.recete = r["name"]                   # a later start (or restart) begins with it too
        return r["name"], []

    def recipes(self) -> list[str]:
        from .core import recipes
        g = self.graph or self._definitions()
        return recipes.names(g)

    def blocks(self) -> list[dict]:
        return blocks_of(self.graph or self._definitions())


def curl_examples(url: str, key: str, https: bool = False) -> str:
    k = "-k " if https else ""
    h = f'-H "X-API-Key: {key}"'
    j = '-H "Content-Type: application/json"'
    return "\n".join([
        f"curl {k}{h} {url}/durum",
        f"curl {k}{h} -X POST {url}/baslat",
        f"curl {k}{h} -X POST {url}/durdur",
        f"curl {k}{h} {url}/degerler",
        f"curl {k}{h} -o kare.jpg {url}/goruntu/<blok-id>",
        f"curl {k}{h} {j} -X POST -d '{{\"seri\": \"A123\"}}' {url}/tetik/<tetik-adi>",
        f"curl {k}{h} {j} -X POST -d '{{\"ad\": \"A ürünü\"}}' {url}/recete",
        f"curl {k}{h} {j} -X POST -d '{{\"ad\": \"sayac\", \"deger\": 0}}' {url}/degisken",
        f"curl {k}{h} {url}/bloklar",
    ])

