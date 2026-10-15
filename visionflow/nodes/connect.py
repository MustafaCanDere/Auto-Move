"""0.9 blocks: shared variables (wireless links, counters that survive restarts) and an incoming webhook."""
from __future__ import annotations

import hmac
import json
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..ai.config import home
from ..core import Node, Param, PortType, register
from .ai_hub import _Feed
from .outputs import _rising

NUM, BOOL, TEXT, ANY = PortType.NUMBER, PortType.BOOL, PortType.TEXT, PortType.ANY
LOGIC, HUB = "4 · Mantık", "6 · AI Merkezi"

# one store for the running flow: every 'Değişken Yaz' / 'Değişken Oku' with the same name shares a value
VARS: dict[str, object] = {}
_LOCK = threading.Lock()


def _store_file():
    return home() / "degiskenler.json"


def _load_persisted() -> dict:
    try:
        return json.loads(_store_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _persist(name: str, value) -> None:
    with _LOCK:
        data = _load_persisted()
        data[name] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
        f = _store_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)


def _same(a, b) -> bool:
    """a == b that also works for images and other arrays (== on an array gives an array, not True/False)."""
    if a is b:
        return True
    if hasattr(a, "shape") or hasattr(b, "shape"):
        try:
            import numpy as np
            return bool(np.array_equal(a, b))
        except Exception:  # noqa: BLE001
            return False
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return False


def _initial(text: str):
    """'0' → 0, '1.5' → 1.5, anything else stays text, empty → None."""
    if text == "":
        return None
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            pass
    return text


@register
class SetVariable(Node):
    setup_params = ("name", "initial", "persist")
    title = "Değişken Yaz"
    category = LOGIC
    description = ("Bir değeri isimle saklar; aynı isimli 'Değişken Oku' blokları onu her yerde okur (kablo "
                   "çekmeden bağlantı). 'Ekle' modu sayaç gibi üstüne ekler. 'Kalıcı' seçilirse program kapanıp "
                   "açılınca değer korunur.")
    inputs = [("value", ANY), ("trigger", BOOL)]
    outputs = [("value", ANY)]
    params = [Param("name", "str", "sayac", "Değişken adı"),
              Param("mode", "choice", "yaz", "İşlem", choices=["yaz", "ekle", "sıfırla"]),
              Param("initial", "str", "0", "Başlangıç değeri"),
              Param("persist", "bool", False, "Kalıcı (kapanınca unutma)")]

    def setup(self):
        name = self.v("name").strip() or "degisken"
        self.state["name"] = name
        saved = _load_persisted() if self.v("persist") else {}
        VARS[name] = saved[name] if name in saved else _initial(self.v("initial"))
        self.state["last_in"] = object()

    def process(self, inputs):
        name, v = self.state["name"], inputs["value"]
        if inputs["trigger"] is not None:
            fire = _rising(self, inputs["trigger"])
        else:                                  # no trigger: act when the incoming value changes
            fire = v is not None and not _same(v, self.state["last_in"])
            self.state["last_in"] = v
        if fire:
            mode, cur = self.v("mode"), VARS.get(name)
            if mode == "sıfırla":
                new = _initial(self.v("initial"))
            elif mode == "ekle":
                try:
                    step = 1 if v is None or isinstance(v, bool) else float(v)
                    total = float(cur or 0) + step
                    new = int(total) if total.is_integer() and not isinstance(v, float) else total
                except (TypeError, ValueError):
                    new = f"{cur or ''}{v}"         # text: append
            else:
                new = v
            VARS[name] = new
            if self.v("persist"):
                _persist(name, new)
        return {"value": VARS.get(name)}


@register
class GetVariable(Node):
    title = "Değişken Oku"
    category = LOGIC
    description = "'Değişken Yaz' ile saklanan değeri okur. Değer değiştiği karede 'changed' doğru olur."
    outputs = [("value", ANY), ("number", NUM), ("text", TEXT), ("changed", BOOL)]
    params = [Param("name", "str", "sayac", "Değişken adı")]

    _START = object()

    def setup(self):
        self.state["prev"] = self._START

    def process(self, inputs):
        v = VARS.get(self.v("name").strip() or "degisken")
        changed = self.state["prev"] is not self._START and not _same(v, self.state["prev"])
        self.state["prev"] = v
        try:
            num = float(v)
        except (TypeError, ValueError):
            num = 0.0
        return {"value": v, "number": num, "text": "" if v is None else str(v), "changed": changed}


# --- incoming webhook ------------------------------------------------------------------------------

class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler(node: "WebhookReceive"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self, body: bytes):
            url = urllib.parse.urlsplit(self.path)
            query = dict(urllib.parse.parse_qsl(url.query))
            token = node.v("token")
            given = self.headers.get("X-Token") or query.pop("token", "")
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if token:
                if not hmac.compare_digest(given.encode(), token.encode()):
                    return self._reply(403, {"ok": False, "hata": "token yanlış"})
            else:
                # no token (this computer only): a web page open in the browser can also reach 127.0.0.1, but it
                # cannot send JSON or a custom header without the browser asking first, and it cannot fake the
                # Host header. So only requests a browser page cannot forge are accepted.
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
                if host not in ("127.0.0.1", "localhost", "::1"):
                    return self._reply(403, {"ok": False, "hata": "sadece bu bilgisayardan"})
                if not (body and ctype == "application/json") and not self.headers.get("X-VisionFlow"):
                    return self._reply(403, {"ok": False, "hata": "token belirleyin ya da JSON gönderin "
                                                                  "(veya X-VisionFlow başlığı ekleyin)"})
            data: object = query
            text = body.decode("utf-8", "replace")
            if body:
                if ctype == "application/x-www-form-urlencoded":
                    data = {**query, **dict(urllib.parse.parse_qsl(text))}
                else:
                    try:
                        data = json.loads(text)
                    except ValueError:
                        data = {**query, "text": text}
            if isinstance(data, dict):
                msg = next((str(data[k]) for k in ("text", "message", "mesaj", "content") if k in data), None)
                text = msg if msg is not None else (json.dumps(data, ensure_ascii=False) if data else text)
            with node.inbox_lock:
                if len(node.inbox) >= 1000:
                    return self._reply(503, {"ok": False, "hata": "kuyruk dolu"})
                node.inbox.append({"id": uuid.uuid4().hex, "path": url.path, "data": data, "text": text})
            self._reply(200, {"ok": True})

        def do_GET(self):
            self._handle(b"")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0:
                return self._reply(400, {"ok": False, "hata": "Content-Length geçersiz"})
            if n > 1_000_000:
                return self._reply(413, {"ok": False, "hata": "çok büyük"})
            self._handle(self.rfile.read(n))

    return Handler


@register
class WebhookReceive(_Feed):
    setup_params = ("port", "listen", "token")
    title = "Dışarıdan Tetik (Webhook Al)"
    category = HUB
    description = ("Program içinde küçük bir web adresi açar: başka bir program, form, Zapier/Make, PLC ağ geçidi "
                   "veya telefonunuz http://bilgisayar:8765/ adresine istek atınca akışı tetikler ve gönderilen "
                   "veriyi verir. Token yoksa sadece bu bilgisayardan gelen JSON istekleri kabul edilir (tarayıcıda açık bir "
                   "sitenin akışı tetiklemesini önler); form, adres parametresi veya ağdan erişim için token belirleyin.")
    inputs = [("next", BOOL)]
    outputs = [("text", TEXT), ("data", ANY), ("path", TEXT), ("new", BOOL), ("waiting", NUM)]
    params = [Param("port", "int", 8765, "Port", 1024, 65535),
              Param("listen", "choice", "sadece bu bilgisayar", "Kim erişebilir",
                    choices=["sadece bu bilgisayar", "yerel ağ"]),
              Param("token", "str", "", "Token (X-Token başlığı veya ?token=)")]
    def poll_interval(self):
        return 0.05                      # the inbox is in memory: check it often

    def setup(self):
        super().setup()
        if self.v("listen") == "yerel ağ" and not self.v("token"):
            raise ValueError("Yerel ağdan erişim için bir token girin (herkes akışınızı tetikleyemesin)")
        self.inbox, self.inbox_lock = [], threading.Lock()
        host = "0.0.0.0" if self.v("listen") == "yerel ağ" else "127.0.0.1"
        try:
            self.server = _Server((host, self.v("port")), _make_handler(self))
        except OSError as exc:
            raise RuntimeError(f"Port {self.v('port')} kullanılamıyor (başka bir program açmış olabilir): {exc}") \
                from None
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def teardown(self):
        srv = getattr(self, "server", None)
        if srv:
            srv.shutdown()
            srv.server_close()
            self.server = None

    def fetch(self):
        with self.inbox_lock:
            items, self.inbox[:] = list(self.inbox), []
        return items

    def item_key(self, item):
        return item["id"]

    def _remember(self, key):
        pass                             # every request is new; nothing to remember across restarts

    def _save_seen(self):
        pass

    def empty_outputs(self):
        return {"text": "", "data": None, "path": ""}

    def outputs_for(self, item):
        return {"text": item["text"], "data": item["data"], "path": item["path"]}
