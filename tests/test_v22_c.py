"""2.2 (c): remote control REST API (api_server.py, 'API Tetik', headless --api, editor dialog) and OPC UA security."""
import json
import ssl
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from visionflow import api_server
from visionflow.core import Graph, load_builtin, registry

load_builtin()

KEY = "test-anahtar-0123456789abcdef"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def call(url, method="GET", body=None, key=KEY, headers=None, ctx=None, raw=False):
    """(status, JSON or bytes, headers) of a real HTTP request."""
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(url, data=data, method=method)
    if key is not None:
        req.add_header("X-API-Key", key)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
            payload, code, hdrs = r.read(), r.status, r.headers
    except urllib.error.HTTPError as e:
        payload, code, hdrs = e.read(), e.code, e.headers
    return code, payload if raw else json.loads(payload.decode()), hdrs


def wait(cond, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.03)
    raise AssertionError("koşul gerçekleşmedi")


def make_flow(path, trigger_idle="bekle (sonraki bloklar çalışmaz)"):
    g = Graph()
    src = g.add_new("TestPattern", width=160, height=120, animate=True)
    num = g.add_new("Number", value=5.0)
    trig = g.add_new("ApiTrigger", name="parti", idle=trigger_idle)
    var = g.add_new("GetVariable", name="api_sayac")
    gray = g.add_new("Grayscale") if "Grayscale" in registry.NODE_REGISTRY else None
    if gray is not None:
        g.connect(src.id, "image", gray.id, "image")
    g.recipes = [{"name": "A", "values": {num.id: {"value": 1.0}}},
                 {"name": "B", "values": {num.id: {"value": 2.0}}}]
    g.save(str(path))
    return src.id, num.id, trig.id, var.id


@pytest.fixture
def headless(tmp_path, home):
    """A headless flow with the API on a free port; yields (base url, ids, controller, state)."""
    flow = tmp_path / "akis.json"
    ids = make_flow(flow)
    state = api_server.FlowState()
    ctl = api_server.HeadlessController(str(flow), state, fps=60, stats=1000)
    settings = {"port": 0, "key": KEY, "rate": 0}
    server = api_server.ApiServer(ctl, state, settings)
    url = server.start()
    stop = threading.Event()
    th = threading.Thread(target=ctl.serve, args=(stop.is_set,), daemon=True)
    th.start()
    wait(lambda: state.running and state.frames > 2)
    yield url, ids, ctl, state, server
    stop.set()
    th.join(15)
    server.stop()
    assert not th.is_alive()


# ------------------------------------------------------------------------------------------------ REST API
def test_status_values_and_blocks(headless):
    url, (src, num, trig, var), ctl, state, _ = headless
    code, d, _ = call(f"{url}/durum")
    assert code == 200 and d["ok"] and d["calisiyor"] is True and d["kare"] > 0 and d["akis"] == "akis.json"
    from visionflow import __version__
    assert d["surum"] == __version__ and "parti" in d["tetikler"] and d["hatalar"] == []
    code, d, _ = call(f"{url}/degerler")
    assert code == 200
    assert d["bloklar"][num]["cikislar"]["value"] == 5.0
    assert src not in d["bloklar"] or "image" not in d["bloklar"][src]["cikislar"]     # images excluded
    code, d, _ = call(f"{url}/degerler?blok={num}")
    assert list(d["bloklar"]) == [num]
    code, d, _ = call(f"{url}/degerler?blok=yok")
    assert code == 404 and "değeri yok" in d["hata"]
    code, d, _ = call(f"{url}/bloklar")
    by_id = {b["id"]: b for b in d["bloklar"]}
    assert by_id[src]["goruntu"] and by_id[trig]["tetik"] == "parti" and by_id[num]["baslik"] == "Sayı"


def test_auth_failures(headless):
    url = headless[0]
    code, d, h = call(f"{url}/durum", key=None)
    assert code == 401 and "eksik" in d["hata"] and d["ok"] is False and "X-API-Key" in h["WWW-Authenticate"]
    code, d, _ = call(f"{url}/durum", key="yanlis-anahtar-yanlis-anahtar")
    assert code == 401 and "yanlış" in d["hata"]
    code, d, _ = call(f"{url}/baslat", "POST", key="x")
    assert code == 401
    code, d, _ = call(f"{url}/durum", key=None, headers={"Authorization": f"Bearer {KEY}"})
    assert code == 200
    code, d, _ = call(f"{url}/openapi.json", key=None)                # public description
    assert code == 200 and d["openapi"].startswith("3.") and "/tetik/{ad}" in d["paths"]
    assert d["components"]["securitySchemes"]["ApiKey"]["name"] == "X-API-Key"


def test_bad_requests(headless):
    url = headless[0]
    code, d, _ = call(f"{url}/olmayan")
    assert code == 404 and "adres yok" in d["hata"]
    code, d, _ = call(url.replace("/api/v1", "/baska"))
    assert code == 404
    code, d, _ = call(f"{url}/durum", "POST", body={})
    assert code == 405 and "GET" in d["hata"]
    code, d, _ = call(f"{url}/baslat")
    assert code == 405
    code, d, _ = call(f"{url}/recete", "POST", body=b"{bozuk")
    assert code == 400 and "JSON" in d["hata"]
    code, d, _ = call(f"{url}/recete", "POST", body={"ad": ""})
    assert code == 400 and "Reçete adı eksik" in d["hata"]
    code, d, _ = call(f"{url}/recete", "POST", body={"ad": True})
    assert code == 400
    code, d, _ = call(f"{url}/degisken", "POST", body=b'{"ad": "x", "deger": NaN}')
    assert code == 400
    code, d, _ = call(f"{url}/goruntu")
    assert code == 400


def test_start_stop(headless):
    url, _, ctl, state, _ = headless
    code, d, _ = call(f"{url}/durdur", "POST")
    assert code == 200 and d["calisiyor"] is False and "durduruldu" in d["mesaj"]
    frames = state.frames
    time.sleep(0.3)
    assert call(f"{url}/durum")[1]["calisiyor"] is False and state.frames == frames
    code, d, _ = call(f"{url}/durdur", "POST")
    assert code == 200 and "zaten" in d["mesaj"]
    code, d, _ = call(f"{url}/baslat", "POST")
    assert code == 200 and d["calisiyor"] is True
    wait(lambda: call(f"{url}/durum")[1]["kare"] > 2)
    code, d, _ = call(f"{url}/baslat", "POST")
    assert "zaten" in d["mesaj"]


def test_recipe_change_while_running_and_stopped(headless):
    url, (src, num, trig, var), ctl, state, _ = headless
    code, d, _ = call(f"{url}/recete")
    assert d["receteler"] == ["A", "B"]
    code, d, _ = call(f"{url}/recete", "POST", body={"ad": "B"})
    assert code == 200 and d["recete"] == "B"
    wait(lambda: call(f"{url}/degerler")[1]["bloklar"][num]["cikislar"]["value"] == 2.0)
    assert call(f"{url}/durum")[1]["recete"] == "B"
    code, d, _ = call(f"{url}/recete", "POST", body={"ad": "Yok böyle"})
    assert code == 400 and d["ok"] is False
    code, d, _ = call(f"{url}/recete", "POST", body={"ad": 1})          # by number
    assert d["recete"] == "A"
    wait(lambda: call(f"{url}/degerler")[1]["bloklar"][num]["cikislar"]["value"] == 1.0)
    # stopped: remembered for the next start
    call(f"{url}/durdur", "POST")
    assert call(f"{url}/recete", "POST", body={"ad": "B"})[0] == 200
    call(f"{url}/baslat", "POST")
    wait(lambda: call(f"{url}/degerler")[1]["bloklar"].get(num, {}).get("cikislar", {}).get("value") == 2.0)


def test_trigger_block(headless):
    url, (src, num, trig, var), ctl, state, _ = headless
    code, d, _ = call(f"{url}/degerler")
    assert trig not in d["bloklar"]                                   # waiting: no request yet
    code, d, _ = call(f"{url}/tetik/parti", "POST", body={"seri": "A123", "adet": 3})
    assert code == 200 and d["ok"] and d["kuyrukta"] >= 1
    out = wait(lambda: call(f"{url}/degerler")[1]["bloklar"].get(trig))["cikislar"]
    assert out["fired"] is True and out["data"] == {"seri": "A123", "adet": 3} and out["count"] == 1
    assert json.loads(out["text"]) == {"seri": "A123", "adet": 3}
    call(f"{url}/tetik/parti", "POST", body={"text": "merhaba"})
    out = wait(lambda: (lambda o: o if o["count"] == 2 else None)(
        call(f"{url}/degerler")[1]["bloklar"][trig]["cikislar"]))
    assert out["text"] == "merhaba"
    code, d, _ = call(f"{url}/tetik/parti", "POST")                    # no body: data None
    assert code == 200
    code, d, _ = call(f"{url}/tetik/yok", "POST", body={})
    assert code == 404 and "API Tetik" in d["hata"] and "parti" in d["hata"]
    code, d, _ = call(f"{url}/tetik/", "POST", body={})
    assert code == 400


def test_trigger_queue_full_and_unregistered():
    hub = api_server.TriggerHub()
    with pytest.raises(KeyError):
        hub.push("a", 1)
    hub.register("a")
    hub.register("a")
    for i in range(api_server.QUEUE_MAX):
        hub.push("a", i)
    with pytest.raises(OverflowError):
        hub.push("a", 0)
    assert hub.pop("a") == (True, 0)
    hub.unregister("a")
    assert hub.names() == ["a"]                       # a second block with the same name still runs
    hub.unregister("a")
    assert hub.names() == [] and hub.pop("a") == (False, None)


def test_trigger_block_idle_modes():
    from visionflow.core import NoNewFrame
    n = registry.create("ApiTrigger")
    n.values.update(name="kip_testi")
    n.setup()
    try:
        with pytest.raises(NoNewFrame):
            n.process({})
        n.values["idle"] = "yanlış ver (akış devam eder)"
        assert n.process({})["fired"] is False
        api_server.TRIGGERS.push("kip_testi", "düz metin")
        api_server.TRIGGERS.push("kip_testi", [1, 2])
        o = n.process({})
        assert o["fired"] and o["text"] == "düz metin" and o["waiting"] == 1 and o["count"] == 1
        assert n.process({})["data"] == [1, 2]
    finally:
        n.teardown()
    assert "kip_testi" not in api_server.TRIGGERS.names()
    for bad in ("", "a b", "x" * 65, "ç/ğ"):
        m = registry.create("ApiTrigger")
        m.values["name"] = bad
        with pytest.raises(ValueError, match="Tetik adı"):
            m.setup()


def test_image_endpoint_returns_jpeg(headless):
    url, (src, *_), ctl, state, _ = headless
    code, data, h = call(f"{url}/goruntu/{src}", raw=True)
    assert code == 200 and h["Content-Type"] == "image/jpeg" and data[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (120, 160, 3)
    code, data, _ = call(f"{url}/goruntu/{src}?genislik=80&kalite=30", raw=True)
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[1] == 80
    code, d, _ = call(f"{url}/goruntu/{src}?genislik=abc")
    assert code == 400 and "tamsayı" in d["hata"]
    code, d, _ = call(f"{url}/goruntu/yok")
    assert code == 404 and "görüntüsü yok" in d["hata"]


@pytest.mark.parametrize("img", [
    np.zeros((10, 12), np.uint8), np.full((10, 12, 4), 200, np.uint8), np.ones((10, 12), bool),
    np.arange(120, dtype=np.uint16).reshape(10, 12) * 500, np.random.rand(10, 12, 3).astype(np.float32),
    np.full((10, 12, 2), 7, np.uint8), np.full((10, 12), np.nan, np.float64)])
def test_to_jpeg_any_picture(img):
    data = api_server.to_jpeg(img)
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED).shape[:2] == (10, 12)


def test_variables(headless):
    url, (src, num, trig, var), ctl, state, _ = headless
    code, d, _ = call(f"{url}/degisken", "POST", body={"ad": "api_sayac", "deger": 41})
    assert code == 200 and d["deger"] == 41
    out = wait(lambda: call(f"{url}/degerler")[1]["bloklar"][var]["cikislar"]
               if call(f"{url}/degerler")[1]["bloklar"][var]["cikislar"].get("number") == 41.0 else None)
    assert out["value"] == 41
    code, d, _ = call(f"{url}/degisken")
    assert d["degiskenler"]["api_sayac"] == 41
    code, d, _ = call(f"{url}/degisken/api_sayac")
    assert d["deger"] == 41
    code, d, _ = call(f"{url}/degisken/olmayan")
    assert code == 404
    code, d, _ = call(f"{url}/degisken", "POST", body={"deger": 1})
    assert code == 400 and "adı eksik" in d["hata"]
    code, d, _ = call(f"{url}/degisken", "POST", body={"ad": "x"})
    assert code == 400
    code, d, _ = call(f"{url}/degisken", "POST", body=[1])
    assert code == 400
    code, d, _ = call(f"{url}/degisken", "POST", body={"ad": "kalici_api", "deger": "metin", "kalici": True})
    assert code == 200
    from visionflow.nodes.connect import _load_persisted
    assert _load_persisted()["kalici_api"] == "metin"
    # images and other objects in variables are shown, not sent as arrays
    from visionflow.nodes import connect
    connect.VARS["resim_api"] = np.zeros((4, 5, 3), np.uint8)
    assert call(f"{url}/degisken/resim_api")[1]["deger"] == "<görüntü 4x5x3>"
    connect.VARS.pop("resim_api")


def test_rate_limit(home):
    state = api_server.FlowState()
    server = api_server.ApiServer(_Dummy(), state, {"port": 0, "key": KEY, "rate": 5})
    url = server.start()
    try:
        codes = [call(f"{url}/durum")[0] for _ in range(20)]
        assert codes.count(429) >= 10 and codes[0] == 200
        code, d, h = call(f"{url}/durum")
        assert code == 429 and "Çok fazla" in d["hata"] and h["Retry-After"] == "1"
        time.sleep(0.5)                                     # refills: allowed again
        assert call(f"{url}/durum")[0] == 200
        # unauthenticated guessing is limited too
        assert 429 in [call(f"{url}/durum", key="tahmin")[0] for _ in range(20)]
    finally:
        server.stop()


class _Dummy:
    def flow_name(self):
        return "deneme"

    def start(self):
        raise ValueError("Akışta blok yok")

    def stop(self):
        return "Akış zaten durmuş"

    def set_recipe(self, wanted):
        raise ValueError("Reçete yok")

    def recipes(self):
        return []

    def blocks(self):
        return []


def test_settings_key_and_security_rules(home):
    s = api_server.load_settings()
    assert len(s["key"]) >= 24 and s["host"] == "127.0.0.1" and s["require_key"] and not s["enabled"]
    assert api_server.load_settings()["key"] == s["key"]                 # kept
    assert api_server.clean_settings({"host": "1.2.3.4", "port": "x", "rate": -5, "https": "evet"}) == \
        {**api_server.DEFAULTS}
    with pytest.raises(ValueError, match="Ağdan erişim"):
        api_server.ApiServer(_Dummy(), api_server.FlowState(),
                             {"host": "0.0.0.0", "require_key": False, "key": KEY}).start()
    with pytest.raises(ValueError, match="çok kısa"):
        api_server.ApiServer(_Dummy(), api_server.FlowState(), {"key": "kisa"}).start()
    with pytest.raises(ValueError, match="sertifika"):
        api_server.ApiServer(_Dummy(), api_server.FlowState(), {"key": KEY, "https": True}).start()
    with pytest.raises(ValueError, match="yüklenemedi"):
        api_server.ApiServer(_Dummy(), api_server.FlowState(), {"key": KEY, "https": True, "cert": "/yok.pem",
                                                                 "keyfile": "/yok.key"}).start()
    # a busy port: Turkish message
    a = api_server.ApiServer(_Dummy(), api_server.FlowState(), {"port": 0, "key": KEY})
    a.start()
    try:
        with pytest.raises(RuntimeError, match="portu açılamadı"):
            api_server.ApiServer(_Dummy(), api_server.FlowState(), {"port": a.port, "key": KEY}).start()
    finally:
        a.stop()


def test_controller_errors_and_without_key(home):
    server = api_server.ApiServer(_Dummy(), api_server.FlowState(), {"port": 0, "key": KEY, "require_key": False})
    url = server.start()
    try:
        code, d, _ = call(f"{url}/durum", key=None)
        assert code == 200 and d["akis"] == "deneme" and d["calisiyor"] is False
        code, d, _ = call(f"{url}/baslat", "POST", body={}, key=None)
        assert code == 400 and d["hata"] == "Akışta blok yok"
        # without a key: a form post (not JSON) or a foreign Host header is refused (browser pages)
        code, d, _ = call(f"{url}/durdur", "POST", key=None, headers={"Content-Type": "text/plain"})
        assert code == 403
        code, d, _ = call(f"{url}/durum", key=None, headers={"Host": "kotu.site:80"})
        assert code == 403
        code, d, _ = call(f"{url}/recete", "POST", body={"ad": "x"}, key=None)
        assert code == 400 and d["hata"] == "Reçete yok"
    finally:
        server.stop()


def test_https_with_self_signed_cert(home, tmp_path):
    pytest.importorskip("cryptography", reason="HTTPS testi için cryptography gerekli")
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    api_server.make_self_signed(cert, key, "127.0.0.1")
    server = api_server.ApiServer(_Dummy(), api_server.FlowState(),
                                  {"port": 0, "key": KEY, "https": True, "cert": str(cert), "keyfile": str(key)})
    url = server.start()
    assert url.startswith("https://127.0.0.1:")
    try:
        ctx = ssl.create_default_context(cafile=str(cert))
        code, d, _ = call(f"{url}/durum", ctx=ctx)
        assert code == 200 and d["akis"] == "deneme"
        with pytest.raises(urllib.error.URLError):          # unknown certificate: refused by a normal client
            call(f"{url}/durum", ctx=ssl.create_default_context())
        # plain HTTP to the HTTPS port does not get an answer and does not break the server
        with pytest.raises(Exception):
            call(url.replace("https://", "http://") + "/durum")
        assert call(f"{url}/durum", ctx=ctx)[0] == 200
    finally:
        server.stop()


def test_headless_cli_flag(tmp_path, home, monkeypatch, capsys):
    """`--api PORT` runs the flow with the API until the program is stopped (Ctrl+C / service stop)."""
    from visionflow import headless
    flow = tmp_path / "akis.json"
    make_flow(flow)
    started = {}
    real_start = api_server.ApiServer.start

    def spy(self):
        started["server"] = self
        return real_start(self)
    monkeypatch.setattr(api_server.ApiServer, "start", spy)
    import signal
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn))
    out = {}
    th = threading.Thread(target=lambda: out.update(code=headless.main([str(flow), "--api", "0", "--fps", "60",
                                                                       "--stats", "1000", "--api-anahtar-goster"])),
                          daemon=True)
    th.start()
    srv = wait(lambda: started.get("server") if started.get("server") and started["server"].running else None)
    url = srv.url
    key = api_server.load_settings()["key"]
    assert srv.settings["host"] == "127.0.0.1" and srv.settings["require_key"]
    wait(lambda: call(f"{url}/durum", key=key)[1]["kare"] > 1)
    assert call(f"{url}/durum")[0] == 401
    assert call(f"{url}/durdur", "POST", key=key)[1]["calisiyor"] is False
    time.sleep(0.3)
    assert th.is_alive()                               # keeps serving after the flow stopped
    assert call(f"{url}/baslat", "POST", key=key)[1]["calisiyor"] is True
    handlers[signal.SIGINT]()                          # Ctrl+C
    th.join(15)
    assert not th.is_alive() and out["code"] == 0 and not srv.running
    printed = capsys.readouterr().out
    assert "Uzaktan kontrol API'si: http://127.0.0.1:" in printed and f"API anahtarı: {key}" in printed


def test_headless_api_bad_port(tmp_path, home, capsys):
    from visionflow import headless
    flow = tmp_path / "akis.json"
    make_flow(flow)
    assert headless.main([str(flow), "--api", "70000"]) == 2
    assert "0-65535" in capsys.readouterr().err


def test_headless_start_failure_reported(tmp_path, home):
    bad = tmp_path / "bozuk.json"
    bad.write_text("{bozuk", encoding="utf-8")
    state = api_server.FlowState()
    ctl = api_server.HeadlessController(str(bad), state, autostart=False)
    stop = threading.Event()
    th = threading.Thread(target=ctl.serve, args=(stop.is_set,), daemon=True)
    th.start()
    try:
        with pytest.raises(ValueError, match="başlatılamadı"):
            ctl.start()
        assert state.status()["hata_sayisi"] >= 1 and "çöktü" in state.status()["son_hatalar"][-1]["hata"]
    finally:
        stop.set()
        th.join(5)


def test_flowstate_publish_keeps_errors_and_plain_values():
    from visionflow.core.graph import NodeResult
    g = Graph()
    a = g.add_new("Number", value=1.0)
    b = g.add_new("Number", value=2.0)
    st = api_server.FlowState()
    st.attach()
    st.publish(g, {a.id: NodeResult({"value": np.float32(1.5)}, 1.0),
                   b.id: NodeResult({}, 1.0, "ValueError: kötü")})
    st.publish(g, {a.id: NodeResult({"value": float("nan")}, 1.0), b.id: NodeResult({}, 1.0, "ValueError: kötü")})
    s = st.status()
    assert s["hata_sayisi"] == 1 and s["hatalar"][0]["blok"] == b.id and s["kare"] == 2      # counted once
    assert st.block_values()[a.id]["cikislar"] == {}                                         # NaN left out
    st.publish(g, {b.id: NodeResult({"value": 2.0}, 1.0)})
    assert st.status()["hatalar"] == []
    assert api_server.plain(np.int64(3)) == 3 and api_server.plain([np.float64(1.5), 2]) == [1.5, 2]
    assert api_server.plain({"a": [1, "x"]}) == {"a": [1, "x"]} and api_server.plain(object()) is None
    assert api_server.plain([np.zeros(3)]) is None and api_server.plain([float("inf")]) is None
    st.detach()
    assert st.status()["calisiyor"] is False


def test_curl_examples():
    text = api_server.curl_examples("https://x:1/api/v1", "K", https=True)
    assert "-k" in text and "X-API-Key: K" in text and "/tetik/" in text and text.count("curl") == 9


# ------------------------------------------------------------------------------------------------ editor
@pytest.fixture
def win(home):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    w.stop()
    w.close()
    app.processEvents()


def _pump_call(app, *args, **kw):
    """An HTTP request from another thread while the Qt event loop runs (start/stop go to the GUI thread)."""
    box = {}
    t = threading.Thread(target=lambda: box.update(r=call(*args, **kw)), daemon=True)
    t.start()
    end = time.monotonic() + 30
    while t.is_alive() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    t.join(1)
    return box["r"]


def test_dialog_opens_and_controls_editor(win, home, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance()
    g = Graph()
    src = g.add_new("TestPattern", width=120, height=90, animate=True)
    num = g.add_new("Number", value=3.0)
    g.add_new("ApiTrigger", name="editor_tetik")
    g.recipes = [{"name": "X", "values": {num.id: {"value": 7.0}}}]
    win.scene.load_graph(g)
    win._loaded(None)
    dlg = win.open_api()
    assert dlg.isVisible() and not dlg.mgr.running and len(dlg.key.text()) >= 24
    assert dlg.key.echoMode() == dlg.key.EchoMode.Password
    key = dlg.key.text()
    assert key in dlg.examples.toPlainText() and "127.0.0.1" in dlg.examples.toPlainText()
    dlg.enabled.setChecked(True)
    dlg.port.setValue(_free_port())
    assert dlg.apply() and dlg.mgr.running
    url = dlg.mgr.server.url
    assert url in dlg.status.text() and api_server.load_settings()["enabled"] is True
    code, d, _ = _pump_call(app, f"{url}/baslat", "POST", key=key)
    assert code == 200 and win.worker is not None and win.worker.api is dlg.mgr.state
    end = time.monotonic() + 15
    while dlg.mgr.state.frames < 3 and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)
    code, d, _ = call(f"{url}/degerler", key=key)
    assert d["bloklar"][num.id]["cikislar"]["value"] == 3.0
    code, data, _ = call(f"{url}/goruntu/{src.id}", key=key, raw=True)
    assert code == 200 and data[:2] == b"\xff\xd8"
    assert call(f"{url}/tetik/editor_tetik", "POST", body={"a": 1}, key=key)[0] == 200
    code, d, _ = _pump_call(app, f"{url}/recete", "POST", body={"ad": "X"}, key=key)
    assert code == 200 and d["recete"] == "X" and win.root.nodes[num.id].values["value"] == 7.0
    code, d, _ = _pump_call(app, f"{url}/bloklar", key=key)
    assert {b["id"] for b in d["bloklar"]} >= {src.id, num.id}
    code, d, _ = _pump_call(app, f"{url}/durdur", "POST", key=key)
    assert code == 200 and win.worker is None and d["calisiyor"] is False
    # new key: the old one stops working
    dlg.regenerate(confirm=False)
    assert dlg.key.text() != key and dlg.mgr.running
    url = dlg.mgr.server.url
    assert call(f"{url}/durum", key=key)[0] == 401
    assert _pump_call(app, f"{url}/durum", key=dlg.key.text())[0] == 200
    # switched off
    dlg.enabled.setChecked(False)
    assert dlg.apply() and not dlg.mgr.running
    with pytest.raises(Exception):
        call(f"{url}/durum", key=dlg.key.text())
    # a setting that cannot work: warned, not left on
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    dlg.enabled.setChecked(True)
    dlg.host.setCurrentIndex(dlg.host.findData("0.0.0.0"))
    dlg.require.setChecked(False)
    assert not dlg.apply() and shown and "anahtarı zorunlu" in shown[0]
    assert api_server.load_settings()["enabled"] is False
    dlg.close()


def test_dialog_makes_https_cert(win, home):
    pytest.importorskip("cryptography")
    dlg = win.open_api()
    assert dlg.make_cert("127.0.0.1")
    assert dlg.https.isChecked() and dlg.cert.text().endswith("sertifika.pem")
    dlg.enabled.setChecked(True)
    dlg.port.setValue(_free_port())
    assert dlg.apply() and dlg.mgr.server.url.startswith("https://")
    assert "curl -k " in dlg.examples.toPlainText()
    dlg.mgr.stop()
    dlg.close()


def test_editor_autostart(home):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = api_server.load_settings()
    s.update(enabled=True, port=_free_port())
    api_server.save_settings(s)
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    try:
        end = time.monotonic() + 5
        while getattr(w, "_api_manager", None) is None and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert w._api_manager.running
        port = w._api_manager.server.port
    finally:
        w.close()
        app.processEvents()
    assert not w._api_manager.running              # closing the window frees the port
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------------------------------------ OPC UA security
def _need_opc_crypto():
    pytest.importorskip("asyncua", reason="asyncua kurulu değil")
    pytest.importorskip("cryptography", reason="OPC UA güvenliği için cryptography gerekli")


def test_opcua_secure_round_trip(home):
    _need_opc_crypto()
    from visionflow import opcua_security as sec
    port = _free_port()
    srv = registry.create("OpcUaServerRead")
    srv.values.update(port=port, name="Komut", vtype="Tamsayı (Int32)", security=sec.SEC_ENCRYPT, user="op",
                      password="gizli")
    pub = registry.create("OpcUaServerPublish")
    pub.values.update(port=port, name="Sonuc", vtype="Ondalık (Double)", security=sec.SEC_ENCRYPT, user="op",
                      password="gizli")
    srv.setup()
    pub.setup()
    try:
        url = f"opc.tcp://127.0.0.1:{port}/visionflow/"
        w = registry.create("OpcUaWrite")
        w.values.update(url=url, node="ns=2;s=VisionFlow.Komut", security=sec.SEC_ENCRYPT, user="op", password="gizli")
        w.setup()
        try:
            from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
            from asyncua import ua
            policy = w.state["client"].aio_obj.security_policy
            assert isinstance(policy, SecurityPolicyBasic256Sha256)
            assert policy.Mode == ua.MessageSecurityMode.SignAndEncrypt
            assert w.process({"value": 42}) == {"ok": True}
            assert srv.process({})["value"] == 42
        finally:
            w.teardown()
        pub.process({"value": 2.5})
        r = registry.create("OpcUaRead")
        r.values.update(url=url, node="ns=2;s=VisionFlow.Sonuc", security=sec.SEC_ENCRYPT, user="op", password="gizli")
        r.setup()
        try:
            assert r.process({})["number"] == 2.5
        finally:
            r.teardown()
        # first use: both certificates are now trusted
        trusted = {p.name for p in (sec.pki_dir() / "trusted").iterdir()}
        assert any("istemci" in n for n in trusted) and any("sunucu" in n for n in trusted)
        assert (sec.pki_dir() / "own" / "server_key.pem").is_file()
        # a client without security cannot even see an endpoint: Turkish explanation
        plain = registry.create("OpcUaRead")
        plain.values.update(url=url, node="ns=2;s=VisionFlow.Komut", user="op", password="gizli")
        with pytest.raises(RuntimeError, match="güvenlik ayarını sunmuyor"):
            plain.setup()
        # wrong password still refused over an encrypted channel
        bad = registry.create("OpcUaRead")
        bad.values.update(url=url, node="ns=2;s=VisionFlow.Komut", security=sec.SEC_ENCRYPT, user="op",
                          password="yanlis")
        with pytest.raises(Exception):
            bad.setup()
        # a server block on the same port with other security settings is refused
        other = registry.create("OpcUaServerPublish")
        other.values.update(port=port, name="Baska", user="op", password="gizli")
        with pytest.raises(ValueError, match="güvenlik"):
            other.setup()
    finally:
        pub.teardown()
        srv.teardown()


def test_opcua_sign_only_and_strict_trust(home):
    _need_opc_crypto()
    from visionflow import opcua_security as sec
    port = _free_port()
    srv = registry.create("OpcUaServerRead")
    srv.values.update(port=port, name="Komut", vtype="Tamsayı (Int32)", security=sec.SEC_SIGN, trust=sec.TRUST_STRICT)
    srv.setup()
    try:
        url = f"opc.tcp://127.0.0.1:{port}/visionflow/"
        c = registry.create("OpcUaWrite")
        c.values.update(url=url, node="ns=2;s=VisionFlow.Komut", security=sec.SEC_SIGN)
        with pytest.raises(RuntimeError, match="güvenilmedi|güvenilenler"):
            c.setup()                         # strict server: the client's certificate is not trusted yet
        rejected = list((sec.pki_dir() / "rejected").glob("*istemci*.der"))
        assert rejected
        trusted = sec.pki_dir() / "trusted"
        rejected[0].replace(trusted / rejected[0].name)                # the user moves it: now accepted
        c.setup()
        try:
            from asyncua import ua
            assert c.state["client"].aio_obj.security_policy.Mode == ua.MessageSecurityMode.Sign
            assert c.process({"value": 7}) == {"ok": True}
            assert srv.process({})["value"] == 7
        finally:
            c.teardown()
    finally:
        srv.teardown()


def test_opcua_trust_first_use_and_changed_certificate(home):
    _need_opc_crypto()
    from visionflow import opcua_security as sec
    from cryptography import x509
    first = x509.load_der_x509_certificate(_another_cert(home, "urn:test:plc"))
    assert sec.check_trust(first, "", sec.TRUST_FIRST, "Sunucu") == "yeni güvenildi"
    assert sec.check_trust(first, "", sec.TRUST_FIRST, "Sunucu") == "güvenilen"
    assert sec.check_trust(first, "", sec.TRUST_STRICT, "Sunucu") == "güvenilen"
    # same application URI, another certificate (another key): possibly an impostor
    changed = x509.load_der_x509_certificate(_another_cert(home, "urn:test:plc"))
    with pytest.raises(PermissionError, match="değişmiş"):
        sec.check_trust(changed, "", sec.TRUST_FIRST, "Sunucu")
    # strict: an unknown certificate is refused and kept in rejected/ for the user
    unknown = x509.load_der_x509_certificate(_another_cert(home, "urn:test:yeni"))
    with pytest.raises(PermissionError, match="güvenilenler listesinde yok"):
        sec.check_trust(unknown, "", sec.TRUST_STRICT, "Sunucu")
    assert len(list((sec.pki_dir() / "rejected").glob("urn_test_*.der"))) == 2


def _another_cert(home, uri):
    import asyncio

    from asyncua.crypto import cert_gen
    from cryptography.x509.oid import ExtendedKeyUsageOID
    folder = home / "baska"
    folder.mkdir(parents=True, exist_ok=True)
    k, c = folder / "k.pem", folder / "c.der"
    for f in (k, c):
        if f.exists():
            f.unlink()
    asyncio.run(cert_gen.setup_self_signed_certificate(k, c, uri, "h", [ExtendedKeyUsageOID.CLIENT_AUTH], {}))
    return c.read_bytes()


def test_opcua_security_defaults_unchanged():
    """Old flows: no security, anonymous, same behaviour."""
    for t in ("OpcUaWrite", "OpcUaRead", "OpcUaServerPublish", "OpcUaServerRead"):
        n = registry.create(t)
        assert n.values["security"] == "Yok" and n.values["trust"] == "İlk bağlantıda güven"
        assert "security" in n.setup_params
    from visionflow import opcua_security as sec
    with pytest.raises(ValueError, match="Bilinmeyen"):
        sec.check_choice("AES")
    old = {"format": "visionflow", "nodes": [{"id": "o", "type": "OpcUaRead", "values": {"url": "opc.tcp://x:1"}}]}
    assert Graph.from_dict(old).nodes["o"].values["security"] == "Yok"
