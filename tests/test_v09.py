"""0.9: shared variables and the incoming webhook."""
import json
import socket
import time
import urllib.error
import urllib.request

import pytest

from visionflow.ai.config import AIConfig
from visionflow.check import check
from visionflow.core import Graph, create, load_builtin
from visionflow.nodes import connect

load_builtin()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    connect.VARS.clear()


def make(t, **v):
    n = create(t)
    n.values.update(v)
    n.setup()
    return n


def test_variable_wireless_link_between_blocks():
    g = Graph()
    num = g.add_new("Number", value=7)
    sv = g.add_new("SetVariable", name="hedef")
    gv = g.add_new("GetVariable", name="hedef")
    g.connect(num.id, "value", sv.id, "value")
    g.setup()
    g.run_once()
    out = g.run_once()[gv.id].outputs            # no wire between them: read on the next frame at the latest
    assert out["value"] == 7 and out["number"] == 7.0 and out["text"] == "7"


def test_variable_add_mode_counts_and_persists():
    sv = make("SetVariable", name="parça", mode="ekle", initial="0", persist=True)
    for trig in (True, False, True, False, True):
        out = sv.process({"value": None, "trigger": trig})
    assert out["value"] == 3
    connect.VARS.clear()
    sv2 = make("SetVariable", name="parça", mode="ekle", initial="0", persist=True)   # restart
    assert sv2.process({"value": None, "trigger": False})["value"] == 3
    rs = make("SetVariable", name="parça", mode="sıfırla", initial="0", persist=True)
    rs.process({"value": None, "trigger": True})
    assert connect.VARS["parça"] == 0


def test_variable_add_numbers_and_text():
    sv = make("SetVariable", name="t", mode="ekle", initial="")
    sv.process({"value": "a", "trigger": None})
    sv.process({"value": "b", "trigger": None})
    assert connect.VARS["t"] == "ab"
    sv = make("SetVariable", name="f", mode="ekle", initial="1.5")
    sv.process({"value": 2.25, "trigger": None})
    assert connect.VARS["f"] == 3.75


def test_get_variable_changed_pulse_and_missing():
    gv = make("GetVariable", name="yok")
    assert gv.process({}) == {"value": None, "number": 0.0, "text": "", "changed": False}
    connect.VARS["yok"] = "x"
    assert gv.process({})["changed"] and not gv.process({})["changed"]


def test_checker_warns_about_unknown_variable_name():
    g = Graph()
    g.add_new("SetVariable", name="sayac")
    gv = g.add_new("GetVariable", name="sayaç")
    full = AIConfig(api_key="k")
    assert any(i.node_id == gv.id and "Değişken Yaz" in i.text for i in check(g, full))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(url, body=None, headers=None):
    req = urllib.request.Request(url, body, headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def pump(n, until, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        out = n.process({"next": None})
        if until(out):
            return out
        time.sleep(0.01)
    raise AssertionError("zaman aşımı")


def test_webhook_receives_json_form_and_query():
    port = free_port()
    n = make("WebhookReceive", port=port, token="t1")
    try:
        base = f"http://127.0.0.1:{port}"
        H = {"X-Token": "t1"}
        assert post(base + "/siparis", json.dumps({"mesaj": "Yeni sipariş", "adet": 3}).encode(),
                    {"Content-Type": "application/json", **H}) == (200, {"ok": True})
        post(base + "/form", "ad=Ali&not=merhaba".encode(), {"Content-Type": "application/x-www-form-urlencoded", **H})
        post(base + "/tetik?x=1&token=t1")
        a = pump(n, lambda o: o["new"])
        b = pump(n, lambda o: o["new"])
        c = pump(n, lambda o: o["new"])
        assert (a["text"], a["data"]["adet"], a["path"]) == ("Yeni sipariş", 3, "/siparis")
        assert b["data"] == {"ad": "Ali", "not": "merhaba"} and c["data"] == {"x": "1"}
    finally:
        n.teardown()


def test_webhook_token_and_restart_on_same_port():
    port = free_port()
    n = make("WebhookReceive", port=port, token="gizli")
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(f"http://127.0.0.1:{port}/", b"{}")
        assert e.value.code == 403
        assert post(f"http://127.0.0.1:{port}/?token=gizli", b"hi")[0] == 200
        assert post(f"http://127.0.0.1:{port}/", b"hi", {"X-Token": "gizli"})[0] == 200
        assert pump(n, lambda o: o["new"])["text"] == "hi"
    finally:
        n.teardown()
    n2 = make("WebhookReceive", port=port)        # stop + start again (editor re-run) must not fail
    n2.teardown()


def test_webhook_lan_requires_token_and_busy_port_is_clear():
    n = create("WebhookReceive")
    n.values.update(listen="yerel ağ", port=free_port())
    assert not n.safe_setup() and "token" in n.setup_error
    port = free_port()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        n = create("WebhookReceive")
        n.values["port"] = port
        assert not n.safe_setup() and "kullanılamıyor" in n.setup_error
    finally:
        blocker.close()


def test_webhook_drives_a_flow():
    port = free_port()
    g = Graph()
    wh = g.add_new("WebhookReceive", port=port)
    sv = g.add_new("SetVariable", name="gelen", mode="ekle")
    g.connect(wh.id, "new", sv.id, "trigger")
    sent = {"n": 0}

    def frame(i, r):
        if i in (3, 4, 5):
            post(f"http://127.0.0.1:{port}/x", b"a", {"X-VisionFlow": "1"})
            sent["n"] += 1
        return connect.VARS.get("gelen") != 3

    g.run(max_frames=2000, max_fps=200, on_frame=frame)
    assert connect.VARS["gelen"] == 3


def test_selftest_reports_everything(tmp_path):
    from visionflow import selftest
    ok, lines = selftest.run(str(tmp_path / "r.txt"), ui=False)
    text = (tmp_path / "r.txt").read_text(encoding="utf-8")
    assert ok and "SONUÇ: her şey çalışıyor" in text and "[TAMAM] Bloklar" in text and "Örnek akışlar" in text


def test_main_switches_route_to_tools(tmp_path, monkeypatch):
    import sys
    from visionflow import __main__ as entry
    g = Graph()
    g.connect(g.add_new("TestPattern").id, "image", g.add_new("Preview").id, "image")
    g.save(tmp_path / "a.json")
    monkeypatch.setattr(sys, "argv", ["VisionFlow", "--headless", str(tmp_path / "a.json"), "--check"])
    assert entry.main() == 0
    monkeypatch.setattr(sys, "argv", ["VisionFlow", "--selftest", "--no-ui", "--out", str(tmp_path / "s.txt")])
    assert entry.main() == 0 and (tmp_path / "s.txt").exists()
