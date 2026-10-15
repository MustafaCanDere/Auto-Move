"""1.0 bug hunt: each test fails on 0.9 code."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from visionflow.ai import claude, mail
from visionflow.ai.config import AIConfig
from visionflow.check import check
from visionflow.core import Graph, create, load_builtin
from visionflow.nodes import connect

load_builtin()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    AIConfig(api_key="test").save()
    connect.VARS.clear()


def make(t, **v):
    n = create(t)
    n.values.update(v)
    n.setup()
    return n


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def status(url, body=None, headers=None, method=None):
    req = urllib.request.Request(url, body, headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


# --- webhook --------------------------------------------------------------------------------------------------

def test_webhook_without_token_ignores_what_a_web_page_could_send():
    port = free_port()
    n = make("WebhookReceive", port=port)
    base = f"http://127.0.0.1:{port}"
    try:
        assert status(base + "/?text=sil") == 403                                   # <img src=...>
        assert status(base + "/", b"text=sil", {"Content-Type": "text/plain"}) == 403   # a plain <form>
        assert status(base + "/", b'{"a":1}', {"Content-Type": "application/json", "Host": "kotu.site:8765"}) == 403
        assert status(base + "/", b'{"text":"tamam"}', {"Content-Type": "application/json"}) == 200
        assert status(base + "/", b"x", {"X-VisionFlow": "1"}) == 200
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(b"POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: -1\r\n\r\n")
        assert b" 400 " in s.recv(200)                                              # 0.9: hung forever
        s.close()
        texts = []
        end = time.time() + 5
        while len(texts) < 2 and time.time() < end:
            out = n.process({"next": None})
            if out["new"]:
                texts.append(out["text"])
            time.sleep(0.01)
        assert texts == ["tamam", "x"]
    finally:
        n.teardown()


# --- variables ----------------------------------------------------------------------------------------------------

def test_variables_carry_images():
    img = np.zeros((4, 4, 3), np.uint8)
    sv, gv = make("SetVariable", name="kare"), make("GetVariable", name="kare")
    assert sv.process({"value": img, "trigger": None})["value"] is img
    assert gv.process({})["changed"] is False
    img2 = img.copy()
    img2[0, 0] = 255
    sv.process({"value": img2, "trigger": None})
    out = gv.process({})
    assert out["changed"] is True and out["value"] is img2


# --- worker: a slider on a source must not restart it ----------------------------------------------------------------

def test_changing_a_live_setting_of_a_source_does_not_restart_it(tmp_path):
    pytest.importorskip("PySide6")
    import cv2
    from visionflow.ui.worker import FlowWorker
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((4, 4, 3), i, np.uint8))
    g = Graph()
    f = g.add_new("ImageFolder", folder=str(tmp_path), delay=0)
    g.setup()
    for _ in range(3):
        g.run_once()
    w = FlowWorker(g.to_dict())
    w.set_param(f.id, "loop", True)
    w._apply_pending(g)
    assert g.run_once()[f.id].outputs["index"] == 4                  # 0.9: back to image 1
    w.set_param(f.id, "folder", str(tmp_path))
    w._apply_pending(g)
    assert g.run_once()[f.id].outputs["index"] == 1                  # a setting read at start still restarts


# --- queued messages survive a stop -----------------------------------------------------------------------------------

def test_telegram_queued_messages_come_back_after_restart(monkeypatch):
    updates = [{"update_id": i, "message": {"text": f"m{i}", "chat": {"id": 111}, "from": {"first_name": "A"}}}
               for i in (5, 6, 7)]
    confirmed = {"offset": 0}

    def fake_get(self, method, **params):
        confirmed["offset"] = max(confirmed["offset"], params.get("offset", 0))
        return [u for u in updates if u["update_id"] >= params.get("offset", 0)]
    Rx = type(create("TelegramReceive"))
    monkeypatch.setattr(Rx, "_get", fake_get)
    n = make("TelegramReceive", token="T", allowed="111")
    n._poll()
    assert n.process({"next": False})["text"] == "m5"                   # m6, m7 still queued
    n._poll()
    n._poll()
    assert confirmed["offset"] == 6                                      # 0.9: 8 → m6 and m7 gone for good
    n.teardown()
    n2 = make("TelegramReceive", token="T", allowed="111")             # flow stopped and started again
    n2._poll()
    assert [i["text"] for i in n2.state["queue"]] == ["m6", "m7"]      # m5 is not answered twice


def test_email_is_marked_read_only_when_handed_out(monkeypatch):
    m1 = SimpleNamespace(uid="1", message_id="<a@x>", sender="A", sender_addr="a@x", subject="s", date="", body="b")
    m2 = SimpleNamespace(uid="2", message_id="<b@x>", sender="B", sender_addr="b@x", subject="s", date="", body="b")
    calls = {"fetch_mark": [], "marked": []}
    monkeypatch.setattr(mail, "fetch_unseen", lambda *a, **k: (calls["fetch_mark"].append(a[4]), [m1, m2])[1])
    monkeypatch.setattr(mail, "mark_seen", lambda uids, *a, **k: calls["marked"].extend(uids))
    n = make("EmailRead", user="u", password="p", host="h", mark_seen=True)
    n._poll()
    assert calls["fetch_mark"] == [False]
    n.process({"next": False})
    end = time.time() + 2
    while not calls["marked"] and time.time() < end:
        time.sleep(0.01)
    assert calls["marked"] == ["1"]                                      # m2 stays unread on the server


def test_imap_has_a_timeout(monkeypatch):
    seen = {}

    class IM:
        def __init__(self, host, timeout=None):
            seen["timeout"] = timeout

        def login(self, *a):
            pass

        def select(self, *a):
            pass

        def uid(self, *a):
            return "OK", [b""]

        def logout(self):
            pass
    monkeypatch.setattr(mail, "IMAP_CLASS", IM)
    mail.fetch_unseen("u", "p", "h")
    assert seen["timeout"] and seen["timeout"] <= 60


def test_smtp_without_encryption_does_not_send_the_password():
    import smtplib

    class S:
        def starttls(self):
            raise smtplib.SMTPNotSupportedError()
    with pytest.raises(mail.MailError, match="STARTTLS"):
        mail.starttls_or_refuse(S(), "smtp.ornek.com")
    mail.starttls_or_refuse(S(), "localhost")                            # a relay on this computer is fine


# --- flows without a camera/timer ---------------------------------------------------------------------------------------

def test_flow_without_a_source_waits_for_the_ai_answer(monkeypatch, tmp_path):
    def create_(**kw):
        time.sleep(0.2)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="cevap")])
    monkeypatch.setattr(claude, "_test_client", SimpleNamespace(beta=SimpleNamespace(
        messages=SimpleNamespace(create=create_))))
    g = Graph()
    t = g.add_new("TextConst", text="soru")
    ai = g.add_new("AskAI", use_memory=False)
    g.connect(t.id, "text", ai.id, "prompt")
    done = []
    frames = g.run(on_frame=lambda i, r: done.append(r[ai.id].outputs.get("done")))
    assert any(done) and frames > 1                                     # 0.9: one frame, answer never seen


def test_headless_stops_when_the_only_source_fails(tmp_path):
    g = Graph()
    g.add_new("VideoFile", path=str(tmp_path / "yok.mp4"))
    t0 = time.time()
    assert g.run(max_fps=50) == 1 and time.time() - t0 < 5             # 0.9: ran forever


# --- recovery file, close, flow folder ------------------------------------------------------------------------------

def test_selftest_and_second_window_keep_the_recovery_file(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    from visionflow.ui.main_window import MainWindow, recovery_file
    rec = recovery_file()
    rec.parent.mkdir(parents=True, exist_ok=True)
    g = Graph()
    g.add_new("Number")
    rec.write_text(json.dumps({"path": str(tmp_path / "proje" / "a.json"), "graph": g.to_dict()}), encoding="utf-8")
    w = MainWindow()
    w.saved_snap = w._snap()
    w.close()
    assert rec.exists()                                                  # 0.9: deleted without asking
    w2 = MainWindow()
    assert w2.check_recovery(ask=False)
    from visionflow.core import node as core_node
    assert core_node.FLOW_DIR == (tmp_path / "proje").resolve()           # relative paths work after a restore
    w2.saved_snap = w2._snap()
    w2.close()
    assert not rec.exists()                                              # restored and closed: gone


# --- research with a paused turn -------------------------------------------------------------------------------------

def test_research_keeps_text_and_sources_from_before_a_pause(monkeypatch):
    res = lambda url: SimpleNamespace(type="web_search_result", url=url, title=url)  # noqa: E731
    replies = [
        SimpleNamespace(stop_reason="pause_turn", content=[
            SimpleNamespace(type="text", text="Birinci bölüm. "),
            SimpleNamespace(type="web_search_tool_result", content=[res("https://a.example")])]),
        SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="web_search_tool_result", content=[res("https://b.example")]),
            SimpleNamespace(type="text", text="İkinci bölüm.")]),
    ]
    sent = []

    def create_(**kw):
        sent.append(kw["messages"])
        return replies[len(sent) - 1]
    monkeypatch.setattr(claude, "_test_client", SimpleNamespace(beta=SimpleNamespace(
        messages=SimpleNamespace(create=create_))))
    report, sources = claude.research("konu")
    assert "Birinci bölüm" in report and "İkinci bölüm" in report
    assert [s["url"] for s in sources] == ["https://a.example", "https://b.example"]


# --- checker -----------------------------------------------------------------------------------------------------------

def test_checker_accepts_a_mail_account_set_on_the_block_and_knows_the_webhook():
    g = Graph()
    rx = g.add_new("EmailRead", user="ben@ornek.com", password="x")
    ai = g.add_new("AskAI")
    g.connect(rx.id, "body", ai.id, "prompt")
    issues = check(g, AIConfig(api_key="k"))
    assert not [i for i in issues if "e-posta hesabı" in i.text.lower()]
    g2 = Graph()
    wh = g2.add_new("WebhookReceive")
    ai2 = g2.add_new("AskAI")
    g2.connect(wh.id, "text", ai2.id, "prompt")
    assert any("next" in i.text and i.node_id == wh.id for i in check(g2, AIConfig(api_key="k")))


# --- video recorder ------------------------------------------------------------------------------------------------------

def test_video_recorder_reports_a_writer_that_did_not_open(tmp_path, monkeypatch):
    import cv2

    class W:
        def __init__(self, *a):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass
    monkeypatch.setattr(cv2, "VideoWriter", W)
    n = make("VideoRecorder", folder=str(tmp_path))
    with pytest.raises(RuntimeError, match="açılamadı"):
        n.process({"image": np.zeros((8, 8, 3), np.uint8), "record": True})


# --- spending limit: two programs at once -------------------------------------------------------------------------------

def test_spending_is_counted_from_two_programs_at_once(tmp_path):
    code = ("import sys; from types import SimpleNamespace as S; from visionflow.ai import claude\n"
            "u = S(input_tokens=1, output_tokens=1)\n"
            "for _ in range(40): claude._record_usage('claude-haiku-5-5', S(usage=u))\n")
    env = {**os.environ, "VISIONFLOW_HOME": os.environ["VISIONFLOW_HOME"]}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env, cwd=root) for _ in range(3)]
    assert all(p.wait(60) == 0 for p in procs)
    assert claude.usage_today()["requests"] == 120                       # 0.9: updates got lost
