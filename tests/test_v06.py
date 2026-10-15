"""0.6: queued AI jobs, e-mail handshake, Telegram bot, web page, folder watch, scheduler, extraction, budget."""
import datetime as dt
import http.server
import json
import threading
import time
from types import SimpleNamespace

import pytest

from visionflow.ai import claude, mail
from visionflow.ai.config import AIConfig
from visionflow.core import Graph, create, load_builtin

load_builtin()


@pytest.fixture(autouse=True)
def ai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    AIConfig(api_key="test").save()


class FakeClaude:
    def __init__(self, fn, delay=0.0, usage=None):
        self.fn, self.delay, self.usage, self.calls = fn, delay, usage, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        time.sleep(self.delay)
        return SimpleNamespace(stop_reason="end_turn", usage=self.usage,
                               content=[SimpleNamespace(type="text", text=self.fn(kw))])


def user_text(kw):
    c = kw["messages"][-1]["content"]
    return c if isinstance(c, str) else c[-1]["text"]


def pump(node, inputs, until, timeout=10.0):
    end = time.time() + timeout
    out = None
    while time.time() < end:
        out = node.process(dict(inputs))
        if until(out):
            return out
        time.sleep(0.01)
    raise AssertionError(f"zaman aşımı: {out}")


# --- bug fixes --------------------------------------------------------------------------------------------

def test_requests_arriving_while_busy_are_queued_not_dropped(monkeypatch):
    fake = FakeClaude(lambda kw: "cevap: " + user_text(kw), delay=0.15)
    monkeypatch.setattr(claude, "_test_client", fake)
    node = create("AskAI")
    node.values["use_memory"] = False
    node.setup()
    answers = []
    for q in ("bir", "iki", "üç"):           # three prompts in three frames, while the first is still running
        out = node.process({"trigger": None, "prompt": q, "context": None, "image": None})
    end = time.time() + 5
    while len(answers) < 3 and time.time() < end:
        out = node.process({"trigger": None, "prompt": "üç", "context": None, "image": None})
        if out["done"]:
            answers.append(out["text"])
        time.sleep(0.01)
    assert answers == ["cevap: bir", "cevap: iki", "cevap: üç"]       # each result gets its own 'done' pulse


RAW = ("From: {n} <{a}>\r\nTo: ben@ornek.com\r\nSubject: Soru {i}\r\nMessage-ID: <m{i}@x>\r\n"
       "Content-Type: text/plain; charset=utf-8\r\n\r\nMerhaba, ben {n}.\r\n")


class FakeIMAP:
    boxes, appended = {}, []

    def __init__(self, host, timeout=None):
        pass

    def login(self, u, p):
        pass

    def select(self, f):
        return "OK", [b"1"]

    def uid(self, cmd, *args):
        if cmd == "search":
            return "OK", [b" ".join(self.boxes)]
        if cmd == "fetch":
            return "OK", [(b"x", self.boxes[args[0]])]
        return "OK", [b""]

    def list(self):
        return "OK", [b'(\\Drafts) "/" "Drafts"']

    def append(self, box, flags, date, data):
        self.appended.append(data)
        return "OK", [b""]

    def logout(self):
        pass


def test_email_replies_go_to_the_right_sender_with_handshake(monkeypatch):
    """Two mails arrive together. With 'next' wired, mail 2 waits until mail 1's reply is saved."""
    FakeIMAP.boxes = {b"1": RAW.format(n="Ali", a="ali@x.com", i=1).encode(),
                      b"2": RAW.format(n="Ayşe", a="ayse@x.com", i=2).encode(),
                      b"3": RAW.format(n="Robot", a="noreply@x.com", i=3).encode()}
    FakeIMAP.appended = []
    monkeypatch.setattr(mail, "IMAP_CLASS", FakeIMAP)
    c = AIConfig.load()
    c.email_user, c.email_password = "ben@ornek.com", "x"
    c.save()
    fake = FakeClaude(lambda kw: "Sayın " + user_text(kw).split("ben ")[1].split(".")[0], delay=0.2)
    monkeypatch.setattr(claude, "_test_client", fake)

    g = Graph.load(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "eposta_asistani.json"))
    by = {n.type_name: n for n in g.nodes.values()}
    by["AskAI"].values["use_memory"] = False
    assert any(cn.dst == by["EmailRead"].id and cn.dst_port == "next" for cn in g.connections)
    g.run(max_frames=1500, max_fps=200, on_frame=lambda i, r: len(FakeIMAP.appended) < 2)
    from email import message_from_bytes, policy
    sent = [message_from_bytes(d, policy=policy.default) for d in FakeIMAP.appended]
    pairs = {m["To"]: m.get_content().splitlines()[0] for m in sent}
    assert pairs == {"ali@x.com": "Sayın Ali", "ayse@x.com": "Sayın Ayşe"}     # noreply skipped


def test_email_seen_by_message_id_not_uid(monkeypatch):
    FakeIMAP.boxes = {b"5": RAW.format(n="Ali", a="ali@x.com", i=1).encode()}
    monkeypatch.setattr(mail, "IMAP_CLASS", FakeIMAP)
    c = AIConfig.load()
    c.email_user, c.email_password = "ben@ornek.com", "x"
    c.save()
    n = create("EmailRead")
    n.setup()
    assert pump(n, {}, lambda o: o["new"])["message_id"] == "<m1@x>"
    FakeIMAP.boxes = {b"99": FakeIMAP.boxes[b"5"]}              # server renumbered the same mail
    n2 = create("EmailRead")
    n2.setup()
    time.sleep(0.2)
    assert not any(n2.process({})["new"] for _ in range(30))


def test_folder_params_use_folder_picker():
    for t, name in [("ImageFolder", "folder"), ("MemoryAdd", "folder"), ("MakeVideo", "images"),
                    ("FolderWatch", "folder"), ("AnomalyDetector", "folder")]:
        assert create(t).param(name).kind == "folder"


# --- spending guard ------------------------------------------------------------------------------------------

def test_usage_recorded_and_daily_limit_blocks(monkeypatch):
    usage = SimpleNamespace(input_tokens=200_000, output_tokens=100_000, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0)
    fake = FakeClaude(lambda kw: "tamam", usage=usage)
    monkeypatch.setattr(claude, "_test_client", fake)
    c = AIConfig.load()
    c.daily_limit_usd = 4.0
    c.save()
    claude.ask("bir")                                  # 0.2M*$4 + 0.1M*$20 = $2.80
    u = claude.usage_today()
    assert u["requests"] == 1 and u["usd"] == pytest.approx(2.8)
    claude.ask("iki")                                  # passes the check (2.80 < 4), total 5.60
    with pytest.raises(claude.AIError, match="harcama sınırı"):
        claude.ask("üç")
    assert len(fake.calls) == 2
    c.daily_limit_usd = 0                              # 0 = no limit
    c.save()
    claude.ask("dört")
    assert len(fake.calls) == 3


# --- extraction --------------------------------------------------------------------------------------------

def test_extract_info_block_uses_json_schema(monkeypatch):
    fake = FakeClaude(lambda kw: json.dumps({"firma": "ABC Ltd", "tutar": "1.250,00 TL", "tarih": "12.10.2026"}))
    monkeypatch.setattr(claude, "_test_client", fake)
    n = create("ExtractInfo")
    n.setup()
    out = pump(n, {"text": "ABC Ltd faturası 1.250,00 TL, 12.10.2026", "trigger": None}, lambda o: o["done"])
    assert (out["a"], out["b"], out["c"]) == ("ABC Ltd", "1.250,00 TL", "12.10.2026")
    fmt = fake.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["required"] == ["firma", "tutar", "tarih"]
    assert fmt["schema"]["additionalProperties"] is False


# --- Telegram bot ------------------------------------------------------------------------------------------

class TelegramAPI(http.server.BaseHTTPRequestHandler):
    updates, sent = [], []

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        off = int(q.get("offset", ["0"])[0])
        res = [u for u in self.updates if u["update_id"] >= off]
        self._json({"ok": True, "result": res})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.sent.append((self.path, json.loads(body)))
        self._json({"ok": True, "result": {}})

    def _json(self, d):
        b = json.dumps(d).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


@pytest.fixture
def tg_server():
    TelegramAPI.updates = [
        {"update_id": 10, "message": {"text": "Merhaba bot", "chat": {"id": 111}, "from": {"first_name": "Ali"}}},
        {"update_id": 11, "message": {"text": "Ben yabancıyım", "chat": {"id": 999}, "from": {"first_name": "X"}}},
        {"update_id": 12, "message": {"text": "Saat kaç?", "chat": {"id": 111}, "from": {"first_name": "Ali"}}},
    ]
    TelegramAPI.sent = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TelegramAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = AIConfig.load()
    c.telegram_token, c.telegram_allowed = "TOK", "111"
    c.save()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_telegram_receive_only_allowed_chats(tg_server, capsys):
    n = create("TelegramReceive")
    n.values.update(api=tg_server, interval=1)
    n.setup()
    first = pump(n, {}, lambda o: o["new"])
    second = pump(n, {}, lambda o: o["new"])
    assert (first["text"], first["chat_id"], first["user"]) == ("Merhaba bot", "111", "Ali")
    assert second["text"] == "Saat kaç?"
    assert "999" in capsys.readouterr().out                    # stranger's ID is logged, message ignored
    n._poll()                                                   # handed-out updates are confirmed on the next poll
    assert n.state["offset"] == 13


def test_telegram_ai_bot_flow(tg_server, monkeypatch):
    fake = FakeClaude(lambda kw: "Cevap: " + user_text(kw), delay=0.05)
    monkeypatch.setattr(claude, "_test_client", fake)
    g = Graph()
    rx = g.add_new("TelegramReceive", api=tg_server, interval=1)
    ai = g.add_new("AskAI", use_memory=False)
    tx = g.add_new("TelegramSend", api=tg_server, text="{v}", cooldown=0)
    for a, ap, b, bp in [(rx, "text", ai, "prompt"), (rx, "new", ai, "trigger"), (ai, "text", tx, "value"),
                         (ai, "done", tx, "trigger"), (rx, "chat_id", tx, "chat_id"), (ai, "done", rx, "next")]:
        g.connect(a.id, ap, b.id, bp)
    g.run(max_frames=2000, max_fps=200, on_frame=lambda i, r: len(TelegramAPI.sent) < 2)
    assert [s[1] for s in TelegramAPI.sent] == [{"chat_id": "111", "text": "Cevap: Merhaba bot"},
                                                {"chat_id": "111", "text": "Cevap: Saat kaç?"}]
    assert all(p == "/botTOK/sendMessage" for p, _ in TelegramAPI.sent)


def test_telegram_send_truncates_long_text():
    n = create("TelegramSend")
    n.values.update(token="T", chat_id="1", text="{v}")
    n.setup()
    req = n.request("x" * 5000, None)
    assert len(json.loads(req.data)["text"]) <= 4096


# --- web page / folder / document / scheduler ----------------------------------------------------------------

def test_web_page_reads_text_and_detects_change():
    page = {"html": "<html><head><title>Fiyat</title><style>x{}</style></head><body><p>Ürün: 100 TL</p>"
                    "<script>alert(1)</script></body></html>"}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            b = page["html"].encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        n = create("WebPage")
        n.values.update(url=f"http://127.0.0.1:{srv.server_address[1]}/", interval=0.2)
        n.setup()
        out = pump(n, {"url": None, "trigger": None}, lambda o: o["done"])
        assert out["title"] == "Fiyat" and out["text"] == "Ürün: 100 TL" and out["changed"] is False
        page["html"] = page["html"].replace("100", "90")
        out = pump(n, {"url": None, "trigger": None}, lambda o: o["done"] and o["changed"])
        assert "90 TL" in out["text"]
        assert n.process({"url": None, "trigger": None})["changed"] is False     # one-frame pulse
    finally:
        srv.shutdown()


def test_folder_watch_waits_for_complete_files(tmp_path):
    (tmp_path / "eski.pdf").write_bytes(b"old")
    n = create("FolderWatch")
    n.values.update(folder=str(tmp_path), interval=0.5)
    n.setup()
    time.sleep(0.1)
    n.process({})
    (tmp_path / "yeni.txt").write_text("merhaba", encoding="utf-8")
    (tmp_path / "resim.gif").write_bytes(b"x")                   # not in the pattern
    out = pump(n, {}, lambda o: o["new"], timeout=5)
    assert out["name"] == "yeni.txt"
    time.sleep(1.2)
    assert not any(n.process({})["new"] for _ in range(5))       # old file and gif never reported
    doc = create("ReadDocument")
    doc.setup()
    assert doc.process({"path": out["path"]})["text"] == "merhaba"


def test_folder_watch_missing_folder_is_setup_error(tmp_path):
    n = create("FolderWatch")
    n.values["folder"] = str(tmp_path / "yok")
    assert not n.safe_setup() and "Klasör yok" in n.setup_error


def test_scheduler_fires_once_per_time_and_respects_days():
    n = create("Scheduler")
    n.values["times"] = "09:00, 18:30"
    clock = {"t": dt.datetime(2026, 10, 12, 8, 59)}                 # a Monday
    n.now = lambda: clock["t"]
    n.setup()
    assert n.process({})["tick"] is False and n.process({})["next"] == "sonraki: 12.10 09:00"
    clock["t"] = dt.datetime(2026, 10, 12, 9, 0, 5)
    assert n.process({})["tick"] is True and n.process({})["tick"] is False
    clock["t"] = dt.datetime(2026, 10, 12, 18, 31)
    assert n.process({})["tick"] is True
    clock["t"] = dt.datetime(2026, 10, 13, 9, 0)
    assert n.process({})["tick"] is True                         # next day again

    w = create("Scheduler")
    w.values.update(times="10:00", days="hafta sonu")
    clock2 = {"t": dt.datetime(2026, 10, 16, 11, 0)}             # Friday, after 10:00
    w.now = lambda: clock2["t"]
    w.setup()
    assert w.process({})["tick"] is False and w.process({})["next"] == "sonraki: 17.10 10:00"
    clock2["t"] = dt.datetime(2026, 10, 17, 10, 0)
    assert w.process({})["tick"] is True

    bad = create("Scheduler")
    bad.values["times"] = "25:99"
    assert not bad.safe_setup() and "anlaşılmadı" in bad.setup_error


def test_scheduler_does_not_fire_for_times_already_past_at_start():
    n = create("Scheduler")
    n.values["times"] = "09:00"
    n.now = lambda: dt.datetime(2026, 10, 12, 10, 0)
    n.setup()
    assert n.process({})["tick"] is False


def test_headless_defaults_to_30_fps():
    import visionflow.headless as h
    import inspect
    assert "default=30" in inspect.getsource(h)


def test_feedback_link_is_not_a_loop_but_real_loops_still_are():
    from visionflow.core.graph import GraphError
    g = Graph()
    rx = g.add_new("FolderWatch")
    ai = g.add_new("AskAI")
    g.connect(rx.id, "name", ai.id, "prompt")
    g.connect(ai.id, "done", rx.id, "next")           # back to a feedback input: allowed
    assert g.order().index(rx.id) < g.order().index(ai.id)
    a, b = g.add_new("PythonScript"), g.add_new("PythonScript")
    g.connect(a.id, "out", b.id, "a")
    with pytest.raises(GraphError):
        g.connect(b.id, "out", a.id, "a")              # an ordinary loop is still refused
    g2 = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert not g2.warnings and len(g2.connections) == len(g.connections)


def test_feedback_input_reads_previous_frame():
    g = Graph()
    fw, py = g.add_new("FolderWatch"), g.add_new("PythonScript", code="state['n'] = state.get('n', 0) + 1\nout = state['n'] == 1")
    seen = []
    orig = fw.process
    fw.process = lambda inputs: (seen.append(inputs["next"]), orig(inputs))[1]
    fw.setup = lambda: fw.state.update(queue=[], current=None, next_poll=1e18, polling=False, error=None,
                                       seen=set(), prev_next=False, held_since=0.0)
    g.connect(py.id, "out", fw.id, "next")
    g.setup()
    for _ in range(3):
        g.run_once()
    assert seen == [False, True, False]     # nothing yet, then the value the script gave on frame 1, ...


def test_feed_moves_on_if_last_step_never_finishes(tmp_path, monkeypatch):
    from visionflow.nodes import ai_hub
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    n = create("FolderWatch")
    n.values.update(folder=str(tmp_path), existing=True, interval=0.5, pattern="*.txt")
    n.setup()
    first = pump(n, {"next": False}, lambda o: o["new"])
    for _ in range(20):
        assert not n.process({"next": False})["new"]
    monkeypatch.setattr(ai_hub._Feed, "HOLD_TIMEOUT", 0.0)
    second = n.process({"next": False})
    assert second["new"] and second["name"] != first["name"]


def test_invoice_example_extracts_into_database(tmp_path, monkeypatch):
    import pathlib
    import sqlite3
    monkeypatch.chdir(tmp_path)
    (tmp_path / "faturalar").mkdir()
    fake = FakeClaude(lambda kw: json.dumps({"firma": "Acar Ltd", "tutar": "1.250 TL", "tarih": "03.10.2026"}))
    monkeypatch.setattr(claude, "_test_client", fake)
    g = Graph.load(str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "fatura_okuyucu.json"))
    by = {n.type_name: n for n in g.nodes.values()}
    by["FolderWatch"].values["interval"] = 0.5

    def frame(i, r):
        if i == 5:
            (tmp_path / "faturalar" / "f1.txt").write_text("FATURA Acar Ltd toplam 1.250 TL", encoding="utf-8")
        return r[by["SQLiteLogger"].id].outputs.get("rows", 0) < 1

    g.run(max_frames=1500, max_fps=200, on_frame=frame)
    rows = sqlite3.connect(tmp_path / "faturalar.db").execute("select firma, tutar, tarih from faturalar").fetchall()
    assert rows == [("Acar Ltd", "1.250 TL", "03.10.2026")]
    assert "Acar Ltd" in user_text(fake.calls[0])


def test_telegram_example_is_wired_for_replies():
    import pathlib
    g = Graph.load(str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "telegram_asistani.json"))
    by = {n.type_name: n for n in g.nodes.values()}
    links = {(g.nodes[c.src].type_name, c.src_port, g.nodes[c.dst].type_name, c.dst_port) for c in g.connections}
    assert ("AskAI", "done", "TelegramReceive", "next") in links
    assert ("TelegramReceive", "chat_id", "TelegramSend", "chat_id") in links
    assert by["TelegramSend"].values["cooldown"] == 0
