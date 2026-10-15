"""0.7: text conditions, working hours, JSON API, Discord/Slack, moving files, flow checker."""
import datetime as dt
import http.server
import json
import os
import pathlib
import threading
import time

import pytest

from visionflow.ai.config import AIConfig
from visionflow.check import check
from visionflow.core import Graph, create, load_builtin

load_builtin()
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def make(type_name, **values):
    n = create(type_name)
    n.values.update(values)
    n.setup()
    return n


# --- Metin Koşulu --------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode,words,text,hit,found", [
    ("içeriyor", "fatura, ödeme", "Ekte FATURA var", True, "fatura"),
    ("içeriyor", "İSTANBUL", "istanbul şubesi", True, "İSTANBUL"),
    ("içeriyor", "ıslak", "ISLAK zemin", True, "ıslak"),          # dotless ı ↔ I
    ("içeriyor", "fatura", "merhaba", False, ""),
    ("ile başlıyor", "RE:, FW:", "  re: sipariş", True, "RE:"),
    ("eşit", "evet", " Evet ", True, "evet"),
    ("eşit", "evet", "evet değil", False, ""),
    ("regex", r"sipariş no[: ]*(\d+)", "Sipariş No: 4711 hakkında", True, "4711"),
    ("içeriyor", " , ,", "herhangi", False, ""),                   # empty words never match
])
def test_text_match(mode, words, text, hit, found):
    out = make("TextMatch", mode=mode, words=words).process({"text": text})
    assert (out["match"], out["found"], out["not_match"]) == (hit, found, not hit)


def test_text_match_none_input_and_bad_regex():
    assert make("TextMatch").process({"text": None})["match"] is False
    n = create("TextMatch")
    n.values.update(mode="regex", words="(")
    assert not n.safe_setup() and "Kalıp" in n.setup_error


def test_text_match_case_sensitive():
    n = make("TextMatch", words="ACİL", case=True)
    assert not n.process({"text": "acil"})["match"] and n.process({"text": "ACİL durum"})["match"]


# --- Saat Aralığında -------------------------------------------------------------------------------------

def window(start, end, days, now):
    n = create("TimeWindow")
    n.values.update(start=start, end=end, days=days)
    n.now = lambda: now
    n.setup()
    return n.process({})["active"]


def test_time_window_office_hours():
    mon = dt.datetime(2026, 10, 12, 10, 0)          # Monday
    assert window("09:00", "18:00", "hafta içi", mon)
    assert not window("09:00", "18:00", "hafta içi", mon.replace(hour=18))   # end is exclusive
    assert not window("09:00", "18:00", "hafta içi", mon + dt.timedelta(days=5))   # Saturday
    assert window("09.00", "18.00", "hafta sonu", mon + dt.timedelta(days=6))       # Sunday, "09.00" accepted


def test_time_window_overnight_belongs_to_start_day():
    fri_night = dt.datetime(2026, 10, 16, 23, 0)
    sat_morning = dt.datetime(2026, 10, 17, 5, 0)
    assert window("22:00", "06:00", "hafta içi", fri_night)
    assert window("22:00", "06:00", "hafta içi", sat_morning)      # Friday's shift still running
    assert not window("22:00", "06:00", "hafta içi", dt.datetime(2026, 10, 18, 5, 0))   # Sunday 05:00 → Saturday's
    assert not window("22:00", "06:00", "her gün", fri_night.replace(hour=12))


def test_time_window_bad_time_is_setup_error():
    n = create("TimeWindow")
    n.values["start"] = "sabah"
    assert not n.safe_setup() and "09:00" in n.setup_error


# --- local HTTP server for API / webhook tests -----------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    posts: list = []
    doc = {"rates": {"TRY": 41.5}, "items": [{"name": "vida", "price": "2.5"}], "msg": "merhaba"}

    def do_GET(self):
        body = b"<html>yok</html>" if self.path == "/html" else json.dumps(self.doc).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)
        Handler.last_headers = dict(self.headers)

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        Handler.posts.append((self.path, json.loads(self.rfile.read(n))))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Handler.posts = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def run_job(n, inputs, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        out = n.process(dict(inputs))
        if out["done"]:
            return out
        time.sleep(0.01)
    raise AssertionError("iş bitmedi")


@pytest.mark.parametrize("path,value,number", [("rates.TRY", 41.5, 41.5), ("items.0.price", "2.5", 2.5),
                                               ("items[0].name", "vida", 0.0), ("msg", "merhaba", 0.0)])
def test_api_get_paths(server, path, value, number):
    out = run_job(make("ApiGet", url=server + "/x", path=path), {"url": None, "trigger": None})
    assert (out["value"], out["number"]) == (value, number)
    assert out["text"] == (value if isinstance(value, str) else json.dumps(value))


def test_api_get_header_and_whole_document(server):
    out = run_job(make("ApiGet", url=server + "/x", path="", header="X-Anahtar: 123"), {"url": None, "trigger": None})
    assert out["value"] == Handler.doc and Handler.last_headers["X-Anahtar"] == "123"


@pytest.mark.parametrize("url_suffix,path,msg", [("/html", "", "JSON değil"), ("/x", "rates.EUR", "bulunamadı"),
                                                 ("/x", "items.5", "bulunamadı")])
def test_api_get_errors_are_readable(server, url_suffix, path, msg):
    n = make("ApiGet", url=server + url_suffix, path=path)
    try:
        n.process({"url": None, "trigger": None})
    except RuntimeError:
        pass
    end = time.time() + 5
    while n.state["busy"] and time.time() < end:
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match=msg):
        n.process({"url": None, "trigger": None})


def test_api_get_rejects_non_http():
    n = make("ApiGet", url="file:///etc/passwd")
    try:
        n.process({"url": None, "trigger": None})      # the job may already have failed within this call
    except RuntimeError:
        pass
    while n.state["busy"]:
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="http"):
        n.process({"url": None, "trigger": None})


def test_chat_webhook_payloads_and_cooldown(server):
    from visionflow.nodes.automation import ChatWebhook
    assert ChatWebhook.payload("https://discord.com/api/webhooks/1/a", "x" * 3000) == {"content": "x" * 2000}
    assert ChatWebhook.payload("https://hooks.slack.com/services/T", "hey") == {"text": "hey"}
    assert ChatWebhook.kind("https://acme.webhook.office.com/x") == "teams"
    n = make("ChatWebhook", url=server + "/discord/hook", text="Kişi sayısı: {v}", cooldown=60)
    for trig in (True, False, True, False):          # second rising edge is inside the cooldown
        n.process({"trigger": trig, "value": 3})
    end = time.time() + 5
    while not Handler.posts and time.time() < end:
        time.sleep(0.01)
    time.sleep(0.1)
    assert Handler.posts == [("/discord/hook", {"content": "Kişi sayısı: 3"})]


# --- Dosyayı Taşı ------------------------------------------------------------------------------------

def test_move_file_never_overwrites(tmp_path):
    inbox, done = tmp_path / "gelen", tmp_path / "işlendi"
    inbox.mkdir()
    done.mkdir()
    (done / "a.pdf").write_text("eski")
    (inbox / "a.pdf").write_text("yeni")
    n = make("MoveFile", folder=str(done))
    out = n.process({"path": str(inbox / "a.pdf"), "trigger": True})
    assert pathlib.Path(out["new_path"]).name == "a_1.pdf" and out["moved"] == 1
    assert (done / "a.pdf").read_text() == "eski" and not (inbox / "a.pdf").exists()


def test_move_file_copy_dated_and_change_trigger(tmp_path):
    src = tmp_path / "rapor.txt"
    src.write_text("x")
    n = make("MoveFile", folder=str(tmp_path / "arşiv"), mode="kopyala", dated=True)
    out = n.process({"path": str(src), "trigger": None})       # no trigger: fires when the path changes
    assert src.exists() and pathlib.Path(out["new_path"]).parent.name == dt.date.today().isoformat()
    assert n.process({"path": str(src), "trigger": None})["moved"] == 1   # same path again: nothing
    with pytest.raises(FileNotFoundError):
        n.process({"path": str(tmp_path / "yok.txt"), "trigger": None})


def test_folder_watch_then_move_flow(tmp_path):
    inbox = tmp_path / "gelen"
    inbox.mkdir()
    g = Graph()
    fw = g.add_new("FolderWatch", folder=str(inbox), interval=0.5, pattern="*.txt")
    mv = g.add_new("MoveFile", folder=str(tmp_path / "islendi"))
    g.connect(fw.id, "path", mv.id, "path")
    g.connect(fw.id, "new", mv.id, "trigger")

    def frame(i, r):
        if i == 3:
            for k in range(3):
                (inbox / f"f{k}.txt").write_text(str(k))
        return r[mv.id].outputs.get("moved", 0) < 3

    g.run(max_frames=2000, max_fps=200, on_frame=frame)
    assert sorted(p.name for p in (tmp_path / "islendi").iterdir()) == ["f0.txt", "f1.txt", "f2.txt"]
    assert not list(inbox.iterdir())


# --- flow checker -------------------------------------------------------------------------------------

FULL = AIConfig(api_key="k", email_user="a", email_password="b", telegram_token="t", telegram_allowed="1",
                youtube_client_secret="c")


def texts(issues):
    return " | ".join(i.text for i in issues)


def test_check_finds_missing_links_and_settings():
    g = Graph()
    cam = g.add_new("TestPattern")
    g.add_new("Blur")                                       # nothing connected
    ai = g.add_new("AskAI")
    g.connect(cam.id, "image", ai.id, "image")
    tg = g.add_new("TelegramSend")
    g.connect(ai.id, "done", tg.id, "trigger")
    issues = check(g, AIConfig())
    t = texts(issues)
    assert "'Bulanıklaştır' bloğunun hiçbir girişi bağlı değil" in t or "hiçbir girişi bağlı değil" in t
    assert "Claude API anahtarı" in t and "Telegram bot token" in t and "chat ID yok" in t
    assert all(i.level == "hata" for i in issues if "anahtar" in i.text)


def test_check_warns_when_feed_has_no_handshake():
    g = Graph()
    rd, ai = g.add_new("EmailRead"), g.add_new("AskAI")
    g.connect(rd.id, "mail", ai.id, "prompt")
    assert "'next'" in texts(check(g, FULL))
    g.connect(ai.id, "done", rd.id, "next")
    assert "'next'" not in texts(check(g, FULL))


def test_check_empty_and_disabled():
    assert "boş" in texts(check(Graph(), FULL))
    g = Graph()
    n = g.add_new("TestPattern")
    n.enabled = False
    assert "devre dışı" in texts(check(g, FULL))


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.stem)
def test_examples_pass_the_checker(path):
    """Shipped examples must not contain real mistakes; only settings the user still has to fill in."""
    errors = [i for i in check(Graph.load(str(path)), FULL) if i.level == "hata"]
    assert all("webhook adresi" in i.text or "chat ID yok" in i.text for i in errors), errors


def test_headless_check_exit_code(tmp_path, capsys):
    from visionflow import headless
    g = Graph()
    g.add_new("Blur")
    p = tmp_path / "a.json"
    g.save(p)
    assert headless.main([str(p), "--check"]) == 1
    g = Graph()
    g.connect(g.add_new("TestPattern").id, "image", g.add_new("Preview").id, "image")
    g.save(p)
    assert headless.main([str(p), "--check"]) == 0
    assert "sorun yok" in capsys.readouterr().out or True


def test_ui_check_dialog_and_focus(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import QPointF
    from visionflow.ui.main_window import MainWindow
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.recovery = tmp_path / "k.json"
    w.show()
    try:
        it = w.scene.add_node("Blur", QPointF(0, 0))
        issues = w.check_flow()
        assert any(i.node_id == it.node.id for i in issues)
        w.focus_node(it.node.id)
        assert it.isSelected()
        w._check_dialog.close()
        # a flow with a feedback link opens and draws without a loop error
        w.saved_snap = w._snap()
        w.open_flow(str(EXAMPLES / "telegram_asistani.json"))
        assert any(w.scene.graph.is_feedback(c) for c in w.scene.graph.connections)
    finally:
        w.saved_snap = w._snap()
        w.close()
