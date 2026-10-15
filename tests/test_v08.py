"""0.8: RSS news feed, delay, daily counter, Excel logger."""
import datetime as dt
import http.server
import threading
import time

import pytest

from visionflow.core import Graph, create, load_builtin

load_builtin()

RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Haberler</title>
{items}</channel></rss>"""
ITEM = """<item><title>{t}</title><link>https://ornek.com/{i}</link><guid>h{i}</guid>
<description><![CDATA[<p>{d} &amp; <b>detay</b></p>]]></description></item>"""
ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Blog</title>
<entry><id>tag:a,1</id><title>Atom yazısı</title><link rel="alternate" href="https://blog/1"/>
<summary>Kısa özet</summary><updated>2026-10-12T10:00:00Z</updated></entry></feed>"""


class Feed(http.server.BaseHTTPRequestHandler):
    items = []

    def do_GET(self):
        body = ATOM if self.path == "/atom" else RSS.format(items="".join(
            ITEM.format(i=i, t=t, d=d) for i, t, d in reversed(Feed.items)))       # newest first, like sites do
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def feed():
    Feed.items = [(1, "Eski haber", "önceden"), (2, "Yapay zeka gelişmesi", "YZ")]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Feed)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def make(t, **v):
    n = create(t)
    n.values.update(v)
    n.setup()
    return n


def pump(n, inputs, until, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        out = n.process(dict(inputs))
        if until(out):
            return out
        time.sleep(0.01)
    raise AssertionError("zaman aşımı")


def test_parse_rss_and_atom():
    from visionflow.nodes.automation import parse_feed
    items = parse_feed(RSS.format(items=ITEM.format(i=7, t="Başlık &amp; Ek", d="Metin")).encode())
    assert items == [{"id": "h7", "title": "Başlık & Ek", "link": "https://ornek.com/7",
                      "summary": "Metin & detay", "date": ""}]
    a = parse_feed(ATOM.encode())[0]
    assert (a["id"], a["title"], a["link"], a["summary"]) == ("tag:a,1", "Atom yazısı", "https://blog/1", "Kısa özet")
    with pytest.raises(ValueError, match="RSS"):
        parse_feed(b"<html><body>degil")


def test_rss_gives_only_news_published_after_start(feed):
    n = make("RssFeed", url=feed + "/rss", interval=30)
    n.state["next_poll"] = 0
    pump(n, {"next": None}, lambda o: not n.state["first"] and not n.state["polling"])
    Feed.items.append((3, "Yeni ürün çıktı", "duyuru"))
    Feed.items.append((4, "İkinci haber", "x"))
    n.state["next_poll"] = 0
    a = pump(n, {"next": None}, lambda o: o["new"])
    b = pump(n, {"next": None}, lambda o: o["new"])
    assert (a["title"], b["title"]) == ("Yeni ürün çıktı", "İkinci haber")       # oldest first
    assert a["link"] == "https://ornek.com/3" and "duyuru" in a["text"]
    n2 = make("RssFeed", url=feed + "/rss", interval=30)                         # restart: nothing repeats
    n2.state["next_poll"] = 0
    pump(n2, {"next": None}, lambda o: not n2.state["polling"] and n2.state["next_poll"] > 0)
    time.sleep(0.2)
    assert not any(n2.process({"next": None})["new"] for _ in range(5))


def test_rss_existing_and_keyword_filter(feed):
    n = make("RssFeed", url=f"{feed}/rss, {feed}/atom", existing=True, words="YAPAY ZEKA, atom")
    got = []
    end = time.time() + 10
    while len(got) < 2 and time.time() < end:
        o = n.process({"next": None})
        if o["new"]:
            got.append(o["title"])
        time.sleep(0.01)
    assert sorted(got) == ["Atom yazısı", "Yapay zeka gelişmesi"]


def test_delay_keeps_every_trigger():
    n = make("Delay", seconds=0.2)
    outs, values = [], []
    t0 = time.monotonic()
    for i in range(6):                             # three rising edges in quick succession
        n.process({"trigger": i % 2 == 0, "value": i})
    while time.monotonic() - t0 < 1:
        o = n.process({"trigger": False, "value": None})
        if o["out"]:
            outs.append(time.monotonic() - t0)
            values.append(o["value"])
    assert values == [0, 2, 4] and outs[0] >= 0.19


def test_daily_counter_resets_and_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path))
    clock = {"t": dt.datetime(2026, 10, 12, 7, 0)}

    def new():
        n = create("DailyCounter")
        n.values.update(name="hat 1", reset_time="06:00")
        n.now = lambda: clock["t"]
        n.setup()
        return n

    n = new()
    for i in range(10):
        out = n.process({"trigger": i % 2 == 0, "reset": None})
    assert out["today"] == 5
    n = new()                                       # program restarted
    assert n.process({"trigger": False, "reset": None})["today"] == 5
    clock["t"] = dt.datetime(2026, 10, 13, 3, 0)   # before 06:00: still the same counting day
    assert n.process({"trigger": True, "reset": None})["today"] == 6
    clock["t"] = dt.datetime(2026, 10, 13, 6, 0)
    out = n.process({"trigger": False, "reset": None})
    assert (out["today"], out["yesterday"], out["total"]) == (0, 6, 6)
    clock["t"] = dt.datetime(2026, 10, 15, 9, 0)   # a day with no production in between
    assert n.process({"trigger": False, "reset": None})["yesterday"] == 0


def test_excel_logger_writes_rows_and_survives_a_locked_file(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "rapor" / "kayit.xlsx"
    n = make("ExcelLogger", path=str(p), names="parça,sonuç,not")
    n.SAVE_EVERY = n.RETRY_EVERY = 0
    n.process({"a": 1, "b": "sağlam", "c": "=HACK()", "trigger": True})
    n.process({"a": 1, "b": "sağlam", "c": "=HACK()", "trigger": False})
    real_save = openpyxl.Workbook.save
    monkeypatch.setattr(openpyxl.Workbook, "save", lambda *a: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(RuntimeError, match="1 satır"):
        n.process({"a": 2, "b": "hatalı", "c": None, "trigger": True})
    monkeypatch.setattr(openpyxl.Workbook, "save", real_save)
    n.process({"a": 2, "b": "hatalı", "c": None, "trigger": False})
    out = n.process({"a": 3, "b": "sağlam", "c": None, "trigger": True})
    assert out == {"rows": 3, "waiting": 0}
    ws = openpyxl.load_workbook(p)["Kayıtlar"]
    rows = list(ws.values)
    assert rows[0] == ("Zaman", "parça", "sonuç", "not")
    assert [r[1:] for r in rows[1:]] == [(1, "sağlam", "=HACK()"), (2, "hatalı", None), (3, "sağlam", None)]
    assert isinstance(rows[1][0], dt.datetime)


def test_excel_logger_without_trigger_writes_on_change(tmp_path):
    pytest.importorskip("openpyxl")
    n = make("ExcelLogger", path=str(tmp_path / "k.xlsx"))
    n.SAVE_EVERY = 0
    for v in (5, 5, 5, 6, 6, 5):
        out = n.process({"a": v, "b": None, "c": None, "trigger": None})
    assert out["rows"] == 3


def test_news_summary_example_is_wired():
    import pathlib
    g = Graph.load(str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "haber_ozeti_telegram.json"))
    links = {(g.nodes[c.src].type_name, c.src_port, g.nodes[c.dst].type_name, c.dst_port) for c in g.connections}
    assert ("AskAI", "done", "RssFeed", "next") in links and ("AskAI", "done", "TelegramSend", "trigger") in links
