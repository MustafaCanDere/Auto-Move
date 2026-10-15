"""0.7 bug fixes found by a full code review, one regression test each."""
import datetime as dt
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_v06 import RAW, FakeIMAP
from visionflow.ai import claude, mail
from visionflow.ai.config import AIConfig
from visionflow.core import Graph, create, load_builtin

load_builtin()


@pytest.fixture(autouse=True)
def ai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    c = AIConfig(api_key="test", email_user="ben@ornek.com", email_password="x")
    c.save()


def make(type_name, **values):
    n = create(type_name)
    n.values.update(values)
    n.setup()
    return n


def wait_idle(n, timeout=5):
    end = time.time() + timeout
    while n.state["busy"] and time.time() < end:
        time.sleep(0.01)


# --- preview thread survives odd images -------------------------------------------------------------

@pytest.mark.parametrize("img", [
    np.zeros((40, 50), bool), np.arange(2000, dtype=np.int64).reshape(40, 50), np.zeros((40, 50, 4), np.uint8),
    np.zeros((40, 50, 1), np.uint8), np.full((40, 50), np.nan, np.float32), np.zeros((40, 50, 2), np.uint8),
], ids=["bool", "int64", "bgra", "1ch", "nan", "2ch"])
def test_odd_images_become_displayable(img):
    pytest.importorskip("PySide6")
    from visionflow.ui.worker import displayable, shrink, to_qimage
    d = displayable(img)
    assert d.dtype == np.uint8 and d.ndim in (2, 3) and (d.ndim == 2 or d.shape[2] == 3)
    q = to_qimage(shrink(d, 20))
    assert q.width() == 20 and not q.isNull()
    assert displayable(np.zeros((0, 0, 3), np.uint8)) is None and to_qimage(np.zeros((0, 5), np.uint8)).isNull()


def test_worker_keeps_running_when_a_block_outputs_a_bool_mask():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from visionflow.ui.worker import FlowWorker
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    g = Graph()
    src = g.add_new("TestPattern", animate=True)
    py = g.add_new("PythonScript", code="out_img = img > 128")
    g.connect(src.id, "image", py.id, "img")
    w = FlowWorker(g.to_dict())
    frames, errors = [], []
    w.frame.connect(lambda payload, fps: frames.append(payload))
    w.failed.connect(errors.append)
    w.start()
    end = time.time() + 10
    while len(frames) < 5 and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    alive = w.isRunning()
    w.stop()
    w.wait(5000)
    assert alive and len(frames) >= 5 and not errors
    assert frames[-1][py.id][3] is not None and not frames[-1][py.id][3].isNull()


# --- documents, memory -------------------------------------------------------------------------------

def test_read_document_never_returns_previous_file(tmp_path):
    old, new = tmp_path / "eski.txt", tmp_path / "yeni.txt"
    old.write_text("ESKİ BELGE", encoding="utf-8")
    n = make("ReadDocument")
    assert n.process({"path": str(old)})["text"] == "ESKİ BELGE"
    with pytest.raises(Exception):
        n.process({"path": str(new)})
    with pytest.raises(Exception):
        n.process({"path": str(new)})              # still an error, not the old text
    new.write_text("YENİ", encoding="utf-8")
    assert n.process({"path": str(new)})["text"] == "YENİ"


def test_memory_keeps_same_named_files_from_different_folders(tmp_path):
    from visionflow.ai.memory import Memory
    docs = tmp_path / "belgeler"
    for year, word in (("2023", "elektrik"), ("2024", "doğalgaz")):
        (docs / year).mkdir(parents=True)
        (docs / year / "fatura.txt").write_text(f"{year} {word} faturası 100 TL", encoding="utf-8")
    m = Memory(tmp_path / "h.db")
    res = m.add_folder(docs)
    assert set(res) == {"2023/fatura.txt", "2024/fatura.txt"}
    assert m.search("elektrik") and m.search("doğalgaz")


# --- e-mail ------------------------------------------------------------------------------------------

def test_older_unread_mail_is_not_starved_by_the_limit(monkeypatch):
    FakeIMAP.boxes = {str(i).encode(): RAW.format(n=f"K{i}", a=f"k{i}@x.com", i=i).encode() for i in range(1, 26)}
    monkeypatch.setattr(mail, "IMAP_CLASS", FakeIMAP)
    n = make("EmailRead", interval=10)
    got = []
    for _ in range(3):                            # three polls of at most 20 each
        n.state["queue"].extend(n.fetch())
        while n.state["queue"]:
            m = n.state["queue"].pop(0)
            got.append(m.uid)
            n._remember(n.item_key(m))
            for k in n.extra_keys(m):
                n._remember(k)
    assert sorted(got, key=int) == [str(i) for i in range(1, 26)] and len(got) == 25


def test_noreply_mails_do_not_fill_the_limit(monkeypatch):
    boxes = {str(i).encode(): RAW.format(n="Robot", a="noreply@x.com", i=i).encode() for i in range(1, 21)}
    boxes[b"21"] = RAW.format(n="Ali", a="ali@x.com", i=21).encode()
    FakeIMAP.boxes = boxes
    monkeypatch.setattr(mail, "IMAP_CLASS", FakeIMAP)
    n = make("EmailRead")
    assert n.fetch() == []                        # 20 robots fill the first batch...
    assert [m.sender_addr for m in n.fetch()] == ["ali@x.com"]     # ...but are skipped from then on


def test_seen_keeps_newest_ids_across_restart(monkeypatch):
    from visionflow.nodes import ai_hub
    monkeypatch.setattr(ai_hub._Feed, "MAX_SEEN", 5)
    n = make("EmailRead")
    for k in ["zzz", "yyy", "aaa", "bbb", "ccc", "ddd", "eee"]:   # newest are alphabetically first
        n._remember(k)
    n._save_seen()
    n2 = make("EmailRead")
    assert list(n2.state["seen"]) == ["aaa", "bbb", "ccc", "ddd", "eee"]


def test_approval_files_never_overwrite(tmp_path):
    msgs = [mail.build_reply("ali@x.com", "Soru", f"cevap {i}", "ben@ornek.com", "") for i in range(3)]
    for m in msgs:
        mail.deliver(m, "onay klasörüne kaydet", folder=str(tmp_path / "onay"))
    files = sorted((tmp_path / "onay").iterdir())
    assert len(files) == 3
    assert sorted(b"cevap" in f.read_bytes() for f in files) == [True] * 3


def test_email_reply_count_counts_queued_replies(monkeypatch, tmp_path):
    n = make("EmailReply", mode="onay klasörüne kaydet", folder=str(tmp_path / "onay"))
    counts = []
    for i in range(4):
        n.process({"trigger": None, "to": "a@x.com", "subject": "s", "body": f"b{i}", "message_id": ""})
    end = time.time() + 5
    while len(counts) < 4 and time.time() < end:
        out = n.process({"trigger": None, "to": "a@x.com", "subject": "s", "body": "b3", "message_id": ""})
        if out["done"]:
            counts.append(out["count"])
        time.sleep(0.01)
    assert counts == [1, 2, 3, 4]


# --- jobs -------------------------------------------------------------------------------------------

class FakeClaude:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        c = kw["messages"][-1]["content"]
        text = c if isinstance(c, str) else c[-1]["text"]
        return SimpleNamespace(stop_reason="end_turn", usage=None, content=[SimpleNamespace(type="text",
                                                                                           text=self.fn(text))])


def test_later_success_clears_earlier_error(monkeypatch):
    def answer(q):
        if "kötü" in q:
            raise RuntimeError("boom")
        return "tamam"
    monkeypatch.setattr(claude, "_test_client", FakeClaude(answer))
    n = make("AskAI", use_memory=False)
    base = {"trigger": None, "context": None, "image": None}
    n.state["busy"] = True                    # hold the worker so both requests run back-to-back in one thread
    n.state["pending"] = [{**base, "prompt": "iyi"}]
    n.state["busy"] = False
    with n._lock:
        n.state["busy"] = True
    n._work({**base, "prompt": "kötü"})       # fails, then the queued "iyi" succeeds
    out = n.process({**base, "prompt": "iyi"})
    assert out["done"] and out["text"] == "tamam"
    assert n.process({**base, "prompt": "iyi"})["text"] == "tamam"     # no stale "boom" afterwards


def test_nan_key_does_not_fire_every_frame(monkeypatch):
    fake = FakeClaude(lambda q: "x")
    monkeypatch.setattr(claude, "_test_client", fake)
    n = make("AskAI", use_memory=False)
    for _ in range(20):
        n.process({"trigger": None, "prompt": float("nan"), "context": None, "image": None})
    wait_idle(n)
    assert fake.calls == 0


def test_request_queued_at_the_last_moment_still_runs(monkeypatch):
    """Stress the hand-over between the worker finishing and a new request arriving."""
    monkeypatch.setattr(claude, "_test_client", FakeClaude(lambda q: q))
    n = make("AskAI", use_memory=False)
    done = []
    for i in range(40):
        out = n.process({"trigger": None, "prompt": f"s{i}", "context": None, "image": None})
        if out["done"]:
            done.append(out["text"])
    end = time.time() + 10
    while len(done) < 40 and time.time() < end:
        out = n.process({"trigger": None, "prompt": "s39", "context": None, "image": None})
        if out["done"]:
            done.append(out["text"])
    wait_idle(n)
    assert done == [f"s{i}" for i in range(40)]          # every request answered, in order


# --- spending limit file ---------------------------------------------------------------------------

def test_usage_file_is_never_seen_half_written():
    resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=100_000, output_tokens=100_000))
    claude._record_usage("claude-opus-5-5", resp)
    first = claude.usage_today()["usd"]
    stop, zero_reads = threading.Event(), []

    def reader():
        while not stop.is_set():
            if claude.usage_today()["usd"] < first:
                zero_reads.append(1)

    t = threading.Thread(target=reader)
    t.start()
    for _ in range(100):
        claude._record_usage("claude-opus-5-5", resp)
    stop.set()
    t.join()
    assert not zero_reads
    assert not [p for p in (claude._usage_file().parent).iterdir() if p.suffix == ".tmp"]


# --- graph / blocks ------------------------------------------------------------------------------------

def test_empty_image_gives_a_clear_error_not_a_crash():
    g = Graph()
    py = g.add_new("PythonScript", code="out_img = img[0:0]")
    src = g.add_new("TestPattern")
    for t in ("Combine", "AverageColor", "Blur"):
        n = g.add_new(t)
        g.connect(py.id, "out_img", n.id, "image" if t != "Combine" else "a")
    g.connect(src.id, "image", py.id, "img")
    g.setup()
    res = g.run_once()
    for nid, r in res.items():
        if g.nodes[nid].type_name in ("Combine", "AverageColor", "Blur"):
            assert r.error and "Boş görüntü" in r.error


def test_resize_tiny_image_and_crop_outside():
    r = make("Resize", scale=0.1)
    assert r.process({"image": np.zeros((1, 1, 3), np.uint8)})["image"].shape[:2] == (1, 1)
    c = make("Crop", x=500, y=500)
    with pytest.raises(ValueError, match="dışında"):
        c.process({"image": np.zeros((100, 100, 3), np.uint8)})


def test_scheduler_started_within_the_minute_still_fires():
    n = create("Scheduler")
    n.values.update(times="09:00", days="her gün")
    n.now = lambda: dt.datetime(2026, 10, 12, 9, 0, 30)
    n.setup()
    assert n.process({})["tick"]


def test_feed_without_next_gives_separate_new_pulses(tmp_path):
    for k in range(3):
        (tmp_path / f"f{k}.txt").write_text("x")
    n = make("FolderWatch", folder=str(tmp_path), existing=True, pattern="*.txt", interval=0.5)
    news = []
    end = time.time() + 5
    while sum(news) < 3 and time.time() < end:
        news.append(n.process({"next": None})["new"])
        time.sleep(0.01)
    assert sum(news) == 3
    assert not any(a and b for a, b in zip(news, news[1:]))     # never two 'new' frames in a row


def test_feed_poll_error_does_not_hold_back_fetched_items(tmp_path):
    n = make("FolderWatch", folder=str(tmp_path))
    n.state.update(queue=[{"path": str(tmp_path / "a.txt"), "stat": (1, 1.0)}], error="ağ hatası", next_poll=1e18)
    assert n.process({"next": None})["name"] == "a.txt"
    with pytest.raises(RuntimeError, match="ağ hatası"):
        n.process({"next": None})


def test_scene_ignores_dropping_the_same_wire_twice(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import QPointF
    from visionflow.ui.scene import EdgeItem, NodeScene
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = NodeScene()
    a, b = s.add_node("TestPattern", QPointF(0, 0)), s.add_node("Blur", QPointF(300, 0))
    out, inp = a.sockets[("image", True)], b.sockets[("image", False)]
    s.try_connect(out, inp)
    s.try_connect(out, inp)
    assert len(s.graph.connections) == 1 and len([i for i in s.items() if isinstance(i, EdgeItem)]) == 1
