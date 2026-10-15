"""2.0 (d): operator panel (HMI), edit lock, production data collector and HTML/PDF production report."""
import datetime as dt
import json
import os
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow.core import Graph, Node, NodeResult, PortType, StopFlow, create, load_builtin, register  # noqa: E402
from visionflow.core.operator import check_password, clean_operator, lock_ok, locked, make_lock  # noqa: E402
from visionflow.production import (ProductionDB, Tally, auto_config, default_image, make_report,  # noqa: E402
                                   render_html, write_pdf)

load_builtin()


@register
class _Part20d(Node):
    """A camera stand-in: one part per frame, every third one bad, stops after ``stop_after`` frames."""
    type_name = "_Part20d"
    outputs = [("image", PortType.IMAGE), ("ok", PortType.BOOL), ("ok_count", PortType.NUMBER),
               ("nok_count", PortType.NUMBER)]
    is_source = True
    stop_after = 9

    def setup(self):
        self.state.update(i=0, ok=0, nok=0)

    def process(self, inputs):
        st = self.state
        st["i"] += 1
        if st["i"] > self.stop_after:
            raise StopFlow()
        ok = st["i"] % 3 != 0
        st["ok" if ok else "nok"] += 1
        img = np.zeros((60, 80, 3), np.uint8)
        cv2.circle(img, (40, 30), 20, (0, 255, 0) if ok else (0, 0, 255), -1)
        return {"image": img, "ok": ok, "ok_count": st["ok"], "nok_count": st["nok"]}


def part_flow(counters=False, lock=None):
    g = Graph()
    src = g.add_new("_Part20d", (0, 0))
    prev = g.add_new("Preview", (260, 0))
    g.connect(src.id, "image", prev.id, "image")
    cfg = {"title": "Hat 1", "image": "", "lamp": {"node": src.id, "port": "ok"}, "history": 5}
    if counters:
        cfg.update(good={"node": src.id, "port": "ok_count"}, bad={"node": src.id, "port": "nok_count"})
    if lock:
        cfg["lock"] = make_lock(lock, iterations=2000)
    g.operator = clean_operator(cfg, g)
    return g, src, prev


def res(**outputs):
    return NodeResult(outputs, 1.0)


# --- config in the flow file -----------------------------------------------------------------------------------

def test_operator_config_round_trip(tmp_path):
    g, src, prev = part_flow(counters=True, lock="1234")
    path = tmp_path / "akis.json"
    g.save(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["operator"]["lamp"] == {"node": src.id, "port": "ok"}
    assert "1234" not in path.read_text(encoding="utf-8")            # only the salted hash is saved
    g2 = Graph.load(str(path))
    assert not g2.warnings
    assert g2.operator == g.operator
    assert check_password(locked(g2), "1234") and not check_password(locked(g2), "12345")


def test_flow_without_operator_has_no_key():
    g = Graph()
    g.add_new("TestPattern")
    assert "operator" not in g.to_dict()
    assert Graph.from_dict(g.to_dict()).operator is None


def test_operator_config_validated_on_load():
    g, src, prev = part_flow()
    data = g.to_dict()
    data["operator"].update(lamp={"node": "yok", "port": "ok"}, good={"node": src.id, "port": "image"},
                            image="hayalet", history="çok", lock={"salt": "zz", "hash": "1", "iterations": 5})
    g2 = Graph.from_dict(data)
    op = g2.operator
    assert op["lamp"] is None and op["good"] is None and op["image"] == "" and op["history"] == 20
    assert "lock" not in op
    assert len(g2.warnings) >= 4 and all("Operatör" in w for w in g2.warnings)
    for bad in ("metin", 5, [1, 2]):
        g3 = Graph.from_dict({**data, "operator": bad})
        assert g3.operator is None and any("operatör" in w.lower() for w in g3.warnings)
    # an image block that exists is kept, history is clamped
    data["operator"].update(image=prev.id, history=10 ** 6)
    assert Graph.from_dict(data).operator["image"] == prev.id
    assert Graph.from_dict(data).operator["history"] == 500


def test_password_hash():
    lock = make_lock("gizli şifre", iterations=2000)
    assert lock_ok(lock) and len(lock["salt"]) == 32
    assert check_password(lock, "gizli şifre")
    assert not check_password(lock, "Gizli şifre") and not check_password(lock, "") and not check_password(lock, None)
    assert make_lock("gizli şifre", iterations=2000)["hash"] != lock["hash"]      # salted
    assert check_password(None, "x")                                             # no lock: always open
    assert not check_password({"salt": "x"}, "x")
    with pytest.raises(ValueError, match="en az 4"):
        make_lock("abc")


def test_auto_config_and_default_image():
    g, src, prev = part_flow()
    assert default_image(g) == prev.id
    assert auto_config(g)["lamp"] == {"node": src.id, "port": "ok"}
    g2 = Graph()
    gray = g2.add_new("Grayscale")
    assert default_image(g2) == gray.id
    assert default_image(Graph()) == ""


# --- counting ----------------------------------------------------------------------------------------------------

def test_tally_counts_lamp_results():
    g, src, prev = part_flow()
    t = Tally(g.operator)
    for ok in (True, True, False):
        t.feed({src.id: res(ok=ok)}, g)
    t.feed({src.id: NodeResult({}, 0.0, waiting="tetik bekleniyor")}, g)       # no part
    snap = t.snapshot()
    assert (snap["good"], snap["bad"], snap["total"]) == (2, 1, 3)
    assert snap["lamp"] is False and snap["yield"] == pytest.approx(66.7)
    assert [h[1] for h in snap["history"]] == [False, True, True]                # newest first
    t.feed({src.id: NodeResult({}, 0.0, "ValueError: kamera yok")}, g)
    assert t.snapshot()["lamp"] is None and "kamera" in t.snapshot()["lamp_error"]
    assert t.take()[:3] == (2, 1, 3)
    assert t.take()[:3] == (0, 0, 0)
    t.reset()
    assert t.snapshot()["total"] == 0 and t.snapshot()["yield"] is None and not t.snapshot()["history"]


def test_tally_counter_outputs_survive_restart():
    g, src, prev = part_flow(counters=True)
    t = Tally(g.operator)
    for okc, nokc in [(1, 0), (2, 0), (2, 1), (2, 1), (0, 0), (1, 0)]:          # counter restarted at 0
        t.feed({src.id: res(ok=True, ok_count=okc, nok_count=nokc)}, g)
    snap = t.snapshot()
    assert (snap["good"], snap["bad"], snap["total"]) == (3, 1, 4)
    assert len(snap["history"]) == 4                                         # one entry per new part, not per frame
    t.feed({src.id: res(ok=True, ok_count="bozuk", nok_count=float("nan"))}, g)
    assert t.snapshot()["total"] == 4


def test_tally_records_block_errors_once_a_minute():
    g, src, prev = part_flow()
    clock = [1000.0]
    t = Tally(g.operator, clock=lambda: clock[0])
    for _ in range(3):
        t.feed({prev.id: NodeResult({}, 0.0, "ValueError: boş"), src.id: res(ok=True)}, g)
    clock[0] += 61
    t.feed({prev.id: NodeResult({}, 0.0, "ValueError: boş")}, g)
    t.feed({prev.id: NodeResult({}, 0.0, "Önceki blok hata verdi")}, g)
    alarms = t.take()[3]
    assert len(alarms) == 2 and "Önizleme" in alarms[0][1]


# --- database and report ------------------------------------------------------------------------------------------

def _filled_db(tmp_path):
    db = ProductionDB(tmp_path / "uretim_verisi.sqlite")
    base = dt.datetime(2026, 5, 4, 8, 0)
    db.add_sample("Hat 1", 50, 2, 52, ts=(base + dt.timedelta(minutes=10)).timestamp())
    db.add_sample("Hat 1", 30, 3, 33, ts=(base + dt.timedelta(minutes=70)).timestamp())
    db.add_sample("Hat 2", 7, 7, 14, ts=(base + dt.timedelta(minutes=75)).timestamp(),
                  alarms=[((base + dt.timedelta(minutes=76)).timestamp(), "Kamera: bağlantı <koptu>")])
    db.add_sample("Hat 1", 0, 0, 0)                                          # nothing: not written
    return db, base


def test_db_summary(tmp_path):
    db, base = _filled_db(tmp_path)
    assert db.flows() == ["Hat 1", "Hat 2"]
    s = db.summary(base, base + dt.timedelta(hours=3), "Hat 1")
    assert (s["good"], s["bad"], s["total"]) == (80, 5, 85)
    assert [r[1:] for r in s["hours"]] == [(50, 2, 52), (30, 3, 33), (0, 0, 0)]
    assert s["alarms"] == []
    all_ = db.summary(base, base + dt.timedelta(hours=3))
    assert all_["total"] == 99 and len(all_["alarms"]) == 1
    with pytest.raises(ValueError, match="Bitiş"):
        db.summary(base, base)
    empty = ProductionDB(tmp_path / "yok" / "x.sqlite").summary(base, base + dt.timedelta(hours=1))
    assert empty["total"] == 0 and empty["yield"] is None


def test_report_html_content(tmp_path):
    db, base = _filled_db(tmp_path)
    out = tmp_path / "rapor" / "r.html"
    html_path, pdf_path, msg = make_report(db, base, base + dt.timedelta(hours=3), out, None, "Vardiya A")
    text = open(html_path, encoding="utf-8").read()
    assert pdf_path is None and msg is None
    assert text.startswith("<!DOCTYPE html>") and "<svg" in text and "Vardiya A" in text
    assert "Tüm akışlar" in text and "04.05.2026 08:00" in text and "04.05.2026 11:00" in text
    assert ">87<" in text and ">12<" in text and ">99<" in text and "%87.9" in text
    assert "&lt;koptu&gt;" in text and "<koptu>" not in text                    # escaped
    assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")     # self-contained
    assert "<script" not in text
    import xml.etree.ElementTree as ET
    svg = ET.fromstring(text[text.index("<svg"):text.index("</svg>") + 6])     # well-formed SVG
    rects = [r for r in svg.iter("{http://www.w3.org/2000/svg}rect") if r.get("fill") == "#2e7d32"]
    assert len(rects) == 3 + 1                                                 # 3 hours + legend
    one = render_html(db.summary(base, base + dt.timedelta(hours=1), "Hat 2"), flow_name="Hat 2")
    assert "Bu dönemde kayıt yok" not in one and "Alarm yok" in one


def test_report_pdf(tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    db, base = _filled_db(tmp_path)
    html_path, pdf_path, msg = make_report(db, base, base + dt.timedelta(hours=3), tmp_path / "r.html", "Hat 1",
                                           pdf=True)
    assert msg is None, msg
    with open(pdf_path, "rb") as f:
        assert f.read(5) == b"%PDF-"
    assert os.path.getsize(pdf_path) > 1000
    assert write_pdf("<p>x</p>", tmp_path / "yok" / "klasor" / "a.pdf") is not None


def test_report_block(tmp_path, monkeypatch):
    from visionflow.core import node as _node
    db, base = _filled_db(tmp_path)
    now = dt.datetime.now()
    db.add_sample("Hat 1", 4, 1, 5, ts=(now - dt.timedelta(minutes=5)).timestamp())
    monkeypatch.setattr(_node, "FLOW_DIR", tmp_path)
    n = create("ProductionReport")
    n.values.update(folder=str(tmp_path / "raporlar"), hours=2)
    n.safe_setup()
    out = n.process({"trigger": None})                     # unconnected: once
    assert out["done"] and out["path"].endswith(".html")
    assert ">5<" in open(out["path"], encoding="utf-8").read()
    assert not n.process({"trigger": None})["done"]
    assert not n.process({"trigger": False})["done"]
    time.sleep(1.05)                                       # report names have seconds
    out2 = n.process({"trigger": True})
    assert out2["done"] and out2["path"] != out["path"]
    assert not n.process({"trigger": True})["done"]        # rising edge only
    n.values["db"] = str(tmp_path / "raporlar")            # a folder, not a database
    n.process({"trigger": False})
    with pytest.raises(ValueError, match="Rapor yazılamadı"):
        n.process({"trigger": True})


# --- UI ------------------------------------------------------------------------------------------------------------

@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_panel_lamp_and_counters_from_fake_frames(qapp, tmp_path):
    from visionflow.ui.operator import OperatorPanel
    g, src, prev = part_flow()
    panel = OperatorPanel(g.to_dict(), str(tmp_path / "hat.json"))
    assert panel.image_id == prev.id and panel.lamp.text() == "BEKLENİYOR"
    panel.tally.feed({src.id: res(ok=True)}, panel.graph)
    panel.refresh()
    assert panel.lamp.text() == "SAĞLAM" and "#2e7d32" in panel.lamp.styleSheet()
    panel.tally.feed({src.id: res(ok=False)}, panel.graph)
    panel.refresh()
    assert panel.lamp.text() == "HATALI" and "#c62828" in panel.lamp.styleSheet()
    assert [panel.counter_labels[k].text() for k in ("good", "bad", "total", "yield")] == ["1", "1", "2", "%50"]
    assert panel.history_list.count() == 2 and "HATALI" in panel.history_list.item(0).text()
    img = np.full((40, 60, 3), 200, np.uint8)
    panel.on_watch(prev.id, img)
    assert not panel.image_label.pixmap().isNull()
    panel.sample()
    s = panel.db.summary(dt.datetime.now() - dt.timedelta(hours=1), dt.datetime.now() + dt.timedelta(minutes=1))
    assert (s["good"], s["bad"]) == (1, 1) and panel.db.path.parent == tmp_path
    panel.reset_counters()
    assert panel.counter_labels["total"].text() == "0" and panel.history_list.count() == 0
    panel.close()


def test_panel_runs_flow(qapp, tmp_path):
    from visionflow.ui.operator import OperatorPanel
    g, src, prev = part_flow(counters=True)
    panel = OperatorPanel(g.to_dict(), str(tmp_path / "hat.json"))
    panel.start()
    assert panel.worker is not None and not panel.start_btn.isEnabled()
    assert panel.worker.wait(10000)
    for _ in range(20):
        qapp.processEvents()
    assert panel.worker is None and panel.start_btn.isEnabled()
    snap = panel.refresh()
    assert (snap["good"], snap["bad"], snap["total"]) == (6, 3, 9)
    assert panel.history_list.count() == 5                    # "last N results"
    assert not panel.image_label.pixmap().isNull()
    s = panel.db.summary(dt.datetime.now() - dt.timedelta(hours=1), dt.datetime.now() + dt.timedelta(minutes=1))
    assert s["total"] == 9                                    # written when the flow stopped
    panel.start()                                             # a second run keeps adding up
    panel.worker.wait(10000)
    for _ in range(20):
        qapp.processEvents()
    assert panel.refresh()["total"] == 18
    panel.close()


def test_panel_lock(qapp, tmp_path, monkeypatch):
    from visionflow.ui import operator as op

    class Editor:
        shown = False

        def show(self):
            self.shown = True

        def raise_(self):
            pass

    g, src, prev = part_flow(lock="1234")
    ed = Editor()
    panel = op.OperatorPanel(g.to_dict(), None, editor=ed)
    assert panel.editor_btn.text().startswith("🔒")
    assert not panel.leave("yanlış") and not ed.shown and "yanlış" in panel.status.text()
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: False)
    assert not panel.close() and not ed.shown                 # closing needs the password too
    assert panel.leave("1234") and ed.shown
    # factory mode (--operator): the editor is made only after the password
    made = []
    panel2 = op.OperatorPanel(g.to_dict(), None, editor_factory=lambda: made.append(Editor()) or made[-1])
    assert not panel2.leave("0000") and not made
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: True)
    assert panel2.leave() and made and made[0].shown
    # no lock: free
    g3, *_ = part_flow()
    panel3 = op.OperatorPanel(g3.to_dict(), None)
    assert panel3.unlock("ne olursa") and not panel3.editor_btn.isVisible()
    panel3.close()


def test_run_operator_from_file(qapp, tmp_path):
    from visionflow.ui.operator import run_operator
    g, src, prev = part_flow()
    g.save(str(tmp_path / "hat.json"))
    panel = run_operator(str(tmp_path / "hat.json"), editor_factory=lambda: None)
    assert panel.flow_name == "hat" and panel.title_label.text() == "Hat 1"
    panel.close()
    with pytest.raises(OSError):
        run_operator(str(tmp_path / "yok.json"))


def test_config_dialog(qapp):
    from visionflow.ui.operator import OperatorConfigDialog
    g, src, prev = part_flow(lock="1234")
    dlg = OperatorConfigDialog(None, g)
    cfg = dlg.result_config()
    assert cfg["lamp"] == {"node": src.id, "port": "ok"} and cfg["lock"] == g.operator["lock"]
    dlg.counter_combos["good"].setCurrentIndex(dlg.counter_combos["good"].findText(f"_Part20d ({src.id}) → ok_count"))
    assert dlg.result_config()["good"] == {"node": src.id, "port": "ok_count"}
    dlg.remove_lock.setChecked(True)
    assert "lock" not in dlg.result_config()
    dlg.pw1.setText("abcd")
    dlg.pw2.setText("abce")
    with pytest.raises(ValueError, match="aynı değil"):
        dlg.result_config()
    dlg.accept()
    assert "aynı değil" in dlg.error.text()
    dlg.pw2.setText("abcd")
    new = dlg.result_config()
    assert check_password(new["lock"], "abcd")
    g.operator = clean_operator(new, g)                   # what the dialog makes passes the file check
    assert Graph.from_dict(g.to_dict()).operator == g.operator


def test_report_dialog(qapp, tmp_path):
    from PySide6.QtCore import QDateTime
    from visionflow.ui.operator import ReportDialog
    db, base = _filled_db(tmp_path)
    dlg = ReportDialog(None, str(tmp_path / "Hat 1.json"))
    assert dlg.db.path == db.path and dlg.flow_combo.currentData() == "Hat 1"
    dlg.start_edit.setDateTime(QDateTime(base))
    dlg.end_edit.setDateTime(QDateTime(base + dt.timedelta(hours=2)))
    html_path, pdf_path, msg = dlg.generate(str(tmp_path / "out.html"))
    assert ">80<" in open(html_path, encoding="utf-8").read()
    assert open(pdf_path, "rb").read(4) == b"%PDF"


@pytest.fixture
def win(qapp):
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    w.stop()
    w.close()
    qapp.processEvents()


def test_editor_menus_and_locked_open(win, tmp_path, monkeypatch):
    from visionflow.ui import operator as op
    texts = [a.text() for m in win.menuBar().actions() for a in m.menu().actions()]
    assert "🏭 Operatör Paneli…" in texts and "📊 Üretim raporu…" in texts and "Operatör paneli ayarları…" in texts
    g, src, prev = part_flow(lock="1234")
    path = str(tmp_path / "kilitli.json")
    g.save(path)
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: False)
    win.open_flow(path)
    assert win.path != path and not win.root.nodes                 # refused
    monkeypatch.setattr(op, "ask_password", lambda parent, lock, *a: check_password(lock, "1234"))
    win.open_flow(path)
    assert win.path == path and win.root.operator["lock"]
    win.open_flow(path, unlocked=True)
    assert win.path == path


def test_editor_opens_panel_and_returns(win, qapp, monkeypatch):
    g, src, prev = part_flow()
    win._set_root(Graph.from_dict(g.to_dict()))
    win.show()
    panel = win.open_operator()
    assert panel is not None and not win.isVisible() and panel.worker is not None
    panel.worker.wait(10000)
    for _ in range(20):
        qapp.processEvents()
    assert panel.refresh()["total"] == 9
    assert panel.leave() and win.isVisible() and panel.worker is None


def test_editor_saves_operator_config(win, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from visionflow.ui import operator as op
    g, src, prev = part_flow()
    g.operator = None
    win._set_root(Graph.from_dict(g.to_dict()))

    def fake_exec(self):
        self.title_edit.setText("Montaj")
        self.config = self.result_config()
        return QDialog.Accepted
    monkeypatch.setattr(op.OperatorConfigDialog, "exec", fake_exec)
    win.edit_operator()
    assert win.root.operator["title"] == "Montaj" and win.root.operator["lamp"]["port"] == "ok"
    assert win.dirty
    win.undo()
    assert win.root.operator is None
