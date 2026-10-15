"""0.8 bug fixes from the second code review, one regression test each."""
import json
import math
import os
import pathlib
import signal
import sqlite3
import time

import numpy as np
import pytest

from visionflow.core import Graph, create, load_builtin

load_builtin()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make(t, **v):
    n = create(t)
    n.values.update(v)
    n.setup()
    return n


@pytest.fixture
def qt():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(qt, tmp_path):
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w.recovery = tmp_path / "k.json"
    w.show()
    yield w
    w.stop()
    w.saved_snap = w._snap()
    w.close()


# --- editor ----------------------------------------------------------------------------------------

def test_paste_and_assistant_actions_from_menu(win, qt):
    from PySide6.QtCore import QPointF
    it = win.scene.add_node("TestPattern", QPointF(0, 0))
    it.setSelected(True)
    win.copy()
    act = next(a for a in win.findChildren(type(win.undo_act)) if a.text() == "Yapıştır")
    n0 = len(win.scene.graph.nodes)
    act.trigger()
    qt.processEvents()
    assert len(win.scene.graph.nodes) == n0 + 1


def test_float_slider_keeps_typed_value(qt):
    from PySide6.QtWidgets import QDoubleSpinBox
    from visionflow.ui.panels import Properties
    node = create("Enhance")
    p = Properties()
    p.show_node(node)
    box = p.findChildren(QDoubleSpinBox)[0]
    name = next(q.name for q in node.params if q.kind == "float")
    for v in (0.29, 0.57, 1.234):
        box.setValue(v)
        assert node.values[name] == pytest.approx(v) and box.value() == pytest.approx(v)


def test_unknown_choice_is_shown_and_first_option_selectable(qt):
    from PySide6.QtWidgets import QComboBox
    from visionflow.ui.panels import Properties
    node = create("Enhance")
    name = next(q for q in node.params if q.kind == "choice")
    node.values[name.name] = "olmayan"
    p = Properties()
    p.show_node(node)
    cb = p.findChildren(QComboBox)[0]
    assert cb.currentText() == "olmayan"
    cb.setCurrentIndex(0)
    assert node.values[name.name] == name.choices[0]


def test_failed_save_keeps_old_path_and_old_file(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    good = tmp_path / "iyi.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(good), ""))
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    win.scene.graph.add_new("TestPattern")
    assert win.save_flow(True)
    before = good.read_text(encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "yok" / "a.json"), ""))
    assert win.save_flow(True) is False and warned
    assert win.path == str(good) and good.read_text(encoding="utf-8") == before


def test_graph_save_is_atomic(tmp_path, monkeypatch):
    g = Graph()
    g.add_new("TestPattern")
    p = tmp_path / "a.json"
    g.save(p)
    before = p.read_text(encoding="utf-8")
    monkeypatch.setattr(json, "dumps", lambda *a, **k: (_ for _ in ()).throw(TypeError("bozuk değer")))
    with pytest.raises(TypeError):
        g.save(p)
    assert p.read_text(encoding="utf-8") == before and not list(tmp_path.glob("*.tmp"))


def test_paste_foreign_clipboard_with_bad_link_is_complete(qt):
    from visionflow.ui.scene import NodeScene
    s = NodeScene()
    snaps = []
    s.structure_changed.connect(lambda: snaps.append(1))
    s.paste({"nodes": [{"id": "a", "type": "TestPattern", "pos": [0, 0]}, {"id": "b", "type": "Blur"},
                       "çöp", {"type": "Grayscale", "pos": None, "values": [1, 2]}],
             "connections": [{"src": "a", "src_port": "yok", "dst": "b", "dst_port": "image"},
                             {"src": "a", "dst": "b"}]})
    assert len(s.graph.nodes) == 3 and snaps


def test_examples_file_detection_is_exact(win, tmp_path):
    from visionflow.ui import main_window
    sibling = pathlib.Path(str(main_window.EXAMPLES) + "_benim")
    try:
        sibling.mkdir()
        g = Graph()
        g.add_new("TestPattern")
        g.save(sibling / "akis.json")
        win.open_flow(str(sibling / "akis.json"))
        assert win.path == str(sibling / "akis.json")
    finally:
        for f in sibling.glob("*"):
            f.unlink()
        sibling.rmdir()


# --- loading flows -------------------------------------------------------------------------------------

def test_load_coerces_and_reports_bad_values():
    g = Graph.from_dict({"version": 99, "nodes": [
        {"id": "a", "type": "TestPattern", "pos": None, "values": {"width": "640", "animate": "false"}},
        {"id": "b", "type": "TestPattern", "values": {"width": "640px", "height": 99999}},
        {"id": "b", "type": "Blur", "values": [1]},
        {"type": "Grayscale"}, "çöp", {"id": "x"}],
        "connections": [{"src": "a", "dst": "b"}, None, {"src": "a", "src_port": "image", "dst": "b",
                                                          "dst_port": "image"}]})
    a, b = g.nodes["a"], g.nodes["b"]
    assert a.values["width"] == 640 and a.values["animate"] is False and a.pos == (0.0, 0.0)
    assert b.values["width"] == 640 and b.values["height"] == 4096        # bad → default, too big → clamped
    assert len(g.nodes) == 4                                                 # duplicate id got a new id
    w = " | ".join(g.warnings)
    assert "daha yeni" in w and "geçersiz" in w and "Aynı kimlikli" in w and "Bozuk" in w


def test_properties_panel_survives_values_from_old_files(qt):
    from visionflow.ui.panels import Properties
    g = Graph.from_dict({"nodes": [{"id": "a", "type": "TestPattern", "values": {"width": "640px"}}]})
    Properties().show_node(g.nodes["a"])


def test_assistant_build_graph_rejects_malformed_output():
    from visionflow.assistant import build_graph
    for bad in (["liste"], {"nodes": "x"}, {"nodes": ["metin", {"type": ["Blur"]}]},
                {"nodes": [{"id": "a", "type": "TestPattern", "values": [1]}], "connections": None},
                {"nodes": [{"id": "a", "type": "TestPattern"}], "connections": [["a", "image", "b", "image"]]}):
        g, errors = build_graph(bad)
        assert errors, bad
    g, errors = build_graph({"nodes": [{"id": "a", "type": "TestPattern"}, {"id": "a", "type": "Blur"}]})
    assert any("aynı id" in e for e in errors) and len(g.nodes) == 1


# --- blocks ----------------------------------------------------------------------------------------

def test_line_counter_counts_a_crossing_through_the_line():
    lc = make("LineCounter", line="0,240,640,240")
    for prev, cy in ((230, 240), (240, 250)):
        det = [{"box": (100, cy - 10, 20, 20), "track_id": 1, "prev_center": (110, prev), "label": "p",
                "class_id": 0, "score": 1}]
        out = lc.process({"detections": det, "image": None, "reset": None})
    assert out["total"] == 1 and out["in"] == 1


def test_label_tool_clamps_boxes_outside_the_image(tmp_path):
    pytest.importorskip("PySide6")
    import cv2
    from visionflow.ui.tools import Dataset
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((100, 200, 3), np.uint8))
    ds = Dataset(tmp_path)
    ds.classes = ["x"]
    ds.save(ds.images[0], [[0, -50, 10, 30, 40], [0, 180, 10, 260, 40], [0, -50, 10, -10, 40]], 200, 100)
    for line in (tmp_path / "a.txt").read_text().splitlines():
        cx, cy, w, h = map(float, line.split()[1:])
        assert 0 <= cx - w / 2 and cx + w / 2 <= 1 and 0 <= cy - h / 2 and cy + h / 2 <= 1
    assert len((tmp_path / "a.txt").read_text().splitlines()) == 2


def test_sample_collector_saves_nothing_when_detector_finds_nothing(tmp_path):
    sc = make("SampleCollector", folder=str(tmp_path), label="saglam")
    sc.process({"image": np.zeros((50, 50, 3), np.uint8), "trigger": True, "detections": []})
    assert not list((tmp_path / "saglam").iterdir())
    sc.process({"image": np.zeros((50, 50, 3), np.uint8), "trigger": False, "detections": None})
    sc.process({"image": np.zeros((50, 50, 3), np.uint8), "trigger": True, "detections": None})
    assert len(list((tmp_path / "saglam").iterdir())) == 1


def test_serial_write_keeps_turkish_and_backslashes():
    sent = []
    n = create("SerialWrite")
    n.values["template"] = "{v}\\n"
    n.state.update(ser=type("S", (), {"write": lambda self, b: sent.append(b)})(), last=None)
    n.process({"value": "kırmızı şişe C:\\new", "trigger": None})
    assert sent == ["kırmızı şişe C:\\new\n".encode()]


@pytest.mark.parametrize("names,expect", [("x,x,y", ["x", "x_2", "y"]), ("id,zaman,c", ["id_2", "zaman_2", "c"])])
def test_sqlite_logger_column_names(tmp_path, names, expect):
    n = make("SQLiteLogger", path=str(tmp_path / "d.db"), table="t", names=names)
    assert n.state["cols"] == expect
    n.process({"a": 1, "b": 2, "c": 3, "trigger": None})
    n.teardown()


def test_sqlite_logger_existing_table_gets_new_columns(tmp_path):
    db = str(tmp_path / "d.db")
    n = make("SQLiteLogger", path=db, table="olay", names="a,b,c")
    n.process({"a": 1, "b": 2, "c": 3, "trigger": None})
    n.teardown()
    n = make("SQLiteLogger", path=db, table="olay", names="adet,b,c")
    n.process({"a": 7, "b": 8, "c": 9, "trigger": None})
    n.teardown()
    rows = sqlite3.connect(db).execute("select a, adet from olay order by id").fetchall()
    assert rows == [(1, None), (None, 7)]


def test_video_recorder_clips_in_the_same_second_do_not_overwrite(tmp_path):
    vr = make("VideoRecorder", folder=str(tmp_path))
    f = np.zeros((64, 64, 3), np.uint8)
    p1 = vr.process({"image": f, "record": True})["path"]
    vr.process({"image": f, "record": False})
    p2 = vr.process({"image": f, "record": True})["path"]
    vr.teardown()
    assert p1 != p2 and len(list(tmp_path.iterdir())) == 2


def test_statistics_ignores_nan():
    for order in ((1.0, float("nan"), 3.0), (float("nan"), 1.0, 3.0)):
        st = make("Statistics")
        for v in order:
            out = st.process({"value": v, "reset": None})
        assert (out["mean"], out["min"], out["max"]) == (2.0, 1.0, 3.0)


def test_web_panel_api_is_valid_json_with_nan():
    import socket
    import urllib.request
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    wp = make("WebPanel", port=port)
    try:
        wp.process({k: (None if k == "image" else float("nan")) for k, _ in wp.inputs})
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/api", timeout=5).read()
        data = json.loads(body, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        assert all(not (isinstance(v, float) and math.isnan(v)) for v in data["values"].values())
    finally:
        wp.teardown()


def test_settings_read_at_start_restart_the_block_during_a_run(tmp_path):
    pytest.importorskip("PySide6")
    from visionflow.ui.worker import FlowWorker
    g = Graph()
    n = g.add_new("SQLiteLogger", path=str(tmp_path / "a.db"), table="t1")
    w = FlowWorker(g.to_dict())
    n.setup()
    w.set_param(n.id, "table", "t2")
    w._apply_pending(g)
    assert n.state["table"] == "t2"
    n.teardown()
    assert "table" in create("SQLiteLogger").setup_params and "port" in create("WebPanel").setup_params


def test_modbus_short_reply_is_a_clear_error():
    import socket
    import struct
    import threading
    from visionflow.nodes.industrial import ModbusClient
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        c, _ = srv.accept()
        c.recv(100)
        c.sendall(struct.pack(">HHHB", 1, 0, 1, 1))     # length 1: no function code at all
        time.sleep(0.2)
        c.close()
    threading.Thread(target=serve, daemon=True).start()
    cl = ModbusClient("127.0.0.1", srv.getsockname()[1], 1)
    with pytest.raises(RuntimeError, match="kısa"):
        cl.request(b"\x03\x00\x00\x00\x01")
    srv.close()


def test_headless_handles_sigterm(tmp_path, monkeypatch):
    from visionflow import headless
    installed = {}
    monkeypatch.setattr(signal, "signal", lambda sig, fn: installed.setdefault(sig, fn))
    g = Graph()
    g.add_new("TestPattern", animate=False)
    p = tmp_path / "a.json"
    g.save(p)
    headless.main([str(p), "--frames", "1"])
    assert signal.SIGTERM in installed and signal.SIGINT in installed


def test_window_settings_follow_visionflow_home(win):
    assert pathlib.Path(os.environ["VISIONFLOW_HOME"]).as_posix() in pathlib.Path(win.settings.fileName()).as_posix()
