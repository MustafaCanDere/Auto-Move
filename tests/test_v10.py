"""1.0: tolerance check, crop a detection, smooth a value, production rate."""
import numpy as np
import pytest

from visionflow.core import Graph, create, load_builtin

load_builtin()


def make(t, **v):
    n = create(t)
    n.values.update(v)
    n.setup()
    return n


def test_tolerance_check_counts_parts_and_yield():
    n = make("ToleranceCheck", low=9.9, high=10.1, nominal=10.0)
    outs = [n.process({"value": v, "trigger": t, "reset": None})
            for v, t in [(10.0, True), (10.0, False), (10.3, True), (10.3, True), (9.8, False), (9.8, True)]]
    last = outs[-1]
    assert (last["ok_count"], last["nok_count"]) == (1, 2)       # held trigger counts once
    assert last["yield"] == pytest.approx(33.33, abs=0.01)
    assert outs[2]["text"] == "ÜST SINIR ÜSTÜNDE" and outs[2]["nok"] and not outs[2]["ok"]
    assert last["text"] == "ALT SINIR ALTINDA" and last["deviation"] == pytest.approx(-0.2)
    n.process({"value": 10.0, "trigger": False, "reset": True})
    assert n.process({"value": 10.0, "trigger": True, "reset": True})["ok_count"] == 1   # reset is an edge


def test_tolerance_check_without_trigger_counts_changes_and_handles_bad_values():
    n = make("ToleranceCheck", low=10.1, high=9.9, missing="sayma")       # swapped limits still work
    for v in (10.0, 10.0, float("nan"), "abc", None, 10.05):
        out = n.process({"value": v, "trigger": None, "reset": None})
    assert (out["ok_count"], out["nok_count"]) == (2, 0)
    assert n.process({"value": None, "trigger": None, "reset": None})["text"] == "ölçüm yok"
    m = make("ToleranceCheck")
    m.process({"value": float("inf"), "trigger": True, "reset": None})
    assert m.process({"value": None, "trigger": None, "reset": None})["nok_count"] == 1


def test_crop_detection_picks_and_clamps():
    img = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3).astype(np.uint8)
    dets = [{"box": (10, 10, 20, 20), "score": 0.9, "label": "plaka #1"},
            {"box": (150, 60, 80, 80), "score": 0.5, "label": "araba #2"}]
    n = make("CropDetection")
    out = n.process({"image": img, "detections": dets})
    assert out["image"].shape == (20, 20, 3) and out["label"] == "plaka #1" and out["found"]
    n.values.update(pick="en büyük", margin=5)
    out = n.process({"image": img, "detections": dets})
    assert out["image"].shape == (45, 55, 3)                      # clamped to the picture edge
    n.values.update(pick="sıra no", index=5)
    assert n.process({"image": img, "detections": dets})["found"] is False
    n.values.update(pick="en yüksek güven", label="Araba", margin=0)
    assert n.process({"image": img, "detections": dets})["label"] == "araba #2"
    assert n.process({"image": img, "detections": [{"box": (500, 500, 5, 5), "score": 1}]})["found"] is False
    assert n.process({"image": None, "detections": dets})["image"] is None


def test_smooth_value_median_ignores_spikes():
    n = make("SmoothValue", window=3)
    for v in (10, 10.2, 99, None, "x"):
        out = n.process({"value": v, "reset": None})
    assert out["value"] == 10.2 and out["ready"] and out["max"] == 99
    n.values.update(mode="ortalama", window=2)
    assert n.process({"value": 1, "reset": None})["value"] == 50.0
    assert n.process({"value": None, "reset": True}) == {"value": 0.0, "min": 0.0, "max": 0.0, "ready": False}


def test_production_rate_per_minute_slow_and_counter_reset():
    n = create("ProductionRate")
    clock = {"t": 1000.0}
    n.now = lambda: clock["t"]
    n.values.update(window=60, target=20)
    n.setup()
    out = None
    for i in range(0, 121):                     # one part every 2 s for two minutes = 30 / min
        clock["t"] = 1000.0 + i * 2
        out = n.process({"count": i})
    assert out["per_minute"] == pytest.approx(30, abs=1) and out["per_hour"] == pytest.approx(1800, abs=60)
    assert out["slow"] is False and out["idle_seconds"] == 0
    for k in range(1, 40):                      # the line stops
        clock["t"] = 1240.0 + k * 2
        out = n.process({"count": 120})
    assert out["slow"] is True and out["per_minute"] < 20 and out["idle_seconds"] == 78
    clock["t"] += 1
    out = n.process({"count": 0})               # counter reset by the operator
    assert out["per_minute"] == 0


def test_new_blocks_chain_in_a_flow():
    g = Graph()
    num = g.add_new("Number", value=10.05)
    smooth = g.add_new("SmoothValue", window=1)
    tol = g.add_new("ToleranceCheck")
    rate = g.add_new("ProductionRate")
    g.connect(num.id, "value", smooth.id, "value")
    g.connect(smooth.id, "value", tol.id, "value")
    g.connect(tol.id, "ok_count", rate.id, "count")
    g.setup()
    res = g.run_once()
    assert res[tol.id].outputs["ok"] and res[tol.id].outputs["ok_count"] == 1 and not res[rate.id].error


def test_shift_report_counts_each_shift_and_survives_counter_reset():
    import datetime as dt
    n = create("ShiftReport")
    clock = {"t": dt.datetime(2026, 10, 12, 7, 0)}
    n.now = lambda: clock["t"]
    n.values.update(times="08:00, 16:00", extra_name="Ortalama çap")
    n.setup()
    feed = lambda ok, nok, extra=None, now=False: n.process({"ok": ok, "nok": nok, "extra": extra, "now": now})  # noqa: E731
    assert feed(500, 20)["shift_ok"] == 0                 # counts from before the flow started are not this shift
    feed(540, 21)
    feed(3, 0)                                            # operator reset the counter
    out = feed(10, 1, 10.04, now=True)
    assert out["report"] and (out["shift_ok"], out["shift_nok"]) == (50, 2)
    assert "Sağlam: 50" in out["text"] and "Ortalama çap: 10.04" in out["text"]
    assert feed(10, 1, now=True)["report"] is False       # manual report is an edge
    clock["t"] = dt.datetime(2026, 10, 12, 8, 0, 30)
    out = feed(12, 1)
    assert out["report"] and "Sağlam: 52" in out["text"] and out["yield"] == pytest.approx(96.3, abs=0.1)
    assert feed(12, 1)["report"] is False and feed(15, 1)["shift_ok"] == 3      # new shift starts from 0
    clock["t"] = dt.datetime(2026, 10, 12, 16, 1)
    assert "07:00" not in feed(15, 1)["text"] and "08:00 – 16:01" in n.state["text"]


def test_roi_values_and_shapes():
    from visionflow.ui.roi import shape_from_values, values_from_points
    assert values_from_points(("rect", "roi"), [(50, 40), (10, 10)], (100, 100)) == {"roi": "10,10,41,31"}
    assert values_from_points(("rect", ("x", "y", "w", "h")), [(-5, -5), (200, 20)], (100, 50)) == \
        {"x": 0, "y": 0, "w": 100, "h": 21}                       # clamped to the picture
    assert values_from_points(("rect", "roi"), [(5, 5), (6, 6)], (100, 100)) is None     # a click, not a drag
    assert values_from_points(("line", "line"), [(0, 50), (30, 50), (99, 50)], (100, 100)) == {"line": "0,50,99,50"}
    assert values_from_points(("quad", "points"), [(0, 0), (9, 0), (9, 9)], (10, 10)) is None
    assert values_from_points(("poly", "zone"), [(1, 1), (8, 1), (8, 8)], (10, 10)) == {"zone": "1,1,8,1,8,8"}
    assert shape_from_values(("rect", "roi"), {"roi": ""}) is None
    assert shape_from_values(("rect", "roi"), {"roi": "1,2,3,4"}) == ("rect", (1, 2, 3, 4))
    assert shape_from_values(("poly", "zone"), {"zone": "1,1,5,1,5"}) is None
    assert shape_from_values(("rect", ("x", "y", "w", "h")), {"x": 1, "y": 2, "w": 3, "h": 4}) == ("rect", (1, 2, 3, 4))


def test_drawing_a_region_on_the_preview_sets_the_block(tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    g = Graph()
    s_node, c_node = g.add_new("TestPattern"), g.add_new("Crop")
    g.connect(s_node.id, "image", c_node.id, "image")
    w.scene.load_graph(g)
    w.history.reset(w._snap())
    src, crop = w.scene.items_by_id[s_node.id], w.scene.items_by_id[c_node.id]
    crop.setSelected(True)
    app.processEvents()
    assert w.watched_id(crop.node) == src.node.id                       # the picture going in, not the crop
    big = np.zeros((960, 1280, 3), np.uint8)
    w.on_watch(src.node.id, big[::2, ::2].copy(), (1280, 960))          # shown shrunk, real size 1280x960
    lab = w.preview.label
    ox, oy, pw, ph = w.preview._pm_rect
    to = lambda x, y: QPoint(int(ox + x * pw / 1280), int(oy + y * ph / 960))  # noqa: E731
    QTest.mousePress(lab, Qt.LeftButton, Qt.NoModifier, to(100, 200))
    QTest.mouseMove(lab, to(500, 600))
    QTest.mouseRelease(lab, Qt.LeftButton, Qt.NoModifier, to(500, 600))
    v = crop.node.values
    assert abs(v["x"] - 100) <= 4 and abs(v["y"] - 200) <= 4 and abs(v["w"] - 400) <= 8 and abs(v["h"] - 400) <= 8
    w.undo()
    assert w.scene.graph.nodes[c_node.id].values["x"] == 0              # one undo step
    w.saved_snap = w._snap()
    w.close()


def test_drawing_a_polygon_zone_with_clicks():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    g = Graph()
    s_node, z_node = g.add_new("TestPattern"), g.add_new("ZoneCounter")
    g.connect(s_node.id, "image", z_node.id, "image")
    w.scene.load_graph(g)
    w.scene.items_by_id[z_node.id].setSelected(True)
    app.processEvents()
    w.on_watch(s_node.id, np.zeros((480, 640, 3), np.uint8), (640, 480))
    ox, oy, pw, ph = w.preview._pm_rect
    to = lambda x, y: QPoint(int(ox + x * pw / 640), int(oy + y * ph / 480))  # noqa: E731
    for x, y in [(100, 100), (500, 100), (500, 400), (100, 400)]:
        QTest.mouseClick(w.preview.label, Qt.LeftButton, Qt.NoModifier, to(x, y))
    QTest.mouseClick(w.preview.label, Qt.RightButton, Qt.NoModifier, to(300, 300))
    pts = [int(v) for v in w.scene.graph.nodes[z_node.id].values["zone"].split(",")]
    assert len(pts) == 8 and all(abs(a - b) <= 3 for a, b in zip(pts, [100, 100, 500, 100, 500, 400, 100, 400]))
    w.saved_snap = w._snap()
    w.close()
