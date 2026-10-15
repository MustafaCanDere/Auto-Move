"""2.0.0 pre-release fixes: AI flow changes, sub-flow grouping with recipes / operator panel, the operator panel and
test-akis inside sub-flows, headless production data, live recipes in nested sub-flows, command line help, locked
recovery, recipes dialog and the sub-flow library."""
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow.core import Graph, Node, NodeResult, PortType, StopFlow, load_builtin, register  # noqa: E402
from visionflow.core import recipes as rc  # noqa: E402
from visionflow.core import graph as vg  # noqa: E402
from visionflow.core.graph import node_at  # noqa: E402
from visionflow.core.operator import clean_operator, make_lock  # noqa: E402
from visionflow.nodes import subflow as sf  # noqa: E402
from visionflow.production import Tally, auto_config, db_path, default_image  # noqa: E402

load_builtin()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@register
class _PartFix(Node):
    """One part per frame (every third bad), ends after 6 frames."""
    type_name = "_PartFix"
    outputs = [("image", PortType.IMAGE), ("ok", PortType.BOOL)]
    is_source = True

    def setup(self):
        self.state["i"] = 0

    def process(self, inputs):
        self.state["i"] += 1
        if self.state["i"] > 6:
            raise StopFlow()
        return {"image": np.zeros((20, 30, 3), np.uint8), "ok": self.state["i"] % 3 != 0}


def pipeline():
    g = Graph()
    src = g.add_new("TestPattern", (0, 0), animate=False)
    gray = g.add_new("Grayscale", (260, 0))
    thr = g.add_new("Threshold", (520, 0))
    cnt = g.add_new("FindContours", (780, 0))
    g.connect(src.id, "image", gray.id, "image")
    g.connect(gray.id, "image", thr.id, "image")
    g.connect(thr.id, "image", cnt.id, "mask")
    return g, src, gray, thr, cnt


def roundtrip(g):
    return Graph.from_dict(json.loads(json.dumps(g.to_dict())))


@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app):
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    w.stop()
    w.close()
    app.processEvents()


# 1 --- AI "change the flow" keeps the operator settings and block colours -------------------------------------------

def test_ai_modify_keeps_operator_lock_and_block_colours(win):
    from visionflow.assistant import build_graph
    g, src, gray, thr, cnt = pipeline()
    thr.color = "#e53935"
    lock = make_lock("1234", iterations=2000)
    g.operator = clean_operator({"title": "Hat", "image": thr.id, "good": {"node": cnt.id, "port": "count"},
                                 "lock": lock}, g)
    win._set_root(g)
    data = {"nodes": [{"id": src.id, "type": "TestPattern", "values": {"animate": True}}] +   # counter removed
            [{"id": n.id, "type": n.type_name} for n in (gray, thr)],
            "connections": [{"src": src.id, "src_port": "image", "dst": gray.id, "dst_port": "image"},
                            {"src": gray.id, "src_port": "image", "dst": thr.id, "dst_port": "image"}]}
    new, errors = build_graph(data, current=win.root)
    assert not errors
    assert new.nodes[thr.id].color == "#e53935"
    win.apply_generated(new, modified=True)
    op = win.root.operator
    assert op["lock"] == lock and op["title"] == "Hat" and op["image"] == thr.id
    assert op["good"] is None                          # its block is gone: checked against the new flow
    assert win.root.nodes[thr.id].color == "#e53935"


# 2 --- grouping / ungrouping keeps recipes and operator outputs -----------------------------------------------------

def test_group_moves_recipe_values_and_operator_refs_into_the_subflow():
    g, src, gray, thr, cnt = pipeline()
    g.recipes = [{"name": "A", "values": {thr.id: {"value": 200}, src.id: {"animate": False}}}]
    g.operator = clean_operator({"image": thr.id, "good": {"node": cnt.id, "port": "count"}}, g)
    sub = sf.group(g, [gray.id, thr.id, cnt.id], "Ölç")
    key = f"{thr.id}/value"
    assert g.recipes[0]["values"] == {src.id: {"animate": False}, sub.id: {key: 200}}
    assert sub.is_exposed(thr.id, "value")
    assert g.operator["image"] == f"{sub.id}/{thr.id}"
    assert g.operator["good"] == {"node": f"{sub.id}/{cnt.id}", "port": "count"}
    g2 = roundtrip(g)                                  # saved and opened again: nothing dropped
    assert g2.recipes[0]["values"][sub.id] == {key: 200}
    assert g2.operator["good"] == {"node": f"{sub.id}/{cnt.id}", "port": "count"}
    assert g2.operator["image"] == f"{sub.id}/{thr.id}"
    assert not rc.apply(g2, g2.recipes[0])
    inner_thr = next(n for n in g2.nodes[sub.id].inner["nodes"] if n["id"] == thr.id)
    assert inner_thr["values"]["value"] == 200
    # and back
    ids = sf.ungroup(g2, sub.id)
    assert set(ids) == {gray.id, thr.id, cnt.id}
    assert g2.recipes[0]["values"] == {src.id: {"animate": False}, thr.id: {"value": 200}}
    assert g2.operator["image"] == thr.id
    assert g2.operator["good"] == {"node": cnt.id, "port": "count"}


def test_ungroup_maps_subflow_output_refs_and_renamed_ids():
    g, src, gray, thr, cnt = pipeline()
    sub = sf.group(g, [gray.id, thr.id], "Maske")
    g.operator = clean_operator({"image": sub.id}, g)
    g.recipes = [{"name": "A", "values": {sub.id: {}}}]
    sub.expose(thr.id, "value")
    g.recipes[0]["values"][sub.id][f"{thr.id}/value"] = 77
    clash = g.add_new("Blur", (0, 500))                # an outside block with the id of one inside
    g.nodes.pop(clash.id)
    clash.id = thr.id
    g.add(clash)
    ids = sf.ungroup(g, sub.id)
    new_thr = next(i for i in ids if g.nodes[i].type_name == "Threshold")
    assert new_thr != thr.id
    assert g.recipes[0]["values"] == {new_thr: {"value": 77}}
    assert g.operator["image"] == new_thr               # the block that gave the sub-flow's picture


def test_editor_group_and_ungroup_keep_recipes(win):
    g, src, gray, thr, cnt = pipeline()
    g.recipes = [{"name": "A", "values": {thr.id: {"value": 99}}}]
    win._set_root(g)
    for nid in (gray.id, thr.id):
        win.scene.items_by_id[nid].setSelected(True)
    item = win.scene.group_selected("Grup")
    sub_id = item.node.id
    assert win.root.recipes[0]["values"] == {sub_id: {f"{thr.id}/value": 99}}
    win.scene.ungroup(item.node)
    assert win.root.recipes[0]["values"] == {thr.id: {"value": 99}}


# 3 --- the operator panel reaches blocks inside sub-flows -----------------------------------------------------------

def nested_part_flow():
    g = Graph()
    src = g.add_new("_PartFix", (0, 0))
    prev = g.add_new("Preview", (260, 0))
    g.connect(src.id, "image", prev.id, "image")
    sub = sf.group(g, [prev.id], "Kontrol")
    return g, src, prev, sub


def test_default_image_auto_config_and_lists_walk_subflows(app):
    from visionflow.ui.operator import OperatorConfigDialog, _outputs
    g, src, prev, sub = nested_part_flow()
    assert default_image(g) == f"{sub.id}/{prev.id}"
    outer = sf.group(g, [src.id], "Kaynak")
    cfg = auto_config(g)
    assert cfg["lamp"] == {"node": f"{outer.id}/{src.id}", "port": "ok"}
    keys = {k: label for k, label, _ in vg.walk_nodes(g)}
    assert keys[f"{outer.id}/{src.id}"] == "Kaynak › _PartFix"
    assert not any(vg.find_node(g, k).type_name in sf.BOUNDARY for k in keys)
    texts = [t for t, _ in _outputs(g, (PortType.BOOL,))]
    assert any(t.startswith("Kaynak › ") and t.endswith("→ ok") for t in texts)
    g.operator = clean_operator({"image": f"{sub.id}/{prev.id}", "lamp": cfg["lamp"]}, g)
    assert g.operator["image"] == f"{sub.id}/{prev.id}" and g.operator["lamp"] == cfg["lamp"]
    dlg = OperatorConfigDialog(None, g)
    assert dlg.image_combo.currentData() == f"{sub.id}/{prev.id}"
    assert dlg.lamp_combo.currentData() == cfg["lamp"]
    assert "Kontrol › " in dlg.image_combo.currentText()
    dlg.deleteLater()


def test_find_node_static_and_running():
    g, src, prev, sub = nested_part_flow()
    assert node_at(g, f"{sub.id}/{prev.id}") is None            # not running
    assert vg.find_node(g, f"{sub.id}/{prev.id}").type_name == "Preview"
    assert vg.find_node(g, f"{sub.id}/yok") is None and vg.find_node(g, "yok/x") is None


# 4 --- F8 skips blocks that only failed because of a block before them ----------------------------------------------

def test_error_items_skip_follow_up_errors(win):
    from PySide6.QtCore import QPointF

    from visionflow.ui import editor_tools as et
    a = win.scene.add_node("Grayscale", QPointF(0, 0))
    b = win.scene.add_node("Threshold", QPointF(300, 0))
    a.set_result(1.0, "ValueError: bozuk", "", None)
    b.set_result(0.0, "Önceki blok hata verdi", "", None)
    assert [it.node.id for it in et.error_items(win.scene)] == [a.node.id]


# 5 --- the operator panel counts a sub-flow failure once ------------------------------------------------------------

def test_operator_panel_counts_subflow_error_once(app):
    from visionflow.ui.operator import OperatorPanel
    g, src, prev, sub = nested_part_flow()
    panel = OperatorPanel(g.to_dict())
    try:
        panel.on_frame({src.id: (1.0, None, "", None),
                        f"{sub.id}/{prev.id}": (1.0, "ValueError: bozuk", "", None),
                        sub.id: (1.0, "InnerError: içerideki 'Önizleme' bloğu: ValueError: bozuk", "", None)}, 10.0)
        assert "1 hata" in panel.state_label.text()
    finally:
        panel._left = True
        panel.close()


# 6 --- a counter's starting value is not counted --------------------------------------------------------------------

def test_tally_first_counter_reading_is_the_baseline():
    t = Tally({"good": {"node": "c", "port": "n"}, "total": {"node": "c", "port": "t"}})
    t.feed({"c": NodeResult({"n": 5000, "t": 5000}, 1.0)})       # a daily counter kept from an earlier run
    assert t.snapshot()["good"] == 0 and t.snapshot()["total"] == 0
    t.feed({"c": NodeResult({"n": 5002, "t": 5002}, 1.0)})
    assert t.snapshot()["good"] == 2 and t.snapshot()["total"] == 2


# 7 --- test-akis finds the player and the recorder inside sub-flows -------------------------------------------------

def make_recording(folder):
    from visionflow.core import create
    rec = create("FrameRecorder")
    rec.values.update(folder=str(folder), new_folder=False, names="a")
    rec.setup()
    for k in range(3):
        img = np.full((20, 30, 3), 40 * k, np.uint8)
        rec.process({"image": img, "trigger": None, "a": k, "b": None, "c": None, "d": None})
    rec.teardown()
    return folder


def test_flowtest_prepare_walks_subflows(tmp_path):
    from visionflow import flowtest
    make_recording(tmp_path / "kayit")
    g = Graph()
    pl = g.add_new("RecordingPlayer", (0, 0), folder="eski/yer", loop=True)
    rec = g.add_new("FrameRecorder", (300, 0), folder=str(tmp_path / "olmamali"))
    g.connect(pl.id, "image", rec.id, "image")
    inner = sf.group(g, [pl.id, rec.id], "İç")
    outer = sf.group(g, [inner.id], "Dış")
    notes = flowtest.prepare(g, str(tmp_path / "kayit"))
    nested = next(n for n in outer.inner["nodes"] if n["id"] == inner.id)["inner"]["nodes"]
    pnd = next(n for n in nested if n["type"] == "RecordingPlayer")
    rnd = next(n for n in nested if n["type"] == "FrameRecorder")
    assert pnd["values"]["folder"] == str(tmp_path / "kayit") and pnd["values"]["loop"] is False
    assert rnd.get("disabled") is True
    assert any("kayıttan okuyor" in n for n in notes) and any("testte kapalı" in n for n in notes)
    assert outer.is_source
    frames = g.run(max_frames=10)
    assert frames == 3 and not (tmp_path / "olmamali").exists()


# 8 --- headless / service runs collect production data --------------------------------------------------------------

def test_headless_run_writes_production_samples(tmp_path, monkeypatch):
    from visionflow import headless
    g = Graph()
    src = g.add_new("_PartFix", (0, 0))
    g.operator = clean_operator({"lamp": {"node": src.id, "port": "ok"}}, g)
    flow = tmp_path / "hat.json"
    g.save(str(flow))
    t = {"v": 0.0}

    def clock():
        t["v"] += 25.0                                  # every call 25 s later: a sample every few frames
        return t["v"]
    monkeypatch.setattr(headless, "_clock", clock)
    assert headless.run_flow(str(flow), fps=0, stats=1e9) == 0
    con = sqlite3.connect(str(db_path(str(flow))))
    try:
        rows = con.execute("SELECT flow, good, bad, total FROM samples").fetchall()
    finally:
        con.close()
    assert len(rows) >= 2 and {r[0] for r in rows} == {"hat"}
    assert (sum(r[1] for r in rows), sum(r[2] for r in rows), sum(r[3] for r in rows)) == (4, 2, 6)


def test_headless_without_operator_settings_writes_nothing(tmp_path):
    from visionflow import headless
    g = Graph()
    g.add_new("_PartFix", (0, 0))
    flow = tmp_path / "hat.json"
    g.save(str(flow))
    assert headless.run_flow(str(flow), fps=0) == 0
    assert not db_path(str(flow)).exists()


# 9 --- recipes: a sub-flow's name, nested sub-flows while running ---------------------------------------------------

def test_recipe_sets_subflow_name_and_title():
    g, src, gray, thr, cnt = pipeline()
    sub = sf.group(g, [gray.id, thr.id], "Eski")
    assert rc.set_value(g, sub.id, "name", "Yeni ad") is None
    assert sub.title == "Yeni ad"


def test_live_recipe_updates_intermediate_running_subflow():
    g, src, gray, thr, cnt = pipeline()
    sub = sf.group(g, [gray.id, thr.id], "İç")
    sub.expose(thr.id, "value")
    outer = sf.group(g, [sub.id], "Dış")
    key = f"{sub.id}/{thr.id}/value"
    assert outer.expose(sub.id, f"{thr.id}/value")
    g.recipes = [{"name": "A", "values": {outer.id: {key: 240}}}]
    g.setup()
    try:
        g.run_once()
        assert not rc.apply(g, g.recipes[0], live=True)
        assert node_at(g, f"{outer.id}/{sub.id}/{thr.id}").values["value"] == 240
        mid = node_at(g, f"{outer.id}/{sub.id}")
        assert mid.values[f"{thr.id}/value"] == 240
        assert next(n for n in mid.inner["nodes"] if n["id"] == thr.id)["values"]["value"] == 240
        mid.teardown()                                 # the sub-flow in between restarts: keeps the new value
        assert mid.safe_setup()
        assert node_at(g, f"{outer.id}/{sub.id}/{thr.id}").values["value"] == 240
    finally:
        g.teardown()


def test_worker_live_change_updates_intermediate_running_subflow(app):
    from visionflow.ui.worker import FlowWorker
    g, src, gray, thr, cnt = pipeline()
    sub = sf.group(g, [gray.id, thr.id], "İç")
    outer = sf.group(g, [sub.id], "Dış")
    w = FlowWorker(g.to_dict())
    live = Graph.from_dict(g.to_dict())
    live.setup()
    try:
        w.set_param(f"{outer.id}/{sub.id}/{thr.id}", "value", 33)
        w._apply_pending(live)
        mid = node_at(live, f"{outer.id}/{sub.id}")
        assert next(n for n in mid.inner["nodes"] if n["id"] == thr.id)["values"]["value"] == 33
    finally:
        live.teardown()


# 10 --- command line help --------------------------------------------------------------------------------------------

def run_main(*args):
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")}
    return subprocess.run([sys.executable, "-m", "visionflow", *args], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=120)


def test_help_prints_commands_without_qt():
    for flag in ("--help", "-h"):
        p = run_main(flag)
        assert p.returncode == 0
        for cmd in ("--headless", "--operator", "servis", "test-akis", "--selftest", "Düzenleyici"):
            assert cmd in p.stdout


def test_unknown_switch_exits_2():
    p = run_main("--bilinmeyen")
    assert p.returncode == 2 and "Bilinmeyen seçenek: --bilinmeyen" in p.stderr


# 11 --- a locked flow is not restored without its password -----------------------------------------------------------

def test_locked_recovery_asks_for_password(win, monkeypatch):
    from visionflow.ui import operator as op
    g, src, gray, thr, cnt = pipeline()
    g.operator = clean_operator({"lock": make_lock("1234", iterations=2000)}, g)
    win.recovery.write_text(json.dumps({"graph": g.to_dict(), "path": None}), encoding="utf-8")
    asked = []
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: asked.append(1) or False)
    assert not win.check_recovery(ask=False)
    assert asked and len(win.root.nodes) == 0
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: True)
    assert win.check_recovery(ask=False)
    assert len(win.root.nodes) == 4
    win.saved_snap = win._snap()


# 12 --- the recipes dialog lets go of the flow when closed with Esc --------------------------------------------------

def test_recipes_dialog_esc_disconnects_and_is_reused(win, app):
    sig = "2structure_changed()"
    base = win.scene.receivers(sig)
    dlg = win.open_recipes()
    assert win.scene.receivers(sig) == base + 1
    dlg.reject()                                       # what Esc does
    app.processEvents()
    assert win.scene.receivers(sig) == base
    again = win.open_recipes()
    assert again is dlg and win.scene.receivers(sig) == base + 1
    again.close()
    assert win.scene.receivers(sig) == base


# 13 --- the library keeps a sub-flow's colour ------------------------------------------------------------------------

def test_library_keeps_block_colour(tmp_path):
    g, src, gray, thr, cnt = pipeline()
    sub = sf.group(g, [gray.id, thr.id], "Renkli")
    sub.color = "#43a047"
    path = sf.save_to_library(sub, tmp_path)
    loaded, warnings = sf.load_from_library(path)
    assert loaded.color == "#43a047" and not warnings
    sub.color = None
    loaded, _ = sf.load_from_library(sf.save_to_library(sub, tmp_path))
    assert loaded.color is None
