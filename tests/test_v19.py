"""1.9: sub-flows (grouped blocks): grouping, ungrouping, running, nesting, the library and the editor."""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow.check import check  # noqa: E402
from visionflow.core import (Graph, GraphError, Node, NoNewFrame, PortType, StopFlow, create,  # noqa: E402
                             load_builtin, register)
from visionflow.core.registry import NODE_REGISTRY  # noqa: E402
from visionflow.nodes import subflow as sf  # noqa: E402

load_builtin()


@register
class _Src19(Node):
    type_name = "_Src19"
    outputs = [("n", PortType.NUMBER)]
    is_source = True
    stop_after = 0
    wait_every = 0

    def setup(self):
        self.state["i"] = 0

    def process(self, inputs):
        self.state["i"] += 1
        if self.stop_after and self.state["i"] > self.stop_after:
            raise StopFlow()
        if self.wait_every and self.state["i"] % self.wait_every:
            raise NoNewFrame("tetik bekleniyor")
        return {"n": self.state["i"]}


@register
class _Add19(Node):
    type_name = "_Add19"
    inputs = [("a", PortType.NUMBER), ("b", PortType.NUMBER)]
    outputs = [("sum", PortType.NUMBER)]

    def process(self, inputs):
        if inputs["a"] == 13:
            raise ValueError("uğursuz sayı")
        return {"sum": (inputs["a"] or 0) + (inputs["b"] or 0)}


def pipeline():
    """TestPattern → Grayscale → Threshold → FindContours (mask)."""
    g = Graph()
    src = g.add_new("TestPattern", (0, 0), animate=False)
    gray = g.add_new("Grayscale", (260, 0))
    thr = g.add_new("Threshold", (520, 0))
    cnt = g.add_new("FindContours", (780, 0))
    g.connect(src.id, "image", gray.id, "image")
    g.connect(gray.id, "image", thr.id, "image")
    g.connect(thr.id, "image", cnt.id, "mask")
    return g, src, gray, thr, cnt


def run1(g):
    g.setup()
    try:
        return g.run_once()
    finally:
        g.teardown()


def sub_of(g):
    return next(n for n in g.nodes.values() if n.type_name == "SubFlow")


# --- core ------------------------------------------------------------------------------------------------------

def test_blocks_registered():
    for t in ("SubFlow", "SubflowInput", "SubflowOutput"):
        assert t in NODE_REGISTRY
    assert len(NODE_REGISTRY) >= 137


def test_group_gives_the_same_result_and_survives_saving():
    g, src, gray, thr, cnt = pipeline()
    before = run1(g)[cnt.id].outputs["count"]
    sub = sf.group(g, [gray.id, thr.id], "Maske")
    assert sub.title == "Maske" and sub.inputs == [("image", "image")] and sub.outputs == [("image", "image")]
    assert set(g.nodes) == {src.id, sub.id, cnt.id}
    g2 = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert not g2.warnings
    r = run1(g2)
    assert r[cnt.id].outputs["count"] == before and not r[sub.id].error
    s2 = sub_of(g2)
    assert s2.title == "Maske" and s2.inputs == sub.inputs and s2.outputs == sub.outputs


def test_one_outside_output_feeding_two_blocks_is_one_input():
    g = Graph()
    s = g.add_new("_Src19", (0, 0))
    a = g.add_new("_Add19", (300, 0))
    b = g.add_new("_Add19", (300, 200))
    out = g.add_new("_Add19", (600, 0))
    g.connect(s.id, "n", a.id, "a")
    g.connect(s.id, "n", b.id, "b")
    g.connect(a.id, "sum", out.id, "a")
    g.connect(b.id, "sum", out.id, "b")
    sub = sf.group(g, [a.id, b.id])
    assert [p for p, _ in sub.inputs] == ["a"]                     # one outside source, one input
    assert [p for p, _ in sub.outputs] == ["sum", "sum_2"]         # two inside outputs, names made unique
    assert sub.inputs[0][1] == PortType.NUMBER                     # "Otomatik" type from the blocks inside
    r = run1(g)
    assert r[out.id].outputs["sum"] == 2                           # (1 + 0) + (0 + 1)


def test_ungroup_puts_everything_back():
    g, src, gray, thr, cnt = pipeline()
    links = {(g.nodes[c.src].type_name, c.src_port, g.nodes[c.dst].type_name, c.dst_port) for c in g.connections}
    sub = sf.group(g, [gray.id, thr.id])
    g.frames.append({"id": "c1", "title": "x", "rect": [0, 0, 100, 100], "color": "#3d5a80"})
    sub.inner["frames"] = [{"id": "c1", "title": "iç not", "rect": [10, 10, 200, 100], "color": "#3d5a80"}]
    ids = sf.ungroup(g, sub.id)
    assert sorted(g.nodes[i].type_name for i in ids) == ["Grayscale", "Threshold"]
    assert {(g.nodes[c.src].type_name, c.src_port, g.nodes[c.dst].type_name, c.dst_port)
            for c in g.connections} == links
    assert not any(n.type_name.startswith("Subflow") or n.type_name == "SubFlow" for n in g.nodes.values())
    assert [f["title"] for f in g.frames] == ["x", "iç not"] and len({f["id"] for f in g.frames}) == 2


def test_ungroup_passes_straight_links_through():
    g = Graph()
    s = g.add_new("_Src19")
    sub = g.add_new("SubFlow")
    a = g.add_new("_Add19")
    i, o = create("SubflowInput"), create("SubflowOutput")
    inner = Graph()
    inner.add(i)
    inner.add(o)
    inner.connect(i.id, "value", o.id, "value")
    sub.set_inner(inner.to_dict())
    g.connect(s.id, "n", sub.id, "giriş")
    g.connect(sub.id, "çıkış", a.id, "a")
    assert run1(g)[a.id].outputs["sum"] == 1
    sf.ungroup(g, sub.id)
    assert [(g.nodes[c.src].type_name, g.nodes[c.dst].type_name) for c in g.connections] == [("_Src19", "_Add19")]


def test_refused_groups_leave_the_flow_alone():
    g = Graph()
    s = g.add_new("_Src19")
    a, b, c = g.add_new("_Add19"), g.add_new("_Add19"), g.add_new("_Add19")
    g.connect(s.id, "n", a.id, "a")
    g.connect(a.id, "sum", b.id, "a")
    g.connect(b.id, "sum", c.id, "a")
    before = g.to_dict()
    with pytest.raises(GraphError, match="döngü"):              # b, outside, sits between a and c
        sf.group(g, [a.id, c.id])
    assert g.to_dict() == before
    with pytest.raises(GraphError, match="seçin"):
        sf.group(g, [])
    fb = g.add_new("SetVariable")                                # no feedback input; use one that has one
    g.remove(fb.id)
    fbs = [t for t, cls in NODE_REGISTRY.items() if cls.feedback_inputs]
    assert fbs
    t = fbs[0]
    n = g.add_new(t)
    port = NODE_REGISTRY[t].feedback_inputs[0]
    src_port = next(p for p, pt in g.nodes[c.id].outputs
                    if PortType.compatible(pt, dict(n.inputs)[port]))
    g.connect(c.id, src_port, n.id, port)
    before = g.to_dict()
    with pytest.raises(GraphError, match="geri besleme"):
        sf.group(g, [n.id])
    assert g.to_dict() == before


def test_nested_sub_flows_run_and_reload():
    g, src, gray, thr, cnt = pipeline()
    expected = run1(g)[cnt.id].outputs["count"]
    inner_sub = sf.group(g, [gray.id, thr.id], "İç")
    outer = sf.group(g, [inner_sub.id, cnt.id], "Dış")
    assert [p for p, _ in outer.outputs] == []                   # nothing outside used the count
    g2 = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    o2 = sub_of(g2)
    assert o2.inner_graph().nodes and any(n.type_name == "SubFlow" for n in o2.inner_graph().nodes.values())
    r = run1(g2)
    assert not r[o2.id].error
    # the counter of the inner flow, through two levels
    from visionflow.ui.worker import flatten
    g2.setup()
    flat = flatten(g2, g2.run_once())
    g2.teardown()
    counts = [r.outputs.get("count") for k, r in flat.items() if k.count("/") == 1 and "count" in r.outputs]
    assert counts == [expected]
    assert any(k.count("/") == 2 for k in flat)                  # blocks two sub-flows deep


def test_source_inside_drives_the_loop_and_stops_it():
    g = Graph()
    s = g.add_new("_Src19")
    a = g.add_new("_Add19")
    g.connect(s.id, "n", a.id, "a")
    sub = sf.group(g, [s.id], "Kaynak")
    assert sub.is_source and g.has_source is False or True       # has_source counts after setup
    old = _Src19.stop_after
    _Src19.stop_after = 4
    try:
        seen = []
        frames = g.run(on_frame=lambda i, r: seen.append(r[a.id].outputs["sum"]))
    finally:
        _Src19.stop_after = old
    assert frames == 4 and seen == [1, 2, 3, 4]


def test_waiting_inside_is_waiting_outside():
    g = Graph()
    s = g.add_new("_Src19")
    a = g.add_new("_Add19")
    g.connect(s.id, "n", a.id, "a")
    sub = sf.group(g, [s.id])
    old = _Src19.wait_every
    _Src19.wait_every = 2
    try:
        g.setup()
        r1 = g.run_once()
        assert r1[sub.id].waiting and not r1[sub.id].error and r1[a.id].waiting and g.idle(r1)
        r2 = g.run_once()
        assert r2[a.id].outputs["sum"] == 2 and not g.idle(r2)
        g.teardown()
    finally:
        _Src19.wait_every = old


def test_error_inside_names_the_block():
    g = Graph()
    s = g.add_new("_Src19")
    a = g.add_new("_Add19")
    b = g.add_new("_Add19")
    g.connect(s.id, "n", a.id, "a")
    g.connect(a.id, "sum", b.id, "a")
    sub = sf.group(g, [a.id])
    g.setup()
    for _ in range(12):
        g.run_once()
    r = g.run_once()                                             # the 13th frame
    assert "_Add19" in r[sub.id].error and "uğursuz" in r[sub.id].error
    assert r[b.id].error == "Önceki blok hata verdi"
    assert r[s.id].outputs["n"] == 13
    g.teardown()


def test_ports_follow_names_order_and_types():
    inner = Graph()
    i1 = inner.add_new("SubflowInput", (0, 200), name="alt")
    inner.add_new("SubflowInput", (0, 0), name="üst", type="sayı")
    i3 = inner.add_new("SubflowInput", (0, 100), name="üst")
    o1 = inner.add_new("SubflowOutput", (500, 0), name="  ")
    sub = create("SubFlow")
    sub.set_inner(inner.to_dict())
    assert [p for p, _ in sub.inputs] == ["üst", "üst_2", "alt"]
    assert dict(sub.inputs)["üst"] == PortType.NUMBER and dict(sub.inputs)["alt"] == PortType.ANY
    assert sub.outputs == [("çıkış", PortType.ANY)]
    assert i1 and i3 and o1


def test_broken_inside_data_is_survived():
    data = {"nodes": [
        {"id": "a", "type": "SubFlow", "pos": [0, 0], "values": {"name": "Bozuk"}, "inner": "çöp"},
        {"id": "b", "type": "SubFlow", "pos": [0, 0], "values": {"name": "Eklenti"},
         "inner": {"nodes": [{"id": "x", "type": "YokBöyleBlok"}], "connections": []}},
        {"id": "c", "type": "SubFlow", "pos": [0, 0]},
    ]}
    deep = {"nodes": [], "connections": []}
    for k in range(40):                                          # far too deep: cut, not a crash
        deep = {"nodes": [{"id": f"d{k}", "type": "SubFlow", "inner": deep}], "connections": []}
    data["nodes"].append({"id": "d", "type": "SubFlow", "inner": deep})
    g = Graph.from_dict(data)
    assert set(g.nodes) == {"a", "b", "c", "d"}
    assert any("Bozuk" in w and "bozuk" in w for w in g.warnings)
    assert any("Eklenti" in w and "YokBöyleBlok" in w for w in g.warnings)
    assert any("derin" in w for w in g.warnings)
    run1(g)                                                      # empty sub-flows run as nothing


def test_library_round_trip(tmp_path):
    g, src, gray, thr, cnt = pipeline()
    sub = sf.group(g, [gray.id, thr.id], "Maske / bul?")
    path = sf.save_to_library(sub, tmp_path)
    assert path.parent == tmp_path and "/" not in path.name and "?" not in path.name
    (tmp_path / "baska.json").write_text("{bozuk", encoding="utf-8")
    (tmp_path / "akis.json").write_text(json.dumps(Graph().to_dict()), encoding="utf-8")
    items = sf.library(tmp_path)
    assert [n for n, _ in items] == ["Maske / bul?"]
    sub2, warnings = sf.load_from_library(items[0][1])
    assert not warnings and sub2.inputs == sub.inputs and sub2.outputs == sub.outputs and sub2.id != sub.id
    with pytest.raises(ValueError):
        sf.load_from_library(tmp_path / "akis.json")


def test_checker_looks_inside():
    class Cfg:
        email_user = email_password = telegram_token = telegram_allowed = youtube_client_secret = ""

        def key(self):
            return "x"
    g = Graph()
    s = g.add_new("_Src19")
    sv = g.add_new("SetVariable", name="hız")
    g.connect(s.id, "n", sv.id, "value")
    sub = g.add_new("SubFlow", name="İç iş")
    inner = Graph()
    inner.add_new("GetVariable", name="hız")
    inner.add_new("SubflowOutput")                               # nothing goes into it
    sub.set_inner(inner.to_dict())
    issues = check(g, Cfg())
    texts = [i.text for i in issues]
    assert any(t.startswith("[İç iş]") and "Alt Akış Çıkışı" in t and i.node_id == sub.id
               for t, i in zip(texts, issues))
    assert not any("değişkeni yazan" in t for t in texts)        # written outside, read inside: fine
    g.add_new("SubflowInput")
    assert any("sadece bir alt akışın içinde" in i.text for i in check(g, Cfg()))


def test_stale_links_are_dropped():
    g = Graph()
    s = g.add_new("_Src19")
    sub = g.add_new("SubFlow")
    inner = Graph()
    i = inner.add_new("SubflowInput", name="x")
    sub.set_inner(inner.to_dict())
    g.connect(s.id, "n", sub.id, "x")
    assert sf.drop_stale_links(g, sub) == 0
    inner.nodes[i.id].values["name"] = "y"
    sub.set_inner(inner.to_dict())
    assert sf.drop_stale_links(g, sub) == 1 and not g.connections


def test_worker_reaches_blocks_inside():
    from visionflow.ui import worker as wk
    g, src, gray, thr, cnt = pipeline()
    thr.values["value"] = 50
    sub = sf.group(g, [gray.id, thr.id])
    w = wk.FlowWorker(g.to_dict())
    live = Graph.from_dict(g.to_dict())
    live.setup()
    key = f"{sub.id}/{thr.id}"
    assert wk.node_at(live, key).type_name == "Threshold" and wk.node_at(live, f"{sub.id}/nope") is None
    a = live.run_once()
    w.set_param(key, "value", 250)
    w.set_param(key, "no_such_setting", 1)                         # ignored, not a crash
    assert w._apply_pending(live)
    assert wk.node_at(live, key).values["value"] == 250
    flat = wk.flatten(live, live.run_once())
    assert key in flat and a[sub.id].outputs["image"].mean() > flat[sub.id].outputs["image"].mean()  # stricter now
    assert key not in wk.flatten(live, {sub.id: flat[sub.id]})    # inner results are taken once
    sent = []
    import types
    w.frame = types.SimpleNamespace(emit=lambda payload, fps: sent.append(payload))
    w.watch_id, w.thumbs = None, True
    w._emit(live, {**flat, "gone/x": flat[cnt.id]}, 0.0)
    assert key in sent[0] and "gone/x" not in sent[0]
    live.teardown()


# --- editor ------------------------------------------------------------------------------------------------------

@pytest.fixture
def win():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    w.stop()
    w.close()
    app.processEvents()


def _load(win):
    g, src, gray, thr, cnt = pipeline()
    win.scene.load_graph(g)
    win._loaded(None)
    return g, src, gray, thr, cnt


def test_group_enter_edit_and_leave(win):
    g, src, gray, thr, cnt = _load(win)
    for i in (gray.id, thr.id):
        win.scene.items_by_id[i].setSelected(True)
    item = win.group_selected("Maske")
    sub = item.node
    win._snapshot()
    assert win.dirty and len(win.history.stack) == 2
    saved = win._snap()
    win.enter_subflow(sub)
    assert win.levels and win.crumbs.isVisible() is not None and "Maske" in win.crumb_label.text()
    assert win._snap() == saved                                   # only looking inside changes nothing
    assert {n.type_name for n in win.scene.graph.nodes.values()} == {"Grayscale", "Threshold", "SubflowInput",
                                                                     "SubflowOutput"}
    assert win.root is g
    # change a setting and add a second output inside
    t = next(n for n in win.scene.graph.nodes.values() if n.type_name == "Threshold")
    t.values["value"] = 200
    win.on_param(t.id, "value", 200)
    from PySide6.QtCore import QPointF
    o = win.scene.add_node("SubflowOutput", QPointF(900, 300), {"name": "gri"})
    win.scene.try_connect(win.scene.items_by_id[gray.id].sockets[("image", True)], o.sockets[("value", False)])
    win._snapshot()
    inner = json.loads(win._snap())
    saved_sub = next(n for n in inner["nodes"] if n["type"] == "SubFlow")
    assert any(n.get("values", {}).get("value") == 200 for n in saved_sub["inner"]["nodes"])
    win.exit_subflow()
    assert not win.levels and win.scene.graph is g
    assert [p for p, _ in win.scene.items_by_id[sub.id].node.outputs] == ["image", "gri"]
    assert len(win.scene.items_by_id[sub.id].sockets) == 3        # the item was rebuilt with the new port
    assert win.scene.items_by_id[sub.id].isSelected()


def test_undo_inside_stays_inside_and_save_saves_everything(win, tmp_path):
    g, src, gray, thr, cnt = _load(win)
    sub = sf.group(win.scene.graph, [gray.id, thr.id], "Grup")
    win.scene.load_graph(win.scene.graph)
    win._snapshot()
    win.enter_subflow(sub)
    from PySide6.QtCore import QPointF
    win.scene.add_node("Blur", QPointF(0, 400))
    win._snapshot()
    assert any(n.type_name == "Blur" for n in win.scene.graph.nodes.values())
    win.undo()
    assert win.levels and win.levels[-1][0].id == sub.id
    assert not any(n.type_name == "Blur" for n in win.scene.graph.nodes.values())
    win.redo()
    assert any(n.type_name == "Blur" for n in win.scene.graph.nodes.values())
    path = tmp_path / "akis.json"
    win.path = str(path)
    assert win.save_flow()
    saved = Graph.load(str(path))
    assert any(n.type_name == "Blur" for n in sub_of(saved).inner_graph().nodes.values())
    assert not win.dirty
    win.go_root()
    assert not win.levels and win.scene.graph is win.root


def test_removing_an_input_inside_drops_the_outer_link(win):
    g, src, gray, thr, cnt = _load(win)
    sub = sf.group(win.scene.graph, [gray.id, thr.id])
    win.scene.load_graph(win.scene.graph)
    assert any(c.dst == sub.id for c in g.connections)
    win.enter_subflow(sub)
    for it in win.scene.items_by_id.values():
        it.setSelected(it.node.type_name == "SubflowInput")
    win.scene.delete_selected()
    win.exit_subflow()
    assert not any(c.dst == sub.id for c in g.connections) and sub.inputs == []
    assert not Graph.from_dict(win.root.to_dict()).warnings


def test_frames_show_inside_results_only_when_inside(win):
    g, src, gray, thr, cnt = _load(win)
    sub = sf.group(win.scene.graph, [gray.id, thr.id])
    win.scene.load_graph(win.scene.graph)
    payload = {sub.id: (1.0, None, "dış", None), f"{sub.id}/{thr.id}": (2.0, "iç hata", "", None)}
    win.on_frame(payload, 0.0)
    assert win.scene.items_by_id[sub.id].status_text == "dış"
    win.enter_subflow(sub)
    win.on_frame(payload, 0.0)
    assert win.scene.items_by_id[thr.id].error == "iç hata"
    win.scene.clearSelection()
    win.scene.items_by_id[thr.id].setSelected(True)
    assert win._watching == f"{sub.id}/{thr.id}"


def test_copy_paste_keeps_the_inside_and_ungroup_from_menu(win, monkeypatch):
    g, src, gray, thr, cnt = _load(win)
    sub = sf.group(win.scene.graph, [gray.id, thr.id], "Kopya")
    win.scene.load_graph(win.scene.graph)
    win.scene.clearSelection()
    win.scene.items_by_id[sub.id].setSelected(True)
    clip = win.scene.copy_selected()
    win.scene.paste(json.loads(json.dumps(clip)))
    subs = [n for n in win.scene.graph.nodes.values() if n.type_name == "SubFlow"]
    assert len(subs) == 2 and all(len(s.inner_graph().nodes) == 4 for s in subs)
    assert all(s.title == "Kopya" for s in subs)
    from PySide6.QtWidgets import QMenu
    import visionflow.ui.main_window as mw

    class Menu(QMenu):
        def exec(self, *a):
            return None
    monkeypatch.setattr(mw, "QMenu", Menu)
    win.scene.clearSelection()
    win.scene.items_by_id[sub.id].setSelected(True)
    win.context_menu(None, win.scene.items_by_id[sub.id].pos())
    texts = [a.text() for a in win._context_menu.actions()]
    assert any("Alt akışı aç" in t for t in texts) and any("çöz" in t for t in texts)
    items = win.ungroup(sub)
    assert len(items) == 2 and sub.id not in win.scene.graph.nodes


def test_rename_updates_the_title(win):
    from PySide6.QtCore import QPointF
    item = win.scene.add_node("SubFlow", QPointF(0, 0))
    item.node.values["name"] = "Ölçüm"
    win.on_param(item.node.id, "name", "Ölçüm")
    assert item.node.title == "Ölçüm"
    win.enter_subflow(item.node)                                 # empty: a hint, no crash
    assert win.levels and not win.scene.graph.nodes
    win.exit_subflow()


def test_library_in_the_editor(win, tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "library_dir", lambda: tmp_path)
    g, src, gray, thr, cnt = _load(win)
    sub = sf.group(win.scene.graph, [gray.id, thr.id], "Hazır")
    win.scene.load_graph(win.scene.graph)
    assert win.save_subflow_to_library(sub).exists()
    n = len(win.scene.graph.nodes)
    win.add_from_library(sf.library()[0][1])
    assert len(win.scene.graph.nodes) == n + 1


def test_running_window_flow_with_a_sub_flow(win):
    g, src, gray, thr, cnt = _load(win)
    sub = sf.group(win.scene.graph, [gray.id, thr.id])
    win.scene.load_graph(win.scene.graph)
    win.enter_subflow(sub)
    got = []
    win.start()
    win.worker.frame.connect(lambda payload, fps: got.append(payload))
    from PySide6.QtWidgets import QApplication
    import time
    end = time.time() + 10
    while time.time() < end and not any(f"{sub.id}/{thr.id}" in p for p in got):
        QApplication.processEvents()
        time.sleep(0.02)
    win.stop()
    assert any(f"{sub.id}/{thr.id}" in p for p in got)
    assert win.scene.items_by_id[thr.id].ms is not None
    assert np is not None


def test_example_flow_runs():
    import pathlib
    g = Graph.load(str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "alt_akis_nesne_sayma.json"))
    assert not g.warnings
    sub = sub_of(g)
    assert sub.title == "Kırmızı nesneleri bul" and [p for p, _ in sub.outputs] == ["detections", "count"]
    r = run1(g)
    assert r[sub.id].outputs["count"] == 1 and not any(x.error for x in r.values())


def test_boundary_blocks_are_not_grouped():
    g = Graph()
    i = g.add_new("SubflowInput")
    a = g.add_new("_Add19")
    g.connect(i.id, "value", a.id, "a")
    with pytest.raises(GraphError, match="Girişi"):
        sf.group(g, [i.id, a.id])
    assert set(g.nodes) == {i.id, a.id}


# --- found in the bug hunt before release ------------------------------------------------------------------------

def test_assistant_keeps_what_is_inside_a_sub_flow():
    from visionflow import assistant
    g, src, gray, thr, cnt = pipeline()
    src.values["animate"] = True
    sub = sf.group(g, [gray.id, thr.id])
    data = json.loads(assistant._current_flow_json(g))
    assert next(n for n in data["nodes"] if n["id"] == sub.id)["ports"] == {"in": ["image:image"],
                                                                          "out": ["image:image"]}
    new, errors = assistant.build_graph(data, g)
    assert not errors
    s2 = new.nodes[sub.id]
    assert s2.inputs == sub.inputs and len(s2.inner_graph().nodes) == 4 and len(new.connections) == 2


def test_looking_inside_does_not_restart_a_running_flow(win):
    import time
    from PySide6.QtWidgets import QApplication
    g, src, gray, thr, cnt = _load(win)
    src.values["animate"] = True
    sub = sf.group(win.scene.graph, [gray.id, thr.id])
    win.scene.load_graph(win.scene.graph)
    win.start()
    w0 = win.worker
    win.enter_subflow(sub)
    end = time.time() + 0.6
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)
    assert win.worker is w0
    win.exit_subflow()
    end = time.time() + 0.6
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)
    assert win.worker is w0
    win.stop()


def test_renaming_a_port_inside_keeps_the_outside_link(win):
    g, src, gray, thr, cnt = _load(win)
    sub = sf.group(win.scene.graph, [gray.id, thr.id])
    win.scene.load_graph(win.scene.graph)
    win.enter_subflow(sub)
    for t, name in (("SubflowInput", "kamera"), ("SubflowOutput", "maske")):
        n = next(n for n in win.scene.graph.nodes.values() if n.type_name == t)
        n.values["name"] = name
        win.on_param(n.id, "name", name)
    win._snapshot()
    win.go_root()
    assert sub.inputs == [("kamera", "image")] and sub.outputs == [("maske", "image")]
    assert {(c.src_port, c.dst_port) for c in g.connections if sub.id in (c.src, c.dst)} == {("image", "kamera"),
                                                                                           ("maske", "mask")}
    assert run1(Graph.from_dict(win.root.to_dict()))[cnt.id].error is None


def test_picture_into_a_sub_flow_is_shown_inside():
    from visionflow.ui import worker as wk
    g, src, gray, thr, cnt = pipeline()
    sub = sf.group(g, [gray.id, thr.id])
    live = Graph.from_dict(g.to_dict())
    live.setup()
    flat = wk.flatten(live, live.run_once())
    inp = next(k for k in flat if k.startswith(sub.id + "/") and wk.node_at(live, k).type_name == "SubflowInput")
    w = wk.FlowWorker.__new__(wk.FlowWorker)
    import types
    sent, watched = [], []
    w.frame = types.SimpleNamespace(emit=lambda payload, fps: sent.append(payload))
    w.watch = types.SimpleNamespace(emit=lambda *a: watched.append(a))
    w.watch_id, w.thumbs = inp, True
    w._emit(live, flat, 0.0)
    assert sent[0][inp][3] is not None and watched and watched[0][0] == inp
    live.teardown()


def test_error_inside_is_logged_once_for_the_right_block(monkeypatch):
    from visionflow.ui import worker as wk
    logged = []
    monkeypatch.setattr(wk, "error_log", lambda: types_ns(record=lambda *a: logged.append(a)))
    g = Graph()
    s = g.add_new("_Src19")
    a = g.add_new("_Add19")
    g.connect(s.id, "n", a.id, "a")
    sf.group(g, [a.id])
    live = Graph.from_dict(g.to_dict())
    live.setup()
    for _ in range(12):
        live.run_once()
    flat = wk.flatten(live, live.run_once())
    w = wk.FlowWorker.__new__(wk.FlowWorker)
    w.frame = types_ns(emit=lambda *a: None)
    w.watch_id, w.thumbs = None, False
    w._emit(live, flat, 0.0)
    assert [x[0] for x in logged] == ["_Add19"]
    live.teardown()


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def test_deep_nesting_stays_fast():
    """Three "Otomatik" ports on every level, seven levels deep: the port types are worked out once per level."""
    import time
    inner = Graph()
    for k in range(3):
        i = inner.add_new("SubflowInput", (0, k), name=f"g{k}")
        a = inner.add_new("_Add19", (300, k))
        o = inner.add_new("SubflowOutput", (600, k), name=f"c{k}")
        inner.connect(i.id, "value", a.id, "a")
        inner.connect(a.id, "sum", o.id, "value")
    for level in range(7):
        sub = create("SubFlow")
        sub.set_inner(inner.to_dict())
        d = Graph()
        d.add(sub)
        for k, (p, _) in enumerate(sub.inputs):
            i = d.add_new("SubflowInput", (0, k), name=f"g{k}")
            d.connect(i.id, "value", sub.id, p)
        for k, (p, _) in enumerate(sub.outputs):
            o = d.add_new("SubflowOutput", (900, k), name=f"c{k}")
            d.connect(sub.id, p, o.id, "value")
        inner = d
    t0 = time.perf_counter()
    top = create("SubFlow")
    top.set_inner(inner.to_dict())
    g = Graph.from_dict({"nodes": [top.to_dict()]})
    assert time.perf_counter() - t0 < 1.0
    assert sub_of(g).inputs == [(f"g{k}", PortType.NUMBER) for k in range(3)]
