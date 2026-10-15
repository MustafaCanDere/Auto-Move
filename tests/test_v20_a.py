"""2.0 (a): exposed settings of sub-flows ("Dışarı açılan ayarlar") and recipes / product types ("Reçeteler")."""
import json
import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow.core import Graph, Node, Param, PortType, StopFlow, load_builtin, register  # noqa: E402
from visionflow.core import recipes as rc  # noqa: E402
from visionflow.core.graph import node_at  # noqa: E402
from visionflow.core.registry import NODE_REGISTRY  # noqa: E402
from visionflow.nodes import subflow as sf  # noqa: E402

load_builtin()


@register
class _Img20(Node):
    """A horizontal gray ramp 0…255 (so a threshold of t keeps about (255 - t) / 256 of the pixels)."""
    type_name = "_Img20"
    outputs = [("image", PortType.IMAGE)]
    is_source = True
    params = [Param("frames", "int", 0, "kare", 0, 1000)]

    def setup(self):
        self.state["n"] = 0

    def process(self, inputs):
        self.state["n"] += 1
        if self.v("frames") and self.state["n"] > self.v("frames"):
            raise StopFlow()
        ramp = np.tile(np.arange(256, dtype=np.uint8), (20, 1))
        return {"image": cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)}


@register
class _Gain20(Node):
    type_name = "_Gain20"
    inputs = [("x", PortType.NUMBER)]
    outputs = [("y", PortType.NUMBER)]
    params = [Param("k", "float", 1.0, "Kazanç", 0, 100), Param("mode", "choice", "a", "Kip", choices=["a", "b"]),
              Param("on", "bool", True, "Açık")]

    def process(self, inputs):
        return {"y": (inputs["x"] or 0) * self.v("k")}


@register
class _Setup20(Node):
    """Reads "k" in setup (like a model path): counts its setups."""
    type_name = "_Setup20"
    inputs = [("x", PortType.NUMBER)]
    outputs = [("y", PortType.NUMBER)]
    params = [Param("k", "int", 1, "Çarpan", 0, 1000), Param("note", "str", "", "Not")]
    setup_params = ("k",)
    setups = 0

    def setup(self):
        type(self).setups += 1
        self.state["k"] = self.v("k")

    def process(self, inputs):
        return {"y": (inputs["x"] or 0) * self.state["k"]}


@register
class _Num20(Node):
    type_name = "_Num20"
    outputs = [("n", PortType.NUMBER)]
    is_source = True
    params = [Param("value", "float", 2.0, "Değer")]

    def process(self, inputs):
        return {"n": self.v("value")}


def ramp_flow():
    """_Img20 → [Grayscale → Threshold] sub-flow → mask out."""
    g = Graph()
    src = g.add_new("_Img20", (0, 0))
    gray = g.add_new("Grayscale", (260, 0))
    thr = g.add_new("Threshold", (520, 0))
    g.connect(src.id, "image", gray.id, "image")
    g.connect(gray.id, "image", thr.id, "image")
    after = g.add_new("Grayscale", (780, 0))                     # makes "image" an output of the sub-flow
    g.connect(thr.id, "image", after.id, "image")
    sub = sf.group(g, [gray.id, thr.id], "Eşik grubu")
    return g, src, sub, gray.id, thr.id


def run1(g):
    g.setup()
    try:
        return g.run_once()
    finally:
        g.teardown()


def white(results, sub):
    return float((results[sub.id].outputs["image"] > 0).mean())


def num_flow():
    """_Num20 → [_Gain20 → _Setup20] sub-flow."""
    g = Graph()
    src = g.add_new("_Num20", (0, 0))
    a = g.add_new("_Gain20", (260, 0))
    b = g.add_new("_Setup20", (520, 0))
    g.connect(src.id, "n", a.id, "x")
    g.connect(a.id, "y", b.id, "x")
    g.connect(b.id, "y", g.add_new("_Gain20", (780, 0)).id, "x")      # makes "y" an output of the sub-flow
    sub = sf.group(g, [a.id, b.id], "Hesap")
    return g, src, sub, a.id, b.id


# --- exposed settings: core ---------------------------------------------------------------------------------------

def test_expose_shows_setting_on_subflow_block():
    g, src, sub, gray, thr = ramp_flow()
    assert [p.name for p in sub.params] == ["name"]
    assert sub.expose(thr, "value")
    p = sub.param(f"{thr}/value")
    assert p.kind == "int" and p.min == 0 and p.max == 255
    assert p.display == "Eşikle: Eşik"
    assert sub.values[f"{thr}/value"] == 127
    assert sub.is_exposed(thr, "value") and not sub.is_exposed(thr, "mode")
    assert sub.inner["exposed"] == [{"node": thr, "param": "value"}]
    # unknown block / setting / boundary blocks cannot be exposed
    assert not sub.expose("yok", "value")
    assert not sub.expose(thr, "yok")
    inp = next(n["id"] for n in sub.inner["nodes"] if n["type"] == "SubflowInput")
    assert not sub.expose(inp, "name")
    # exposing twice keeps one entry
    assert sub.expose(thr, "value")
    assert len(sub.inner["exposed"]) == 1
    # the class keeps its own list
    assert [p.name for p in type(sub).params] == ["name"]


def test_exposed_value_changes_the_block_inside():
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    low = white(run1(g), sub)
    sub.set_exposed(f"{thr}/value", 250)
    high = white(run1(g), sub)
    assert low == pytest.approx(128 / 256, abs=0.01) and high == pytest.approx(5 / 256, abs=0.01)
    inner_thr = next(n for n in sub.inner["nodes"] if n["id"] == thr)
    assert inner_thr["values"]["value"] == 250


def test_exposed_survives_save_load_and_old_readers_ignore_it(tmp_path):
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    sub.expose(thr, "mode")
    sub.set_exposed(f"{thr}/value", 33)
    path = tmp_path / "a.json"
    g.save(str(path))
    g2 = Graph.load(str(path))
    s2 = g2.nodes[sub.id]
    assert not g2.warnings
    assert [p.name for p in s2.params] == ["name", f"{thr}/value", f"{thr}/mode"]
    assert s2.values[f"{thr}/value"] == 33 and s2.values[f"{thr}/mode"] == "binary"
    assert white(run1(g2), s2) == pytest.approx((256 - 34) / 256, abs=0.01)
    # a reader without the feature: the inner flow still loads (the extra key is ignored)
    inner = json.loads(path.read_text(encoding="utf-8"))["nodes"]
    raw = next(n for n in inner if n["type"] == "SubFlow")["inner"]
    assert "exposed" in raw and not Graph.from_dict(raw).warnings


def test_each_copy_keeps_its_own_values():
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    d = sub.to_dict()
    copies = []
    for v in (10, 200):
        c = NODE_REGISTRY["SubFlow"]()
        c.load_extra(json.loads(json.dumps(d)), [])
        c.set_exposed(f"{thr}/value", v)
        copies.append(c)
    assert copies[0].values[f"{thr}/value"] == 10 and copies[1].values[f"{thr}/value"] == 200
    assert sub.values[f"{thr}/value"] == 127
    vals = [next(n for n in c.inner["nodes"] if n["id"] == thr)["values"]["value"] for c in copies]
    assert vals == [10, 200]


def test_library_keeps_exposed(tmp_path):
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    sub.set_exposed(f"{thr}/value", 77)
    path = sf.save_to_library(sub, tmp_path)
    loaded, warnings = sf.load_from_library(path)
    assert not warnings and loaded.values[f"{thr}/value"] == 77
    assert loaded.param(f"{thr}/value").display == "Eşikle: Eşik"


def test_exposed_removed_when_block_or_setting_disappears():
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    inner = sub.inner_graph()
    inner.remove(thr)
    sf.apply_inner(g, sub, inner.to_dict())          # what the editor does after deleting the block inside
    assert sub.exposed_keys() == [] and f"{thr}/value" not in sub.values
    assert "exposed" not in sub.inner
    assert "exposed" not in json.dumps(g.to_dict())


def test_set_inner_keeps_exposed_and_takes_values_edited_inside():
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    inner = sub.inner_graph()
    inner.nodes[thr].values["value"] = 99           # edited on the canvas inside the sub-flow
    sf.apply_inner(g, sub, inner.to_dict())
    assert sub.values[f"{thr}/value"] == 99 and sub.is_exposed(thr, "value")


def test_broken_exposed_entries_in_a_file_are_ignored():
    g, src, sub, gray, thr = ramp_flow()
    d = g.to_dict()
    nd = next(n for n in d["nodes"] if n["type"] == "SubFlow")
    nd["inner"]["exposed"] = ["x", {"node": 5, "param": "value"}, {"node": thr, "param": "nope"},
                              {"node": thr, "param": "value"}, {"node": thr, "param": "value"}, None,
                              {"node": "ghost", "param": "value"}]
    next(n for n in nd["inner"]["nodes"] if n["id"] == thr)["values"]["value"] = "çok"   # bad value too
    g2 = Graph.from_dict(d)
    s2 = g2.nodes[sub.id]
    assert s2.exposed_keys() == [f"{thr}/value"] and s2.values[f"{thr}/value"] == 127
    assert s2.inner["exposed"] == [{"node": thr, "param": "value"}]
    nd["inner"]["exposed"] = "bozuk"
    assert Graph.from_dict(d).nodes[sub.id].exposed_keys() == []


def test_nested_subflow_exposes_an_exposed_setting():
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    outer = sf.group(g, [sub.id], "Dış")
    assert outer.expose(sub.id, f"{thr}/value")
    key = f"{sub.id}/{thr}/value"
    assert outer.param(key).display == "Eşik grubu: Eşikle: Eşik"
    outer.set_exposed(key, 240)
    res = run1(g)
    assert float((res[outer.id].outputs["image"] > 0).mean()) == pytest.approx(15 / 256, abs=0.01)
    g2 = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert g2.nodes[outer.id].values[key] == 240
    # the running block deep inside is reachable with the same path
    g2.setup()
    try:
        assert node_at(g2, f"{outer.id}/{sub.id}/{thr}").values["value"] == 240
    finally:
        g2.teardown()


def test_live_change_reaches_running_inner_block_through_worker_key():
    pytest.importorskip("PySide6.QtCore")
    from visionflow.ui.worker import FlowWorker
    g, src, sub, a, b = num_flow()
    sub.expose(b, "k")
    w = FlowWorker(g.to_dict())
    live = Graph.from_dict(g.to_dict())
    live.setup()
    try:
        before = _Setup20.setups
        w.set_param(f"{sub.id}/{b}", "k", 5)
        w._apply_pending(live)
        assert _Setup20.setups == before + 1               # a setup setting: only that block restarted
        assert live.run_once()[sub.id].outputs["y"] == 10.0
    finally:
        live.teardown()


def test_ungroup_after_expose_puts_values_back():
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    sub.set_exposed(f"{thr}/value", 42)
    sf.ungroup(g, sub.id)
    assert g.nodes[thr].values["value"] == 42


# --- recipes: core ------------------------------------------------------------------------------------------------

def recipe_flow():
    g = Graph()
    src = g.add_new("_Num20", (0, 0))
    a = g.add_new("_Gain20", (260, 0))
    g.connect(src.id, "n", a.id, "x")
    g.recipes = [{"name": "Ürün A", "values": {a.id: {"k": 3.0}}},
                 {"name": "Ürün B", "values": {a.id: {"k": 10.0, "mode": "b"}}}]
    return g, src, a


def test_recipes_roundtrip_and_old_files():
    g, src, a = recipe_flow()
    d = json.loads(json.dumps(g.to_dict()))
    assert [r["name"] for r in d["recipes"]] == ["Ürün A", "Ürün B"]
    g2 = Graph.from_dict(d)
    assert g2.recipes == g.recipes and not g2.warnings
    assert "recipes" not in Graph().to_dict()                     # no key when there are none
    d.pop("recipes")
    assert Graph.from_dict(d).recipes == []


def test_recipes_validated_on_load():
    g, src, a = recipe_flow()
    d = g.to_dict()
    d["recipes"] = [
        {"name": "  Ürün   A ", "values": {a.id: {"k": "4", "mode": 7, "yok": 1}, "ghost": {"k": 1}}},
        {"name": "ürün a", "values": {}},                          # same name (case/spaces do not count)
        {"name": "", "values": {}}, "çöp", {"values": {}},
        {"name": "C", "values": {a.id: {"k": 1e9, "on": "evet"}, src.id: "bozuk"}},
        {"name": "D", "values": "bozuk"},
        {"name": "E", "values": {a.id: {"k": [1, 2]}}},
    ]
    g2 = Graph.from_dict(d)
    names = [r["name"] for r in g2.recipes]
    assert names == ["Ürün A", "C", "D", "E"]
    assert g2.recipes[0]["values"] == {a.id: {"k": 4.0}}            # "7" is no choice: dropped with a warning
    assert g2.recipes[1]["values"] == {a.id: {"k": 100.0, "on": True}}   # pulled into range
    assert g2.recipes[2]["values"] == {} and g2.recipes[3]["values"] == {}
    w = "\n".join(g2.warnings)
    assert "Aynı adlı" in w and "Bozuk reçete" in w and "geçersiz" in w
    d["recipes"] = "bozuk"
    g3 = Graph.from_dict(d)
    assert g3.recipes == [] and any("Reçete" in x for x in g3.warnings)


def test_export_drops_deleted_blocks():
    g, src, a = recipe_flow()
    g.remove(a.id)
    assert g.to_dict()["recipes"] == [{"name": "Ürün A", "values": {}}, {"name": "Ürün B", "values": {}}]
    assert g.recipes[0]["values"]                                  # undo can bring the block back


def test_find_by_name_and_number():
    g, src, a = recipe_flow()
    assert rc.find(g, "ürün b")["name"] == "Ürün B"
    assert rc.find(g, " Ürün  A ")["name"] == "Ürün A"
    assert rc.find(g, 2)["name"] == "Ürün B"
    assert rc.find(g, 2.0)["name"] == "Ürün B"
    assert rc.find(g, np.int64(1))["name"] == "Ürün A"
    assert rc.find(g, "1")["name"] == "Ürün A"
    assert rc.find(g, 0, base=0)["name"] == "Ürün A"
    for bad, msg in ((3, "numaralı reçete yok"), (0, "numaralı reçete yok"), (1.5, "tam sayı"),
                     (float("nan"), "tam sayı"), ("Ürün C", "adlı reçete yok"), (True, "bekleniyor"),
                     (None, "bekleniyor"), ([1], "bekleniyor")):
        with pytest.raises(ValueError, match=msg):
            rc.find(g, bad)
    with pytest.raises(ValueError, match="hiç reçete yok"):
        rc.find(Graph(), "A")


def test_apply_recipe_not_running():
    g, src, a = recipe_flow()
    assert g.apply_recipe("Ürün B") == []
    assert a.values["k"] == 10.0 and a.values["mode"] == "b"
    g.recipes[0]["values"][a.id]["k"] = "çok"                       # broken by hand after loading
    errs = g.apply_recipe("Ürün A")
    assert len(errs) == 1 and "geçersiz" in errs[0] and a.values["k"] == 10.0


def test_capture_values():
    g, src, a = recipe_flow()
    sel = g.add_new("RecipeSelect")
    note = g.add_new("Note")
    a.values["k"] = 7.5
    vals = rc.capture(g, [a.id, sel.id, note.id, "ghost"])
    assert vals == {a.id: {"k": 7.5, "mode": "a", "on": True}}


def test_recipe_select_setting_applies_before_first_frame():
    g, src, a = recipe_flow()
    sel = g.add_new("RecipeSelect", recipe="ürün b")
    g.setup()
    try:
        r = g.run_once()
        assert r[a.id].outputs["y"] == 20.0                       # 2 × 10 already on frame 1
        assert r[sel.id].outputs == {"name": "Ürün B", "index": 2, "changed": True}
        assert g.recipe_events == [("Ürün B", [])]
        assert g.run_once()[sel.id].outputs["changed"] is False
    finally:
        g.teardown()


def test_recipe_select_input_switches_on_next_frame():
    g, src, a = recipe_flow()
    num = g.add_new("_Num20", value=1.0)
    sel = g.add_new("RecipeSelect")
    g.connect(num.id, "n", sel.id, "recipe")
    g.setup()
    try:
        r = g.run_once()
        assert r[sel.id].outputs["name"] == "Ürün A" and r[a.id].outputs["y"] == 2.0   # default k=1 this frame
        assert g.run_once()[a.id].outputs["y"] == 6.0                                # applied between frames
        num.values["value"] = 2.0
        r = g.run_once()
        assert r[sel.id].outputs["changed"] and r[sel.id].outputs["index"] == 2
        assert g.run_once()[a.id].outputs["y"] == 20.0
        num.values["value"] = 9.0
        r = g.run_once()
        assert "numaralı reçete yok" in r[sel.id].error and a.values["k"] == 10.0     # unknown: kept as it was
    finally:
        g.teardown()


def test_recipe_select_zero_based_and_text_and_bad_inputs():
    g, src, a = recipe_flow()
    sel = g.add_new("RecipeSelect", base="0'dan")
    g.setup()
    try:
        assert sel.process({"recipe": np.float32(1.0)})["name"] == "Ürün B"
        assert sel.process({"recipe": "Ürün A"})["index"] == 0
        assert sel.process({"recipe": "   "})["name"] == "Ürün A"     # empty text: no change
        assert sel.process({"recipe": None})["changed"] is False
        for bad in (True, np.zeros(3), {"a": 1}):
            with pytest.raises(ValueError, match="bekleniyor"):
                sel.process({"recipe": bad})
    finally:
        g.teardown()


def test_recipe_select_without_recipes_or_flow():
    g = Graph()
    sel = g.add_new("RecipeSelect")
    assert run1(g)[sel.id].outputs == {"name": "", "index": 0, "changed": False}
    sel.values["recipe"] = "A"
    r = run1(g)
    assert "hiç reçete yok" in r[sel.id].error
    lone = NODE_REGISTRY["RecipeSelect"]()
    lone.safe_setup()
    with pytest.raises(ValueError, match="akışın içinde"):
        lone.process({"recipe": "A"})


def test_live_recipe_restarts_setup_blocks_once_and_reaches_inside_subflows():
    g, src, sub, a, b = num_flow()
    sub.expose(a, "k")
    sub.expose(b, "k")
    g.recipes = [{"name": "X", "values": {sub.id: {f"{a}/k": 2.0, f"{b}/k": 3}}}]
    sel = sf.group(g, [g.add_new("RecipeSelect", recipe="").id], "Seçici")   # the selector inside a sub-flow
    g.setup()
    try:
        assert g.run_once()[sub.id].outputs["y"] == 2.0
        inner_sel = next(n for n in sel.state["graph"].nodes.values() if n.type_name == "RecipeSelect")
        inner_sel.values["recipe"] = "X"
        before = _Setup20.setups
        g.run_once()                                           # requested during this frame …
        assert g.run_once()[sub.id].outputs["y"] == 12.0       # … applied before the next: 2 × 2 × 3
        assert _Setup20.setups == before + 1                   # only the block reading "k" in setup restarted
        assert sub.values[f"{b}/k"] == 3                       # the block (and its saved inner flow) too
        assert next(n for n in sub.inner["nodes"] if n["id"] == b)["values"]["k"] == 3
    finally:
        g.teardown()


def test_graph_run_applies_recipe_and_reports():
    g, src, a = recipe_flow()
    g.add_new("RecipeSelect", recipe="2")
    out = []
    g.run(max_frames=2, on_frame=lambda n, r: out.append(r[a.id].outputs["y"]))
    assert out == [20.0, 20.0]


# --- headless ------------------------------------------------------------------------------------------------------

def test_headless_recete(tmp_path, capsys):
    from visionflow import headless
    g, src, a = recipe_flow()
    sel = g.add_new("RecipeSelect", recipe="Ürün A")           # the command line wins over the block's setting
    path = tmp_path / "akis.json"
    g.save(str(path))
    assert headless.main([str(path), "--frames", "2", "--fps", "0", "--plugins", str(tmp_path),
                          "--recete", "ürün b"]) == 0
    out = capsys.readouterr().out
    assert "Reçete: Ürün B" in out and "Reçete uygulandı: Ürün B" in out
    assert headless.main([str(path), "--frames", "1", "--plugins", str(tmp_path), "--recete", "Yok"]) == 2
    assert "Reçete seçilemedi" in capsys.readouterr().err
    assert headless.main([str(path), "--frames", "1", "--fps", "0", "--plugins", str(tmp_path),
                          "--recete", "1"]) == 0
    assert "Reçete: Ürün A" in capsys.readouterr().out
    del sel


def test_headless_sets_selectors_inside_subflows():
    from visionflow.headless import _set_selectors
    g, src, a = recipe_flow()
    s = g.add_new("RecipeSelect", recipe="Ürün A")
    sub = sf.group(g, [s.id], "S")
    _set_selectors(g, "Ürün B")
    assert next(n for n in sub.inner["nodes"] if n["type"] == "RecipeSelect")["values"]["recipe"] == "Ürün B"


# --- editor ---------------------------------------------------------------------------------------------------------

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


def _load(win, g):
    win.scene.load_graph(g)
    win._loaded(None)


def _labels(props):
    from PySide6.QtWidgets import QLabel
    return [w.text() for w in props.form_host.findChildren(QLabel)]


def test_expose_from_properties_inside_subflow(win):
    g, src, sub, gray, thr = ramp_flow()
    _load(win, g)
    win.props.show_node(sub)
    assert win.props._exposed(sub, "name") is None              # not inside a sub-flow: no menu
    win.enter_subflow(sub)
    t = win.scene.graph.nodes[thr]
    win.scene.items_by_id[thr].setSelected(True)
    assert win.props.node is t and win.props._exposed(t, "value") is False
    menu = win.props.expose_menu(t, "value")
    act = menu.actions()[0]
    assert act.text() == "Bu ayarı alt akış bloğunda göster" and not act.isChecked()
    act.setChecked(True)                                         # → expose_toggled → toggle_exposed
    assert sub.is_exposed(thr, "value")
    assert any(x.startswith("📌 ") and "Eşik" in x for x in _labels(win.props))
    # the canvas context menu has the same switch
    from PySide6.QtCore import QPoint, QPointF, QTimer
    QTimer.singleShot(0, lambda: win._context_menu.close())
    win.context_menu(QPoint(0, 0), QPointF(0, 0))
    sub_menu = next(a.menu() for a in win._context_menu.actions() if a.text() == "Ayarı alt akış bloğunda göster")
    checks = {a.text(): a.isChecked() for a in sub_menu.actions()}
    assert checks == {"Yöntem": False, "Eşik": True}
    # the boundary blocks have nothing to expose
    inp = next(n for n in win.scene.graph.nodes.values() if n.type_name == "SubflowInput")
    assert win._expose_state(inp, "name") is None
    win.exit_subflow()
    win.scene.clearSelection()
    win.scene.items_by_id[sub.id].setSelected(True)
    assert win.props.node is sub and "Eşikle: Eşik" in _labels(win.props)


def test_editing_exposed_setting_on_block_and_undo(win, tmp_path):
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    _load(win, g)
    win.props.show_node(sub)
    win.props._emit(sub, f"{thr}/value", 200)                    # typed into the settings panel
    win._snapshot()
    assert next(n for n in sub.inner["nodes"] if n["id"] == thr)["values"]["value"] == 200
    win.enter_subflow(sub)
    assert win.scene.graph.nodes[thr].values["value"] == 200     # seen inside too
    win.exit_subflow()
    win.undo()
    s = win.root.nodes[sub.id]
    assert s.values[f"{thr}/value"] == 127 and s.is_exposed(thr, "value")
    win.redo()
    assert win.root.nodes[sub.id].values[f"{thr}/value"] == 200
    path = tmp_path / "x.json"
    win.path = str(path)
    assert win.save_flow()
    assert Graph.load(str(path)).nodes[sub.id].values[f"{thr}/value"] == 200


def test_copy_paste_subflow_keeps_separate_values(win):
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    _load(win, g)
    win.scene.clearSelection()
    win.scene.items_by_id[sub.id].setSelected(True)
    win.scene.duplicate_selected()
    copy_ = next(n for n in win.scene.graph.nodes.values() if n.type_name == "SubFlow" and n.id != sub.id)
    assert copy_.is_exposed(thr, "value")
    win.props._emit(copy_, f"{thr}/value", 5)
    win.on_param(copy_.id, f"{thr}/value", 5)
    assert sub.values[f"{thr}/value"] == 127 and copy_.values[f"{thr}/value"] == 5
    g2 = Graph.from_dict(json.loads(win._snap()))
    assert sorted(n.values[f"{thr}/value"] for n in g2.nodes.values() if n.type_name == "SubFlow") == [5, 127]


def test_deleting_inner_block_removes_exposed(win):
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    _load(win, g)
    win.enter_subflow(sub)
    win.scene.clearSelection()
    win.scene.items_by_id[thr].setSelected(True)
    win.scene.delete_selected()
    win.exit_subflow()
    assert sub.exposed_keys() == [] and [p.name for p in sub.params] == ["name"]


def test_exposed_change_goes_to_worker_with_inner_key(win):
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    _load(win, g)

    class FakeWorker:
        def __init__(self):
            self.sent = []

        def set_param(self, *a):
            self.sent.append(a)
    win.worker = FakeWorker()
    try:
        win.on_param(sub.id, f"{thr}/value", 9)
        win.on_param(sub.id, "name", "Yeni")
        assert win.worker.sent == [(f"{sub.id}/{thr}", "value", 9), (sub.id, "name", "Yeni")]
    finally:
        win.worker = None


def test_recipes_dialog_create_capture_apply_rename_delete(win, tmp_path):
    g, src, a = recipe_flow()
    g.recipes = []
    sel = g.add_new("RecipeSelect", recipe="P1")
    _load(win, g)
    dlg = win.open_recipes()
    try:
        assert dlg.list.count() == 0 and not dlg.apply_btn.isEnabled()
        win.scene.items_by_id[a.id].setSelected(True)
        win.scene.items_by_id[sel.id].setSelected(True)
        a.values["k"] = 4.0
        r = dlg.new_recipe("P1")
        assert r["values"] == {a.id: {"k": 4.0, "mode": "a", "on": True}}     # the selector is not captured
        assert dlg.table.rowCount() == 3
        assert dlg.new_recipe("p1") is None and "zaten var" in dlg.status.text()
        assert dlg.new_recipe("   ") is None
        win.scene.clearSelection()
        dlg.new_recipe("P2")
        assert dlg.current()["values"] == {}
        assert dlg.capture_selected() == 0                                  # nothing selected
        win.scene.items_by_id[a.id].setSelected(True)
        a.values["k"] = 8.0
        assert dlg.capture_selected() == 3
        # typed into the table: checked like a setting
        row = next(i for i in range(dlg.table.rowCount()) if dlg.table.item(i, 1).text() == "Kazanç")
        dlg.table.item(row, 2).setText("500")
        assert dlg.current()["values"][a.id]["k"] == 100.0                  # pulled into range
        row = next(i for i in range(dlg.table.rowCount()) if dlg.table.item(i, 1).text() == "Kazanç")
        dlg.table.item(row, 2).setText("abc")
        assert dlg.current()["values"][a.id]["k"] == 100.0 and "Geçersiz" in dlg.status.text()
        row = next(i for i in range(dlg.table.rowCount()) if dlg.table.item(i, 1).text() == "Açık")
        assert dlg.remove_rows([row]) == 1 and "on" not in dlg.current()["values"][a.id]
        # apply P1 in the editor
        dlg.list.setCurrentRow(0)
        assert dlg.apply_current() == [] and a.values["k"] == 4.0
        # rename: the selector set to it follows
        assert dlg.rename_recipe("Ürün 1") and sel.values["recipe"] == "Ürün 1"
        assert not dlg.rename_recipe("P2")
        win._snapshot()
        saved = json.loads(win._snap())
        assert [r["name"] for r in saved["recipes"]] == ["Ürün 1", "P2"]
        assert dlg.delete_recipe(confirm=False) and [r["name"] for r in win.root.recipes] == ["P2"]
        win._snapshot()
        win.undo()                                                          # the deleted recipe comes back
        assert [r["name"] for r in win.root.recipes] == ["Ürün 1", "P2"]
        assert dlg.list.count() == 2
        path = tmp_path / "r.json"
        win.path = str(path)
        assert win.save_flow()
        assert [r["name"] for r in Graph.load(str(path)).recipes] == ["Ürün 1", "P2"]
    finally:
        dlg.close()


def test_recipe_for_exposed_setting_applied_while_inside_subflow(win):
    g, src, sub, gray, thr = ramp_flow()
    sub.expose(thr, "value")
    g.recipes = [{"name": "Koyu", "values": {sub.id: {f"{thr}/value": 30}}}]
    _load(win, g)
    win.enter_subflow(sub)
    assert win.apply_recipe("Koyu") == []
    assert win.levels and win.scene.graph.nodes[thr].values["value"] == 30   # the open inside shows it
    win.exit_subflow()                                                       # and leaving keeps it
    assert win.root.nodes[sub.id].values[f"{thr}/value"] == 30
    assert win.apply_recipe("yok") == ["'yok' adlı reçete yok"]


def test_worker_recipe_mirrors_values_into_editor(win):
    g, src, a = recipe_flow()
    _load(win, g)
    win.scene.items_by_id[a.id].setSelected(True)
    win.on_worker_recipe("Ürün B", [])
    assert a.values["k"] == 10.0
    from PySide6.QtWidgets import QDoubleSpinBox
    assert any(isinstance(w, QDoubleSpinBox) and w.value() == 10.0 for w in win.props.form_host.findChildren(QDoubleSpinBox))


def test_running_editor_switches_recipe(win):
    from PySide6.QtWidgets import QApplication
    import time
    g, src, a = recipe_flow()
    num = g.add_new("_Num20", value=1.0)
    sel = g.add_new("RecipeSelect")
    g.connect(num.id, "n", sel.id, "recipe")
    _load(win, g)
    win.start()
    try:
        t0 = time.time()
        while a.values["k"] != 3.0 and time.time() - t0 < 10:
            QApplication.processEvents()
            time.sleep(0.02)
        assert a.values["k"] == 3.0                     # Ürün A came from the running flow into the editor
    finally:
        win.stop()
