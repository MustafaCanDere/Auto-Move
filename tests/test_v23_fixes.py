"""2.3 fixes: bugs at the seams of the four 2.3 features (screen designer, 3D / depth, dataset manager, station
manager + API in --operator) found after they were merged."""
import json
import os
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow.core import Graph, load_builtin  # noqa: E402
from visionflow.core import node as _node  # noqa: E402
from visionflow.core import screens as S  # noqa: E402
from visionflow.core.operator import clean_operator  # noqa: E402

load_builtin()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(_node, "FLOW_DIR", None)
    monkeypatch.setattr(_node, "FLOW_FILE", None)
    return tmp_path / "home"


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(home, qapp):
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    panel = w.operator_panel()
    if panel is not None:
        panel.leave(open_editor=False)
    w.stop()
    w.close()
    qapp.processEvents()


def pump_until(app, cond, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def screen_flow(selector="A"):
    """TestPattern; a Number switched by recipes A/B (and a 'Reçete Seç' set to A); a designed screen showing the
    Number's setting, an input writing it and a recipe selector."""
    g = Graph()
    src = g.add_new("TestPattern", width=64, height=48, animate=True)
    num = g.add_new("Number", value=0.0)
    other = g.add_new("Number", value=5.0)
    sel = g.add_new("RecipeSelect", recipe=selector) if selector is not None else None
    g.recipes = [{"name": "A", "values": {num.id: {"value": 1.0}}},
                 {"name": "B", "values": {num.id: {"value": 2.0}}}]
    ws = [S.new_widget("number", 0, 0, "set", src={"node": num.id, "setting": "value"}),
          S.new_widget("input", 0, 200, "inp", target={"node": other.id, "setting": "value"}, lock=False),
          S.new_widget("recipe", 0, 400, "rcp", lock=False),
          S.new_widget("button", 300, 400, "rbtn", action="reçete", recipe="B", lock=False)]
    screen = S.new_screen()
    screen["pages"][0]["widgets"] = ws
    g.operator = clean_operator({"screen": screen, "layout": "screen"}, g)
    return g, src, num, other, sel


def selector_of(data, sel):
    return next(n for n in data["nodes"] if n["id"] == sel.id)["values"]["recipe"]


# ---------------------------------------------------------------- recipes: API ↔ designed screen ↔ panel / editor
def test_operator_api_recipe_reaches_the_designed_screen_and_survives_screen_writes(home, qapp):
    """--operator: a recipe chosen over the API used to change only the panel's graph_data. The designed screen
    kept showing the old recipe / old setting values while stopped, and the next setting written on the screen
    rebuilt graph_data from the stale flow copy: the recipe's values were silently lost for the next start."""
    from visionflow.ui.api_dialog import PanelController
    from visionflow.ui.operator import OperatorPanel
    from visionflow import api_server
    g, src, num, other, sel = screen_flow()
    panel = OperatorPanel(g.to_dict(), None)
    try:
        view = panel.screen_view
        assert view is not None and panel.worker is None
        ctl = PanelController(panel, api_server.FlowState())
        assert ctl.set_recipe("B") == ("B", [])
        view.poll()
        assert view.item("set").value == 2.0                  # the screen shows the recipe's value at once
        assert view.item("rcp").value == "B" and ctl.current_recipe() == "B"
        assert view.edit_input(view.item("inp"), "7")         # an operator input afterwards ...
        data = panel.graph_data
        node = next(n for n in data["nodes"] if n["id"] == num.id)
        assert node["values"]["value"] == 2.0                 # ... keeps the recipe for the next start
        assert next(n for n in data["nodes"] if n["id"] == other.id)["values"]["value"] == 7.0
        assert selector_of(data, sel) == "B"                  # 'Reçete Seç' does not switch back to A
        with pytest.raises(ValueError):
            ctl.set_recipe("yok")
    finally:
        panel.leave(open_editor=False)


def test_screen_recipe_while_stopped_reaches_api_and_next_start(home, qapp):
    """A recipe chosen on the designed screen while the flow was stopped did not reach /durum (recipe_name stayed
    old), and the 'Reçete Seç' block switched the next start back to its own recipe."""
    from visionflow.ui.api_dialog import PanelController
    from visionflow.ui.operator import OperatorPanel
    from visionflow import api_server
    g, src, num, other, sel = screen_flow()
    panel = OperatorPanel(g.to_dict(), None)
    try:
        view = panel.screen_view
        assert view.choose_recipe(view.item("rbtn"), "B")
        assert PanelController(panel, api_server.FlowState()).current_recipe() == "B"
        assert selector_of(panel.graph_data, sel) == "B"
        panel.start()
        assert pump_until(qapp, lambda: view.feed.snapshot()["frames"] > 2)
        panel.stop()
        assert panel.recipe_name == "B"                       # the start did not switch back to A
        assert view.item("rcp").value == "B"
    finally:
        panel.leave(open_editor=False)


def test_editor_api_recipe_reaches_the_panel_screen(win, qapp):
    """The editor's API (or recipe dialog) while its operator panel shows a designed screen: the panel's flow copy
    stayed old (stale values on the screen, lost again by the next screen write)."""
    from visionflow.ui.api_dialog import manager
    g, src, num, other, sel = screen_flow()
    win._set_root(Graph.from_dict(g.to_dict()))
    panel = win.open_operator()
    assert panel is not None and panel.screen_view is not None
    panel.stop()
    view = panel.screen_view
    name, errors = manager(win).controller.set_recipe("B")
    assert name == "B" and not errors
    view.poll()
    assert view.item("set").value == 2.0 and view.item("rcp").value == "B"
    assert view.edit_input(view.item("inp"), "9")
    node = next(n for n in panel.graph_data["nodes"] if n["id"] == num.id)
    assert node["values"]["value"] == 2.0 and selector_of(panel.graph_data, sel) == "B"
    assert win.root.nodes[num.id].values["value"] == 2.0 and win.root.nodes[other.id].values["value"] == 9.0
    # a recipe from the screen reaches the editor (its API reports it) and its 'Reçete Seç' blocks
    assert view.choose_recipe(view.item("rcp"), "A")
    assert win.active_recipe == "A" and win.root.nodes[sel.id].values["recipe"] == "A"
    assert win.root.nodes[num.id].values["value"] == 1.0


def test_operator_changes_are_handed_to_the_editor_opened_later(home, qapp, tmp_path):
    """--operator: the editor is made only when the operator leaves the panel and loads the flow FILE, so settings
    written on the designed screen and the recipe chosen there were lost (and reverted at the next run)."""
    from visionflow.ui.operator import OperatorPanel
    g, src, num, other, sel = screen_flow()
    path = tmp_path / "akis.json"
    g.save(str(path))
    made = []

    def editor():
        from visionflow.ui.main_window import MainWindow
        w = MainWindow()
        w._confirm_discard = lambda: True
        w.open_flow(str(path), unlocked=True)
        made.append(w)
        return w
    panel = OperatorPanel(Graph.load(str(path)).to_dict(), str(path), editor_factory=editor)
    view = panel.screen_view
    assert view.edit_input(view.item("inp"), "4")
    assert view.choose_recipe(view.item("rcp"), "B")
    assert panel.leave()
    w = made[0]
    try:
        assert w.root.nodes[other.id].values["value"] == 4.0
        assert w.root.nodes[num.id].values["value"] == 2.0 and w.active_recipe == "B"
        assert w.path == str(path)
        w._snap_timer.stop()
        w._snapshot()
        assert w._snap() != w.saved_snap                       # an unsaved change, the file is untouched
        assert json.loads(path.read_text(encoding="utf-8"))["nodes"]
    finally:
        w.stop()
        w.close()
        qapp.processEvents()


def test_unchanged_operator_panel_leaves_the_editor_as_loaded(home, qapp, tmp_path):
    from visionflow.ui.operator import OperatorPanel
    g, *_ = screen_flow()
    path = tmp_path / "akis.json"
    g.save(str(path))
    made = []

    def editor():
        from visionflow.ui.main_window import MainWindow
        w = MainWindow()
        w.open_flow(str(path), unlocked=True)
        made.append(w)
        return w
    panel = OperatorPanel(Graph.load(str(path)).to_dict(), str(path), editor_factory=editor)
    assert panel.leave()
    w = made[0]
    try:
        assert w._snap() == w.saved_snap
    finally:
        w.close()
        qapp.processEvents()


# ---------------------------------------------------------------- designed screen ↔ 3D outputs
def depth_results():
    g = Graph.load(os.path.join(os.path.dirname(__file__), "..", "examples", "derinlik_hacim_olcum.json"))
    from visionflow.core.graph import flatten
    g.setup()
    try:
        flat = flatten(g, g.run_once())
    finally:
        g.teardown()
    return g, flat


def test_image_and_value_widgets_on_the_same_any_output(qapp):
    """An image widget bound to an ANY output (intrinsics, plane, 'truth') turned every value widget of that
    output into 'Görüntü değil' with no value."""
    from visionflow.ui.screen_runtime import ScreenView
    g, flat = depth_results()
    screen = S.new_screen()
    screen["pages"][0]["widgets"] = [
        S.new_widget("image", 0, 0, "img", src={"node": "sahne", "port": "intrinsics"}),
        S.new_widget("value", 0, 400, "txt", src={"node": "sahne", "port": "intrinsics"}),
        S.new_widget("image", 400, 0, "dimg", src={"node": "sahne", "port": "depth"}),
        S.new_widget("value", 400, 400, "dtxt", src={"node": "sahne", "port": "depth"}),
        S.new_widget("value", 600, 400, "pl", src={"node": "zemin", "port": "plane"}),
        S.new_widget("number", 800, 400, "cnt", src={"node": "nesne", "port": "detections"})]
    view = ScreenView(screen, g)
    try:
        view.feed.feed(flat, g)
        view.poll()
        it = view.item
        assert it("img").error == "Görüntü değil"
        assert it("txt").error == "" and str(it("txt").value).startswith("fx=")
        assert it("dimg").qimage is not None and not it("dimg").qimage.isNull() and it("dimg").error == ""
        assert it("dtxt").value == "<dizi 480x640>"
        assert "eğim" in str(it("pl").value) and it("cnt").value == len(flat["nesne"].outputs["detections"])
        # a block error reaches both kinds of widget
        from visionflow.core import NodeResult
        view.feed.feed({"sahne": NodeResult({}, 1.0, error="kamera koptu")}, g)
        view.poll()
        assert it("img").error == "kamera koptu" and it("txt").error == "kamera koptu"
    finally:
        view.stop_timer()


# ---------------------------------------------------------------- float32 depth pictures: API, preview, archive
def holey_depth():
    d = np.full((60, 80), 800.0, np.float32)
    d[20:40, 20:60] = 760.0                                   # a box 40 mm high
    d[::7, ::5] = np.nan                                      # the camera's missing pixels
    return d


def test_api_jpeg_and_preview_keep_contrast_of_depth_with_holes():
    """NaN holes were made 0 before stretching, so the measured 760…800 mm were squeezed into 242…255: the box
    was invisible in the API picture, the station preview and the editor / screen pictures."""
    from visionflow.api_server import stretch_u8, to_jpeg
    d = holey_depth()
    u = stretch_u8(d)
    assert u[25, 21] == 0 and u[5, 1] == 255 and u[0, 0] == 0          # box, floor, hole
    j = cv2.imdecode(np.frombuffer(to_jpeg(d), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert int(j[5:15, 1:4].mean()) - int(j[25:35, 31:34].mean()) > 200
    from visionflow.ui.worker import displayable
    shown = displayable(d)
    assert shown.dtype == np.uint8 and int(shown[5, 1]) - int(shown[25, 21]) > 200
    assert not stretch_u8(np.full((4, 4), np.nan, np.float32)).any()
    assert not stretch_u8(np.full((4, 4), 5.0, np.float32)).any()
    assert (stretch_u8(np.array([[1, 2], [3, 4]], np.uint16)) == [[0, 85], [170, 255]]).all()


def test_archive_jpeg_of_depth_is_not_all_white():
    """'Parçayı Arşivle' with a depth map in mm clipped everything above 255 to white."""
    from visionflow.trace import to_jpeg_bgr
    j = to_jpeg_bgr(holey_depth())
    assert j.dtype == np.uint8 and int(j[5, 1]) - int(j[25, 21]) > 200
    # pictures that were right before stay the same
    f01 = np.full((4, 4), 0.5, np.float32)
    assert (to_jpeg_bgr(f01) == 127).all()
    f255 = np.full((4, 4), 200.0, np.float32)
    assert (to_jpeg_bgr(f255) == 200).all()
    assert (to_jpeg_bgr(np.full((4, 4), 40000, np.uint16)) == 40000 >> 8).all()
