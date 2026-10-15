"""2.0 editor tools: align / distribute, snap to grid, mini map, image export, wire drop, insert into a wire,
flow search, error cycling, light theme and block colours."""
import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from visionflow.core import Graph, PortType, load_builtin  # noqa: E402
from visionflow.core.node import clean_color  # noqa: E402
from visionflow.ui import editor_tools as et  # noqa: E402
from visionflow.ui import theme  # noqa: E402
from visionflow.ui.scene import NODE_W, NodeItem  # noqa: E402

load_builtin()


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_win():
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    w.resize(1200, 800)
    w.show()
    return w


@pytest.fixture
def win(app):
    out = sys.stdout
    w = _make_win()
    yield w
    w.stop()
    w.close()
    sys.stdout = out
    app.processEvents()


def no_menu_exec(monkeypatch):
    import visionflow.ui.main_window as mw

    class Menu(QtWidgets.QMenu):
        def exec(self, *a):
            return None
    monkeypatch.setattr(mw, "QMenu", Menu)


def flush(win):
    if win._snap_timer.isActive():
        win._snap_timer.stop()
    win._snapshot()


def three(win):
    s = win.scene
    a = s.add_node("Grayscale", QPointF(0, 0))
    b = s.add_node("Blur", QPointF(300, 40))
    c = s.add_node("Canny", QPointF(700, -30))
    s.clearSelection()
    win._loaded(None)
    return a, b, c


# --- align / distribute -------------------------------------------------------------------------------------------

def test_align_math():
    rects = [(0, 0, 100, 50), (200, 30, 100, 80), (50, 100, 60, 20)]
    assert et.align_positions(rects, "left") == [(0, 0), (0, 30), (0, 100)]
    assert et.align_positions(rects, "right") == [(200, 0), (200, 30), (240, 100)]
    assert et.align_positions(rects, "top") == [(0, 0), (200, 0), (50, 0)]
    assert et.align_positions(rects, "bottom") == [(0, 70), (200, 40), (50, 100)]
    # centre x of all = (0 + 300) / 2 = 150
    assert et.align_positions(rects, "hcenter") == [(100, 0), (100, 30), (120, 100)]
    # centre y of all = (0 + 120) / 2 = 60
    assert et.align_positions(rects, "vcenter") == [(0, 35), (200, 20), (50, 50)]
    assert et.align_positions([], "left") == []
    with pytest.raises(ValueError):
        et.align_positions(rects, "diagonal")


def test_distribute_math():
    rects = [(0, 0, 100, 10), (500, 0, 100, 10), (130, 0, 100, 10)]
    # span 0..600, widths 300 → gap 150: x = 0, 250, 500 (input order kept in the result)
    assert et.distribute_positions(rects, "horizontal") == [(0, 0), (500, 0), (250, 0)]
    v = et.distribute_positions([(0, 0, 10, 40), (0, 10, 10, 40), (0, 300, 10, 40)], "vertical")
    assert v == [(0, 0), (0, 150), (0, 300)]
    # overlapping: centres spread evenly instead of negative gaps
    o = et.distribute_positions([(0, 0, 100, 10), (10, 0, 100, 10), (40, 0, 100, 10)], "horizontal")
    assert [x for x, _ in o] == [0, 20, 40]
    assert et.distribute_positions(rects[:2], "horizontal") == [(0, 0), (500, 0)]
    with pytest.raises(ValueError):
        et.distribute_positions(rects, "sideways")


def test_align_selected_is_one_undo_step(win):
    a, b, c = three(win)
    for it in (a, b, c):
        it.setSelected(True)
    before = [it.node.pos for it in (a, b, c)]
    idx = win.history.index
    assert et.arrange_selected(win.scene, "top") == 2
    flush(win)
    assert {it.node.pos[1] for it in (a, b, c)} == {-30}
    assert win.history.index == idx + 1
    win.undo()
    assert [n.pos for n in (win.scene.graph.nodes[i] for i in (a.node.id, b.node.id, c.node.id))] == before


def test_align_needs_selection(win):
    a, b, c = three(win)
    msgs = []
    win.scene.message.connect(msgs.append)
    a.setSelected(True)
    assert et.arrange_selected(win.scene, "left") == 0
    b.setSelected(True)
    assert et.arrange_selected(win.scene, "horizontal") == 0      # distributing needs three
    assert any("en az 3" in m for m in msgs)


def test_align_in_context_menu(win, monkeypatch):
    a, b, c = three(win)
    a.setSelected(True)
    b.setSelected(True)
    no_menu_exec(monkeypatch)
    win.context_menu(QPoint(0, 0), QPointF(0, 0))
    titles = [act.text() for act in win._context_menu.actions()]
    assert "Hizala" in titles and "Blok rengi" in titles
    edit = next(a.menu() for a in win.menuBar().actions() if a.text() == "Dü&zen")
    assert "Hizala" in [x.text() for x in edit.actions()]


# --- snap to grid -------------------------------------------------------------------------------------------------

def test_snap_math():
    assert et.snap_value(29) == 20 and et.snap_value(31) == 40 and et.snap_value(-11) == -20
    assert et.snap_value(7, 0) == 7
    p = et.snap_point(QPointF(13, 47))
    assert (p.x(), p.y()) == (20, 40)


def _drag(view, a: QPointF, b: QPointF, steps=6):
    vp = view.viewport()
    pa, pb = view.mapFromScene(a), view.mapFromScene(b)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, pa)
    for k in range(1, steps + 1):
        QTest.mouseMove(vp, pa + (pb - pa) * k / steps)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, pb)


def test_snap_while_dragging_and_remembered(win, app):
    s = win.scene
    it = s.add_node("Grayscale", QPointF(3, 7))
    win.view.centerOn(QPointF(100, 100))
    win.snap_act.trigger()
    assert s.snap_grid == 20
    assert it.node.pos == (3, 7)              # turning it on does not move anything by itself
    _drag(win.view, QPointF(50, 10), QPointF(113, 58))
    x, y = it.node.pos
    assert x % 20 == 0 and y % 20 == 0 and (x, y) != (0, 0)
    # programmatic moves (load, undo, align) are not snapped
    it.setPos(5, 5)
    assert it.node.pos == (5, 5)
    w2 = _make_win()
    try:
        assert w2.snap_act.isChecked() and w2.scene.snap_grid == 20
        w2.snap_act.trigger()
        assert w2.scene.snap_grid == 0
    finally:
        w2.close()
    win.snap_act.setChecked(False)
    et.set_snap(win, False)


def test_drag_without_snap_is_free(win):
    s = win.scene
    it = s.add_node("Grayscale", QPointF(3, 7))
    win.view.centerOn(QPointF(100, 100))
    et.set_snap(win, False)
    _drag(win.view, QPointF(50, 10), QPointF(113, 58))
    assert it.node.pos != (3, 7) and (it.node.pos[0] % 20, it.node.pos[1] % 20) != (0, 0)


# --- mini map -------------------------------------------------------------------------------------------------------

def test_minimap_moves_view(win, app):
    three(win)
    et.set_minimap(win, True)
    mm = win.minimap
    assert mm.isVisible()
    assert mm.geometry().right() <= win.view.width() and mm.geometry().bottom() <= win.view.height()
    target = QPointF(700, 0)
    world, s, ox, oy = mm.mapping()
    p = QPointF(ox + (target.x() - world.left()) * s, oy + (target.y() - world.top()) * s)
    QTest.mouseClick(mm, Qt.LeftButton, Qt.NoModifier, p.toPoint())
    centre = win.view.mapToScene(win.view.viewport().rect().center())
    assert abs(centre.x() - target.x()) < 30 and abs(centre.y() - target.y()) < 30
    mm.grab()                                   # paints without error
    win.minimap_act.trigger()
    assert not mm.isVisible() and win.settings.value("minimap", type=bool) is False
    et.set_minimap(win, True)


# --- export as an image ---------------------------------------------------------------------------------------------

def test_export_png_and_svg(win, tmp_path):
    a, b, c = three(win)
    win.scene.try_connect(a.sockets[("image", True)], b.sockets[("image", False)])
    assert len(win.scene.graph.connections) == 1
    b.setSelected(True)
    png = et.export_image(win.scene, tmp_path / "akis.png")
    assert png.stat().st_size > 1000
    img = QImage(str(png))
    r = et.blocks_rect(win.scene)
    assert img.width() >= r.width() * 1.9 and img.height() >= r.height() * 1.9
    assert b.isSelected()                       # selection is put back
    try:
        svg = et.export_image(win.scene, tmp_path / "akis.svg")
    except ValueError as exc:                   # PySide6 without QtSvg: clear message
        assert "QtSvg" in str(exc)
    else:
        text = svg.read_text(encoding="utf-8", errors="ignore")
        assert svg.stat().st_size > 500 and "<svg" in text
    assert et.export_dialog(win, str(tmp_path / "x.png")) is not None
    assert (tmp_path / "x.png").stat().st_size > 0


def test_export_errors(win, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="blok yok"):
        et.export_image(win.scene, tmp_path / "bos.png")
    win.scene.add_node("Grayscale", QPointF(0, 0))
    with pytest.raises(ValueError, match="Desteklenmeyen"):
        et.export_image(win.scene, tmp_path / "akis.txt")
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: warned.append(a[2]))
    assert et.export_dialog(win, str(tmp_path / "akis.gif")) is None and warned
    assert "Akışı resim olarak kaydet…" in [a.text() for a in
                                             next(a.menu() for a in win.menuBar().actions()
                                                  if a.text() == "&Dosya").actions()]


# --- wire dropped on empty canvas -------------------------------------------------------------------------------------

def test_compatible_blocks():
    takes_image = et.compatible_blocks(PortType.IMAGE, need_input=True)
    assert takes_image["Grayscale"] == "image"
    assert "TestPattern" not in takes_image                  # a source has no input
    gives_number = et.compatible_blocks(PortType.NUMBER, need_input=False)
    from visionflow.core.registry import NODE_REGISTRY
    for tn, port in gives_number.items():
        t = dict(NODE_REGISTRY[tn].outputs)[port]
        assert PortType.compatible(t, PortType.NUMBER)
    assert et.best_port([("a", "any"), ("b", "image")], "image", True) == "b"
    assert et.best_port([("a", "number")], "image", True) is None


def test_wire_drop_offers_fitting_blocks_and_connects(win, app):
    s = win.scene
    src = s.add_node("TestPattern", QPointF(0, 0))
    win._loaded(None)
    idx = win.history.index
    win.view.centerOn(QPointF(200, 50))
    sock = src.sockets[("image", True)]
    _drag(win.view, sock.scenePos(), QPointF(380, 60))
    qa = win.quick_add
    assert qa.isVisible() and qa.allowed is not None
    shown = [qa.list.item(i).data(Qt.UserRole) for i in range(qa.list.count())]
    assert "Grayscale" in shown and "TestPattern" not in shown
    assert set(shown) <= set(et.compatible_blocks(PortType.IMAGE, True))
    QTest.keyClicks(qa.edit, "gri")
    assert qa.list.currentItem().data(Qt.UserRole) == "Grayscale"
    QTest.keyClick(qa.edit, Qt.Key_Return)
    new = next(n for n in s.graph.nodes.values() if n.type_name == "Grayscale")
    assert [(c.src, c.src_port, c.dst, c.dst_port) for c in s.graph.connections] == \
        [(src.node.id, "image", new.id, "image")]
    assert new.pos[0] > 380 - 1                                # placed to the right of the drop point
    flush(win)
    assert win.history.index == idx + 1                        # add + connect = one undo step
    win.undo()
    assert [n.type_name for n in win.scene.graph.nodes.values()] == ["TestPattern"]
    # a normal quick add afterwards is not filtered and does not connect
    win._show_quick_add(None, None)
    assert qa.allowed is None and qa.list.count() > len(shown)
    qa.close()


def test_wire_drop_from_input_adds_a_source(win):
    s = win.scene
    dst = s.add_node("Grayscale", QPointF(400, 0))
    sock = dst.sockets[("image", False)]
    assert et.on_wire_dropped(win, sock, QPointF(100, 30))
    assert "TestPattern" in win.quick_add.allowed
    item = et._quick_chosen(win, "TestPattern")
    win.quick_add.close()
    assert item.node.pos[0] + NODE_W < 100
    assert [(c.src, c.dst, c.dst_port) for c in s.graph.connections] == [(item.node.id, dst.node.id, "image")]


def test_wire_drop_on_block_does_not_open_list(win):
    s = win.scene
    a = s.add_node("TestPattern", QPointF(0, 0))
    b = s.add_node("Math", QPointF(400, 0))
    hits = []
    s.wire_dropped.connect(lambda *x: hits.append(x))
    win.view.centerOn(QPointF(300, 50))
    _drag(win.view, a.sockets[("image", True)].scenePos(), b.pos() + QPointF(90, 60))
    assert hits == []
    win.quick_add.close()


# --- insert a dragged block into a wire -------------------------------------------------------------------------------

def _wire(win):
    s = win.scene
    a = s.add_node("TestPattern", QPointF(0, 0))
    b = s.add_node("Threshold", QPointF(600, 0))
    s.try_connect(a.sockets[("image", True)], b.sockets[("image", False)])
    g = s.add_node("Grayscale", QPointF(300, 300))
    win._loaded(None)
    return a, b, g


def test_insert_into_wire(win):
    s = win.scene
    a, b, g = _wire(win)
    s.clearSelection()
    g.setSelected(True)
    edge = a.edges[0]
    g.setPos(300, edge.path().pointAtPercent(0.5).y() - 40)
    assert et.update_insert_target(s, g) is edge and edge.highlight
    idx = win.history.index
    assert et.finish_insert(s, g)
    assert not edge.highlight and s.insert_edge is None
    conns = {(c.src, c.src_port, c.dst, c.dst_port) for c in s.graph.connections}
    assert conns == {(a.node.id, "image", g.node.id, "image"), (g.node.id, "image", b.node.id, "image")}
    flush(win)
    assert win.history.index == idx + 1
    win.undo()
    conns = {(c.src, c.dst) for c in win.scene.graph.connections}
    assert conns == {(a.node.id, b.node.id)}


def test_insert_by_mouse_drag(win):
    s = win.scene
    a, b, g = _wire(win)
    win.view.centerOn(QPointF(400, 150))
    y = a.edges[0].path().pointAtPercent(0.5).y()
    _drag(win.view, QPointF(350, 310), QPointF(350, y + 10))
    assert len(s.graph.connections) == 2 and len(g.edges) == 2


def test_no_insert_when_ports_do_not_fit_or_block_is_connected(win):
    s = win.scene
    a, b, g = _wire(win)
    edge = a.edges[0]
    mid = edge.path().pointAtPercent(0.5)
    num = s.add_node("Math", QPointF(mid.x() - 50, mid.y() - 40))     # numbers only: does not fit an image wire
    s.clearSelection()
    num.setSelected(True)
    assert et.update_insert_target(s, num) is None
    # a block that already has a wire is never inserted
    c = s.add_node("Blur", QPointF(mid.x() - 50, mid.y() - 40))
    s.try_connect(a.sockets[("image", True)], c.sockets[("image", False)])
    s.clearSelection()
    c.setSelected(True)
    assert et.update_insert_target(s, c) is None
    # two selected blocks: no insert
    s.clearSelection()
    g.setPos(mid.x() - 50, mid.y() - 40)
    g.setSelected(True)
    b.setSelected(True)
    assert et.update_insert_target(s, g) is None
    assert not et.finish_insert(s, g)


# --- search the flow ------------------------------------------------------------------------------------------------

def test_fuzzy_score():
    assert et.fuzzy_score("", "Gri") == 0
    assert et.fuzzy_score("gri", "Griye Çevir Grayscale") == 0
    assert et.fuzzy_score("esk", "Eşik Threshold") is not None        # Turkish letters folded
    assert et.fuzzy_score("thd", "Eşik Threshold") >= 1000              # letters in order, not a substring
    assert et.fuzzy_score("xyz", "Eşik Threshold") is None
    assert et.fuzzy_score("es th", "Eşik Threshold") is not None
    assert et.fuzzy_score("gr", "Grayscale") < et.fuzzy_score("gr", "Bugra")


def test_flow_search_jumps_to_block(win):
    a, b, c = three(win)
    c.error = "bozuk ayar"
    fs = et.show_flow_search(win)
    assert fs is not None and fs.isVisible() and fs.list.count() == 3
    QTest.keyClicks(fs.edit, "cny")
    assert fs.list.count() == 1
    assert fs.list.item(0).data(Qt.UserRole) == c.node.id and "bozuk ayar" in fs.list.item(0).text()
    QTest.keyClick(fs.edit, Qt.Key_Return)
    assert not fs.isVisible()
    assert win.scene.selected_node() is c.node
    centre = win.view.mapToScene(win.view.viewport().rect().center())
    assert c.sceneBoundingRect().adjusted(-40, -40, 40, 40).contains(centre)


def test_flow_search_empty_flow(win):
    assert et.show_flow_search(win) is None


# --- jump between failing blocks ---------------------------------------------------------------------------------

def test_error_cycling(win):
    a, b, c = three(win)
    assert et.goto_next_error(win) is None
    a.set_result(1.0, "hata 1", "", None)
    c.set_result(1.0, "hata 2", "", None)
    order = [it.node.id for it in et.error_items(win.scene)]
    assert order == [c.node.id, a.node.id]                      # top to bottom
    seen = [et.goto_next_error(win).node.id for _ in range(3)]
    assert seen == [c.node.id, a.node.id, c.node.id]
    assert win.scene.selected_node() is c.node
    win.goto_error_act.trigger()
    assert win.scene.selected_node() is a.node
    assert win.goto_error_act.shortcut().toString() == "F8"
    a.clear_result()
    assert [it.node.id for it in et.error_items(win.scene)] == [c.node.id]


# --- theme --------------------------------------------------------------------------------------------------------

def test_theme_switch_persists(win, app):
    try:
        win.theme_acts["light"].trigger()
        assert theme.current_theme == "light" and theme.CANVAS["bg"] == theme.CANVAS_THEMES["light"]["bg"]
        assert win.settings.value("theme") == "light"
        assert app.styleSheet() == theme.LIGHT_STYLESHEET
        assert win.theme_acts["light"].isChecked() and not win.theme_acts["dark"].isChecked()
        win.scene.add_node("Grayscale", QPointF(0, 0))
        img = win.view.grab().toImage()
        assert img.pixelColor(5, 5).lightness() > 200              # the canvas is light now
        theme.set_theme("dark")                                    # a new window reads the saved choice
        w2 = _make_win()
        try:
            assert theme.current_theme == "light" and w2.theme_acts["light"].isChecked()
        finally:
            w2.close()
    finally:
        win.theme_acts["dark"].trigger()
    assert theme.current_theme == "dark" and win.settings.value("theme") == "dark"
    assert app.styleSheet() == theme.STYLESHEET
    assert theme.set_theme("pembe") == "dark"


# --- block colour -----------------------------------------------------------------------------------------------

def test_clean_color():
    assert clean_color("#E53935") == "#e53935"
    for bad in ("red", "#12345", "#ff_fff", "#+12345", 123, None, ["#ffffff"], "#gggggg"):
        assert clean_color(bad) is None


def test_block_color_saved_loaded_undone(win, tmp_path):
    a, b, c = three(win)
    b.setSelected(True)
    idx = win.history.index
    assert et.set_block_color(win.scene, "#43A047") == 1
    assert b.node.color == "#43a047"
    flush(win)
    assert win.history.index == idx + 1
    d = b.node.to_dict()
    assert d["color"] == "#43a047" and "color" not in a.node.to_dict()
    path = tmp_path / "renk.json"
    win.scene.graph.save(str(path))
    g = Graph.load(str(path))
    assert g.nodes[b.node.id].color == "#43a047" and g.nodes[a.node.id].color is None
    # bad colours in a file are ignored
    data = json.loads(path.read_text(encoding="utf-8"))
    data["nodes"][0]["color"] = "kırmızı"
    assert all(n.color in (None, "#43a047") for n in Graph.from_dict(data).nodes.values())
    # paste keeps it
    win.copy()
    win.paste(QPointF(0, 500))
    pasted = [it for it in win.scene.selectedItems() if isinstance(it, NodeItem)]
    assert pasted and pasted[0].node.color == "#43a047"
    img = QImage(400, 300, QImage.Format_ARGB32)                 # the colour label is painted in the header
    img.fill(0)
    p = QPainter(img)
    win.scene.render(p, QRectF(img.rect()), QRectF(b.pos().x(), b.pos().y(), 200, 150))
    p.end()
    hc = img.pixelColor(int(100 * 2), 10)
    assert (hc.red(), hc.green(), hc.blue()) == (0x43, 0xa0, 0x47)
    with pytest.raises(ValueError):
        et.set_block_color(win.scene, "mavi")
    win.undo()
    win.undo()
    assert win.scene.graph.nodes[b.node.id].color is None


def test_block_color_menu(win, monkeypatch):
    a, b, c = three(win)
    a.setSelected(True)
    no_menu_exec(monkeypatch)
    win.context_menu(QPoint(0, 0), QPointF(0, 0))
    sub = next(x.menu() for x in win._context_menu.actions() if x.text() == "Blok rengi")
    names = [x.text() for x in sub.actions()]
    assert "Kırmızı" in names and names[0].startswith("Renksiz")
    next(x for x in sub.actions() if x.text() == "Mavi").trigger()
    assert a.node.color == "#1e88e5"
    sub.actions()[0].trigger()
    assert a.node.color is None


def test_export_huge_flow_is_scaled_down(win, tmp_path):
    win.scene.add_node("Grayscale", QPointF(0, 0))
    win.scene.add_node("Blur", QPointF(4500, 4500))
    img = QImage(str(et.export_image(win.scene, tmp_path / "buyuk.png")))
    assert 0 < img.width() <= et.MAX_SIDE and img.width() * img.height() <= et.MAX_PIXELS * 1.01
