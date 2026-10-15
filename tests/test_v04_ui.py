"""0.4 editor features: auto layout, autosave / crash recovery, help, paste keeps bypass."""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QPointF  # noqa: E402

from visionflow.core import Graph, load_builtin  # noqa: E402
from visionflow.ui.main_window import MainWindow  # noqa: E402

load_builtin()


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app, tmp_path):
    w = MainWindow()
    w.recovery = tmp_path / "kurtarma.json"
    w.show()
    yield w
    w.stop()
    w.saved_snap = w._snap()
    w.close()


def chain(win, n=4):
    items = [win.scene.add_node(t, QPointF(i * 37, (n - i) * 91))
             for i, t in enumerate(["TestPattern", "Blur", "Grayscale", "Canny"][:n])]
    for a, b in zip(items, items[1:]):
        win.scene.graph.connect(a.node.id, "image", b.node.id, "image")
    win.scene.load_graph(win.scene.graph)
    return [it.node.id for it in items]


def test_graph_auto_layout_columns():
    g = Graph()
    src = g.add_new("TestPattern")
    a, b = g.add_new("Blur", pos=(0, 500)), g.add_new("Grayscale", pos=(0, 100))
    note = g.add_new("Note", pos=(999, 999))
    g.connect(src.id, "image", a.id, "image")
    g.connect(src.id, "image", b.id, "image")
    g.auto_layout(dx=200, gap=10, heights={src.id: 100, a.id: 100, b.id: 100})
    assert src.pos[0] == 0 and a.pos[0] == b.pos[0] == 200
    assert b.pos[1] < a.pos[1]                       # keeps the user's vertical order
    assert src.pos[1] == pytest.approx(55)           # single block centered on its 2-block neighbour column
    assert note.pos == (999, 999)


def test_auto_layout_in_editor_is_undoable(win):
    ids = chain(win)
    before = {i: win.scene.graph.nodes[i].pos for i in ids}
    win._snapshot()
    win.scene.auto_layout()
    win._snapshot()
    xs = [win.scene.items_by_id[i].pos().x() for i in ids]
    assert xs == sorted(xs) and len(set(xs)) == 4
    assert len({win.scene.items_by_id[i].pos().y() for i in ids}) == 1
    win.undo()
    assert {i: win.scene.graph.nodes[i].pos for i in ids} == before


def test_autosave_and_recovery(win, app):
    chain(win, 3)
    win.autosave()
    assert win.recovery.exists()
    data = json.loads(win.recovery.read_text(encoding="utf-8"))
    assert len(data["graph"]["nodes"]) == 3

    other = MainWindow()                       # "next start" after a crash
    other.recovery = win.recovery
    assert other.check_recovery(ask=False)
    assert len(other.scene.graph.nodes) == 3 and len(other.scene.graph.connections) == 2
    assert other.dirty                          # restored work must still be saved by the user
    other.saved_snap = other._snap()
    other.close()


def test_save_clears_recovery(win, tmp_path):
    chain(win, 2)
    win.autosave()
    assert win.recovery.exists()
    win.path = str(tmp_path / "a.json")
    win.save_flow()
    assert not win.recovery.exists()
    win.autosave()                              # nothing unsaved -> nothing written
    assert not win.recovery.exists()


def test_corrupt_recovery_is_ignored(win):
    win.recovery.write_text("{bozuk", encoding="utf-8")
    assert not win.check_recovery(ask=False)
    assert not win.recovery.exists()


def test_paste_keeps_disabled_flag(win):
    item = win.scene.add_node("Blur", QPointF(0, 0))
    item.node.enabled = False
    win.scene.clearSelection()
    item.setSelected(True)
    clip = win.scene.copy_selected()
    win.scene.paste(clip, QPointF(300, 0))
    pasted = [n for n in win.scene.graph.nodes.values() if n.id != item.node.id]
    assert len(pasted) == 1 and pasted[0].enabled is False


def test_help_and_menus_exist(win):
    titles = [a.text() for a in win.menuBar().actions()]
    assert "&Yardım" in titles
    texts = [a.text() for m in win.menuBar().findChildren(QtWidgets.QMenu) for a in m.actions()]
    assert "Otomatik yerleştir" in texts and "Kısayollar ve kullanım" in texts
