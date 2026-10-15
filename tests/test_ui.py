"""Editor tests, run headless with the Qt offscreen platform."""
import os
import pathlib

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from visionflow.core import load_builtin  # noqa: E402
from visionflow.ui.main_window import MainWindow  # noqa: E402

load_builtin()
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def win():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    yield w
    w.stop()
    w.saved_snap = w._snap()   # skip the "save changes?" prompt
    w.close()
    app.processEvents()


def wait_for(cond, ms=4000):
    for _ in range(ms // 20):
        QtWidgets.QApplication.processEvents()
        if cond():
            return True
        QTest.qWait(20)
    return False


def flush(win):
    win._snap_timer.stop()
    win._snapshot()


def test_undo_redo_add_and_delete(win):
    sc = win.scene
    sc.add_node("Grayscale", QPointF(0, 0))
    flush(win)
    sc.add_node("Blur", QPointF(250, 0))
    flush(win)
    assert len(sc.graph.nodes) == 2
    win.undo()
    assert len(sc.graph.nodes) == 1
    win.undo()
    assert len(sc.graph.nodes) == 0
    win.redo()
    win.redo()
    assert len(sc.graph.nodes) == 2
    sc.clearSelection()
    win.select_all()
    sc.delete_selected()
    flush(win)
    assert not sc.graph.nodes
    win.undo()
    assert len(sc.graph.nodes) == 2


def test_copy_paste_keeps_internal_links(win):
    win.open_flow(str(EXAMPLES / "kenar_algilama.json"))
    sc = win.scene
    n0, c0 = len(sc.graph.nodes), len(sc.graph.connections)
    win.select_all()
    win.copy()
    win.paste(QPointF(0, 800))
    assert len(sc.graph.nodes) == 2 * n0 and len(sc.graph.connections) == 2 * c0
    assert win.dirty


def test_delete_key_ignored_while_typing(win):
    sc = win.scene
    item = sc.add_node("ColorRange", QPointF(0, 0))
    item.setSelected(True)
    QtWidgets.QApplication.processEvents()
    line_edits = win.props.findChildren(QtWidgets.QAbstractSpinBox)
    assert line_edits
    line_edits[0].setFocus()
    QTest.keyClick(line_edits[0], Qt.Key_Delete)
    QTest.keyClick(line_edits[0], Qt.Key_F)
    assert len(sc.graph.nodes) == 1          # node survived
    win.view.setFocus()
    QTest.keyClick(win.view.viewport(), Qt.Key_Delete)
    assert len(sc.graph.nodes) == 0


def test_run_example_and_live_param(win):
    win.open_flow(str(EXAMPLES / "renkli_nesne_sayma.json"))
    sc = win.scene
    fc = next(i for i in sc.items_by_id.values() if i.node.type_name == "FindContours")
    win.start()
    assert wait_for(lambda: "count=1" in fc.status_text), fc.status_text
    cr = next(i for i in sc.items_by_id.values() if i.node.type_name == "ColorRange")
    for k, v in (("h_min", 40), ("h_max", 41)):   # a hue band nothing in the scene has
        cr.node.values[k] = v
        win.on_param(cr.node.id, k, v)
    assert wait_for(lambda: "count=0" in fc.status_text), fc.status_text
    assert not any(i.error for i in sc.items_by_id.values())
    win.stop()
    assert win.worker is None


def test_rewire_while_running_restarts(win):
    win.open_flow(str(EXAMPLES / "kenar_algilama.json"))
    sc = win.scene
    win.start()
    assert wait_for(lambda: all(i.ms is not None for i in sc.items_by_id.values()))
    th = sc.add_node("Threshold", QPointF(0, 400))
    canny = next(i for i in sc.items_by_id.values() if i.node.type_name == "Canny")
    sc.try_connect(canny.sockets[("image", True)], th.sockets[("image", False)])
    assert wait_for(lambda: th.ms is not None and th.error is None)


def test_save_and_reload(win, tmp_path):
    win.open_flow(str(EXAMPLES / "renkli_nesne_sayma.json"))
    path = tmp_path / "x.json"
    win.path = str(path)
    assert win.save_flow()
    assert not win.dirty
    win.new_flow()
    win.open_flow(str(path))
    assert len(win.scene.graph.nodes) == 7 and win.path == str(path)
