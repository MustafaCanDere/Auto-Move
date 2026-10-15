"""0.3 editor features: bypass, notes, quick add, labelling tool, training window."""
import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from visionflow.core import Graph, load_builtin  # noqa: E402
from visionflow.train import main as train_main  # noqa: E402
from visionflow.ui.main_window import MainWindow  # noqa: E402
from visionflow.ui.tools import Dataset, LabelingDialog, TrainDialog  # noqa: E402

load_builtin()


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app):
    w = MainWindow()
    w.show()
    yield w
    w.stop()
    w.saved_snap = w._snap()
    w.close()


def test_disabled_block_is_bypassed_and_saved():
    g = Graph()
    s = g.add_new("TestPattern", animate=False)
    gray = g.add_new("Grayscale")
    canny = g.add_new("Canny")
    g.connect(s.id, "image", gray.id, "image")
    g.connect(gray.id, "image", canny.id, "image")
    gray.enabled = False
    g.setup()
    res = g.run_once()
    assert res[gray.id].outputs["image"].ndim == 3      # colour image passed straight through
    assert res[canny.id].error is None
    g2 = Graph.from_dict(g.to_dict())
    assert not g2.nodes[gray.id].enabled


def test_toggle_disable_from_editor(win):
    item = win.scene.add_node("Blur", QPointF(0, 0))
    item.setSelected(True)
    win.scene.toggle_enabled_selected()
    assert not item.node.enabled
    win.scene.toggle_enabled_selected()
    assert item.node.enabled


def test_note_grows_with_text(win):
    note = win.scene.add_node("Note", QPointF(0, 0))
    h0 = note.height
    note.node.values["text"] = "\n".join(["satır"] * 10)
    win.on_param(note.node.id, "text", note.node.values["text"])
    assert note.height > h0


def test_quick_add_filters_and_adds(win):
    win._show_quick_add(None, None)
    qa = win.quick_add
    QTest.keyClicks(qa.edit, "anomali tespiti")
    assert qa.list.count() == 1
    QTest.keyClick(qa.edit, Qt.Key_Return)
    assert [n.type_name for n in win.scene.graph.nodes.values()] == ["AnomalyDetector"]


def test_labeling_tool_writes_yolo(app, tmp_path):
    (tmp_path / "images").mkdir()
    for i in range(3):
        cv2.imwrite(str(tmp_path / "images" / f"{i}.jpg"), np.full((100, 200, 3), 90, np.uint8))
    dlg = LabelingDialog(folder=str(tmp_path))
    dlg.resize(1000, 700)
    dlg.show()
    dlg.add_class("vida")
    assert dlg.canvas.current_class == 1
    c = dlg.canvas
    a, b = c.to_widget(20, 10), c.to_widget(60, 90)
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, a.toPoint())
    QTest.mouseMove(c, b.toPoint())
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, b.toPoint())
    dlg.go(1)
    parts = (tmp_path / "labels" / "0.txt").read_text().split()
    assert parts[0] == "1" and float(parts[1]) == pytest.approx(0.2, abs=0.02)
    assert (tmp_path / "classes.txt").read_text().splitlines() == ["nesne", "vida"]
    assert "train: \"images\"" in (tmp_path / "data.yaml").read_text()
    ds = Dataset(tmp_path)
    assert ds.labelled_count() == 1
    dlg.close()


def test_train_window_command(app, tmp_path):
    dlg = TrainDialog()
    dlg.data.setText(str(tmp_path / "data.yaml"))
    dlg.device.setCurrentText("cpu")
    cmd = dlg.command()
    assert cmd[:3] == ["-u", "-m", "visionflow.train"] and "--device" in cmd and "cpu" in cmd


def test_train_script_reports_missing_ultralytics(capsys, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("ultralytics"):
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert train_main(["--data", "x.yaml"]) == 2
    assert "pip install ultralytics" in capsys.readouterr().out
