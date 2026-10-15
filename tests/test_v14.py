"""1.4: automatic labelling in the labelling tool (model suggestions, label all, copy previous, undo box)."""
import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from visionflow.core import load_builtin  # noqa: E402
from visionflow.ui.tools import Dataset, LabelingDialog  # noqa: E402

load_builtin()


@pytest.fixture(autouse=True)
def app(monkeypatch):
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    msgs = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *x, **k: msgs.append(x[2]))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *x, **k: msgs.append(x[2]))
    a.msgs = msgs
    return a


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "images").mkdir()
    for i in range(4):
        cv2.imwrite(str(tmp_path / "images" / f"{i}.jpg"), np.full((100, 200, 3), 90, np.uint8))
    return tmp_path


def fake(dets):
    calls = []

    def detect(img):
        calls.append(img.shape)
        return [dict(d) for d in dets]
    detect.calls = calls
    return detect


def lines(folder, name):
    p = folder / "labels" / f"{name}.txt"
    return p.read_text().splitlines() if p.exists() else []


def test_suggest_adds_model_boxes_and_new_classes(folder):
    dlg = LabelingDialog(folder=str(folder))
    dlg.detector = fake([{"box": (20, 10, 40, 80), "label": "vida", "score": 0.9},
                         {"box": (120, 10, 30, 30), "label": "kişi #3", "score": 0.8},
                         {"box": (0, 0, 0, 0), "label": "x", "score": 0.9},          # empty: skipped
                         {"box": (float("nan"), 0, 5, 5), "label": "x", "score": 0.9},
                         {"label": "kutusuz"}])
    dlg.suggest()
    assert dlg.ds.classes == ["vida", "kişi #3"]               # the empty dataset's "nesne" placeholder is replaced
    assert [b[0] for b in dlg.canvas.boxes] == [0, 1]
    dlg.suggest()                                                # same boxes again: not doubled
    assert len(dlg.canvas.boxes) == 2
    assert len(lines(folder, "0")) == 2
    dlg.close()


def test_suggest_keeps_existing_classes_and_boxes(folder):
    (folder / "classes.txt").write_text("vida\nsomun")
    (folder / "labels").mkdir(exist_ok=True)
    (folder / "labels" / "0.txt").write_text("1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.05 0.05 0.05 0.1 0.1")   # + polygon
    dlg = LabelingDialog(folder=str(folder))
    dlg.detector = fake([{"box": (80, 40, 40, 20), "label": "somun", "score": 0.9},     # = the existing box
                         {"box": (10, 10, 20, 20), "label": "vida", "score": 0.9}])
    dlg.suggest()
    assert dlg.ds.classes == ["vida", "somun"] and len(dlg.canvas.boxes) == 2
    assert len(lines(folder, "0")) == 3 and lines(folder, "0")[-1].startswith("0 0.1 0.1 0.05")
    dlg.close()


def test_label_all_only_touches_unlabelled_images(folder, app):
    (folder / "labels").mkdir(exist_ok=True)
    (folder / "labels" / "1.txt").write_text("0 0.5 0.5 0.1 0.1")
    (folder / "classes.txt").write_text("vida")
    dlg = LabelingDialog(folder=str(folder))
    det = fake([{"box": (20, 10, 40, 80), "label": "vida", "score": 0.9}])
    dlg.detector = det
    dlg.label_all()
    assert len(det.calls) == 3                                   # 0, 2, 3
    assert lines(folder, "1") == ["0 0.5 0.5 0.1 0.1"]
    for n in ("0", "2", "3"):
        assert lines(folder, n) and lines(folder, n)[0].startswith("0 ")
    assert len(dlg.canvas.boxes) == 1                            # the shown image reloaded with its new box
    assert "3 resme 3 kutu" in app.msgs[-1]
    dlg.label_all()
    assert app.msgs[-1] == "Etiketsiz resim yok."
    dlg.close()


def test_broken_model_does_not_crash(folder, app):
    dlg = LabelingDialog(folder=str(folder))

    def broken(img):
        raise RuntimeError("model bozuk")
    dlg.detector = broken
    dlg.suggest()
    dlg.label_all()
    assert all("model bozuk" in m for m in app.msgs[-2:])
    assert not lines(folder, "0")
    dlg.close()


def test_missing_model_file_is_reported(folder, app, tmp_path):
    dlg = LabelingDialog(folder=str(folder))
    assert dlg.choose_model(str(tmp_path / "yok.onnx")) is False and dlg.detector is None
    assert app.msgs
    dlg.close()


def test_copy_previous_scales_and_skips_duplicates(folder):
    cv2.imwrite(str(folder / "images" / "1.jpg"), np.full((200, 400, 3), 90, np.uint8))   # twice the size
    dlg = LabelingDialog(folder=str(folder))
    dlg.canvas.boxes.append([0, 20, 10, 60, 90])
    dlg._changed()
    dlg.go(1)
    dlg.copy_previous()
    assert dlg.canvas.boxes == [[0, pytest.approx(40, abs=1), pytest.approx(20, abs=1), pytest.approx(120, abs=1),
                                 pytest.approx(180, abs=1)]]
    dlg.copy_previous()
    assert len(dlg.canvas.boxes) == 1
    dlg.go(2)
    assert len(Dataset(folder).load(folder / "images" / "1.jpg", 400, 200)) == 1
    dlg.go(0)
    dlg.copy_previous()                                          # first image: nothing before it
    dlg.close()


def test_undo_box(folder):
    dlg = LabelingDialog(folder=str(folder))
    dlg.canvas.boxes += [[0, 1, 1, 50, 50], [0, 60, 10, 90, 40]]
    dlg._changed()
    dlg.undo_box()
    assert len(dlg.canvas.boxes) == 1 and len(lines(folder, "0")) == 1
    dlg.close()


def test_real_onnx_model_suggests_boxes(folder, tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    raw = np.zeros((1, 84, 8400), np.float32)
    raw[0, :4, 0] = (320, 320, 100, 100)
    raw[0, 4 + 2, 0] = 0.95                                     # class 2 = car
    const = helper.make_node("Constant", [], ["output0"], value=helper.make_tensor(
        "v", TensorProto.FLOAT, raw.shape, raw.flatten().tolist()))
    graph = helper.make_graph([const], "fake_yolo",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                              [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "fake.onnx"
    onnx.save(model, path)
    dlg = LabelingDialog(folder=str(folder))
    assert dlg.choose_model(str(path)) and "fake.onnx" in dlg.model_label.text()
    dlg.suggest()
    assert dlg.ds.classes == ["car"] and len(dlg.canvas.boxes) == 1
    dlg.close()



# --- 1.4 bug hunt ---------------------------------------------------------------------------------------------------

def test_label_all_never_overwrites_boxes_drawn_meanwhile(folder, app):
    dlg = LabelingDialog(folder=str(folder))
    dlg.go(2)

    def detect(img):
        if not dlg.canvas.boxes:                  # the user draws on image 2 while the model runs on image 0
            dlg.canvas.boxes.append([0, 5, 5, 50, 50])
            dlg._changed()
        return [{"box": (100, 10, 40, 40), "label": "vida", "score": 0.9}]
    dlg.detector = detect
    dlg.label_all()
    assert lines(folder, "2")[0].startswith("0 0.1375")            # the user's own box survived
    dlg.close()


def test_label_all_stops_when_another_folder_is_opened(folder, tmp_path, app):
    other = tmp_path / "baska"
    other.mkdir()
    cv2.imwrite(str(other / "x.jpg"), np.zeros((50, 50, 3), np.uint8))
    dlg = LabelingDialog(folder=str(folder))
    calls = []

    def detect(img):
        calls.append(1)
        if len(calls) == 1:
            dlg.open_folder(str(other))
        return [{"box": (10, 10, 20, 20), "label": "vida", "score": 0.9}]
    dlg.detector = detect
    dlg.label_all()
    assert not (other / "labels" / "0.txt").exists() and len(calls) == 1
    dlg.close()


def test_class_names_are_kept_whole_and_on_one_line(folder):
    (folder / "classes.txt").write_text("parca #1\nparca #2\nvida \n")
    dlg = LabelingDialog(folder=str(folder))
    dlg.detector = fake([{"box": (10, 10, 30, 30), "label": "parca #2", "score": 0.9},
                         {"box": (60, 10, 30, 30), "label": "vida", "score": 0.9},
                         {"box": (110, 10, 30, 30), "label": "a\nb", "score": 0.9}])
    dlg.suggest()
    assert dlg.ds.classes == ["parca #1", "parca #2", "vida", "a b"]
    assert [b[0] for b in dlg.canvas.boxes] == [1, 2, 3]
    dlg.close()
    assert Dataset(folder).classes == ["parca #1", "parca #2", "vida", "a b"]


def test_flat_folder_with_same_stem_keeps_one_label_file(tmp_path, app):
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((50, 50, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((50, 50, 3), np.uint8))
    dlg = LabelingDialog(folder=str(tmp_path))
    det = fake([{"box": (10, 10, 20, 20), "label": "vida", "score": 0.9}])
    dlg.detector = det
    dlg.label_all()
    assert len(det.calls) == 1 and len(lines(tmp_path, "a") if (tmp_path / "labels").exists() else
                                       (tmp_path / "a.txt").read_text().splitlines()) == 1
    dlg.close()


def test_boxes_outside_the_picture_are_not_suggested(folder):
    dlg = LabelingDialog(folder=str(folder))
    dlg.detector = fake([{"box": (500, 500, 40, 40), "label": "vida", "score": 0.9},
                         {"box": (-20, -20, 60, 60), "label": "vida", "score": 0.9}])
    dlg.suggest()
    assert dlg.canvas.boxes == [[0, 0.0, 0.0, 40.0, 40.0]]
    dlg.close()
