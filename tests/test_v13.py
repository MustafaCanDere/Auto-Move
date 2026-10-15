"""1.3: kendi kendine öğrenen görüntü (uncertain samples → teach → retrain), data augmentation."""
import json
import os
import pathlib
import time

import cv2
import numpy as np
import pytest

from visionflow.ai.config import AIConfig
from visionflow.core import Graph, create, load_builtin
from visionflow.nodes import learning

load_builtin()
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    AIConfig(api_key="test").save()


def make(t, **v):
    n = create(t)
    n.values.update(v)
    n.setup()
    return n


def shape(kind, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((80, 80, 3), 30, np.uint8)
    c = tuple(int(x) for x in rng.integers(150, 255, 3))
    if kind == "top":
        cv2.circle(img, (40, 40), 25 + int(rng.integers(-3, 3)), c, -1)
    else:
        cv2.rectangle(img, (14, 14), (66 + int(rng.integers(-3, 3)), 66), c, -1)
    return img


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "veri"
    for kind in ("top", "kutu"):
        (root / kind).mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(root / kind / f"{i}.png"), shape(kind, i))
    return root


def test_review_folder_is_not_a_class(data):
    (data / "_incelenecek").mkdir()
    cv2.imwrite(str(data / "_incelenecek" / "a.png"), shape("top"))
    m = learning.QuickModel().fit(data)
    assert m.classes == ["kutu", "top"]


def test_augmentation_adds_samples_and_still_classifies(data):
    plain = learning.QuickModel().fit(data)
    aug = learning.QuickModel(augment=True).fit(data)
    assert len(aug.y) == 6 * len(plain.y)
    assert aug.predict(shape("top", 9))[0] == "top" and aug.predict(shape("kutu", 9))[0] == "kutu"


def test_classifier_augment_param(data):
    n = make("QuickClassifier", folder=str(data), augment=True)
    assert n.process({"image": shape("top", 5), "detections": None})["label"] == "top"
    assert len(n.state["model"].y) == 36


def test_collector_saves_only_uncertain_and_waits_between(data):
    n = make("UncertainCollector", folder=str(data), low=0.0, high=0.75, interval=60)
    img = shape("top")
    assert n.process({"image": img, "score": 0.95, "label": "top", "detections": None}) == {"saved": 0, "uncertain": False}
    out = n.process({"image": img, "score": 0.5, "label": "top", "detections": None})
    assert out == {"saved": 1, "uncertain": True}
    n.process({"image": img, "score": 0.4, "label": "top", "detections": None})        # too soon
    files = list((data / "_incelenecek").glob("*.png"))
    assert len(files) == 1 and learning.guess_from_name(files[0].name) == ("top", 0.5)
    assert n.process({"image": img, "score": float("nan"), "label": "", "detections": None})["uncertain"] is False


def test_collector_crops_uncertain_detections_and_respects_limit(data):
    n = make("UncertainCollector", folder=str(data), interval=0, max_files=3)
    img = np.zeros((100, 100, 3), np.uint8)
    dets = [{"box": [10, 10, 20, 20], "label": "a/b", "score": 0.3},
            {"box": [50, 50, 20, 20], "label": "kutu", "score": 0.99},
            {"box": [60, 10, 10, 10], "label": "top", "score": 0.45}]
    n.process({"image": img, "score": None, "label": None, "detections": dets})
    files = sorted((data / "_incelenecek").glob("*.png"))
    assert len(files) == 2 and all("/" not in f.name for f in files)
    assert cv2.imread(str(files[0])).shape[:2] == (40, 40)                      # 20 px box + 10 px pad
    for _ in range(3):
        n.process({"image": img, "score": None, "label": None, "detections": dets})
    assert len(list((data / "_incelenecek").glob("*.png"))) == 3


def test_collector_needs_score_or_detections(data):
    n = make("UncertainCollector", folder=str(data))
    with pytest.raises(ValueError):
        n.process({"image": shape("top"), "score": None, "label": None, "detections": None})


def test_example_flow_runs(tmp_path, monkeypatch):
    import shutil
    work = tmp_path / "w"
    shutil.copytree(EXAMPLES / "ornek_veri" / "sekiller", work / "examples" / "ornek_veri" / "sekiller")
    monkeypatch.chdir(work)
    g = Graph.from_dict(json.loads((EXAMPLES / "kendi_kendine_ogrenen.json").read_text(encoding="utf-8")))
    assert not g.warnings
    errs = g.setup()
    assert not errs
    for _ in range(3):
        res = g.run_once()
    assert not [r.error for r in res.values() if r.error]


def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_teach_dialog_moves_samples_and_classifier_retrains(data, monkeypatch):
    qapp()
    from visionflow.ui.teach_dialog import TeachDialog
    clf = make("QuickClassifier", folder=str(data))
    clf.process({"image": shape("top"), "detections": None})
    before = len(clf.state["model"].y)
    col = make("UncertainCollector", folder=str(data), interval=0)
    for i in range(3):
        col.process({"image": shape("top", 20 + i), "score": 0.3, "label": "kutu", "detections": None})
        time.sleep(0.002)
    dlg = TeachDialog(folder=str(data))
    assert len(dlg.files) == 3 and "kutu" in dlg.guess.text()
    assert [dlg.grid.itemAt(i).widget().text() for i in range(dlg.grid.count())] == ["1  kutu", "2  top"]
    dlg._pick_index(1)                         # key "2" → top
    dlg.delete()
    dlg.undo()                                 # the deleted one comes back
    assert len(dlg.files) == 2
    dlg.skip()
    dlg.put("top")
    dlg.put("top")
    assert dlg.current() is None and len(list((data / "top").glob("*.png"))) == 6
    monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getText", lambda *a, **k: ("leke", True))
    dlg.undo()
    dlg.new_class()
    assert len(list((data / "leke").glob("*.png"))) == 1
    dlg.reject()
    assert not list((data / "_incelenecek").glob("_silinen*"))
    os.utime(data, (time.time() + 5, time.time() + 5))
    clf.state["checked"] = 29                  # next frame looks at the folder again
    clf.process({"image": shape("top"), "detections": None})
    assert len(clf.state["model"].y) == before + 3 and clf.state["model"].classes == ["kutu", "leke", "top"]


def test_menu_opens_teach_on_flow_folder(tmp_path):
    qapp()
    from visionflow.ui.main_window import MainWindow
    win = MainWindow()
    try:
        from PySide6.QtCore import QPointF
        win.scene.add_node("UncertainCollector", QPointF(0, 0))
        node = next(iter(win.scene.graph.nodes.values()))
        node.values["folder"] = str(tmp_path / "benim")
        dlg = win.open_teach()
        assert dlg.root == tmp_path / "benim"
        dlg.close()
    finally:
        win._confirm_discard = lambda: True
        win.close()


# --- 1.3 bug hunt ---------------------------------------------------------------------------------------------------

def test_collecting_does_not_retrain(data):
    clf = make("QuickClassifier", folder=str(data))
    col = make("UncertainCollector", folder=str(data), interval=0)
    clf.process({"image": shape("top"), "detections": None})
    model = clf.state["model"]
    for i in range(60):
        col.process({"image": shape("top", i), "score": 0.3, "label": "top", "detections": None})
        clf.process({"image": shape("top"), "detections": None})
    assert clf.state["model"] is model                               # same model: nothing was retrained
    cv2.imwrite(str(data / "top" / "yeni.png"), shape("top", 99))
    clf.state["checked"] = 29
    clf.process({"image": shape("top"), "detections": None})
    assert clf.state["model"] is not model


def test_augmentation_keeps_scores_meaningful(data):
    aug = learning.QuickModel(k=3, augment=True).fit(data)
    # the 3 nearest neighbours must be 3 different photos, not 3 variants of one
    blend = cv2.addWeighted(shape("top", 3), 0.5, shape("kutu", 3), 0.5, 0)
    assert aug.predict(blend)[1] < 1.0


def test_crop_box_handles_boxes_off_the_image():
    img = np.zeros((100, 200, 3), np.uint8)
    assert learning.crop_box(img, [-50, 10, 20, 20], 10) is None
    assert learning.crop_box(img, [190, 90, 50, 50], 0).shape[:2] == (10, 10)
    assert learning.crop_box(img, [float("nan"), 0, 5, 5]) is None
    assert learning.crop_box(img, None) is None
    n = make("UncertainCollector", folder="x", interval=0)
    out = n.process({"image": img, "score": None, "label": None,
                     "detections": [{"box": [-50, 10, 20, 20], "score": 0.3}, {"label": "a", "score": 0.3}]})
    assert out["saved"] == 0


def test_empty_frames_are_not_collected(data):
    n = make("UncertainCollector", folder=str(data), interval=0)
    assert n.process({"image": shape("top"), "score": 0.0, "label": "", "detections": None})["uncertain"] is False


def test_two_collectors_never_overwrite_each_other(data, monkeypatch):
    class Frozen(learning.dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 12, 0, 0)
    monkeypatch.setattr(learning.dt, "datetime", Frozen)
    a = make("UncertainCollector", folder=str(data), interval=0)
    b = make("UncertainCollector", folder=str(data), interval=0)
    b.id = a.id                                                       # worst case: even the ids clash
    for n in (a, b):
        n.process({"image": shape("top"), "score": 0.3, "label": "top", "detections": None})
    assert len(list((data / "_incelenecek").glob("*.png"))) == 2


def test_labels_survive_the_file_name():
    for label in ("vida #12", "çizik__derin", "a/b", ""):
        name = f"x_0__{learning.safe_label(label)}__0.40.png"
        got, score = learning.guess_from_name(name)
        assert score == 0.4 and "__" not in got and got
    assert learning.guess_from_name("x__" + learning.safe_label("vida #12") + "__0.40.png")[0] == "vida_12"


def test_teach_windows_do_not_share_the_trash(data):
    qapp()
    from visionflow.ui.teach_dialog import TeachDialog
    col = make("UncertainCollector", folder=str(data), interval=0)
    for i in range(2):
        col.process({"image": shape("top", i), "score": 0.3, "label": "top", "detections": None})
    d1, d2 = TeachDialog(folder=str(data)), TeachDialog(folder=str(data))
    d2.delete()
    d1.reject()
    d2.undo()
    assert len(list((data / "_incelenecek").glob("*.png"))) == 2
    d2.reject()


def test_enter_in_folder_box_does_not_open_a_picker(data, monkeypatch):
    app = qapp()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from visionflow.ui.teach_dialog import TeachDialog
    opened = []
    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getExistingDirectory", lambda *a, **k: opened.append(1) or "")
    d = TeachDialog(folder=str(data))
    d.show()
    d.folder_edit.setFocus()
    QTest.keyClick(d.folder_edit, Qt.Key_Return)
    app.processEvents()
    assert opened == []
    d.reject()


def test_new_class_with_bad_name_is_refused(data, monkeypatch):
    qapp()
    from PySide6.QtWidgets import QInputDialog, QMessageBox
    from visionflow.ui.teach_dialog import TeachDialog
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    (data / "dosya").write_text("x")
    d = TeachDialog(folder=str(data))
    for name in ("dosya", "CON", "_gizli"):
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, n=name, **k: (n, True))
        d.new_class()
    assert len(warned) == 3
    d.reject()


def test_teach_menu_opens_one_window():
    qapp()
    from visionflow.ui.main_window import MainWindow
    win = MainWindow()
    try:
        assert win.open_teach() is win.open_teach()
    finally:
        win._teach.reject()
        win._confirm_discard = lambda: True
        win.close()


def test_certainty_is_low_for_in_between_images(data):
    m = learning.QuickModel(k=3).fit(data)
    clear = m.predict_full(shape("top", 7))[3]
    blend = m.predict_full(cv2.addWeighted(shape("top", 3), 0.5, shape("kutu", 3), 0.5, 0))[3]
    assert 0 <= blend < clear <= 1
    n = make("QuickClassifier", folder=str(data))
    out = n.process({"image": shape("top"), "detections": [{"box": [0, 0, 80, 80], "score": 1.0}]})
    assert "certainty" in out and "certainty" in out["detections"][0]


def test_example_flow_collects_hard_cases(tmp_path, monkeypatch):
    import shutil
    work = tmp_path / "w"
    shutil.copytree(EXAMPLES / "ornek_veri" / "sekiller", work / "examples" / "ornek_veri" / "sekiller")
    monkeypatch.chdir(work)
    g = Graph.from_dict(json.loads((EXAMPLES / "kendi_kendine_ogrenen.json").read_text(encoding="utf-8")))
    g.setup()
    for _ in range(60):
        g.run_once()
    assert list((work / "examples/ornek_veri/sekiller/_incelenecek").glob("*.png"))


def test_few_similar_samples_still_classify(data):
    # 3 photos per class on a plain background: most features are constant, which used to blow up the scaling
    # and made the classifier guess (9 of 40 circles right)
    m = learning.QuickModel(k=3).fit(data)
    assert sum(m.predict(shape("top", s))[0] == "top" for s in range(5, 45)) >= 38
    assert sum(m.predict(shape("kutu", s))[0] == "kutu" for s in range(5, 45)) >= 38
