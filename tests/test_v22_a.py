"""2.2: Az Örnekle Nesne Bul — few-shot object detection engine, block, box drawing/training window and example."""
import json
import os
import pathlib
import sys
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _fewshot_det_synth import evaluate, scene  # noqa: E402
from visionflow import fewshot_det as fd  # noqa: E402
from visionflow.core import Graph, create, load_builtin  # noqa: E402

load_builtin()
ROOT = pathlib.Path(__file__).resolve().parents[1]
TWO = ("braket", "etiket")


def score_on(model, seeds, kinds, **kw):
    tp = fp = fn = 0
    for s in seeds:
        img, gt = scene(s, kinds, **kw)
        a, b, c = evaluate(model.detect(img), gt)
        tp, fp, fn = tp + a, fp + b, fn + c
    return tp / max(1, tp + fn), tp / max(1, tp + fp)


@pytest.fixture(scope="module")
def one():
    data = [scene(s) for s in range(4)]
    return fd.DetModel.train([d[0] for d in data], [d[1] for d in data], ["braket"]), data


@pytest.fixture(scope="module")
def two():
    data = [scene(s, TWO) for s in range(5)]
    return fd.DetModel.train([d[0] for d in data], [d[1] for d in data], list(TWO))


# --- engine ---------------------------------------------------------------------------------------------------------

def test_single_class_recall_precision_on_unseen(one):
    model, _ = one
    # unseen scenes: parts rotated ±30°, scaled 0.8–1.2, among same-coloured circles/triangles/bars and clutter
    recall, precision = score_on(model, range(100, 115), ("braket",))
    assert recall >= 0.9 and precision >= 0.9, (recall, precision)
    info = model.info["classes"]["braket"]
    assert info["examples"] == sum(len(d[1]) for d in one[1]) and info["train_accuracy"] > 0.95


def test_trained_on_three_pictures():
    data = [scene(s, ("etiket",)) for s in range(50, 53)]
    model = fd.DetModel.train([d[0] for d in data], [d[1] for d in data], ["etiket"])
    recall, precision = score_on(model, range(1050, 1062), ("etiket",))
    assert recall >= 0.9 and precision >= 0.9, (recall, precision)


def test_multi_class(two):
    recall, precision = score_on(two, range(300, 315), TWO)
    assert recall >= 0.9 and precision >= 0.9, (recall, precision)
    img, gt = scene(301, TWO)
    dets = two.detect(img)
    for d in dets:                                    # the YOLO blocks' detection format (+ rotated box)
        assert set(d) >= {"box", "score", "class_id", "label", "angle", "poly", "center", "size"}
        assert d["label"] == TWO[d["class_id"]] and 0 <= d["score"] <= 1 and len(d["poly"]) == 4
        assert all(isinstance(v, int) for v in d["box"])
    # class filter by name or number
    only = two.detect(img, classes="etiket")
    assert only and all(d["label"] == "etiket" for d in only)
    assert [d["box"] for d in two.detect(img, classes=[1])] == [d["box"] for d in only]
    with pytest.raises(ValueError, match="sınıfı modelde yok"):
        two.detect(img, classes="vida")
    assert len(two.detect(img, max_objects=1)) == 1
    assert two.detect(img, score=0.999999) == [] or all(d["score"] >= 0.999999 for d in two.detect(img, score=0.999999))


def test_robust_to_lighting_blur_noise(two):
    rng = np.random.default_rng(0)
    for fx in (lambda i: np.clip(i * 1.25 + 10, 0, 255).astype(np.uint8), lambda i: (i * 0.7).astype(np.uint8),
               lambda i: cv2.GaussianBlur(i, (0, 0), 1.2),
               lambda i: np.clip(i + rng.normal(0, 10, i.shape), 0, 255).astype(np.uint8)):
        tp = fp = fn = 0
        for s in range(400, 406):
            img, gt = scene(s, TWO)
            a, b, c = evaluate(two.detect(fx(img)), gt)
            tp, fp, fn = tp + a, fp + b, fn + c
        assert tp / (tp + fn) >= 0.85 and tp / max(1, tp + fp) >= 0.85


def test_angle_and_rotated_box():
    """One upright example; the found angle follows the part's rotation (counter-clockwise positive)."""
    from _fewshot_det_synth import _paste, _part
    rng = np.random.default_rng(1)

    def make(angle, scale=1.0, seed=0):
        r = np.random.default_rng(seed)
        img = (np.full((360, 480, 3), 120) + r.normal(0, 8, (360, 480, 3))).clip(0, 255).astype(np.uint8)
        p, m = _part("braket", rng)
        return img, _paste(img, None, p, m, 240, 180, angle, scale)

    img0, box0 = make(0)
    model = fd.DetModel.train([img0], [[(*box0, 0)]], ["braket"])
    for angle in (-25, 15, 40, 120):
        img, gt = make(angle, 1.1, seed=angle + 100)
        dets = model.detect(img)
        assert len(dets) == 1, (angle, dets)
        d = dets[0]
        assert abs(((d["angle"] - angle) + 180) % 360 - 180) < 6, (angle, d["angle"])
        assert fd.box_iou(d["box"], gt) > 0.7
        assert abs(d["center"][0] - 240) < 4 and abs(d["center"][1] - 180) < 4


def test_save_load_identical(one, tmp_path):
    model, _ = one
    p = model.save(tmp_path / "parca")
    assert p.endswith(".vfdet")
    again = fd.DetModel.load(p)
    assert again.classes == model.classes and again.trained_classes == model.trained_classes
    for s in (500, 501, 502):
        img, _ = scene(s)
        a, b = model.detect(img), again.detect(img)
        assert [(d["box"], d["score"], d["angle"]) for d in a] == [(d["box"], d["score"], d["angle"]) for d in b]
    # bad files: Turkish messages
    with pytest.raises(ValueError, match="Model dosyası yok"):
        fd.DetModel.load(tmp_path / "yok.vfdet")
    junk = tmp_path / "junk.vfdet"
    junk.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="okunamadı"):
        fd.DetModel.load(junk)
    other = tmp_path / "other.vfdet"
    np.savez(other.open("wb"), header=np.frombuffer(json.dumps({"format": "x"}).encode(), np.uint8))
    with pytest.raises(ValueError, match="modeli değil"):
        fd.DetModel.load(other)


def test_training_errors_are_turkish():
    img, gt = scene(0)
    with pytest.raises(ValueError, match="Hiç kutu çizilmedi"):
        fd.DetModel.train([img], [[]], ["braket"])
    with pytest.raises(ValueError, match="önce resim ekleyin"):
        fd.DetModel.train([], [], ["braket"])
    with pytest.raises(ValueError, match="çok küçük"):
        fd.DetModel.train([img], [[(10, 10, 3, 40, 0)]], ["braket"])
    with pytest.raises(ValueError, match="Geçersiz sınıf"):
        fd.DetModel.train([img], [[(*gt[0][:4], 5)]], ["braket"])
    with pytest.raises(ValueError, match="Ölçek aralığı"):
        fd.DetModel.train([img], [gt], ["braket"], scale_range=(1.2, 1.5))
    with pytest.raises(ValueError, match="bağlı değil"):
        fd.DetModel.train([None], [gt], ["braket"])
    with pytest.raises(ValueError, match="eğitilmemiş"):
        fd.DetModel(["a"]).detect(img)


def test_any_picture_kind(one):
    model, _ = one
    img, gt = scene(120)
    ref = [d["box"] for d in model.detect(img)]
    assert len(ref) >= len(gt) - 1
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    assert [d["box"] for d in model.detect(bgra)] == ref
    u16 = img.astype(np.uint16) * 257
    assert [d["box"] for d in model.detect(u16)] == ref
    f32 = img.astype(np.float32) / 255.0
    assert len(model.detect(f32)) == len(ref)
    # tiny, flat and empty pictures
    assert model.detect(np.zeros((5, 7, 3), np.uint8)) == []
    assert model.detect(np.full((480, 640), 128, np.uint8)) == []
    with pytest.raises(ValueError, match="boş"):
        model.detect(np.zeros((0, 0, 3), np.uint8))
    with pytest.raises(ValueError, match="bağlı değil"):
        model.detect(None)


def test_grayscale_training():
    data = [scene(s) for s in range(4)]
    gray = [cv2.cvtColor(d[0], cv2.COLOR_BGR2GRAY) for d in data]
    model = fd.DetModel.train(gray, [d[1] for d in data], ["braket"], angle_range=45)
    tp = fp = fn = 0
    for s in range(600, 608):
        img, gt = scene(s)
        a, b, c = evaluate(model.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), gt)
        tp, fp, fn = tp + a, fp + b, fn + c
    assert tp / (tp + fn) >= 0.9 and tp / max(1, tp + fp) >= 0.9


def test_speed_640x480(one, two):
    model, _ = one
    img, _ = scene(700)
    model.detect(img)                                   # first call builds the rotated templates
    times = []
    for s in range(700, 706):
        img, _ = scene(s)
        t = time.perf_counter()
        model.detect(img)
        times.append(time.perf_counter() - t)
    # about 60–100 ms on a 4-core CPU; loose bound for busy test machines
    assert float(np.median(times)) < 0.6, times


# --- the block --------------------------------------------------------------------------------------------------------

def test_block_in_flow_with_detection_blocks(one, tmp_path):
    model, _ = one
    path = model.save(tmp_path / "m.vfdet")
    img, gt = scene(130)
    cv2.imwrite(str(tmp_path / "t.png"), img)
    g = Graph()
    src = g.add_new("ImageFile", path=str(tmp_path / "t.png"))
    det = g.add_new("FewShotDetect", model=path)
    draw = g.add_new("DrawDetections")
    filt = g.add_new("FilterDetections", labels="braket")
    g.connect(src.id, "image", det.id, "image")
    g.connect(src.id, "image", draw.id, "image")
    g.connect(det.id, "detections", draw.id, "detections")
    g.connect(det.id, "detections", filt.id, "detections")
    out = {}
    g.run(on_frame=lambda n, r: out.update(r))
    assert out[det.id].error is None, out[det.id].error
    assert out[det.id].outputs["count"] == len(gt)
    assert out[filt.id].outputs["count"] == len(gt)
    assert out[draw.id].outputs["image"].shape == img.shape
    assert out[det.id].outputs["image"].shape == img.shape
    from visionflow.nodes.fewshot import last_input
    assert last_input(det.id) is not None


def test_block_settings_and_errors(two, tmp_path):
    node = create("FewShotDetect")
    node.setup()
    img, gt = scene(310, TWO)
    with pytest.raises(ValueError, match="Model dosyası seçilmedi"):
        node.process({"image": img})
    node.values["model"] = str(tmp_path / "yok.vfdet")
    with pytest.raises(ValueError, match="Model dosyası yok"):
        node.process({"image": img})
    node.values["model"] = two.save(tmp_path / "iki")
    out = node.process({"image": img})
    assert out["count"] == len(out["detections"]) >= len(gt) - 1
    node.values["classes"] = "braket"
    assert all(d["label"] == "braket" for d in node.process({"image": img})["detections"])
    node.values["classes"] = ""
    node.values["max_objects"] = 1
    assert node.process({"image": img})["count"] == 1
    node.values["max_objects"] = 50
    node.values["score"] = 0.99999
    assert node.process({"image": img})["count"] <= out["count"]
    with pytest.raises(ValueError, match="bağlı değil"):
        node.process({"image": None})
    # counting and tracking blocks accept the detections
    cnt = create("CountByLabel")
    cnt.values["labels"] = "braket,etiket"
    res = cnt.process({"detections": out["detections"]})
    assert res["a"] + res["b"] == out["count"]
    trk = create("ObjectTracker")
    trk.setup()
    r1 = trk.process({"detections": out["detections"]})
    r2 = trk.process({"detections": out["detections"]})
    assert r1["count"] == r2["count"] == out["count"] and r2["total"] == out["count"]


def test_block_reloads_saved_model(one, tmp_path):
    model, _ = one
    p = model.save(tmp_path / "m.vfdet")
    node = create("FewShotDetect")
    node.values["model"] = p
    node.setup()
    img, _ = scene(140)
    node.process({"image": img})
    first = node.state["model"]
    time.sleep(0.01)
    model.save(p)
    os.utime(p, ns=(time.time_ns(), time.time_ns() + 5_000_000))
    node.process({"image": img})
    assert node.state["model"] is not first


# --- example ----------------------------------------------------------------------------------------------------------

def test_example_flow_runs(monkeypatch):
    monkeypatch.chdir(ROOT)
    g = Graph.from_dict(json.loads((ROOT / "examples" / "az_ornekle_nesne_bulma.json").read_text("utf-8")))
    out = {}
    g.run(on_frame=lambda n, r: out.update(r))
    det = next(n for n in g.nodes.values() if n.type_name == "FewShotDetect")
    assert out[det.id].error is None, out[det.id].error
    assert out[det.id].outputs["count"] == 4
    labels = sorted(d["label"] for d in out[det.id].outputs["detections"])
    assert labels == ["braket", "braket", "etiket", "etiket"]
    folder = ROOT / "examples" / "ornek_veri" / "az_ornek_tespit"
    assert sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) < 300_000


# --- the window -------------------------------------------------------------------------------------------------------

@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_dialog_boxes_train_save_apply(app, tmp_path):
    from visionflow.ui.det_teach_dialog import DetTeachDialog
    node = create("FewShotDetect")
    applied = {}
    dlg = DetTeachDialog(None, node, lambda k, v: applied.__setitem__(k, v))
    dlg.show()
    with pytest.raises(ValueError, match="önce resim ekleyin"):
        dlg.train()
    with pytest.raises(ValueError, match="Önce bir resim"):
        dlg.add_box(1, 1, 20, 20)
    data = [scene(s, TWO) for s in range(5)]
    for s, (img, gt) in enumerate(data):
        assert dlg.add_image(img, f"r{s}") == s and dlg.current == s
    with pytest.raises(ValueError, match="Hiç kutu çizilmedi"):
        dlg.train()
    assert dlg.add_class("etiket") == 1
    dlg.rename_class(0, "braket")
    with pytest.raises(ValueError, match="zaten var"):
        dlg.add_class("Etiket")
    with pytest.raises(ValueError, match="virgül"):
        dlg.add_class("a,b")
    for s, (img, gt) in enumerate(data):
        for x, y, w, h, c in gt:
            dlg.add_box(x, y, w, h, cls=c, index=s)
    # editing: move, resize, delete, undo
    dlg.set_current(0)
    b0 = list(dlg.images[0]["boxes"][0])
    dlg.move_box(0, 15, -5)
    assert dlg.images[0]["boxes"][0][:2] == [b0[0] + 15, max(0, b0[1] - 5)]
    dlg.set_box(0, 0, 0, 30, 30)
    dlg.delete_box(0)
    assert len(dlg.images[0]["boxes"]) == len(data[0][1]) - 1
    assert dlg.undo() and dlg.undo() and dlg.undo()
    assert dlg.images[0]["boxes"][0] == b0
    with pytest.raises(ValueError, match="çok küçük"):
        dlg.add_box(5, 5, 3, 3)
    with pytest.raises(ValueError, match="Geçersiz sınıf"):
        dlg.add_box(5, 5, 30, 30, cls=7)
    # train → live finds on the pictures
    model = dlg.train()
    assert model.trained_classes == [0, 1]
    dets = dlg.detections(0)
    assert dets is not None and len(dets) >= len(data[0][1]) - 1
    # unseen picture: add it, accept the finds as boxes (the user then fixes mistakes), retrain
    img, gt = scene(900, TWO)
    k = dlg.add_image(img, "yeni")
    found = dlg.detections(k)
    tp, fp, fn = evaluate(found, gt)
    assert tp >= len(gt) - 1
    assert dlg.accept_detections() == len(found)
    assert dlg.accept_detections() == 0                    # already there: not added twice
    assert len(dlg.images[k]["boxes"]) == len(found)
    dlg.undo()
    assert dlg.images[k]["boxes"] == []
    dlg.render()
    # save → the block's setting is set and the block finds the parts
    saved = dlg.save_model(str(tmp_path / "dlg"))
    assert applied["model"] == saved and saved.endswith(".vfdet")
    assert fd.DetModel.load(saved).classes == ["braket", "etiket"]
    node.values["model"] = saved
    node.setup()
    out = node.process({"image": img})
    a, b, c = evaluate(out["detections"], gt)
    assert a >= len(gt) - 1 and b <= 1
    # pictures of any kind; the flow's last picture; removing pictures
    dlg.add_image((img.astype(np.uint16) * 257)[:, :, 0], "16bit")
    assert dlg.add_from_flow()
    n = len(dlg.images)
    dlg.remove_current()
    assert len(dlg.images) == n - 1 and dlg.current == n - 2
    # a second window on the same block starts from the saved model's classes and settings
    dlg2 = DetTeachDialog(None, node)
    assert [c[0] for c in dlg2.classes] == ["braket", "etiket"] and dlg2.model is not None
    dlg2.close()
    dlg.close()


def test_dialog_files_and_folder(app, tmp_path):
    from visionflow.ui.det_teach_dialog import DetTeachDialog
    dlg = DetTeachDialog(None, None)
    with pytest.raises(ValueError, match="Klasör yok"):
        dlg.add_folder(tmp_path / "yok")
    with pytest.raises(ValueError, match="resim yok"):
        dlg.add_folder(tmp_path)
    cv2.imwrite(str(tmp_path / "a.png"), scene(0)[0])
    (tmp_path / "b.png").write_bytes(b"bozuk")
    assert dlg.add_folder(tmp_path) == 1
    assert "b.png" in dlg.status.text()
    assert not dlg.add_from_flow()                         # no block: nothing to take
    with pytest.raises(ValueError, match="önce 'Eğit'"):
        dlg.save_model(str(tmp_path / "x"))
    dlg.close()


def test_dialog_mouse(app):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from visionflow.ui.det_teach_dialog import DetTeachDialog
    dlg = DetTeachDialog(None, None)
    dlg.resize(1100, 700)
    dlg.show()
    dlg.add_image(scene(0)[0])
    c = dlg.canvas
    mid = QPoint(c.width() // 2, c.height() // 2)
    # drag on empty space → a new box
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, mid)
    QTest.mouseMove(c, mid + QPoint(60, 40))
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, mid + QPoint(60, 40))
    boxes = dlg.images[0]["boxes"]
    assert len(boxes) == 1 and boxes[0][2] > 20 and boxes[0][3] > 15
    before = list(boxes[0])
    # drag inside it → moved
    inside = mid + QPoint(30, 20)
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, inside)
    QTest.mouseMove(c, inside + QPoint(25, 0))
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, inside + QPoint(25, 0))
    after = dlg.images[0]["boxes"][0]
    assert after[0] > before[0] and after[2:4] == before[2:4]
    # a plain click on empty space does not make a box; right click on the box deletes it
    QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, QPoint(5, 5))
    assert len(dlg.images[0]["boxes"]) == 1
    sp = c.to_screen(after[0] + after[2] / 2, after[1] + after[3] / 2)
    QTest.mouseClick(c, Qt.RightButton, Qt.NoModifier, QPoint(int(sp.x()), int(sp.y())))
    assert dlg.images[0]["boxes"] == []
    dlg.close()


def test_main_window_opens_block_dialog(app):
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QPushButton
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    try:
        item = w.scene.add_node("FewShotDetect", QPointF(0, 0))
        node = item.node if hasattr(item, "node") else item
        w.props.show_node(node)
        btn = w.props.findChild(QPushButton, "block_dialog")
        assert btn is not None and "Eğit" in btn.text()
        btn.click()
        dlg = w._block_dialogs[node.id]
        assert type(dlg).__name__ == "DetTeachDialog" and dlg.isVisible() and dlg.node is node
        assert w.open_block_dialog(node.id) is dlg
        dlg.close()
    finally:
        w.close()
