"""2.1: Az Örnekle Segmentasyon — few-shot segmentation engine, block, painting/training window and example."""
import json
import os
import pathlib

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow import fewshot_seg as fs  # noqa: E402
from visionflow.core import Graph, create, load_builtin  # noqa: E402

load_builtin()
ROOT = pathlib.Path(__file__).resolve().parents[1]


# --- synthetic data -------------------------------------------------------------------------------------------------

def synth(seed, h=240, w=320, kinds=("striped",)):
    """Noisy grey background with textured shapes. Returns (image, ground truth class map) where class k + 1 is
    ``kinds[k]``: "striped" = orange striped, "blue" = flat bluish."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.normal(0, 18, (h, w, 3)) + (120, 110, 100)).clip(0, 255).astype(np.uint8), (0, 0), 1)
    gt = np.zeros((h, w), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    stripes = ((np.sin(xx * 0.8 + yy * 0.3) > 0) * 60).astype(np.int16)
    tex = {"striped": (np.stack([stripes] * 3, -1) + (40, 140, 170) + rng.normal(0, 10, (h, w, 3))),
           "blue": (np.zeros((h, w, 3)) + (200, 90, 40) + rng.normal(0, 8, (h, w, 3)))}
    for _ in range(int(rng.integers(3, 6))):
        k = int(rng.integers(len(kinds)))
        cx, cy, r = int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40)), int(rng.integers(18, 36))
        m = np.zeros((h, w), np.uint8)
        if rng.random() < .5:
            cv2.circle(m, (cx, cy), r, 1, -1)
        else:
            cv2.rectangle(m, (cx - r, cy - r // 2), (cx + r, cy + r // 2), 1, -1)
        gt[m > 0] = k + 1
    for k, kind in enumerate(kinds):
        t = tex[kind].clip(0, 255).astype(np.uint8)
        img[gt == k + 1] = t[gt == k + 1]
    return img, gt


def strokes(gt, seed, n=6, radius=4):
    """A user's few brush dabs: ``n`` per class, inside each class (a little away from the edges)."""
    rng = np.random.default_rng(seed)
    lab = np.zeros(gt.shape, np.uint8)
    for c in np.unique(gt):
        inner = cv2.erode((gt == c).astype(np.uint8), np.ones((9, 9) if c else (7, 7), np.uint8))
        ys, xs = np.nonzero(inner)
        if not len(xs):
            continue
        dab = np.zeros(gt.shape, np.uint8)
        for _ in range(n):
            i = rng.integers(len(xs))
            cv2.circle(dab, (int(xs[i]), int(ys[i])), radius, 1, -1)
        lab[(dab > 0) & (inner > 0)] = c + 1
    return lab


@pytest.fixture(scope="module")
def trained():
    data = [synth(s) for s in range(3)]
    model = fs.SegModel.train([d[0] for d in data], [strokes(d[1], s) for s, d in enumerate(data)],
                              ["arka plan", "parça"])
    return model


# --- engine ---------------------------------------------------------------------------------------------------------

def test_few_strokes_segment_unseen_images(trained):
    assert trained.info["images"] == 3 and trained.info["train_accuracy"] > 0.95
    ious = []
    for s in range(100, 106):
        img, gt = synth(s)
        pred = trained.predict(img, smooth=1, min_area=30)
        assert pred.shape == gt.shape and pred.dtype == np.uint8
        ious.append(fs.iou(pred == 1, gt == 1))
    assert min(ious) > 0.85, ious


def test_save_load_round_trip_identical(trained, tmp_path):
    path = trained.save(tmp_path / "m")                   # extension added
    assert path.endswith(".vfseg") and pathlib.Path(path).stat().st_size < 200_000
    back = fs.SegModel.load(path)
    assert back.classes == ["arka plan", "parça"] and back.colors == trained.colors
    for s in (200, 201):
        img, _ = synth(s)
        assert np.array_equal(back.predict(img, smooth=2, min_area=40), trained.predict(img, smooth=2, min_area=40))


def test_multi_class():
    data = [synth(s, kinds=("striped", "blue")) for s in range(4)]
    model = fs.SegModel.train([d[0] for d in data], [strokes(d[1], s) for s, d in enumerate(data)],
                              ["arka plan", "çizgili", "mavi"])
    for s in (300, 301, 302):
        img, gt = synth(s, kinds=("striped", "blue"))
        pred = model.predict(img, smooth=1, min_area=30)
        for c in (1, 2):
            if (gt == c).sum() > 500:
                assert fs.iou(pred == c, gt == c) > 0.8, (s, c)
        areas = fs.class_areas(pred, 3)
        assert abs(sum(areas) - 100) < 0.01


def test_image_kinds_and_scales(trained):
    img, gt = synth(150)
    base = trained.predict(img)
    gray16 = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257)
    assert trained.predict(gray16).shape == gt.shape
    assert np.array_equal(trained.predict(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)), base)
    assert np.array_equal(trained.predict(img.astype(np.float32) / 255.0), base)
    assert np.array_equal(trained.predict((img.astype(np.uint16) * 257)), base)
    big = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_NEAREST)        # works on a smaller copy
    big_gt = cv2.resize(gt, (1280, 960), interpolation=cv2.INTER_NEAREST)
    assert fs.iou(trained.predict(big, smooth=1) == 1, big_gt == 1) > 0.8
    tiny = trained.predict(np.zeros((1, 1, 3), np.uint8))
    assert tiny.shape == (1, 1)


def test_turkish_errors(tmp_path):
    img, gt = synth(1)
    with pytest.raises(ValueError, match="hiç resim yok"):
        fs.SegModel.train([], [], ["arka plan", "nesne"])
    with pytest.raises(ValueError, match="Hiç boyanmış piksel yok"):
        fs.SegModel.train([img], [np.zeros(gt.shape, np.uint8)], ["arka plan", "nesne"])
    with pytest.raises(ValueError, match="en az iki sınıfı"):
        fs.SegModel.train([img], [(gt == 1).astype(np.uint8) * 2], ["arka plan", "nesne"])
    with pytest.raises(ValueError, match="En az iki sınıf"):
        fs.SegModel.train([img], [strokes(gt, 0)], ["tek"])
    with pytest.raises(ValueError, match="boyutu resimle aynı"):
        fs.SegModel.train([img], [np.ones((10, 10), np.uint8)], ["arka plan", "nesne"])
    with pytest.raises(ValueError, match="tanımsız sınıf"):
        fs.SegModel.train([img], [np.full(gt.shape, 5, np.uint8)], ["arka plan", "nesne"])
    with pytest.raises(ValueError, match="bağlı değil"):
        fs.SegModel.train([None], [strokes(gt, 0)], ["arka plan", "nesne"])
    with pytest.raises(ValueError, match="Model dosyası yok"):
        fs.SegModel.load(tmp_path / "yok.vfseg")
    bad = tmp_path / "bozuk.vfseg"
    bad.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="okunamadı"):
        fs.SegModel.load(bad)
    other = tmp_path / "baska.vfseg"
    with open(other, "wb") as fh:
        np.savez(fh, header=np.frombuffer(json.dumps({"format": "x"}).encode(), np.uint8))
    with pytest.raises(ValueError, match="değil"):
        fs.SegModel.load(other)
    with pytest.raises(ValueError, match="eğitilmemiş"):
        fs.SegModel(["a", "b"]).predict(img)
    with pytest.raises(ValueError):
        fs.grabcut_rect(img, (0, 0, 2, 2))


def test_remove_small_and_overlay():
    lab = np.zeros((50, 50), np.uint8)
    lab[5:8, 5:8] = 1          # 9 px
    lab[20:40, 20:40] = 1      # 400 px
    out = fs.remove_small(lab, 50)
    assert out[6, 6] == 0 and out[30, 30] == 1
    ov = fs.overlay(np.full((50, 50), 100, np.uint8), out, [(0, 0, 0), (0, 0, 255)], 0.5)
    assert ov.shape == (50, 50, 3) and tuple(ov[0, 0]) == (100, 100, 100) and ov[30, 30, 2] > 150


def test_grabcut_rect_finds_object():
    img = np.full((120, 160, 3), 30, np.uint8)
    cv2.circle(img, (80, 60), 25, (40, 200, 230), -1)
    m = fs.grabcut_rect(img, (45, 25, 70, 70))
    gt = np.zeros(m.shape, np.uint8)
    cv2.circle(gt, (80, 60), 25, 1, -1)
    assert fs.iou(m, gt > 0) > 0.85


# --- the block ------------------------------------------------------------------------------------------------------

def test_block(trained, tmp_path):
    path = trained.save(tmp_path / "parca.vfseg")
    node = create("FewShotSegment")
    assert node.title == "Az Örnekle Segmentasyon" and node.dialog
    node.values.update(model=path, min_area=30)
    node.setup()
    img, gt = synth(120)
    out = node.process({"image": img})
    assert out["mask"].dtype == np.uint8 and set(np.unique(out["mask"])) <= {0, 255}
    assert fs.iou(out["mask"] > 0, gt == 1) > 0.85
    n_gt = cv2.connectedComponentsWithStats((gt == 1).astype(np.uint8))[0] - 1
    assert out["count"] == len(out["contours"]) == len(out["detections"])
    assert abs(out["count"] - n_gt) <= 1
    assert out["overlay"].shape == img.shape
    assert abs(out["area"] - 100 * (gt == 1).mean()) < 3
    assert set(out["areas"]) == {"arka plan", "parça"}
    # class numbers, choosing the class by name or number, grayscale 16-bit input
    node.values.update(mask_mode=node.param("mask_mode").choices[1], target="arka plan")
    out2 = node.process({"image": (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257)})
    assert out2["mask"].max() <= 1
    node.values["target"] = "1"
    assert node.process({"image": img})["area"] == pytest.approx(out["area"])
    from visionflow.nodes.fewshot import last_input
    assert last_input(node.id).shape == img.shape


def test_block_errors_and_reload(trained, tmp_path):
    node = create("FewShotSegment")
    node.setup()
    img, _ = synth(5)
    with pytest.raises(ValueError, match="Model dosyası seçilmedi"):
        node.process({"image": img})
    node.values["model"] = str(tmp_path / "yok.vfseg")
    with pytest.raises(ValueError, match="Model dosyası yok"):
        node.process({"image": img})
    path = trained.save(tmp_path / "m.vfseg")
    node.values["model"] = path
    with pytest.raises(ValueError, match="bağlı değil"):
        node.process({"image": None})
    with pytest.raises(ValueError, match="resim bağlanmalı"):
        node.process({"image": "metin"})
    node.values["target"] = "kedi"
    with pytest.raises(ValueError, match="modelde yok"):
        node.process({"image": img})
    node.values["target"] = ""
    node.process({"image": img})
    # the model is saved again (e.g. renamed class from the window): the block notices without a restart
    m2 = fs.SegModel.load(path)
    m2.classes[1] = "yeni ad"
    st = os.stat(path)
    m2.save(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert "yeni ad" in node.process({"image": img})["areas"]


def test_example_flow_runs():
    path = ROOT / "examples" / "az_ornekle_segmentasyon.json"
    g = Graph.load(str(path))
    assert not g.warnings
    assert not g.setup()
    res = g.run_once()
    assert not [r.error for r in res.values() if r.error]
    seg = next(n for n in g.nodes.values() if n.type_name == "FewShotSegment")
    assert res[seg.id].outputs["count"] >= 2
    for f in (ROOT / "examples" / "ornek_veri" / "az_ornek").iterdir():
        assert f.stat().st_size < 200_000


# --- the window -----------------------------------------------------------------------------------------------------

@pytest.fixture
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_dialog_paint_train_save(app, tmp_path):
    from visionflow.ui.seg_teach_dialog import SegTeachDialog
    node = create("FewShotSegment")
    applied = {}
    dlg = SegTeachDialog(None, node, lambda k, v: applied.__setitem__(k, v))
    dlg.show()
    with pytest.raises(ValueError, match="önce resim ekleyin"):
        dlg.train()
    data = [synth(s) for s in range(3)]
    for s, (img, gt) in enumerate(data):
        i = dlg.add_image(img, f"r{s}")
        assert dlg.current == i
        lab = strokes(gt, s)
        for c in (0, 1):                                    # paint each dab as a one-point stroke
            ys, xs = np.nonzero(lab == c + 1)
            for k in range(0, len(xs), 7):
                dlg.paint_stroke([(xs[k], ys[k])], cls=c, radius=2)
    with pytest.raises(ValueError, match="Hiç boyanmış|en az iki"):
        dlg2 = SegTeachDialog(None, None)
        dlg2.add_image(data[0][0])
        dlg2.train()
    model = dlg.train()
    pred = dlg.prediction()
    assert pred is not None and pred.shape == data[-1][1].shape
    img, gt = synth(400)
    assert fs.iou(model.predict(img, smooth=1, min_area=30) == 1, gt == 1) > 0.85
    # undo removes the last stroke only
    before = dlg.images[dlg.current]["labels"].copy()
    dlg.paint_stroke([(10, 10), (60, 10)], cls=1, radius=3)
    assert not np.array_equal(before, dlg.images[dlg.current]["labels"])
    assert dlg.undo()
    assert np.array_equal(before, dlg.images[dlg.current]["labels"])
    dlg.paint_stroke([(10, 10)], erase=True, radius=3)
    # classes
    with pytest.raises(ValueError, match="zaten var"):
        dlg.add_class("NESNE")
    dlg.rename_class(1, "Nesne")                           # only the case changes: allowed
    dlg.rename_class(1, "parça")
    assert dlg.add_class("leke") == 2
    with pytest.raises(ValueError, match="boş"):
        dlg.rename_class(2, "  ")
    # save → the block's setting is set and the block uses it
    saved = dlg.save_model(str(tmp_path / "dlg"))
    assert applied["model"] == saved and saved.endswith(".vfseg")
    assert fs.SegModel.load(saved).classes == ["arka plan", "parça"]
    node.values["model"] = saved
    node.setup()
    assert node.process({"image": img})["count"] >= 1
    # a picture of any kind; the flow's last picture; removing pictures
    dlg.add_image((img.astype(np.uint16) * 257)[:, :, 0], "16bit")
    from visionflow.nodes.fewshot import last_input
    assert last_input(node.id) is not None and dlg.add_from_flow()
    n = len(dlg.images)
    dlg.remove_current()
    assert len(dlg.images) == n - 1 and dlg.current == n - 2
    # prediction → labels, GrabCut, clear
    dlg.set_current(0)
    dlg.prediction_to_labels()
    assert dlg.images[0]["labels"].min() >= 1
    dlg.clear_current()
    assert not dlg.images[0]["labels"].any()
    gimg = np.full((120, 160, 3), 30, np.uint8)
    cv2.circle(gimg, (80, 60), 25, (40, 200, 230), -1)
    gi = dlg.add_image(gimg, "grabcut")
    with pytest.raises(ValueError, match="arka plan dışında"):
        dlg.grabcut((45, 25, 70, 70), cls=0)
    assert dlg.grabcut((45, 25, 70, 70), cls=1) > 1000
    assert (dlg.images[gi]["labels"] == 2).sum() > 1000 and (dlg.images[gi]["labels"] == 1).any()
    dlg.render()
    dlg.close()


def test_dialog_mouse_painting(app):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from visionflow.ui.seg_teach_dialog import SegTeachDialog
    dlg = SegTeachDialog(None, None)
    dlg.resize(1000, 700)
    dlg.show()
    dlg.add_image(synth(0)[0])
    dlg.set_class(1)
    c = dlg.canvas
    mid = QPoint(c.width() // 2, c.height() // 2)
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, mid)
    QTest.mouseMove(c, mid + QPoint(40, 0))
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, mid + QPoint(40, 0))
    lab = dlg.images[0]["labels"]
    assert (lab == 2).sum() > 50
    x, y = c.to_image(mid)
    assert lab[int(y), int(x)] == 2
    dlg.set_tool("erase")
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, mid)
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, mid)
    assert lab is dlg.images[0]["labels"] and dlg.images[0]["labels"][int(y), int(x)] == 0
    dlg.close()


def test_main_window_opens_block_dialog(app, tmp_path):
    from visionflow.ui.main_window import MainWindow
    from PySide6.QtWidgets import QPushButton
    w = MainWindow()
    w._confirm_discard = lambda: True
    try:
        from PySide6.QtCore import QPointF
        item = w.scene.add_node("FewShotSegment", QPointF(0, 0))
        node = item.node if hasattr(item, "node") else item
        w.props.show_node(node)
        btn = w.props.findChild(QPushButton, "block_dialog")
        assert btn is not None and "Eğit" in btn.text()
        btn.click()
        dlg = w._block_dialogs[node.id]
        assert dlg.isVisible() and dlg.node is node
        assert w.open_block_dialog(node.id) is dlg               # one window per block
        img, gt = synth(0)
        dlg.add_image(img)
        dlg.paint_stroke([(5, 5), (40, 5)], cls=0, radius=3)
        lab = strokes(gt, 0)
        ys, xs = np.nonzero(lab == 2)
        dlg.paint_stroke([(xs[0], ys[0])], cls=1, radius=2)
        dlg.train()
        saved = dlg.save_model(str(tmp_path / "w.vfseg"))
        assert node.values["model"] == saved                  # set like typing it in
        dlg.close()
    finally:
        w.stop()
        w.close()
