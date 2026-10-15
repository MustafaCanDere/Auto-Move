"""1.8: camera calibration, mm measuring (scale, bird's-eye plane, distance) and comment frames in the editor."""
import json
import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow import calibration as cal  # noqa: E402
from visionflow.core import Graph, create, load_builtin  # noqa: E402
from visionflow.core.registry import NODE_REGISTRY  # noqa: E402

load_builtin()

K_TRUE = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
D_TRUE = np.array([-0.25, 0.08, 0, 0, 0])
SQ_MM = 25.0
BOARD = cal.chessboard(9, 6, 60, 40)      # first inner corner of the texture at (100, 100), 60 px per square


def render(rvec, tvec, K=K_TRUE, D=D_TRUE, size=(640, 480), extra=None):
    """Photograph the board plane with a pinhole camera that has lens distortion. `extra(texture)` may draw on it."""
    w, h = size
    tex = cv2.cvtColor(BOARD, cv2.COLOR_GRAY2BGR)
    if extra:
        tex = np.pad(tex, ((0, 400), (0, 400), (0, 0)), constant_values=255)
        extra(tex)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = cv2.undistortPoints(np.stack([xs.ravel(), ys.ravel()], 1)[:, None], K, D).reshape(-1, 2)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    rays = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ R
    o = -R.T @ np.asarray(tvec, float)
    s = -o[2] / rays[:, 2]
    P = o[None] + s[:, None] * rays
    u = (P[:, 0] / SQ_MM) * 60 + 100
    v = (P[:, 1] / SQ_MM) * 60 + 100
    return cv2.remap(tex, u.reshape(h, w).astype(np.float32), v.reshape(h, w).astype(np.float32), cv2.INTER_LINEAR,
                     borderValue=(255, 255, 255))


def views(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return {f"{i}.png": render([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2)],
                               [-100 + rng.uniform(-30, 30), -60 + rng.uniform(-30, 30), rng.uniform(450, 650)])
            for i in range(n)}


@pytest.fixture(scope="module")
def calib():
    return cal.calibrate(views(), 9, 6, SQ_MM)


def test_blocks_registered():
    for t in ("LensUndistort", "ScaleFromReference", "PlaneRectify", "DistanceMeasure"):
        assert t in NODE_REGISTRY
    assert len(NODE_REGISTRY) >= 134


def test_calibration_recovers_the_camera(calib, tmp_path):
    assert calib.rms < 0.5 and calib.images_used == 12
    assert calib.K[0, 0] == pytest.approx(800, rel=0.02) and calib.K[1, 2] == pytest.approx(240, abs=8)
    assert calib.D[0] == pytest.approx(-0.25, abs=0.04)
    path = tmp_path / "k.json"
    calib.save(path)
    back = cal.Calibration.load(path)
    assert back.rms == calib.rms and np.allclose(back.K, calib.K) and back.board == [9, 6, 25.0]
    assert set(back.per_image) == {f"{i}.png" for i in range(12)}


def test_calibration_errors(tmp_path):
    with pytest.raises(ValueError, match="en az 3"):
        cal.calibrate({"a.png": np.full((480, 640, 3), 255, np.uint8)}, 9, 6, 25)
    imgs = views(4)
    imgs["b.png"] = cv2.resize(imgs["0.png"], (320, 240))              # another resolution: left out
    imgs["c.png"] = np.full((480, 640, 3), 255, np.uint8)               # no board: left out
    c = cal.calibrate(imgs, 9, 6, 25)
    assert c.images_used == 4 and c.skipped == ["b.png", "c.png"]
    with pytest.raises(FileNotFoundError, match="Kamera Kalibrasyonu"):
        cal.Calibration.load(tmp_path / "yok.json")
    (tmp_path / "bozuk.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="okunamadı"):
        cal.Calibration.load(tmp_path / "bozuk.json")


def straightness(img):
    """How far the board's corner rows are from straight lines (pixels): lens bending makes them curve."""
    c = cal.find_board(img, 9, 6)
    err = 0.0
    for row in c.reshape(6, 9, 2):
        A = np.c_[row[:, 0], np.ones(9)]
        coef, *_ = np.linalg.lstsq(A, row[:, 1], rcond=None)
        err = max(err, float(np.abs(A @ coef - row[:, 1]).max()))
    return err


def test_undistort_straightens_lines(calib, tmp_path):
    path = tmp_path / "k.json"
    calib.save(path)
    img = render([0.05, 0.05, 0.0], [-100, -60, 330])                  # board filling the picture: strong bending
    n = create("LensUndistort")
    n.values.update(file=str(path))
    assert n.safe_setup(), n.setup_error
    out = n.process({"image": img})["image"]
    assert out.shape == img.shape
    assert straightness(out) < straightness(img) / 3
    half = n.process({"image": cv2.resize(img, (320, 240))})["image"]   # the same camera at half resolution
    assert half.shape == (240, 320, 3) and straightness(half) < straightness(cv2.resize(img, (320, 240))) * 0.75


def test_undistort_uses_default_file(monkeypatch, tmp_path, calib):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path))
    n = create("LensUndistort")
    assert not n.safe_setup() and "Kamera Kalibrasyonu" in n.setup_error
    calib.save(cal.default_path())
    assert n.safe_setup()


def test_scale_from_chessboard_and_lock():
    img = render([0, 0, 0], [-100, -60, 500], D=np.zeros(5))         # straight above: 800/500 = 1.6 px per mm
    n = create("ScaleFromReference")
    assert n.safe_setup()
    out = n.process({"image": img})
    assert out["found"] and out["px_per_mm"] == pytest.approx(1.6, rel=0.01)
    blank = np.full_like(img, 255)
    assert n.process({"image": blank})["px_per_mm"] == out["px_per_mm"]          # locked: the board may go
    n.values["lock"] = False
    n.state["found"] = None
    out = n.process({"image": blank})
    assert not out["found"] and out["px_per_mm"] == 0.0


def test_scale_from_aruco():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 7, 200)
    img = np.full((400, 400), 255, np.uint8)
    img[100:300, 100:300] = marker
    n = create("ScaleFromReference")
    n.values.update(method="ArUco işaretçi", square_mm=50.0, marker_id=7)
    assert n.safe_setup()
    assert n.process({"image": img})["px_per_mm"] == pytest.approx(4.0, rel=0.02)   # 200 px / 50 mm
    n.values.update(marker_id=3, lock=False)
    n.state["found"] = None
    assert not n.process({"image": img})["found"]


def test_plane_rectify_measures_at_an_angle():
    def part(tex):                                   # a 100 x 50 mm plate next to the board on the same table
        x0, y0 = 100 + int(250 / SQ_MM * 60), 100 + int(40 / SQ_MM * 60)
        cv2.rectangle(tex, (x0, y0), (x0 + int(100 / SQ_MM * 60), y0 + int(50 / SQ_MM * 60)), (0, 0, 200), -1)
    img = render([0.45, -0.3, 0.1], [-150, -40, 700], D=np.zeros(5), extra=part)   # camera looks at an angle
    n = create("PlaneRectify")
    n.values.update(out_ppm=4.0, width_mm=450, height_mm=250, margin_mm=20)
    assert n.safe_setup()
    out = n.process({"image": img})
    assert out["found"] and out["px_per_mm"] == 4.0 and out["image"].shape[:2] == (1000, 1800)
    red = cv2.inRange(out["image"], (0, 0, 120), (90, 90, 255))
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(red))
    assert w / 4.0 == pytest.approx(100, abs=2.5) and h / 4.0 == pytest.approx(50, abs=2.5)
    assert (x / 4.0 - 20) == pytest.approx(250 - 0, abs=3)                 # at 250 mm from the first corner


def test_distance_measure():
    dets = [{"box": (100, 100, 20, 20), "score": 0.5, "class_id": 0, "label": "delik"},
            {"box": (400, 500, 20, 20), "score": 0.9, "class_id": 0, "label": "delik"},
            {"box": (10, 10, 4, 4), "score": 0.95, "class_id": 0, "label": "toz"}]
    n = create("DistanceMeasure")
    out = n.process({"image": np.zeros((600, 600, 3), np.uint8), "detections": dets, "px_per_mm": None})
    assert out["found"] and out["distance"] == pytest.approx(math_dist((12, 12), (110, 110)), abs=0.01)
    n.values["pick"] = "En büyük iki"
    out = n.process({"image": None, "detections": dets, "px_per_mm": 2.0})
    assert out["distance"] == pytest.approx(250.0) and out["dx"] == 150.0 and out["dy"] == 200.0
    assert out["image"] is None
    n.values.update(pick="En emin iki", px_per_mm=4.0)
    assert n.process({"image": None, "detections": dets, "px_per_mm": None})["distance"] == pytest.approx(
        math_dist((12, 12), (410, 510)) / 4, abs=0.01)
    obb = [{"box": (0, 0, 50, 50), "center": (10.0, 10.0), "score": 1}, {"box": (0, 0, 1, 1), "center": (13.0, 14.0),
                                                                          "score": 1}]
    assert n.process({"image": None, "detections": obb, "px_per_mm": 1.0})["distance"] == 5.0
    assert n.process({"image": None, "detections": dets[:1], "px_per_mm": None})["found"] is False


def math_dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def test_measure_objects_takes_scale_input():
    img = np.zeros((200, 300, 3), np.uint8)
    c = np.array([[[50, 50]], [[150, 50]], [[150, 100]], [[50, 100]]], np.int32)
    n = create("MeasureObjects")
    out = n.process({"image": img, "contours": [c], "px_per_mm": 2.0})
    assert out["width"] == 50.0 and "mm" in out["detections"][0]["label"]
    assert n.process({"image": img, "contours": [c], "px_per_mm": None})["width"] == 100.0


def test_find_board_orientation_is_stable():
    img = render([0, 0, 0], [-100, -60, 500], D=np.zeros(5))
    a = cal.find_board(img, 9, 6)
    b = cal.find_board(cv2.rotate(img, cv2.ROTATE_180), 9, 6)
    h, w = img.shape[:2]
    assert a[0, 1] < a[-1, 1] and b[0, 1] < b[-1, 1]                     # always starts at the top
    assert np.allclose(np.array([w - 1, h - 1]) - b[-1], a[0], atol=1.0)


# --- comment frames ------------------------------------------------------------------------------------------------

def test_frames_saved_loaded_and_cleaned(tmp_path):
    g = Graph()
    g.add(create("TestPattern"))
    g.frames.append({"id": "c1", "title": "Giriş\nkamera burada", "rect": [0, 0, 300, 200], "color": "#3d5a80"})
    data = json.loads(json.dumps(g.to_dict()))
    back = Graph.from_dict(data)
    assert back.frames == g.frames
    data["frames"] += [{"rect": [1, 2]}, {"rect": [0, 0, 5, 5], "color": "red; drop", "title": 7},
                       {"rect": [float("nan"), 0, 1, 1]}, "x"]
    g2 = Graph.from_dict(data)
    assert len(g2.frames) == 2 and g2.frames[1]["color"] == "#3d5a80" and g2.frames[1]["title"] == "7"
    assert g2.frames[1]["rect"][2:] == [80.0, 60.0]                       # minimum size
    assert sum("çerçeve" in w for w in g2.warnings) == 3
    assert "frames" not in Graph().to_dict()                                # old readers see nothing new


@pytest.fixture
def win():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    w.close()
    app.processEvents()


def test_frame_in_editor_moves_blocks_and_undoes(win):
    from PySide6.QtCore import QPointF, QRectF
    from visionflow.ui.scene import FrameItem
    sc = win.scene
    a = sc.add_node("TestPattern", QPointF(100, 100))
    b = sc.add_node("GaussianBlur" if "GaussianBlur" in NODE_REGISTRY else "TestPattern", QPointF(800, 100))
    sc.clearSelection()
    a.setSelected(True)
    f = sc.add_frame(title="Kamera")
    assert isinstance(f, FrameItem) and f.rect_in_scene().contains(a.sceneBoundingRect())
    assert not f.rect_in_scene().contains(b.sceneBoundingRect())
    assert win.scene.graph.to_dict()["frames"][0]["title"] == "Kamera"
    # drag by the title: the block inside comes along, the one outside stays
    sc.clearSelection()
    f._carry = [it for it in sc.items(f.rect_in_scene()) if it is a]
    f._last = f.pos()
    f.setPos(f.pos() + QPointF(50, 30))
    assert a.node.pos == pytest.approx((150, 130)) and b.node.pos == (800, 100)
    assert sc.graph.frames[0]["rect"][:2] == [round(f.pos().x(), 1), round(f.pos().y(), 1)]
    sc.rename_frame(f, "Giriş bölümü\nAçıklama satırı")
    sc.recolor_frame(f)
    win._snapshot()
    saved = sc.graph.to_dict()
    assert saved["frames"][0]["title"].startswith("Giriş") and saved["frames"][0]["color"] != "#3d5a80"
    sc.clearSelection()
    f.setSelected(True)
    sc.delete_selected()
    assert sc.graph.frames == [] and not sc.frames() and a.node.id in sc.graph.nodes   # blocks stay
    win._snapshot()
    win.undo()
    assert len(win.scene.graph.frames) == 1 and len(win.scene.frames()) == 1
    empty = win.scene.add_frame(at=QPointF(2000, 2000))
    assert empty.rect_in_scene() == QRectF(2000, 2000, 420, 260)
    assert win.scene.frame_at(QPointF(2010, 2100)) is empty
    win.scene.remove_frame(empty)
    assert len(win.scene.graph.frames) == 1


def test_frame_resize_and_paint(win):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QImage, QPainter
    f = win.scene.add_frame(at=QPointF(0, 0))
    f.prepareGeometryChange()
    f.w, f.h = 600, 400
    f._store()
    assert win.scene.graph.frames[-1]["rect"][2:] == [600, 400]
    img = QImage(700, 500, QImage.Format_ARGB32)
    p = QPainter(img)
    f.paint(p, None)
    p.end()
    assert f._in_handle(QPointF(595, 395)) and not f._in_handle(QPointF(300, 200))


def test_calibration_dialog(tmp_path, win):
    from visionflow.ui.calibration_dialog import CalibrationDialog
    from PySide6.QtWidgets import QApplication
    for name, img in views(6, seed=1).items():
        cv2.imwrite(str(tmp_path / name), img)
    dlg = CalibrationDialog(win, str(tmp_path))
    dlg.out.setText(str(tmp_path / "out" / "kamera.json"))
    board = dlg.save_board(str(tmp_path / "tahta.png"))
    assert cv2.imread(board).shape[:2] == ((6 + 1) * 120 + 160, (9 + 1) * 120 + 160)
    dlg.run()
    dlg.thread.join(60)
    QApplication.processEvents()
    assert dlg.result is not None and "Kalibrasyon tamam" in dlg.report.toPlainText(), dlg.report.toPlainText()
    assert "tahta.png" in dlg.report.toPlainText()                      # the saved board is not a photo
    assert (tmp_path / "out" / "kamera.json").is_file()
    dlg.folder.setText(str(tmp_path / "bos"))
    dlg.run()
    dlg.thread.join(10)
    QApplication.processEvents()
    assert "Olmadı" in dlg.report.toPlainText()
    assert win.open_calibration() is win.open_calibration()


# --- fixes from the pre-release bug hunt ------------------------------------------------------------------------

def test_locked_scale_follows_setting_changes():
    img = render([0, 0, 0], [-100, -60, 500], D=np.zeros(5))
    blank = np.full_like(img, 255)
    n = create("ScaleFromReference")
    assert n.safe_setup()
    assert n.process({"image": img})["px_per_mm"] == pytest.approx(1.6, rel=0.01)
    n.values["square_mm"] = 10.0                                         # locked, but the size is new
    assert n.process({"image": blank})["px_per_mm"] == pytest.approx(4.0, rel=0.01)
    n.values["cols"] = 8                                                 # another board: search again
    assert not n.process({"image": blank})["found"]
    n.values.update(cols=9, lock=False)
    assert n.process({"image": img})["found"]
    out = n.process({"image": blank})                                    # not locked: the board left the picture
    assert not out["found"] and out["px_per_mm"] == 0.0


def test_plane_rectify_board_size_change_does_not_crash():
    img = render([0, 0, 0], [-100, -60, 500], D=np.zeros(5))
    n = create("PlaneRectify")
    assert n.safe_setup()
    assert n.process({"image": img})["found"]
    n.values["cols"] = 8
    out = n.process({"image": np.full_like(img, 255)})
    assert not out["found"]


def test_undistort_refuses_other_aspect_ratio(calib, tmp_path):
    path = tmp_path / "k.json"
    calib.save(path)
    n = create("LensUndistort")
    n.values.update(file=str(path))
    assert n.safe_setup()
    with pytest.raises(ValueError, match="en-boy"):
        n.process({"image": np.zeros((720, 1280, 3), np.uint8)})


def test_rubber_band_inside_a_frame_leaves_the_frame(win):
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QPainterPath
    sc = win.scene
    a = sc.add_node("TestPattern", QPointF(100, 100))
    sc.clearSelection()
    a.setSelected(True)
    f = sc.add_frame()
    sc.clearSelection()
    band = QPainterPath()
    band.addRect(a.sceneBoundingRect().adjusted(-5, -5, 5, 5))
    sc.setSelectionArea(band)
    assert a.isSelected() and not f.isSelected()
    sc.delete_selected()
    assert len(sc.graph.frames) == 1
    inside = f.rect_in_scene().center()
    assert sc.frame_at(inside) is f                                     # the body still has its menu
    small = sc.add_frame(rect=QRectF(f.pos().x() + 10, f.pos().y() + 40, 150, 100))
    assert sc.frame_at(small.rect_in_scene().center()) is small          # the innermost frame


def test_frames_alone_count_as_unsaved_work(win):
    from PySide6.QtCore import QPointF
    win.scene.add_frame(at=QPointF(0, 0))
    win._snapshot()
    assert win.dirty and "*" in win.windowTitle() or (win._title() or "*" in win.windowTitle())
    asked = []
    from PySide6.QtWidgets import QMessageBox
    orig = QMessageBox.question
    QMessageBox.question = lambda *a, **k: asked.append(1) or QMessageBox.Discard
    try:
        del win._confirm_discard
        assert win._confirm_discard() and asked
    finally:
        QMessageBox.question = orig
        win._confirm_discard = lambda: True


def test_assistant_keeps_frames(win):
    from PySide6.QtCore import QPointF
    win.scene.add_node("TestPattern", QPointF(0, 0))
    win.scene.add_frame(at=QPointF(0, 0), title="Benim notum")
    new = Graph()
    new.add(create("TestPattern"))
    win.apply_generated(new, True)
    assert [f["title"] for f in win.scene.graph.frames] == ["Benim notum"]


def test_duplicate_frame_ids_are_renamed():
    g = Graph.from_dict({"nodes": [], "frames": [{"id": "a", "rect": [0, 0, 100, 100]},
                                                 {"id": "a", "rect": [0, 0, 100, 100]}]})
    assert len({f["id"] for f in g.frames}) == 2
