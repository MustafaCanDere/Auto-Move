"""0.9 bug hunt: each test fails on 0.8 code."""
import os
import pathlib
import shutil
import sys

import cv2
import numpy as np
import pytest

from visionflow.ai.config import AIConfig
from visionflow.core import Graph, create, load_builtin
from visionflow.core import node as core_node
from visionflow.nodes import ai as ai_nodes
from visionflow.nodes.ai_hub import _Feed

load_builtin()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(core_node, "FLOW_DIR", None)


def make(t, **v):
    n = create(t)
    n.values.update(v)
    n.setup()
    return n


# --- Klasörü İzle never forgets files that are still in the folder ------------------------------------------

def test_folder_watch_does_not_report_old_files_again_after_many_new_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(_Feed, "MAX_SEEN", 3)          # a small limit stands in for 5000 files
    for i in range(6):
        (tmp_path / f"eski{i}.txt").write_text("x")
    n = make("FolderWatch", folder=str(tmp_path), pattern="*.txt")
    for i in range(4):
        (tmp_path / f"yeni{i}.txt").write_text("y")
    n._poll()
    n._poll()                                          # second look: sizes unchanged → ready
    names = [pathlib.Path(i["path"]).name for i in n.state["queue"]]
    assert sorted(names) == [f"yeni{i}.txt" for i in range(4)]
    for _ in range(8):
        n.process({"next": None})
    n.state["queue"].clear()
    n._poll()
    assert n.state["queue"] == []                      # 0.8: the oldest were forgotten and came back as "new"
    (tmp_path / "eski0.txt").unlink()
    n._poll()
    assert not any("eski0.txt" in k for k in n.state["seen"])     # files that left are forgotten


# --- Kamera reconnects -------------------------------------------------------------------------------------

class FakeCap:
    def __init__(self, frames):
        self.frames, self.released = list(frames), False

    def isOpened(self):
        return not self.released

    def read(self):
        f = self.frames.pop(0) if self.frames else None
        return (f is not None), f

    def release(self):
        self.released = True

    def set(self, *a):
        return True


def test_camera_reconnects_after_the_stream_drops(monkeypatch):
    frame = np.zeros((4, 4, 3), np.uint8)
    caps = [FakeCap([frame] + [None] * 10), FakeCap([frame] * 3)]
    Camera = type(create("Camera"))
    monkeypatch.setattr(Camera, "_open", lambda self: caps.pop(0))
    n = make("Camera", source="0")
    assert n.process({})["image"] is frame
    for _ in range(5):
        with pytest.raises(RuntimeError, match="görüntü alınamadı"):
            n.process({})
    assert n.process({})["image"] is frame              # reopened with the second capture
    assert caps == []


def test_camera_video_file_end_still_stops_the_flow(tmp_path, monkeypatch):
    from visionflow.core.node import StopFlow
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    Camera = type(create("Camera"))
    monkeypatch.setattr(Camera, "_open", lambda self: FakeCap([]))
    n = make("Camera", source=str(video))
    with pytest.raises(StopFlow):
        n.process({})


# --- YOLO -----------------------------------------------------------------------------------------------------

def test_model_names_accepts_dict_list_and_comma_text():
    assert ai_nodes.model_names("{0: 'kedi', 1: 'köpek'}") == ["kedi", "köpek"]
    assert ai_nodes.model_names("['vida', 'somun']") == ["vida", "somun"]
    assert ai_nodes.model_names("vida, somun") == ["vida", "somun"]
    assert ai_nodes.model_names(None) == []


def test_yolo_small_input_v8_layout_is_not_mistaken_for_v5():
    # 2 classes, only 4 candidate boxes: (1, 4+nc, N) = (1, 6, 4); 0.8 guessed the layout from rows < cols
    out = np.zeros((1, 6, 4), np.float32)
    out[0, :, 0] = [50, 50, 20, 20, 0.1, 0.9]
    dets = ai_nodes.yolo_postprocess(out, 0.5, 0.5, 1.0, (0, 0), ["vida", "somun"])
    assert [(d["label"], d["box"]) for d in dets] == [("somun", (40, 40, 20, 20))]
    v5 = np.zeros((1, 3, 6), np.float32)                         # 1-class YOLOv5: (N, 5+1) is also 6 wide
    v5[0, 0] = [50, 50, 20, 20, 0.9, 0.95]
    dets = ai_nodes.yolo_postprocess(v5, 0.5, 0.5, 1.0, (0, 0), ["vida"])
    assert [(d["label"], d["box"]) for d in dets] == [("vida", (40, 40, 20, 20))]


def test_yolo_end_to_end_output_and_rectangular_model(monkeypatch):
    ort = pytest.importorskip("onnxruntime")

    class Sess:
        def __init__(self, path, providers=None):
            self.seen_shape = None

        def get_inputs(self):
            return [type("I", (), {"name": "images", "shape": [1, 3, 384, 640]})()]

        def get_modelmeta(self):
            return type("M", (), {"custom_metadata_map": {"names": "{0: 'kedi', 1: 'köpek'}"}})()

        def run(self, _, feed):
            Sess.blob_shape = feed["images"].shape
            out = np.zeros((1, 300, 6), np.float32)
            out[0, 0] = [100, 52, 200, 152, 0.8, 1]            # x1,y1,x2,y2,score,class in letterboxed pixels
            return [out]

    monkeypatch.setattr(ort, "InferenceSession", Sess)
    n = make("YoloDetector", model="model.onnx")
    out = n.process({"image": np.zeros((360, 640, 3), np.uint8)})
    assert Sess.blob_shape == (1, 3, 384, 640)                  # not squashed to 640x640
    assert out["count"] == 1
    d = out["detections"][0]
    assert d["label"] == "köpek" and d["box"] == (100, 40, 100, 100)


def test_blocks_restart_when_settings_read_in_setup_change():
    want = {"YoloDetector": {"model", "classes", "size"}, "ImageClassifier": {"model"},
            "ArucoDetector": {"dict"}, "MotionDetector": {"history"}, "Scheduler": set(),
            "FolderWatch": {"folder", "pattern"}}
    for t, names in want.items():
        sp = set(type(create(t)).setup_params)
        assert sp and names <= sp, t


# --- data.yaml, colour, paths ---------------------------------------------------------------------------------

def test_data_yaml_quotes_tricky_class_names(tmp_path):
    yaml = pytest.importorskip("yaml")
    from visionflow.nodes.learning import write_data_yaml
    root = tmp_path / "proje #2"
    root.mkdir()
    p = write_data_yaml(root, ["no", "null", "kusur: çizik", "1"])
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["names"] == {0: "no", 1: "null", 2: "kusur: çizik", 3: "1"}
    assert data["path"].endswith("proje #2")


def test_average_color_roi_outside_and_partly_outside_the_image():
    img = np.zeros((50, 50, 3), np.uint8)
    img[:, :] = (0, 0, 255)
    n = make("AverageColor", roi="40,40,100,100")
    assert n.process({"image": img})["color"] == "kırmızı"
    n.values["roi"] = "80,80,10,10"
    with pytest.raises(ValueError, match="resmin dışında"):
        n.process({"image": img})


def test_relative_paths_are_found_next_to_the_flow(tmp_path, monkeypatch):
    proj = tmp_path / "proje"
    (proj / "veri").mkdir(parents=True)
    (proj / "veri" / "a.txt").write_text("1")
    g = Graph()
    g.add_new("FolderWatch", folder="veri", pattern="*.txt")
    g.save(str(proj / "akis.json"))
    elsewhere = tmp_path / "baska"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)                     # e.g. started from a desktop shortcut
    g2 = Graph.load(str(proj / "akis.json"))
    fw = next(iter(g2.nodes.values()))
    assert pathlib.Path(fw.v("folder")) == (proj / "veri").resolve()
    assert fw.values["folder"] == "veri"              # the saved setting itself is unchanged
    assert core_node.resolve_path("yok/klasor") == "yok/klasor"     # outputs that do not exist yet stay as typed


def test_media_reads_images_from_turkish_folder_names(tmp_path):
    from visionflow.ai.media import _imread
    folder = tmp_path / "Masaüstü çıktı"
    folder.mkdir()
    ok, buf = cv2.imencode(".png", np.full((5, 5, 3), 200, np.uint8))
    buf.tofile(str(folder / "şekil.png"))
    assert _imread(folder / "şekil.png").shape == (5, 5, 3)
    assert _imread(folder / "yok.png") is None


def test_training_uses_an_installed_python_in_the_packaged_app(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from visionflow.ui.tools import TrainDialog
    exe, args = TrainDialog.program()
    assert exe == sys.executable and args[-2:] == ["-m", "visionflow.train"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/python3" if name == "python3" else None)
    exe, args = TrainDialog.program()
    assert exe == "/usr/bin/python3" and args == ["-u", str(tmp_path / "train.py")]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert TrainDialog.program() == (None, [])


# --- Excel ------------------------------------------------------------------------------------------------------

def test_excel_logger_batches_rows_and_writes_the_rest_on_stop(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "k.xlsx"
    saves = []
    real_save = openpyxl.Workbook.save
    monkeypatch.setattr(openpyxl.Workbook, "save", lambda self, f: (saves.append(f), real_save(self, f)))
    n = make("ExcelLogger", path=str(p))
    for i in range(20):
        out = n.process({"a": i, "b": "=1+1", "c": "-abc", "trigger": i % 2 == 0})
    assert len(saves) == 1 and out["rows"] == 1 and out["waiting"] == 9     # 0.8: one save per row
    n.teardown()
    rows = list(openpyxl.load_workbook(p).active.values)
    assert len(rows) == 11 and rows[2][1:] == (2, "=1+1", "-abc")         # formula text kept as text


def test_excel_logger_damaged_file_is_reported_and_not_retried_every_frame(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "k.xlsx"
    p.write_text("bu bir excel değil")
    loads = []
    real = openpyxl.load_workbook
    monkeypatch.setattr(openpyxl, "load_workbook", lambda *a, **k: (loads.append(1), real(*a, **k))[1])
    n = make("ExcelLogger", path=str(p))
    for i in range(10):
        with pytest.raises(RuntimeError, match="bozuk"):
            n.process({"a": i, "b": None, "c": None, "trigger": True if i % 2 == 0 else False})
    assert len(loads) == 1
    assert n.state["pending"][0][1] == 0 and len(n.state["pending"]) == 5    # rows are kept, not lost


# --- Çizgi Geçiş Sayacı ------------------------------------------------------------------------------------------

def test_line_counter_forgets_objects_that_left_long_ago():
    n = make("LineCounter", line="0,50,100,50")
    n.FORGET_AFTER = 10
    for tid in range(200):                              # a conveyor: every part gets a new id
        n.process({"detections": [{"track_id": tid, "box": (40, 55, 10, 10), "prev_center": (45, 40)}],
                   "image": None, "reset": None})
    assert n.state["n_in"] + n.state["n_out"] == 200
    assert len(n.state["side"]) < 70 and len(n.state["counted"]) < 70


# --- Dosyayı Taşı ------------------------------------------------------------------------------------------------

def test_move_file_failure_leaves_no_half_copy_and_retries(tmp_path, monkeypatch):
    src = tmp_path / "gelen" / "fatura.pdf"
    src.parent.mkdir()
    src.write_bytes(b"pdf")
    calls = []

    def locked_move(a, b):
        calls.append(1)
        if len(calls) == 1:
            shutil.copy2(a, b)                          # across drives: copy worked, deleting the original not
            raise PermissionError("kullanımda")
        os.replace(a, b)
    monkeypatch.setattr(shutil, "move", locked_move)
    n = make("MoveFile", folder=str(tmp_path / "islendi"))
    with pytest.raises(RuntimeError, match="taşınamadı"):
        n.process({"path": str(src), "trigger": None})
    assert src.exists() and not (tmp_path / "islendi" / "fatura.pdf").exists()
    assert n.process({"path": str(src), "trigger": None})["moved"] == 0      # waits before trying again
    n.state["retry_at"] = 0
    out = n.process({"path": str(src), "trigger": None})
    assert out["moved"] == 1 and out["new_path"].endswith("fatura.pdf") and not src.exists()


# --- AI Merkezi -------------------------------------------------------------------------------------------------

def test_ai_center_new_chat_while_waiting_does_not_bring_the_old_chat_back(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    from visionflow.ai import claude
    from visionflow.ui.ai_center import AICenter, SESSION
    from test_ai_hub import FakeClaude
    AIConfig(api_key="test").save()
    monkeypatch.setattr(claude, "_test_client", FakeClaude(lambda kw: "eski cevap"))
    w = AICenter()
    pending = []
    monkeypatch.setattr(w, "_run", lambda fn, ok, failed: pending.append((fn, ok)))
    w.chat_in.setPlainText("eski soru")
    w.send()
    w.new_chat()                                        # pressed while the answer is on its way
    fn, ok = pending[0]
    ok(fn())
    assert w.memory.history(SESSION) == []
    assert "eski cevap" not in w.transcript.toPlainText() and w.send_btn.isEnabled()
    w.close()


def test_ai_center_window_is_reused_and_deleted_on_close():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PySide6.QtCore import QCoreApplication, QEvent
    from visionflow.ui.main_window import MainWindow
    AIConfig(api_key="test").save()
    win = MainWindow()
    a = win.open_ai_center()
    assert win.open_ai_center() is a                    # 0.8: a new window (and memory) every Ctrl+M
    a.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    b = win.open_ai_center()
    assert b is not a
    b.close()
    win.saved_snap = win._snap()
    win.close()



def test_read_document_understands_notepad_unicode_and_bom(tmp_path):
    from visionflow.ai.memory import read_document
    (tmp_path / "a.txt").write_bytes("İade süresi 14 gün".encode("utf-16"))
    (tmp_path / "b.txt").write_bytes("﻿Şube: İzmir".encode("utf-8"))
    assert read_document(tmp_path / "a.txt") == "İade süresi 14 gün"
    assert read_document(tmp_path / "b.txt") == "Şube: İzmir"
