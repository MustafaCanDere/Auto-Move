"""2.1 (c): inference device selection (CPU / CUDA / TensorRT / OpenVINO / DirectML / CoreML), fallback, flow and
computer default, per-block override, speed test, CLI and the editor dialog. Real GPUs are not available here: a
fake onnxruntime stands in for them."""
import json
import sys
import types

import numpy as np
import pytest

from visionflow import accel
from visionflow.core import Graph, load_builtin
from visionflow.core.registry import NODE_REGISTRY, create

load_builtin()

ONNX_BLOCKS = ("YoloDetector", "YoloSegment", "YoloPose", "YoloOBB", "ImageClassifier", "PatchCore")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("VISIONFLOW_HOME", str(h))
    return h


class FakeSess:
    """Stand-in for onnxruntime.InferenceSession: records what it was asked for."""
    created: list = []
    fail: set = set()              # providers whose start raises
    silent: set = set()            # providers onnxruntime drops silently (missing CUDA libraries)

    def __init__(self, path, providers=None):
        names = [p[0] if isinstance(p, tuple) else p for p in providers or []]
        if names and names[0] in FakeSess.fail:
            raise RuntimeError("CUDA failure 100: no CUDA-capable device")
        self.path, self.providers = path, providers
        self._used = [n for n in names if n not in FakeSess.silent] or ["CPUExecutionProvider"]
        FakeSess.created.append(self)

    def get_providers(self):
        return self._used

    def get_inputs(self):
        return [types.SimpleNamespace(name="images", shape=[1, 3, 64, 64])]

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map={"names": "{0: 'vida'}"})

    def run(self, _, feed):
        return [np.zeros((1, 5, 10), np.float32)]


@pytest.fixture
def fake(monkeypatch):
    FakeSess.created, FakeSess.fail, FakeSess.silent = [], set(), set()
    mod = types.SimpleNamespace(InferenceSession=FakeSess,
                                get_available_providers=lambda: list(mod.providers),
                                providers=["CUDAExecutionProvider", "OpenVINOExecutionProvider",
                                           "TensorrtExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(accel, "_ort", lambda: mod)
    monkeypatch.setattr(accel, "tiny_model", lambda: b"fake_onnx_model")
    return mod


def names(sess):
    return [p[0] if isinstance(p, tuple) else p for p in sess.providers]


# --- provider selection / fallback ---------------------------------------------------------------------------------

def test_explicit_device_uses_its_provider_with_cpu_behind(fake):
    s = accel.session("m.onnx", accel.CUDA)
    assert names(s) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert dict([s.providers[0]])["CUDAExecutionProvider"]["device_id"] == 0
    assert accel.device_of(s) == "CUDA"
    s = accel.session("m.onnx", accel.CPU)
    assert names(s) == ["CPUExecutionProvider"] and accel.device_of(s) == "CPU"
    s = accel.session("m.onnx", accel.OPENVINO)
    assert names(s)[0] == "OpenVINOExecutionProvider" and accel.device_of(s) == "OpenVINO"


def test_auto_prefers_cuda_then_cpu_without_warning(fake, capsys):
    assert accel.device_of(accel.session("m.onnx")) == "CUDA"
    fake.providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert accel.device_of(accel.session("m.onnx", accel.AUTO)) == "OpenVINO"
    fake.providers = ["AzureExecutionProvider", "CPUExecutionProvider"]
    s = accel.session("m.onnx", "Genel ayar")
    assert names(s) == ["CPUExecutionProvider"] and accel.device_of(s) == "CPU"
    assert "UYARI" not in capsys.readouterr().out


def test_missing_device_falls_back_to_cpu_with_turkish_warning(fake, capsys):
    fake.providers = ["CPUExecutionProvider"]
    s = accel.session("m.onnx", accel.DIRECTML, who="YOLO")
    assert names(s) == ["CPUExecutionProvider"]
    assert accel.device_of(s) == accel.FALLBACK == "CPU (yedek)"
    out = capsys.readouterr().out
    assert "UYARI" in out and "[YOLO]" in out and "DirectML" in out and "CPU kullanılıyor" in out


def test_device_that_fails_to_start_falls_back(fake, capsys):
    FakeSess.fail = {"CUDAExecutionProvider"}
    s = accel.session("m.onnx", accel.CUDA)
    assert names(s) == ["CPUExecutionProvider"] and accel.device_of(s) == "CPU (yedek)"
    assert "başlatılamadı" in capsys.readouterr().out


def test_silent_onnxruntime_fallback_is_reported(fake, capsys):
    FakeSess.silent = {"CUDAExecutionProvider"}
    s = accel.session("m.onnx", accel.CUDA)
    assert accel.device_of(s) == "CPU (yedek)"
    assert "UYARI" in capsys.readouterr().out


def test_cpu_failure_is_not_hidden(fake):
    FakeSess.fail = {"CPUExecutionProvider"}
    with pytest.raises(RuntimeError):
        accel.session("m.onnx", accel.CPU)


def test_fp16_option_reaches_tensorrt_and_openvino(fake):
    s = accel.session("m.onnx", accel.TENSORRT, fp16=True)
    assert names(s) == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    opts = s.providers[0][1]
    assert opts["trt_fp16_enable"] is True and opts["trt_engine_cache_enable"] is True
    assert accel.session("m.onnx", accel.TENSORRT).providers[0][1]["trt_fp16_enable"] is False
    assert accel.session("m.onnx", accel.OPENVINO, fp16=True).providers[0][1]["precision"] == "FP16"
    assert "precision" not in accel.session("m.onnx", accel.OPENVINO).providers[0][1]


def test_tensorrt_without_cuda_provider_keeps_only_available_ones(fake):
    fake.providers = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
    assert names(accel.session("m.onnx", accel.TENSORRT)) == ["TensorrtExecutionProvider", "CPUExecutionProvider"]


def test_missing_onnxruntime_gives_turkish_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        accel.session("m.onnx")
    assert accel.ort_providers() == []
    assert "onnxruntime yok" in accel.status_line()


# --- defaults: block / flow / computer -----------------------------------------------------------------------------

def test_settings_files_and_bad_values(home):
    assert accel.pc_settings() == {}
    accel.save_pc_settings(device=accel.OPENVINO, fp16=True)
    assert accel.pc_settings() == {"device": accel.OPENVINO, "fp16": True}
    with pytest.raises(ValueError):
        accel.save_pc_settings(device="Kuantum")
    (home / accel.SETTINGS_FILE).write_text("{bozuk", encoding="utf-8")
    assert accel.pc_settings() == {}
    (home / accel.SETTINGS_FILE).write_text(json.dumps({"device": "yok", "fp16": "evet"}), encoding="utf-8")
    assert accel.pc_settings() == {}
    assert accel.default_device() == accel.AUTO


def _yolo_flow(device_setting):
    g = Graph()
    y = g.add_new("YoloDetector", model="m.onnx", device=device_setting)
    return g, y


def test_block_override_flow_default_and_computer_default(fake):
    accel.save_pc_settings(device=accel.OPENVINO)
    g, y = _yolo_flow(accel.GLOBAL)
    assert g.setup() == {}
    assert y.state["device"] == "OpenVINO"              # computer setting
    g.accel = {"device": accel.CUDA}
    g.setup()
    assert y.state["device"] == "CUDA"                  # flow beats computer
    y.values["device"] = accel.CPU
    g.setup()
    assert y.state["device"] == "CPU"                   # block beats flow
    out = y.process({"image": np.zeros((64, 64, 3), np.uint8)})
    assert out["device"] == "CPU" and out["count"] == 0


def test_flow_fp16_reaches_the_block_session(fake):
    g, y = _yolo_flow(accel.TENSORRT)
    g.accel = {"fp16": True}
    g.setup()
    assert FakeSess.created[-1].providers[0][1]["trt_fp16_enable"] is True


def test_block_inside_sub_flow_uses_the_main_flow_device(fake):
    outer = Graph()
    outer.accel = {"device": accel.OPENVINO}
    inner, y = _yolo_flow(accel.GLOBAL)
    inner.parent = outer
    inner.setup()
    assert y.state["device"] == "OpenVINO"


def test_block_without_flow_uses_computer_setting(fake):
    accel.save_pc_settings(device=accel.CPU)
    n = create("ImageClassifier")
    n.values["model"] = "m.onnx"
    n.setup()
    assert n.state["device"] == "CPU"


def test_every_onnx_block_has_device_setting_and_output():
    for t in ONNX_BLOCKS:
        cls = NODE_REGISTRY[t]
        n = create(t)
        assert n.param("device").choices == accel.BLOCK_CHOICES and n.values["device"] == accel.GLOBAL, t
        assert ("device", "text") in cls.outputs, t
        assert "device" in cls.setup_params, t


def test_flow_device_is_saved_and_loaded(tmp_path):
    g = Graph()
    g.add_new("YoloDetector", device=accel.CUDA)
    g.accel = {"device": accel.DIRECTML, "fp16": True}
    p = tmp_path / "akis.json"
    g.save(str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["accel"] == {"device": accel.DIRECTML, "fp16": True}
    g2 = Graph.load(str(p))
    assert g2.accel == {"device": accel.DIRECTML, "fp16": True} and not g2.warnings
    assert next(iter(g2.nodes.values())).values["device"] == accel.CUDA
    assert "accel" not in Graph().to_dict()             # old-style file stays as it was
    assert Graph.from_dict({"nodes": []}).accel == {}
    bad = Graph.from_dict({"nodes": [], "accel": {"device": "Kuantum"}})
    assert bad.accel == {} and any("bilinmeyen cihaz" in w for w in bad.warnings)
    assert Graph.from_dict({"nodes": [], "accel": "x"}).warnings


# --- real CPU inference still works --------------------------------------------------------------------------------

def test_real_cpu_inference_with_classifier_and_yolo(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    path = tmp_path / "tiny.onnx"
    path.write_bytes(accel.tiny_model())
    g = Graph()
    s = g.add_new("TestPattern", animate=False, width=320, height=240)
    c = g.add_new("ImageClassifier", model=str(path), size=224)
    g.connect(s.id, "image", c.id, "image")
    raw = np.zeros((1, 84, 8400), np.float32)
    raw[0, :4, 0] = (320, 320, 100, 100)
    raw[0, 4 + 2, 0] = 0.95
    const = helper.make_node("Constant", [], ["output0"], value=helper.make_tensor(
        "v", TensorProto.FLOAT, raw.shape, raw.flatten().tolist()))
    model = helper.make_model(helper.make_graph(
        [const], "y", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "y.onnx")
    y = g.add_new("YoloDetector", model=str(tmp_path / "y.onnx"))
    g.connect(s.id, "image", y.id, "image")
    assert g.setup() == {}
    res = g.run_once()
    g.teardown()
    assert res[c.id].error is None and res[c.id].outputs["device"] == "CPU"
    assert 0 <= res[c.id].outputs["class_id"] < 32
    assert res[y.id].outputs["count"] == 1 and res[y.id].outputs["device"] == "CPU"


def test_requesting_a_missing_gpu_on_this_pc_still_runs_on_cpu(tmp_path, capsys):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("bu bilgisayarda CUDA var")
    path = tmp_path / "tiny.onnx"
    path.write_bytes(accel.tiny_model())
    n = create("ImageClassifier")
    n.values.update(model=str(path), device=accel.CUDA)
    n.setup()
    out = n.process({"image": np.zeros((50, 60), np.uint8)})
    assert out["device"] == "CPU (yedek)" and "UYARI" in capsys.readouterr().out


def test_patchcore_reports_its_device(tmp_path, fake):
    import cv2
    from visionflow.nodes import patchcore as pc
    folder = tmp_path / "ok"
    folder.mkdir()
    for i in range(3):
        img = np.full((64, 64, 3), 120 + i, np.uint8)
        cv2.imwrite(str(folder / f"{i}.png"), img)
    n = create("PatchCore")
    n.values.update(folder=str(folder), size=64)
    n.setup()
    out = n.process({"image": np.full((64, 64, 3), 121, np.uint8)})
    assert out["device"] == "CPU (yerleşik)"
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    node = create("PatchCore")
    node.values["device"] = accel.OPENVINO
    f = pc.make_features(pc.BACKBONES[2], str(model), 64, node)
    assert f.device == "OpenVINO" and names(f.sess)[0] == "OpenVINOExecutionProvider"


# --- detection, OpenCV DNN, benchmark, CLI, self-test ----------------------------------------------------------------

def test_available_devices_and_status(fake, monkeypatch):
    devs = {d["name"]: d for d in accel.available_devices()}
    assert set(devs) == set(accel.DEVICES[1:])
    assert devs[accel.CUDA]["available"] and devs[accel.OPENVINO]["available"]
    assert not devs[accel.DIRECTML]["available"] and "directml" in devs[accel.DIRECTML]["note"]
    assert "NVIDIA CUDA" in accel.status_line()
    monkeypatch.setattr(accel, "opencv_cuda_count", lambda: 2)
    assert "OpenCV CUDA: 2" in accel.status_line()
    fake.providers = ["CPUExecutionProvider"]
    assert "yalnız CPU" in accel.status_line()


class FakeNet:
    def __init__(self):
        self.backend = self.target = None

    def setPreferableBackend(self, b):
        self.backend = b

    def setPreferableTarget(self, t):
        self.target = t


def test_opencv_backend(monkeypatch, capsys):
    import cv2
    net = FakeNet()
    assert accel.opencv_backend(net, accel.CPU) == "CPU"
    assert (net.backend, net.target) == (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    monkeypatch.setattr(accel, "opencv_cuda_count", lambda: 0)
    assert accel.opencv_backend(net, accel.CUDA) == "CPU (yedek)" and "UYARI" in capsys.readouterr().out
    assert accel.opencv_backend(net, accel.AUTO) == "CPU"
    monkeypatch.setattr(accel, "opencv_cuda_count", lambda: 1)
    if hasattr(cv2.dnn, "DNN_BACKEND_CUDA"):
        assert accel.opencv_backend(net, accel.AUTO, fp16=True) == "CUDA"
        assert (net.backend, net.target) == (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16)


def test_benchmark_rows_with_errors(fake):
    FakeSess.fail = {"CUDAExecutionProvider"}
    rows = {r["device"]: r for r in accel.benchmark([accel.CPU, accel.CUDA], runs=2, model=b"x")}
    assert rows[accel.CPU]["used"] == "CPU" and rows[accel.CPU]["ms"] >= 0
    assert rows[accel.CUDA]["used"] == "CPU (yedek)"
    FakeSess.fail = {"CPUExecutionProvider"}
    rows = accel.benchmark([accel.CPU], runs=1, model=b"x")
    assert rows[0]["error"].startswith("RuntimeError") and rows[0]["ms"] is None


def test_benchmark_without_any_model(monkeypatch):
    monkeypatch.setattr(accel, "tiny_model", lambda: None)
    with pytest.raises(RuntimeError, match="pip install onnx"):
        accel.benchmark()


def test_gpu_test_cli(monkeypatch, capsys):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from visionflow.__main__ import main
    monkeypatch.setattr(sys, "argv", ["visionflow", "gpu-test", "--tekrar", "3"])
    assert main() == 0
    out = capsys.readouterr().out
    assert "CPUExecutionProvider" in out and "[VAR] CPU" in out and "ms/kare" in out and "NVIDIA CUDA" in out
    monkeypatch.setattr(sys, "argv", ["visionflow", "--gpu-test", "--hizli"])
    assert main() == 0
    assert "ms/kare" not in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["visionflow", "gpu-test", "--tekrar", "x"])
    assert main() == 2
    monkeypatch.setattr(sys, "argv", ["visionflow", "--help"])
    main()
    assert "gpu-test" in capsys.readouterr().out


def test_selftest_lists_gpu_status(tmp_path):
    from visionflow import selftest
    _, lines = selftest.run(str(tmp_path / "r.txt"), ui=False)
    assert any("Hızlandırma (GPU)" in line for line in lines)


# --- editor dialog -------------------------------------------------------------------------------------------------

@pytest.fixture
def win():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    w.stop()
    w.close()
    app.processEvents()


def test_dialog_sets_flow_and_computer_default_and_benchmarks(win, fake):
    from visionflow.ui.accel_dialog import PC
    assert any("Hızlandırma" in a.text() for a in win.findChildren(type(win.undo_act)))
    dlg = win.open_accel()
    assert dlg.table.rowCount() == len(accel.DEVICES) - 1
    assert dlg.flow_combo.currentText() == PC and dlg.pc_combo.currentText() == accel.AUTO
    dlg.flow_combo.setCurrentText(accel.CUDA)
    dlg.fp16.setChecked(True)
    assert win.root.accel == {"device": accel.CUDA, "fp16": True}
    assert '"accel"' in win._snap()
    dlg.pc_combo.setCurrentText(accel.OPENVINO)
    assert accel.pc_settings()["device"] == accel.OPENVINO
    dlg.flow_combo.setCurrentText(PC)
    dlg.fp16.setChecked(False)
    assert win.root.accel == {}
    dlg.run_benchmark(wait=True)
    speeds = [dlg.table.item(r, 2).text() for r in range(dlg.table.rowCount())]
    assert any("ms" in s for s in speeds)
    assert "En hızlı" in dlg.status.text()
    dlg.close()


def test_dialog_reads_flow_setting_of_an_opened_flow(win, tmp_path):
    g = Graph()
    g.accel = {"device": accel.OPENVINO}
    p = tmp_path / "a.json"
    g.save(str(p))
    win.open_flow(str(p))
    dlg = win.open_accel()
    assert dlg.flow_combo.currentText() == accel.OPENVINO and not dlg.fp16.isChecked()
    dlg.close()


def test_qt_own_buttons_are_turkish(qapp=None):
    from PySide6.QtWidgets import QApplication, QDialogButtonBox
    from visionflow.__main__ import _turkish_qt
    app = QApplication.instance() or QApplication([])
    _turkish_qt(app)
    bb = QDialogButtonBox(QDialogButtonBox.Close)
    assert bb.button(QDialogButtonBox.Close).text() == "Kapat"
