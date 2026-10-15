"""Tests for the 0.3 blocks: learning, OCR, segmentation, S7, OPC UA, web panel."""
import json
import shutil
import socket
import time
import urllib.request

import cv2
import numpy as np
import pytest

from visionflow.core import create, load_builtin

load_builtin()
rng = np.random.default_rng(42)


def run_node(type_name, inputs=None, frames=1, **values):
    node = create(type_name)
    node.values.update(values)
    node.setup()
    try:
        out = None
        for i in range(frames):
            inp = {name: None for name, _ in node.inputs}
            inp.update(inputs(i) if callable(inputs) else (inputs or {}))
            out = node.process(inp)
        return out
    finally:
        node.teardown()


def sample(kind: str) -> np.ndarray:
    img = np.full((80, 80, 3), rng.integers(20, 60), np.uint8)
    c = (int(rng.integers(30, 50)), int(rng.integers(30, 50)))
    r = int(rng.integers(14, 22))
    if kind == "kirmizi_daire":
        cv2.circle(img, c, r, (0, 0, int(rng.integers(180, 255))), -1)
    else:
        cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), (int(rng.integers(180, 255)), 60, 0), -1)
    return img


def make_dataset(root, n=8):
    for cls in ("kirmizi_daire", "mavi_kare"):
        (root / cls).mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(root / cls / f"{i}.png"), sample(cls))


def test_quick_classifier_learns_from_folder(tmp_path):
    make_dataset(tmp_path)
    for cls in ("kirmizi_daire", "mavi_kare"):
        for _ in range(5):
            out = run_node("QuickClassifier", {"image": sample(cls)}, folder=str(tmp_path))
            assert out["label"] == cls


def test_quick_classifier_on_detections(tmp_path):
    make_dataset(tmp_path)
    scene = np.zeros((100, 200, 3), np.uint8)
    scene[10:90, 10:90] = sample("mavi_kare")
    scene[10:90, 110:190] = sample("kirmizi_daire")
    dets = [{"box": (10, 10, 80, 80), "score": 1, "class_id": 0, "label": "x"},
            {"box": (110, 10, 80, 80), "score": 1, "class_id": 0, "label": "x"}]
    out = run_node("QuickClassifier", {"image": scene, "detections": dets}, folder=str(tmp_path), pad=0)
    assert [d["label"] for d in out["detections"]] == ["mavi_kare", "kirmizi_daire"]


def test_sample_collector_then_train(tmp_path):
    for cls in ("kirmizi_daire", "mavi_kare"):
        run_node("SampleCollector", lambda i, c=cls: {"image": sample(c), "trigger": i % 2 == 0}, frames=12,
                 folder=str(tmp_path), label=cls)
    assert len(list((tmp_path / "mavi_kare").glob("*.png"))) == 6
    assert run_node("QuickClassifier", {"image": sample("mavi_kare")}, folder=str(tmp_path))["label"] == "mavi_kare"


def good_part():
    img = np.full((120, 120, 3), 140, np.uint8)
    cv2.rectangle(img, (20, 20), (100, 100), (60, 90, 200), -1)
    cv2.circle(img, (60, 60), 15, (230, 230, 230), -1)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def test_anomaly_detector_finds_scratch(tmp_path):
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"{i}.png"), good_part())
    ok = run_node("AnomalyDetector", {"image": good_part()}, folder=str(tmp_path), size=128)
    bad_img = good_part()
    cv2.line(bad_img, (30, 80), (90, 85), (0, 0, 0), 3)
    bad = run_node("AnomalyDetector", {"image": bad_img}, folder=str(tmp_path), size=128)
    assert ok["ok"] and not bad["ok"]
    x, y, w, h = bad["detections"][0]["box"]
    assert 20 <= x <= 40 and 70 <= y <= 90


def test_dataset_recorder_writes_yolo(tmp_path):
    img = np.zeros((100, 200, 3), np.uint8)
    dets = [{"box": (50, 25, 20, 50), "score": 1, "class_id": 0, "label": "vida #3"}]
    run_node("DatasetRecorder", {"image": img, "detections": dets, "trigger": True}, folder=str(tmp_path))
    label = next((tmp_path / "labels").glob("*.txt")).read_text().split()
    assert label == ["0", "0.300000", "0.500000", "0.100000", "0.500000"]
    assert "0: \"vida\"" in (tmp_path / "data.yaml").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract kurulu değil")
def test_ocr_reads_label():
    img = np.full((120, 520, 3), 255, np.uint8)
    cv2.putText(img, "LOT 4821", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    out = run_node("OCR", {"image": img}, mode="tek satır", lang="eng")
    assert "4821" in out["text"] and out["number"] == 4821


def test_yolo_segment_fake_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    nc, nm, n = 2, 4, 8400
    out0 = np.zeros((1, 4 + nc + nm, n), np.float32)
    out0[0, :4, 0] = (320, 320, 200, 200)
    out0[0, 4 + 1, 0] = 0.9
    out0[0, 4 + nc:, 0] = (10, 0, 0, 0)          # mask = sigmoid(10 * proto0)
    protos = np.full((1, nm, 160, 160), -1, np.float32)
    protos[0, 0, 60:100, 60:100] = 1              # a 160px square in the middle (640 scale)

    def const(name, arr):
        return helper.make_node("Constant", [], [name], value=helper.make_tensor(
            name + "_v", TensorProto.FLOAT, arr.shape, arr.flatten().tolist()))
    graph = helper.make_graph(
        [const("output0", out0), const("output1", protos)], "fake_seg",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, out0.shape),
         helper.make_tensor_value_info("output1", TensorProto.FLOAT, protos.shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "seg.onnx")
    out = run_node("YoloSegment", {"image": np.zeros((640, 640, 3), np.uint8)}, model=str(tmp_path / "seg.onnx"),
                   classes="vida,somun")
    assert out["count"] == 1 and out["detections"][0]["label"] == "somun"
    assert out["detections"][0]["area"] == pytest.approx(160 * 160, rel=0.1)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_web_panel_serves_api_and_snapshot():
    node = create("WebPanel")
    node.values.update(port=free_port(), names="Adet,Durum")
    node.setup()
    try:
        out = node.process({"image": np.zeros((50, 80, 3), np.uint8), "a": 7, "b": True, "c": None, "d": None})
        base = f"http://127.0.0.1:{node.v('port')}"
        assert out["url"].endswith(str(node.v("port")))
        api = json.loads(urllib.request.urlopen(base + "/api", timeout=3).read())
        assert api["values"] == {"Adet": 7, "Durum": "✔"}
        jpg = urllib.request.urlopen(base + "/snapshot.jpg", timeout=3).read()
        assert jpg[:2] == b"\xff\xd8"
        assert b"VisionFlow" in urllib.request.urlopen(base + "/", timeout=3).read()
    finally:
        node.teardown()


def test_siemens_s7_read_write():
    snap7 = pytest.importorskip("snap7")
    port = free_port()
    server = snap7.server.Server()
    db = (snap7.type.WordLen.Byte.ctype * 16)() if hasattr(snap7.type, "WordLen") else bytearray(16)
    server.register_area(snap7.type.SrvArea.DB, 1, db)
    server.start(tcp_port=port)
    try:
        common = dict(host="127.0.0.1", port=port, db=1, rack=0, slot=1)
        assert run_node("S7Write", {"value": 1234}, kind="int", offset=2, **common)["ok"]
        assert run_node("S7Read", kind="int", offset=2, **common)["value"] == 1234
        run_node("S7Write", {"value": True}, kind="bool", offset=0, bit=3, **common)
        assert run_node("S7Read", kind="bool", offset=0, bit=3, **common)["bool"] is True
        run_node("S7Write", {"value": 2.5}, kind="real", offset=4, **common)
        assert run_node("S7Read", kind="real", offset=4, **common)["value"] == 2.5
    finally:
        server.stop()
        server.destroy()


def test_opc_ua_read_write():
    sync = pytest.importorskip("asyncua.sync")
    port = free_port()
    server = sync.Server()
    server.set_endpoint(f"opc.tcp://127.0.0.1:{port}/vf/")
    idx = server.register_namespace("vf")
    var = server.nodes.objects.add_object(idx, "Makine").add_variable(f"ns={idx};s=Makine.Sayac", "Sayac", 0)
    var.set_writable()
    server.start()
    try:
        common = dict(url=f"opc.tcp://127.0.0.1:{port}/vf/", node=f"ns={idx};s=Makine.Sayac")
        assert run_node("OpcUaWrite", {"value": 42}, **common)["ok"]
        time.sleep(0.2)
        assert run_node("OpcUaRead", **common)["number"] == 42
    finally:
        server.stop()
