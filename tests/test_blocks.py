"""Behavioural tests for individual blocks."""
import socket
import struct
import threading
import time

import cv2
import numpy as np
import pytest

from visionflow.core import Graph, create, load_builtin
from visionflow.nodes.industrial import ModbusClient

load_builtin()


def run_node(type_name, inputs=None, frames=1, **values):
    node = create(type_name)
    node.values.update(values)
    node.setup()
    out = None
    for i in range(frames):
        inp = {name: None for name, _ in node.inputs}
        inp.update(inputs(i) if callable(inputs) else (inputs or {}))
        out = node.process(inp)
    node.teardown()
    return out


def scene_with_red(hue_img=(0, 0, 255)):
    img = np.full((200, 300, 3), 30, np.uint8)
    cv2.circle(img, (150, 100), 40, hue_img, -1)
    return img


def test_color_range_wraps_red():
    # red at hue ~178 must be found with the 170..10 wrap range
    hsv = np.full((100, 100, 3), (178, 255, 255), np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    out = run_node("ColorRange", {"image": img}, h_min=170, h_max=10)
    assert out["ratio"] > 99


def test_qr_reader():
    enc = cv2.QRCodeEncoder.create()
    qr = cv2.resize(enc.encode("VF-12345"), None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(qr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    out = run_node("CodeReader", {"image": cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)}, kind="QR")
    assert out["found"] and out["text"] == "VF-12345"


def test_aruco():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 7, 200)
    img = cv2.copyMakeBorder(marker, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    out = run_node("ArucoDetector", {"image": cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)})
    assert out["count"] == 1 and out["detections"][0]["class_id"] == 7


def test_motion_detector():
    still = np.full((120, 160, 3), 80, np.uint8)
    moving = still.copy()
    cv2.rectangle(moving, (40, 40), (100, 100), (255, 255, 255), -1)
    out = run_node("MotionDetector", lambda i: {"image": still if i < 30 else moving}, frames=31)
    assert out["motion"] and out["ratio"] > 5


def test_reference_compare(tmp_path):
    ref = np.full((100, 100, 3), 120, np.uint8)
    path = tmp_path / "ref.png"
    cv2.imwrite(str(path), ref)
    good = run_node("ReferenceCompare", {"image": ref.copy()}, reference=str(path))
    bad_img = ref.copy()
    cv2.circle(bad_img, (50, 50), 12, (0, 0, 0), -1)
    bad = run_node("ReferenceCompare", {"image": bad_img}, reference=str(path))
    assert good["ok"] and not bad["ok"] and len(bad["detections"]) == 1


def test_measure_in_mm():
    mask = np.zeros((200, 300), np.uint8)
    cv2.rectangle(mask, (50, 50), (149, 99), 255, -1)    # 100 x 50 px
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = run_node("MeasureObjects", {"image": mask, "contours": cnts}, px_per_mm=2.0)
    assert out["width"] == pytest.approx(50, abs=1) and out["height"] == pytest.approx(25, abs=1)


def test_average_color_and_sharpness():
    assert run_node("AverageColor", {"image": np.full((50, 50, 3), (255, 0, 0), np.uint8)})["color"] == "mavi"
    assert run_node("AverageColor", {"image": np.zeros((50, 50, 3), np.uint8)})["color"] == "siyah"
    noise = np.random.default_rng(0).integers(0, 255, (100, 100), dtype=np.uint8)
    assert run_node("Sharpness", {"image": noise})["sharp"]
    assert not run_node("Sharpness", {"image": np.full((100, 100), 128, np.uint8)})["sharp"]


def det(x, y, cid=0):
    return {"box": (x, y, 20, 20), "score": 1.0, "class_id": cid, "label": "kutu"}


def test_tracker_keeps_ids_and_line_counter_counts_once():
    g = Graph()
    py = g.add_new("PythonScript", code=(
        "i = state.setdefault('i', 0); state['i'] = i + 1\n"
        "out = [{'box': (100, 10 + i * 15, 20, 20), 'score': 1.0, 'class_id': 0, 'label': 'kutu'}]"))
    tr = g.add_new("ObjectTracker")
    lc = g.add_new("LineCounter", line="0,100,300,100")
    g.connect(py.id, "out", tr.id, "detections")
    g.connect(tr.id, "detections", lc.id, "detections")
    g.setup()
    ids = set()
    for _ in range(15):
        res = g.run_once()
        ids |= {d["track_id"] for d in res[tr.id].outputs["detections"]}
    assert ids == {1}
    assert res[lc.id].outputs["in"] == 1 and res[lc.id].outputs["out"] == 0


def test_logic_blocks():
    assert run_node("Latch", lambda i: {"set": i == 0, "reset": False}, frames=3)["out"]
    assert not run_node("Latch", lambda i: {"set": i == 0, "reset": i == 2}, frames=3)["out"]
    assert run_node("HoldPulse", lambda i: {"signal": i == 0}, frames=3, seconds=10)["out"]
    assert run_node("Switch", {"cond": False, "a": 1, "b": 2})["out"] == 2
    assert run_node("TextFormat", {"a": 3, "b": "x"}, template="{a}-{b}")["text"] == "3-x"
    assert run_node("Statistics", lambda i: {"value": i}, frames=5, window=3)["mean"] == 3.0
    t = create("IntervalTimer")
    t.values["seconds"] = 0.05
    t.setup()
    end = time.monotonic() + 0.3
    ticked = False
    while time.monotonic() < end and not ticked:
        time.sleep(0.02)
        ticked = t.process({})["tick"]
    assert ticked


def test_image_utils():
    img = scene_with_red()
    assert run_node("Combine", {"a": img, "b": img})["image"].shape == (200, 600, 3)
    assert run_node("Combine", {"a": img, "b": img}, mode="dikey")["image"].shape == (400, 300, 3)
    assert run_node("BrightnessContrast", {"image": img}, brightness=50)["image"].mean() > img.mean()
    assert run_node("Equalize", {"image": img})["image"].shape == img.shape
    assert run_node("DrawText", {"image": img, "a": 5})["image"].shape == img.shape
    assert run_node("DrawShape", {"image": img}, shape="çizgi")["image"].shape == img.shape


def test_video_recorder(tmp_path):
    img = scene_with_red()
    out = run_node("VideoRecorder", lambda i: {"image": img, "record": i < 5}, frames=6, folder=str(tmp_path))
    assert not out["recording"]
    cap = cv2.VideoCapture(out["path"])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5


class FakePLC(threading.Thread):
    """Tiny Modbus TCP server: stores coils/registers in dicts."""

    def __init__(self):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen()
        self.port = self.srv.getsockname()[1]
        self.coils, self.regs = {}, {}

    def run(self):
        conn, _ = self.srv.accept()
        while True:
            hdr = conn.recv(7)
            if len(hdr) < 7:
                return
            tid, _, length, unit = struct.unpack(">HHHB", hdr)
            pdu = conn.recv(length - 1)
            fn, addr, val = struct.unpack(">BHH", pdu[:5])
            if fn == 5:
                self.coils[addr] = val == 0xFF00
                resp = pdu
            elif fn == 6:
                self.regs[addr] = val
                resp = pdu
            elif fn == 3:
                vals = [self.regs.get(addr + i, 0) for i in range(val)]
                resp = struct.pack(f">BB{val}H", 3, 2 * val, *vals)
            else:
                resp = struct.pack(">BB", fn | 0x80, 1)
            conn.sendall(struct.pack(">HHHB", tid, 0, len(resp) + 1, unit) + resp)


def test_modbus_write_and_read():
    plc = FakePLC()
    plc.start()
    out = run_node("ModbusWrite", lambda i: {"value": i >= 1}, frames=3, host="127.0.0.1", port=plc.port,
                   address=4)
    assert out["ok"] and plc.coils[4] is True
    plc2 = FakePLC()
    plc2.start()
    c = ModbusClient("127.0.0.1", plc2.port)
    c.write_register(10, 1234)
    assert c.read_registers(10) == [1234]
    c.close()
