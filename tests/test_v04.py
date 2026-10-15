"""Tests for the 0.4 blocks and the 0.4 robustness fixes."""
import http.server
import json
import socketserver
import sqlite3
import threading
import time

import cv2
import numpy as np
import pytest

from visionflow.core import Graph, StopFlow, create, load_builtin
from visionflow.nodes.more import classify_shape

load_builtin()


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


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


# --- blocks ------------------------------------------------------------------------------------------

def test_image_folder_iterates_and_stops(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((10, 10, 3), i * 50, np.uint8))
    (tmp_path / "notlar.txt").write_text("x")
    node = create("ImageFolder")
    node.values.update(folder=str(tmp_path), delay=0)
    node.setup()
    outs = [node.process({}) for _ in range(3)]
    assert [o["name"] for o in outs] == ["0.png", "1.png", "2.png"]
    assert outs[-1]["index"] == 3 and outs[-1]["total"] == 3
    with pytest.raises(StopFlow):
        node.process({})
    node.values["loop"] = True
    assert node.process({})["name"] == "0.png"


def test_image_folder_missing_folder_is_setup_error(tmp_path):
    node = create("ImageFolder")
    node.values["folder"] = str(tmp_path / "yok")
    assert node.safe_setup() is False
    assert "Klasör yok" in node.setup_error


def test_perspective_warp_maps_corners():
    img = np.zeros((200, 200, 3), np.uint8)
    img[50:150, 50:150] = 255
    out = run_node("PerspectiveWarp", {"image": img}, points="50,50,150,50,150,150,50,150", width=100, height=100)
    assert out["image"].shape == (100, 100, 3)
    assert out["image"][5:95, 5:95].min() == 255
    with pytest.raises(ValueError):
        run_node("PerspectiveWarp", {"image": img}, points="1,2,3")


def _contour(draw):
    m = np.zeros((300, 300), np.uint8)
    draw(m)
    return cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]


@pytest.mark.parametrize("draw,expected", [
    (lambda m: cv2.fillPoly(m, [np.int32([[150, 30], [270, 250], [30, 250]])], 255), "üçgen"),
    (lambda m: cv2.rectangle(m, (50, 50), (200, 200), 255, -1), "kare"),
    (lambda m: cv2.rectangle(m, (20, 100), (280, 180), 255, -1), "dikdörtgen"),
    (lambda m: cv2.fillPoly(m, [cv2.ellipse2Poly((150, 150), (100, 100), 0, 0, 360, 72)[:-1]], 255), "beşgen"),
    (lambda m: cv2.fillPoly(m, [cv2.ellipse2Poly((150, 150), (100, 100), 0, 0, 360, 60)[:-1]], 255), "altıgen"),
    (lambda m: cv2.circle(m, (150, 150), 100, 255, -1), "daire"),
    (lambda m: cv2.ellipse(m, (150, 150), (130, 50), 0, 0, 360, 255, -1), "elips"),
])
def test_classify_shape(draw, expected):
    assert classify_shape(_contour(draw)) == expected


def test_shape_detector_filter():
    m = np.zeros((300, 600), np.uint8)
    cv2.circle(m, (100, 150), 60, 255, -1)
    cv2.rectangle(m, (300, 100), (400, 200), 255, -1)
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    assert run_node("ShapeDetector", {"contours": cs})["count"] == 2
    out = run_node("ShapeDetector", {"contours": cs}, only="daire")
    assert out["count"] == 1 and out["detections"][0]["label"] == "daire"


def test_hough_circles_counts_and_measures():
    img = np.full((300, 400, 3), 40, np.uint8)
    for x in (80, 200, 320):
        cv2.circle(img, (x, 150), 40, (230, 230, 230), -1)
    out = run_node("HoughCircles", {"image": img}, min_r=20, max_r=60, min_dist=50)
    assert out["count"] == 3
    assert all(abs(d["radius"] - 40) <= 4 for d in out["detections"])
    assert out["image"].shape == img.shape


def test_count_by_label():
    dets = [{"label": n, "box": (0, 0, 1, 1)} for n in ("car #1", "car #2", "person", "dog")]
    out = run_node("CountByLabel", {"detections": dets}, labels="person,car,truck")
    assert (out["a"], out["b"], out["c"]) == (1, 2, 0)
    assert out["summary"] == "car: 2, dog: 1, person: 1"


def test_zone_counter():
    dets = [{"label": "a", "box": (110, 110, 20, 20)}, {"label": "b", "box": (10, 10, 20, 20)}]
    out = run_node("ZoneCounter", {"detections": dets, "image": np.zeros((400, 600, 3), np.uint8)},
                   zone="100,100,300,100,300,300,100,300")
    assert out["count"] == 1 and out["occupied"] and out["inside"][0]["label"] == "a"
    assert out["image"].shape == (400, 600, 3)
    out = run_node("ZoneCounter", {"detections": dets}, zone="100,100,300,100,300,300,100,300", min_count=2)
    assert not out["occupied"] and out["image"] is None


@pytest.mark.parametrize("mode", ["keskinleştir", "gürültü azalt", "kenar koruyan yumuşatma", "detay artır"])
def test_enhance_modes(mode):
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    assert run_node("Enhance", {"image": img}, mode=mode)["image"].shape == img.shape
    if mode != "detay artır":
        assert run_node("Enhance", {"image": img[:, :, 0].copy()}, mode=mode)["image"].shape == (60, 80)


def test_background_subtract_finds_new_object():
    bg = np.full((120, 160, 3), 90, np.uint8)

    def frames(i):
        f = bg.copy()
        if i == 59:
            cv2.rectangle(f, (40, 40), (80, 80), (250, 250, 250), -1)
        return {"image": f}
    out = run_node("BackgroundSubtract", frames, frames=60)
    assert out["mask"][60, 60] == 255 and out["mask"][10, 10] == 0
    assert 5 < out["ratio"] < 15


def test_chart_draws_and_resets():
    node = create("Chart")
    node.values.update(points=50, width=300, height=150)
    node.setup()
    for i in range(80):
        out = node.process({"a": i, "b": 80 - i, "reset": False})
    assert out["image"].shape == (150, 300, 3)
    assert len(node.state["a"]) == 50
    node.process({"a": None, "b": None, "reset": True})
    assert node.state["a"] == []


def test_sqlite_logger(tmp_path):
    db = tmp_path / "k.db"
    node = create("SQLiteLogger")
    node.values.update(path=str(db), table="sayım; drop", names="adet,durum,etiket")
    node.setup()
    for v in (1, 1, 2, 2, 3):                                   # change-driven: 1, 2, 3
        node.process({"a": v, "b": np.bool_(v > 1), "c": [{"label": "x"}], "trigger": None})
    node.process({"a": 9, "b": None, "c": None, "trigger": True})   # trigger rising -> write
    node.process({"a": 10, "b": None, "c": None, "trigger": True})  # held -> no write
    assert node.process({"a": 10, "b": None, "c": None, "trigger": False})["rows"] == 4
    node.teardown()
    con = sqlite3.connect(db)
    rows = con.execute('SELECT adet, durum, etiket FROM "sayım__drop"').fetchall()
    assert rows == [(1, 0, '["x"]'), (2, 1, '["x"]'), (3, 1, '["x"]'), (9, None, None)]


# --- notifications --------------------------------------------------------------------------------------

class _SMTPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        w = lambda s: self.wfile.write(s.encode() + b"\r\n")  # noqa: E731
        w("220 test")
        while line := self.rfile.readline().decode():
            cmd = line.strip().upper()
            if cmd.startswith(("EHLO", "HELO")):
                w("250 ok")
            elif cmd == "DATA":
                w("354 go")
                data = b""
                while (chunk := self.rfile.readline()) != b".\r\n":
                    data += chunk
                self.server.messages.append(data)
                w("250 queued")
            elif cmd == "QUIT":
                w("221 bye")
                return
            elif cmd.startswith(("MAIL", "RCPT", "RSET", "NOOP")):
                w("250 ok")
            else:
                w("502 not implemented")


def test_email_send_via_local_smtp():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _SMTPHandler)
    srv.messages = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        node = create("EmailSend")
        node.values.update(smtp="127.0.0.1", port=srv.server_address[1], to="a@b.c", user="",
                           body="Hata: {v}", cooldown=100)
        node.setup()
        img = np.zeros((20, 20, 3), np.uint8)
        node.process({"trigger": True, "image": img, "value": 7})
        node.process({"trigger": False, "image": img, "value": 7})
        node.process({"trigger": True, "image": img, "value": 8})   # within cooldown -> skipped
        assert wait_for(lambda: node.state["sent"] == 1)
        time.sleep(0.2)
        assert len(srv.messages) == 1
        raw = srv.messages[0]
        assert b"Hata: 7" in raw and b"goruntu.jpg" in raw
    finally:
        srv.shutdown()


def test_email_requires_recipient():
    node = create("EmailSend")
    node.setup()
    with pytest.raises(ValueError):
        node.process({"trigger": True, "image": None, "value": 1})


def test_telegram_send_to_local_api():
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            got.append((self.path, self.headers["Content-Type"], self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        node = create("TelegramSend")
        node.values.update(token="T0K", chat_id="42", cooldown=0, api=f"http://127.0.0.1:{srv.server_address[1]}")
        node.setup()
        node.process({"trigger": True, "image": None, "value": "kapak yok"})
        node.process({"trigger": False, "image": None, "value": None})
        node.process({"trigger": True, "image": np.zeros((10, 10, 3), np.uint8), "value": 3})
        assert wait_for(lambda: len(got) == 2 and node.state["sent"] == 2)
        by_path = {p: (ct, body) for p, ct, body in got}
        assert json.loads(by_path["/botT0K/sendMessage"][1]) == {"chat_id": "42", "text": "⚠️ Olay algılandı: kapak yok"}
        ct, body = by_path["/botT0K/sendPhoto"]
        assert ct.startswith("multipart/form-data") and b'name="photo"' in body and b"\xff\xd8" in body
    finally:
        srv.shutdown()


# --- 0.4 robustness fixes ---------------------------------------------------------------------------------

def test_setup_error_isolated_to_its_node(tmp_path):
    g = Graph()
    bad = g.add_new("ImageFolder", folder=str(tmp_path / "yok"))
    gray_bad = g.add_new("Grayscale")
    g.connect(bad.id, "image", gray_bad.id, "image")
    good = g.add_new("TestPattern", animate=False)
    gray = g.add_new("Grayscale")
    g.connect(good.id, "image", gray.id, "image")
    errors = g.setup()
    assert list(errors) == [bad.id]
    res = g.run_once()
    assert "Klasör yok" in res[bad.id].error
    assert res[gray.id].error is None and res[gray.id].outputs["image"].ndim == 2
    g.teardown()


def test_failed_source_alone_does_not_spin(tmp_path):
    g = Graph()
    g.add_new("ImageFolder", folder=str(tmp_path / "yok"))
    g.setup()
    assert not g.has_source
    g.teardown()


def test_load_skips_unknown_blocks():
    g = Graph()
    a = g.add_new("TestPattern")
    b = g.add_new("Grayscale")
    g.connect(a.id, "image", b.id, "image")
    data = g.to_dict()
    data["nodes"].append({"id": "x1", "type": "UzaylıBlok", "pos": [0, 0], "values": {}})
    data["connections"].append({"src": "x1", "src_port": "out", "dst": b.id, "dst_port": "image"})
    g2 = Graph.from_dict(data)
    assert len(g2.nodes) == 2 and len(g2.connections) == 1
    assert any("UzaylıBlok" in w for w in g2.warnings)


def test_disabled_flag_roundtrip():
    g = Graph()
    n = g.add_new("Blur")
    n.enabled = False
    g2 = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert g2.nodes[n.id].enabled is False
