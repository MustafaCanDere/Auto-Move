"""1.1 bug hunt: each test fails on 1.0 code."""
import json
import os
import socket
import time
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from visionflow import assistant
from visionflow.ai.config import AIConfig
from visionflow.core import Graph, create, load_builtin, registry

load_builtin()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    AIConfig(api_key="test").save()


def make(t, **v):
    n = create(t)
    n.values.update(v)
    n.setup()
    return n


def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --- web panel ---------------------------------------------------------------------------------------------------

def test_web_panel_page_never_puts_values_into_html():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    n = make("WebPanel", port=port)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            page = r.read().decode()
    finally:
        n.teardown()
    assert "innerHTML" not in page and "textContent=v" in page


# --- assistant ---------------------------------------------------------------------------------------------------

def _current():
    g = Graph()
    cam = g.add_new("TestPattern")
    tg = g.add_new("TelegramSend", token="123:ABC", chat_id="42")
    mail = g.add_new("EmailSend", password="S3cret!", to="a@b.c")
    cam.pos, tg.pos, mail.enabled = (123, 456), (400, 456), False
    return g, cam, tg, mail


def test_assistant_never_sends_passwords_and_keeps_them():
    g, cam, tg, mail = _current()
    sent = assistant._current_flow_json(g)
    assert "S3cret!" not in sent and "123:ABC" not in sent and "a@b.c" in sent
    data = json.loads(sent)
    data["nodes"].append({"id": "yeni", "type": "Preview", "values": {}})
    data["connections"] = [{"src": cam.id, "src_port": "image", "dst": "yeni", "dst_port": "image"}]
    new, errors = assistant.build_graph(data, g)
    assert not errors
    assert new.nodes[tg.id].values["token"] == "123:ABC" and new.nodes[mail.id].values["password"] == "S3cret!"


def test_assistant_change_keeps_ids_positions_and_disabled_blocks():
    g, cam, tg, mail = _current()
    data = json.loads(assistant._current_flow_json(g))
    data["nodes"].append({"id": "yeni", "type": "Preview", "values": {}})
    data["connections"] = [{"src": cam.id, "src_port": "image", "dst": "yeni", "dst_port": "image"}]
    new, _ = assistant.build_graph(data, g)
    assert set(g.nodes) <= set(new.nodes)                                  # 1.0: every id was new
    assert new.nodes[cam.id].pos == (123, 456) and new.nodes[mail.id].enabled is False
    added = next(n for nid, n in new.nodes.items() if nid not in g.nodes)
    assert added.pos[0] > 400                                               # new block next to the old ones


# --- labeling tool ---------------------------------------------------------------------------------------------

def test_labeling_keeps_class_names_from_data_yaml_and_polygon_lines(tmp_path):
    pytest.importorskip("PySide6")
    import cv2
    from visionflow.ui.tools import Dataset
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.png"), np.zeros((100, 100, 3), np.uint8))
    (tmp_path / "data.yaml").write_text("path: .\ntrain: images\nval: images\nnames:\n  0: vida\n  1: somun\n",
                                        encoding="utf-8")
    (tmp_path / "labels" / "a.txt").write_text("1 0.1 0.1 0.2 0.1 0.2 0.2\n0 0.5 0.5 0.2 0.2 0.93\n",
                                               encoding="utf-8")
    ds = Dataset(tmp_path)
    assert ds.classes == ["vida", "somun"]                                  # 1.0: ["nesne"] after the first edit
    img = ds.images[0]
    boxes = ds.load(img, 100, 100)
    assert len(boxes) == 1 and boxes[0][0] == 0                             # the 6-column box is editable
    boxes.append([1, 10, 10, 30, 30])
    ds.save(img, boxes, 100, 100)
    lines = (tmp_path / "labels" / "a.txt").read_text(encoding="utf-8").splitlines()
    assert "1 0.1 0.1 0.2 0.1 0.2 0.2" in lines and len(lines) == 3         # polygon kept
    assert '1: "somun"' in (tmp_path / "data.yaml").read_text(encoding="utf-8")


# --- plugins ------------------------------------------------------------------------------------------------------

def test_a_broken_plugin_does_not_stop_the_others(tmp_path, capsys):
    (tmp_path / "a_bozuk.py").write_text("import olmayan_paket_xyz\n", encoding="utf-8")
    (tmp_path / "b_iyi.py").write_text(
        "from visionflow.core import Node, register\n"
        "@register\nclass EklentiDeneme11(Node):\n    title = 'Deneme'\n    def process(self, inputs):\n"
        "        return {}\n", encoding="utf-8")
    loaded = registry.load_plugins(tmp_path)
    assert loaded == ["b_iyi"] and "EklentiDeneme11" in registry.NODE_REGISTRY
    assert "a_bozuk.py" in registry.PLUGIN_ERRORS and "a_bozuk.py" in capsys.readouterr().out


# --- blocks ----------------------------------------------------------------------------------------------------------

def test_hough_circles_works_at_the_highest_roundness():
    import cv2
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (100, 100), 40, 255, -1)
    n = create("HoughCircles")
    n.values["roundness"] = n.param("roundness").max
    n.setup()
    n.process({"image": img})                                               # 1.0: cv2.error at 1.0


def test_tolerance_check_does_not_count_a_stuck_nan():
    n = make("ToleranceCheck")
    for _ in range(5):
        out = n.process({"value": float("nan"), "trigger": None, "reset": None})
    assert out["nok_count"] == 0 and out["yield"] == 100.0


def test_csv_logger_keeps_rows_while_the_file_is_locked(tmp_path, monkeypatch):
    import builtins
    path = tmp_path / "alt" / "k.csv"
    n = make("CsvLogger", path=str(path))
    n.RETRY_EVERY = 0
    real_open, locked = builtins.open, {"on": True}

    def fake_open(f, *a, **k):
        if str(f) == str(path) and locked["on"]:
            raise PermissionError("Excel")
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, "open", fake_open)
    for i in range(3):
        with pytest.raises(RuntimeError, match=f"{i + 1} satır"):
            n.process({"a": i, "b": None, "trigger": None})
    locked["on"] = False
    n.process({"a": 3, "b": None, "trigger": None})
    rows = path.read_text(encoding="utf-8-sig").splitlines()
    assert rows[0] == "zaman;a;b" and [r.split(";")[1] for r in rows[1:]] == ["0", "1", "2", "3"]


def test_turkish_letters_are_drawn_not_question_marks():
    pytest.importorskip("PIL")
    from visionflow.nodes._text import put_text
    import cv2
    a = np.zeros((60, 300, 3), np.uint8)
    b = a.copy()
    put_text(a, "ŞĞİ çıkan", (5, 40), 1.0, (255, 255, 255), 2)
    cv2.putText(b, "??? ??kan", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    assert a.any() and not np.array_equal(a > 0, b > 0)
    g = np.zeros((40, 200), np.uint8)                                       # gray images too
    put_text(g, "Ölçüm", (2, 30), 0.8, (200,), 1)
    assert g.max() > 100
    out = make("DrawText", text="Sıcaklık: {a}").process({"image": np.zeros((80, 300, 3), np.uint8), "a": 5,
                                                          "b": None})
    assert out["image"].any()


# --- memory ------------------------------------------------------------------------------------------------------------

def test_emptied_document_is_forgotten_when_added_again(tmp_path):
    from visionflow.ai.memory import Memory
    m = Memory(str(tmp_path / "h.db"))
    m.add_text("Vida fiyatı 5 TL. Somun fiyatı 2 TL.", source="fiyat.txt", replace=True)
    assert m.search("vida")
    m.add_text("   ", source="fiyat.txt", replace=True)
    assert not m.search("vida") and "fiyat.txt" not in [s[0] for s in m.sources()]
    m.close()


# --- editor --------------------------------------------------------------------------------------------------------------

def test_block_search_ignores_turkish_letters():
    qapp()
    from visionflow.ui.panels import Palette, search_fold
    assert "izle" in search_fold("Klasörü İzle") and "kirp" in search_fold("Kırp (ROI)")
    p = Palette()
    p._filter("istatistik")
    hits = [p.tree.topLevelItem(i).child(j).text(0) for i in range(p.tree.topLevelItemCount())
            for j in range(p.tree.topLevelItem(i).childCount()) if not p.tree.topLevelItem(i).child(j).isHidden()]
    assert "İstatistik" in hits


def test_wheel_zooms_back_in_after_fit_all_on_a_big_flow():
    app = qapp()
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(900, 700)
    w.show()
    g = Graph()
    for i in range(30):
        n = g.add_new("Blur")
        n.pos = (i % 10 * 600, i // 10 * 900)
    w.scene.load_graph(g)
    w.view.fit_all()
    app.processEvents()
    z0 = w.view.zoom
    assert z0 < 0.2
    ev = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                     Qt.NoModifier, Qt.ScrollEnd, False)
    w.view.wheelEvent(ev)
    assert w.view.zoom > z0                                                 # 1.0: stuck
    w.saved_snap = w._snap()
    w.close()


def test_training_output_split_inside_a_letter_is_read(monkeypatch):
    qapp()
    from visionflow.ui.tools import TrainDialog
    d = TrainDialog()
    got = []
    d.finished_model.connect(got.append)
    data = "epoch 1\nBİTTİ: models/vida.onnx\n".encode("utf-8")
    cut = data.index("İ".encode()) + 1                                      # inside the two bytes of "İ"
    chunks = [data[:cut], data[cut:]]
    d.proc = SimpleNamespace(readAllStandardOutput=lambda: chunks.pop(0))
    d._read()
    d._read()
    assert got == ["models/vida.onnx"]
    d.proc = None
    d.close()


# --- YouTube --------------------------------------------------------------------------------------------------------------

def test_expired_youtube_sign_in_starts_a_new_one(monkeypatch, tmp_path):
    import sys
    import types
    from visionflow.ai import media
    from visionflow.ai.config import home
    token = home() / "youtube_token.json"
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text("{}", encoding="utf-8")
    secret = tmp_path / "client_secret.json"
    secret.write_text("{}", encoding="utf-8")

    class Creds:
        expired, refresh_token, valid = True, "r", False

        @classmethod
        def from_authorized_user_file(cls, *a):
            return cls()

        def refresh(self, req):
            raise RuntimeError("invalid_grant")

    class NewCreds:
        valid = True

        def to_json(self):
            return '{"yeni": 1}'
    flow = SimpleNamespace(run_local_server=lambda port: NewCreds())
    mods = {
        "google": types.ModuleType("google"), "google.auth": types.ModuleType("google.auth"),
        "google.auth.transport": types.ModuleType("google.auth.transport"),
        "google.auth.transport.requests": SimpleNamespace(Request=lambda: None),
        "google.oauth2": types.ModuleType("google.oauth2"),
        "google.oauth2.credentials": SimpleNamespace(Credentials=Creds),
        "google_auth_oauthlib": types.ModuleType("google_auth_oauthlib"),
        "google_auth_oauthlib.flow": SimpleNamespace(InstalledAppFlow=SimpleNamespace(
            from_client_secrets_file=lambda *a: flow)),
        "googleapiclient": types.ModuleType("googleapiclient"),
        "googleapiclient.discovery": SimpleNamespace(build=lambda *a, **k: "servis"),
    }
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    assert media.youtube_service(str(secret)) == "servis"                    # 1.0: RefreshError forever
    assert json.loads(token.read_text(encoding="utf-8")) == {"yeni": 1}
    time.sleep(0)
