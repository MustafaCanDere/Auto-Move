"""2.3 (a): Ekran Tasarımcısı — operator screens designed by drag and drop, bound to block outputs / settings /
shared variables, shown as the operator panel (also with --operator)."""
import json
import os
import threading
import time
import types

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow.core import Graph, Node, NodeResult, Param, PortType, StopFlow, load_builtin, register  # noqa: E402
from visionflow.core import screens as S  # noqa: E402
from visionflow.core.operator import clean_operator, make_lock  # noqa: E402
from visionflow.nodes import connect  # noqa: E402
from visionflow.nodes import subflow as sf  # noqa: E402

load_builtin()


@register
class _Part23a(Node):
    """A camera stand-in: a picture, ok (every third bad), a measured value and counters; stops after N frames."""
    type_name = "_Part23a"
    outputs = [("image", PortType.IMAGE), ("ok", PortType.BOOL), ("value", PortType.NUMBER),
               ("ok_count", PortType.NUMBER), ("nok_count", PortType.NUMBER), ("label", PortType.TEXT)]
    params = [Param("gain", "float", 1.0, "Kazanç", 0.0, 10.0), Param("count", "int", 3, "Adet", 1, 20),
              Param("mode", "choice", "a", "Mod", choices=["a", "b"]), Param("on", "bool", False, "Açık")]
    is_source = True
    stop_after = 12

    def setup(self):
        self.state.update(i=0, ok=0, nok=0)

    def process(self, inputs):
        st = self.state
        st["i"] += 1
        if self.stop_after and st["i"] > self.stop_after:
            raise StopFlow()
        ok = st["i"] % 3 != 0
        st["ok" if ok else "nok"] += 1
        img = np.zeros((60, 80, 3), np.uint8)
        cv2.circle(img, (40, 30), 20, (0, 255, 0) if ok else (0, 0, 255), -1)
        return {"image": img, "ok": ok, "value": st["i"] * 10.0 * self.v("gain"), "ok_count": st["ok"],
                "nok_count": st["nok"], "label": f"parça {st['i']}"}


@register
class _Fail23a(Node):
    type_name = "_Fail23a"
    inputs = [("x", PortType.NUMBER)]
    outputs = [("y", PortType.NUMBER)]

    def process(self, inputs):
        raise ValueError("ölçüm başarısız")


def flow(stop_after=12, screen=None, lock=None, layout="screen"):
    _Part23a.stop_after = stop_after
    g = Graph()
    src = g.add_new("_Part23a", (0, 0))
    prev = g.add_new("Preview", (260, 0))
    g.connect(src.id, "image", prev.id, "image")
    cfg = {"title": "Hat", "image": "", "lamp": {"node": src.id, "port": "ok"}, "history": 5,
           "good": {"node": src.id, "port": "ok_count"}, "bad": {"node": src.id, "port": "nok_count"}}
    if screen is not None:
        cfg["screen"] = screen(src) if callable(screen) else screen
        cfg["layout"] = layout
    if lock:
        cfg["lock"] = make_lock(lock, iterations=2000)
    g.operator = clean_operator(cfg, g)
    return g, src


def basic_screen(src):
    """One page with every widget kind bound to the source block."""
    sid = src.id
    ws = [S.new_widget("number", 0, 0, "num", src={"node": sid, "port": "value"}, decimals=0, unit="mm",
                       thresholds=[[50, "#ff9800"], [100, "#c62828"]]),
          S.new_widget("lamp", 300, 0, "lamp", src={"node": sid, "port": "ok"}),
          S.new_widget("bar", 0, 200, "bar", src={"node": sid, "port": "value"}, max=200.0),
          S.new_widget("trend", 400, 200, "trend", src={"node": sid, "port": "value"}, points=50),
          S.new_widget("image", 700, 0, "img", src={"node": sid, "port": "image"}),
          S.new_widget("value", 0, 500, "lbl", src={"node": sid, "port": "label"}, text="Son: {}"),
          S.new_widget("counters", 400, 500, "cnt"),
          S.new_widget("alarms", 800, 500, "alm"),
          S.new_widget("button", 0, 600, "btn", action="değişken", target={"var": "hmi_v"}, on_value="1",
                       off_value="0"),
          S.new_widget("input", 300, 600, "inp", target={"node": sid, "setting": "gain"}, min=0.0, max=5.0),
          S.new_widget("recipe", 600, 600, "rcp"),
          S.new_widget("text", 900, 700, "txt", text="Başlık"),
          S.new_widget("rect", 0, 700, "box"),
          S.new_widget("picture", 300, 700, "pic", file="yok.png")]
    return {"size": [1280, 800], "pages": [{"name": "Ana", "widgets": ws}, {"name": "İkinci", "widgets": []}]}


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _clean_vars():
    connect.VARS.clear()
    yield
    connect.VARS.clear()


def res(**outputs):
    return NodeResult(outputs, 1.0)


def pump(qapp, n=10):
    for _ in range(n):
        qapp.processEvents()


def render(view, w=1280, h=800):
    """The current page drawn into an image (what the operator sees at the design resolution)."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    view.scene.render(p, QRectF(0, 0, w, h), QRectF(0, 0, w, h))
    p.end()
    return img


# --- model --------------------------------------------------------------------------------------------------------

def test_every_widget_type_round_trips():
    used = set()
    for wtype in S.WIDGETS:
        wd = S.new_widget(wtype, 10, 20, S.unique_id(used))
        used.add(wd["id"])
        back = S.clean_widget(json.loads(json.dumps(wd)), None, [], set())
        assert back == wd, wtype
        assert S.widget_problem(wd) is None
    with pytest.raises(ValueError):
        S.new_widget("uçan_daire")


def test_clean_screen_is_robust():
    g, src = flow()
    warnings = []
    data = {"size": ["x", None], "rate": 999, "grid": -5, "pages": [
        "bozuk",
        {"name": "A", "bg": "kırmızı", "widgets": [
            {"type": "gelecek_öğe", "id": "z", "rect": [1, 2, 3, 4], "props": {"a": 1}},
            {"type": "number", "id": "n", "rect": [0, 0, "nan", 5], "props": {"src": {"node": "yok", "port": "v"},
                                                                          "decimals": "çok", "color": "mavi"}},
            {"type": "number", "id": "n", "props": {"src": {"node": src.id}}},          # broken binding, same id
            {"type": "lamp", "props": {"src": {"node": src.id, "port": "nope"}}},
            {"type": "image", "props": {"src": {"node": src.id, "port": "ok"}}},       # not a picture
            {"type": "button", "props": {"target": {"counter": "good"}}},              # cannot write a counter
            42, {"rect": [0, 0, 1, 1]}]}]}
    sc = S.clean_screen(data, g, warnings)
    assert sc["size"] == list(S.DEFAULT_SIZE) and sc["rate"] == 30 and sc["grid"] == 1
    page = sc["pages"][0]
    assert page["bg"] == "#15161a" and len(page["widgets"]) == 6
    ids = [w["id"] for w in page["widgets"]]
    assert len(set(ids)) == len(ids)
    unknown, num, num2, lamp, img, btn = page["widgets"]
    assert unknown["type"] == "gelecek_öğe" and unknown["props"] == {"a": 1}      # kept for a newer version
    assert "Bilinmeyen öğe türü" in S.widget_problem(unknown, g)
    assert num["rect"][2:] == [240.0, 140.0] and num["props"]["decimals"] == 1 and num["props"]["color"] == "#e8e8e8"
    assert num["props"]["src"] == {"node": "yok", "port": "v"} and "Blok bulunamadı" in S.widget_problem(num, g)
    assert num2["props"]["src"] is None
    assert "çıkışı yok" in S.widget_problem(lamp, g)
    assert "görüntü çıkışı değil" in S.widget_problem(img, g)
    assert btn["props"]["target"] is None
    text = "\n".join(warnings)
    assert "bilinmeyen öğe türü" in text and "yer tutucu" in text and "bozuk" in text
    assert S.clean_screen("çöp", g, warnings) is None
    assert S.clean_screen({}, g)["pages"][0]["name"] == "Ana sayfa"


def test_flow_file_round_trip(tmp_path):
    g, src = flow(screen=basic_screen)
    path = str(tmp_path / "hat.json")
    g.save(path)
    g2 = Graph.load(path)
    assert not g2.warnings
    assert g2.operator["layout"] == "screen"
    assert g2.operator["screen"] == g.operator["screen"]
    assert g2.operator["screen"]["pages"][0]["widgets"][0]["props"]["thresholds"] == [[50.0, "#ff9800"],
                                                                                     [100.0, "#c62828"]]
    # the block is gone: the file still opens, the widget keeps its binding and warns
    data = json.loads(open(path, encoding="utf-8").read())
    data["nodes"] = [n for n in data["nodes"] if n["type"] != "_Part23a"]
    g3 = Graph.from_dict(data)
    assert any("Ekran tasarımı" in w and "yer tutucu" in w for w in g3.warnings)
    num = g3.operator["screen"]["pages"][0]["widgets"][0]
    assert num["props"]["src"]["node"] == src.id and S.widget_problem(num, g3)
    # no screen: the old operator settings stay as they were
    g4, _ = flow()
    assert "screen" not in g4.operator and "layout" not in g4.operator
    # a layout without a screen falls back to the automatic panel
    g5 = Graph()
    g5.operator = clean_operator({"layout": "screen"}, g5)
    assert "layout" not in g5.operator


def test_values_thresholds_and_inputs():
    assert S.format_number(1234.5, 1, "mm", True) == "1.234,5 mm"
    assert S.format_number(None) == "–" and S.format_number(True, 0) == "1" and S.format_number("abc") == "abc"
    th = S.parse_thresholds("80=#C62828; 50=#ff9800")
    assert th == [[50.0, "#ff9800"], [80.0, "#c62828"]]
    assert S.pick_color(10, th, "#fff") == "#fff" and S.pick_color(60, th, "#fff") == "#ff9800"
    assert S.pick_color(80, th, "#fff") == "#c62828" and S.pick_color("yok", th, "#fff") == "#fff"
    with pytest.raises(ValueError, match="okunamadı"):
        S.parse_thresholds("50 kırmızı")
    p = Param("t", "int", 5, "Eşik", 0, 255)
    assert S.parse_input("12", p) == 12
    for bad, msg in (("abc", "Sayı"), ("300", "aralığında"), ("1.5", "Tam sayı"), ("12", "aralığında")):
        with pytest.raises(ValueError, match=msg):
            S.parse_input(bad, p, *((0.0, 10.0) if bad == "12" else (0.0, 0.0)))
    assert S.parse_input("2,5", Param("f", "float", 1.0)) == 2.5
    assert S.parse_input("evet", Param("b", "bool", False)) is True
    with pytest.raises(ValueError):
        S.parse_input("belki", Param("b", "bool", False))
    with pytest.raises(ValueError, match="Seçenekler"):
        S.parse_input("c", Param("c", "choice", "a", choices=["a", "b"]))
    with pytest.raises(ValueError):
        S.parse_input("x", Param("c", "code", ""))
    assert S.parse_input("7", None) == 7 and S.parse_input("7.5", None) == 7.5 and S.parse_input("abc", None) == "abc"
    with pytest.raises(ValueError):
        S.parse_input("70", None, 0.0, 10.0)


def test_write_setting_inside_subflow_and_grouping_keeps_bindings():
    g = Graph()
    src = g.add_new("_Part23a", (0, 0))
    thr = g.add_new("Threshold", (200, 0))
    g.connect(src.id, "image", thr.id, "image")
    name = next(p.name for p in thr.params if p.kind in ("int", "float"))
    screen = {"pages": [{"name": "A", "widgets": [
        S.new_widget("input", 0, 0, "i", target={"node": thr.id, "setting": name}),
        S.new_widget("image", 0, 100, "im", src={"node": thr.id, "port": "image"})]}]}
    g.operator = clean_operator({"screen": screen, "layout": "screen"}, g)
    sub = sf.group(g, [thr.id], "Maske")
    widgets = g.operator["screen"]["pages"][0]["widgets"]
    key = f"{sub.id}/{thr.id}"
    assert widgets[0]["props"]["target"] == {"node": key, "setting": name}
    assert widgets[1]["props"]["src"]["node"] == key and S.widget_problem(widgets[1], g) is None
    S.write_setting(g, key, name, 77)
    g2 = Graph.from_dict(g.to_dict())
    from visionflow.core.graph import find_node
    assert find_node(g2, key).values[name] == 77
    with pytest.raises(ValueError):
        S.write_setting(g, "yok", name, 1)
    sf.ungroup(g, sub.id)
    assert g.operator["screen"]["pages"][0]["widgets"][0]["props"]["target"]["node"] == thr.id


# --- runtime --------------------------------------------------------------------------------------------------------

def make_view(g, host=None):
    from visionflow.ui.screen_runtime import ScreenHost, ScreenView
    return ScreenView(g.operator["screen"], g, host or ScreenHost(g))


def test_feed_updates_values_colours_and_image(qapp):
    from visionflow.production import Tally
    g, src = flow(screen=basic_screen)
    from visionflow.ui.screen_runtime import ScreenHost
    tally = Tally(g.operator)
    view = make_view(g, ScreenHost(g, tally=tally))
    img = np.zeros((60, 80, 3), np.uint8)
    img[:] = (0, 255, 0)
    results = {src.id: res(image=img, ok=True, value=60.0, ok_count=1, nok_count=0, label="p1")}
    view.feed.feed(results, g)
    tally.feed(results, g)
    view.poll()
    it = view.item
    assert it("num").value == 60.0 and it("lamp").lamp_state() is True and it("lbl").value == "p1"
    assert it("img").qimage is not None and not it("img").qimage.isNull()
    assert it("trend").trend == [60.0] and it("cnt").value["good"] == 1
    shot = render(view)
    from PySide6.QtGui import QColor
    lamp_c = shot.pixelColor(380, 20)                     # lamp centre: the "on" green
    assert lamp_c.green() > 100 and lamp_c.red() < 80
    view.feed.feed({src.id: res(image=img, ok=False, value=120.0, ok_count=1, nok_count=1, label="p2")}, g)
    view.poll()
    assert it("lamp").lamp_state() is False and it("num").value == 120.0
    assert S.pick_color(it("num").value, it("num").props["thresholds"], "#fff") == "#c62828"
    shot = render(view)
    assert shot.pixelColor(380, 20).red() > 150 and shot.pixelColor(380, 20).green() < 80
    assert it("trend").trend == [60.0, 120.0]
    # a block error shows HATA on the widget and lands in the alarm list once
    for _ in range(3):
        view.feed.feed({src.id: NodeResult({}, 1.0, error="kamera koptu")}, g)
    view.poll()
    assert it("num").error == "kamera koptu" and it("lamp").lamp_state() is None
    assert len(it("alm").value) == 1 and "kamera koptu" in it("alm").value[0][2]
    # waiting frames change nothing; images are rate limited but never lost forever
    view.feed.feed({src.id: NodeResult({}, 1.0, waiting="tetik bekleniyor")}, g)
    view.poll()
    assert it("num").error == "kamera koptu"
    assert isinstance(render(view), type(shot))
    view.stop_timer()
    QColor  # noqa: B018


def test_odd_values_do_not_break_the_screen(qapp):
    g, src = flow(screen=basic_screen)
    view = make_view(g)
    odd = [None, float("nan"), np.float32(3.5), np.array([1, 2, 3]), [1, 2], {"a": 1}, "metin", np.bool_(True),
           np.zeros((5, 5), np.uint16)]
    for v in odd:
        view.feed.feed({src.id: res(image=v, ok=v, value=v, label=v)}, g)
        view.poll()
        render(view)
    assert view.item("num").value == "<dizi 5x5>" and view.item("lbl").value == "<dizi 5x5>"
    for img in (np.zeros((10, 10), np.uint16), np.zeros((10, 10, 4), np.uint8), np.zeros((10, 10), np.float32),
                np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), bool)):
        view.feed.feed({src.id: res(image=img)}, g)
        view.poll()
        render(view)
    assert view.item("img").qimage is not None
    view.stop_timer()


def test_placeholder_for_bad_references(qapp):
    g, src = flow(screen=basic_screen)
    data = g.to_dict()
    data["nodes"] = [n for n in data["nodes"] if n["type"] != "_Part23a"]
    data["operator"]["screen"]["pages"][0]["widgets"].append({"type": "gelecek_öğe", "rect": [0, 0, 50, 50]})
    g2 = Graph.from_dict(data)
    view = make_view(g2)
    probs = {it.wd["id"]: it.problem for it in view.items}
    assert "Blok bulunamadı" in probs["num"] and probs["txt"] is None and probs["cnt"] is None
    assert any(p and "Bilinmeyen" in p for p in probs.values())
    view.poll()
    shot = render(view)
    amber = [shot.pixelColor(x, y) for x in range(0, 240) for y in (1, 2)]   # dashed frame of 'num'
    assert sum(1 for c in amber if c.red() > 150 and c.green() > 100 and c.blue() < 100) > 50
    view.stop_timer()


def test_buttons_write_variables_and_settings(qapp, monkeypatch):
    from visionflow.ui import operator as op
    from visionflow.ui.screen_runtime import ScreenHost

    class FakeWorker:
        def __init__(self):
            self.params, self.recipes = [], []

        def set_param(self, *a):
            self.params.append(a)

        def request_recipe(self, name):
            self.recipes.append(name)

    g, src = flow(screen=basic_screen, lock="1234")
    worker = FakeWorker()
    written = []
    host = ScreenHost(g, g.operator["lock"], worker=lambda: worker, after_write=lambda *a: written.append(a))
    view = make_view(g, host)
    btn = view.item("btn")
    view.press(btn)                                        # momentary: 1 while pressed, 0 when released
    assert connect.VARS["hmi_v"] == 1
    view.release(btn)
    assert connect.VARS["hmi_v"] == 0
    btn.wd["props"]["mode"] = "aç/kapa"
    view.poll()
    view.release(btn)
    assert connect.VARS["hmi_v"] == 1
    view.poll()
    assert btn.button_active()
    view.release(btn)
    assert connect.VARS["hmi_v"] == 0
    btn.wd["props"]["on_value"] = "doğru"
    view.release(btn)
    assert connect.VARS["hmi_v"] is True
    # a setting button on a locked flow asks for the password
    btn.wd["props"].update(action="ayar", mode="anlık", target={"node": src.id, "setting": "count"}, on_value="9",
                           off_value="3")
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: False)
    view.press(btn)
    view.release(btn)
    assert not worker.params and "Şifre yanlış" in view.last_message
    asked = []
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: asked.append(1) or True)
    view.press(btn)
    assert worker.params[-1] == (src.id, "count", 9) and g.nodes[src.id].values["count"] == 9
    view.release(btn)
    assert worker.params[-1] == (src.id, "count", 3) and len(asked) == 1     # unlocked for a while
    assert written[-1] == (src.id, "count", 3)
    btn.wd["props"]["on_value"] = "999"                   # out of the setting's range: refused
    view.press(btn)
    assert g.nodes[src.id].values["count"] == 3 and "Yazılamadı" in view.last_message
    view.release(btn)
    btn.wd["props"]["lock"] = False
    host.unlocked_until = 0
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: pytest.fail("no password for this button"))
    view.press(btn)
    view.release(btn)
    # start / stop / reset / page buttons
    calls = []
    host._start, host._stop, host._reset = (lambda: calls.append("start")), (lambda: calls.append("stop")), \
        (lambda: calls.append("reset"))
    for act in ("başlat", "durdur", "sayaçları sıfırla"):
        btn.wd["props"]["action"] = act
        view.release(btn)
    assert calls == ["start", "stop", "reset"]
    btn.wd["props"].update(action="sayfa", page="İkinci")
    view.release(btn)
    assert view.page_index == 1 and view.tabs.currentIndex() == 1
    assert view.go_page("-") and view.page_index == 0 and view.go_page("2") and view.page_index == 1
    assert not view.go_page("yok") and "bulunamadı" in view.last_message
    view.stop_timer()


def test_input_validation_and_lock(qapp, monkeypatch):
    from visionflow.ui import operator as op
    from visionflow.ui import screen_runtime as rt
    g, src = flow(screen=basic_screen, lock="1234")
    host = rt.ScreenHost(g, g.operator["lock"])
    view = make_view(g, host)
    inp = view.item("inp")
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: False)
    assert not view.edit_input(inp, "2") and g.nodes[src.id].values["gain"] == 1.0
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: True)
    assert not view.edit_input(inp, "abc") and "Geçersiz" in view.last_message
    assert not view.edit_input(inp, "7") and "aralığında" in view.last_message      # widget range 0…5
    assert view.edit_input(inp, "2,5") and g.nodes[src.id].values["gain"] == 2.5
    view.poll()
    assert inp.value == 2.5
    monkeypatch.setattr(rt, "ask_value", lambda *a, **k: ("4", True))
    assert view.edit_input(inp) and g.nodes[src.id].values["gain"] == 4.0
    monkeypatch.setattr(rt, "ask_value", lambda *a, **k: ("", False))
    assert not view.edit_input(inp)
    # choice / bool settings are picked from a list; a variable input takes text or numbers
    inp.wd["props"].update(target={"node": src.id, "setting": "mode"}, min=0.0, max=0.0)
    seen = {}
    monkeypatch.setattr(rt, "ask_value", lambda p, t, l, cur, items=None: seen.update(items=items) or ("b", True))
    assert view.edit_input(inp) and seen["items"] == ["a", "b"] and g.nodes[src.id].values["mode"] == "b"
    inp.wd["props"]["target"] = {"var": "hedef"}
    assert view.edit_input(inp, "12") and connect.VARS["hedef"] == 12
    inp.wd["props"]["target"] = {"node": "yok", "setting": "x"}
    assert not view.edit_input(inp, "1") and "bulunamadı" in view.last_message
    view.stop_timer()


def test_recipe_selector(qapp, monkeypatch):
    from visionflow.ui import screen_runtime as rt
    g, src = flow(screen=basic_screen)
    g.recipes = [{"name": "A", "values": {src.id: {"gain": 2.0}}}, {"name": "B", "values": {src.id: {"gain": 3.0}}}]
    view = make_view(g)
    rcp = view.item("rcp")
    monkeypatch.setattr(rt, "ask_value", lambda *a, **k: ("B", True))
    assert view.choose_recipe(rcp) and g.nodes[src.id].values["gain"] == 3.0
    view.poll()
    assert rcp.value == "B"
    assert not view.choose_recipe(rcp, "yok") and "reçete yok" in view.last_message
    g.recipes = []
    assert not view.choose_recipe(rcp) and "reçete yok" in view.last_message
    view.stop_timer()


# --- the operator panel with a designed screen --------------------------------------------------------------------

def test_panel_shows_designed_screen_and_runs(qapp, tmp_path):
    from visionflow.ui.operator import OperatorPanel
    g, src = flow(screen=basic_screen)
    panel = OperatorPanel(g.to_dict(), str(tmp_path / "hat.json"))
    assert panel.screen_view is not None and panel.classic.isHidden()
    panel.start()
    assert panel.worker.screen is panel.screen_view.feed
    assert panel.worker.wait(10000)
    pump(qapp, 20)
    assert panel.worker is None
    view = panel.screen_view
    view.poll()
    assert view.item("num").value == 120.0                   # 12 frames × 10
    assert view.item("trend").trend[-3:] == [100.0, 110.0, 120.0] and len(view.item("trend").trend) == 12
    assert view.item("img").qimage is not None
    assert view.item("cnt").value["total"] == 12 and view.item("cnt").value["bad"] == 4
    assert view.item("lamp").lamp_state() is False           # frame 12 is a bad part
    # a setting written on the screen is kept for the next start
    assert view.edit_input(view.item("inp"), "0.5")
    assert next(n for n in panel.graph_data["nodes"] if n["id"] == src.id)["values"]["gain"] == 0.5
    panel.start()
    panel.worker.wait(10000)
    pump(qapp, 20)
    view.poll()
    assert view.item("num").value == 60.0                    # 12 × 10 × 0.5
    panel.close()
    pump(qapp)


def test_panel_screen_scales_and_hides_bar(qapp, tmp_path):
    from visionflow.ui.operator import OperatorPanel

    def screen(src):
        sc = basic_screen(src)
        sc["show_bar"] = False
        return sc
    g, src = flow(screen=screen)
    panel = OperatorPanel(g.to_dict(), None)
    assert panel.bottom_bar.isHidden()
    for w, h in ((1920, 1080), (800, 600), (1024, 1280)):
        panel.resize(w, h)
        panel.show()
        pump(qapp)
        view = panel.screen_view.view
        view.fit()
        s = view.transform().m11()
        vw, vh = view.viewport().width(), view.viewport().height()
        assert abs(s - min(vw / 1280, vh / 800)) < 0.02          # the whole design fits, aspect kept
    panel.close()
    # layout "auto" keeps the classic panel even with a screen saved
    g2, _ = flow(screen=basic_screen, layout="auto")
    panel2 = OperatorPanel(g2.to_dict(), None)
    assert panel2.screen_view is None and not panel2.classic.isHidden()
    panel2.close()


def test_operator_command_line_with_designed_screen(qapp, tmp_path, monkeypatch):
    from visionflow import __main__ as main_mod
    from visionflow.ui import operator as op
    g, src = flow(stop_after=5, screen=basic_screen)
    path = str(tmp_path / "hat.json")
    g.save(path)
    made = []
    real = op.run_operator

    def capture(p, editor_factory=None):
        panel = real(p, editor_factory)
        made.append(panel)
        return panel
    monkeypatch.setattr(op, "run_operator", capture)

    def run_loop():
        panel = made[0]
        assert panel.isFullScreen() or panel.isVisible()
        assert panel.worker is not None and panel.worker.wait(10000)
        pump(qapp, 20)
        return 0
    assert main_mod._operator(types.SimpleNamespace(exec=run_loop), [path], []) == 0
    panel = made[0]
    assert panel.screen_view is not None and panel.screen_view.item("num").value == 50.0
    panel.close()
    pump(qapp)


def test_repeated_start_stop_leaves_nothing_running(qapp, tmp_path):
    from visionflow.ui.operator import OperatorPanel
    g, src = flow(stop_after=0, screen=basic_screen)
    panel = OperatorPanel(g.to_dict(), str(tmp_path / "hat.json"))
    before = threading.active_count()
    for _ in range(5):
        panel.start()
        t0 = time.time()
        while panel.screen_view.feed.frames < 3 and time.time() - t0 < 10:
            pump(qapp, 2)
            time.sleep(0.01)
        assert panel.screen_view.feed.frames >= 3
        panel.stop()
        assert panel.worker is None
    pump(qapp, 10)
    assert threading.active_count() <= before
    assert panel.screen_view.timer.isActive()
    panel.close()
    pump(qapp)
    assert not panel.screen_view.timer.isActive() and not panel.refresh_timer.isActive()
    _Part23a.stop_after = 12


# --- the designer -----------------------------------------------------------------------------------------------------

def test_designer_editing_undo_copy_paste(qapp):
    from visionflow.ui.screen_designer import ScreenDesigner
    g, src = flow()
    d = ScreenDesigner(None, g)
    d.ask_unsaved = lambda: "discard"
    a = d.add_widget("number", 13, 27)
    assert a["rect"][:2] == [10.0, 30.0]                       # snapped to the 10 px grid
    b = d.add_widget("lamp", 300, 100)
    assert [w["id"] for w in d.page["widgets"]] == [a["id"], b["id"]]
    assert d.set_prop(a["id"], "src", {"node": src.id, "port": "value"})
    assert d.set_prop(a["id"], "rect", [100, 100, 200, 100])
    assert d.items[0].pos().x() == 100 and d.items[0].w == 200
    n_hist = len(d.history)
    d.undo()
    assert d.page["widgets"][0]["rect"] == [10.0, 30.0, 240.0, 140.0]
    d.undo()
    assert d.page["widgets"][0]["props"]["src"] is None
    d.redo()
    d.redo()
    assert d.page["widgets"][0]["rect"] == [100.0, 100.0, 200.0, 100.0] and len(d.history) == n_hist
    # copy / paste: new ids, offset, bindings kept; paste again goes further
    d.select([a["id"]])
    assert d.copy() == 1
    new = d.paste()
    assert len(new) == 1 and new[0] not in (a["id"], b["id"])
    pasted = next(w for w in d.page["widgets"] if w["id"] == new[0])
    assert pasted["rect"][:2] == [120.0, 120.0] and pasted["props"]["src"] == {"node": src.id, "port": "value"}
    new2 = d.paste()
    assert next(w for w in d.page["widgets"] if w["id"] == new2[0])["rect"][:2] == [140.0, 140.0]
    d.undo()
    assert len(d.page["widgets"]) == 3
    # z-order
    d.select([a["id"]])
    d.z_order("front")
    assert d.page["widgets"][-1]["id"] == a["id"] and d.items[-1].zValue() == 2
    d.z_order("back")
    assert d.page["widgets"][0]["id"] == a["id"]
    d.z_order("up")
    assert d.page["widgets"][1]["id"] == a["id"]
    # align / distribute (multi-select)
    d.select([w["id"] for w in d.page["widgets"]])
    assert len(d.selected_ids()) == 3
    d.align("sol")
    assert len({w["rect"][0] for w in d.page["widgets"]}) == 1
    d.align("dikey dağıt")
    ys = sorted((w["rect"][1], w["rect"][3]) for w in d.page["widgets"])
    gaps = [ys[i + 1][0] - (ys[i][0] + ys[i][1]) for i in range(2)]
    assert abs(gaps[0] - gaps[1]) < 0.02
    # moving with the mouse is kept (and undoable)
    it = d.items[0]
    it.setPos(it.pos().x() + 37, it.pos().y())
    d.items_edited()
    assert d.page["widgets"][0]["rect"][0] == pytest.approx(it.pos().x())
    # delete
    d.select([a["id"]])
    d.delete_selected()
    assert a["id"] not in [w["id"] for w in d.page["widgets"]]
    d.undo()
    assert a["id"] in [w["id"] for w in d.page["widgets"]]
    # pages
    d.add_page()
    assert d.page_index == 1 and d.page["name"] == "Sayfa 2" and not d.items
    d.rename_page("Ayarlar")
    d.add_widget("button", action="sayfa", page="1")
    d.remove_page()
    assert len(d.screen["pages"]) == 1
    d.undo()
    assert [p["name"] for p in d.screen["pages"]] == ["Ana sayfa", "Ayarlar"]
    d.remove_page()
    d.remove_page()
    assert len(d.screen["pages"]) == 1 and "silinemez" in d.info.text()
    # unknown types can not be added; history is bounded
    assert d.add_widget("uçan_daire") is None
    d.close()


def test_designer_resize_and_property_editor(qapp):
    from visionflow.ui.screen_designer import ScreenDesigner, SourcePicker
    g, src = flow()
    d = ScreenDesigner(None, g)
    d.ask_unsaved = lambda: "discard"
    wd = d.add_widget("bar", 0, 0)
    it = d.items[0]
    it.resize_by("br", 33, 47, (0, 0, 320, 80), 10)
    assert (it.w, it.h) == (350, 130)
    it.resize_by("tl", 26, 500, (0, 0, 320, 80), 10)
    assert it.pos().x() == 30 and it.h >= 4
    d.items_edited()
    assert wd is d.page["widgets"][0] and d.page["widgets"][0]["rect"][0] == 30
    d.select([wd["id"]])
    d._build_props()
    assert "style" in d.prop_widgets and "src" in d.prop_widgets
    d.prop_widgets["style"].setCurrentText("gösterge")
    assert d.page["widgets"][0]["props"]["style"] == "gösterge"
    th = d.prop_widgets["thresholds"]
    th.setText("70=#ff0000; 30=#00ff00")
    th.editingFinished.emit()
    assert d.page["widgets"][0]["props"]["thresholds"] == [[30.0, "#00ff00"], [70.0, "#ff0000"]]
    th.setText("bozuk")
    th.editingFinished.emit()
    assert "border" in th.styleSheet() and d.page["widgets"][0]["props"]["thresholds"]
    # the source tree has blocks (also inside sub-flows), settings, variables and counters
    g2 = Graph()
    s2 = g2.add_new("_Part23a", (0, 0))
    thr = g2.add_new("Threshold", (200, 0))
    g2.connect(s2.id, "image", thr.id, "image")
    sub = sf.group(g2, [thr.id], "Maske")
    g2.add_new("SetVariable", (0, 300), name="hız")
    pick = SourcePicker(None, g2, "source")
    leaves = [json.loads(i.data(0, 0x0100)) for i in pick._all() if i.data(0, 0x0100)]
    assert {"node": f"{sub.id}/{thr.id}", "port": "image"} in leaves
    assert {"node": s2.id, "setting": "gain"} in leaves and {"var": "hız"} in leaves and {"counter": "yield"} in leaves
    assert pick.select_ref({"node": s2.id, "port": "ok"}) and pick.ref() == {"node": s2.id, "port": "ok"}
    pick.filter.setText("nok_count")
    visible = [i for i in pick._all() if not i.isHidden() and i.data(0, 0x0100)]
    assert [json.loads(i.data(0, 0x0100))["port"] for i in visible] == ["nok_count"]
    img_pick = SourcePicker(None, g2, "image_source")
    ports = {json.loads(i.data(0, 0x0100)).get("port") for i in img_pick._all() if i.data(0, 0x0100)}
    assert "image" in ports and "ok" not in ports
    tgt = SourcePicker(None, g2, "target")
    tgt.var_edit.setText("yeni")
    assert tgt.ref() == {"var": "yeni"}
    for p in (pick, img_pick, tgt):
        p.close()
    d.close()


def test_designer_preview_and_apply_in_editor(qapp, monkeypatch):
    from visionflow.ui.main_window import MainWindow
    win = MainWindow()
    win._confirm_discard = lambda: True
    try:
        g, src = flow(stop_after=0)
        win.scene.load_graph(g)
        win.history.reset(win._snap())
        d = win.open_screen_designer()
        assert win.open_screen_designer() is d             # one window
        d.ask_unsaved = lambda: "discard"
        d.add_widget("number", 0, 0, src={"node": src.id, "port": "value"})
        assert d.dirty()
        # live preview while the editor runs the flow
        win.start()
        view = d.toggle_preview(True)
        assert win.worker.screen is view.feed and d.add_widget("lamp") is None
        t0 = time.time()
        while view.feed.frames < 3 and time.time() - t0 < 10:
            pump(qapp, 2)
            time.sleep(0.01)
        view.poll()
        assert view.items[0].value is not None
        d.toggle_preview(False)
        assert win.worker.screen is None and d.preview is None
        win.stop()
        assert d.apply()
        op = win.root.operator
        assert op["layout"] == "screen" and op["screen"]["pages"][0]["widgets"][0]["props"]["src"]["node"] == src.id
        assert not d.dirty()
        win.undo()                                         # the editor's undo takes the screen back out
        assert not (win.root.operator or {}).get("screen")
        win.redo()
        assert win.root.operator["screen"]["pages"][0]["widgets"]
        # the operator settings dialog keeps the screen and can switch the layout
        from visionflow.ui.operator import OperatorConfigDialog
        dlg = OperatorConfigDialog(None, win.root)
        assert dlg.layout_combo.isEnabled()
        dlg.layout_combo.setCurrentIndex(0)
        cfg = dlg.result_config()
        assert cfg["layout"] == "auto" and cfg["screen"]["pages"][0]["widgets"]
        d.close()
        # the menu entry exists
        texts = [a.text() for m in win.menuBar().actions() if m.menu() for a in m.menu().actions()]
        assert any("Ekran tasarımcısı" in t for t in texts)
    finally:
        win.stop()
        win.close()
        pump(qapp)


def test_designer_open_panel_from_editor(qapp, monkeypatch):
    """The editor's operator panel uses the designed screen, and settings changed there reach the editor."""
    from visionflow.ui.main_window import MainWindow
    win = MainWindow()
    win._confirm_discard = lambda: True
    try:
        g, src = flow(stop_after=0, screen=basic_screen)
        win.scene.load_graph(g)
        win.history.reset(win._snap())
        panel = win.open_operator()
        assert panel.screen_view is not None and panel.worker is not None
        assert panel.screen_view.edit_input(panel.screen_view.item("inp"), "3")
        assert win.root.nodes[src.id].values["gain"] == 3.0
        assert panel.leave()
        assert panel.worker is None
    finally:
        win.stop()
        win.close()
        pump(qapp)


def test_designer_drop_from_palette_and_unknown_widgets_survive(qapp):
    from PySide6.QtCore import QMimeData, QPointF, Qt
    from PySide6.QtGui import QDropEvent
    from visionflow.ui.screen_designer import MIME, ScreenDesigner
    g, src = flow()
    screen = {"pages": [{"name": "A", "widgets": [{"type": "gelecek_öğe", "id": "z", "rect": [0, 0, 50, 50],
                                                    "props": {"x": [1, 2]}}]}]}
    g.operator = clean_operator({"screen": screen}, g)
    d = ScreenDesigner(None, g)
    d.ask_unsaved = lambda: "discard"
    d.resize(1200, 800)
    d.show()
    pump(qapp)
    assert d.items[0].problem and "tanınmıyor" in d.info.text()
    mime = QMimeData()
    mime.setData(MIME, b"lamp")
    target = d.canvas.mapFromScene(QPointF(400, 300))
    ev = QDropEvent(QPointF(target), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    d.canvas.dropEvent(ev)
    lamp = d.page["widgets"][-1]
    assert lamp["type"] == "lamp" and abs(lamp["rect"][0] - 400) <= 10 and abs(lamp["rect"][1] - 300) <= 10
    assert d.apply()
    saved = g.operator["screen"]["pages"][0]["widgets"]
    assert saved[0] == {"type": "gelecek_öğe", "id": "z", "rect": [0.0, 0.0, 50.0, 50.0], "props": {"x": [1, 2]}}
    assert g.operator["layout"] == "auto"              # the flow's choice is kept (it had the automatic panel)
    # preview without a running flow: buttons work on shared variables, settings stay out of the flow
    d.add_widget("button", 0, 0, action="değişken", target={"var": "önizleme"}, on_value="5")
    view = d.toggle_preview(True)
    btn = view.items[-1]
    view.press(btn)
    assert connect.VARS["önizleme"] == 5
    view.release(btn)
    d.toggle_preview(False)
    d.close()
