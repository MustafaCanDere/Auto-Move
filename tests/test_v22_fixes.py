"""2.2 fixes: bugs at the seams of the four 2.2 features (few-shot detection, traceability archive, REST API /
OPC UA security, template gallery) found after they were merged."""
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visionflow import trace  # noqa: E402
from visionflow.core import Graph, load_builtin  # noqa: E402
from visionflow.core import node as _node  # noqa: E402

load_builtin()

KEY = "fix-anahtar-0123456789abcdef"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(_node, "FLOW_DIR", None)
    monkeypatch.setattr(_node, "FLOW_FILE", None)
    return tmp_path / "home"


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(home, qapp):
    from visionflow.ui.main_window import MainWindow
    w = MainWindow()
    w._confirm_discard = lambda: True
    yield w
    panel = w.operator_panel()
    if panel is not None:
        panel.leave(open_editor=False)
    w.stop()
    w.close()
    qapp.processEvents()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def call(url, method="GET", body=None, key=KEY):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-API-Key", key)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def pump_call(app, *args, **kw):
    """An HTTP request from another thread while the Qt event loop runs (the API's GUI calls need it)."""
    box = {}
    t = threading.Thread(target=lambda: box.update(r=call(*args, **kw)), daemon=True)
    t.start()
    end = time.monotonic() + 40
    while t.is_alive() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    t.join(1)
    return box["r"]


def pump_until(app, cond, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def recipe_flow(archive=None, selector="A"):
    """TestPattern → (optional) Parçayı Arşivle; a Number switched by recipes A/B; a 'Reçete Seç' set to A."""
    g = Graph()
    src = g.add_new("TestPattern", width=64, height=48, animate=True)
    num = g.add_new("Number", value=0.0)
    sel = g.add_new("RecipeSelect", recipe=selector) if selector is not None else None
    arc = None
    if archive is not None:
        arc = g.add_new("ArchivePart", folder=str(archive), images="Hiçbiri")
        g.connect(src.id, "image", arc.id, "image")
    g.recipes = [{"name": "A", "values": {num.id: {"value": 1.0}}},
                 {"name": "B", "values": {num.id: {"value": 2.0}}}]
    return g, src, num, sel, arc


# ------------------------------------------------------------------------------------------- API ↔ operator panel
def test_api_controls_the_operator_panel_not_a_second_copy(win, qapp):
    """The operator panel runs the flow with the editor hidden: 'baslat' used to start a second copy of the flow
    in the hidden editor (two cameras, double PLC writes) and the API never saw the panel's run."""
    from visionflow.ui.api_dialog import manager
    g, src, num, sel, arc = recipe_flow(selector=None)
    win._set_root(Graph.from_dict(g.to_dict()))
    mgr = manager(win)
    url = mgr.start({"enabled": True, "port": _free_port(), "key": KEY, "rate": 0})
    try:
        panel = win.open_operator()
        assert panel is not None and panel.worker is not None
        assert panel.worker.api is mgr.state                      # the panel's run is reported
        assert pump_until(qapp, lambda: mgr.state.running and mgr.state.frames > 2)
        code, d = pump_call(qapp, f"{url}/baslat", "POST")
        assert code == 200 and "zaten" in d["mesaj"] and win.worker is None
        code, d = pump_call(qapp, f"{url}/durdur", "POST")
        assert code == 200 and panel.worker is None and win.worker is None and d["calisiyor"] is False
        code, d = pump_call(qapp, f"{url}/baslat", "POST")
        assert code == 200 and panel.worker is not None and win.worker is None
        # a recipe from the API reaches the flow the panel runs
        code, d = pump_call(qapp, f"{url}/recete", "POST", body={"ad": "B"})
        assert code == 200 and d["recete"] == "B"
        assert pump_until(qapp, lambda: call(f"{url}/degerler")[1]["bloklar"].get(num.id, {})
                          .get("cikislar", {}).get("value") == 2.0)
        assert panel.graph_data["nodes"][[n["id"] for n in panel.graph_data["nodes"]].index(num.id)]["values"][
            "value"] == 2.0                                        # and a restart of the panel keeps it
        # switched off while the panel runs: the panel's worker stops reporting
        mgr.stop()
        assert panel.worker.api is None
    finally:
        mgr.stop()


# ------------------------------------------------------------------------------------------- recipes ↔ archive
def test_archive_records_recipe_switched_from_outside(tmp_path, home):
    """'Parçayı Arşivle' took the recipe from the 'Reçete Seç' block's state, which stays on its own recipe when
    the API / Reçeteler window switches the running flow: parts were archived under the wrong recipe."""
    g, src, num, sel, arc = recipe_flow(tmp_path / "arsiv")
    g.setup()
    try:
        g.run_once()
        g.request_recipe("B")                     # what the API (headless and editor) does
        g.run_once()
        assert g.nodes[num.id].values["value"] == 2.0
    finally:
        g.teardown()
    recs = trace.Archive(tmp_path / "arsiv").search(newest_first=False)
    assert [r["recipe"] for r in recs] == ["A", "B"]


def test_archive_records_recipe_given_at_headless_start(tmp_path, home, capsys):
    from visionflow import headless
    g, src, num, sel, arc = recipe_flow(tmp_path / "arsiv", selector=None)
    flow = tmp_path / "akis.json"
    g.save(str(flow))
    assert headless.run_flow(str(flow), frames=2, fps=0, recete="B") == 0
    recs = trace.Archive(tmp_path / "arsiv").search()
    assert recs and {r["recipe"] for r in recs} == {"B"}


def test_editor_api_recipe_survives_restart_and_reaches_running_flow(win, qapp, tmp_path):
    """Editor + API: the switch reached the running blocks only, a 'Reçete Seç' block switched back at the next
    start (headless --api keeps it), and the archive kept the old recipe."""
    from visionflow.ui.api_dialog import manager
    g, src, num, sel, arc = recipe_flow(tmp_path / "arsiv")
    win._set_root(Graph.from_dict(g.to_dict()))
    mgr = manager(win)
    url = mgr.start({"enabled": True, "port": _free_port(), "key": KEY, "rate": 0})
    try:
        win.start()
        assert pump_until(qapp, lambda: mgr.state.recipe == "A" and mgr.state.frames > 2)
        code, d = pump_call(qapp, f"{url}/recete", "POST", body={"ad": "B"})
        assert code == 200 and d["recete"] == "B"
        assert win.root.nodes[sel.id].values["recipe"] == "B"
        assert pump_until(qapp, lambda: mgr.state.recipe == "B")   # the running flow reported the switch
        frames = mgr.state.frames
        assert pump_until(qapp, lambda: mgr.state.frames > frames + 3)
        win.stop()
        win.start()                                                 # restart: stays on B
        assert pump_until(qapp, lambda: mgr.state.frames > 2)
        assert mgr.state.recipe == "B" and win.root.nodes[num.id].values["value"] == 2.0
        win.stop()
    finally:
        mgr.stop()
    recs = trace.Archive(tmp_path / "arsiv").search(limit=5)
    assert recs and recs[0]["recipe"] == "B"


def test_reçeteler_window_switch_while_running_goes_through_flow(win, qapp):
    """MainWindow.apply_recipe on a running flow: the flow itself switches (API state follows)."""
    g, src, num, sel, arc = recipe_flow(selector=None)
    win._set_root(Graph.from_dict(g.to_dict()))
    from visionflow.ui.api_dialog import manager
    mgr = manager(win)
    mgr.start({"enabled": True, "port": 0, "key": KEY, "rate": 0})
    try:
        win.start()
        assert pump_until(qapp, lambda: mgr.state.frames > 2)
        assert win.apply_recipe("B") == []
        assert pump_until(qapp, lambda: mgr.state.recipe == "B")
        assert pump_until(qapp, lambda: mgr.state.block_values().get(num.id, {}).get("cikislar", {})
                          .get("value") == 2.0)
        win.stop()
    finally:
        mgr.stop()


def test_worker_request_recipe_unknown_name_reports(qapp):
    from visionflow.ui.worker import FlowWorker
    g, src, num, sel, arc = recipe_flow(selector=None)
    w = FlowWorker(g.to_dict())
    errors = []
    w.failed.connect(errors.append)
    g2 = Graph.from_dict(g.to_dict())
    w.request_recipe("yok")
    assert w._apply_pending(g2) is True
    assert errors and "yok" in errors[0]


# ------------------------------------------------------------------------------------------- template gallery
def test_every_example_has_gallery_text_and_category():
    """The few-shot detection example (2.2) had no gallery entry: shown as 'Az ornekle nesne bulma' (no Turkish
    letters), without description and under no category."""
    from visionflow import templates as T
    for t in T.list_templates():
        assert t["categories"] and set(t["categories"]) <= set(T.CATEGORIES), t["name"]
        assert t["description"], t["name"]
        assert t["new"] or t["name"] in T.EXAMPLE_META, t["name"]
    t = next(t for t in T.list_templates() if t["name"] == "az_ornekle_nesne_bulma")
    assert t["title"] == "Az örnekle nesne bulma" and T.matches(t, "Yapay zeka", "örnek")


# ------------------------------------------------------------------------------------------- dialog-trained models
EXAMPLE_MODEL = os.path.join(os.path.dirname(__file__), "..", "examples", "ornek_veri", "az_ornek_tespit",
                             "nesne_modeli.vfdet")


def test_teach_dialog_saves_relative_model_next_to_the_flow(tmp_path, home, qapp, monkeypatch):
    """'Kutu Çiz ve Eğit' saved a relative model path into the folder the program was started from, while the block
    looks for it next to the flow file: started from another folder (or as a service) the model was missing."""
    from visionflow import fewshot_det as fd
    from visionflow.core import create
    from visionflow.ui.det_teach_dialog import DetTeachDialog
    proj, other = tmp_path / "proje", tmp_path / "baska"
    proj.mkdir()
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setattr(_node, "FLOW_DIR", proj)
    node = create("FewShotDetect")
    node.values["model"] = "modeller/yeni"
    applied = {}
    dlg = DetTeachDialog(None, node, lambda k, v: applied.__setitem__(k, v))
    try:
        dlg.model = fd.DetModel.load(EXAMPLE_MODEL)
        dlg.classes = [[n, tuple(c)] for n, c in zip(dlg.model.classes, dlg.model.colors)]
        saved = dlg.save_model()
    finally:
        dlg.close()
    assert (proj / "modeller" / "yeni.vfdet").is_file() and not (other / "modeller").exists()
    assert saved == str(proj / "modeller" / "yeni.vfdet")
    assert applied["model"] == "modeller/yeni.vfdet"            # stays relative: the folder can be moved
    monkeypatch.chdir(tmp_path)                                  # started from anywhere: still found
    node.values["model"] = applied["model"]
    node.setup()
    assert node.process({"image": np.zeros((80, 80, 3), np.uint8)})["count"] == 0


def test_save_target_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_node, "FLOW_DIR", None)
    target, setting = _node.save_target("m/a", ".vfdet")          # no flow file: absolute
    assert target == setting == str(tmp_path / "m" / "a.vfdet")
    absolute = str(tmp_path / "x.vfseg")
    assert _node.save_target(absolute, ".vfseg") == (absolute, absolute)
    flow_dir = tmp_path / "f"
    (flow_dir / "m").mkdir(parents=True)
    (flow_dir / "m" / "var.vfdet").write_bytes(b"")
    monkeypatch.setattr(_node, "FLOW_DIR", flow_dir)
    assert _node.save_target("m/var.vfdet") == (str(flow_dir / "m" / "var.vfdet"), "m/var.vfdet")   # existing
    assert _node.save_target("m/yeni.vfdet") == (str(flow_dir / "m" / "yeni.vfdet"), "m/yeni.vfdet")
    assert _node.save_target("") == ("", "")


# ------------------------------------------------------------------------------------------- archive window
def test_archive_window_opens_the_flows_archive_folder(win, qapp, tmp_path):
    """Araçlar → İzlenebilirlik arşivi and the operator panel's 'Arşiv' button always showed the default folder:
    a line archiving to its own folder showed an empty archive."""
    from visionflow.nodes.subflow import group
    from visionflow.ui.operator import OperatorPanel
    win.open_trace()
    assert win._trace_browser.archive.root == trace.default_root()          # no archive block: the default
    win._trace_browser.close()
    w = trace.ArchiveWriter(tmp_path / "hat_arsivi")
    w.submit(serial="SN-1", ok=True)
    assert w.close()
    g, src, num, sel, arc = recipe_flow(tmp_path / "hat_arsivi", selector=None)
    group(g, [arc.id])                                                       # inside a sub-flow too
    win._set_root(Graph.from_dict(g.to_dict()))
    win.open_trace()
    br = win._trace_browser
    assert br.archive.root == tmp_path / "hat_arsivi" and br.total == 1
    br.close()
    panel = OperatorPanel(g.to_dict(), None)
    try:
        panel.open_trace()
        assert panel._trace_browser.archive.root == tmp_path / "hat_arsivi"
        panel._trace_browser.close()
    finally:
        panel._left = True
        panel.close()


def test_refused_locked_flow_does_not_take_over_flow_file(win, tmp_path, monkeypatch):
    """Opening a locked flow and giving up at the password left FLOW_DIR / FLOW_FILE on that file: the flow still
    open archived its parts under the other flow's name and looked for its relative files in the other folder."""
    from visionflow.core.operator import clean_operator, make_lock
    from visionflow.ui import operator as op
    mine, other = tmp_path / "benim", tmp_path / "kilitli"
    mine.mkdir()
    other.mkdir()
    g, *_ = recipe_flow(selector=None)
    g.save(str(mine / "hat1.json"))
    win.open_flow(str(mine / "hat1.json"))
    assert _node.FLOW_FILE == (mine / "hat1.json").resolve()
    locked = Graph()
    locked.add_new("TestPattern")
    locked.operator = clean_operator({"lock": make_lock("1234", iterations=2000)}, locked)
    locked.save(str(other / "hat2.json"))
    monkeypatch.setattr(op, "ask_password", lambda *a, **k: False)
    win.open_flow(str(other / "hat2.json"))
    assert win.path == str(mine / "hat1.json")
    assert _node.FLOW_FILE == (mine / "hat1.json").resolve() and _node.FLOW_DIR == mine.resolve()


# ------------------------------------------------------------------------------------------- OPC UA security
def test_opc_server_blocks_without_security_ignore_trust_choice(home):
    """Two 'OPC UA Sunucusu' blocks on one port without security refused to start when only their (unused)
    client-certificate choice differed."""
    pytest.importorskip("asyncua", reason="asyncua kurulu değil")
    from visionflow.nodes.plc_server import _OpcServer
    port = _free_port()
    a = _OpcServer.acquire(port, "", "", "Yok", "İlk bağlantıda güven")
    try:
        b = _OpcServer.acquire(port, "", "", "Yok", "Sadece güvenilen listesi")
        assert b is a
        b.release()
    finally:
        a.release()


# ------------------------------------------------------------------------------------------- Node.dialog hook
def test_block_dialog_follows_the_block_after_undo_and_subflow(win, qapp, tmp_path):
    """The 'Kutu Çiz ve Eğit' window kept the block object it was opened for: after an undo, a recipe switch or
    leaving / re-entering a sub-flow (new block objects) saving the model silently did not set the block, while the
    window said 'blok bu modeli kullanıyor'; the button then opened a second, empty window."""
    from visionflow import fewshot_det as fd
    from visionflow.nodes.subflow import group
    g = Graph()
    src = g.add_new("TestPattern", width=64, height=48)
    det = g.add_new("FewShotDetect")
    g.connect(src.id, "image", det.id, "image")
    sub = group(g, [det.id])
    win._set_root(Graph.from_dict(g.to_dict()))
    win.enter_subflow(win.scene.graph.nodes[sub.id])
    dlg = win.open_block_dialog(det.id)
    dlg.model = fd.DetModel.load(EXAMPLE_MODEL)
    dlg.classes = [[n, tuple(c)] for n, c in zip(dlg.model.classes, dlg.model.colors)]
    dlg.add_image(np.zeros((60, 80, 3), np.uint8), "çizilen")
    win._restore(win._snap())                                   # undo / recipe: new block objects
    first = str(tmp_path / "bir.vfdet")
    dlg.save_model(first)
    assert dlg.applied and win.scene.graph.nodes[det.id].values["model"] == first
    win.exit_subflow()
    second = str(tmp_path / "iki.vfdet")
    dlg.path_edit.setText(second)
    dlg.save_ui()
    assert not dlg.applied and "açık değil" in dlg.status.text()     # told, not silently ignored
    win.enter_subflow(win.scene.graph.nodes[sub.id])
    again = win.open_block_dialog(det.id)
    assert again is dlg and len(dlg.images) == 1 and dlg.node is win.scene.graph.nodes[det.id]
    dlg.save_model(second)
    assert dlg.applied and win.scene.graph.nodes[det.id].values["model"] == second
    win.exit_subflow()
    inner = Graph.from_dict(win.root.to_dict()).nodes[sub.id].inner_graph()
    assert inner.nodes[det.id].values["model"] == second          # written into the sub-flow
    dlg.close()
