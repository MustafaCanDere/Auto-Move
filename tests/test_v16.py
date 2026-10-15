"""1.6: industrial cameras (Basler pypylon, GenICam/Harvesters) against fake SDKs, and NoNewFrame in the graph."""
import os
import types

import cv2
import numpy as np
import pytest

from visionflow.core import Graph, Node, NoNewFrame, PortType, create, load_builtin, register
from visionflow.core.registry import NODE_REGISTRY
from visionflow.nodes import machine_vision as mv
from visionflow.nodes.machine_vision import TRIGGERS, IndustrialCamera, to_bgr

load_builtin()


# --- a fake Basler pypylon ------------------------------------------------------------------------------------

class FakeFeature:
    def __init__(self, cam, name):
        self.cam, self.name = cam, name

    def SetValue(self, v):
        if self.name in self.cam.readonly:
            raise RuntimeError("not writable")
        self.cam.features[self.name] = v

    def Execute(self):
        self.cam.features[self.name] = self.cam.features.get(self.name, 0) + 1
        self.cam.pending += 1


class FakeResult:
    def __init__(self, arr, ok=True):
        self.arr, self.ok, self.released = arr, ok, False

    def IsValid(self):
        return self.arr is not None

    def GrabSucceeded(self):
        return self.ok

    def GetErrorDescription(self):
        return "paket kaybı"

    def Release(self):
        self.released = True


def make_pylon(serials=("22001", "22002"), frames=None):
    world = types.SimpleNamespace(cams=[], frames=frames)

    class Info:
        def __init__(self, s):
            self.s = s

        def GetSerialNumber(self):
            return self.s

        def GetModelName(self):
            return "acA1920-40gc"

    class Cam:
        known = {"TriggerSelector", "TriggerMode", "TriggerSource", "TriggerSoftware", "AcquisitionMode",
                 "ExposureAuto", "ExposureTime", "GainAuto", "Gain"}

        def __init__(self, info):
            self.info, self.features, self.readonly, self.pending = info, {}, set(), 0
            self.opened = self.grabbing = False
            self.closed = 0
            world.cams.append(self)

        def __getattr__(self, name):
            if name in Cam.known:
                return FakeFeature(self, name)
            raise RuntimeError(f"node {name} not existing")      # pypylon does not raise AttributeError

        def Open(self):
            self.opened = True

        def Close(self):
            self.closed += 1

        def StartGrabbing(self, strategy):
            self.grabbing = True

        def StopGrabbing(self):
            self.grabbing = False

        def IsGrabbing(self):
            return self.grabbing

        def RetrieveResult(self, ms, handling):
            if self.features.get("TriggerMode") == "On" and not self.pending:
                return FakeResult(None)
            self.pending = max(0, self.pending - 1)
            if world.frames:
                return world.frames.pop(0)
            return FakeResult(np.full((4, 6, 3), 50, np.uint8))

    class Factory:
        def EnumerateDevices(self):
            return [Info(s) for s in serials]

        def CreateDevice(self, info):
            return info

    class Converter:
        def Convert(self, res):
            return types.SimpleNamespace(GetArray=lambda: res.arr)

    factory = Factory()
    mod = types.SimpleNamespace(
        TlFactory=types.SimpleNamespace(GetInstance=lambda: factory), InstantCamera=Cam,
        ImageFormatConverter=Converter, PixelType_BGR8packed=1, OutputBitAlignment_MsbAligned=2,
        GrabStrategy_LatestImageOnly=3, TimeoutHandling_Return=4)
    return mod, world


@pytest.fixture
def pylon(monkeypatch):
    mod, world = make_pylon()
    monkeypatch.setattr(mv._PylonCamera, "module", mod)
    return world


def cam_node(**values):
    n = create("IndustrialCamera")
    n.values.update(values)
    return n


def test_registered_and_count():
    assert "IndustrialCamera" in NODE_REGISTRY and IndustrialCamera.is_source
    assert len(NODE_REGISTRY) >= 126


def test_free_run_with_exposure_and_gain(pylon, capsys):
    n = cam_node(serial="22002", exposure=5000.0, gain=3.0)
    assert n.safe_setup(), n.setup_error
    cam = pylon.cams[0]
    assert cam.info.GetSerialNumber() == "22002" and cam.grabbing
    assert cam.features["TriggerMode"] == "Off" and cam.features["ExposureTime"] == 5000.0
    assert cam.features["ExposureAuto"] == "Off" and cam.features["Gain"] == 3.0
    out = n.process({"trigger": None})
    assert out["image"].shape == (4, 6, 3)
    n.values["exposure"] = 800.0                         # changes while running, without reconnecting
    n.process({"trigger": None})
    assert cam.features["ExposureTime"] == 800.0 and len(pylon.cams) == 1
    assert "bağlandı" in capsys.readouterr().out
    n.teardown()
    assert cam.closed == 1 and not cam.grabbing
    n.teardown()
    assert cam.closed == 1


def test_camera_defaults_are_left_alone(pylon):
    n = cam_node()                                       # exposure 0 / gain -1: the camera's own settings
    assert n.safe_setup()
    f = pylon.cams[0].features
    assert "ExposureTime" not in f and "Gain" not in f and "ExposureAuto" not in f
    assert pylon.cams[0].info.GetSerialNumber() == "22001"   # empty serial: the first camera


def test_pick_by_index_and_unknown_serial(pylon):
    n = cam_node(serial="1")
    assert n.safe_setup()
    assert pylon.cams[-1].info.GetSerialNumber() == "22002"
    n = cam_node(serial="999")
    assert not n.safe_setup()
    assert "999" in n.setup_error and "22001, 22002" in n.setup_error


def test_no_camera_and_missing_package(monkeypatch):
    mod, _ = make_pylon(serials=())
    monkeypatch.setattr(mv._PylonCamera, "module", mod)
    n = cam_node()
    assert not n.safe_setup() and "kamera bulunamadı" in n.setup_error
    monkeypatch.setattr(mv._PylonCamera, "module", None)
    monkeypatch.setitem(__import__("sys").modules, "pypylon", None)
    n = cam_node()
    assert not n.safe_setup() and "pip install pypylon" in n.setup_error


def test_software_trigger(pylon):
    n = cam_node(trigger_mode=TRIGGERS[1])
    assert n.safe_setup()
    cam = pylon.cams[0]
    assert cam.features["TriggerMode"] == "On" and cam.features["TriggerSource"] == "Software"
    assert n.process({"trigger": None})["image"] is not None        # not connected: every frame
    with pytest.raises(NoNewFrame):
        n.process({"trigger": False})
    n.process({"trigger": True})                                     # rising edge: one picture
    with pytest.raises(NoNewFrame):
        n.process({"trigger": True})                                 # still on: no second picture
    assert cam.features["TriggerSoftware"] == 2


def test_hardware_trigger_waits_without_error(pylon):
    n = cam_node(trigger_mode=TRIGGERS[3])
    assert n.safe_setup()
    cam = pylon.cams[0]
    assert cam.features["TriggerSource"] == "Line1"
    with pytest.raises(NoNewFrame, match="donanım tetiği"):
        n.process({})
    cam.pending = 1                                                  # a pulse on Line1
    assert n.process({})["image"].shape == (4, 6, 3)
    assert n.state["cam"].cam is cam and len(pylon.cams) == 1         # waiting never reconnects


def test_bad_frame_and_timeouts_reconnect(pylon):
    n = cam_node()
    assert n.safe_setup()
    pylon.frames = [FakeResult(np.zeros((2, 2, 3), np.uint8), ok=False)]
    with pytest.raises(RuntimeError, match="paket kaybı"):
        n.process({})
    assert n.state["cam"] is None and pylon.cams[0].closed == 1
    assert n.process({})["image"] is not None                        # reconnected on the next frame
    assert len(pylon.cams) == 2
    pylon.frames = [FakeResult(None)] * 3
    for _ in range(3):
        with pytest.raises(RuntimeError, match="kare gelmedi"):
            n.process({})
    assert n.state["cam"] is None                                    # three silent frames: reconnect


def test_readonly_feature_warns(pylon, monkeypatch, capsys):
    orig = make_pylon()[0].InstantCamera

    class RO(orig):
        def __init__(self, info):
            super().__init__(info)
            self.readonly = {"ExposureTime"}
    monkeypatch.setattr(mv._PylonCamera.module, "InstantCamera", RO)
    n = cam_node(exposure=100.0)
    assert n.safe_setup()
    assert "pozlama ayarını kabul etmedi" in capsys.readouterr().out


# --- a fake Harvesters (GenICam) --------------------------------------------------------------------------------

def make_harvesters(devices, frame, fmt="Mono8", ch=1):
    log = []

    class Node:
        def __init__(self, nm, name):
            self.nm, self.name = nm, name

        @property
        def value(self):
            return self.nm.values.get(self.name)

        @value.setter
        def value(self, v):
            self.nm.values[self.name] = v

        def execute(self):
            log.append(("execute", self.name))

    class NodeMap:
        names = {"TriggerSelector", "TriggerMode", "TriggerSource", "TriggerSoftware", "AcquisitionMode",
                 "ExposureAuto", "ExposureTimeAbs", "GainAuto", "Gain"}

        def __init__(self):
            self.values = {}

        def __getattr__(self, name):
            if name in NodeMap.names:
                return Node(self, name)
            raise AttributeError(name)

    class TimeoutException(Exception):
        pass

    class Buffer:
        def __init__(self):
            comp = types.SimpleNamespace(data=frame.reshape(-1), height=frame.shape[0], width=frame.shape[1],
                                         num_components_per_pixel=ch, data_format=fmt)
            self.payload = types.SimpleNamespace(components=[comp])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            log.append(("queue",))

    class IA:
        def __init__(self, index):
            self.remote_device = types.SimpleNamespace(node_map=NodeMap())
            self.timeouts = 0
            log.append(("create", index))

        def start(self):
            log.append(("start",))

        def fetch(self, timeout):
            if self.timeouts:
                self.timeouts -= 1
                raise TimeoutException()
            return Buffer()

        def stop(self):
            log.append(("stop",))

        def destroy(self):
            log.append(("destroy",))

    class Harvester:
        def __init__(self):
            self.files = []
            self.device_info_list = []

        def add_file(self, f):
            self.files.append(f)
            log.append(("cti", f))

        def update(self):
            self.device_info_list = [types.SimpleNamespace(serial_number=s, model="MV-CA050-10GC",
                                                           vendor="Hikrobot") for s in devices]

        def create(self, index):
            self.ia = IA(index)
            log.append(("ia", self.ia))
            return self.ia

        def reset(self):
            log.append(("reset",))

    return types.SimpleNamespace(Harvester=Harvester), log


@pytest.fixture
def cti(tmp_path):
    f = tmp_path / "MvProducerGEV.cti"
    f.write_bytes(b"")
    return str(f)


def test_genicam_bayer_camera(monkeypatch, cti):
    raw = np.zeros((4, 4), np.uint8)
    raw[0::2, 0::2] = 200                                      # red pixels of an RG pattern
    mod, log = make_harvesters(["K001", "K002"], raw, "BayerRG8")
    monkeypatch.setattr(mv._GenICamCamera, "module", mod)
    n = cam_node(backend=mv.BACKENDS[1], serial="K002", cti=cti, exposure=1200.0, trigger_mode=TRIGGERS[2])
    assert n.safe_setup(), n.setup_error
    assert ("cti", cti) in log and ("create", 1) in log and ("start",) in log
    ia = next(e[1] for e in log if e[0] == "ia")
    vals = ia.remote_device.node_map.values
    assert vals["ExposureTimeAbs"] == 1200.0                  # older cameras: the ...Abs name
    assert vals["TriggerMode"] == "On" and vals["TriggerSource"] == "Line0"
    img = n.process({})["image"]
    assert img.shape == (4, 4, 3)
    assert img[..., 2].mean() > img[..., 0].mean()            # red stays red after demosaicing
    ia.timeouts = 1
    with pytest.raises(NoNewFrame):                           # hardware trigger: a time-out just means "no pulse"
        n.process({})
    n.teardown()
    assert log[-3:] == [("stop",), ("destroy",), ("reset",)]


def test_genicam_timeout_in_free_run(monkeypatch, cti):
    mod, log = make_harvesters(["K001"], np.zeros((2, 3), np.uint8))
    monkeypatch.setattr(mv._GenICamCamera, "module", mod)
    n = cam_node(backend=mv.BACKENDS[1], cti=cti)
    assert n.safe_setup()
    next(e[1] for e in log if e[0] == "ia").timeouts = 1
    with pytest.raises(RuntimeError, match="kare gelmedi"):
        n.process({})
    assert n.process({})["image"].shape == (2, 3, 3)


def test_genicam_cti_search(monkeypatch, tmp_path):
    mod, log = make_harvesters(["K001"], np.zeros((2, 2), np.uint8))
    monkeypatch.setattr(mv._GenICamCamera, "module", mod)
    monkeypatch.setattr(mv, "cti_files", lambda: [])
    n = cam_node(backend=mv.BACKENDS[1])
    assert not n.safe_setup() and "MVS" in n.setup_error
    n = cam_node(backend=mv.BACKENDS[1], cti=str(tmp_path / "yok.cti"))
    assert not n.safe_setup() and "yok.cti" in n.setup_error
    assert not any(e[0] == "cti" for e in log)


def test_cti_files_found_on_gentl_path(monkeypatch, tmp_path):
    (tmp_path / "a.cti").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"")
    monkeypatch.setenv("GENICAM_GENTL64_PATH", str(tmp_path) + os.pathsep + str(tmp_path))
    found = mv.cti_files()
    assert found.count(str(tmp_path / "a.cti")) == 1 and not any(f.endswith(".txt") for f in found)


def test_list_cameras(monkeypatch, cti, pylon):
    mod, log = make_harvesters(["K001"], np.zeros((2, 2), np.uint8))
    monkeypatch.setattr(mv._GenICamCamera, "module", mod)
    assert mv.list_cameras(mv.BACKENDS[0])[1] == {"serial": "22002", "model": "acA1920-40gc", "vendor": "Basler"}
    assert mv.list_cameras(mv.BACKENDS[1], cti) == [{"serial": "K001", "model": "MV-CA050-10GC",
                                                     "vendor": "Hikrobot"}]
    assert log[-1] == ("reset",)


@pytest.mark.parametrize("fmt,data,expect", [
    ("Mono8", np.full((2, 2), 77, np.uint8), 77),
    ("Mono12", np.full((2, 2), 4095, np.uint16), 255),
    ("Mono10", np.full((2, 2), 512, np.uint16), 128),
    ("RGB8", np.dstack([np.full((2, 2), 9, np.uint8), np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8)]), None),
])
def test_to_bgr(fmt, data, expect):
    img = to_bgr(data, fmt)
    assert img.shape == (2, 2, 3) and img.dtype == np.uint8
    if expect is None:
        assert img[0, 0].tolist() == [0, 0, 9]
    else:
        assert int(img[0, 0, 0]) == expect


def test_to_bgr_rejects_packed_and_unknown():
    with pytest.raises(RuntimeError, match="Sıkıştırılmış"):
        to_bgr(np.zeros((2, 3), np.uint8), "Mono12p")
    with pytest.raises(RuntimeError, match="Desteklenmeyen"):
        to_bgr(np.zeros((2, 2), np.uint8), "YUV422_8")
    assert to_bgr(np.zeros((2, 2, 3), np.uint8), "BGR8").shape == (2, 2, 3)
    bayer = to_bgr(np.zeros((4, 4), np.uint8), "BayerGB8")
    assert bayer.shape == (4, 4, 3)
    assert cv2.COLOR_BayerGR2BGR == mv._BAYER["BayerGB"]


# --- NoNewFrame in the graph ------------------------------------------------------------------------------------

@register
class _WaitSource16(Node):
    type_name = "_WaitSource16"
    outputs = [("n", PortType.NUMBER)]
    is_source = True
    every = 2                    # one real frame in `every` polls

    def setup(self):
        self.state["i"] = 0

    def process(self, inputs):
        self.state["i"] += 1
        if self.state["i"] % self.every:
            raise NoNewFrame("bekliyor")
        return {"n": self.state["i"]}


@register
class _Count16(Node):
    type_name = "_Count16"
    inputs = [("n", PortType.NUMBER)]
    outputs = [("count", PortType.NUMBER)]

    def setup(self):
        self.state["c"] = 0

    def process(self, inputs):
        self.state["c"] += 1
        return {"count": self.state["c"]}


def test_graph_skips_blocks_after_a_waiting_source():
    g = Graph()
    s, c = create("_WaitSource16"), create("_Count16")
    g.add(s)
    g.add(c)
    g.connect(s.id, "n", c.id, "n")
    g.setup()
    r1 = g.run_once()
    assert r1[s.id].waiting == "bekliyor" and not r1[s.id].error
    assert r1[c.id].waiting and not r1[c.id].error and c.state["c"] == 0      # not run on an old value
    r2 = g.run_once()
    assert not r2[s.id].waiting and r2[c.id].outputs == {"count": 1}
    g.teardown()


def test_worker_keeps_last_picture_while_waiting(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from visionflow.core.graph import NodeResult
    from visionflow.ui.worker import FlowWorker
    g = Graph()
    s, c = create("_WaitSource16"), create("_Count16")
    g.add(s)
    g.add(c)
    w = FlowWorker.__new__(FlowWorker)
    sent = []
    w.frame = types.SimpleNamespace(emit=lambda payload, fps: sent.append(payload))
    w.watch_id, w.thumbs = None, True
    w._emit(g, {s.id: NodeResult({}, 1.0, waiting="donanım tetiği bekleniyor"),
                c.id: NodeResult({}, 0.0, waiting="önceki blok bekliyor")}, 0.0)
    assert sent == [{s.id: (1.0, None, "donanım tetiği bekleniyor", None)}]


# --- the finder window -----------------------------------------------------------------------------------------

def test_camera_finder_and_menu(pylon, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from visionflow.ui.main_window import MainWindow
    win = MainWindow()
    win._confirm_discard = lambda: True
    dlg = win.open_camera_finder()
    assert win.open_camera_finder() is dlg

    def search():
        dlg.find()
        if dlg.thread is not None:
            dlg.thread.join(5)
        app.processEvents()
    search()
    assert dlg.list.topLevelItemCount() == 2 and dlg.btn_find.isEnabled()
    dlg.list.setCurrentItem(dlg.list.topLevelItem(1))
    dlg.add()
    cams = [n for n in win.scene.graph.nodes.values() if n.type_name == "IndustrialCamera"]
    assert len(cams) == 1 and cams[0].values["serial"] == "22002"
    monkeypatch.setattr(mv._PylonCamera, "module", make_pylon(serials=())[0])
    search()
    assert dlg.list.topLevelItemCount() == 0 and "bulunamadı" in dlg.info.text()
    monkeypatch.setattr(mv._PylonCamera, "module", None)
    monkeypatch.setitem(__import__("sys").modules, "pypylon", None)
    search()
    assert "pip install pypylon" in dlg.info.text()
    dlg.backend.setCurrentIndex(1)                       # GenICam while a flow runs: refused, nothing searched
    dlg.running = lambda: True
    dlg.thread = None
    dlg.find()
    assert dlg.thread is None and "durdurun" in dlg.info.text()
    dlg.reject()
    win.close()
    app.processEvents()


# --- fixes from the pre-release bug hunt ------------------------------------------------------------------------

def test_refused_trigger_is_an_error(pylon, monkeypatch):
    orig = mv._PylonCamera.module.InstantCamera

    class NoLine0(orig):
        def __init__(self, info):
            super().__init__(info)
            self.readonly = {"TriggerSource"}
    monkeypatch.setattr(mv._PylonCamera.module, "InstantCamera", NoLine0)
    n = cam_node(trigger_mode=TRIGGERS[2])
    assert not n.safe_setup() and "Line0" in n.setup_error
    assert all(c.closed == 1 for c in pylon.cams)          # the camera is not left open


def test_raw_gain_on_older_basler(pylon, monkeypatch, capsys):
    orig = mv._PylonCamera.module.InstantCamera

    class Raw:
        def __init__(self, cam):
            self.cam = cam

        def SetValue(self, v):
            self.cam.features["GainRaw"] = v

        def GetMin(self):
            return 0

        def GetMax(self):
            return 500

    class Old(orig):
        def __init__(self, info):
            super().__init__(info)
            self.readonly = {"Gain"}

        def __getattr__(self, name):
            if name == "GainRaw":
                return Raw(self)
            return orig.__getattr__(self, name)
    monkeypatch.setattr(mv._PylonCamera.module, "InstantCamera", Old)
    n = cam_node(gain=6.0)
    assert n.safe_setup()
    assert pylon.cams[0].features["GainRaw"] == round(6.0 / mv.RAW_GAIN_DB)
    n.values["gain"] = 40.0
    n.process({})
    assert pylon.cams[0].features["GainRaw"] == 500        # kept inside the camera's range
    assert "GainRaw" in capsys.readouterr().out


def test_software_trigger_throws_away_a_late_picture(pylon):
    n = cam_node(trigger_mode=TRIGGERS[1])
    assert n.safe_setup()
    cam = pylon.cams[0]
    stale, fresh = FakeResult(np.zeros((4, 6, 3), np.uint8)), FakeResult(np.full((4, 6, 3), 9, np.uint8))
    cam.pending = 1                                        # the answer to an earlier, timed-out trigger
    pylon.frames = [stale, fresh]
    assert int(n.process({"trigger": None})["image"][0, 0, 0]) == 9
    assert stale.released


def test_feedback_keeps_last_value_while_waiting():
    from visionflow.core.graph import NodeResult
    g = Graph()
    s, c = create("_WaitSource16"), create("_Count16")
    g.add(s)
    g.add(c)
    g.connect(s.id, "n", c.id, "n")
    g.setup()
    g._previous = {c.id: NodeResult({"count": 5}, 0.0)}
    g.run_once()                                           # waiting frame
    assert g._previous[c.id].outputs == {"count": 5}
    g.run_once()
    assert g._previous[c.id].outputs == {"count": 1}


def test_waiting_polls_are_not_frames():
    g = Graph()
    s, c = create("_WaitSource16"), create("_Count16")
    g.add(s)
    g.add(c)
    g.connect(s.id, "n", c.id, "n")
    seen = []
    assert g.run(max_frames=3, on_frame=lambda n, r: seen.append(n)) == 3
    assert seen == [1, 2, 3] and s.state["i"] == 6           # six polls, three real frames


def test_worker_restarts_a_block_once_per_batch(pylon):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from visionflow.ui.worker import FlowWorker
    mod = mv._PylonCamera.module
    g = Graph()
    n = cam_node(serial="999")                             # fails to start
    g.add(n)
    assert not n.safe_setup()
    w = FlowWorker(g.to_dict())
    failed = []
    w.failed.connect(failed.append)
    calls = []
    real = mod.TlFactory.GetInstance

    def counting():
        calls.append(1)
        return real()
    mod.TlFactory.GetInstance = counting
    for v in (100.0, 200.0, 300.0, 400.0):
        w.set_param(n.id, "exposure", v)
    w._apply_pending(g)
    assert len(failed) == 1 and n.values["exposure"] == 400.0
    assert len(calls) <= 2                                 # one start attempt, not four


def test_worker_shows_a_frame_that_came_between_emits(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from visionflow.ui import worker as wk
    monkeypatch.setattr(_WaitSource16, "every", 5)       # a trigger every 5 polls, a screen update every 3
    g = Graph()
    s, c = create("_WaitSource16"), create("_Count16")
    g.add(s)
    g.add(c)
    g.connect(s.id, "n", c.id, "n")
    w = wk.FlowWorker(g.to_dict())
    w.limiter = wk.FrameLimiter(0)
    sent = []
    monkeypatch.setattr(w, "_emit", lambda graph, results, fps: sent.append(
        {graph.nodes[k].type_name: (v.waiting, v.outputs) for k, v in results.items()}))
    clock = iter(i * 0.02 for i in range(10 ** 6))        # 20 ms per frame, emits every 50 ms
    monkeypatch.setattr(wk, "time", types.SimpleNamespace(perf_counter=lambda: next(clock), sleep=lambda s: None))
    orig = wk.Graph.run_once
    runs = []

    def counted(self):
        runs.append(1)
        if len(runs) > 40:
            w.stop()
        return orig(self)
    monkeypatch.setattr(wk.Graph, "run_once", counted)
    w.run()
    counts = sorted({v[1]["count"] for p in sent for k, v in p.items() if k == "_Count16" and not v[0]})
    assert counts == list(range(1, counts[-1] + 1)) and counts[-1] >= 5   # every counted frame was shown
