"""2.0 (e): service / systemd / Task Scheduler, crash watchdog, installer script, frame recorder + replay, flow test."""
import glob
import json
import os
import pathlib
import re
import sys
import types

import cv2
import numpy as np
import pytest

from visionflow.core import Graph, StopFlow, create, load_builtin

load_builtin()
ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- service
@pytest.fixture
def flow_file(tmp_path):
    g = Graph()
    g.add_new("TestPattern", animate=False)
    p = tmp_path / "hat akışı.json"
    g.save(str(p))
    return p


def test_systemd_unit_text(flow_file):
    from visionflow import service
    text = service.systemd_unit(str(flow_file), "hat 1")
    assert "Restart=always" in text and "RestartSec=" in text
    exec_line = next(line for line in text.splitlines() if line.startswith("ExecStart="))
    assert "--headless" in exec_line and "--yeniden-baslat" in exec_line
    assert service._systemd_quote(str(flow_file.resolve())) in exec_line            # path with a space is quoted
    assert "WantedBy=default.target" in text and "User=" not in text
    assert "Description=VisionFlow akışı: hat_1" in text
    system = service.systemd_unit(str(flow_file), "hat", system=True, user="op")
    assert "WantedBy=multi-user.target" in system and "User=op" in system
    assert service.unit_path("hat", True) == pathlib.Path("/etc/systemd/system/visionflow-hat.service")


def test_systemd_dry_run_prints_and_writes_nothing(flow_file, tmp_path, monkeypatch):
    from visionflow import service
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    lines, calls = [], []
    code = service.main(["kur", "--akis", str(flow_file), "--yazdir"], out=lines.append,
                        runner=calls.append, platform="linux")
    out = "\n".join(lines)
    assert code == 0 and not calls
    assert "Restart=always" in out and "systemctl --user enable --now visionflow-hat_akisi.service" in out
    assert not (tmp_path / "cfg").exists()


def test_systemd_install_and_remove(flow_file, tmp_path, monkeypatch):
    from visionflow import service
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    calls = []
    runner = lambda cmd: calls.append(cmd) or types.SimpleNamespace(returncode=0)  # noqa: E731
    assert service.main(["kur", "--akis", str(flow_file), "--ad", "hat"], out=lambda *_: None, runner=runner,
                        platform="linux") == 0
    unit = tmp_path / "cfg/systemd/user/visionflow-hat.service"
    assert unit.is_file() and "Restart=always" in unit.read_text()
    assert calls == [["systemctl", "--user", "daemon-reload"],
                     ["systemctl", "--user", "enable", "--now", "visionflow-hat.service"]]
    calls.clear()
    assert service.main(["durum", "--ad", "hat"], out=lambda *_: None, runner=runner, platform="linux") == 0
    assert calls == [["systemctl", "--user", "status", "--no-pager", "visionflow-hat.service"]]
    calls.clear()
    assert service.main(["kaldir", "--ad", "hat"], out=lambda *_: None, runner=runner, platform="linux") == 0
    assert not unit.exists()
    assert calls[0][-3:] == ["disable", "--now", "visionflow-hat.service"] and calls[-1][-1] == "daemon-reload"
    # a failing systemctl is reported through the exit code
    bad = lambda cmd: types.SimpleNamespace(returncode=3)  # noqa: E731
    assert service.main(["baslat", "--ad", "hat"], out=lambda *_: None, runner=bad, platform="linux") == 3


def test_schtasks_commands(flow_file):
    from visionflow import service
    [cmd] = service.schtasks_commands("kur", "hat", str(flow_file))
    assert cmd[:4] == ["schtasks", "/Create", "/TN", "VisionFlow\\hat"]
    tr = cmd[cmd.index("/TR") + 1]
    assert f'"{flow_file.resolve()}"' in tr and "--headless" in tr and "--yeniden-baslat" in tr
    assert cmd[cmd.index("/SC") + 1] == "ONLOGON" and "/RU" not in cmd and cmd[-1] == "/F"
    [sysc] = service.schtasks_commands("kur", "hat", str(flow_file), system=True)
    assert sysc[sysc.index("/SC") + 1] == "ONSTART" and sysc[sysc.index("/RU") + 1] == "SYSTEM"
    assert service.schtasks_commands("kaldir", "hat") == [["schtasks", "/Delete", "/TN", "VisionFlow\\hat", "/F"]]
    assert service.schtasks_commands("baslat", "hat")[0][1] == "/Run"
    assert service.schtasks_commands("durdur", "hat")[0][1] == "/End"
    assert service.schtasks_commands("durum", "hat")[0][1] == "/Query"


def test_windows_without_pywin32_falls_back_to_task_scheduler(flow_file, monkeypatch):
    from visionflow import service
    monkeypatch.setitem(sys.modules, "win32serviceutil", None)      # import fails
    assert service.have_pywin32() is False
    lines, calls = [], []
    runner = lambda cmd: calls.append(cmd) or types.SimpleNamespace(returncode=0)  # noqa: E731
    code = service.main(["kur", "--akis", str(flow_file)], out=lines.append, runner=runner, platform="win32")
    assert code == 0
    assert "pip install pywin32" in lines[0]
    assert calls and calls[0][:2] == ["schtasks", "/Create"] and "VisionFlow\\hat_akisi" in calls[0]
    assert any("Görev kuruldu" in line for line in lines)
    # --yazdir with the forced method only prints
    lines.clear(), calls.clear()
    assert service.main(["kaldir", "--ad", "x", "--yontem", "gorev", "--yazdir"], out=lines.append, runner=runner,
                        platform="win32") == 0
    assert not calls and any("schtasks /Delete /TN VisionFlow\\x /F" in line for line in lines)


def test_windows_with_pywin32_installs_a_service(flow_file, monkeypatch):
    from visionflow import service
    calls = []
    fake = types.ModuleType("win32serviceutil")
    fake.InstallService = lambda cls, name, display, **kw: calls.append(("install", cls, name, kw["startType"]))
    fake.SetServiceCustomOption = lambda name, key, value: calls.append(("option", name, key, value))
    fake.StartService = lambda name: calls.append(("start", name))
    fake.StopService = lambda name: calls.append(("stop", name))
    fake.RemoveService = lambda name: calls.append(("remove", name))
    fake.QueryServiceStatus = lambda name: (0, 4)
    ws = types.ModuleType("win32service")
    ws.SERVICE_AUTO_START = 2
    monkeypatch.setitem(sys.modules, "win32serviceutil", fake)
    monkeypatch.setitem(sys.modules, "win32service", ws)
    lines = []
    assert service.main(["kur", "--akis", str(flow_file), "--ad", "hat"], out=lines.append, platform="win32") == 0
    assert calls[0][0] == "install" and calls[0][1].endswith("service.VisionFlowService")
    assert calls[0][2] == "VisionFlow_hat" and calls[0][3] == 2
    assert calls[1] == ("option", "VisionFlow_hat", "akis", str(flow_file.resolve()))
    assert calls[2] == ("start", "VisionFlow_hat")
    lines.clear()
    assert service.main(["durum", "--ad", "hat"], out=lines.append, platform="win32") == 0
    assert "çalışıyor" in lines[0]
    calls.clear()
    assert service.main(["kaldir", "--ad", "hat"], out=lines.append, platform="win32") == 0
    assert calls == [("stop", "VisionFlow_hat"), ("remove", "VisionFlow_hat")]
    # an error of the service manager (not an administrator) is a message, not a crash
    fake.StartService = lambda name: (_ for _ in ()).throw(OSError("Erişim engellendi"))
    lines.clear()
    assert service.main(["baslat", "--ad", "hat"], out=lines.append, platform="win32") == 1
    assert "Erişim engellendi" in lines[0] and "Yönetici" in lines[0]


def test_service_argument_errors(tmp_path, capsys):
    from visionflow import service
    with pytest.raises(SystemExit):
        service.parse_args(["kur"])                                   # no flow
    with pytest.raises(SystemExit):
        service.parse_args(["kur", "--akis", str(tmp_path / "yok.json")])
    with pytest.raises(SystemExit):
        service.parse_args(["durum"])                                 # neither name nor flow
    with pytest.raises(SystemExit):
        service.parse_args(["sil", "--ad", "x"])
    assert "--akis" in capsys.readouterr().err
    assert service.parse_args(["durum", "--akis", "a/b/Hat Akışı 2.json"]).ad == "Hat_Akisi_2"
    assert service.safe_name("../../etc") == "etc" and service.safe_name("!!!") == "akis"


def test_main_dispatches_service_without_qt(flow_file, monkeypatch, capsys):
    from visionflow import __main__ as entry
    monkeypatch.setattr(sys, "argv", ["VisionFlow", "servis", "kur", "--akis", str(flow_file), "--yazdir",
                                      "--yontem", "systemd"])
    assert entry.main() == 0
    assert "Restart=always" in capsys.readouterr().out


# ---------------------------------------------------------------- watchdog
def test_watchdog_restarts_with_backoff(monkeypatch, tmp_path):
    from visionflow import headless
    sleeps = []
    monkeypatch.setattr(headless, "_sleep", sleeps.append)
    runs = {"n": 0}

    def flaky():
        runs["n"] += 1
        if runs["n"] <= 3:
            raise RuntimeError(f"kamera koptu {runs['n']}")
        return 0
    log = headless.get_logger(tmp_path / "k" / "h.log")
    assert headless.watchdog(flaky, log=log) == 0
    assert runs["n"] == 4
    assert sum(sleeps) == pytest.approx(1 + 2 + 4) and max(sleeps) <= 0.5
    for h in log.handlers:
        h.flush()
    text = (tmp_path / "k" / "h.log").read_text(encoding="utf-8")
    assert "kamera koptu 1" in text and "Traceback" in text and "yeniden başlatılacak" in text


def test_watchdog_gives_up_and_stops(monkeypatch):
    from visionflow import headless
    sleeps = []
    monkeypatch.setattr(headless, "_sleep", sleeps.append)

    def boom():
        raise ValueError("bozuk")
    assert headless.watchdog(boom, max_restarts=2) == 1
    assert sum(sleeps) == pytest.approx(1 + 2)
    # the backoff is capped
    sleeps.clear()
    assert headless.watchdog(boom, max_restarts=9) == 1
    assert sum(sleeps) == pytest.approx(1 + 2 + 4 + 8 + 16 + 32 + 60 + 60 + 60)
    # a stop request during the wait ends it at once
    stop = {"flag": False}
    monkeypatch.setattr(headless, "_sleep", lambda s: stop.update(flag=True))
    assert headless.watchdog(boom, should_stop=lambda: stop["flag"]) == 1


def test_watchdog_resets_backoff_after_a_long_run(monkeypatch):
    from visionflow import headless
    sleeps, now = [], {"t": 0.0}
    monkeypatch.setattr(headless, "_sleep", sleeps.append)
    monkeypatch.setattr(headless, "_clock", lambda: now["t"])

    def long_then_crash():
        now["t"] += 1000                    # ran for a long time before crashing
        raise RuntimeError("x")
    headless.watchdog(long_then_crash, max_restarts=3)
    assert sum(sleeps) == pytest.approx(3 * 1.0)


def test_headless_restart_option_logs_to_home(tmp_path, monkeypatch, capsys):
    from visionflow import headless
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(headless, "_sleep", lambda s: None)
    code = headless.main([str(tmp_path / "olmayan.json"), "--yeniden-baslat", "--en-fazla-yeniden", "1"])
    assert code == 1
    log = tmp_path / "home" / "kayitlar" / "headless.log"
    assert log.is_file() and "Akış çöktü" in log.read_text(encoding="utf-8")
    assert "yeniden başlatılacak" in capsys.readouterr().err


def test_headless_restart_runs_a_good_flow_once(flow_file, tmp_path, monkeypatch, capsys):
    from visionflow import headless
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    assert headless.main([str(flow_file), "--frames", "2", "--yeniden-baslat"]) == 0
    assert "Bitti: 1 kare" in capsys.readouterr().out


# ---------------------------------------------------------------- installer
def iss_sections():
    text = (ROOT / "installer" / "visionflow.iss").read_text(encoding="utf-8-sig")
    sections, current = {}, None
    for line in text.splitlines():
        s = line.strip()
        m = re.fullmatch(r"\[(\w+)\]", s)
        if m:
            current = m.group(1)
            sections[current] = []
        elif current and s and not s.startswith(";"):
            sections[current].append(s)
    return text, sections


def entry(line):
    return {k: v.strip().strip('"') for k, v in re.findall(r'(\w+):\s*("(?:[^"]|"")*"|[^;]+)', line)}


def test_installer_script_structure_and_files(flow_file):
    text, sec = iss_sections()
    for name in ("Setup", "Languages", "Tasks", "Files", "Icons", "Registry", "Run", "UninstallRun", "Code"):
        assert name in sec, name
    langs = [entry(line) for line in sec["Languages"]]
    assert {lang["Name"] for lang in langs} == {"turkish", "english"}
    assert any("Turkish.isl" in lang["MessagesFile"] for lang in langs)
    tasks = {entry(line)["Name"] for line in sec["Tasks"]}
    assert {"desktopicon", "fileassoc", "autostart"} <= tasks
    assert any("{autodesktop}" in line and "desktopicon" in line for line in sec["Icons"])
    assert "turkish.FlowFileDesc=VisionFlow akışı" in text
    assert any(".json\\OpenWithProgids" in line for line in sec["Registry"])
    assert "ChangesAssociations=yes" in sec["Setup"]
    # every file taken from the repository exists (dist\ is the PyInstaller output, built first)
    defines = dict(re.findall(r'#define\s+(\w+)\s+"([^"]*)"', text))
    sources = [entry(line)["Source"] for line in sec["Files"]]
    assert any("dist\\VisionFlow\\*" in s for s in sources)
    for src in sources:
        src = re.sub(r"\{#(\w+)\}", lambda m: defines[m.group(1)], src).replace("\\", "/")
        path = (ROOT / "installer" / src).resolve()
        if path.relative_to(ROOT).parts[0] == "dist":
            continue
        assert glob.glob(str(path)), f"kurulumdaki dosya depoda yok: {src}"
    # the spec / entry point the build script uses exist
    bat = (ROOT / "installer" / "build_windows.bat").read_text()
    assert "visionflow.spec" in bat and "installer\\visionflow.iss" in bat and "ISCC" in bat
    assert (ROOT / "visionflow.spec").is_file() and (ROOT / "run_visionflow.py").is_file()
    assert b"\r\n" in (ROOT / "installer" / "build_windows.bat").read_bytes()      # cmd.exe wants CRLF
    # the autostart / uninstall commands are valid service commands
    from visionflow import service
    run = next(entry(line) for line in sec["Run"] if "servis kur" in line)
    assert run.get("Tasks") == "autostart"
    params = run["Parameters"].replace('""', '"').replace("{code:GetFlowPath}", str(flow_file))
    import shlex
    args = service.parse_args(shlex.split(params)[1:])
    assert args.islem == "kur" and args.ad == "VisionFlow"
    unrun = entry(sec["UninstallRun"][0])["Parameters"]
    assert service.parse_args(unrun.split()[1:]).islem == "kaldir"
    assert "function GetFlowPath" in text and "WizardIsTaskSelected('autostart')" in text


# ---------------------------------------------------------------- recorder / player
class Clock:
    def __init__(self):
        self.t = 100.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(round(s, 6))
        self.t += s


@pytest.fixture
def clock(monkeypatch):
    from visionflow.nodes import recording
    c = Clock()
    monkeypatch.setattr(recording, "_clock", c)
    monkeypatch.setattr(recording, "_sleep", c.sleep)
    return c


def frame(k, size=(120, 160)):
    img = np.zeros(size + (3,), np.uint8)
    for i in range(k):
        cv2.circle(img, (20 + i * 35, 60), 12, (255, 255, 255), -1)
    return img


def test_recorder_ring_buffer_and_manifest(tmp_path, clock):
    rec = create("FrameRecorder")
    rec.values.update(folder=str(tmp_path / "rec"), new_folder=False, mode="tetikte", pre=2, post=1,
                      names="sayi,durum")
    rec.setup()
    trig = [False] * 5 + [True] + [False] * 4
    outs = []
    for i, t in enumerate(trig):
        clock.t += 0.1
        outs.append(rec.process({"image": frame(1), "trigger": t, "a": i, "b": np.float32(i / 2), "c": None,
                                 "d": None}))
    rec.teardown()
    m = json.loads((tmp_path / "rec" / "manifest.json").read_text(encoding="utf-8"))
    assert m["format"] == "visionflow-kayit" and m["names"] == ["sayi", "durum", "c", "d"]
    got = [f["values"]["sayi"] for f in m["frames"]]
    assert got == [3, 4, 5, 6]                   # 2 before, the trigger frame, 1 after
    assert [o["recording"] for o in outs] == [False] * 5 + [True, True, False, False, False]
    assert m["frames"][1]["values"]["durum"] == 2.0 and isinstance(m["frames"][1]["values"]["durum"], float)
    ts = [f["t"] for f in m["frames"]]
    assert ts == pytest.approx([0.3, 0.4, 0.5, 0.6])          # capture time kept for the buffered frames
    assert all((tmp_path / "rec" / f["file"]).is_file() for f in m["frames"])
    assert outs[-1]["count"] == 4 and outs[-1]["folder"] == str(tmp_path / "rec")
    # running again into the same folder continues the recording
    rec.setup()
    rec.process({"image": frame(2), "trigger": True, "a": 99, "b": None, "c": None, "d": None})
    rec.teardown()
    m = json.loads((tmp_path / "rec" / "manifest.json").read_text(encoding="utf-8"))
    assert len(m["frames"]) == 5 and m["frames"][-1]["values"]["sayi"] == 99
    assert len({f["file"] for f in m["frames"]}) == 5 and m["frames"][-1]["t"] > m["frames"][-2]["t"]


def test_recorder_new_subfolder_each_run_and_bad_input(tmp_path, clock):
    rec = create("FrameRecorder")
    rec.values.update(folder=str(tmp_path / "r"))
    rec.setup()
    first = rec.state["folder"]
    rec.process({"image": frame(1), "trigger": None, "a": None, "b": None, "c": None, "d": None})
    rec.teardown()
    rec.setup()
    assert rec.state["folder"] != first and first.parent == rec.state["folder"].parent
    with pytest.raises(ValueError, match="resim"):
        rec.process({"image": "x", "trigger": None, "a": None, "b": None, "c": None, "d": None})
    rec.values["mode"] = "tetikte"
    with pytest.raises(ValueError, match="tetik"):
        rec.process({"image": frame(1), "trigger": None, "a": None, "b": None, "c": None, "d": None})
    # a frame without an image keeps the values
    rec.values["mode"] = "her kare"
    rec.process({"image": None, "trigger": None, "a": {"x": np.int64(3)}, "b": [1, 2], "c": object(), "d": None})
    rec.teardown()
    m = json.loads((rec.state["folder"] / "manifest.json").read_text(encoding="utf-8"))
    assert m["frames"][0]["file"] is None and m["frames"][0]["values"]["a"] == {"x": 3}
    assert m["frames"][0]["values"]["b"] == [1, 2] and isinstance(m["frames"][0]["values"]["c"], str)


def test_recorder_disk_guard_and_max_files(tmp_path, clock, monkeypatch):
    from visionflow.nodes import recording
    rec = create("FrameRecorder")
    rec.values.update(folder=str(tmp_path / "r"), new_folder=False, max_files=2, min_free_mb=0)
    rec.setup()
    inp = {"image": frame(1), "trigger": None, "a": None, "b": None, "c": None, "d": None}
    rec.process(inp)
    rec.process(inp)
    with pytest.raises(RuntimeError, match="En fazla"):
        rec.process(inp)
    with pytest.raises(RuntimeError, match="En fazla"):
        rec.process(inp)
    rec.teardown()
    assert len(list((tmp_path / "r").glob("*.png"))) == 2
    assert len(json.loads((tmp_path / "r" / "manifest.json").read_text())["frames"]) == 2
    # low disk space
    monkeypatch.setattr(recording.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=50e6))
    rec2 = create("FrameRecorder")
    rec2.values.update(folder=str(tmp_path / "r2"), new_folder=False, min_free_mb=100)
    rec2.setup()
    with pytest.raises(RuntimeError, match="Disk"):
        rec2.process(inp)
    rec2.teardown()
    assert not list((tmp_path / "r2").glob("*.png"))


@pytest.mark.parametrize("img", [
    np.full((20, 30), 7, np.uint8),
    np.full((20, 30, 3), 40000, np.uint16),
    np.dstack([np.full((20, 30), v, np.uint8) for v in (1, 2, 3, 200)]),
])
def test_record_and_replay_keep_image_types(tmp_path, clock, img):
    rec = create("FrameRecorder")
    rec.values.update(folder=str(tmp_path / "r"), new_folder=False)
    rec.setup()
    rec.process({"image": img, "trigger": None, "a": 1, "b": None, "c": None, "d": None})
    rec.teardown()
    pl = create("RecordingPlayer")
    pl.values.update(folder=str(tmp_path / "r"), speed="en hızlı")
    pl.setup()
    out = pl.process({})
    assert out["image"].dtype == img.dtype and out["image"].shape == img.shape and np.array_equal(out["image"], img)
    assert out["a"] == 1 and out["frame"] == 1


def test_recorder_converts_float_and_bool_images(tmp_path, clock):
    from visionflow.nodes.recording import to_png_image
    assert to_png_image(np.full((4, 4), 0.5, np.float32)).max() == 127
    assert to_png_image(np.full((4, 4), 300.0)).max() == 255
    assert to_png_image(np.ones((4, 4), bool)).max() == 255


def make_recording(folder, clock, counts, dt=0.5):
    rec = create("FrameRecorder")
    rec.values.update(folder=str(folder), new_folder=False, names="beklenen")
    rec.setup()
    for k in counts:
        rec.process({"image": frame(k), "trigger": None, "a": k, "b": None, "c": None, "d": None})
        clock.t += dt
    rec.teardown()
    return folder


def test_replay_timing_speed_and_loop(tmp_path, clock):
    make_recording(tmp_path / "r", clock, [1, 2, 3], dt=0.5)
    pl = create("RecordingPlayer")
    pl.values.update(folder=str(tmp_path / "r"))
    pl.setup()
    outs = [pl.process({}) for _ in range(3)]
    assert clock.sleeps == [0.5, 0.5]                     # original timing
    assert [o["a"] for o in outs] == [1, 2, 3] and [o["time"] for o in outs] == pytest.approx([0, 0.5, 1.0])
    with pytest.raises(StopFlow):
        pl.process({})
    # slow processing: no extra waiting, it catches up
    clock.sleeps.clear()
    pl.setup()
    pl.process({})
    clock.t += 2.0
    pl.process({})
    assert clock.sleeps == []
    # twice as fast
    pl.values["factor"] = 2.0
    pl.setup()
    for _ in range(3):
        pl.process({})
    assert clock.sleeps == [0.25, 0.25]
    # as fast as possible + loop
    clock.sleeps.clear()
    pl.values.update(speed="en hızlı", loop=True)
    pl.setup()
    vals = [pl.process({})["a"] for _ in range(7)]
    assert vals == [1, 2, 3, 1, 2, 3, 1] and clock.sleeps == []


def test_replay_errors(tmp_path, clock):
    pl = create("RecordingPlayer")
    for folder, msg in [("", "seçilmedi"), (str(tmp_path / "yok"), "yok"), (str(tmp_path), "manifest")]:
        pl.values["folder"] = folder
        with pytest.raises(ValueError, match=msg):
            pl.setup()
    (tmp_path / "manifest.json").write_text("{bozuk")
    with pytest.raises(ValueError, match="okunamadı"):
        pl.setup()
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "visionflow-kayit", "frames": []}))
    with pytest.raises(ValueError, match="boş"):
        pl.setup()
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "visionflow-kayit",
                                                        "frames": [{"file": "000001.png", "t": 0}]}))
    pl.setup()
    with pytest.raises(RuntimeError, match="kare dosyası yok"):
        pl.process({})
    # through the graph: setup error on the block, not a crash
    g = Graph()
    g.add_new("RecordingPlayer", folder=str(tmp_path / "yok"))
    assert "Kayıt klasörü yok" in next(iter(g.setup().values()))


# ---------------------------------------------------------------- flow test
def counting_flow(path, b=2.5, source="ImageFile"):
    g = Graph()
    src = g.add_new(source, path=str(path.parent / "x.png")) if source == "ImageFile" else g.add_new(source)
    gray = g.add_new("Grayscale")
    th = g.add_new("Threshold")
    fc = g.add_new("FindContours", min_area=10)
    cmp = g.add_new("Compare", op=">", b=b)
    g.connect(src.id, "image", gray.id, "image")
    g.connect(gray.id, "image", th.id, "image")
    g.connect(th.id, "image", fc.id, "mask")
    g.connect(fc.id, "count", cmp.id, "a")
    g.save(str(path))
    return g, fc, cmp


def test_flow_test_end_to_end(tmp_path, clock, capsys):
    from visionflow import flowtest
    make_recording(tmp_path / "kayit", clock, [1, 2, 3, 4, 2])
    flow = tmp_path / "akis.json"
    g, fc, cmp = counting_flow(flow)
    exp = tmp_path / "beklenen.json"
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp),
                          "--kaydet-beklenen", "--cikislar", f"Nesne Bul (Kontur).count,{cmp.id}.result"]) == 0
    data = json.loads(exp.read_text(encoding="utf-8"))
    assert data["cikislar"] == [f"{fc.id}.count", f"{cmp.id}.result"]
    assert [r[f"{fc.id}.count"] for r in data["kareler"]] == [1, 2, 3, 4, 2]
    assert [r[f"{cmp.id}.result"] for r in data["kareler"]] == [False, False, True, True, False]
    assert clock.sleeps == []                               # replayed as fast as possible
    capsys.readouterr()
    # same flow: pass
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp)]) == 0
    out = capsys.readouterr().out
    assert "SONUÇ: GEÇTİ" in out and "Resim Dosyası" in out and "| Karşılaştır.result" in out
    # a changed limit decides differently on frame 3: fail
    counting_flow(flow, b=3.5)
    g2 = Graph.load(str(flow))
    assert set(g2.nodes) != set(g.nodes)                    # new ids: the saved keys must match the new flow
    capsys.readouterr()
    code = flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp)])
    assert code == 2 and "akışta yok" in capsys.readouterr().out


def test_flow_test_detects_a_changed_decision(tmp_path, clock, capsys):
    from visionflow import flowtest
    make_recording(tmp_path / "kayit", clock, [1, 2, 3, 4, 2])
    flow = tmp_path / "akis.json"
    counting_flow(flow)
    exp = tmp_path / "beklenen.json"
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp),
                          "--kaydet-beklenen"]) == 0
    data = json.loads(flow.read_text(encoding="utf-8"))
    for nd in data["nodes"]:                                # edit the same flow: stricter limit
        if nd["type"] == "Compare":
            nd["values"]["b"] = 3.5
    flow.write_text(json.dumps(data), encoding="utf-8")
    capsys.readouterr()
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp)]) == 1
    out = capsys.readouterr().out
    assert "SONUÇ: KALDI" in out and "KALDI" in out.split("SONUÇ")[0]
    assert "Karşılaştır.result kare 3: beklenen true, bulunan false" in out
    assert re.search(r"Nesne Bul \(Kontur\)\.count\s*\|\s*5\s*\|\s*5\s*\|\s*0\s*\|\s*GEÇTİ", out)
    # a shorter recording than expected is a failure too
    capsys.readouterr()
    for nd in data["nodes"]:
        if nd["type"] == "Compare":
            nd["values"]["b"] = 2.5
    flow.write_text(json.dumps(data), encoding="utf-8")
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp), "--kare", "3"]) == 1
    assert "Kare sayısı farklı: beklenen 5, bulunan 3" in capsys.readouterr().out


def test_flow_test_with_player_block_tolerance_and_errors(tmp_path, clock, capsys):
    from visionflow import flowtest
    make_recording(tmp_path / "kayit", clock, [1, 3])
    g = Graph()
    pl = g.add_new("RecordingPlayer", folder="eski/yer", loop=True)
    fc_in = g.add_new("Grayscale")
    m = g.add_new("Math", op="*")
    rec = g.add_new("FrameRecorder", folder=str(tmp_path / "olmamali"))
    g.connect(pl.id, "image", fc_in.id, "image")
    g.connect(pl.id, "a", m.id, "a")
    g.connect(pl.id, "image", rec.id, "image")
    flow = tmp_path / "a.json"
    g.save(str(flow))
    exp = tmp_path / "e.json"
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp),
                          "--kaydet-beklenen"]) == 0
    out = capsys.readouterr().out
    assert "bloğu testte kapalı" in out and not (tmp_path / "olmamali").exists()
    data = json.loads(exp.read_text(encoding="utf-8"))
    assert data["cikislar"] == [f"{m.id}.result"] and len(data["kareler"]) == 2      # loop is switched off
    data["kareler"][0][f"{m.id}.result"] += 1e-9                                      # within the tolerance
    exp.write_text(json.dumps(data), encoding="utf-8")
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp)]) == 0
    data["kareler"][0][f"{m.id}.result"] += 0.1
    exp.write_text(json.dumps(data), encoding="utf-8")
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp)]) == 1
    assert flowtest.main([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp),
                          "--tolerans", "0.5"]) == 0
    capsys.readouterr()
    for args, msg in [
        ([str(flow), "--kayit", str(tmp_path / "yok"), "--beklenen", str(exp)], "Kayıt klasörü yok"),
        ([str(tmp_path / "yok.json"), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp)], "bulunamadı"),
        ([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(tmp_path / "yok.json")], "kaydet-beklenen"),
        ([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(flow)], "beklenen değerler dosyası değil"),
        ([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp), "--cikislar", "Matematik.yok"],
         "çıkışı yok"),
        ([str(flow), "--kayit", str(tmp_path / "kayit"), "--beklenen", str(exp), "--cikislar", "Yok.x"],
         "bulunamadı"),
    ]:
        assert flowtest.main(args) == 2, msg
        assert msg in capsys.readouterr().out


def test_flow_test_needs_a_single_source(tmp_path, clock, capsys):
    from visionflow import flowtest
    make_recording(tmp_path / "kayit", clock, [1])
    g = Graph()
    g.add_new("TestPattern")
    g.add_new("TestPattern")
    g.save(str(tmp_path / "a.json"))
    assert flowtest.main([str(tmp_path / "a.json"), "--kayit", str(tmp_path / "kayit"), "--beklenen",
                          str(tmp_path / "e.json"), "--kaydet-beklenen"]) == 2
    assert "birden fazla" in capsys.readouterr().out
    g = Graph()
    g.add_new("Number")
    g.save(str(tmp_path / "b.json"))
    assert flowtest.main([str(tmp_path / "b.json"), "--kayit", str(tmp_path / "kayit"), "--beklenen",
                          str(tmp_path / "e.json"), "--kaydet-beklenen"]) == 2
    assert "Kayıttan Oynat" in capsys.readouterr().out


def test_main_dispatches_flow_test(tmp_path, clock, monkeypatch, capsys):
    from visionflow import __main__ as entry
    make_recording(tmp_path / "kayit", clock, [2, 3])
    flow = tmp_path / "akis.json"
    counting_flow(flow, source="TestPattern")
    exp = tmp_path / "e.json"
    monkeypatch.setattr(sys, "argv", ["VisionFlow", "test-akis", str(flow), "--kayit", str(tmp_path / "kayit"),
                                      "--beklenen", str(exp), "--kaydet-beklenen"])
    assert entry.main() == 0
    monkeypatch.setattr(sys, "argv", sys.argv[:-1])
    assert entry.main() == 0
    out = capsys.readouterr().out
    assert "'Test Görüntüsü' yerine kayıt oynatılıyor" in out and "GEÇTİ" in out


def test_blocks_are_registered():
    from visionflow.core import NODE_REGISTRY
    assert NODE_REGISTRY["FrameRecorder"].title == "Kare Kaydedici"
    assert NODE_REGISTRY["RecordingPlayer"].title == "Kayıttan Oynat"
    assert NODE_REGISTRY["RecordingPlayer"].is_source is True
    assert os.path.basename(ROOT / "visionflow" / "nodes" / "recording.py") == "recording.py"
