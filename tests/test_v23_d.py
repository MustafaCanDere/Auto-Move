"""2.3 (d): İstasyon Yöneticisi (several flows, each in its own process) and the 2.2 gaps it closes:
REST API in --operator mode, 'servis kur --api', headless --api with the watchdog limits, /durum recipe of a new flow."""
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import cv2
import pytest

from visionflow import api_server, headless, service
from visionflow import stations as stn
from visionflow.core import Graph, load_builtin
from visionflow.core import node as _node

from test_v22_fixes import KEY, _free_port, call, pump_call, pump_until, recipe_flow

load_builtin()
ROOT = pathlib.Path(__file__).resolve().parents[1]

CRASH_CODE = ('import os\nstate["n"] = state.get("n", 0) + 1\n'
              'if state["n"] > 3:\n    os._exit(3)\nout_img = img')


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))       # children inherit it
    monkeypatch.setattr(_node, "FLOW_DIR", None)
    monkeypatch.setattr(_node, "FLOW_FILE", None)
    return tmp_path / "home"


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pattern_flow(path, code=None):
    g = Graph()
    src = g.add_new("TestPattern", (0, 0), width=96, height=72)
    if code is None:
        out = g.add_new("Preview", (200, 0))
        g.connect(src.id, "image", out.id, "image")
    else:
        py = g.add_new("PythonScript", (200, 0), code=code)
        g.connect(src.id, "image", py.id, "img")
    g.save(str(path))
    return path


def write_stations(path, items):
    path.write_text(json.dumps({"istasyonlar": items}, ensure_ascii=False), encoding="utf-8")
    return path


def wait(sup, cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        sup.tick()
        if cond():
            return True
        time.sleep(0.03)
    return False


def row(sup, name):
    return next(r for r in sup.status() if r["ad"] == name)


def dead(pid, timeout=10.0):
    if sys.platform == "win32":
        timeout = max(timeout, 5.0)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not stn.pid_alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def supervisors():
    made = []
    yield made
    for s in made:
        s.shutdown(timeout=3)


# ------------------------------------------------------------------------------------------- station file
def test_station_file_roundtrip_defaults_and_relative_paths(tmp_path):
    f = write_stations(tmp_path / "hat.json", [
        {"ad": " Hat 1 ", "akis": "a.json"},
        {"ad": "Hat 2", "akis": str(tmp_path / "b.json"), "recete": 2, "oncelik": "dusuk", "cekirdekler": [0, 0],
         "api_portu": 8799, "bekleme_sn": 5, "en_fazla_bekleme_sn": 1}])
    a, b = stn.load_stations(f)
    assert a.ad == "Hat 1" and a.otomatik_baslat and a.yeniden_baslat and a.en_fazla_yeniden == 5 and a.fps == 30
    assert a.flow_path(tmp_path) == tmp_path / "a.json"
    assert b.recete == "2" and b.cekirdekler == [0] and b.api_portu == 8799
    assert b.en_fazla_bekleme_sn == 5            # never shorter than the first wait
    stn.save_stations(tmp_path / "out.json", [a, b])
    raw = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["istasyonlar"]
    assert raw[0] == {"ad": "Hat 1", "akis": "a.json"}                  # defaults are not written
    assert stn.load_stations(tmp_path / "out.json") == [a, b]
    assert stn.relative_flow(str(tmp_path / "x" / "c.json"), f) == "x/c.json"
    # a bare list works too, UTF-8 with BOM (Notepad) too
    (tmp_path / "liste.json").write_bytes(b"\xef\xbb\xbf" + json.dumps([{"ad": "Ç", "akis": "a.json"}]).encode())
    assert stn.load_stations(tmp_path / "liste.json")[0].ad == "Ç"


@pytest.mark.parametrize("items, text", [
    ([{"ad": "A", "akis": "a.json"}, {"ad": "a", "akis": "b.json"}], "iki istasyon"),
    ([{"ad": "Hat ı", "akis": "a.json"}, {"ad": "Hat i", "akis": "b.json"}], "dosya adı"),
    ([{"ad": "A", "akis": "a", "api_portu": 9000}, {"ad": "B", "akis": "b", "api_portu": 9000}], "aynı API portu"),
    ([{"ad": "A", "akis": "a.json", "renk": 1}], "bilinmeyen alan"),
    ([{"akis": "a.json"}], "'ad'"),
    ([{"ad": "A"}], "'akis'"),
    ([{"ad": "A", "akis": "a", "oncelik": "max"}], "oncelik"),
    ([{"ad": "A", "akis": "a", "cekirdekler": [9999]}], "çekirdek"),
    ([{"ad": "A", "akis": "a", "cekirdekler": "0,1"}], "listesi"),
    ([{"ad": "A", "akis": "a", "api_portu": 70000}], "api_portu"),
    ([{"ad": "A", "akis": "a", "en_fazla_yeniden": 1.5}], "tam sayı"),
    ([{"ad": "A", "akis": "a", "fps": "hızlı"}], "sayı"),
    ([{"ad": "A", "akis": "a", "otomatik_baslat": "evet"}], "true ya da false"),
    (["A"], "JSON nesnesi"),
])
def test_station_file_errors_are_turkish(tmp_path, items, text):
    with pytest.raises(ValueError, match=text):
        stn.load_stations(write_stations(tmp_path / "s.json", items))


def test_station_file_missing_or_broken(tmp_path):
    with pytest.raises(ValueError, match="bulunamadı"):
        stn.load_stations(tmp_path / "yok.json")
    (tmp_path / "bozuk.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="geçerli JSON"):
        stn.load_stations(tmp_path / "bozuk.json")
    (tmp_path / "nesne.json").write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="'istasyonlar'"):
        stn.load_stations(tmp_path / "nesne.json")


# ------------------------------------------------------------------------------------------- real processes
def test_three_stations_one_crashes_only_it_restarts_with_backoff_then_gives_up(tmp_path, home, supervisors):
    pattern_flow(tmp_path / "a.json")
    pattern_flow(tmp_path / "crash.json", CRASH_CODE)
    f = write_stations(tmp_path / "hat.json", [
        {"ad": "Hat 1", "akis": "a.json", "fps": 25},
        {"ad": "Hat 2", "akis": "a.json", "fps": 25, "oncelik": "dusuk", "cekirdekler": [0]},
        {"ad": "Çöken", "akis": "crash.json", "fps": 25, "bekleme_sn": 0.3, "en_fazla_yeniden": 2}])
    starts = []
    sup = stn.Supervisor(stn.load_stations(f), source=f,
                         log=lambda m: starts.append(time.monotonic()) if m.startswith("Çöken: başlatılıyor") else None)
    supervisors.append(sup)
    sup.start_all(only_auto=True)
    assert len(sup.pids()) == 3 and len(set(sup.pids())) == 3
    pids = {n: sup.runs[n].proc.pid for n in ("Hat 1", "Hat 2")}
    assert wait(sup, lambda: sup.runs["Çöken"].state == "coktu")
    assert wait(sup, lambda: all(row(sup, n)["durum"] == "calisiyor" and row(sup, n)["kare"] > 5
                                 for n in ("Hat 1", "Hat 2")))
    crashed = row(sup, "Çöken")
    assert crashed["yeniden"] == 2 and crashed["pid"] is None and crashed["cikis_kodu"] == 3
    assert "çıkış kodu 3" in crashed["son_hata"]
    assert len(starts) == 3
    assert starts[1] - starts[0] >= 0.3 and starts[2] - starts[1] >= 0.6       # 0.3 s, then twice as long
    for n, pid in pids.items():                                                # the others never noticed
        r = row(sup, n)
        assert r["pid"] == pid and r["yeniden"] == 0 and r["fps"] > 0 and r["sure_sn"] > 0
        assert r["hata_sayisi"] == 0 and r["son_hata"] == ""
    if hasattr(os, "sched_getaffinity"):
        assert os.sched_getaffinity(pids["Hat 2"]) == {0}
    # a live thumbnail: the preview JPEG of the station's 'Önizleme' block
    img = cv2.imread(str(sup.preview_path("Hat 1")))
    assert img is not None and img.shape[:2] == (72, 96)
    log = sup.log_tail("Çöken")
    assert "vazgeçildi" in log and log.count("başlatılıyor") == 3
    sup.shutdown()
    for pid in pids.values():
        assert dead(pid, 1)
    assert sup.pids() == [] and not sup.busy
    assert all(r["durum"] == "durdu" for r in sup.status() if r["ad"] != "Çöken")
    assert not sup.stop_path("Hat 1").exists()
    try:
        import psutil
    except ImportError:
        return
    assert not [c for c in psutil.Process().children(recursive=True) if c.pid in pids.values()]


def test_stop_restart_and_clean_exit_codes(tmp_path, home, supervisors):
    pattern_flow(tmp_path / "a.json")
    f = write_stations(tmp_path / "hat.json", [{"ad": "A", "akis": "a.json", "fps": 30},
                                               {"ad": "Kısa", "akis": "a.json", "kare": 5, "fps": 0}])
    sup = stn.Supervisor(stn.load_stations(f), source=f)
    supervisors.append(sup)
    sup.start("A")
    sup.start("Kısa")
    assert wait(sup, lambda: row(sup, "A")["durum"] == "calisiyor")
    first = sup.runs["A"].proc.pid
    assert wait(sup, lambda: row(sup, "Kısa")["durum"] == "bitti")           # a flow that ends is not restarted
    assert row(sup, "Kısa")["cikis_kodu"] == 0 and row(sup, "Kısa")["kare"] == 5
    sup.restart("A")
    assert wait(sup, lambda: sup.runs["A"].proc is not None and sup.runs["A"].proc.pid != first
                and row(sup, "A")["durum"] == "calisiyor")
    assert dead(first, 1) and row(sup, "A")["yeniden"] == 0                    # a manual restart is not a crash
    sup.stop("A")
    assert row(sup, "A")["durum"] == "duruyor"
    assert wait(sup, lambda: row(sup, "A")["durum"] == "durdu", 10)
    assert row(sup, "A")["cikis_kodu"] == 0                                    # stopped itself (stop file)
    with pytest.raises(ValueError, match="istasyon yok"):
        sup.start("yok")


def test_crash_reason_bad_recipe_missing_flow(tmp_path, home, supervisors):
    (tmp_path / "bozuk.json").write_text("{bozuk", encoding="utf-8")
    pattern_flow(tmp_path / "a.json")
    f = write_stations(tmp_path / "hat.json", [
        {"ad": "Bozuk", "akis": "bozuk.json", "yeniden_baslat": False},
        {"ad": "Reçetesiz", "akis": "a.json", "recete": "yok"},
        {"ad": "Yok", "akis": "yok.json"}])
    sup = stn.Supervisor(stn.load_stations(f), source=f)
    supervisors.append(sup)
    sup.start_all()
    assert row(sup, "Yok")["durum"] == "hata" and "bulunamadı" in row(sup, "Yok")["son_hata"]
    assert wait(sup, lambda: row(sup, "Bozuk")["durum"] == "coktu" and row(sup, "Reçetesiz")["durum"] == "hata")
    assert "Error" in row(sup, "Bozuk")["son_hata"] and row(sup, "Bozuk")["yeniden"] == 0   # the traceback's line
    assert row(sup, "Reçetesiz")["cikis_kodu"] == 2 and row(sup, "Reçetesiz")["yeniden"] == 0
    assert "Reçete seçilemedi" in sup.log_tail("Reçetesiz")


def test_station_that_ignores_stop_is_killed_after_timeout(tmp_path, home, supervisors):
    pattern_flow(tmp_path / "yavas.json", 'import time\nstate["n"] = state.get("n", 0) + 1\n'
                                          'if state["n"] > 2:\n    time.sleep(60)\nout_img = img')
    f = write_stations(tmp_path / "hat.json", [{"ad": "Yavaş", "akis": "yavas.json"}])
    sup = stn.Supervisor(stn.load_stations(f), source=f, kill_timeout=1.0)
    supervisors.append(sup)
    sup.start("Yavaş")
    pid = sup.runs["Yavaş"].proc.pid
    assert wait(sup, lambda: row(sup, "Yavaş")["kare"] >= 2)
    t0 = time.monotonic()
    sup.shutdown()
    assert time.monotonic() - t0 < 6
    assert row(sup, "Yavaş")["durum"] == "durdu" and dead(pid, 1)
    assert "zorla kapatılıyor" in sup.log_tail("Yavaş")


def test_set_stations_removes_and_updates(tmp_path, home, supervisors):
    pattern_flow(tmp_path / "a.json")
    f = write_stations(tmp_path / "hat.json", [{"ad": "A", "akis": "a.json"}, {"ad": "B", "akis": "a.json"}])
    sup = stn.Supervisor(stn.load_stations(f), source=f)
    supervisors.append(sup)
    sup.start("B")
    pid = sup.runs["B"].proc.pid
    a = sup.runs["A"].st
    sup.set_stations([stn.Station(ad="A", akis="a.json", fps=5)])                  # B removed: stopped
    assert list(sup.runs) == ["A"] and sup.runs["A"].st.fps == 5 and a.fps == 30
    assert dead(pid, 1)
    with pytest.raises(ValueError):
        sup.set_stations([stn.Station(ad="X", akis="a"), stn.Station(ad="x", akis="b")])


def test_api_station_reports_over_rest_and_status_file(tmp_path, home, supervisors):
    pattern_flow(tmp_path / "a.json")
    port = _free_port()
    api_server.save_settings({"port": port, "key": KEY, "rate": 0})
    f = write_stations(tmp_path / "hat.json", [{"ad": "API", "akis": "a.json", "api_portu": port, "fps": 20}])
    sup = stn.Supervisor(stn.load_stations(f), source=f)
    supervisors.append(sup)
    sup.start("API")
    assert "--api" in sup.command(sup.runs["API"].st) and "--yeniden-baslat" in sup.command(sup.runs["API"].st)
    assert wait(sup, lambda: row(sup, "API")["durum"] == "calisiyor" and row(sup, "API")["kare"] > 3)
    code, d = call(f"http://127.0.0.1:{port}/api/v1/durum")
    assert code == 200 and d["calisiyor"] and d["akis"] == "a.json"
    code, d = call(f"http://127.0.0.1:{port}/api/v1/durdur", "POST")
    assert code == 200
    assert wait(sup, lambda: row(sup, "API")["durum"] == "bekliyor")              # the process (and API) stays
    assert sup.runs["API"].proc is not None
    sup.shutdown()
    assert row(sup, "API")["durum"] == "durdu"


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL of the supervisor")
def test_children_stop_when_the_supervisor_dies(tmp_path, home):
    pattern_flow(tmp_path / "a.json")
    f = write_stations(tmp_path / "hat.json", [{"ad": "A", "akis": "a.json", "fps": 20}])
    proc = subprocess.Popen([sys.executable, "-m", "visionflow", "istasyonlar", str(f), "--durum-araligi", "0"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    status = home / "istasyonlar" / "hat" / "A.json"
    try:
        end = time.monotonic() + 20
        child = None
        while time.monotonic() < end and child is None:
            try:
                d = json.loads(status.read_text(encoding="utf-8"))
                child = d["pid"] if d.get("calisiyor") else None
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.05)
        assert child
        proc.kill()                           # no chance to stop its stations
        proc.wait(5)
        assert dead(child, 10)                # the station noticed and stopped by itself
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


def test_cli_supervisor_runs_until_sigterm_and_check_options(tmp_path, home):
    pattern_flow(tmp_path / "a.json")
    f = write_stations(tmp_path / "hat.json", [{"ad": "A", "akis": "a.json", "fps": 20},
                                               {"ad": "Elle", "akis": "a.json", "otomatik_baslat": False}])
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "visionflow", "istasyonlar", str(f), "--durum-araligi", "0.5"],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    status = home / "istasyonlar" / "hat" / "A.json"
    try:
        end = time.monotonic() + 20
        child = None
        while time.monotonic() < end and child is None:
            try:
                d = json.loads(status.read_text(encoding="utf-8"))
                child = d["pid"] if d.get("calisiyor") and d.get("kare", 0) > 3 else None
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.05)
        assert child
        # a second supervisor of the same file is refused
        other = subprocess.run([sys.executable, "-m", "visionflow", "istasyonlar", str(f)], cwd=ROOT,
                               capture_output=True, env=env, timeout=30)
        assert other.returncode == 3 and "zaten çalışıyor" in other.stdout.decode()
        if os.name == "posix":
            proc.send_signal(signal.SIGTERM)
        else:
            (home / "istasyonlar" / "hat" / "dur").touch()
        out = proc.communicate(timeout=20)[0].decode()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, out
    assert "Çalışıyor" in out and "İstasyon yöneticisi durdu." in out
    assert "Elle: başlatılıyor" not in out                              # not an auto-start station
    assert dead(child, 1)
    assert not (home / "istasyonlar" / "hat" / "yonetici.pid").exists()

    assert stn.main([str(f), "--denetle"]) == 0
    assert stn.main([str(tmp_path / "ornek.json"), "--ornek"]) == 0
    assert stn.load_stations(tmp_path / "ornek.json")[0].ad == "Hat 1"
    assert stn.main([str(tmp_path / "ornek.json"), "--denetle"]) == 1       # its hat1.json does not exist
    assert stn.main([str(tmp_path / "ornek.json"), "--ornek"]) == 1          # never overwritten
    assert stn.run_supervisor(tmp_path / "yok.json", lambda: True, out=lambda m: None) == 2


def test_cli_entry_point_listed():
    from visionflow import __main__ as m
    assert "istasyonlar" in m.KNOWN and "istasyonlar hat.json" in m.HELP


# ------------------------------------------------------------------------------------------- the window
def test_station_manager_window(tmp_path, home, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from visionflow.ui import station_manager as sm
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    pattern_flow(tmp_path / "a.json")
    pattern_flow(tmp_path / "b.json")

    class Editor:
        opened = []

        def open_flow(self, p):
            self.opened.append(p)

        def show(self):
            pass

        raise_ = activateWindow = show

    w = sm.StationManager(Editor())
    try:
        w.show()
        assert not w.b_add.isEnabled()
        assert w.new_file(str(tmp_path / "hat.json"))
        assert w.add_station(stn.Station(ad="Hat 1", akis=str(tmp_path / "a.json"), fps=20))
        assert w.add_station(stn.Station(ad="Hat 2", akis="b.json", fps=20))
        assert not w.add_station(stn.Station(ad="hat 1", akis="b.json"))        # same name
        assert warned and "iki istasyon" in warned[-1]
        saved = stn.load_stations(tmp_path / "hat.json")
        assert [s.ad for s in saved] == ["Hat 1", "Hat 2"]
        assert w.table.rowCount() == 2 and w.table.item(0, 2).text() == "Hat 1"
        w.select("Hat 1")
        w.start_selected()
        assert pump_until(qapp, lambda: w.table.item(0, 3).text().startswith("Çalışıyor")
                          and w.table.item(0, 1).data(sm.Qt.DecorationRole) is not None, 20)
        assert w.b_stop.isEnabled() and not w.b_start.isEnabled()
        assert pump_until(qapp, lambda: "başlatılıyor" in w.log_view.toPlainText())
        assert w.preview.pixmap() is not None and not w.preview.pixmap().isNull()
        assert w.table.item(1, 3).text() == "Durdu"                              # only the selected one
        w.open_flow()
        assert Editor.opened == [str(tmp_path / "a.json")]
        # edit the running one: new settings from its next start
        assert w.edit_station(stn.Station(ad="Hat 1", akis="a.json", fps=10))
        assert "yeniden başlatıldığında" in w.info.text() and stn.load_stations(tmp_path / "hat.json")[0].fps == 10
        w.select("Hat 2")
        assert w.remove_station(confirm=False)
        assert [s.ad for s in stn.load_stations(tmp_path / "hat.json")] == ["Hat 1"] and w.table.rowCount() == 1
        asked = []
        w.ask_stop = lambda: asked.append(1) or False
        w.close()                                         # refused: a station runs
        assert asked and w.isVisible() and w.sup is not None
        w.ask_stop = lambda: True
        pid = w.sup.runs["Hat 1"].proc.pid
        w.close()
        assert w.sup is None and dead(pid, 1)
    finally:
        if w.sup is not None:
            w.sup.shutdown(timeout=3)
        w.close()


def test_station_manager_open_reuses_window_and_bad_file(tmp_path, home, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from visionflow.ui import station_manager as sm
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))

    class Win:
        pass
    win = Win()
    w = sm.open_station_manager(win)
    try:
        assert sm.open_station_manager(win) is w and w.isVisible()
        (tmp_path / "bozuk.json").write_text("{", encoding="utf-8")
        assert not w.load(tmp_path / "bozuk.json")
        assert warned and "geçerli JSON" in warned[0] and w.sup is None
        assert not w.add_station(stn.Station(ad="A", akis="a.json"))
    finally:
        w.close()
    from visionflow.ui.main_window import MainWindow
    assert hasattr(MainWindow, "open_stations")


def test_station_dialog_checks_input(tmp_path, home, qapp):
    from visionflow.ui.station_manager import StationDialog
    g, *_ = recipe_flow(selector=None)
    g.save(str(tmp_path / "r.json"))
    dlg = StationDialog(None, None, tmp_path)
    dlg.name.setText("Hat 1")
    dlg.flow.setText("r.json")
    dlg._load_recipes()
    assert [dlg.recipe.itemText(i) for i in range(dlg.recipe.count())] == ["", "A", "B"]
    dlg.recipe.setCurrentText("B")
    dlg.cpus.setText("0;x")
    dlg.accept()
    assert dlg.result_station is None and "çekirdek" in dlg.error.text()
    dlg.cpus.setText("0")
    dlg.api_port.setValue(8800)
    dlg.accept()
    st = dlg.result_station
    assert st.ad == "Hat 1" and st.recete == "B" and st.cekirdekler == [0] and st.api_portu == 8800
    dlg2 = StationDialog(None, st, tmp_path)
    assert dlg2.name.text() == "Hat 1" and dlg2.recipe.currentText() == "B" and dlg2.api_port.value() == 8800
    dlg2.name.setText("")
    dlg2.accept()
    assert dlg2.result_station is None and "'ad'" in dlg2.error.text()


# ------------------------------------------------------------------------------------------- 2.2 gaps
def test_gap_a_operator_mode_starts_the_api(tmp_path, home, qapp, monkeypatch):
    """--operator akis.json: the REST API opens when "program açılınca da açılsın" is set, and controls the panel."""
    from visionflow import __main__ as main_mod
    from visionflow.ui import operator as op
    from visionflow.ui.api_dialog import manager
    port = _free_port()
    api_server.save_settings({"enabled": True, "port": port, "key": KEY, "rate": 0})
    g, src, num, sel, arc = recipe_flow(selector=None)
    flow = tmp_path / "akis.json"
    g.save(str(flow))
    made = []
    real = op.run_operator
    monkeypatch.setattr(op, "run_operator", lambda *a, **k: made.append(real(*a, **k)) or made[-1])

    class App:
        def exec(self):
            return 0

    assert main_mod._operator(App(), [str(flow)], []) == 0
    panel = made[0]
    try:
        mgr = manager(panel)
        assert mgr.running
        url = f"http://127.0.0.1:{port}/api/v1"
        assert pump_until(qapp, lambda: mgr.state.running and mgr.state.frames > 2)
        code, d = pump_call(qapp, f"{url}/durum")
        assert code == 200 and d["calisiyor"] and d["akis"] == "akis.json" and d["recete"] == ""
        code, d = pump_call(qapp, f"{url}/recete", "POST", body={"ad": "B"})
        assert code == 200 and d["recete"] == "B" and panel.recipe_name == "B"
        assert pump_until(qapp, lambda: mgr.state.block_values().get(num.id, {}).get("cikislar", {})
                          .get("value") == 2.0)
        code, d = pump_call(qapp, f"{url}/durdur", "POST")
        assert code == 200 and panel.worker is None
        code, d = pump_call(qapp, f"{url}/durum")
        assert d["calisiyor"] is False and d["recete"] == "B"
        code, d = pump_call(qapp, f"{url}/baslat", "POST")
        assert code == 200 and panel.worker is not None
        assert pump_until(qapp, lambda: mgr.state.running and mgr.state.recipe == "B")   # restarted with B
        code, d = pump_call(qapp, f"{url}/bloklar")
        assert code == 200 and any(b["id"] == num.id for b in d["bloklar"])
    finally:
        panel.leave(open_editor=False)
    assert not manager(panel).running                 # the port is free for the editor opened from the panel


def test_gap_b_service_install_with_api_and_station_file(tmp_path, home):
    flow = pattern_flow(tmp_path / "a.json")
    out = []
    assert service.main(["kur", "--akis", str(flow), "--api", "8790", "--api-ag", "--yazdir", "--yontem", "systemd"],
                        out=out.append) == 0
    text = "\n".join(out)
    assert "--headless" in text and "--api 8790 --api-ag" in text and "--yeniden-baslat" in text
    out.clear()
    assert service.main(["kur", "--akis", str(flow), "--api", "8790", "--yazdir", "--yontem", "gorev"],
                        out=out.append) == 0
    assert "--api 8790" in "\n".join(out) and "--api-ag" not in "\n".join(out)
    st = write_stations(tmp_path / "hat 1.json", [{"ad": "A", "akis": "a.json"}])
    out.clear()
    assert service.main(["kur", "--istasyonlar", str(st), "--yazdir", "--yontem", "systemd"], out=out.append) == 0
    text = "\n".join(out)
    assert "visionflow-hat_1.service" in text and " istasyonlar " in text and "--headless" not in text
    assert service.stations_command(str(st))[-2:] == ["istasyonlar", str(st.resolve())]
    with pytest.raises(SystemExit):
        service.main(["kur", "--akis", str(flow), "--istasyonlar", str(st), "--yazdir"], out=out.append)
    with pytest.raises(SystemExit):
        service.main(["kur", "--istasyonlar", str(st), "--api", "8790", "--yazdir"], out=out.append)
    with pytest.raises(SystemExit):
        service.main(["kur", "--akis", str(flow), "--api", "0", "--yazdir"], out=out.append)
    (tmp_path / "bozuk.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(SystemExit):
        service.main(["kur", "--istasyonlar", str(tmp_path / "bozuk.json"), "--yazdir"], out=out.append)


def test_gap_c_headless_api_watchdog_backoff_and_limit(tmp_path, home, monkeypatch):
    """--api with --yeniden-baslat --en-fazla-yeniden 2: waits 1 s then 2 s, gives up after 2 restarts and waits
    for 'baslat' (used to restart every 5 s forever and ignore --en-fazla-yeniden)."""
    flow = pattern_flow(tmp_path / "a.json")
    runs, sleeps = [], []
    monkeypatch.setattr(headless, "_sleep", lambda s: sleeps.append(s))

    def crash(*a, **k):
        runs.append(time.monotonic())
        raise RuntimeError("kamera sürücüsü çöktü")
    monkeypatch.setattr(headless, "run_flow", crash)
    api_server.save_settings({"port": 0, "key": KEY, "rate": 0})
    seen = {}

    def should_stop():
        if len(runs) >= 3:
            seen.setdefault("t", time.monotonic())
            return time.monotonic() - seen["t"] > 0.5          # stays waiting after giving up
        return len(runs) > 10
    assert headless.main([str(flow), "--api", "0", "--yeniden-baslat", "--en-fazla-yeniden", "2"],
                         should_stop=should_stop) == 1
    assert len(runs) == 3
    assert sum(sleeps) == pytest.approx(3.0)                   # 1 s + 2 s
    log = (home / "kayitlar" / "headless.log").read_text(encoding="utf-8")
    assert "kamera sürücüsü çöktü" in log and "En fazla yeniden başlatma sayısına (2)" in log


def test_gap_c_controller_restart_count_resets_on_baslat(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(headless, "_sleep", lambda s: None)
    calls = []

    def crash(*a, **k):
        calls.append(1)
        raise RuntimeError("x")
    monkeypatch.setattr(headless, "run_flow", crash)
    ctl = api_server.HeadlessController("akis.json", api_server.FlowState())
    ctl.restart_on_crash, ctl.max_restarts = True, 1
    done = threading.Event()
    t = threading.Thread(target=ctl.serve, args=(done.is_set,), daemon=True)
    t.start()
    try:
        end = time.monotonic() + 10
        while (ctl._want or len(calls) < 2) and time.monotonic() < end:
            time.sleep(0.01)
        assert len(calls) == 2 and ctl.restarts == 1 and not ctl._want     # 1 restart, then it gave up
        assert "En fazla" in ctl.state.last_errors[-1]["hata"]
        with pytest.raises(ValueError, match="başlatılamadı"):
            ctl.start()                                                     # 'baslat': a full new set of restarts
        assert len(calls) == 4
    finally:
        done.set()
        t.join(5)


def test_gap_d_flowstate_attach_forgets_previous_recipe():
    st = api_server.FlowState()
    st.set_recipe("Eski akışın reçetesi")
    st.attach()
    assert st.recipe == ""
    st.attach("A")
    assert st.recipe == "A"
    ctl = api_server.HeadlessController("x.json", st, recete="B")
    ctl.attach(None)
    assert st.recipe == "B" and ctl.current_recipe() == "B"
    ctl.recete = None
    ctl.attach(None)
    assert st.recipe == ""


def test_gap_d_editor_api_recipe_of_newly_opened_flow(home, qapp, tmp_path):
    from visionflow.ui.api_dialog import manager
    from visionflow.ui.main_window import MainWindow
    win = MainWindow()
    win._confirm_discard = lambda: True
    mgr = manager(win)
    port = _free_port()
    url = mgr.start({"enabled": True, "port": port, "key": KEY, "rate": 0})
    try:
        g, src, num, sel, arc = recipe_flow(selector=None)
        win._set_root(Graph.from_dict(g.to_dict()))
        win._loaded(None)
        assert win.apply_recipe("B") == []
        code, d = pump_call(qapp, f"{url}/durum")
        assert d["recete"] == "B" and not d["calisiyor"]
        win.start()
        assert pump_until(qapp, lambda: mgr.state.running and mgr.state.frames > 2)
        assert mgr.state.recipe == "B"                            # started with the recipe applied before
        win.stop()
        # another flow is opened: its runs (and /durum) do not carry the old flow's recipe
        g2 = Graph()
        g2.add_new("TestPattern", width=64, height=48)
        win._set_root(g2)
        win._loaded(None)
        code, d = pump_call(qapp, f"{url}/durum")
        assert d["recete"] == "" and pump_call(qapp, f"{url}/recete")[1]["recete"] == ""
        win.start()
        assert pump_until(qapp, lambda: mgr.state.running and mgr.state.frames > 2)
        code, d = pump_call(qapp, f"{url}/durum")
        assert d["calisiyor"] and d["recete"] == ""
        win.stop()
    finally:
        mgr.stop()
        win.stop()
        win.close()


def test_one_supervisor_per_station_file(tmp_path, home, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from visionflow.ui import station_manager as sm
    f = write_stations(tmp_path / "hat.json", [{"ad": "A", "akis": "a.json"}])
    sup = stn.Supervisor(stn.load_stations(f), source=f)
    assert sup.claim() == 0 and sup.claim() == 0                # its own mark is no conflict
    (sup.run_dir / "yonetici.pid").write_text(str(os.getppid()))     # a live other process runs it
    assert stn.Supervisor(stn.load_stations(f), source=f).claim() == os.getppid()
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    w = sm.StationManager(None)
    try:
        assert not w.load(f) and w.sup is None and "başka bir istasyon yöneticisinde" in warned[0]
    finally:
        w.close()
    (sup.run_dir / "yonetici.pid").write_text("999999999")          # left over by a crashed supervisor
    assert sup.claim() == 0
    sup.release()
    assert not (sup.run_dir / "yonetici.pid").exists()


def test_recipe_by_number_is_reported_by_name(tmp_path, home):
    g, *_ = recipe_flow(selector=None)
    g.save(str(tmp_path / "r.json"))
    state = api_server.FlowState()
    ctl = api_server.HeadlessController(str(tmp_path / "r.json"), state, recete="2")
    assert headless.run_flow(str(tmp_path / "r.json"), frames=2, fps=0, recete="2", api=ctl) == 0
    assert state.recipe == "B" and ctl.current_recipe() == "B"
