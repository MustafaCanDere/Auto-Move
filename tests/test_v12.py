"""1.2: Kendini Geliştir (the program examines itself and improves its own code with the user's approval)."""
import datetime as dt
import json
import os
import pathlib

import pytest

from visionflow import evolve
from visionflow.ai.config import AIConfig
from visionflow.core import Graph, load_builtin

load_builtin()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    AIConfig(api_key="test").save()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A small stand-in for the program's folder, so nothing real is changed."""
    root = tmp_path / "prog"
    (root / "visionflow" / "nodes").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "plugins").mkdir()
    (root / "visionflow" / "nodes" / "demo.py").write_text("def area(w, h):\n    return w + h\n", encoding="utf-8")
    monkeypatch.setattr(evolve, "editable_root", lambda: root)
    return root


def answer(*edits, summary="Alan hesabı düzeltildi."):
    return json.dumps({"ozet": summary, "degisiklikler": list(edits)}, ensure_ascii=False)


FIX = {"dosya": "visionflow/nodes/demo.py", "eski": "return w + h", "yeni": "return w * h"}
REPORT = {"version": "x", "frozen": False, "errors": [], "plugin_errors": {}, "time": "now", "tests_ok": True}


# --- what may be changed ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["visionflow/nodes/more.py", "tests/test_new.py", "plugins/fix.py",
                                  "./visionflow/core/node.py", "visionflow\\nodes\\more.py"])
def test_allowed_paths(path):
    assert evolve.allowed(path) is None


@pytest.mark.parametrize("path", ["visionflow/evolve.py", "visionflow/ai/claude.py", "visionflow/ai/config.py",
                                  "visionflow/Evolve.py", "VisionFlow/ai/Config.py", "visionflow//evolve.py",
                                  "visionflow/./evolve.py", "../visionflow/nodes/more.py", "/etc/passwd.py",
                                  "C:/Windows/x.py", "visionflow/nodes/../../x.py", "README.md", "run.bat",
                                  "examples/x.py", "plugins/sub/x.py", "",
                                  # a package of the same name would replace the protected module on import
                                  "visionflow/ai/config/__init__.py", "visionflow/ai/claude/x.py",
                                  "visionflow/evolve/__init__.py", "visionflow/__init__.py", "visionflow/ai/__init__.py",
                                  # Windows drops trailing dots and spaces: these ARE ai/claude.py and ai/config.py
                                  "visionflow/ai./claude.py", "visionflow/ai /config.py", "visionflow/ai/claude.py.",
                                  "Visionflow/new.py", "TESTS/x.py", "visionflow/a|b.py"])
def test_refused_paths(path):
    assert evolve.allowed(path)


def test_packaged_program_may_only_write_plugins(monkeypatch):
    monkeypatch.setattr(evolve, "frozen", lambda: True)
    assert evolve.allowed("plugins/fix.py") is None
    assert evolve.allowed("visionflow/nodes/more.py")


# --- remembering errors -------------------------------------------------------------------------------------------

def test_error_log_writes_each_error_once_and_groups_them():
    log = evolve.error_log()
    for _ in range(50):
        log.record("Camera", "kamera 0 açılamadı", "visionflow/nodes/inputs.py:10 process")
    log.record("Camera", "kamera 1 açılamadı")
    log.record("Blur", "Önceki blok hata verdi")             # follow-up errors are not a cause
    log.record("Blur", "")
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1                                   # "kamera 1" is the same error as "kamera 0"
    s = log.summary()
    assert len(s) == 1 and s[0]["count"] == 1 and s[0]["block"] == "Camera"
    assert s[0]["where"] == "visionflow/nodes/inputs.py:10 process"
    log.clear()
    assert log.summary() == []


def test_error_log_follows_settings_folder(tmp_path, monkeypatch):
    a = evolve.error_log()
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "other"))
    assert evolve.error_log() is not a and evolve.error_log().path.parent == tmp_path / "other"


def test_graph_reports_where_an_error_happened():
    g = Graph()
    img = g.add_new("ImageFile", path="yok.png")
    r = g.run_once()[img.id]
    assert r.error
    assert "visionflow/" in r.where and ".py:" in r.where


def test_worker_records_block_errors():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from visionflow.ui.worker import FlowWorker
    g = Graph()
    g.add_new("ImageFile", path="yok.png")
    w = FlowWorker(g.to_dict())
    g2 = Graph.from_dict(g.to_dict())
    w._emit(g2, g2.run_once(), 0.0)
    assert [e["block"] for e in evolve.error_log().summary()] == ["ImageFile"]


def test_report_text_lists_errors():
    evolve.error_log().record("Camera", "kamera açılamadı", "visionflow/nodes/inputs.py:5 setup")
    rep = evolve.self_report(run_tests=False)
    text = evolve.report_text(rep)
    assert "Camera" in text and "kamera açılamadı" in text and "inputs.py:5" in text


# --- checking a proposed change -----------------------------------------------------------------------------------

def test_parse_answer_with_text_around_json():
    summary, edits = evolve._parse("Tamam:\n```json\n" + answer(FIX) + "\n```")
    assert summary == "Alan hesabı düzeltildi." and edits[0].new == "return w * h"
    with pytest.raises(evolve.EvolveError):
        evolve._parse("JSON yok")


def test_check_edits_finds_problems(project):
    E = evolve.Edit
    assert evolve.check_edits([E("visionflow/nodes/demo.py", "return w + h", "return w * h")]) == []
    assert "bulunamadı" in evolve.check_edits([E("visionflow/nodes/demo.py", "return w - h", "x")])[0]
    (project / "visionflow" / "nodes" / "two.py").write_text("a = 1\na = 1\n", encoding="utf-8")
    assert "2 kez" in evolve.check_edits([E("visionflow/nodes/two.py", "a = 1", "a = 2")])[0]
    assert "sözdizimi" in evolve.check_edits([E("visionflow/nodes/demo.py", "return w + h", "return (w")])[0]
    assert "korumalı" in evolve.check_edits([E("visionflow/evolve.py", "x", "y")])[0]
    assert "zaten var" in evolve.check_edits([E("visionflow/nodes/demo.py", "", "x = 1", True)])[0]
    assert evolve.check_edits([E("plugins/yeni.py", "", "x = 1\n", True)]) == []


# --- asking, testing on a copy and correcting itself --------------------------------------------------------------

def test_propose_corrects_itself_until_the_tests_pass(project, monkeypatch):
    tried = []

    def fake_try(edits):
        tried.append([e.new for e in edits])
        return (edits[0].new == "return w * h"), "== Testler: ..."
    monkeypatch.setattr(evolve, "try_in_copy", fake_try)
    answers = iter([answer({**FIX, "eski": "return w - h"}),          # 1: old text not in the file
                    answer({**FIX, "yeni": "return w / h"}),          # 2: tests fail on the copy
                    answer(FIX)])                                      # 3: good
    asked = []

    def ask(prompt, system, history):
        asked.append((prompt, len(history)))
        return next(answers)
    steps = []
    prop = evolve.propose("alan yanlış", REPORT, ask=ask, on_step=steps.append)
    assert prop.tests_ok is True and prop.rounds == 3 and prop.edits[0].new == "return w * h"
    assert tried == [["return w / h"], ["return w * h"]]               # round 1 never reached the copy
    assert "bulunamadı" in asked[1][0] and "Testler başarısız" in asked[2][0]
    assert [n for _, n in asked] == [0, 2, 4]                          # Claude sees its earlier answers
    assert any("sorguluyor" in s for s in steps)
    assert "return w + h" in (project / "visionflow/nodes/demo.py").read_text()   # nothing real changed


def test_propose_gives_up_after_max_rounds(project, monkeypatch):
    monkeypatch.setattr(evolve, "try_in_copy", lambda edits: (False, "FAILED test_x"))
    calls = []
    prop = evolve.propose("", REPORT, ask=lambda p, s, h: calls.append(1) or answer(FIX))
    assert len(calls) == evolve.MAX_ROUNDS and prop.tests_ok is False
    with pytest.raises(evolve.EvolveError):
        evolve.apply(prop)


def test_propose_without_changes(project):
    prop = evolve.propose("", REPORT, ask=lambda p, s, h: answer(summary="Kamera kablosunu kontrol edin."))
    assert prop.edits == [] and "Kamera" in prop.summary


def test_protected_file_never_reaches_the_copy(project, monkeypatch):
    monkeypatch.setattr(evolve, "try_in_copy", lambda edits: pytest.fail("must not be tried"))
    bad = {"dosya": "visionflow/ai/config.py", "eski": "daily_limit", "yeni": "daily_limit * 0"}
    prop = evolve.propose("", REPORT, ask=lambda p, s, h: answer(bad))
    assert prop.tests_ok is False and "korumalı" in prop.test_output


def test_try_in_copy_changes_only_the_copy(project, monkeypatch):
    seen = {}

    def fake_checks(root, timeout=900):
        seen["root"] = root
        seen["text"] = (root / "visionflow/nodes/demo.py").read_text()
        return True, "ok"
    monkeypatch.setattr(evolve, "run_checks", fake_checks)
    ok, _ = evolve.try_in_copy([evolve.Edit(**{"path": FIX["dosya"], "old": FIX["eski"], "new": FIX["yeni"]})])
    assert ok and "w * h" in seen["text"] and seen["root"] != project and not seen["root"].exists()
    assert "w + h" in (project / "visionflow/nodes/demo.py").read_text()


def test_packaged_program_tries_plugins_by_loading_them(project, monkeypatch):
    monkeypatch.setattr(evolve, "frozen", lambda: True)
    good = evolve.Edit("plugins/ok.py", "", "X = 1\n", True)
    bad = evolve.Edit("plugins/bad.py", "", "raise RuntimeError('olmadı')\n", True)
    assert evolve.try_in_copy([good])[0] is True
    ok, out = evolve.try_in_copy([bad])
    assert not ok and "olmadı" in out
    from visionflow.core.registry import PLUGIN_ERRORS
    assert "bad" not in " ".join(PLUGIN_ERRORS)                        # the real list is left as it was


# --- applying and undoing -----------------------------------------------------------------------------------------

def _tested(*edits):
    return evolve.Proposal("Düzeltme", list(edits), 1, True, "ok", "hedef", fingerprint=evolve.code_fingerprint())


def test_apply_backs_up_and_undo_restores(project):
    demo = project / "visionflow/nodes/demo.py"
    prop = _tested(evolve.Edit(FIX["dosya"], FIX["eski"], FIX["yeni"]),
                  evolve.Edit("tests/test_demo.py", "", "def test_x():\n    pass\n", True))
    evolve.apply(prop)
    assert "w * h" in demo.read_text() and (project / "tests/test_demo.py").exists()
    h = evolve.history()
    assert h[0]["id"] == prop.id and h[0]["undone"] is False
    with pytest.raises(evolve.EvolveError):
        evolve.apply(prop)                                             # once only
    evolve.undo(prop.id)
    assert "w + h" in demo.read_text() and not (project / "tests/test_demo.py").exists()
    assert evolve.history()[0]["undone"]


def test_apply_refuses_untested_or_stale_changes(project):
    with pytest.raises(evolve.EvolveError):
        evolve.apply(evolve.Proposal("x", [evolve.Edit(FIX["dosya"], FIX["eski"], FIX["yeni"])], 1, None))
    prop = _tested(evolve.Edit(FIX["dosya"], FIX["eski"], FIX["yeni"]))
    (project / "visionflow/nodes/demo.py").write_text("def area(w, h):\n    return 0\n", encoding="utf-8")
    with pytest.raises(evolve.EvolveError, match="artık"):
        evolve.apply(prop)


def test_undo_refuses_odd_ids():
    for bad in ("../x", "", "a/b"):
        with pytest.raises(evolve.EvolveError):
            evolve.undo(bad)


# --- daily self-review --------------------------------------------------------------------------------------------

def test_daily_review_is_off_by_default_and_then_daily():
    now = dt.datetime(2026, 10, 12, 9, 0)
    assert evolve.daily_due(now) is False
    evolve.save_settings({"daily": True, "last": ""})
    assert evolve.daily_due(now) is True
    evolve.save_settings({"daily": True, "last": (now - dt.timedelta(hours=5)).isoformat()})
    assert evolve.daily_due(now) is False
    assert evolve.daily_due(now + dt.timedelta(hours=20)) is True


def test_daily_review_only_prepares_a_proposal(project, monkeypatch):
    from visionflow.core import registry
    monkeypatch.setattr(registry, "PLUGIN_ERRORS", {})                # earlier tests may have left plugin errors
    monkeypatch.setattr(evolve, "run_checks", lambda root=None, timeout=900: (True, "ok"))
    monkeypatch.setattr(evolve, "try_in_copy", lambda edits: (True, "ok"))
    evolve.save_settings({"daily": True, "last": ""})
    assert evolve.daily_review(ask=lambda p, s, h: pytest.fail("no errors: Claude is not asked")) is None
    evolve.error_log().record("Area", "yanlış alan", "visionflow/nodes/demo.py:2 area")
    prop = evolve.daily_review(ask=lambda p, s, h: answer(FIX))
    assert prop is not None and evolve.load_pending().id == prop.id
    assert "w + h" in (project / "visionflow/nodes/demo.py").read_text()          # not applied
    assert evolve.daily_due() is False
    evolve.apply(evolve.load_pending())
    assert evolve.load_pending() is None


def test_files_for_includes_error_locations(project):
    rep = {**REPORT, "errors": [{"block": "X", "error": "e", "where": "visionflow/nodes/demo.py:2 area", "count": 1}]}
    assert "visionflow/nodes/demo.py" in evolve._files_for(rep, "")


def test_diff_keeps_lines_apart(project):
    d = _tested(evolve.Edit(FIX["dosya"], FIX["eski"], FIX["yeni"])).diff()
    assert "-return w + h\n+return w * h" in d and "satır 2" in d


def test_proposal_roundtrip():
    p = _tested(evolve.Edit("plugins/a.py", "", "x = 1", True))
    q = evolve.Proposal.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.to_dict() == p.to_dict()
    assert evolve.Proposal.from_dict({"summary": "s"}).id


# --- window -------------------------------------------------------------------------------------------------------

def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(cond, app, secs=10):
    import time
    end = time.time() + secs
    while not cond() and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_dialog_proposes_and_applies(project, monkeypatch):
    app = qapp()
    from PySide6.QtWidgets import QMessageBox
    from visionflow.ui.evolve_dialog import EvolveDialog
    monkeypatch.setattr(evolve, "try_in_copy", lambda edits: (True, "== Testler: GEÇTİ"))
    monkeypatch.setattr(evolve, "run_checks", lambda root=None, timeout=900: (True, "ok"))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dlg = EvolveDialog(ask=lambda p, s, h: answer(FIX))
    assert not dlg.btn_apply.isEnabled() and not dlg.daily.isChecked()
    dlg.goal.setText("alan yanlış")
    dlg.request()
    assert wait_for(lambda: dlg.btn_apply.isEnabled(), app), dlg.status.text()
    assert "w * h" in dlg.diff.toPlainText() and "GEÇTİ" in dlg.tests.toPlainText()
    dlg.apply()
    assert "w * h" in (project / "visionflow/nodes/demo.py").read_text()
    assert dlg.hist.topLevelItemCount() == 1 and "yeniden başlatın" in dlg.status.text()
    dlg.hist.setCurrentItem(dlg.hist.topLevelItem(0))
    dlg.undo_selected()
    assert "w + h" in (project / "visionflow/nodes/demo.py").read_text()
    dlg.daily.setChecked(True)
    assert evolve.settings()["daily"] is True
    dlg.close()


def test_dialog_shows_waiting_proposal(project):
    qapp()
    from visionflow.ui.evolve_dialog import EvolveDialog
    evolve.save_pending(_tested(evolve.Edit(FIX["dosya"], FIX["eski"], FIX["yeni"])))
    dlg = EvolveDialog(ask=lambda p, s, h: "")
    assert dlg.btn_apply.isEnabled() and "bekleyen" in dlg.status.text()
    dlg.close()


def test_main_window_has_menu_entry():
    qapp()
    from visionflow.ui.main_window import MainWindow
    win = MainWindow()
    try:
        names = [a.text() for a in win.findChildren(type(win.menuBar().actions()[0]))]
        assert any("Kendini Geliştir" in n for n in names)
        dlg = win.open_evolve()
        dlg.close()
    finally:
        win.close()


# --- 1.2 bug hunt ---------------------------------------------------------------------------------------------------

def test_code_reaching_for_the_rules_is_refused(project):
    e = evolve.Edit("plugins/x.py", "", "import visionflow.evolve as e\ne.PROTECTED.clear()\n", True)
    assert "evolve" in evolve.check_edits([e])[0]


def test_edits_to_one_file_are_checked_in_order(project):
    demo = project / "visionflow/nodes/demo.py"
    demo.write_text("def g():\n    return 1\n", encoding="utf-8")
    first = evolve.Edit(FIX["dosya"], "def g():", "def g0():\n    return 1\n# end\ndef g():")
    second = evolve.Edit(FIX["dosya"], "return 1\n", "return 9\n")       # now twice in the file
    assert "2 kez" in " ".join(evolve.check_edits([first, second]))
    removed = evolve.Edit(FIX["dosya"], "    return 1\n", "")
    assert "bulunamadı" in " ".join(evolve.check_edits([removed, evolve.Edit(FIX["dosya"], "return 1", "return 2")]))
    ok = [evolve.Edit(FIX["dosya"], "def g():", "def h():"), evolve.Edit(FIX["dosya"], "return 1", "return 2")]
    assert evolve.check_edits(ok) == []
    evolve.apply(_tested(*ok))
    assert demo.read_text() == "def h():\n    return 2\n"


def test_undo_never_wipes_a_newer_change(project):
    demo = project / "visionflow/nodes/demo.py"
    a = _tested(evolve.Edit(FIX["dosya"], "return w + h", "return w * h"))
    evolve.apply(a)
    b = _tested(evolve.Edit(FIX["dosya"], "return w * h", "return w * h * 1"))
    evolve.apply(b)
    with pytest.raises(evolve.EvolveError, match="yeniden değişti"):
        evolve.undo(a.id)
    assert "w * h * 1" in demo.read_text()
    evolve.undo(b.id)
    evolve.undo(a.id)
    assert "w + h" in demo.read_text()


def test_undo_ignores_tampered_paths(project, tmp_path):
    a = _tested(evolve.Edit(FIX["dosya"], "return w + h", "return w * h"))
    folder = evolve.apply(a)
    meta = json.loads((folder / "bilgi.json").read_text(encoding="utf-8"))
    meta["backups"].append({"path": "visionflow/ai/config.py", "existed": False})
    (folder / "bilgi.json").write_text(json.dumps(meta), encoding="utf-8")
    (project / "visionflow/ai").mkdir()
    (project / "visionflow/ai/config.py").write_text("x = 1\n", encoding="utf-8")
    evolve.undo(a.id)
    assert (project / "visionflow/ai/config.py").exists()


def test_failed_write_puts_everything_back(project, monkeypatch):
    (project / "visionflow/nodes/b.py").write_text("B = 1\n", encoding="utf-8")
    prop = _tested(evolve.Edit(FIX["dosya"], "return w + h", "return w * h"), evolve.Edit("visionflow/nodes/b.py", "B = 1", "B = 2"))
    real = pathlib.Path.write_text

    def flaky(self, *a, **k):
        if self.name == "b.py":
            raise PermissionError("dosya kilitli")
        return real(self, *a, **k)
    monkeypatch.setattr(pathlib.Path, "write_text", flaky)
    with pytest.raises(evolve.EvolveError, match="hiçbir şey değişmedi"):
        evolve.apply(prop)
    monkeypatch.setattr(pathlib.Path, "write_text", real)
    assert "w + h" in (project / FIX["dosya"]).read_text() and evolve.history() == []
    evolve.apply(prop)                                                    # can be retried
    assert "B = 2" in (project / "visionflow/nodes/b.py").read_text()


def test_proposal_tested_on_other_code_must_be_retested(project, monkeypatch):
    prop = _tested(evolve.Edit(FIX["dosya"], "return w + h", "return w * h"))
    (project / "visionflow/nodes/other.py").write_text("X = 1\n", encoding="utf-8")   # e.g. another change
    assert evolve.needs_retest(prop)
    with pytest.raises(evolve.EvolveError, match="yeniden test"):
        evolve.apply(prop)
    monkeypatch.setattr(evolve, "try_in_copy", lambda edits: (True, "ok"))
    evolve.apply(evolve.retest(prop))
    assert "w * h" in (project / FIX["dosya"]).read_text()


def test_packaged_program_never_loads_a_proposed_plugin_itself(project, monkeypatch, tmp_path):
    from visionflow.core.registry import NODE_REGISTRY
    monkeypatch.setattr(evolve, "frozen", lambda: True)
    marker = tmp_path / "ran.txt"
    code = (f"open({str(marker)!r}, 'w').write('x')\n"
            "from visionflow.core import register\nfrom visionflow.nodes.processing import Grayscale\n"
            "@register\nclass Grayscale(Grayscale):\n    EVIL = True\n")
    before = NODE_REGISTRY["Grayscale"]
    ok, out = evolve.try_in_copy([evolve.Edit("plugins/evil.py", "", code, True)])
    assert ok, out
    assert NODE_REGISTRY["Grayscale"] is before                         # the running program is untouched
    assert marker.exists()                                              # it did run, in a separate process


def test_daily_review_does_not_ask_twice_about_the_same_error(project, monkeypatch):
    from visionflow.core import registry
    monkeypatch.setattr(registry, "PLUGIN_ERRORS", {})
    monkeypatch.setattr(evolve, "run_checks", lambda root=None, timeout=900: (True, "ok"))
    monkeypatch.setattr(evolve, "try_in_copy", lambda edits: (True, "ok"))
    evolve.error_log().record("Camera", "kamera 0 açılamadı", "")
    calls = []
    for _ in range(3):
        evolve.daily_review(ask=lambda p, s, h: calls.append(1) or answer())
    assert len(calls) == 1
    evolve.error_log().record("Blur", "yeni hata", "")
    evolve.daily_review(ask=lambda p, s, h: calls.append(1) or answer())
    assert len(calls) == 2


def test_error_log_ignores_changing_numbers_and_stays_small():
    log = evolve.error_log()
    log.record("Camera", "kamera kayboldu", "")
    for i in range(3000):
        log.record("Zaman", f"ValueError: t={i}.{i * 7} (kare {i})" + "x" * 900, "")
    assert {e["block"] for e in log.summary()} == {"Camera", "Zaman"}
    for i in range(3000):
        log._seen.clear()
        log.record(f"B{i % 50}", f"hata {i}" + "y" * 1200, "")
    assert log.path.stat().st_size < 420_000


def test_where_skips_user_scripts_and_venv(tmp_path):
    from visionflow.core.node import where
    ns = {}
    exec(compile("def boom():\n    raise ValueError('x')\n", "<Python Kodu ab12>", "exec"), ns)
    venv = evolve.SOURCE_ROOT / ".venv_test_where"
    try:
        venv.mkdir(exist_ok=True)
        (venv / "fakenp.py").write_text("def boom():\n    raise ValueError('np')\n", encoding="utf-8")
        import importlib.util
        spec = importlib.util.spec_from_file_location("fakenp_where", venv / "fakenp.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fn in (ns["boom"], mod.boom):
            try:
                fn()
            except ValueError as exc:
                w = where(exc)
            assert "Python Kodu" not in w and ".venv" not in w
            assert w.startswith("tests/") is False and "test_v12.py" not in w
    finally:
        import shutil
        shutil.rmtree(venv, ignore_errors=True)


def test_setup_errors_have_a_location():
    from visionflow.core import create
    n = create("ImageFile")
    n.values["path"] = "yok.png"
    n.setup = lambda: n._open_missing()                       # a failing setup inside the program's code
    n._open_missing = lambda: evolve.allowed(None)            # AttributeError raised in visionflow/evolve.py
    g = Graph()
    g.add(n)
    g.setup()
    r = g.run_once()[n.id]
    assert r.error and "visionflow/evolve.py" in r.where


def test_main_window_print_reaches_the_log():
    qapp()
    from visionflow.ui.main_window import MainWindow
    import sys
    real = sys.stdout
    win = MainWindow()
    try:
        got = []
        win.stream.text.connect(got.append)
        print("merhaba")                                    # a block's print() must not raise
        assert "merhaba" in "".join(got)
        g = Graph()
        g.add_new("ImageFile", path="yok.png")
        g.add_new("Number", value=1)
        assert g.setup() is not None                       # Graph.setup prints setup errors
    finally:
        sys.stdout = real
        win.close()


def test_dialog_is_single_and_can_discard_a_waiting_proposal(project):
    qapp()
    from visionflow.ui.evolve_dialog import EvolveDialog
    evolve.save_pending(_tested(evolve.Edit(FIX["dosya"], FIX["eski"], FIX["yeni"])))
    dlg = EvolveDialog(ask=lambda p, s, h: "")
    dlg.discard()
    assert evolve.load_pending() is None and not dlg.btn_apply.isEnabled()
    dlg.close()
    stale = _tested(evolve.Edit(FIX["dosya"], "return 42", "return 0"))     # no longer fits the file
    evolve.save_pending(stale)
    EvolveDialog(ask=lambda p, s, h: "").close()
    assert evolve.load_pending() is None
