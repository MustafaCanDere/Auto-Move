import os
import pathlib
import sys
import tempfile

# keep crash-recovery files of the test run away from the user's real profile
os.environ.setdefault("VISIONFLOW_HOME", tempfile.mkdtemp(prefix="visionflow-test-"))


_status = {"code": 0}


def pytest_sessionfinish(session, exitstatus):
    _status["code"] = int(exitstatus)


def _run_from_command_line(config) -> bool:
    """`pytest ...` or `python -m pytest ...`, not pytest.main() inside another program (an IDE, a script),
    not an xdist worker, and not under coverage (which saves its data at exit)."""
    argv0 = pathlib.Path(sys.argv[0] if sys.argv else "")
    cli = argv0.stem in ("pytest", "py.test") or (argv0.name == "__main__.py" and argv0.parent.name == "pytest")
    return cli and not hasattr(config, "workerinput") and "coverage" not in sys.modules \
        and not os.environ.get("VISIONFLOW_NORMAL_EXIT")


def pytest_unconfigure(config):
    if "visionflow.ai.media" in sys.modules:
        try:
            sys.modules["visionflow.ai.media"].stop_tts_worker()
        except Exception:
            pass
    if sys.platform != "win32" and "PySide6.QtWidgets" in sys.modules:
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.closeAllWindows()
                app.quit()
        except Exception:
            pass
    if _run_from_command_line(config):
        sys.stdout.flush()
        sys.stderr.flush()
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.ExitProcess(int(_status["code"]))
        else:
            os._exit(_status["code"])

