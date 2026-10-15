"""Installation self-test: ``VisionFlow --selftest`` (or ``python -m visionflow --selftest``).

Checks that every block loads, the examples open, a small flow runs, the optional AI/office packages are there
and the editor window can be built. The report is printed and also written to a file, because the Windows
build has no console window.
"""
from __future__ import annotations

import importlib
import pathlib
import sys
import time
import traceback

OPTIONAL = [("anthropic", "Claude (AI Merkezi)"), ("edge_tts", "doğal Türkçe ses"), ("pyttsx3", "internetsiz ses"),
            ("imageio_ffmpeg", "video üretme"), ("pypdf", "PDF okuma"), ("docx", "Word okuma"),
            ("openpyxl", "Excel'e kaydet"), ("onnxruntime", "YOLO / ONNX modelleri")]


def run(out_path: str | None = None, ui: bool = True) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    def check(label: str, fn, essential: bool = True):
        nonlocal ok
        t0 = time.perf_counter()
        try:
            detail = fn()
            lines.append(f"[TAMAM] {label}{': ' + str(detail) if detail else ''} ({(time.perf_counter() - t0):.1f} sn)")
        except Exception as exc:  # noqa: BLE001
            ok = ok and not essential
            lines.append(f"[{'HATA' if essential else 'EKSİK'}] {label}: {type(exc).__name__}: {exc}")
            if essential:
                lines.append("        " + traceback.format_exc().strip().splitlines()[-2].strip())

    from . import __version__
    from .core import Graph, load_builtin
    from .core.registry import NODE_REGISTRY
    lines.append(f"VisionFlow Studio {__version__} kurulum testi · Python {sys.version.split()[0]} · {sys.platform}")

    check("Bloklar", lambda: (load_builtin(), f"{len(NODE_REGISTRY)} blok")[1])

    def examples():
        from .ui.main_window import EXAMPLES
        files = sorted(pathlib.Path(EXAMPLES).glob("*.json"))
        if not files:
            raise FileNotFoundError(f"örnek akış bulunamadı: {EXAMPLES}")
        bad = [f"{f.name}: {w}" for f in files for w in Graph.load(str(f)).warnings]
        if bad:
            raise ValueError("; ".join(bad[:3]))
        return f"{len(files)} örnek açıldı"
    check("Örnek akışlar", examples)

    def flow():
        g = Graph()
        a, b, c = g.add_new("TestPattern"), g.add_new("Blur"), g.add_new("Canny")
        g.connect(a.id, "image", b.id, "image")
        g.connect(b.id, "image", c.id, "image")
        errors = []
        frames = g.run(max_frames=5, on_frame=lambda i, r: errors.extend(x.error for x in r.values() if x.error))
        if errors or frames != 5:
            raise RuntimeError(errors[0] if errors else f"{frames} kare")
        return "5 kare işlendi"
    check("Görüntü işleme (OpenCV)", flow)

    def ffmpeg():
        from .ai.media import ffmpeg_exe
        return ffmpeg_exe()
    for mod, what in OPTIONAL:
        check(f"{what} ({mod})", lambda m=mod: getattr(importlib.import_module(m), "__version__", "var"),
              essential=False)
    check("ffmpeg", ffmpeg, essential=False)

    def gpu():
        from .accel import status_line
        return status_line()
    check("Hızlandırma (GPU)", gpu, essential=False)

    def home_dir():
        from .ai.config import home
        h = home()
        h.mkdir(parents=True, exist_ok=True)
        probe = h / "yazma_testi.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return str(h)
    check("Ayar klasörü yazılabilir", home_dir)

    if ui:
        def window():
            from PySide6.QtWidgets import QApplication
            from .ui.main_window import MainWindow
            app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841
            w = MainWindow()
            n = sum(1 for _ in w.palette.findChildren(object))
            w.saved_snap = w._snap()
            w.close()
            return f"pencere kuruldu ({n} öğe)"
        check("Editör penceresi (Qt)", window)

    lines.append("SONUÇ: " + ("her şey çalışıyor" if ok else "sorun var, yukarıdaki HATA satırlarına bakın"))
    report = "\n".join(lines)
    if out_path is None:
        from .ai.config import home
        out_path = str(home() / "kurulum_testi.txt")
    try:
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_path).write_text(report + "\n", encoding="utf-8")
        lines.append(f"Rapor: {out_path}")
    except OSError:
        pass
    return ok, lines


def main(argv=None) -> int:
    argv = list(argv or [])
    out = argv[argv.index("--out") + 1] if "--out" in argv and argv.index("--out") + 1 < len(argv) else None
    ok, lines = run(out, ui="--no-ui" not in argv)
    for line in lines:
        try:
            print(line)
        except (OSError, ValueError, AttributeError):   # windowed .exe: no console to print to
            pass
    return 0 if ok else 1
