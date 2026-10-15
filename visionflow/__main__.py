"""Start the editor: ``python -m visionflow``"""
import pathlib
import sys

from .core import load_builtin, load_plugins


HELP = """VisionFlow Studio — kullanım:

  python -m visionflow [akis.json]              Düzenleyiciyi açar (isteğe bağlı olarak bir akışla)
  python -m visionflow --operator akis.json     Yalnız operatör paneli: tam ekran, akış çalışır
  python -m visionflow --headless akis.json     Akışı arayüz olmadan çalıştırır (--help ile seçenekler)
  python -m visionflow servis kur --akis akis.json
                                                Akışı bilgisayar açılınca çalıştırır (kur, kaldir, baslat,
                                                durdur, durum; --help ile seçenekler; --api PORT ile REST API,
                                                --istasyonlar hat.json ile bütün istasyonlar)
  python -m visionflow istasyonlar hat.json     İstasyon dosyasındaki akışları (kameraları) aynı anda, her birini
                                                ayrı işlemde çalıştırır; çökeni yeniden başlatır (--help)
  python -m visionflow test-akis akis.json --kayit KLASÖR --beklenen beklenen.json
                                                Kaydı akıştan geçirip kararları beklenenle karşılaştırır
  python -m visionflow arsiv --seri ABC123     İzlenebilirlik arşivinde arar (--disa-aktar out.csv --baslangic
                                                2026-10-01 ile dışa aktarır; --help ile seçenekler)
  python -m visionflow veriseti istatistik KLASÖR
                                                Veri seti araçları: istatistik, bol, donustur, etiketle
                                                (--help ile seçenekler)
  python -m visionflow gpu-test                 Hızlandırma cihazlarını (CUDA, OpenVINO…) listeler, hız testi yapar
  python -m visionflow --selftest               Kurulumu ve blokları kendi kendine dener
  python -m visionflow --check-plugins A B      (iç kullanım) önerilen eklentiyi dener
  python -m visionflow -h | --help              Bu yardımı gösterir

Paketlenmiş programda 'python -m visionflow' yerine VisionFlow.exe yazın."""

KNOWN = ("--selftest", "--check-plugins", "--headless", "servis", "--servis", "test-akis", "--test-akis", "--operator",
         "--gpu-test", "--arsiv", "--veriseti", "istasyonlar", "--istasyonlar")


def _turkish_qt(app) -> None:
    """Qt's own texts (Close, Cancel, Yes/No buttons, file dialogs) in Turkish; silently skipped if missing."""
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

    tr = QTranslator(app)
    if tr.load(QLocale("tr_TR"), "qtbase", "_", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(tr)
        app._vf_translator = tr                 # keep a reference


def main() -> int:
    args = sys.argv[1:]
    if args[:1] in (["-h"], ["--help"], ["/?"]):     # before Qt: works on a PC without a screen
        print(HELP)
        return 0
    if args[:1] and args[0].startswith("--") and args[0] not in KNOWN:
        print(f"Bilinmeyen seçenek: {args[0]}\n\n{HELP}", file=sys.stderr)
        return 2
    # the packaged .exe has no python -m: these switches give it the command-line tools too
    if args[:1] == ["--selftest"]:
        from .selftest import main as selftest
        return selftest(args[1:])
    if args[:1] == ["--check-plugins"] and len(args) == 3:     # Kendini Geliştir tries a proposed plugin here
        from .evolve import check_plugins
        return check_plugins(args[1], args[2])
    if args[:1] == ["--headless"]:
        from .headless import main as headless
        return headless(args[1:])
    if args[:1] in (["servis"], ["--servis"]):          # run a flow at boot (service / task / systemd)
        from .service import main as service
        return service(args[1:])
    if args[:1] in (["istasyonlar"], ["--istasyonlar"]):     # several flows at once, each in its own process
        from .stations import main as stations
        return stations(args[1:])
    if args[:1] in (["test-akis"], ["--test-akis"]):    # replay a recording and compare the decisions
        from .flowtest import main as flowtest
        return flowtest(args[1:])
    if args[:1] in (["gpu-test"], ["--gpu-test"]):     # inference devices and a small speed test
        from .accel import main as gpu_test
        return gpu_test(args[1:])
    if args[:1] in (["arsiv"], ["--arsiv"]):           # part traceability archive: search / export
        from .trace import cli as trace_cli
        return trace_cli(args[1:])
    if args[:1] in (["veriseti"], ["--veriseti"]):     # dataset tools: stats, split, convert, auto-label
        from .datasets import cli as dataset_cli
        return dataset_cli(args[1:])
    # the editor (Qt) is imported only here: the command-line tools above also run on a PC without a screen
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow
    from .ui.theme import STYLESHEET
    load_builtin()
    # next to the .exe when frozen, else the working folder
    base = pathlib.Path(sys.executable).parent if getattr(sys, "frozen", False) else pathlib.Path.cwd()
    plugins = load_plugins(base / "plugins")
    app = QApplication(sys.argv)
    app.setApplicationName("VisionFlow Studio")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    _turkish_qt(app)
    icon_path = pathlib.Path(__file__).resolve().parent / "resources" / "app_icon.png"
    if not icon_path.exists():
        icon_path = pathlib.Path(__file__).resolve().parent.parent / "installer" / "app_icon.png"
    if icon_path.exists():
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(icon_path)))
    if args[:1] == ["--operator"]:
        return _operator(app, args[1:], plugins)
    win = MainWindow()
    if plugins:
        print(f"Eklentiler yüklendi: {', '.join(plugins)}")
    from .core.registry import PLUGIN_ERRORS
    for name, err in PLUGIN_ERRORS.items():         # printed again now that the log panel shows it
        print(f"HATA: eklenti {name} yüklenemedi: {err}")
    win.show()
    if not win.check_recovery():
        if len(sys.argv) > 1:
            win.open_flow(sys.argv[1])
        else:                                       # first start: the template gallery ("Yeni akış")
            from .ui.template_gallery import maybe_show_on_start
            maybe_show_on_start(win)
    return app.exec()


def _operator(app, args, plugins) -> int:
    """``--operator akis.json``: only the operator panel, full screen, with the flow running. The editor opens from
    the panel (after the password when the flow is locked)."""
    from PySide6.QtWidgets import QMessageBox
    from .ui.operator import run_operator
    if len(args) != 1:
        print("Kullanım: python -m visionflow --operator akis.json")
        return 2
    path = args[0]

    def editor():
        from .ui.main_window import MainWindow
        win = MainWindow()
        win.open_flow(path, unlocked=True)
        return win

    try:
        panel = run_operator(path, editor_factory=editor)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "Operatör paneli", f"Akış açılamadı:\n{path}\n\n{exc}")
        return 1
    if plugins:
        print(f"Eklentiler yüklendi: {', '.join(plugins)}")
    from .ui.api_dialog import autostart
    autostart(panel, log=print)            # the remote control API, when "program açılınca da açılsın" is set
    panel.showFullScreen()
    panel.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
