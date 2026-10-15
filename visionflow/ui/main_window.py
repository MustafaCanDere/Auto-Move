"""Main editor window."""
from __future__ import annotations

import json
import os
import pathlib
import sys
import threading

from PySide6.QtCore import QMimeData, QObject, QPointF, QSettings, QStandardPaths, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QScrollArea, QSpinBox,
                               QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from .. import __version__

from ..core import Graph, by_category
from . import editor_tools
from .panels import LogPanel, Palette, Preview, Properties, QuickAdd
from .scene import NodeItem, NodeScene, NodeView
from .worker import FlowWorker

# in a PyInstaller build the bundled data lives under sys._MEIPASS
EXAMPLES = pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(__file__).resolve().parents[2])) / "examples"
CLIP_MIME = "application/x-visionflow-clip"
AUTOSAVE_MS = 30_000
SHORTCUTS = [
    ("Blok ekle", "Soldan sürükle-bırak, çift tık veya Tab / Ctrl+Space ile hızlı ekle"),
    ("Bağla", "Çıkış noktasından girişe sürükle"),
    ("Çalıştır / Durdur", "F5 / Shift+F5"),
    ("Operatör paneli (tam ekran: F11)", "F9"),
    ("Geri al / Yinele", "Ctrl+Z / Ctrl+Y"),
    ("Kopyala / Yapıştır / Çoğalt", "Ctrl+C / Ctrl+V / Ctrl+D"),
    ("Sil", "Del"),
    ("Tümünü seç", "Ctrl+A"),
    ("Devre dışı bırak / aç", "Ctrl+E"),
    ("Otomatik yerleştir", "Ctrl+Shift+L"),
    ("Akışı denetle (eksik bağlantı, ayar)", "Ctrl+Shift+D"),
    ("Hepsini sığdır", "F"),
    ("Blok ara", "Ctrl+F"),
    ("Akışta ara (bloğa git)", "Ctrl+P"),
    ("Hatalı bloğa git (sırayla)", "F8"),
    ("Hizala / dağıt, blok rengi", "Düzen menüsü veya sağ tık"),
    ("Kabloyu boşluğa bırak", "Uyan bloklar listelenir, seçilen eklenip bağlanır"),
    ("Bloğu kablonun üzerine bırak", "Blok bağlantının arasına eklenir"),
    ("Yakınlaş / kaydır", "Fare tekerleği / sağ veya orta tuşla sürükle"),
    ("Yeni / Aç / Kaydet", "Ctrl+N / Ctrl+O / Ctrl+S"),
    ("AI Merkezi (sohbet, hafıza, ayarlar)", "Ctrl+M"),
    ("Yapay zeka ile akış kur", "Ctrl+K"),
    ("Etiketleme / Model eğit", "Ctrl+L / Ctrl+T"),
    ("Örnekleri öğret (programın emin olmadığı görüntüler)", "Ctrl+Shift+O"),
    ("Yorum çerçevesi (seçilen blokların etrafına)", "Ctrl+G · başlığından sürükle, köşesinden boyutlandır"),
    ("Alt akış yap (seçilenleri tek blokta topla)", "Ctrl+Alt+G · çift tıkla içine gir, Alt+↑ ile üst akışa dön"),
    ("Kendini Geliştir (programın kendi kodunu iyileştirmesi)", "Ctrl+Shift+G"),
    ("Endüstriyel kameraları bul (GigE / USB3 Vision)", "Araçlar menüsü"),
    ("Bu yardım", "F1"),
]


def _set_flow_dir(path) -> None:
    """Relative file settings of the open flow are looked up next to its file (see core.node.resolve_path)."""
    from ..core import node as _node
    _node.FLOW_DIR = pathlib.Path(path).resolve().parent if path else None
    _node.FLOW_FILE = pathlib.Path(path).resolve() if path else None


def recovery_file() -> pathlib.Path:
    base = os.environ.get("VISIONFLOW_HOME") or QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    return pathlib.Path(base or pathlib.Path.home() / ".visionflow") / "kurtarma.json"
MAX_HISTORY = 100


class _Stream(QObject):
    text = Signal(str)

    def write(self, s):
        self.text.emit(s)

    def flush(self):
        pass


class _Note(QObject):
    text = Signal(str)


class History:
    """Snapshot-based undo/redo over the graph's JSON form."""

    def __init__(self):
        self.stack: list[str] = []
        self.index = -1

    def reset(self, snap: str):
        self.stack, self.index = [snap], 0

    def push(self, snap: str) -> bool:
        if self.index >= 0 and self.stack[self.index] == snap:
            return False
        del self.stack[self.index + 1:]
        self.stack.append(snap)
        if len(self.stack) > MAX_HISTORY:
            self.stack.pop(0)
        self.index = len(self.stack) - 1
        return True

    def undo(self) -> str | None:
        if self.index > 0:
            self.index -= 1
            return self.stack[self.index]
        return None

    def redo(self) -> str | None:
        if self.index < len(self.stack) - 1:
            self.index += 1
            return self.stack[self.index]
        return None


class PerformanceHUD(QWidget):
    """Modern real-time performance and system status HUD widget for the status bar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(8)

        self.fps_badge = QLabel("⚪ Durdu")
        self.fps_badge.setStyleSheet(
            "background: #181d28; color: #94a3b8; border: 1px solid #2a3346; "
            "border-radius: 9px; padding: 2px 10px; font-weight: 600; font-size: 8.5pt;"
        )
        layout.addWidget(self.fps_badge)

        self.ms_badge = QLabel("⚡ 0.0 ms")
        self.ms_badge.setStyleSheet(
            "background: #0f1c2e; color: #38bdf8; border: 1px solid #16365c; "
            "border-radius: 9px; padding: 2px 8px; font-weight: 500; font-size: 8.5pt;"
        )
        layout.addWidget(self.ms_badge)

        self.nodes_badge = QLabel("🧩 0 Blok")
        self.nodes_badge.setStyleSheet(
            "background: #0c1f17; color: #34d399; border: 1px solid #14422e; "
            "border-radius: 9px; padding: 2px 8px; font-weight: 500; font-size: 8.5pt;"
        )
        layout.addWidget(self.nodes_badge)

        self.err_badge = QLabel("")
        self.err_badge.setVisible(False)
        self.err_badge.setStyleSheet(
            "background: #2d1217; color: #f87171; border: 1px solid #571e29; "
            "border-radius: 9px; padding: 2px 8px; font-weight: 600; font-size: 8.5pt;"
        )
        layout.addWidget(self.err_badge)

    def set_stopped(self, node_count=0):
        self.fps_badge.setText("⚪ Durdu")
        self.fps_badge.setStyleSheet(
            "background: #181d28; color: #94a3b8; border: 1px solid #2a3346; "
            "border-radius: 9px; padding: 2px 10px; font-weight: 600; font-size: 8.5pt;"
        )
        self.ms_badge.setText("⚡ 0.0 ms")
        self.nodes_badge.setText(f"🧩 {node_count} Blok")
        self.err_badge.setVisible(False)

    def set_running(self):
        self.fps_badge.setText("🔄 Başlıyor…")
        self.fps_badge.setStyleSheet(
            "background: #0f2b3d; color: #38bdf8; border: 1px solid #0284c7; "
            "border-radius: 9px; padding: 2px 10px; font-weight: 600; font-size: 8.5pt;"
        )

    def update_stats(self, fps: float, total_ms: float, errors: list[str], node_count: int):
        if fps >= 25.0:
            color = "#34d399"
            bg = "#0c281e"
            border = "#059669"
            txt = f"🟢 {fps:.1f} FPS"
        elif fps >= 12.0:
            color = "#fbbf24"
            bg = "#292110"
            border = "#d97706"
            txt = f"🟡 {fps:.1f} FPS"
        else:
            color = "#f87171"
            bg = "#301318"
            border = "#dc2626"
            txt = f"🔴 {fps:.1f} FPS"

        self.fps_badge.setText(txt)
        self.fps_badge.setStyleSheet(
            f"background: {bg}; color: {color}; border: 1px solid {border}; "
            "border-radius: 9px; padding: 2px 10px; font-weight: 600; font-size: 8.5pt;"
        )
        self.ms_badge.setText(f"⚡ {total_ms:.1f} ms/kare")
        self.nodes_badge.setText(f"🧩 {node_count} Blok")

        if errors:
            self.err_badge.setVisible(True)
            self.err_badge.setText(f"⛔ {len(errors)} Hata")
            self.err_badge.setToolTip("\n".join(errors))
        else:
            self.err_badge.setVisible(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1500, 900)
        # Window icon
        icon_path = pathlib.Path(__file__).resolve().parent.parent / "resources" / "app_icon.png"
        if not icon_path.exists():
            icon_path = pathlib.Path(__file__).resolve().parents[2] / "installer" / "app_icon.png"
        if icon_path.exists():
            app_icon = QIcon(str(icon_path))
            self.setWindowIcon(app_icon)
            QGuiApplication.setWindowIcon(app_icon)
        self.path: str | None = None
        self.worker: FlowWorker | None = None
        self.history = History()
        self.saved_snap = ""
        self.levels: list[tuple] = []     # open sub-flows, outermost first: (sub-flow block, its inner graph)
        self._root: Graph | None = None
        self._navigating = False          # showing another level of the flow is not an edit: no restart
        # VISIONFLOW_HOME (portable install, tests) keeps window settings and recent files next to everything else
        home = os.environ.get("VISIONFLOW_HOME")
        self.settings = QSettings(os.path.join(home, "pencere.ini"), QSettings.IniFormat) if home else \
            QSettings("VisionFlow", "Studio")

        self.scene = NodeScene()
        self.view = NodeView(self.scene)
        self.setCentralWidget(self.view)

        self.palette = Palette()
        self.props = Properties()
        self.preview = Preview()
        self.logp = LogPanel()
        self._dock("Bloklar", self.palette, Qt.LeftDockWidgetArea)
        preview_dock = self._dock("Önizleme", self.preview, Qt.RightDockWidgetArea)
        props_scroll = QScrollArea()
        props_scroll.setWidget(self.props)
        props_scroll.setWidgetResizable(True)
        props_scroll.setFrameShape(QScrollArea.NoFrame)
        props_dock = self._dock("Ayarlar", props_scroll, Qt.RightDockWidgetArea)
        QTimer.singleShot(0, lambda: (self.resizeDocks([props_dock], [380], Qt.Horizontal),
                                      self.resizeDocks([preview_dock, props_dock], [300, 500], Qt.Vertical)))
        log_dock = self._dock("Günlük", self.logp, Qt.BottomDockWidgetArea)
        QTimer.singleShot(0, lambda: self.resizeDocks([log_dock], [110], Qt.Vertical))

        self.palette.add_requested.connect(self.add_at_center)
        self.palette.create_custom_requested.connect(self.open_custom_block_studio)
        self.props.param_changed.connect(self.on_param)
        self.props.expose_state = self._expose_state
        self.props.expose_toggled.connect(self.toggle_exposed)
        self.props.dialog_requested.connect(self.open_block_dialog)
        self.preview.roi_drawn.connect(self.on_roi_drawn)
        self.scene.selectionChanged.connect(self.on_selection)
        self.scene.structure_changed.connect(self.on_structure)
        self.scene.edited.connect(self._schedule_snapshot)
        self.scene.message.connect(lambda m: self.statusBar().showMessage(m, 4000))
        self.scene.subflow_open.connect(self.enter_subflow)
        self.view.context_requested.connect(self.context_menu)
        self.quick_add = QuickAdd(self)
        self._quick_pos = QPointF()
        self.quick_add.chosen.connect(lambda t: editor_tools._quick_chosen(self, t))
        self.view.quick_add_requested.connect(self._show_quick_add)

        self.stream = _Stream()
        self.stream.text.connect(self.logp.log)
        sys.stdout = self.stream

        self.hud = PerformanceHUD(self)
        self.statusBar().addPermanentWidget(self.hud)
        self.fps_label = self.hud.fps_badge
        self._restart_timer = QTimer(self, singleShot=True, interval=300, timeout=self._restart)
        self._snap_timer = QTimer(self, singleShot=True, interval=400, timeout=self._snapshot)
        self._build_actions()
        editor_tools.install(self)        # align, snap, mini map, image export, flow search, F8, theme, colours
        self.new_flow()
        self.recovery = recovery_file()
        self._owns_recovery = False   # only delete a recovery file this window wrote (or restored)
        self._autosave_timer = QTimer(self, interval=AUTOSAVE_MS, timeout=self.autosave)
        self._autosave_timer.start()
        if self.settings.value("geometry") is not None:
            self.restoreGeometry(self.settings.value("geometry"))
            self.restoreState(self.settings.value("window_state"))
        QTimer.singleShot(5000, self._daily_review)
        from .api_dialog import autostart
        QTimer.singleShot(0, self, lambda: autostart(self))    # remote control API, when it was left on

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _act(self, menu, text, slot, shortcut=None, toolbar=None, checkable=False, canvas_only=False):
        a = QAction(text, self, checkable=checkable)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if canvas_only:
            # keys like Del / F / Ctrl+C must not fire while typing in a settings field
            a.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            self.view.addAction(a)
        a.triggered.connect(slot)
        if menu is not None:
            menu.addAction(a)
        if toolbar:
            toolbar.addAction(a)
        return a

    def _build_actions(self):
        tb = self.addToolBar("Ana")
        tb.setObjectName("Ana")
        tb.setMovable(False)
        m = self.menuBar().addMenu("&Dosya")
        self._act(m, "Yeni…", self.new_from_gallery, "Ctrl+N", tb)       # template gallery (ui/template_gallery.py)
        self._act(m, "Boş akış", self.new_flow, "Ctrl+Shift+N")
        self._act(m, "Aç…", self.open_flow, "Ctrl+O", tb)
        self._act(m, "Kaydet", self.save_flow, "Ctrl+S", tb)
        self._act(m, "Farklı Kaydet…", lambda: self.save_flow(True), "Ctrl+Shift+S")
        self.recent_menu = m.addMenu("Son açılanlar")
        self._refresh_recent()
        ex = m.addMenu("Örnekler")
        for f in sorted(EXAMPLES.glob("*.json")):
            ex.addAction(f.stem.replace("_", " "), lambda p=str(f): self.open_flow(p))
        m.addSeparator()
        self._act(m, "Çıkış", self.close, "Ctrl+Q")

        e = self.menuBar().addMenu("Dü&zen")
        self.undo_act = self._act(e, "Geri al", self.undo, "Ctrl+Z", tb)
        self.redo_act = self._act(e, "Yinele", self.redo, "Ctrl+Y", tb)
        e.addSeparator()
        self._act(e, "Kopyala", self.copy, "Ctrl+C", canvas_only=True)
        # lambdas: QAction.triggered passes checked=False, which would land in the first parameter
        self._act(e, "Yapıştır", lambda: self.paste(), "Ctrl+V", canvas_only=True)
        self._act(e, "Çoğalt", self.scene.duplicate_selected, "Ctrl+D", canvas_only=True)
        self._act(e, "Sil", self.scene.delete_selected, "Del", canvas_only=True)
        self._act(e, "Tümünü seç", self.select_all, "Ctrl+A", canvas_only=True)
        self._act(e, "Devre dışı bırak / aç", self.scene.toggle_enabled_selected, "Ctrl+E", canvas_only=True)
        self._act(e, "Yorum çerçevesi ekle (seçilenlerin etrafına)", self.add_frame, "Ctrl+G", canvas_only=True)
        self._act(e, "Alt akış yap (seçilenleri grupla)…", lambda: self.group_selected(), "Ctrl+Alt+G",
                  canvas_only=True)
        self._act(e, "Üst akışa dön", self.exit_subflow, "Alt+Up")
        e.addSeparator()
        self._act(e, "Blok ara", lambda: self.palette.search.setFocus(), "Ctrl+F")

        tb.addSeparator()
        r = self.menuBar().addMenu("&Çalıştır")
        self.run_act = self._act(r, "▶  Çalıştır", self.start, "F5", tb)
        self.stop_act = self._act(r, "■  Durdur", self.stop, "Shift+F5", tb)
        self.stop_act.setEnabled(False)
        r.addSeparator()
        self._act(r, "✔  Akışı denetle…", self.check_flow, "Ctrl+Shift+D", tb)
        r.addSeparator()
        self._act(r, "🏭 Operatör Paneli…", self.open_operator, "F9")
        self._act(r, "Operatör paneli ayarları…", self.edit_operator)
        tb.addWidget(QLabel("  Maks FPS "))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(0, 240)
        self.fps_spin.setValue(int(self.settings.value("max_fps", 30)))
        self.fps_spin.setToolTip("0 = sınırsız. Düşük değer işlemci kullanımını azaltır.")
        self.fps_spin.valueChanged.connect(self.on_fps_limit)
        tb.addWidget(self.fps_spin)

        v = self.menuBar().addMenu("&Görünüm")
        self.thumb_act = self._act(v, "Bloklarda küçük önizleme", self.toggle_thumbs, checkable=True)
        self.thumb_act.setChecked(True)
        self._act(v, "Hepsini sığdır", self.view.fit_all, "F", canvas_only=True)
        self._act(v, "Otomatik yerleştir", self.scene.auto_layout, "Ctrl+Shift+L")

        tb.addSeparator()
        self._act(None, "📷 Resim", lambda: editor_tools.export_dialog(self), None, tb)

        t = self.menuBar().addMenu("&Araçlar")
        self._act(t, "🧠 AI Merkezi…", self.open_ai_center, "Ctrl+M", tb)
        self._act(t, "✨ Yapay Zeka ile Akış Kur…", lambda: self.open_assistant(), "Ctrl+K", tb)
        self._act(t, "🎨 Özel Blok Geliştirme Stüdyosu…", self.open_custom_block_studio, "Ctrl+Shift+B", tb)
        t.addSeparator()
        self._act(t, "Etiketleme Aracı (YOLO)…", self.open_labeler, "Ctrl+L")
        self._act(t, "Örnekleri Öğret (emin olmadıkları)…", self.open_teach, "Ctrl+Shift+O")
        self._act(t, "YOLO Modeli Eğit…", self.open_trainer, "Ctrl+T")
        self._act(t, "🗃 Veri seti yöneticisi…", self.open_dataset_manager)
        self._act(t, "📷 Endüstriyel Kameraları Bul…", self.open_camera_finder)
        self._act(t, "📐 Kamera Kalibrasyonu (mm ölçüm)…", self.open_calibration)
        self._act(t, "📊 Üretim raporu…", self.open_report)
        self._act(t, "🗂 İzlenebilirlik arşivi…", self.open_trace)
        self._act(t, "📋 Reçeteler (ürün tipleri)…", self.open_recipes)
        self._act(t, "🖥 Ekran tasarımcısı…", self.open_screen_designer)
        self._act(t, "⚡ Hızlandırma (GPU)…", self.open_accel)
        self._act(t, "🌐 Uzaktan kontrol (API)…", self.open_api)
        self._act(t, "🏭 İstasyon yöneticisi…", self.open_stations)
        self._act(t, "Hızlı blok ekle", lambda: self._show_quick_add(None, None), "Ctrl+Space")
        t.addSeparator()
        self._act(t, "🔧 Kendini Geliştir…", self.open_evolve, "Ctrl+Shift+G")

        self.crumbs = self.addToolBar("Konum")
        self.crumbs.setObjectName("Konum")
        self.crumbs.setMovable(False)
        self.crumb_label = QLabel()
        self.crumbs.addWidget(self.crumb_label)
        self.crumbs.addAction("⬆ Üst akışa dön", self.exit_subflow)
        self.crumbs.addAction("⌂ Ana akış", self.go_root)
        self.crumbs.setVisible(False)

        h = self.menuBar().addMenu("&Yardım")
        self._act(h, "Kısayollar ve kullanım", self.show_help, "F1")
        self._act(h, "Hakkında", lambda: QMessageBox.about(
            self, "VisionFlow Studio",
            f"<b>VisionFlow Studio {__version__}</b><br>Kod yazmadan görüntü işleme, yapay zeka ve otomasyon."
            f"<br>{sum(len(v) for v in by_category().values())} blok"))
        self._update_undo_actions()

    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Kısayollar")
        dlg.resize(560, 520)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        rows = "".join(f"<tr><td style='padding:3px 14px 3px 0'><b>{a}</b></td><td>{b}</td></tr>" for a, b in SHORTCUTS)
        tb.setHtml(f"<h3>Kısayollar</h3><table>{rows}</table>"
                   "<h3>İpuçları</h3><ul>"
                   "<li>Çalışırken bir blok seçip ayarlarını değiştirin, sonuç anında görünür.</li>"
                   "<li>Önizlemede fareyi gezdirince piksel rengi (BGR / HSV) yazar.</li>"
                   "<li>Program beklenmedik kapanırsa açılışta son akışınızı kurtarmayı önerir.</li>"
                   "<li>Kendi bloğunuz için <code>plugins/</code> klasörüne .py dosyası koyun.</li></ul>")
        lay.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.exec()

    # autosave / crash recovery ----------------------------------------------------
    def autosave(self):
        """Write the current flow to the recovery file while there are unsaved changes."""
        try:
            if self.dirty and (self.root.nodes or self.root.frames):
                self.recovery.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.recovery.with_suffix(".tmp")
                tmp.write_text(json.dumps({"path": self.path, "graph": self._root_dict()},
                                          ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self.recovery)
                self._owns_recovery = True
            else:
                self._drop_recovery()
        except OSError as exc:
            print(f"Otomatik kayıt yapılamadı: {exc}")

    def _drop_recovery(self):
        """Delete the recovery file, but never one left by a crash that nobody has looked at yet, or one that
        another open VisionFlow window is keeping (closing a second window or the self-test must not lose it)."""
        if self._owns_recovery:
            self.recovery.unlink(missing_ok=True)
            self._owns_recovery = False

    def check_recovery(self, ask=True) -> bool:
        """Offer to restore a flow left behind by a crash. Returns True when one was restored."""
        if not self.recovery.exists():
            return False
        try:
            data = json.loads(self.recovery.read_text(encoding="utf-8"))
            graph = Graph.from_dict(data["graph"])
        except Exception:  # noqa: BLE001
            self.recovery.unlink(missing_ok=True)
            return False
        if ask and QMessageBox.question(
                self, "Kurtarma", f"Program son kez düzgün kapanmadı. Kaydedilmemiş akış ({len(graph.nodes)} blok) "
                "geri yüklensin mi?") != QMessageBox.Yes:
            self.recovery.unlink(missing_ok=True)
            return False
        from ..core.operator import locked
        from . import operator as _op
        if locked(graph) and not _op.ask_password(self, locked(graph), "Kurtarılan akış kilitli. Geri yüklemek "
                                                  "için şifreyi girin:"):
            return False                  # the file stays: the right person can restore it later
        self._set_root(graph)
        self.path = data.get("path")
        self._owns_recovery = True
        _set_flow_dir(self.path)
        self.history.reset(self._snap())
        self.saved_snap = ""          # restored work counts as unsaved
        self._update_undo_actions()
        self._title()
        QTimer.singleShot(60, self.view.fit_all)
        return True

    # history / dirty state -------------------------------------------------------------
    def _snap(self) -> str:
        return json.dumps(self._root_dict(), sort_keys=True, ensure_ascii=False)

    # sub-flows -----------------------------------------------------------------------
    @property
    def root(self) -> Graph:
        """The whole flow (the canvas may show the inside of a sub-flow)."""
        return self._root if self.levels else self.scene.graph

    def _prefix(self) -> str:
        """Key prefix of the blocks on the canvas in the running flow's results ("subflow id/" when inside one)."""
        return "".join(f"{node.id}/" for node, _ in self.levels)

    def _sync(self):
        """Write what is being edited inside open sub-flows back into their blocks, innermost first, and drop links
        of the outer flow to inputs/outputs that were removed inside."""
        from ..nodes.subflow import apply_inner
        for i in range(len(self.levels) - 1, -1, -1):
            node, inner = self.levels[i]
            data = inner.to_dict()
            if data != node.inner and apply_inner(self.levels[i - 1][1] if i else self._root, node, data) \
                    and self.worker:
                self._restart_timer.start()      # a renamed or new port: the running flow is wired differently

    def _root_dict(self) -> dict:
        self._sync()
        return self.root.to_dict()

    def _set_root(self, graph: Graph):
        self.levels, self._root = [], None
        self.scene.load_graph(graph)
        self._update_crumbs()

    def _update_crumbs(self):
        inside = bool(self.levels)
        self.crumbs.setVisible(inside)
        names = ["Ana akış"] + [node.title for node, _ in self.levels]
        self.crumb_label.setText("  📍 " + "  ›  ".join(names) + "   ")

    def enter_subflow(self, node):
        """Show the inside of a sub-flow block on the canvas (its blocks can be edited there)."""
        if node is None or node.type_name != "SubFlow":
            return
        self._sync()
        if not self.levels:
            self._root = self.scene.graph
        warnings: list[str] = []
        inner = node.inner_graph(warnings)
        for w in warnings:
            self.logp.log(f"Uyarı: {w}")
        self.levels.append((node, inner))
        self._show(inner)
        self._update_crumbs()
        if not inner.nodes:
            self.statusBar().showMessage("Alt akış boş: blok ekleyin; girişler ve çıkışlar için 'Alt Akış Girişi / "
                                         "Çıkışı' bloklarını kullanın", 8000)
        QTimer.singleShot(30, self.view.fit_all)

    def exit_subflow(self):
        if not self.levels:
            return
        self._sync()
        node, _ = self.levels.pop()
        parent = self.levels[-1][1] if self.levels else self._root
        if not self.levels:
            self._root = None
        self._show(parent)
        self._update_crumbs()
        item = self.scene.items_by_id.get(node.id)
        if item is not None:
            item.setSelected(True)
            QTimer.singleShot(30, lambda: self.view.centerOn(item))

    def _show(self, graph: Graph):
        self._navigating = True
        try:
            self.scene.load_graph(graph)
        finally:
            self._navigating = False

    def go_root(self):
        while self.levels:
            self.exit_subflow()

    def group_selected(self, name: str | None = None):
        """Pack the selected blocks into a sub-flow block, asking for its name."""
        if not any(isinstance(it, NodeItem) for it in self.scene.selectedItems()):
            self.statusBar().showMessage("Alt akış yapmak için önce blokları seçin", 4000)
            return None
        if name is None:
            name, ok = QInputDialog.getText(self, "Alt akış yap", "Alt akışın adı:", text="Alt Akış")
            if not ok:
                return None
        item = self.scene.group_selected(name)
        if item is not None:
            self.statusBar().showMessage("Alt akış oluşturuldu: içini görmek için çift tıklayın (geri almak için "
                                         "Ctrl+Z)", 6000)
        return item

    def ungroup(self, node):
        items = self.scene.ungroup(node)
        if items:
            self.statusBar().showMessage(f"Alt akış çözüldü: {len(items)} blok", 4000)
        return items

    def save_subflow_to_library(self, node):
        from ..nodes import subflow
        try:
            path = subflow.save_to_library(node)
        except OSError as exc:
            QMessageBox.warning(self, "Kaydedilemedi", str(exc))
            return None
        self.statusBar().showMessage(f"Kütüphaneye kaydedildi: {path}", 5000)
        return path

    def add_from_library(self, path, pos: QPointF | None = None):
        from ..nodes import subflow
        try:
            sub, warnings = subflow.load_from_library(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Açılamadı", str(exc))
            return None
        for w in warnings:
            self.logp.log(f"Uyarı: {w}")
        if pos is None:
            pos = self.view.mapToScene(self.view.viewport().rect().center()) - QPointF(95, 40)
        return self.scene.paste({"format": "visionflow-clip", "nodes": [sub.to_dict()], "connections": []}, pos)

    def _schedule_snapshot(self):
        self._snap_timer.start()

    def _snapshot(self):
        self.history.push(self._snap())
        self._update_undo_actions()
        self._title()

    def _update_undo_actions(self):
        self.undo_act.setEnabled(self.history.index > 0)
        self.redo_act.setEnabled(self.history.index < len(self.history.stack) - 1)

    def _restore(self, snap: str | None):
        if snap is None:
            return
        path = [node.id for node, _ in self.levels]
        root = Graph.from_dict(json.loads(snap))
        self._set_root(root)
        if path:                                 # stay inside the same sub-flow when it is still there
            levels, cur = [], root
            for nid in path:
                node = cur.nodes.get(nid)
                if node is None or node.type_name != "SubFlow":
                    break
                cur = node.inner_graph()
                levels.append((node, cur))
            if levels:
                self._root, self.levels = root, levels
                self.scene.load_graph(cur)
                self._update_crumbs()
        self._update_undo_actions()
        self._title()

    def undo(self):
        if self._snap_timer.isActive():   # flush a pending edit first so it can be undone
            self._snap_timer.stop()
            self._snapshot()
        self._restore(self.history.undo())

    def redo(self):
        self._restore(self.history.redo())

    @property
    def dirty(self) -> bool:
        return self._snap() != self.saved_snap

    def _confirm_discard(self) -> bool:
        if not self.dirty or not (self.root.nodes or self.root.frames):
            return True
        ans = QMessageBox.question(self, "Kaydedilmemiş değişiklikler",
                                   "Akışta kaydedilmemiş değişiklikler var. Kaydedilsin mi?",
                                   QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if ans == QMessageBox.Save:
            return self.save_flow()
        return ans == QMessageBox.Discard

    # file handling ----------------------------------------------------------------
    def _title(self):
        name = os.path.basename(self.path) if self.path else "Yeni akış"
        star = "*" if self.dirty and (self.root.nodes or self.root.frames) else ""
        self.setWindowTitle(f"{name}{star} — VisionFlow Studio")

    def _loaded(self, path: str | None):
        self.path = path
        self.active_recipe = ""                  # the recipe this flow runs with (remote control API /durum)
        self.saved_snap = self._snap()
        self.history.reset(self.saved_snap)
        self._update_undo_actions()
        self._title()

    def new_flow(self):
        if not self._confirm_discard():
            return
        self.stop()
        self._set_root(Graph())
        self._loaded(None)

    def new_from_gallery(self):
        from .template_gallery import new_from_gallery
        return new_from_gallery(self)

    def open_flow(self, path=None, unlocked=False):
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Akış aç", "", "VisionFlow akışı (*.json)")
            if not path:
                return
        self.stop()
        from ..core import node as _node
        before = (_node.FLOW_DIR, _node.FLOW_FILE)     # Graph.load points them at the new file
        try:
            graph = Graph.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Açılamadı", str(exc))
            return
        from ..core.operator import locked
        from . import operator as _op
        if not unlocked and locked(graph) and not _op.ask_password(self, locked(graph), "Bu akış kilitli. "
                                                                    "Düzenlemek için şifreyi girin:"):
            # the open flow stays: its relative files and its name in the archive too
            _node.FLOW_DIR, _node.FLOW_FILE = before
            return
        self._set_root(graph)
        if graph.warnings:
            for w in graph.warnings:
                self.logp.log(f"Uyarı: {w}")
            QMessageBox.warning(self, "Akış kısmen açıldı", "\n".join(graph.warnings[:10]))
        is_example = pathlib.Path(path).resolve().parent == EXAMPLES.resolve()
        self._loaded(None if is_example else path)
        if not is_example:
            self._add_recent(path)
        QTimer.singleShot(60, self.view.fit_all)

    def save_flow(self, save_as=False) -> bool:
        if save_as or not self.path:
            path, _ = QFileDialog.getSaveFileName(self, "Akışı kaydet", "akis.json", "VisionFlow akışı (*.json)")
            if not path:
                return False
        else:
            path = self.path
        try:
            self._sync()
            self.root.save(path)
        except OSError as exc:
            QMessageBox.warning(self, "Kaydedilemedi", f"{path}\n\n{exc}")
            return False
        self.path = path
        _set_flow_dir(path)
        self.saved_snap = self._snap()
        self._drop_recovery()
        self._add_recent(self.path)
        self._title()
        self.statusBar().showMessage(f"Kaydedildi: {self.path}", 3000)
        return True

    def _add_recent(self, path):
        recent = [p for p in self.settings.value("recent", [], list) if p != path]
        self.settings.setValue("recent", [path, *recent][:8])
        self._refresh_recent()

    def _refresh_recent(self):
        self.recent_menu.clear()
        recent = [p for p in self.settings.value("recent", [], list) if os.path.exists(p)]
        for p in recent:
            self.recent_menu.addAction(p, lambda p=p: self.open_flow(p))
        self.recent_menu.setEnabled(bool(recent))

    # editing ------------------------------------------------------------------------
    def add_at_center(self, type_name):
        c = self.view.mapToScene(self.view.viewport().rect().center())
        self.scene.add_node(type_name, c - QPointF(95, 40))

    def select_all(self):
        for it in self.scene.items_by_id.values():
            it.setSelected(True)

    def copy(self):
        clip = self.scene.copy_selected()
        if not clip:
            return
        md = QMimeData()
        data = json.dumps(clip, ensure_ascii=False)
        md.setData(CLIP_MIME, data.encode())
        md.setText(data)
        QGuiApplication.clipboard().setMimeData(md)
        self.statusBar().showMessage(f"{len(clip['nodes'])} blok kopyalandı", 2000)

    def paste(self, at: QPointF | None = None):
        md = QGuiApplication.clipboard().mimeData()
        raw = bytes(md.data(CLIP_MIME)).decode() if md.hasFormat(CLIP_MIME) else md.text()
        try:
            clip = json.loads(raw)
        except (ValueError, TypeError):
            return
        if isinstance(clip, dict) and clip.get("format") in ("visionflow-clip", "visionflow"):
            if at is None:
                at = self.view.mapToScene(self.view.mapFromGlobal(self.cursor().pos()))
                if not self.view.viewport().rect().contains(self.view.mapFromScene(at)):
                    at = None
            self.scene.paste(clip, at)

    def _show_quick_add(self, global_pos, scene_pos):
        if global_pos is None:
            center = self.view.viewport().rect().center()
            global_pos, scene_pos = self.view.mapToGlobal(center), self.view.mapToScene(center)
        self._quick_pos = scene_pos
        self._pending_wire = None
        self.quick_add.popup(global_pos)

    def open_teach(self):
        from .teach_dialog import TeachDialog
        # the data folder of the flow's collector or classifier, so the window opens on the right samples
        folder = next((n.values.get("folder") for t in ("UncertainCollector", "QuickClassifier", "SampleCollector")
                       for n in self.root.nodes.values() if n.type_name == t and n.values.get("folder")), "veri")
        dlg = getattr(self, "_teach", None)
        try:
            if dlg is not None and dlg.isVisible():
                dlg.raise_()
                dlg.activateWindow()
                return dlg                      # one window: two would fight over the same samples
        except RuntimeError:
            pass
        if dlg is not None:
            dlg.deleteLater()
        dlg = self._teach = TeachDialog(self, folder)
        dlg.show()
        return dlg

    def open_block_dialog(self, node_id: str):
        """A block's own window (Node.dialog = (button text, "module.Class" in visionflow.ui)), one per block.
        The window is made as Class(parent, node, apply) where apply(setting, value) sets a setting like typing it."""
        import importlib
        node = self.scene.graph.nodes.get(node_id)
        spec = getattr(node, "dialog", None)
        if not spec:
            return None
        dialogs = self.__dict__.setdefault("_block_dialogs", {})
        dlg = dialogs.get(node_id)
        try:
            old = getattr(dlg, "node", None) if dlg is not None else None
            # the same block (undo, a recipe or leaving / entering a sub-flow make new block objects): the window
            # with the pictures drawn so far, now for the block on the canvas
            if old is not None and old.id == node.id and old.type_name == node.type_name:
                dlg.node = node
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()
                return dlg
        except RuntimeError:                    # deleted by Qt
            pass
        mod_name, _, cls_name = spec[1].rpartition(".")
        cls = getattr(importlib.import_module(f".{mod_name}", __package__), cls_name)

        def apply(name, value, nid=node.id, kind=node.type_name) -> bool:
            """False when the block is not on the canvas now (e.g. its sub-flow was left): nothing was set."""
            n = self.scene.graph.nodes.get(nid)
            if n is None or n.type_name != kind:
                return False
            if n.values.get(name) == value:
                return True
            n.values[name] = n.param(name).coerce(value, clamp=True)
            self.on_param(n.id, name, n.values[name])
            if self.props.node is n:
                self.props.show_node(n)
            return True
        dlg = dialogs[node_id] = cls(self, node, apply)
        dlg.show()
        return dlg

    def open_calibration(self):
        from .calibration_dialog import CalibrationDialog
        dlg = getattr(self, "_calibration", None)
        try:
            if dlg is not None and dlg.isVisible():
                dlg.raise_()
                return dlg
        except RuntimeError:
            pass
        dlg = self._calibration = CalibrationDialog(self)
        dlg.show()
        return dlg

    def add_frame(self):
        center = self.view.mapToScene(self.view.viewport().rect().center())
        return self.scene.add_frame(at=center - QPointF(210, 130))

    def open_camera_finder(self):
        from .camera_finder import CameraFinder

        def add(backend, serial):
            center = self.view.mapToScene(self.view.viewport().rect().center())
            self.scene.add_node("IndustrialCamera", center, {"backend": backend, "serial": serial})
        dlg = getattr(self, "_camera_finder", None)
        try:
            if dlg is not None and dlg.isVisible():
                dlg.raise_()
                return dlg
        except RuntimeError:
            pass
        dlg = self._camera_finder = CameraFinder(self, add, running=lambda: self.worker is not None)
        dlg.show()
        return dlg

    def open_custom_block_studio(self):
        from .custom_block_dialog import CustomBlockStudioDialog
        dlg = getattr(self, "_custom_block_dialog", None)
        if dlg is not None:
            try:
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()
                return dlg
            except RuntimeError:
                pass
        dlg = self._custom_block_dialog = CustomBlockStudioDialog(self, on_saved=self.palette.rebuild)
        dlg.show()
        return dlg

    def open_evolve(self):
        from .evolve_dialog import EvolveDialog
        dlg = getattr(self, "_evolve", None)
        if dlg is not None:
            try:
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()
                return dlg                     # one window: a running proposal keeps showing its progress
            except RuntimeError:
                pass
        dlg = self._evolve = EvolveDialog(self)
        dlg.show()
        return dlg

    def _daily_review(self):
        """'Her gün kendini incele': examine itself in the background; a tested proposal only waits for the user."""
        from .. import evolve
        from ..ai.config import AIConfig
        if not evolve.daily_due() or not AIConfig.load().key():
            return
        note = self._daily_note = _Note(self)
        note.text.connect(self.logp.log)
        note.text.connect(lambda m: self.statusBar().showMessage(m, 15000))

        def work():       # a plain daemon thread: closing the window must not wait for the tests
            try:
                prop = evolve.daily_review()
                msg = (f"Kendini Geliştir: günlük inceleme bir düzeltme hazırladı ({prop.summary[:120]}). "
                       "Araçlar → Kendini Geliştir'den inceleyip uygulayabilirsiniz.") if prop else ""
            except Exception as exc:  # noqa: BLE001
                msg = f"Kendini Geliştir: günlük inceleme yapılamadı: {exc}"
            if msg:
                try:
                    note.text.emit(msg)
                except RuntimeError:          # window already closed
                    pass
        threading.Thread(target=work, daemon=True, name="kendini-incele").start()

    def open_ai_center(self):
        from .ai_center import AICenter
        dlg = getattr(self, "_ai_center", None)
        if dlg is not None:
            try:
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()
                return dlg                         # one window: pressing Ctrl+M again brings it to the front
            except RuntimeError:                   # it was closed and deleted
                pass
        dlg = AICenter(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.destroyed.connect(lambda *_: setattr(self, "_ai_center", None))
        self._ai_center = dlg
        dlg.show()
        return dlg

    def open_assistant(self, client=None):
        from .assistant_dialog import AssistantDialog
        self.go_root()                    # the assistant reads and replaces the whole flow
        dlg = AssistantDialog(self, self.scene.graph, client=client)
        dlg.apply_graph.connect(self.apply_generated)
        dlg.show()
        return dlg

    def apply_generated(self, graph: Graph, modified: bool):
        """Put an assistant-made flow on the canvas as one undoable step."""
        self.stop()
        if self._snap_timer.isActive():
            self._snap_timer.stop()
            self._snapshot()
        self.go_root()
        notes = [n for n in self.scene.graph.nodes.values() if n.type_name == "Note"] if modified else []
        for n in notes:          # the assistant never sees notes or comment frames; keep the user's
            graph.add(n)
        if modified:
            graph.frames = [dict(f) for f in self.scene.graph.frames]
            from ..core import recipes
            graph.recipes = recipes.clean(recipes.export(self.scene.graph), graph, [])
            if self.scene.graph.operator:        # operator panel settings and the edit lock stay
                from ..core.operator import clean_operator
                graph.operator = clean_operator(self.scene.graph.operator, graph, [])
        self._set_root(graph)
        if not modified:
            self.path = None
            _set_flow_dir(None)
            self.scene.auto_layout()     # real block heights (thumbnails) instead of the estimate
        self._snapshot()
        QTimer.singleShot(60, self.view.fit_all)
        self.statusBar().showMessage(f"Yapay zeka akışı uygulandı ({len(graph.nodes)} blok). Geri almak için Ctrl+Z", 6000)

    def open_labeler(self):
        from .tools import LabelingDialog
        dlg = LabelingDialog(self)
        dlg.show()

    def open_dataset_manager(self, folder: str | None = None):
        """Araçlar → Veri seti yöneticisi: statistics, split, export, augmentation, auto-labelling."""
        from .dataset_manager import DatasetManager
        dlg = DatasetManager(self, folder)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.show()
        return dlg

    def open_trainer(self):
        from .tools import TrainDialog
        dlg = TrainDialog(self)
        dlg.finished_model.connect(lambda p: self.logp.log(f"Yeni model hazır: {p}"))
        dlg.show()

    def context_menu(self, global_pos, scene_pos):
        menu = QMenu(self)
        frame = self.scene.frame_at(scene_pos)
        if frame is not None:
            menu.addAction("Çerçeve başlığını düzenle…", lambda: self.scene.rename_frame(frame))
            menu.addAction("Çerçeve rengini değiştir", lambda: self.scene.recolor_frame(frame))
            menu.addAction("Çerçeveyi sil (bloklar kalır)", lambda: self.scene.remove_frame(frame))
            menu.addSeparator()
        sel = self.scene.selected_node()
        if sel is not None and sel.type_name == "SubFlow":
            menu.addAction("📂 Alt akışı aç (içini düzenle)", lambda: self.enter_subflow(sel))
            menu.addAction("Alt akışı çöz (blokları geri koy)", lambda: self.ungroup(sel))
            menu.addAction("Alt akış kütüphanesine kaydet", lambda: self.save_subflow_to_library(sel))
            menu.addSeparator()
        if self.levels:
            if sel is not None and sel.params and self._expose_state(sel, sel.params[0].name) is not None:
                exp = menu.addMenu("Ayarı alt akış bloğunda göster")
                for p in sel.params:
                    a = exp.addAction(p.display)
                    a.setCheckable(True)
                    a.setChecked(bool(self._expose_state(sel, p.name)))
                    a.toggled.connect(lambda on, nid=sel.id, name=p.name: self.toggle_exposed(nid, name, on))
            menu.addAction("⬆ Üst akışa dön", self.exit_subflow)
            menu.addSeparator()
        if any(isinstance(it, NodeItem) for it in self.scene.selectedItems()):
            menu.addAction("Seçilenleri alt akış yap (tek blokta topla)…", lambda: self.group_selected())
            menu.addAction("Seçilenleri yorum çerçevesine al", lambda: self.scene.add_frame())
        else:
            menu.addAction("Yorum çerçevesi ekle", lambda: self.scene.add_frame(at=scene_pos))
        if self.scene.selected_node() or any(self.scene.selectedItems()):
            menu.addAction("Kopyala", self.copy)
            menu.addAction("Çoğalt", self.scene.duplicate_selected)
            menu.addAction("Sil", self.scene.delete_selected)
            menu.addAction("Devre dışı bırak / aç", self.scene.toggle_enabled_selected)
            menu.addSeparator()
        editor_tools.extend_context_menu(self, menu)
        menu.addAction("Yapıştır", lambda: self.paste(scene_pos))
        add = menu.addMenu("Blok ekle")
        for cat, classes in by_category().items():
            sub = add.addMenu(cat)
            for cls in classes:
                sub.addAction(cls.title, lambda t=cls.type_name: self.scene.add_node(t, scene_pos))
        from ..nodes.subflow import library
        saved = library()
        lib = menu.addMenu("Alt akış kütüphanesi")
        for name, path in saved:
            lib.addAction(name, lambda p=path: self.add_from_library(p, scene_pos))
        lib.setEnabled(bool(saved))
        self._context_menu = menu            # tests read it
        menu.exec(global_pos)

    def watched_id(self, node) -> str | None:
        """The block whose image the preview shows: the block itself, or for a block with a drawable region the
        block that feeds its image input (the region is drawn on the picture going in, not the result)."""
        if node is None:
            return None
        if node.roi:
            src = next((c.src for c in self.scene.graph.connections
                        if c.dst == node.id and c.dst_port == "image"), None)
            if src:
                return src
        return node.id

    def on_selection(self):
        node = self.scene.selected_node()
        self.props.show_node(node)
        self._watching = self._full(self.watched_id(node))
        if self.worker:
            self.worker.watch_id = self._watching
        self.preview.set_image(None)
        self.preview.set_roi(node.roi if node is not None and node.roi and self._watching != self._full(node.id) else None,
                             node.values if node is not None else None)

    def on_roi_drawn(self, values: dict):
        """A region drawn on the preview: set it on the selected block like typing it in, undo included."""
        node = self.scene.selected_node()
        if node is None or not node.roi:
            return
        for name, value in values.items():
            node.values[name] = node.param(name).coerce(value, clamp=True)
            self.on_param(node.id, name, node.values[name])
        self.props.show_node(node)
        self.statusBar().showMessage("Bölge ayarlandı", 3000)

    def _full(self, local_id):
        return f"{self._prefix()}{local_id}" if local_id else local_id

    def on_param(self, node_id, name, value):
        self._schedule_snapshot()
        item = self.scene.items_by_id.get(node_id)
        if item is not None and item.is_note:
            item.refresh()
        if item is not None and item.is_subflow and name == "name":
            item.node.refresh()                  # the title on the canvas
            item.refresh()
        if item is not None and item.is_subflow and "/" in name:
            item.node.set_exposed(name, value)   # an exposed setting: it is used by the block inside
        self._send_live(node_id, name, value)

    def _send_live(self, node_id, name, value):
        """A changed setting to the running flow. A sub-flow's exposed setting "inner id/setting" goes straight to
        the running block inside ("sub-flow id/inner id")."""
        if not self.worker:
            return
        node = self.scene.graph.nodes.get(node_id)
        if node is not None and node.type_name == "SubFlow" and "/" in name:
            path, _, name = name.rpartition("/")
            node_id = f"{node_id}/{path}"
        self.worker.set_param(self._full(node_id), name, value)

    # exposed settings of sub-flows --------------------------------------------------
    def _expose_state(self, node, name):
        """Inside a sub-flow: is setting `name` of block `node` shown on the sub-flow block (None: not possible)."""
        if not self.levels or node is None or node.type_name in ("SubflowInput", "SubflowOutput", "Note") \
                or self.scene.graph.nodes.get(node.id) is not node:
            return None
        return self.levels[-1][0].is_exposed(node.id, name)

    def toggle_exposed(self, node_id, name, on=True) -> bool:
        """Show (or hide) a setting of a block inside the open sub-flow on the sub-flow block."""
        if not self.levels:
            return False
        self._sync()                              # a block just added inside must be in the sub-flow first
        sub = self.levels[-1][0]
        ok = sub.expose(node_id, name, on)
        if ok:
            node = self.scene.graph.nodes.get(node_id)
            label = node.param(name).display if node is not None else name
            self.statusBar().showMessage(f"'{label}' alt akış bloğunda {'gösteriliyor' if on else 'gizlendi'}", 4000)
            self._schedule_snapshot()
            if self.props.node is not None:
                self.props.show_node(self.props.node)
        return ok

    # recipes --------------------------------------------------------------------------
    def open_recipes(self):
        from .recipes_dialog import RecipesDialog
        dlg = getattr(self, "_recipes", None)
        try:
            if dlg is not None and dlg.isVisible():
                dlg.raise_()
                dlg.activateWindow()
                return dlg
        except RuntimeError:                     # already deleted by Qt
            dlg = None
        self.go_root()                           # recipes hold blocks of the main flow
        if dlg is None:                          # one dialog, shown again (not a new one each time)
            dlg = self._recipes = RecipesDialog(self)
        dlg.show()
        dlg.raise_()
        return dlg

    def open_accel(self):
        """Araçlar → Hızlandırma (GPU)…: devices, the flow's / computer's default and a speed test."""
        from .accel_dialog import AccelDialog
        dlg = AccelDialog(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.show()
        return dlg

    def open_stations(self):
        """Araçlar → İstasyon yöneticisi…: several flows at once, each in its own process (ui/station_manager.py)."""
        from .station_manager import open_station_manager
        return open_station_manager(self)

    def open_api(self):
        """Araçlar → Uzaktan kontrol (API)…: REST API for MES/ERP (ui/api_dialog.py)."""
        from .api_dialog import ApiDialog
        dlg = ApiDialog(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.show()
        return dlg

    def apply_recipe(self, name) -> list[str]:
        """Put a recipe's values into the blocks of the flow (and of the running flow). Returns the problems."""
        from ..core import recipes
        self._sync()
        r = recipes.get(self.root, name)
        if r is None:
            return [f"'{name}' adlı reçete yok"]
        errors = recipes.apply(self.root, r)
        self.active_recipe = r["name"]
        panel = self.operator_panel()
        worker = self.worker or (panel.worker if panel is not None else None)
        if worker is not None:
            # the running flow switches between frames like a 'Reçete Seç' block does (blocks restarted if needed,
            # the API and 'Parçayı Arşivle' learn the new recipe)
            worker.request_recipe(r["name"])
        if panel is not None:      # the panel's flow copy (its designed screen, its next start) gets it too
            panel.apply_recipe(r["name"], from_editor=True)
        self._values_changed()
        self._schedule_snapshot()
        for e in errors:
            self.logp.log(f"Reçete '{r['name']}': {e}")
        return errors

    def _values_changed(self):
        """Settings of the whole flow were changed from outside the settings panel: show them."""
        if self.levels:                          # the open sub-flow's blocks are copies: reload them
            self._navigating = True
            try:
                self._restore(json.dumps(self.root.to_dict(), ensure_ascii=False))
            finally:
                self._navigating = False
        else:
            for it in self.scene.items_by_id.values():
                if it.is_subflow:
                    it.node.refresh()            # its title may have changed
                if it.is_note or it.is_subflow:
                    it.refresh()
        node = self.scene.selected_node()
        self.props.show_node(node)
        self._title()

    def on_worker_recipe(self, name, errors):
        """The running flow switched to a recipe ('Reçete Seç'): show the same values in the editor."""
        from ..core import recipes
        self.logp.log(f"Reçete uygulandı: {name}")
        for e in errors:
            self.logp.log(f"Reçete '{name}': {e}")
        self.statusBar().showMessage(f"Reçete: {name}", 4000)
        self.active_recipe = name
        self._sync()
        r = recipes.get(self.root, name)
        if r is not None:
            recipes.apply(self.root, r)
            self._values_changed()

    def on_structure(self):
        # rewiring while running: restart shortly so the change applies immediately
        if self.worker and not self._navigating:
            self._restart_timer.start()
        sel = self.scene.selected_node()
        if sel is not None and sel.roi and self._full(self.watched_id(sel)) != getattr(self, "_watching", None):
            self.on_selection()                  # the region block got (or lost) its input picture
        cur = self.props.node
        if cur and self.scene.graph.nodes.get(cur.id) is not cur:
            self.props.show_node(self.scene.selected_node())

    def toggle_thumbs(self, on):
        self.scene.set_thumbs_visible(on)
        if self.worker:
            self.worker.thumbs = on

    def on_fps_limit(self, value):
        self.settings.setValue("max_fps", value)
        if self.worker:
            self.worker.limiter.max_fps = value

    # checking ------------------------------------------------------------------------
    def flow_issues(self):
        from ..check import check
        self._sync()
        return check(self.root)

    def check_flow(self):
        """List the flow's problems; double-click one to jump to its block."""
        issues = self.flow_issues()
        dlg = QDialog(self)
        dlg.setWindowTitle("Akış denetimi")
        dlg.resize(620, 360)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Sorun bulunamadı. Akış çalışmaya hazır." if not issues else
                             f"{len(issues)} konu bulundu. Bloğa gitmek için çift tıklayın:"))
        lst = QListWidget()
        icons = {"hata": "⛔", "uyarı": "⚠️", "bilgi": "ℹ️"}
        for i in issues:
            item = QListWidgetItem(f"{icons.get(i.level, '')}  {i.text}")
            item.setData(Qt.UserRole, i.node_id)
            lst.addItem(item)
        lst.itemDoubleClicked.connect(lambda it: self.focus_node(it.data(Qt.UserRole)))
        lay.addWidget(lst)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        self._check_dialog = dlg      # tests read it
        dlg.show()
        return issues

    def focus_node(self, node_id):
        if node_id and node_id not in self.scene.items_by_id and self.levels:
            self.go_root()                       # flow check issues name blocks of the whole flow
        it = self.scene.items_by_id.get(node_id) if node_id else None
        if it is None:
            return
        self.scene.clearSelection()
        it.setSelected(True)
        self.view.centerOn(it)

    # operator panel / production report ----------------------------------------------
    def open_operator(self):
        """Hide the editor and run the flow in the full-screen operator panel."""
        from .operator import OperatorPanel
        if not self.root.nodes:
            self.statusBar().showMessage("Önce birkaç blok ekleyin", 3000)
            return None
        self.stop()
        try:
            panel = OperatorPanel(self._root_dict(), self.path, editor=self)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Operatör paneli", f"Açılamadı: {exc}")
            return None
        self._operator = panel
        panel.showMaximized()
        self.hide()
        panel.start()
        return panel

    def operator_panel(self):
        """The operator panel opened from this editor while it is open (the editor is hidden then), else None."""
        panel = getattr(self, "_operator", None)
        try:
            return panel if panel is not None and not panel._left else None
        except RuntimeError:                       # deleted by Qt
            return None

    def edit_operator(self):
        from .operator import OperatorConfigDialog
        self._sync()
        dlg = OperatorConfigDialog(self, self.root)
        self._operator_dialog = dlg          # tests read it
        if dlg.exec() == QDialog.Accepted:
            self.root.operator = dlg.config
            self._snapshot()
            self.statusBar().showMessage("Operatör paneli ayarları akışa kaydedildi (dosyayı kaydetmeyi unutmayın)",
                                         5000)

    def open_screen_designer(self):
        """Araçlar → Ekran tasarımcısı: the operator screen designed by drag and drop (ui/screen_designer.py)."""
        from .screen_designer import ScreenDesigner
        self._sync()
        old = getattr(self, "_screen_designer", None)
        try:
            if old is not None and old.isVisible():
                old.raise_()
                old.activateWindow()
                return old
        except RuntimeError:                       # deleted by Qt
            pass
        dlg = ScreenDesigner(self, self.root, editor=self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        self._screen_designer = dlg
        dlg.show()
        return dlg

    def open_report(self):
        from .operator import ReportDialog
        self._report_dialog = ReportDialog(self, self.path)
        self._report_dialog.show()

    def open_trace(self):
        from ..nodes.traceability import flow_archive
        from .trace_browser import open_browser
        self._sync()
        self._trace_browser = open_browser(self, flow_archive(self.root))     # the flow's own archive folder

    # running -------------------------------------------------------------------------
    def start(self):
        if self.worker:
            return
        if not self.root.nodes:
            self.statusBar().showMessage("Önce birkaç blok ekleyin", 3000)
            return
        try:
            self.root.order()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Akış hatalı", str(exc))
            return
        for it in self.scene.items_by_id.values():
            it.clear_result()
        try:
            errors = [i for i in self.flow_issues() if i.level == "hata"]
        except Exception:  # noqa: BLE001  (the checker must never block running)
            errors = []
        for i in errors:
            self.logp.log(f"[Denetim] {i.text}")
        if errors:
            self.statusBar().showMessage(f"Akış denetimi {len(errors)} sorun buldu: Çalıştır → Akışı denetle "
                                         f"(Ctrl+Shift+D)", 8000)
        self.worker = FlowWorker(self._root_dict())
        node = self.scene.selected_node()
        self._watching = self._full(self.watched_id(node))
        self.worker.watch_id = self._watching
        self.worker.thumbs = self.thumb_act.isChecked()
        self.worker.limiter.max_fps = self.fps_spin.value()
        self.worker.frame.connect(self.on_frame)
        self.worker.watch.connect(self.on_watch)
        self.worker.failed.connect(self.on_failed)
        self.worker.recipe.connect(self.on_worker_recipe)
        self.worker.finished.connect(self._worker_done)
        from .api_dialog import worker_state
        self.worker.api = worker_state(self)        # remote control API (if on) gets every frame
        self.worker.recipe_name = getattr(self, "active_recipe", "")
        self.worker.start()
        self.run_act.setEnabled(False)
        self.stop_act.setEnabled(True)
        self.hud.set_running()

    def stop(self):
        w = self.worker
        if not w:
            return
        w.stop()
        if not w.wait(5000):
            self.logp.log("Uyarı: akış 5 sn içinde durmadı (kamera yanıt vermiyor olabilir)")
        self._worker_done()

    def _restart(self):
        if self.worker:
            self.stop()
            self.start()

    def _worker_done(self):
        if self.worker and self.worker.isRunning():
            return
        self.worker = None
        self.run_act.setEnabled(True)
        self.stop_act.setEnabled(False)
        self.hud.set_stopped(len(self.scene.items_by_id))

    def on_watch(self, node_id, img, size=None):
        if node_id == getattr(self, "_watching", None) and self.scene.selected_node() is not None:
            self.preview.set_image(img, size)

    def on_failed(self, msg):
        self.logp.log(f"HATA: {msg}")
        self.statusBar().showMessage(msg, 5000)

    def on_frame(self, payload: dict, fps: float):
        errors = []
        prefix = self._prefix()
        for key, (ms, err, text, qimg) in payload.items():
            nid = key[len(prefix):] if key.startswith(prefix) else None
            item = self.scene.items_by_id.get(nid) if nid and "/" not in nid else None
            if item:
                item.set_result(ms, err, text, qimg)
                if err and err != "Önceki blok hata verdi":
                    errors.append(f"{item.node.title}: {err}")
        total = sum(v[0] for k, v in payload.items() if "/" not in k)
        self.hud.update_stats(fps, total, errors, len(self.scene.items_by_id))

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        self.stop()
        if self.worker is not None and self.worker.isRunning():
            # a camera/PLC that does not answer: wait a little longer, but never quit with the flow thread still
            # running (Qt aborts, and video files / connections are not closed properly)
            self.statusBar().showMessage("Akış durduruluyor…")
            if not self.worker.wait(10000):
                QMessageBox.warning(self, "Akış durmuyor", "Akış hâlâ durmadı (bir cihaz yanıt vermiyor olabilir). "
                                    "Birkaç saniye sonra tekrar kapatmayı deneyin.")
                e.ignore()
                return
            self._worker_done()
        self._autosave_timer.stop()
        from .api_dialog import shutdown
        shutdown(self)                           # the remote control API's port is freed
        self._drop_recovery()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())
        self.scene.selectionChanged.disconnect(self.on_selection)
        sys.stdout = sys.__stdout__
        super().closeEvent(e)
