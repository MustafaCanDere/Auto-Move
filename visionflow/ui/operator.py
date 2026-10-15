"""Operator panel (HMI): a big live picture, a SAĞLAM / HATALI lamp, large counters and the last results for the
production line, plus its settings dialog, the edit-lock password prompt and the production report dialog."""
from __future__ import annotations

import datetime as dt
import pathlib
import time

from PySide6.QtCore import QDateTime, Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateTimeEdit, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from ..core import Graph, PortType
from ..core.graph import find_node, inner_error, walk_nodes
from ..core.operator import COUNTERS, HISTORY_MAX, check_password, make_lock
from ..production import (SAMPLE_S, ProductionDB, Tally, auto_config, db_path, default_image, make_report,
                          store_sample)
from .worker import FlowWorker, to_qimage

SAMPLE_MS = int(SAMPLE_S * 1000)    # counts are written to the production database every minute
REFRESH_MS = 100
GOOD, BAD, IDLE, ERR = "#2e7d32", "#c62828", "#455a64", "#ef6c00"


def ask_password(parent, lock, why: str = "Devam etmek için düzenleme şifresini girin:") -> bool:
    """True when there is no lock or the right password was typed. Tests replace this function."""
    if not lock:
        return True
    pw, ok = QInputDialog.getText(parent, "Düzenleme kilidi", why, QLineEdit.Password)
    if not ok:
        return False
    if check_password(lock, pw):
        return True
    QMessageBox.warning(parent, "Düzenleme kilidi", "Şifre yanlış.")
    return False


def _big(text: str, px: int, bold=True) -> QLabel:
    lab = QLabel(text)
    lab.setAlignment(Qt.AlignCenter)
    lab.setStyleSheet(f"font-size: {px}px; {'font-weight: bold;' if bold else ''}")
    return lab


class OperatorPanel(QWidget):
    """Runs the flow in its own FlowWorker and shows only what the line operator needs. ``editor`` is the editor
    window that opened it (shown again when the panel is left); ``editor_factory`` makes one (``--operator``)."""

    def __init__(self, graph_data: dict, flow_path: str | None = None, editor=None, editor_factory=None):
        super().__init__()
        self.graph_data = graph_data
        self.graph = Graph.from_dict(graph_data)
        self.flow_path = flow_path
        self.flow_name = pathlib.Path(flow_path).stem if flow_path else "Adsız akış"
        self.editor, self.editor_factory = editor, editor_factory
        self.cfg = dict(self.graph.operator or auto_config(self.graph))
        self.lock = self.cfg.get("lock")
        self.image_id = self.cfg.get("image") or default_image(self.graph)
        self.tally = Tally(self.cfg)
        self.db = ProductionDB(db_path(flow_path))
        self.worker: FlowWorker | None = None
        self.recipe_name = ""               # the recipe the flow runs with (remote control API)
        self._left = False
        self._changed = False               # settings / recipe changed here (handed to an editor opened later)
        self._hist_key = None
        self.setWindowTitle(f"Operatör Paneli — {self.cfg.get('title') or self.flow_name}")
        self.setStyleSheet("QWidget { background: #15161a; color: #e8e8e8; } "
                           "QPushButton { font-size: 20px; padding: 14px 22px; background: #33363c; border-radius: 6px; }"
                           "QPushButton:hover { background: #44484f; } QListWidget { font-size: 18px; }")
        self.resize(1280, 800)
        self._build()
        self.refresh_timer = QTimer(self, interval=REFRESH_MS, timeout=self.refresh)
        self.sample_timer = QTimer(self, interval=SAMPLE_MS, timeout=self.sample)
        self.refresh_timer.start()
        self.refresh()

    # --- layout --------------------------------------------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        # the automatic layout; with a designed screen ("layout": "screen") it stays hidden behind it
        self.classic = QWidget()
        classic = QVBoxLayout(self.classic)
        classic.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.title_label = _big(self.cfg.get("title") or self.flow_name, 28)
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.state_label = _big("Durdu", 20, bold=False)
        self.clock_label = _big("", 24, bold=False)
        top.addWidget(self.title_label, 1)
        top.addWidget(self.state_label)
        top.addSpacing(24)
        top.addWidget(self.clock_label)
        classic.addLayout(top)

        mid = QHBoxLayout()
        self.image_label = QLabel("Görüntü bekleniyor")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("background: #000; font-size: 20px; color: #888;")
        mid.addWidget(self.image_label, 3)

        side = QVBoxLayout()
        self.lamp = _big("BEKLENİYOR", 56)
        self.lamp.setMinimumHeight(150)
        side.addWidget(self.lamp)
        grid = QGridLayout()
        self.counter_labels = {}
        for i, (key, name, color) in enumerate([("good", "Sağlam", GOOD), ("bad", "Hatalı", BAD),
                                                ("total", "Toplam", "#e8e8e8"), ("yield", "Verim", "#64b5f6")]):
            box = QVBoxLayout()
            box.addWidget(_big(name, 18, bold=False))
            lab = _big("0", 48)
            lab.setStyleSheet(lab.styleSheet() + f" color: {color};")
            box.addWidget(lab)
            self.counter_labels[key] = lab
            grid.addLayout(box, i // 2, i % 2)
        side.addLayout(grid)
        side.addWidget(_big("Son sonuçlar", 18, bold=False))
        self.history_list = QListWidget()
        side.addWidget(self.history_list, 1)
        mid.addLayout(side, 2)
        classic.addLayout(mid, 1)
        root.addWidget(self.classic, 1)
        self.screen_view = None
        screen = self.cfg.get("screen")
        if self.cfg.get("layout") == "screen" and screen:
            from .screen_runtime import ScreenView
            self.screen_view = ScreenView(screen, self.graph, self._screen_host())
            root.addWidget(self.screen_view, 1)
            self.classic.hide()
            root.setContentsMargins(0, 0, 0, 0)

        self.bottom_bar = QWidget()
        bottom = QHBoxLayout(self.bottom_bar)
        bottom.setContentsMargins(6, 4, 6, 4)
        self.start_btn = QPushButton("▶  Başlat")
        self.stop_btn = QPushButton("■  Durdur")
        self.reset_btn = QPushButton("Sayaçları sıfırla")
        self.full_btn = QPushButton("Tam ekran (F11)")
        self.trace_btn = QPushButton("Arşiv")
        self.trace_btn.setToolTip("İzlenebilirlik arşivi: seri numarasıyla parça arama")
        self.trace_btn.clicked.connect(self.open_trace)
        self.editor_btn = QPushButton("🔒 Düzenleyici" if self.lock else "Düzenleyici")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.reset_btn.clicked.connect(self.reset_counters)
        self.full_btn.clicked.connect(self.toggle_fullscreen)
        self.editor_btn.clicked.connect(lambda: self.leave())
        for b in (self.start_btn, self.stop_btn, self.reset_btn):
            bottom.addWidget(b)
        bottom.addStretch(1)
        bottom.addWidget(self.trace_btn)
        bottom.addWidget(self.full_btn)
        bottom.addWidget(self.editor_btn)
        self.editor_btn.setVisible(self.editor is not None or self.editor_factory is not None)
        root.addWidget(self.bottom_bar)
        if self.screen_view is not None and not screen.get("show_bar", True):
            self.bottom_bar.hide()           # the designed screen has its own buttons (F11 / Esc still work)
        self.status = QLabel(" ")
        self.status.setStyleSheet("color: #ffb74d; font-size: 14px;")
        root.addWidget(self.status)
        self._update_buttons()

    def _update_buttons(self):
        running = self.worker is not None
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    # --- running -------------------------------------------------------------------------------------------------
    def start(self):
        if self.worker is not None:
            return
        try:
            self.graph.order()
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Akış başlatılamadı: {exc}")
            return
        self.status.setText(" ")
        w = FlowWorker(self.graph_data)
        w.thumbs = False                   # the panel draws no block thumbnails
        w.watch_id = self.image_id or None
        w.tally = self.tally
        if self.screen_view is not None:
            self.screen_view.feed.new_run()
            w.screen = self.screen_view.feed
        # the editor's remote control API (if on) reports this run; without an editor (--operator) the panel's own
        from .api_dialog import worker_state
        w.api = worker_state(self.editor if self.editor is not None else self)
        w.recipe_name = (getattr(self.editor, "active_recipe", "") or "") if self.editor is not None \
            else self.recipe_name
        w.recipe.connect(self._on_recipe)
        w.frame.connect(self.on_frame)
        w.watch.connect(self.on_watch)
        w.failed.connect(self.on_failed)
        w.finished.connect(self._worker_done)
        self.worker = w
        w.start()
        self.sample_timer.start()
        self.state_label.setText("Çalışıyor…")
        self._update_buttons()

    def _on_recipe(self, name, errors):
        self.recipe_name = name
        if self.editor is not None:        # the editor's API reports the recipe the flow runs with
            self.editor.active_recipe = name
        if self.screen_view is not None:
            self.screen_view.host.recipe_name = name
            for e in errors:
                self.screen_view.feed.alarm(f"Reçete '{name}': {e}", "uyarı")

    def stop(self):
        w = self.worker
        if w is None:
            return
        w.stop()
        if not w.wait(5000):
            self.status.setText("Uyarı: akış 5 sn içinde durmadı (kamera yanıt vermiyor olabilir)")
        self._worker_done()

    def _worker_done(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = None
        self.sample_timer.stop()
        self.sample()                      # what was counted since the last minute sample
        self.state_label.setText("Durdu")
        self._update_buttons()

    def on_frame(self, payload: dict, fps: float):
        errs = sum(1 for k, v in payload.items() if v[1] and v[1] != "Önceki blok hata verdi"
                   and not (str(v[1]).startswith("InnerError:") and inner_error(find_node(self.graph, k), v[1])))
        self.state_label.setText(f"Çalışıyor · {fps:.1f} FPS" + (f" · {errs} hata" if errs else "") if fps else
                                 "Çalışıyor" + (f" · {errs} hata" if errs else ""))

    def on_watch(self, node_id, img, size=None):
        if node_id != self.image_id:
            return
        self._img = to_qimage(img)
        self._show_image()

    def _show_image(self):
        qimg = getattr(self, "_img", None)
        if qimg is None or qimg.isNull():
            return
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio,
                                                                  Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._show_image()

    def on_failed(self, msg):
        self.status.setText(f"HATA: {msg}")
        self.tally.alarm(msg)
        if self.screen_view is not None:
            self.screen_view.feed.alarm(msg)

    # --- designed screen -----------------------------------------------------------------------------------------
    def _screen_host(self):
        from .screen_runtime import ScreenHost
        return ScreenHost(self.graph, self.lock, worker=lambda: self.worker, tally=self.tally, start=self.start,
                          stop=self.stop, reset=self.reset_counters, recipe=self._screen_recipe,
                          after_write=self._screen_wrote, message=lambda t: self.status.setText(t))

    def _editor_root(self):
        root = getattr(self.editor, "root", None) if self.editor is not None else None
        return root if isinstance(root, Graph) else None

    def _screen_wrote(self, key, name, value):
        """A setting changed on the designed screen: the panel's next start keeps it, and so does the editor."""
        self.graph_data = self.graph.to_dict()
        self._changed = True
        root = self._editor_root()
        if root is not None:
            from ..core.screens import write_setting
            try:
                write_setting(root, key, name, value)
                self.editor._values_changed()
                self.editor._schedule_snapshot()
            except Exception:  # noqa: BLE001  (the editor's flow may have changed meanwhile)
                pass

    def _screen_recipe(self, name) -> list[str]:
        return self.apply_recipe(name)

    def apply_recipe(self, name, from_editor: bool = False) -> list[str]:
        """Switch to a recipe (designed screen, remote control API, or the editor's API / recipe dialog with
        ``from_editor``): its values go into the panel's flow (shown on the screen, kept for the next start, with
        the 'Reçete Seç' blocks following it like headless --api does), the running flow switches between frames,
        and the editor that opened the panel gets it too. Returns the problems."""
        from ..core import recipes
        from ..headless import _set_selectors
        r = recipes.get(self.graph, name)
        if r is None:
            return [f"'{name}' adlı reçete yok"]
        errors = recipes.apply(self.graph, r)
        _set_selectors(self.graph, r["name"])
        self.recipe_name = r["name"]
        self.graph_data = self.graph.to_dict()
        self._changed = True
        if self.screen_view is not None:
            self.screen_view.host.recipe_name = r["name"]
            self.screen_view.host._settings.clear()        # the setting values shown come from the flow again
        if from_editor:
            return errors
        if self.worker is not None:
            self.worker.request_recipe(r["name"])
        root = self._editor_root()
        if root is not None and recipes.get(root, r["name"]) is not None:
            try:
                recipes.apply(root, recipes.get(root, r["name"]))
                _set_selectors(root, r["name"])
                self.editor.active_recipe = r["name"]
                self.editor._values_changed()
                self.editor._schedule_snapshot()
            except Exception:  # noqa: BLE001
                pass
        return errors

    def sample(self):
        """Write the counts of the last minute (and new alarms) to the production database."""
        try:
            store_sample(self.tally, self.db, self.flow_name)
        except Exception as exc:  # noqa: BLE001  (a full disk must not stop the line)
            self.status.setText(f"Üretim verisi kaydedilemedi: {exc}")

    def reset_counters(self):
        self.tally.reset()
        self._hist_key = None
        self.refresh()

    # --- display -------------------------------------------------------------------------------------------------
    def refresh(self):
        snap = self.tally.snapshot()
        self.clock_label.setText(time.strftime("%H:%M:%S"))
        if not self.cfg.get("lamp") and not self.tally.counter_mode:
            text, color, tip = "LAMBA SEÇİLMEDİ", IDLE, "Düzenleyicide Çalıştır → Operatör paneli ayarları"
        elif snap["lamp_error"]:
            text, color, tip = "HATA", ERR, snap["lamp_error"]
        elif snap["lamp"] is None:
            text, color, tip = "BEKLENİYOR", IDLE, ""
        else:
            text, color, tip = ("SAĞLAM", GOOD, "") if snap["lamp"] else ("HATALI", BAD, "")
        self.lamp.setText(text)
        self.lamp.setToolTip(tip)
        self.lamp.setStyleSheet(f"font-size: 56px; font-weight: bold; color: white; background: {color}; "
                                f"border-radius: 12px;")
        for k in COUNTERS:
            self.counter_labels[k].setText(str(snap[k]))
        self.counter_labels["yield"].setText("–" if snap["yield"] is None else f"%{snap['yield']:g}")
        hist = snap["history"]
        key = (len(hist), hist[0] if hist else None)
        if key != self._hist_key:
            self._hist_key = key
            self.history_list.clear()
            for ts, ok, label in hist:
                item = QListWidgetItem(f"{time.strftime('%H:%M:%S', time.localtime(ts))}   {label}")
                item.setForeground(QColor("#81c784" if ok else "#ef5350"))
                self.history_list.addItem(item)
        return snap

    def open_trace(self):
        from ..nodes.traceability import flow_archive
        from .trace_browser import open_browser
        self._trace_browser = open_browser(self, flow_archive(self.graph))    # the flow's own archive folder

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        elif e.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
        else:
            super().keyPressEvent(e)

    # --- leaving / lock ------------------------------------------------------------------------------------------
    def unlock(self, password: str | None = None) -> bool:
        """The edit lock allows leaving: no lock, the given password matches, or (None) the prompt was answered."""
        if not self.lock:
            return True
        if password is not None:
            return check_password(self.lock, password)
        return ask_password(self, self.lock, "Düzenleyiciyi açmak için şifreyi girin:")

    def leave(self, password: str | None = None, open_editor: bool = True) -> bool:
        """Stop the flow, close the panel and go to the editor (password first when the flow is locked)."""
        if not self.unlock(password):
            self.status.setText("Şifre yanlış: düzenleyici açılmadı")
            return False
        self.stop()
        self._left = True
        self.refresh_timer.stop()
        self._api_off()                # the editor opened now uses the port for its own API
        if self.screen_view is not None:
            self.screen_view.stop_timer()
        if open_editor:       # shown before the panel closes, so the program does not quit with its last window
            if self.editor is None and self.editor_factory is not None:
                self.editor = self.editor_factory()
                self._hand_over()
            if self.editor is not None:
                self.editor.show()
                self.editor.raise_()
        self.close()
        return True

    def _hand_over(self):
        """``--operator``: the editor made now loaded the flow file; what was changed on the panel (settings written
        on the designed screen, a recipe chosen there or over the API) must not be lost: the editor gets the panel's
        flow as an unsaved change, and the recipe it runs with."""
        ed = self.editor
        if ed is None or not self._changed or not hasattr(ed, "_restore"):
            return
        try:
            import json
            ed._restore(json.dumps(self.graph_data, ensure_ascii=False))
            ed._schedule_snapshot()
            ed.active_recipe = self.recipe_name
        except Exception as exc:  # noqa: BLE001
            print(f"Uyarı: paneldeki değişiklikler düzenleyiciye aktarılamadı: {exc}")

    def _api_off(self):
        """The panel's own remote control API (``--operator`` without an editor) is switched off."""
        from .api_dialog import shutdown
        shutdown(self)

    def closeEvent(self, e):
        if not self._left:
            if not self.unlock():
                self.status.setText("Paneli kapatmak için şifre gerekli")
                e.ignore()
                return
            self.stop()
            self._left = True
            self.refresh_timer.stop()
            self._api_off()
            if self.screen_view is not None:
                self.screen_view.stop_timer()
            if self.editor is not None:           # opened from the editor: go back to it
                self.editor.show()
        super().closeEvent(e)


# --- settings dialog -------------------------------------------------------------------------------------------------

def _outputs(graph: Graph, kinds) -> list[tuple[str, dict]]:
    """Outputs of the given types, of blocks inside sub-flows too ("subflow id/inner id", "Alt akış › Blok")."""
    out = []
    for key, label, node in walk_nodes(graph):
        for port, t in node.outputs:
            if t in kinds:
                out.append((f"{label} ({key}) → {port}", {"node": key, "port": port}))
    return out


class OperatorConfigDialog(QDialog):
    """What the operator panel shows, and the edit lock. ``result_config()`` is saved as the flow's "operator"."""

    def __init__(self, parent, graph: Graph):
        super().__init__(parent)
        self.setWindowTitle("Operatör paneli ayarları")
        self.graph = graph
        cfg = dict(graph.operator or auto_config(graph))
        self.old_lock = cfg.get("lock")
        form = QFormLayout(self)
        self.title_edit = QLineEdit(cfg.get("title", ""))
        self.title_edit.setPlaceholderText("Boş = akış dosyasının adı")
        form.addRow("Başlık", self.title_edit)
        self.image_combo = QComboBox()
        self.image_combo.addItem("Otomatik (Önizleme bloğu)", "")
        for key, label, node in walk_nodes(graph):
            if any(t in (PortType.IMAGE, PortType.ANY) for _, t in node.outputs):
                self.image_combo.addItem(f"{label} ({key})", key)
        self._select(self.image_combo, cfg.get("image", ""))
        form.addRow("Canlı görüntü", self.image_combo)
        self.lamp_combo = self._ref_combo(_outputs(graph, (PortType.BOOL,)), cfg.get("lamp"), "(yok)")
        form.addRow("SAĞLAM / HATALI lambası", self.lamp_combo)
        nums = _outputs(graph, (PortType.NUMBER, PortType.BOOL))
        self.counter_combos = {}
        for key, label in [("good", "Sağlam sayacı"), ("bad", "Hatalı sayacı"), ("total", "Toplam sayacı")]:
            empty = "(lambadan say)" if key != "total" else "(sağlam + hatalı)"
            self.counter_combos[key] = self._ref_combo(nums, cfg.get(key), empty)
            form.addRow(label, self.counter_combos[key])
        hint = QLabel("Sayaç seçilmezse lambanın her yeni sonucu bir parça sayılır (tetikli kamera ile kullanın).")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        self.history_spin = QSpinBox()
        self.history_spin.setRange(1, HISTORY_MAX)
        self.history_spin.setValue(int(cfg.get("history", 20)))
        form.addRow("Son sonuç sayısı", self.history_spin)
        self.screen = cfg.get("screen")
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Otomatik (hazır düzen)", "auto")
        self.layout_combo.addItem("Tasarım ekranı", "screen")
        self._select(self.layout_combo, cfg.get("layout", "auto") if self.screen else "auto")
        self.layout_combo.setEnabled(bool(self.screen))
        if not self.screen:
            self.layout_combo.setToolTip("Önce Araçlar → Ekran tasarımcısı ile bir ekran tasarlayın")
        form.addRow("Panel düzeni", self.layout_combo)
        self.pw1, self.pw2 = QLineEdit(), QLineEdit()
        for w in (self.pw1, self.pw2):
            w.setEchoMode(QLineEdit.Password)
        self.pw1.setPlaceholderText("Değiştirmek için yazın" if self.old_lock else "Boş = kilit yok")
        form.addRow("Düzenleme şifresi", self.pw1)
        form.addRow("Şifre (tekrar)", self.pw2)
        self.remove_lock = QCheckBox("Kilidi kaldır")
        self.remove_lock.setVisible(bool(self.old_lock))
        form.addRow(self.remove_lock)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #ef5350;")
        form.addRow(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    @staticmethod
    def _select(combo, data):
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _ref_combo(self, items, current, empty):
        c = QComboBox()
        c.addItem(empty, None)
        for text, ref in items:
            c.addItem(text, ref)
        self._select(c, current)
        return c

    def result_config(self) -> dict:
        """The chosen settings. Raises ValueError (Turkish) for a bad password."""
        cfg = {"title": self.title_edit.text().strip(), "image": self.image_combo.currentData() or "",
               "lamp": self.lamp_combo.currentData(), "history": self.history_spin.value(),
               **{k: c.currentData() for k, c in self.counter_combos.items()}}
        if self.screen:                     # the designed screen stays (Araçlar → Ekran tasarımcısı edits it)
            cfg["screen"] = self.screen
            cfg["layout"] = self.layout_combo.currentData()
        p1, p2 = self.pw1.text(), self.pw2.text()
        if p1 or p2:
            if p1 != p2:
                raise ValueError("Şifreler aynı değil")
            cfg["lock"] = make_lock(p1)
        elif self.old_lock and not self.remove_lock.isChecked():
            cfg["lock"] = self.old_lock
        return cfg

    def accept(self):
        try:
            self.config = self.result_config()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()


# --- production report dialog ------------------------------------------------------------------------------------

class ReportDialog(QDialog):
    """Araçlar → Üretim raporu: an HTML (and PDF) report of the counts the operator panel collected."""

    def __init__(self, parent=None, flow_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Üretim raporu")
        self.db = ProductionDB(db_path(flow_path))
        self.flow_path = flow_path
        form = QFormLayout(self)
        form.addRow(QLabel(f"Veri dosyası: {self.db.path}"))
        now = QDateTime.currentDateTime()
        self.start_edit = QDateTimeEdit(now.addSecs(-8 * 3600))
        self.end_edit = QDateTimeEdit(now)
        for w in (self.start_edit, self.end_edit):
            w.setDisplayFormat("dd.MM.yyyy HH:mm")
            w.setCalendarPopup(True)
        form.addRow("Başlangıç", self.start_edit)
        form.addRow("Bitiş", self.end_edit)
        self.flow_combo = QComboBox()
        self.flow_combo.addItem("Tüm akışlar", "")
        try:
            flows = self.db.flows()
        except Exception:  # noqa: BLE001
            flows = []
        for f in flows:
            self.flow_combo.addItem(f, f)
        if flow_path:
            OperatorConfigDialog._select(self.flow_combo, pathlib.Path(flow_path).stem)
        form.addRow("Akış", self.flow_combo)
        self.title_edit = QLineEdit("Üretim raporu")
        form.addRow("Başlık", self.title_edit)
        self.pdf_check = QCheckBox("PDF de oluştur")
        self.pdf_check.setChecked(True)
        form.addRow(self.pdf_check)
        bb = QDialogButtonBox()
        bb.addButton("Rapor oluştur…", QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Close)
        bb.accepted.connect(self.ask_and_generate)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def generate(self, out_html) -> tuple[str, str | None, str | None]:
        start = self.start_edit.dateTime().toPython()
        end = self.end_edit.dateTime().toPython()
        return make_report(self.db, start, end, out_html, self.flow_combo.currentData() or None,
                           self.title_edit.text().strip() or "Üretim raporu", self.pdf_check.isChecked())

    def ask_and_generate(self):
        default = str(self.db.path.parent / f"uretim_raporu_{dt.datetime.now():%Y%m%d_%H%M}.html")
        path, _ = QFileDialog.getSaveFileName(self, "Raporu kaydet", default, "HTML rapor (*.html)")
        if not path:
            return
        try:
            html_path, pdf_path, msg = self.generate(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Rapor oluşturulamadı", str(exc))
            return
        text = f"Rapor yazıldı:\n{html_path}" + (f"\n{pdf_path}" if pdf_path else "") + (f"\n\n{msg}" if msg else "")
        QMessageBox.information(self, "Üretim raporu", text)
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))


def run_operator(path: str, editor_factory=None) -> OperatorPanel:
    """``python -m visionflow --operator akis.json``: the panel alone, full screen, running the flow."""
    graph = Graph.load(path)
    for w in graph.warnings:
        print(f"Uyarı: {w}")
    panel = OperatorPanel(graph.to_dict(), path, editor_factory=editor_factory)
    return panel
