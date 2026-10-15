"""Araçlar → İstasyon yöneticisi…: several flows (stations / cameras) at once, each in its own process.

The window owns a stations.Supervisor and calls its tick() from a timer; the stations are child processes, so a
crashing camera never freezes the editor or the other stations. Pictures come from the preview JPEG every station
writes once a second, the log from its log file.
"""
from __future__ import annotations

import os
import pathlib

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .. import stations as stn

TICK_MS = 300
THUMB_W, THUMB_H = 96, 72
COLUMNS = ["", "Önizleme", "İstasyon", "Durum", "FPS", "Kare", "Hata", "Yeniden", "Süre", "Son hata"]


def _duration(sec: float) -> str:
    sec = int(sec)
    if sec <= 0:
        return "-"
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class StationDialog(QDialog):
    """Add / edit one station. ``station()`` gives the checked result (None when cancelled)."""

    def __init__(self, parent, station: stn.Station | None = None, base: pathlib.Path | None = None):
        super().__init__(parent)
        self.base = base
        self.setWindowTitle("İstasyonu düzenle" if station else "İstasyon ekle")
        st = station or stn.Station(ad="", akis="")
        form = QFormLayout(self)
        self.name = QLineEdit(st.ad)
        self.name.setPlaceholderText("örn. Hat 1 – kapak kontrol")
        form.addRow("Ad", self.name)
        row = QHBoxLayout()
        self.flow = QLineEdit(st.akis)
        self.flow.setPlaceholderText("akış dosyası (.json)")
        pick = QPushButton("Seç…")
        pick.clicked.connect(self.pick_flow)
        row.addWidget(self.flow, 1)
        row.addWidget(pick)
        form.addRow("Akış", row)
        self.recipe = QComboBox()
        self.recipe.setEditable(True)
        self.recipe.addItem("")
        self.recipe.setCurrentText(st.recete)
        self.recipe.setToolTip("Akış bu reçeteyle (ürün tipiyle) başlar; boş = akıştaki ayarlarla")
        form.addRow("Reçete", self.recipe)
        self.auto = QCheckBox("Yönetici başlayınca bu istasyon da başlasın (servis / komut satırı)")
        self.auto.setChecked(st.otomatik_baslat)
        form.addRow("", self.auto)
        self.restart = QCheckBox("Çökerse yeniden başlat")
        self.restart.setChecked(st.yeniden_baslat)
        form.addRow("", self.restart)
        self.max_restarts = QSpinBox()
        self.max_restarts.setRange(0, 1_000_000)
        self.max_restarts.setSpecialValueText("sınırsız")
        self.max_restarts.setValue(st.en_fazla_yeniden)
        form.addRow("En fazla yeniden başlatma (art arda)", self.max_restarts)
        self.backoff = QDoubleSpinBox()
        self.backoff.setRange(0, 3600)
        self.backoff.setDecimals(1)
        self.backoff.setSuffix(" sn")
        self.backoff.setValue(st.bekleme_sn)
        form.addRow("İlk bekleme (her çöküşte iki katı)", self.backoff)
        self.backoff_max = QDoubleSpinBox()
        self.backoff_max.setRange(0, 86400)
        self.backoff_max.setDecimals(1)
        self.backoff_max.setSuffix(" sn")
        self.backoff_max.setValue(st.en_fazla_bekleme_sn)
        form.addRow("En uzun bekleme", self.backoff_max)
        self.priority = QComboBox()
        for key in stn.PRIORITIES:
            self.priority.addItem(stn.PRIORITY_LABELS[key], key)
        self.priority.setCurrentIndex(stn.PRIORITIES.index(st.oncelik))
        form.addRow("İşlemci önceliği", self.priority)
        self.cpus = QLineEdit(", ".join(str(c) for c in st.cekirdekler))
        self.cpus.setPlaceholderText(f"boş = hepsi; örn. 0, 1  (bu bilgisayarda {os.cpu_count() or 1} çekirdek)")
        form.addRow("Çekirdekler", self.cpus)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0, 1000)
        self.fps.setDecimals(1)
        self.fps.setSpecialValueText("sınırsız")
        self.fps.setValue(st.fps)
        form.addRow("En fazla FPS", self.fps)
        self.frames = QSpinBox()
        self.frames.setRange(0, 2_000_000_000)
        self.frames.setSpecialValueText("sınırsız")
        self.frames.setValue(min(st.kare, 2_000_000_000))
        form.addRow("Bu kadar kareden sonra dur", self.frames)
        self.api_port = QSpinBox()
        self.api_port.setRange(0, 65535)
        self.api_port.setSpecialValueText("kapalı")
        self.api_port.setValue(st.api_portu)
        form.addRow("REST API portu", self.api_port)
        self.api_net = QCheckBox("API ağa açık (0.0.0.0)")
        self.api_net.setChecked(st.api_ag)
        form.addRow("", self.api_net)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #e57373;")
        self.error.setWordWrap(True)
        form.addRow(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.flow.editingFinished.connect(self._load_recipes)
        self._load_recipes()
        self.result_station: stn.Station | None = None

    def pick_flow(self):
        path, _ = QFileDialog.getOpenFileName(self, "Akış seç", str(self.base or ""), "Akış (*.json)")
        if path:
            self.flow.setText(path)
            self._load_recipes()

    def _load_recipes(self):
        current = self.recipe.currentText()
        names = []
        flow = self.flow.text().strip()
        if flow:
            try:
                from ..core import Graph, load_builtin, recipes
                load_builtin()
                p = stn.Station(ad="x", akis=flow).flow_path(self.base)
                names = recipes.names(Graph.load(str(p))) if p.is_file() else []
            except Exception:  # noqa: BLE001  (a broken flow: the recipe can still be typed)
                names = []
        self.recipe.blockSignals(True)
        self.recipe.clear()
        self.recipe.addItems([""] + names)
        self.recipe.setCurrentText(current)
        self.recipe.blockSignals(False)

    def raw(self) -> dict:
        cpus_text = self.cpus.text().replace(";", ",").strip()
        try:
            cpus = [int(c) for c in cpus_text.split(",") if c.strip()]
        except ValueError:
            cpus = "geçersiz"
        return {"ad": self.name.text(), "akis": self.flow.text(), "recete": self.recipe.currentText(),
                "otomatik_baslat": self.auto.isChecked(), "yeniden_baslat": self.restart.isChecked(),
                "en_fazla_yeniden": self.max_restarts.value(), "bekleme_sn": self.backoff.value(),
                "en_fazla_bekleme_sn": self.backoff_max.value(), "oncelik": self.priority.currentData(),
                "cekirdekler": cpus, "api_portu": self.api_port.value(), "api_ag": self.api_net.isChecked(),
                "fps": self.fps.value(), "kare": self.frames.value()}

    def accept(self):
        try:
            self.result_station = stn.station_from_dict(self.raw())
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()


class StationManager(QWidget):
    """The station manager window. ``editor`` (optional) is the main window: 'Akışı aç' opens flows in it."""

    def __init__(self, editor=None, path=None):
        super().__init__(None)
        self.editor = editor
        self.path: pathlib.Path | None = None
        self.sup: stn.Supervisor | None = None
        self._thumbs: dict[str, tuple[float, QPixmap]] = {}
        self._log_tick = 0
        self.setWindowTitle("İstasyon Yöneticisi")
        self.resize(1180, 760)
        self._build()
        self.timer = QTimer(self, interval=TICK_MS, timeout=self.tick)
        if path:
            self.load(path)
        self._update_buttons()

    # --- layout ---------------------------------------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.file_label = QLabel("İstasyon dosyası açık değil")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        b_new, b_open = QPushButton("Yeni…"), QPushButton("Aç…")
        b_new.clicked.connect(self.new_file)
        b_open.clicked.connect(self.open_file)
        top.addWidget(self.file_label, 1)
        top.addWidget(b_new)
        top.addWidget(b_open)
        root.addLayout(top)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(THUMB_H + 6)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selected_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self.edit_station())

        bar = QHBoxLayout()
        self.b_start, self.b_stop, self.b_restart = (QPushButton("▶ Başlat"), QPushButton("■ Durdur"),
                                                     QPushButton("⟳ Yeniden başlat"))
        self.b_start_all, self.b_stop_all = QPushButton("▶ Tümünü başlat"), QPushButton("■ Tümünü durdur")
        self.b_add, self.b_edit, self.b_del = QPushButton("Ekle…"), QPushButton("Düzenle…"), QPushButton("Sil")
        self.b_open_flow = QPushButton("Akışı düzenleyicide aç")
        for b, fn in ((self.b_start, self.start_selected), (self.b_stop, self.stop_selected),
                      (self.b_restart, self.restart_selected), (self.b_start_all, self.start_all),
                      (self.b_stop_all, self.stop_all), (self.b_add, self.add_station),
                      (self.b_edit, self.edit_station), (self.b_del, self.remove_station),
                      (self.b_open_flow, self.open_flow)):
            b.clicked.connect(fn)
        for b in (self.b_start, self.b_stop, self.b_restart):
            bar.addWidget(b)
        bar.addSpacing(16)
        bar.addWidget(self.b_start_all)
        bar.addWidget(self.b_stop_all)
        bar.addStretch(1)
        for b in (self.b_add, self.b_edit, self.b_del, self.b_open_flow):
            bar.addWidget(b)

        bottom = QSplitter(Qt.Horizontal)
        self.preview = QLabel("Önizleme yok")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setStyleSheet("background: #000; color: #888;")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QFont("monospace", 9))
        self.log_view.setPlaceholderText("Seçili istasyonun kaydı")
        bottom.addWidget(self.preview)
        bottom.addWidget(self.log_view)
        bottom.setStretchFactor(1, 1)

        split = QSplitter(Qt.Vertical)
        holder = QWidget()
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.table, 1)
        hl.addLayout(bar)
        split.addWidget(holder)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)
        self.info = QLabel(" ")
        self.info.setStyleSheet("color: #ffb74d;")
        root.addWidget(self.info)

    # --- file -----------------------------------------------------------------------------------------------
    def load(self, path) -> bool:
        """Open a station file (the stations of the file open before are stopped first)."""
        try:
            stations = stn.load_stations(path)
        except ValueError as exc:
            self._warn(str(exc))
            return False
        sup = stn.Supervisor(stations, source=pathlib.Path(path).resolve())
        other = sup.claim()
        if other:
            stn._LIVE.discard(sup)
            self._warn(f"Bu istasyon dosyası başka bir istasyon yöneticisinde çalışıyor (işlem {other}: servis ya da "
                       "komut satırı). Aynı istasyonlar iki kez çalışmasın diye burada açılmadı; önce onu durdurun.")
            return False
        self._close_supervisor()
        self.path = pathlib.Path(path).resolve()
        self.sup = sup
        self._thumbs.clear()
        self.file_label.setText(f"İstasyon dosyası: {self.path}")
        self._fill()
        self.timer.start()
        self.info.setText(f"{len(stations)} istasyon yüklendi. Kayıtlar: {self.sup.run_dir}")
        return True

    def new_file(self, path=None) -> bool:
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Yeni istasyon dosyası", "istasyonlar.json",
                                                  "İstasyon dosyası (*.json)")
            if not path:
                return False
        try:
            stn.save_stations(path, [])
        except (OSError, ValueError) as exc:
            self._warn(f"Dosya yazılamadı: {exc}")
            return False
        return self.load(path)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "İstasyon dosyası aç", "", "İstasyon dosyası (*.json)")
        if path:
            self.load(path)

    def _save(self) -> bool:
        if self.sup is None or self.path is None:
            return False
        try:
            stn.save_stations(self.path, self.sup.stations)
            return True
        except (OSError, ValueError) as exc:
            self._warn(f"İstasyon dosyası kaydedilemedi: {exc}")
            return False

    def _warn(self, msg: str):
        self.info.setText(msg)
        QMessageBox.warning(self, "İstasyon yöneticisi", msg)

    # --- table ----------------------------------------------------------------------------------------------
    def _fill(self):
        names = [s.ad for s in self.sup.stations] if self.sup else []
        keep = self.selected_name()
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            for col in range(len(COLUMNS)):
                if self.table.item(row, col) is None:
                    self.table.setItem(row, col, QTableWidgetItem(""))
            self.table.item(row, 2).setText(name)
            lamp = self.table.item(row, 0)
            lamp.setText("●")
            f = lamp.font()
            f.setPointSize(20)
            lamp.setFont(f)
            lamp.setTextAlignment(Qt.AlignCenter)
        if keep in names:
            self.table.selectRow(names.index(keep))
        elif names and not self.table.selectedItems():
            self.table.selectRow(0)
        self.refresh()

    def selected_name(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 2)
        return item.text() if item else None

    def select(self, name: str) -> None:
        for row in range(self.table.rowCount()):
            if self.table.item(row, 2).text() == name:
                self.table.selectRow(row)
                return

    def _thumb(self, name: str) -> QPixmap | None:
        p = self.sup.preview_path(name)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return None
        old = self._thumbs.get(name)
        if old and old[0] == mtime:
            return old[1]
        pix = QPixmap(str(p))
        if pix.isNull():                    # being replaced right now: keep the old one
            return old[1] if old else None
        self._thumbs[name] = (mtime, pix)
        return pix

    def refresh(self):
        if self.sup is None:
            return
        rows = {r["ad"]: r for r in self.sup.status()}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 2).text()
            r = rows.get(name)
            if r is None:
                continue
            self.table.item(row, 0).setForeground(QColor(r["renk"]))
            self.table.item(row, 0).setToolTip(r["durum_metni"])
            pix = self._thumb(name)
            thumb = self.table.item(row, 1)
            if pix is not None:
                thumb.setData(Qt.DecorationRole, pix.scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation))
            else:
                thumb.setData(Qt.DecorationRole, None)
            text = r["durum_metni"]
            if r["durum"] == "yeniden":
                text += f" ({r['bekleme_sn']:g} sn)"
            if r["pid"]:
                text += f"  [pid {r['pid']}]"
            vals = {3: text, 4: f"{r['fps']:.1f}" if r["fps"] else "-", 5: str(r["kare"]), 6: str(r["hata_sayisi"]),
                    7: str(r["yeniden"]), 8: _duration(r["sure_sn"]), 9: str(r["son_hata"] or "")}
            for col, v in vals.items():
                it = self.table.item(row, col)
                if it.text() != v:
                    it.setText(v)
            self.table.item(row, 9).setToolTip(str(r["son_hata"] or ""))
        self._show_selected()
        self._update_buttons()

    def _selected_changed(self):
        self._log_tick = 0
        self._show_selected(force_log=True)
        self._update_buttons()

    def _show_selected(self, force_log: bool = False):
        name = self.selected_name()
        if self.sup is None or name is None:
            self.preview.setText("Önizleme yok")
            self.preview.setPixmap(QPixmap())
            self.log_view.setPlainText("")
            return
        pix = self._thumb(name)
        if pix is not None:
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Önizleme yok (istasyon çalışmıyor ya da akışta görüntü yok)")
        self._log_tick += 1
        if force_log or self._log_tick % 3 == 0:          # the log about once a second
            text = self.sup.log_tail(name, 300)
            if text != self.log_view.toPlainText():
                bar = self.log_view.verticalScrollBar()
                at_end = bar.value() >= bar.maximum() - 2
                self.log_view.setPlainText(text)
                if at_end or force_log:
                    bar.setValue(bar.maximum())

    def _update_buttons(self):
        has = self.sup is not None
        name = self.selected_name()
        run = self.sup.runs.get(name) if has and name else None
        alive = run is not None and run.proc is not None
        self.b_start.setEnabled(run is not None and (not alive or run.stopping))
        self.b_stop.setEnabled(run is not None and (alive or run.state == "yeniden") and not run.stopping)
        self.b_restart.setEnabled(run is not None)
        self.b_start_all.setEnabled(has and bool(self.sup.runs))
        self.b_stop_all.setEnabled(has and self.sup.busy)
        self.b_add.setEnabled(has)
        self.b_edit.setEnabled(run is not None)
        self.b_del.setEnabled(run is not None)
        self.b_open_flow.setEnabled(run is not None and self.editor is not None)

    # --- actions --------------------------------------------------------------------------------------------
    def tick(self):
        if self.sup is not None:
            self.sup.tick()
            self.refresh()

    def _act(self, fn, *args):
        try:
            fn(*args)
        except ValueError as exc:
            self._warn(str(exc))
        self.refresh()

    def start_selected(self):
        if self.selected_name():
            self._act(self.sup.start, self.selected_name())

    def stop_selected(self):
        if self.selected_name():
            self._act(self.sup.stop, self.selected_name())

    def restart_selected(self):
        if self.selected_name():
            self._act(self.sup.restart, self.selected_name())

    def start_all(self):
        if self.sup is not None:
            self._act(self.sup.start_all)

    def stop_all(self):
        if self.sup is not None:
            self._act(self.sup.stop_all)

    def _dialog(self, station=None) -> stn.Station | None:
        dlg = StationDialog(self, station, self.path.parent if self.path else None)
        self._dialog_open = dlg                 # tests fill it in
        if dlg.exec() != QDialog.Accepted:
            return None
        st = dlg.result_station
        if st is not None and self.path is not None:
            st.akis = stn.relative_flow(st.akis, self.path)
        return st

    def add_station(self, station: stn.Station | None = None) -> bool:
        if self.sup is None:
            return False
        st = station or self._dialog()
        if st is None:
            return False
        return self._set(self.sup.stations + [st], st.ad)

    def edit_station(self, station: stn.Station | None = None) -> bool:
        name = self.selected_name()
        if self.sup is None or name is None:
            return False
        st = station or self._dialog(self.sup.runs[name].st)
        if st is None:
            return False
        new = [st if s.ad == name else s for s in self.sup.stations]
        if not self._set(new, st.ad):           # renamed: the old name's process is stopped
            return False
        run = self.sup.runs[st.ad]
        if run.proc is not None:
            self.info.setText(f"'{st.ad}' yeni ayarlarla yeniden başlatıldığında çalışır (⟳ Yeniden başlat).")
        return True

    def remove_station(self, confirm: bool = True) -> bool:
        name = self.selected_name()
        if self.sup is None or name is None:
            return False
        if confirm:
            if QMessageBox.question(self, "İstasyonu sil", f"'{name}' istasyonu listeden silinsin mi? "
                                    "(Çalışıyorsa durdurulur; akış dosyası silinmez.)") != QMessageBox.Yes:
                return False
        return self._set([s for s in self.sup.stations if s.ad != name])

    def _set(self, stations, select: str | None = None) -> bool:
        try:
            self.sup.set_stations(stations)
        except ValueError as exc:
            self._warn(str(exc))
            return False
        self._save()
        self._fill()
        if select:
            self.select(select)
        return True

    def open_flow(self):
        name = self.selected_name()
        if self.editor is None or self.sup is None or name is None:
            return
        path = self.sup.runs[name].st.flow_path(self.sup.base)
        if not path.is_file():
            self._warn(f"Akış dosyası bulunamadı: {path}")
            return
        self.editor.open_flow(str(path))
        self.editor.show()
        self.editor.raise_()
        self.editor.activateWindow()

    # --- closing --------------------------------------------------------------------------------------------
    def ask_stop(self) -> bool:
        """Closing with stations running: stop them? (Tests replace this.)"""
        return QMessageBox.question(self, "İstasyon yöneticisi",
                                    "Çalışan istasyonlar var. Pencere kapanınca hepsi durdurulacak. Devam edilsin mi?"
                                    ) == QMessageBox.Yes

    def _close_supervisor(self):
        self.timer.stop()
        if self.sup is not None:
            self.sup.shutdown()
            self.sup = None

    def closeEvent(self, e):
        if self.sup is not None and self.sup.busy and not self.ask_stop():
            e.ignore()
            return
        self._close_supervisor()
        super().closeEvent(e)


def open_station_manager(win) -> StationManager:
    """One station manager per editor window: shown again when it is already open."""
    mgr = getattr(win, "_station_manager", None)
    try:
        if mgr is not None and mgr.isVisible():
            mgr.raise_()
            mgr.activateWindow()
            return mgr
    except RuntimeError:            # deleted by Qt
        mgr = None
    if mgr is None or mgr.sup is None:
        mgr = StationManager(win)
        win._station_manager = mgr
    mgr.show()
    return mgr
