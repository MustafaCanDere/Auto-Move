"""Araçlar → İzlenebilirlik arşivi…: search the parts archived by 'Parçayı Arşivle' (date, serial, verdict, flow,
station, recipe), look at their pictures and values, and export them (CSV, Excel, ZIP with pictures)."""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import cv2
import numpy as np
from PySide6.QtCore import QDateTime, QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDateTimeEdit, QDialog, QFileDialog,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout,
                               QWidget)

from .. import trace
from .worker import shrink, to_qimage

PAGE = 100
THUMB = 64
ALL = "Hepsi"
GOOD, BAD = "#2e7d32", "#c62828"
COLS = ["", "Zaman", "Seri no", "Lot", "Sonuç", "Sebep", "İstasyon", "Akış", "Reçete"]


def _thumb(path: str) -> QIcon | None:
    """A small picture for the table, decoded at 1/4 or 1/8 size (fast for big JPEGs)."""
    if not path:
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return None
    img = shrink(img, THUMB * 2)
    return QIcon(QPixmap.fromImage(to_qimage(img)))


class TraceBrowser(QDialog):
    def __init__(self, parent=None, root=None):
        super().__init__(parent)
        self.setWindowTitle("İzlenebilirlik arşivi")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(1200, 760)
        self.archive = trace.Archive(root)
        self.page = 0
        self.total = 0
        self.records: list[dict] = []
        self._image = None

        root_lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Arşiv klasörü:"))
        self.folder_edit = QLineEdit(str(self.archive.root))
        self.folder_edit.setReadOnly(True)
        top.addWidget(self.folder_edit, 1)
        change = QPushButton("Değiştir…")
        change.clicked.connect(self.choose_folder)
        top.addWidget(change)
        root_lay.addLayout(top)

        f = QGridLayout()
        self.date_check = QCheckBox("Tarih aralığı")
        now = QDateTime.currentDateTime()
        self.start_edit = QDateTimeEdit(now.addDays(-1))
        self.end_edit = QDateTimeEdit(now.addSecs(3600))
        for w in (self.start_edit, self.end_edit):
            w.setDisplayFormat("dd.MM.yyyy HH:mm")
            w.setCalendarPopup(True)
            w.setEnabled(False)
        self.date_check.toggled.connect(self.start_edit.setEnabled)
        self.date_check.toggled.connect(self.end_edit.setEnabled)
        f.addWidget(self.date_check, 0, 0)
        f.addWidget(self.start_edit, 0, 1)
        f.addWidget(QLabel("–"), 0, 2)
        f.addWidget(self.end_edit, 0, 3)
        f.addWidget(QLabel("Seri no / lot içerir"), 0, 4)
        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText("ör. ABC123")
        self.serial_edit.returnPressed.connect(self.search)
        f.addWidget(self.serial_edit, 0, 5)
        f.addWidget(QLabel("Sonuç"), 0, 6)
        self.verdict_combo = QComboBox()
        self.verdict_combo.addItems([ALL, "Sağlam", "Hatalı"])
        f.addWidget(self.verdict_combo, 0, 7)
        self.combos = {}
        for (key, label), col in zip([("flow", "Akış"), ("station", "İstasyon"), ("recipe", "Reçete")], (0, 4, 6)):
            f.addWidget(QLabel(label), 1, col)
            c = QComboBox()
            c.setMinimumWidth(140)
            self.combos[key] = c
            f.addWidget(c, 1, col + 1)
        f.addWidget(QLabel("Sebep"), 1, 2)
        self.reason_edit = QLineEdit()
        self.reason_edit.setPlaceholderText("içerir")
        self.reason_edit.returnPressed.connect(self.search)
        f.addWidget(self.reason_edit, 1, 3)
        search_btn = QPushButton("🔎 Ara")
        search_btn.setDefault(True)
        search_btn.clicked.connect(self.search)
        f.addWidget(search_btn, 1, 8)
        root_lay.addLayout(f)

        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setIconSize(QSize(THUMB, THUMB))
        self.table.verticalHeader().setDefaultSectionSize(THUMB + 4)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.show_selected)
        split.addWidget(self.table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel("Görüntü yok")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("background: #111; color: #888;")
        rl.addWidget(self.image_label, 3)
        self.annotated_check = QCheckBox("İşaretli görüntüyü göster")
        self.annotated_check.setChecked(True)
        self.annotated_check.toggled.connect(self.show_selected)
        rl.addWidget(self.annotated_check)
        self.values_view = QTextBrowser()
        rl.addWidget(self.values_view, 2)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root_lay.addWidget(split, 1)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Önceki")
        self.next_btn = QPushButton("Sonraki ▶")
        self.prev_btn.clicked.connect(lambda: self.go(self.page - 1))
        self.next_btn.clicked.connect(lambda: self.go(self.page + 1))
        self.page_label = QLabel()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.page_label)
        nav.addWidget(self.next_btn)
        nav.addSpacing(20)
        self.summary_label = QLabel()
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        nav.addWidget(self.summary_label, 1)
        exp_sel = QPushButton("Seçilenleri dışa aktar…")
        exp_all = QPushButton("Filtrelenenleri dışa aktar…")
        exp_sel.clicked.connect(lambda: self.ask_export(selected=True))
        exp_all.clicked.connect(lambda: self.ask_export(selected=False))
        nav.addWidget(exp_sel)
        nav.addWidget(exp_all)
        root_lay.addLayout(nav)

        self.reload_choices()
        self.search()

    # --- data ---------------------------------------------------------------------------------------------------
    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Arşiv klasörü", str(self.archive.root))
        if path:
            self.set_root(path)

    def set_root(self, path):
        self.archive = trace.Archive(path)
        self.folder_edit.setText(str(self.archive.root))
        self.reload_choices()
        self.search()

    def reload_choices(self):
        for key, c in self.combos.items():
            cur = c.currentText()
            c.blockSignals(True)
            c.clear()
            c.addItem(ALL)
            try:
                c.addItems(self.archive.distinct(key))
            except ValueError:
                pass
            i = c.findText(cur)
            c.setCurrentIndex(max(0, i))
            c.blockSignals(False)

    def current_filter(self) -> trace.Filter:
        flt = trace.Filter(serial=self.serial_edit.text(), reason=self.reason_edit.text())
        if self.date_check.isChecked():
            flt.start = self.start_edit.dateTime().toPython().timestamp()
            flt.end = self.end_edit.dateTime().toPython().timestamp()
        v = self.verdict_combo.currentText()
        flt.verdict = None if v == ALL else (v == "Sağlam")
        for key, c in self.combos.items():
            setattr(flt, key, "" if c.currentText() == ALL else c.currentText())
        return flt

    def search(self):
        self.page = 0
        self.refresh()

    def go(self, page: int):
        pages = max(1, -(-self.total // PAGE))
        self.page = min(max(0, page), pages - 1)
        self.refresh(count=False)

    def refresh(self, count: bool = True):
        flt = self.current_filter()
        try:
            if count:
                self.total = self.archive.count(flt)
                self.show_summary(self.archive.summary(flt))
            self.records = self.archive.search(flt, limit=PAGE, offset=self.page * PAGE)
        except ValueError as exc:
            self.records, self.total = [], 0
            self.summary_label.setText(f"<span style='color:{BAD}'>{exc}</span>")
        self.fill_table()
        pages = max(1, -(-self.total // PAGE))
        self.page_label.setText(f"Sayfa {self.page + 1}/{pages} ({self.total} kayıt)")
        self.prev_btn.setEnabled(self.page > 0)
        self.next_btn.setEnabled(self.page + 1 < pages)

    def show_summary(self, s: dict):
        if not self.archive.exists():
            self.summary_label.setText("Bu klasörde henüz arşiv yok. Akışa 'Parçayı Arşivle' bloğunu ekleyin.")
            return
        pct = "–" if s["fail_pct"] is None else f"%{s['fail_pct']:.1f}"
        reasons = ", ".join(f"{r} ({n})" for r, n in s["top_reasons"][:3]) or "–"
        self.summary_label.setText(
            f"<b>{s['count']}</b> parça · <span style='color:{GOOD}'>sağlam {s['good']}</span> · "
            f"<span style='color:{BAD}'>hatalı {s['bad']}</span> · hata oranı <b>{pct}</b>"
            f"<br>En sık sebepler: {reasons}")

    def fill_table(self):
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.records))
        for row, r in enumerate(self.records):
            verdict = "—" if r["ok"] is None else ("SAĞLAM" if r["ok"] else "HATALI")
            cells = ["", f"{r['time']:%d.%m.%Y %H:%M:%S}", r["serial"], r["lot"], verdict, r["reason"],
                     r["station"], r["flow"], r["recipe"]]
            for col, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if col == 0:
                    it.setData(Qt.UserRole, r["id"])
                    icon = _thumb(r["ann_path"] or r["img_path"])
                    if icon is not None:
                        it.setIcon(icon)
                if col == 4 and r["ok"] is not None:
                    it.setForeground(Qt.white)
                    it.setBackground(Qt.darkGreen if r["ok"] else Qt.darkRed)
                self.table.setItem(row, col, it)
        self.table.setColumnWidth(0, THUMB + 8)
        if self.records:
            self.table.selectRow(0)
        else:
            self.show_record(None)

    def selected_records(self) -> list[dict]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.records[r] for r in rows if r < len(self.records)]

    def show_selected(self):
        sel = self.selected_records()
        self.show_record(sel[0] if sel else None)

    def show_record(self, r: dict | None):
        self._image = None
        if r is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Görüntü yok")
            self.values_view.setHtml("")
            return
        path = r["ann_path"] if self.annotated_check.isChecked() and r["ann_path"] else r["img_path"]
        if not path and r["ann_path"]:
            path = r["ann_path"]
        img = trace.read_image(path) if path else None
        self._image = img
        self._show_image()
        if img is None:
            self.image_label.setText("Görüntü yok" if not path else "Görüntü dosyası bulunamadı")
        verdict = ("—" if r["ok"] is None else
                   f"<b style='color:{GOOD if r['ok'] else BAD}'>{'SAĞLAM' if r['ok'] else 'HATALI'}</b>")
        rows = "".join(f"<tr><td><b>{_esc(k)}</b></td><td>{_esc(_fmt(v))}</td></tr>" for k, v in r["values"].items())
        when = f"{r['time']:%d.%m.%Y %H:%M:%S.%f}"[:-3]
        self.values_view.setHtml(
            f"<p>{verdict} · {when}</p>"
            f"<p>Seri no: <b>{_esc(r['serial']) or '–'}</b> · Lot: {_esc(r['lot']) or '–'}<br>"
            f"İstasyon: {_esc(r['station']) or '–'} · Akış: {_esc(r['flow']) or '–'} · "
            f"Reçete: {_esc(r['recipe']) or '–'}<br>Sebep: {_esc(r['reason']) or '–'}<br>"
            f"Kayıt no: {_esc(r['rid'])}</p>"
            f"<h4>Ölçülen değerler</h4><table cellpadding='3'>{rows or '<tr><td>Değer yok</td></tr>'}</table>")

    def _show_image(self):
        if self._image is None:
            self.image_label.setPixmap(QPixmap())
            return
        pm = QPixmap.fromImage(to_qimage(self._image))
        self.image_label.setPixmap(pm.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._show_image()

    # --- export -------------------------------------------------------------------------------------------------
    def export_to(self, path, selected: bool) -> str:
        if selected:
            recs = self.selected_records()
            if not recs:
                raise ValueError("Önce tablodan kayıt seçin")
            recs = self.archive.by_ids([r["id"] for r in recs])
            return self.archive.export(path, records=recs)
        return self.archive.export(path, flt=self.current_filter())

    def ask_export(self, selected: bool):
        name = f"arsiv_{dt.datetime.now():%Y%m%d_%H%M}"
        path, filt = QFileDialog.getSaveFileName(
            self, "Dışa aktar", str(pathlib.Path.home() / f"{name}.csv"),
            "CSV (*.csv);;Excel (*.xlsx);;ZIP, görüntülerle (*.zip)")
        if not path:
            return
        if not pathlib.Path(path).suffix:
            path += ".xlsx" if "xlsx" in filt else ".zip" if "zip" in filt else ".csv"
        try:
            msg = self.export_to(path, selected)
        except ValueError as exc:
            QMessageBox.warning(self, "Dışa aktarılamadı", str(exc))
            return
        QMessageBox.information(self, "Dışa aktarıldı", msg)


def _esc(text) -> str:
    import html
    return html.escape(str(text))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def open_browser(parent=None, root=None) -> TraceBrowser:
    dlg = TraceBrowser(parent, root)
    dlg.show()
    return dlg
