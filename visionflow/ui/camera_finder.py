"""Araçlar → Endüstriyel kameraları bul: lists the GigE / USB3 Vision cameras this computer sees and adds the chosen
one to the flow as an Endüstriyel Kamera block."""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout)

from ..nodes.machine_vision import BACKENDS, cti_files, list_cameras


class CameraFinder(QDialog):
    found = Signal(object, object)               # (cameras or None, error text)

    def __init__(self, parent=None, on_add=None, running=None):
        super().__init__(parent)
        self.setWindowTitle("Endüstriyel kameraları bul")
        self.resize(640, 380)
        self.on_add = on_add                         # fn(backend, serial): put a camera block into the flow
        self.running = running or (lambda: False)    # is a flow running? (it may be using the GenTL producer)
        self.thread: threading.Thread | None = None
        self.found.connect(self._show)
        self.backend = QComboBox()
        self.backend.addItems(BACKENDS)
        self.btn_find = QPushButton("Ara")
        self.btn_find.clicked.connect(self.find)
        top = QHBoxLayout()
        top.addWidget(QLabel("Kamera türü:"))
        top.addWidget(self.backend, 1)
        top.addWidget(self.btn_find)
        self.list = QTreeWidget()
        self.list.setHeaderLabels(["Üretici", "Model", "Seri no"])
        self.list.setColumnWidth(0, 140)
        self.list.setColumnWidth(1, 260)
        self.list.itemDoubleClicked.connect(lambda *_: self.add())
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.btn_add = QPushButton("Akışa ekle")
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self.add)
        self.list.currentItemChanged.connect(lambda cur, _: self.btn_add.setEnabled(cur is not None))
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.info)
        lay.addWidget(self.btn_add, alignment=Qt.AlignRight)

    def find(self):
        """Search in the background: GigE discovery can take seconds and must not freeze the window."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.list.clear()
        self.btn_add.setEnabled(False)
        backend = self.backend.currentText()
        if not backend.startswith("Basler") and self.running():
            # a second Harvester's reset() can unload the .cti the running flow is grabbing through
            self.info.setText("Akış çalışırken GenICam kameralar aranamaz. Önce akışı durdurun.")
            return
        self.info.setText("Kameralar aranıyor…")
        self.btn_find.setEnabled(False)
        self._backend = backend

        def work():
            try:
                cams, err = list_cameras(backend), None
            except Exception as exc:  # noqa: BLE001
                cams, err = None, str(exc)
            try:
                self.found.emit(cams, err)
            except RuntimeError:                     # the window was closed meanwhile
                pass
        self.thread = threading.Thread(target=work, daemon=True, name="kamera-ara")
        self.thread.start()

    def _show(self, cams, err):
        self.btn_find.setEnabled(True)
        if err is not None:
            self.info.setText(f"Aranamadı: {err}")
            return
        for c in cams:
            self.list.addTopLevelItem(QTreeWidgetItem([c["vendor"], c["model"], c["serial"]]))
        if cams:
            self.list.setCurrentItem(self.list.topLevelItem(0))
            self.info.setText(f"{len(cams)} kamera bulundu. Birini seçip Akışa ekle'ye basın.")
        else:
            hint = ("" if self._backend.startswith("Basler") else
                    f" Kullanılan .cti dosyaları: {', '.join(cti_files()) or 'yok'}.")
            self.info.setText("Kamera bulunamadı. Kablo, güç/PoE ve kameranın IP adresinin bilgisayarla aynı ağda "
                              "olduğunu kontrol edin (üreticinin IP ayar aracı ile)." + hint)

    def add(self):
        item = self.list.currentItem()
        if item is None or self.on_add is None:
            return
        self.on_add(self.backend.currentText(), item.text(2))
        self.info.setText(f"{item.text(1)} ({item.text(2)}) akışa eklendi.")
