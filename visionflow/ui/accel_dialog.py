"""Araçlar → Hızlandırma (GPU)…: detected inference devices, the default device and a speed test."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .. import accel

PC = "Bilgisayar ayarı"          # flow choice: use this computer's device
_ALIVE: set = set()              # speed tests still running after their dialog closed (kept until they end)


class _Bench(QThread):
    done = Signal(object)          # list of rows, or an error text

    def run(self):
        try:
            self.done.emit(accel.benchmark())
        except Exception as exc:  # noqa: BLE001
            self.done.emit(str(exc))


class AccelDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Hızlandırma (GPU)")
        self.resize(700, 420)
        self._bench: _Bench | None = None
        lay = QVBoxLayout(self)
        hint = QLabel("Yapay zeka blokları (YOLO, sınıflandırma, PatchCore) modeli seçilen cihazda çalıştırır. Cihaz "
                      "yoksa veya açılamazsa CPU ile devam edilir ve blokta 'CPU (yedek)' yazar. Blokların 'Cihaz' "
                      "ayarı bu seçimi tek blok için değiştirir. Değişiklik akış yeniden başlatılınca geçerli olur.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        lay.addWidget(hint)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Cihaz", "Durum", "Hız testi", "Not"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)
        form = QFormLayout()
        self.pc_combo = QComboBox()
        self.pc_combo.addItems(accel.DEVICES)
        self.flow_combo = QComboBox()
        self.flow_combo.addItems([PC] + accel.DEVICES)
        self.fp16 = QCheckBox("FP16 (yarı hassasiyet; TensorRT / OpenVINO'da daha hızlı, sonuç çok az değişebilir)")
        form.addRow("Bu bilgisayarda varsayılan:", self.pc_combo)
        form.addRow("Bu akış için:", self.flow_combo)
        form.addRow("", self.fp16)
        lay.addLayout(form)
        self.bench_btn = QPushButton("⏱ Hız testi")
        self.bench_btn.clicked.connect(lambda: self.run_benchmark())
        lay.addWidget(self.bench_btn)
        self.status = QLabel(" ")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        lay.addWidget(bb)
        self._filling = False
        self.pc_combo.currentTextChanged.connect(self._pc_changed)
        self.flow_combo.currentTextChanged.connect(self._flow_changed)
        self.fp16.toggled.connect(self._flow_changed)
        self.reload()

    @property
    def graph(self):
        return self.win.root

    def reload(self):
        """Fill from the current flow and this computer's settings (again after undo / opening another flow)."""
        self._filling = True
        try:
            self.pc_combo.setCurrentText(accel.pc_settings().get("device", accel.AUTO))
            acc = self.graph.accel or {}
            self.flow_combo.setCurrentText(acc.get("device") or PC)
            self.fp16.setChecked(bool(acc.get("fp16", False)))
            self._fill_devices()
        finally:
            self._filling = False

    def _fill_devices(self, bench: dict | None = None):
        devs = accel.available_devices()
        self.table.setRowCount(len(devs))
        for r, d in enumerate(devs):
            speed = ""
            if bench and d["name"] in bench:
                b = bench[d["name"]]
                speed = f"HATA: {b['error']}" if b["error"] else f"{b['ms']:.2f} ms ({b['used']})"
            cells = [d["name"], "✔ var" if d["available"] else "— yok", speed, d["note"]]
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        auto = accel.auto_device()
        self.status.setText(f"{accel.status_line()} · Otomatik seçim bu bilgisayarda: {auto}")

    def _pc_changed(self, text):
        if self._filling:
            return
        try:
            accel.save_pc_settings(device=text)
            self.status.setText(f"Bu bilgisayarın varsayılanı: {text}")
        except (OSError, ValueError) as exc:
            self.status.setText(f"Kaydedilemedi: {exc}")

    def _flow_changed(self, *_):
        if self._filling:
            return
        acc = {}
        if self.flow_combo.currentText() != PC:
            acc["device"] = self.flow_combo.currentText()
        if self.fp16.isChecked():
            acc["fp16"] = True
        if acc != (self.graph.accel or {}):
            self.graph.accel = acc
            self.win._schedule_snapshot()
            self.win._title()
            self.status.setText("Akışın cihaz ayarı değişti; akışı kaydedin ve yeniden başlatın")

    def run_benchmark(self, wait: bool = False):
        if self._bench is not None and self._bench.isRunning():
            return
        self.bench_btn.setEnabled(False)
        self.status.setText("Hız testi çalışıyor… (TensorRT ilk seferde birkaç dakika sürebilir)")
        self._bench = b = _Bench()       # no parent: closing the dialog must not destroy a running test
        _ALIVE.add(b)
        b.finished.connect(lambda: _ALIVE.discard(b))
        b.done.connect(self._bench_done)
        b.start()
        if wait:
            self._bench.wait()
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()

    def _bench_done(self, result):
        self.bench_btn.setEnabled(True)
        if isinstance(result, str):
            self.status.setText(f"Hız testi yapılamadı: {result}")
            return
        self.results = result
        self._fill_devices({r["device"]: r for r in result})
        ok = [r for r in result if not r["error"]]
        if ok:
            best = min(ok, key=lambda r: r["ms"])
            self.status.setText(f"En hızlı: {best['device']} ({best['ms']:.2f} ms/kare, küçük test ağı)")
