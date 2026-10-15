"""Araçlar → Kamera Kalibrasyonu: lens calibration from chessboard photos, saved for the Lens Bozulmasını Düzelt block."""
from __future__ import annotations

import pathlib
import threading

import cv2
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout)

from .. import calibration as cal


class CalibrationDialog(QDialog):
    done_signal = Signal(object, object)          # (Calibration or None, error text)
    step_signal = Signal(str)

    def __init__(self, parent=None, folder: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Kamera Kalibrasyonu")
        self.resize(640, 520)
        self.thread: threading.Thread | None = None
        self.result: cal.Calibration | None = None
        info = QLabel("1) <b>Satranç tahtası kaydet</b> ile tahtayı yazdırın ve düz bir yüzeye yapıştırın. "
                      "2) Kamerayla tahtanın 10–20 fotoğrafını çekin: farklı açılar, resmin köşeleri ve kenarları "
                      "dahil. 3) Klasörü seçip <b>Kalibre et</b>'e basın. Sonuç, <i>Lens Bozulmasını Düzelt</i> "
                      "bloğunun kullandığı dosyaya kaydedilir.")
        info.setWordWrap(True)
        self.folder = QLineEdit(folder)
        browse = QPushButton("Seç…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        self.cols, self.rows = QSpinBox(), QSpinBox()
        for s, v in ((self.cols, 9), (self.rows, 6)):
            s.setRange(2, 50)
            s.setValue(v)
        self.square = QDoubleSpinBox()
        self.square.setRange(0.1, 1000)
        self.square.setValue(25.0)
        self.square.setSuffix(" mm")
        self.out = QLineEdit(str(cal.default_path()))
        form = QFormLayout()
        form.addRow("Fotoğraf klasörü:", row)
        form.addRow("İç köşe (yatay):", self.cols)
        form.addRow("İç köşe (dikey):", self.rows)
        form.addRow("Kare kenarı:", self.square)
        form.addRow("Kaydedilecek dosya:", self.out)
        self.btn_board = QPushButton("Satranç tahtası kaydet…")
        self.btn_board.clicked.connect(self.save_board)
        self.btn_run = QPushButton("Kalibre et")
        self.btn_run.clicked.connect(self.run)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_board)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_run)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addLayout(form)
        lay.addLayout(buttons)
        lay.addWidget(self.report, 1)
        self.done_signal.connect(self._finished)
        self.step_signal.connect(self.report.setPlainText)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Satranç tahtası fotoğrafları", self.folder.text())
        if d:
            self.folder.setText(d)

    def save_board(self, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Satranç tahtası", "satranc_tahtasi.png", "PNG (*.png)")
            if not path:
                return None
        img = cal.chessboard(self.cols.value(), self.rows.value(), 120, 80)
        ok, buf = cv2.imencode(".png", img)
        if ok:
            buf.tofile(path)
        self.report.setPlainText(f"Kaydedildi: {path}\nYazdırırken 'sayfaya sığdır' kullanmayın; yazdırdıktan "
                                 "sonra bir karenin kenarını cetvelle ölçüp 'Kare kenarı'na yazın.")
        return path

    def run(self):
        if self.thread is not None and self.thread.is_alive():
            return
        folder, out = self.folder.text().strip(), self.out.text().strip()
        cols, rows, sq = self.cols.value(), self.rows.value(), self.square.value()
        self.btn_run.setEnabled(False)
        self.report.setPlainText("Fotoğraflar okunuyor…")

        def work():
            try:
                images = cal.read_folder(folder)
                if not images:
                    raise ValueError(f"Klasörde fotoğraf yok: {folder}")
                c = cal.calibrate(images, cols, rows, sq, on_step=self._step)
                c.save(out)
                res, err = c, None
            except Exception as exc:  # noqa: BLE001
                res, err = None, str(exc)
            try:
                self.done_signal.emit(res, err)
            except RuntimeError:                  # window closed meanwhile
                pass
        self.thread = threading.Thread(target=work, daemon=True, name="kalibrasyon")
        self.thread.start()

    def _step(self, text):
        try:
            self.step_signal.emit(text)
        except RuntimeError:
            pass

    def _finished(self, c, err):
        self.btn_run.setEnabled(True)
        if err:
            self.report.setPlainText(f"Olmadı: {err}")
            return
        self.result = c
        quality = "çok iyi" if c.rms < 0.5 else "kabul edilebilir" if c.rms < 1.0 else "zayıf: daha net fotoğraflar çekin"
        worst = sorted(c.per_image.items(), key=lambda kv: -kv[1])[:5]
        lines = [f"Kalibrasyon tamam: {c.images_used} fotoğraf kullanıldı.",
                 f"Ortalama hata: {c.rms:.3f} piksel ({quality}).",
                 f"Kaydedildi: {pathlib.Path(self.out.text().strip())}", "",
                 "En çok hata veren fotoğraflar (1 pikselin üstündekileri silip tekrar deneyebilirsiniz):"]
        lines += [f"  {name}: {e:.3f} px" for name, e in worst]
        if c.skipped:
            lines += ["", "Kullanılmayan (tahta bulunamadı veya farklı çözünürlük): " + ", ".join(c.skipped[:20])]
        self.report.setPlainText("\n".join(lines))
