"""Örnekleri Öğret: go through the images the program was unsure about (Emin Olmadıklarını Topla) and put each one
into the right class folder with one key press. Hızlı Sınıflandırıcı notices the new samples and retrains itself."""
from __future__ import annotations

import pathlib
import shutil
import uuid

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QDialog, QFileDialog, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from ..core.node import resolve_path
from ..nodes.learning import REVIEW_DIR, _imread, _images_in, class_folders, guess_from_name
from .worker import shrink, to_qimage


WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class TeachDialog(QDialog):
    def __init__(self, parent=None, folder: str = "veri"):
        super().__init__(parent)
        self.setWindowTitle("Örnekleri Öğret")
        self.resize(760, 640)
        self.files: list[pathlib.Path] = []
        self.moved: list[tuple[pathlib.Path, pathlib.Path]] = []      # (from, to) for "Geri al"
        self.trash_name = f"_silinen_{uuid.uuid4().hex[:8]}"     # this window's own: closing another keeps it
        self._pix: tuple[pathlib.Path, QPixmap] | None = None

        self.folder_edit = QLineEdit(folder)
        self.folder_edit.editingFinished.connect(self.reload)
        browse = QPushButton("Seç…")
        browse.clicked.connect(self._browse)
        top = QHBoxLayout()
        top.addWidget(QLabel("Veri seti klasörü:"))
        top.addWidget(self.folder_edit, 1)
        top.addWidget(browse)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.image = QLabel(alignment=Qt.AlignCenter)
        self.image.setMinimumSize(480, 360)
        self.image.setStyleSheet("background:#111; border-radius:6px")
        self.guess = QLabel(alignment=Qt.AlignCenter)
        self.buttons = QWidget()
        self.grid = QGridLayout(self.buttons)
        self.grid.setContentsMargins(0, 0, 0, 0)

        self.btn_new = QPushButton("Yeni sınıf…")
        self.btn_new.clicked.connect(self.new_class)
        self.btn_skip = QPushButton("Atla (Boşluk)")
        self.btn_skip.clicked.connect(self.skip)
        self.btn_delete = QPushButton("Sil (Del)")
        self.btn_delete.clicked.connect(self.delete)
        self.btn_undo = QPushButton("Geri al (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        row = QHBoxLayout()
        for b in (self.btn_new, self.btn_skip, self.btn_delete, self.btn_undo):
            row.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.info)
        lay.addWidget(self.image, 1)
        lay.addWidget(self.guess)
        lay.addWidget(self.buttons)
        lay.addLayout(row)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.skip)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        for i in range(9):
            QShortcut(QKeySequence(str(i + 1)), self, lambda i=i: self._pick_index(i))
        for b in self.findChildren(QPushButton):     # Enter in the folder box must not "click" a button
            b.setAutoDefault(False)
            b.setDefault(False)
        self.reload()

    # --- folders ------------------------------------------------------------------------------------------------
    @property
    def root(self) -> pathlib.Path:
        return pathlib.Path(resolve_path(self.folder_edit.text().strip() or "veri"))

    @property
    def review(self) -> pathlib.Path:
        return self.root / REVIEW_DIR

    def classes(self) -> list[str]:
        names = [d.name for d in class_folders(self.root)]
        # empty class folders made with "Yeni sınıf" count too
        if self.root.is_dir():
            names += [d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))
                      and d.name not in names]
        return sorted(names)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Veri seti klasörü", str(self.root))
        if d:
            self.folder_edit.setText(d)
            self.reload()

    def reload(self):
        self.files = _images_in(self.review)
        self._rebuild_buttons()
        self.show_current()

    def _rebuild_buttons(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for i, name in enumerate(self.classes()):
            b = QPushButton(f"{i + 1}  {name}" if i < 9 else name)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, n=name: self.put(n))
            self.grid.addWidget(b, i // 4, i % 4)

    # --- current image ------------------------------------------------------------------------------------------
    def current(self) -> pathlib.Path | None:
        while self.files and not self.files[0].exists():           # removed by hand meanwhile
            self.files.pop(0)
        return self.files[0] if self.files else None

    def show_current(self):
        cur = self.current()
        for b in (self.btn_skip, self.btn_delete):
            b.setEnabled(cur is not None)
        self.btn_undo.setEnabled(bool(self.moved))
        if cur is None:
            self.image.setPixmap(QPixmap())
            self.image.setText("İncelenecek örnek yok.\n\nAkışa 'Emin Olmadıklarını Topla' bloğunu ekleyin; program "
                               "emin olmadığı görüntüleri buraya toplar.")
            self.guess.setText("")
            self.info.setText(f"İnceleme klasörü: {self.review}")
            return
        self.info.setText(f"{len(self.files)} örnek bekliyor. Doğru sınıfın düğmesine (veya 1–9 tuşuna) basın; "
                          "Hızlı Sınıflandırıcı yeni örnekle kendini yeniden eğitir.")
        if self._pix is None or self._pix[0] != cur:              # decode once, not on every resize
            img = _imread(cur)
            self._pix = (cur, QPixmap.fromImage(to_qimage(shrink(img, 900))) if img is not None else None)
        pix = self._pix[1]
        if pix is None:
            self.image.setText(f"Okunamadı: {cur.name}")
        else:
            self.image.setPixmap(pix.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label, score = guess_from_name(cur.name)
        self.guess.setText(f"Programın tahmini: <b>{label}</b> (güven {score:.2f})" if label and score is not None
                           else "")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.current() is not None:
            self.show_current()

    # --- actions ------------------------------------------------------------------------------------------------
    def _pick_index(self, i: int):
        names = self.classes()
        if i < len(names):
            self.put(names[i])

    def put(self, name: str):
        cur = self.current()
        if cur is None:
            return
        dst_dir = self.root / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / cur.name
        n = 1
        while dst.exists():
            dst = dst_dir / f"{cur.stem}_{n}{cur.suffix}"
            n += 1
        try:
            shutil.move(str(cur), str(dst))
        except OSError as exc:
            QMessageBox.warning(self, "Taşınamadı", str(exc))
            return
        self.moved.append((cur, dst))
        self.files.pop(0)
        self.show_current()

    def skip(self):
        if self.current() is not None:
            self.files.append(self.files.pop(0))
            self.show_current()

    def delete(self):
        cur = self.current()
        if cur is None:
            return
        trash = self.review / self.trash_name           # kept while "Geri al" is possible, removed on close
        trash.mkdir(parents=True, exist_ok=True)
        dst = trash / cur.name
        try:
            shutil.move(str(cur), str(dst))
        except OSError as exc:
            QMessageBox.warning(self, "Silinemedi", str(exc))
            return
        self.moved.append((cur, dst))
        self.files.pop(0)
        self.show_current()

    def undo(self):
        if not self.moved:
            return
        src, dst = self.moved.pop()
        try:
            shutil.move(str(dst), str(src))
        except OSError as exc:
            QMessageBox.warning(self, "Geri alınamadı", str(exc))
            return
        self.files.insert(0, src)
        self.show_current()

    def new_class(self):
        name, ok = QInputDialog.getText(self, "Yeni sınıf", "Sınıf adı (ör. cizik, leke):")
        name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or "")).strip()
        if not ok or not name:
            return
        if name.startswith(("_", ".")) or name.upper().split(".")[0] in WINDOWS_RESERVED:
            QMessageBox.warning(self, "Yeni sınıf", f"'{name}' sınıf adı olarak kullanılamaz.")
            return
        try:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Yeni sınıf", f"Klasör oluşturulamadı: {exc}")
            return
        self._rebuild_buttons()
        self.put(name)

    def done(self, r):
        shutil.rmtree(self.review / self.trash_name, ignore_errors=True)
        super().done(r)

    def closeEvent(self, ev):
        shutil.rmtree(self.review / self.trash_name, ignore_errors=True)
        super().closeEvent(ev)
