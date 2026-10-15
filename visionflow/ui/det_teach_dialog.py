"""Kutu Çiz ve Eğit (Az Örnekle Nesne Bul): draw boxes around the objects on a few pictures (one class per box), press
Eğit, look at what the detector finds on each picture, add boxes it missed (or accept its finds) and train again; then
save the model for the block. Picture loading and drawing helpers are shared with the segmentation window."""
from __future__ import annotations

import pathlib

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from .. import fewshot_det as fd
from ..core.node import resolve_path, save_target
from .seg_teach_dialog import IMAGE_EXTS, _fit, _qimage, _read

MAX_UNDO = 60
HANDLE = 7                 # corner handle size on screen (px)


def _qcolor(bgr) -> QColor:
    return QColor(int(bgr[2]), int(bgr[1]), int(bgr[0]))


class BoxCanvas(QWidget):
    """Shows a picture with its boxes and the detector's finds. Drag on empty space draws a box, drag inside a box
    moves it, drag a corner of the selected box resizes it, right click (or Delete) removes it, double click on a
    find turns it into a box. Everything is reported in picture pixels."""
    box_drawn = Signal(int, int, int, int)
    box_changed = Signal(int, int, int, int, int)      # index, x, y, w, h
    box_selected = Signal(int)
    delete_requested = Signal(int)
    detection_accepted = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pix: QPixmap | None = None
        self.img_size = (1, 1)
        self.boxes: list[list] = []                    # [x, y, w, h, cls] (shown; owned by the dialog)
        self.dets: list[dict] = []
        self.colors: list = []
        self.names: list[str] = []
        self.selected = -1
        self._drag: str | None = None                  # "new" | "move" | "resize"
        self._start: QPointF | None = None
        self._cur: QPointF | None = None
        self._orig: list | None = None
        self._corner = 0

    def set_content(self, pix, size, boxes, dets, colors, names, selected=-1):
        self.pix, self.img_size = pix, size
        self.boxes, self.dets, self.colors, self.names = boxes, dets, colors, names
        self.selected = selected if 0 <= selected < len(boxes) else -1
        self.update()

    # geometry
    def _target(self) -> QRectF:
        w, h = self.img_size
        s = min(self.width() / max(1, w), self.height() / max(1, h))
        return QRectF((self.width() - w * s) / 2, (self.height() - h * s) / 2, w * s, h * s)

    def _scale(self) -> float:
        return self._target().width() / max(1, self.img_size[0])

    def to_image(self, pos) -> tuple[float, float]:
        r = self._target()
        s = self._scale()
        return (pos.x() - r.x()) / s, (pos.y() - r.y()) / s

    def to_screen(self, x, y) -> QPointF:
        r = self._target()
        s = self._scale()
        return QPointF(r.x() + x * s, r.y() + y * s)

    def _box_rect(self, b) -> QRectF:
        return QRectF(self.to_screen(b[0], b[1]), self.to_screen(b[0] + b[2], b[1] + b[3]))

    def _corner_at(self, pos) -> int:
        """0 = none, 1..4 = top-left, top-right, bottom-right, bottom-left of the selected box."""
        if not 0 <= self.selected < len(self.boxes):
            return 0
        r = self._box_rect(self.boxes[self.selected])
        for i, c in enumerate((r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft())):
            if abs(c.x() - pos.x()) <= HANDLE and abs(c.y() - pos.y()) <= HANDLE:
                return i + 1
        return 0

    def box_at(self, pos) -> int:
        """The smallest box under a screen position, -1 when none."""
        x, y = self.to_image(pos)
        hits = [(b[2] * b[3], i) for i, b in enumerate(self.boxes)
                if b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]]
        return min(hits)[1] if hits else -1

    def det_at(self, pos) -> int:
        x, y = self.to_image(pos)
        hits = [(d["box"][2] * d["box"][3], i) for i, d in enumerate(self.dets)
                if d["box"][0] <= x <= d["box"][0] + d["box"][2] and d["box"][1] <= y <= d["box"][1] + d["box"][3]]
        return min(hits)[1] if hits else -1

    def _live(self, i):
        """Box i as it is during a drag."""
        b = list(self.boxes[i])
        if self._drag in ("move", "resize") and i == self.selected and self._orig is not None and self._cur:
            x0, y0 = self.to_image(self._start)
            x1, y1 = self.to_image(self._cur)
            dx, dy = x1 - x0, y1 - y0
            ox, oy, ow, oh = self._orig[:4]
            if self._drag == "move":
                b[0], b[1] = ox + dx, oy + dy
            else:
                left, top, right, bottom = ox, oy, ox + ow, oy + oh
                if self._corner in (1, 4):
                    left += dx
                if self._corner in (2, 3):
                    right += dx
                if self._corner in (1, 2):
                    top += dy
                if self._corner in (3, 4):
                    bottom += dy
                b[:4] = [min(left, right), min(top, bottom), abs(right - left), abs(bottom - top)]
        return b

    # painting
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        if self.pix is None:
            p.setPen(QColor("#aaa"))
            p.drawText(self.rect(), Qt.AlignCenter, "Resim ekleyin: 'Klasör…', 'Dosya…' veya 'Akıştan al'")
            return
        r = self._target()
        p.drawPixmap(r, self.pix, QRectF(self.pix.rect()))
        for d in self.dets:                             # the detector's finds: dashed, with the score
            col = _qcolor(self.colors[d["class_id"]] if d["class_id"] < len(self.colors) else (255, 255, 255))
            p.setPen(QPen(col, 1, Qt.DashLine))
            if d.get("poly"):
                pts = [self.to_screen(x, y) for x, y in d["poly"]]
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    p.drawLine(a, b)
            rect = self._box_rect(d["box"])
            p.drawRect(rect)
            p.drawText(rect.bottomLeft() + QPointF(2, 12), f"{d['label']} {d['score']:.2f}")
        for i, _ in enumerate(self.boxes):              # the user's boxes: solid
            b = self._live(i)
            c = int(b[4])
            col = _qcolor(self.colors[c] if c < len(self.colors) else (255, 255, 255))
            rect = self._box_rect(b)
            p.setPen(QPen(col, 3 if i == self.selected else 2))
            p.drawRect(rect)
            name = self.names[c] if c < len(self.names) else str(c)
            p.fillRect(QRectF(rect.x(), rect.y() - 16, 8 * len(name) + 8, 16), col)
            p.setPen(QColor("#000"))
            p.drawText(QPointF(rect.x() + 3, rect.y() - 4), name)
            if i == self.selected:
                p.setPen(QPen(QColor("#fff"), 1))
                for c2 in (rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()):
                    p.drawRect(QRectF(c2.x() - HANDLE / 2, c2.y() - HANDLE / 2, HANDLE, HANDLE))
        if self._drag == "new" and self._start is not None and self._cur is not None:
            p.setPen(QPen(QColor("#ffeb3b"), 2, Qt.DashLine))
            p.drawRect(QRectF(self._start, self._cur).normalized())

    # mouse
    def mousePressEvent(self, e):
        if self.pix is None:
            return
        pos = e.position()
        if e.button() == Qt.RightButton:
            i = self.box_at(pos)
            if i >= 0:
                self.delete_requested.emit(i)
            return
        if e.button() != Qt.LeftButton:
            return
        self._start = self._cur = pos
        corner = self._corner_at(pos)
        i = self.box_at(pos)
        if corner:
            self._drag, self._corner = "resize", corner
        elif i >= 0:
            self.selected = i
            self.box_selected.emit(i)
            self._drag = "move"
        else:
            self._drag = "new"
        self._orig = list(self.boxes[self.selected]) if self._drag != "new" else None
        self.update()

    def mouseMoveEvent(self, e):
        if self._drag:
            self._cur = e.position()
            self.update()
        elif self.pix is not None:
            self.setCursor(Qt.SizeFDiagCursor if self._corner_at(e.position())
                           else Qt.SizeAllCursor if self.box_at(e.position()) >= 0 else Qt.CrossCursor)

    def mouseReleaseEvent(self, e):
        if not self._drag or e.button() != Qt.LeftButton:
            return
        self._cur = e.position()
        kind, self._drag = self._drag, None
        if kind == "new":
            (x0, y0), (x1, y1) = self.to_image(self._start), self.to_image(self._cur)
            self._start = self._cur = None
            self.box_drawn.emit(int(round(min(x0, x1))), int(round(min(y0, y1))), int(round(abs(x1 - x0))),
                                int(round(abs(y1 - y0))))
        elif 0 <= self.selected < len(self.boxes):
            self._drag = kind
            b = self._live(self.selected)
            self._drag = None
            moved = [int(round(v)) for v in b[:4]] != [int(round(v)) for v in self._orig[:4]]
            self._start = self._cur = self._orig = None
            if moved:
                self.box_changed.emit(self.selected, *(int(round(v)) for v in b[:4]))
        self.update()

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and self.box_at(e.position()) < 0:
            i = self.det_at(e.position())
            if i >= 0:
                self.detection_accepted.emit(i)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected >= 0:
            self.delete_requested.emit(self.selected)
        else:
            super().keyPressEvent(e)


class DetTeachDialog(QDialog):
    """The box drawing / training window. ``node`` is the Az Örnekle Nesne Bul block (or None), ``apply(name,
    value)`` sets one of its settings (used to point it at the saved model)."""

    def __init__(self, parent=None, node=None, apply=None):
        super().__init__(parent)
        self.node, self.apply = node, apply
        self.setWindowTitle("Az Örnekle Nesne Bul — Kutu Çiz ve Eğit")
        self.resize(1150, 740)
        self.images: list[dict] = []            # {"name", "img" (BGR8), "boxes": [[x, y, w, h, cls], …]}
        self.current = -1
        self.selected = -1
        self.classes: list[list] = [["nesne", fd.default_color(0)]]
        self.active_class = 0
        self.model: fd.DetModel | None = None
        self._dets: dict[int, list] = {}        # image index → detections of the current model
        self._undo: list[tuple[int, list]] = []
        self._build()
        self._load_existing_model()
        self._refresh_classes()
        self.render()

    # --- layout ---------------------------------------------------------------------------------------------------
    def _build(self):
        self.canvas = BoxCanvas()
        self.canvas.box_drawn.connect(self._drawn)
        self.canvas.box_changed.connect(lambda i, x, y, w, h: self._ui(lambda: self.set_box(i, x, y, w, h)))
        self.canvas.box_selected.connect(self._select)
        self.canvas.delete_requested.connect(lambda i: self._ui(lambda: self.delete_box(i)))
        self.canvas.detection_accepted.connect(lambda i: self._ui(lambda: self.accept_detections([i])))

        def button(text, slot, tip=None):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False: slot())
            if tip:
                b.setToolTip(tip)
            return b

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.set_current)
        img_btns = QHBoxLayout()
        for b in (button("Klasör…", self._add_folder_ui), button("Dosya…", self._add_files_ui),
                  button("Akıştan al", self.add_from_flow, "Bloğun son çalıştırmada gördüğü resmi ekler"),
                  button("Kaldır", self.remove_current)):
            img_btns.addWidget(b)

        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._class_picked)
        cls_btns = QHBoxLayout()
        for b in (button("Sınıf ekle", self._add_class_ui), button("Yeniden adlandır", self._rename_class_ui),
                  button("Kutuyu bu sınıfa al", lambda: self._ui(self.set_selected_class),
                         "Seçili kutunun sınıfını seçili sınıf yapar")):
            cls_btns.addWidget(b)

        self.btn_undo = button("Geri al (Ctrl+Z)", self.undo)
        edit_row = QHBoxLayout()
        for b in (self.btn_undo, button("Kutuyu sil (Del)", lambda: self._ui(lambda: self.delete_box(self.selected))),
                  button("Hepsini sil", self.clear_current)):
            edit_row.addWidget(b)
        self.btn_accept = button("Bulunanları kutuya çevir", lambda: self._ui(self.accept_detections),
                                 "Modelin bu resimde bulduklarını kutu olarak ekler (çakışanlar hariç); sonra "
                                 "yalnızca hataları düzeltin. Tek bir bulunanı çift tıklayarak da ekleyebilirsiniz.")

        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(0, 180)
        self.angle_spin.setValue(180)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setToolTip("Parçalar en çok kaç derece dönmüş görünebilir (180 = her yöne). Küçük değer "
                                   "daha hızlı ve daha az yanlış alarm demektir.")
        self.scale_lo = QDoubleSpinBox()
        self.scale_lo.setRange(0.3, 1.0)
        self.scale_lo.setSingleStep(0.05)
        self.scale_lo.setValue(0.8)
        self.scale_hi = QDoubleSpinBox()
        self.scale_hi.setRange(1.0, 3.0)
        self.scale_hi.setSingleStep(0.05)
        self.scale_hi.setValue(1.2)
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Dönme ±"))
        opt_row.addWidget(self.angle_spin)
        opt_row.addWidget(QLabel("Ölçek"))
        opt_row.addWidget(self.scale_lo)
        opt_row.addWidget(QLabel("–"))
        opt_row.addWidget(self.scale_hi)

        self.btn_train = button("Eğit", self.train_ui)
        self.btn_train.setStyleSheet("font-weight:bold; padding:6px")
        self.show_pred = QCheckBox("Bulunanları göster")
        self.show_pred.setChecked(True)
        self.show_pred.toggled.connect(lambda _: self.render())
        self.score_slider = QSlider(Qt.Horizontal)
        self.score_slider.setRange(5, 99)
        self.score_slider.setValue(50)
        self.score_label = QLabel("0.50")
        self.score_slider.valueChanged.connect(self._score_changed)
        pred_row = QHBoxLayout()
        pred_row.addWidget(self.show_pred)
        pred_row.addWidget(QLabel("Eşik:"))
        pred_row.addWidget(self.score_slider, 1)
        pred_row.addWidget(self.score_label)

        self.path_edit = QLineEdit(self._default_model_path())
        save_row = QHBoxLayout()
        save_row.addWidget(self.path_edit, 1)
        save_row.addWidget(button("…", self._browse_save))
        self.btn_save = button("Modeli kaydet ve bloğa uygula", self.save_ui)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setObjectName("hint")

        side = QVBoxLayout()
        side.addWidget(QLabel("<b>Resimler</b>"))
        side.addWidget(self.image_list, 2)
        side.addLayout(img_btns)
        side.addWidget(QLabel("<b>Sınıflar</b> (çizilecek kutunun sınıfı, 1–9 tuşları)"))
        side.addWidget(self.class_list, 1)
        side.addLayout(cls_btns)
        box_help = QLabel("<b>Kutular</b> — boş yerde sürükleyin: yeni kutu; kutuyu sürükleyin: taşı; "
                          "köşeden: boyut; sağ tık / Del: sil")
        box_help.setWordWrap(True)
        side.addWidget(box_help)
        side.addLayout(edit_row)
        side.addWidget(self.btn_accept)
        side.addWidget(QLabel("<b>Eğitim</b>"))
        side.addLayout(opt_row)
        side.addWidget(self.btn_train)
        side.addLayout(pred_row)
        side.addWidget(QLabel("<b>Model dosyası</b>"))
        side.addLayout(save_row)
        side.addWidget(self.btn_save)
        side.addWidget(self.status)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(370)

        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(side_w)

        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("T"), self, self.train_ui)
        for i in range(9):
            QShortcut(QKeySequence(str(i + 1)), self, lambda i=i: self.set_class(i))
        self.status.setText("1) Resim ekleyin  2) Bulunacak her nesnenin etrafına kutu çizin (resimdeki hepsine)  "
                            "3) Eğit  4) Kaçırılanlara kutu ekleyip yeniden eğitin  5) Kaydedin.")

    def _ui(self, fn):
        """Run an edit from a mouse/button action; a Turkish error goes to the status line."""
        try:
            return fn()
        except ValueError as exc:
            self.status.setText(f"<span style='color:#ef5350'>{exc}</span>")
            return None

    def _default_model_path(self) -> str:
        cur = str(self.node.values.get("model", "")) if self.node is not None else ""
        return cur or "modeller/nesne_bulucu.vfdet"

    def _load_existing_model(self):
        """The block's current model: its classes, colours and settings, so a retrain keeps the same names."""
        path = resolve_path(self.path_edit.text().strip())
        if not path or not pathlib.Path(path).is_file():
            return
        try:
            m = fd.DetModel.load(path)
        except ValueError:
            return
        self.model = m
        self.classes = [[n, tuple(c)] for n, c in zip(m.classes, m.colors)]
        self.angle_spin.setValue(int(round(m.angle_range)))
        self.scale_lo.setValue(m.scale_range[0])
        self.scale_hi.setValue(m.scale_range[1])

    # --- pictures -------------------------------------------------------------------------------------------------
    def add_image(self, img, name: str = "resim") -> int:
        """Add a picture (any depth / channels); returns its index. Raises ValueError for a non-picture."""
        bgr = _fit(img)
        self.images.append({"name": name, "img": bgr, "boxes": []})
        self.image_list.addItem(name)
        self.image_list.setCurrentRow(len(self.images) - 1)       # → set_current
        return len(self.images) - 1

    def add_files(self, paths) -> int:
        n, bad = 0, []
        for p in paths:
            img = _read(p)
            if img is None:
                bad.append(pathlib.Path(p).name)
                continue
            try:
                self.add_image(img, pathlib.Path(p).name)
                n += 1
            except ValueError:
                bad.append(pathlib.Path(p).name)
        if bad:
            self.status.setText(f"Okunamayan dosyalar: {', '.join(bad[:5])}" + (" …" if len(bad) > 5 else ""))
        return n

    def add_folder(self, folder) -> int:
        folder = pathlib.Path(resolve_path(str(folder)))
        if not folder.is_dir():
            raise ValueError(f"Klasör yok: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"Klasörde resim yok: {folder}")
        return self.add_files(files[:200])

    def add_from_flow(self) -> bool:
        from ..nodes.fewshot import last_input
        img = last_input(self.node.id) if self.node is not None else None
        if img is None:
            self.status.setText("Akıştan resim yok: akışı bir kez çalıştırın, blok gördüğü son resmi buraya verir.")
            return False
        self.add_image(img, f"akıştan {sum(1 for d in self.images if d['name'].startswith('akıştan')) + 1}")
        return True

    def _add_folder_ui(self):
        d = QFileDialog.getExistingDirectory(self, "Resim klasörü")
        if d:
            try:
                self.add_folder(d)
            except ValueError as exc:
                QMessageBox.warning(self, "Resim ekle", str(exc))

    def _add_files_ui(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Resim seç", "", "Resimler (*.png *.jpg *.jpeg *.bmp *.tif "
                                                                         "*.tiff *.webp)")
        if files:
            self.add_files(files)

    def remove_current(self):
        if self.current < 0:
            return
        i = self.current
        self.images.pop(i)
        self._dets = {(k if k < i else k - 1): v for k, v in self._dets.items() if k != i}
        self._undo = [(k if k < i else k - 1, v) for k, v in self._undo if k != i]
        self.image_list.blockSignals(True)
        self.image_list.takeItem(i)
        self.image_list.blockSignals(False)
        self.current = -1
        self.set_current(min(i, len(self.images) - 1))

    def set_current(self, i: int):
        self.selected = -1
        if i < 0 or i >= len(self.images):
            self.current = -1
            self.render()
            return
        self.current = i
        if self.image_list.currentRow() != i:
            self.image_list.blockSignals(True)
            self.image_list.setCurrentRow(i)
            self.image_list.blockSignals(False)
        self.render()

    def _refresh_image_names(self):
        for i, d in enumerate(self.images):
            it = self.image_list.item(i)
            if it is not None:
                it.setText(f"{d['name']}  ({len(d['boxes'])} kutu)" if d["boxes"] else d["name"])

    # --- classes --------------------------------------------------------------------------------------------------
    def _refresh_classes(self):
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for i, (name, col) in enumerate(self.classes):
            pm = QPixmap(14, 14)
            pm.fill(_qcolor(col))
            self.class_list.addItem(QListWidgetItem(QIcon(pm), f"{i + 1}  {name}"))
        self.class_list.setCurrentRow(min(self.active_class, len(self.classes) - 1))
        self.class_list.blockSignals(False)
        self.render()

    def _class_picked(self, i):
        if 0 <= i < len(self.classes):
            self.active_class = i

    def set_class(self, i: int):
        if 0 <= i < len(self.classes):
            self.active_class = i
            self.class_list.setCurrentRow(i)

    def _clean_name(self, name: str, skip: int = -1) -> str:
        name = " ".join(str(name or "").split())
        if not name:
            raise ValueError("Sınıf adı boş olamaz")
        if "," in name:
            raise ValueError("Sınıf adında virgül olamaz (bloktaki sınıf süzgeci virgülle ayırır)")
        if any(c[0].casefold() == name.casefold() for i, c in enumerate(self.classes) if i != skip):
            raise ValueError(f"'{name}' adında bir sınıf zaten var")
        return name

    def add_class(self, name: str, color=None) -> int:
        if len(self.classes) >= fd.MAX_CLASSES:
            raise ValueError(f"En çok {fd.MAX_CLASSES} sınıf olabilir")
        name = self._clean_name(name)
        self.classes.append([name, tuple(color) if color else fd.default_color(len(self.classes))])
        self.active_class = len(self.classes) - 1
        self._refresh_classes()
        return self.active_class

    def rename_class(self, i: int, name: str):
        if not 0 <= i < len(self.classes):
            raise ValueError("Sınıf seçilmedi")
        self.classes[i][0] = self._clean_name(name, skip=i)
        if self.model is not None and i < len(self.model.classes):
            self.model.classes[i] = self.classes[i][0]
            for dets in self._dets.values():
                for d in dets:
                    if d["class_id"] == i:
                        d["label"] = self.classes[i][0]
        self._refresh_classes()

    def _add_class_ui(self):
        name, ok = QInputDialog.getText(self, "Sınıf ekle", "Sınıf adı (ör. vida, somun, etiket):")
        if ok:
            try:
                self.add_class(name)
            except ValueError as exc:
                QMessageBox.warning(self, "Sınıf ekle", str(exc))

    def _rename_class_ui(self):
        i = self.active_class
        name, ok = QInputDialog.getText(self, "Yeniden adlandır", "Yeni ad:", text=self.classes[i][0])
        if ok:
            try:
                self.rename_class(i, name)
            except ValueError as exc:
                QMessageBox.warning(self, "Yeniden adlandır", str(exc))

    # --- boxes ----------------------------------------------------------------------------------------------------
    def _index(self, index):
        i = self.current if index is None else int(index)
        if not 0 <= i < len(self.images):
            raise ValueError("Önce bir resim ekleyin")
        return i

    def _push_undo(self, i):
        self._undo.append((i, [list(b) for b in self.images[i]["boxes"]]))
        del self._undo[:-MAX_UNDO]

    def _clip(self, i, x, y, w, h):
        H, W = self.images[i]["img"].shape[:2]
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
        if x1 - x0 < fd.MIN_BOX or y1 - y0 < fd.MIN_BOX:
            raise ValueError(f"Kutu çok küçük: en az {fd.MIN_BOX}x{fd.MIN_BOX} piksel olmalı (resmin içinde)")
        return [x0, y0, x1 - x0, y1 - y0]

    def _changed(self, i):
        self._refresh_image_names()
        if i == self.current:
            self.render()

    def add_box(self, x, y, w, h, cls: int | None = None, index: int | None = None) -> int:
        """Add a box (picture pixels) of a class (default: the chosen one); one undo step. Returns its number."""
        i = self._index(index)
        cls = self.active_class if cls is None else int(cls)
        if not 0 <= cls < len(self.classes):
            raise ValueError("Geçersiz sınıf")
        box = self._clip(i, x, y, w, h)
        self._push_undo(i)
        self.images[i]["boxes"].append(box + [cls])
        if i == self.current:
            self.selected = len(self.images[i]["boxes"]) - 1
        self._changed(i)
        return len(self.images[i]["boxes"]) - 1

    def set_box(self, bi: int, x, y, w, h, index: int | None = None):
        i = self._index(index)
        boxes = self.images[i]["boxes"]
        if not 0 <= bi < len(boxes):
            raise ValueError("Kutu seçilmedi")
        box = self._clip(i, x, y, w, h)
        self._push_undo(i)
        boxes[bi][:4] = box
        self._changed(i)

    def move_box(self, bi: int, dx, dy, index: int | None = None):
        i = self._index(index)
        boxes = self.images[i]["boxes"]
        if not 0 <= bi < len(boxes):
            raise ValueError("Kutu seçilmedi")
        H, W = self.images[i]["img"].shape[:2]
        x, y, w, h = boxes[bi][:4]
        x = min(max(0, x + int(round(dx))), max(0, W - w))      # moving keeps the size: stop at the edges
        y = min(max(0, y + int(round(dy))), max(0, H - h))
        self.set_box(bi, x, y, w, h, index=i)

    def delete_box(self, bi: int, index: int | None = None):
        i = self._index(index)
        boxes = self.images[i]["boxes"]
        if not 0 <= bi < len(boxes):
            raise ValueError("Silinecek kutuyu seçin")
        self._push_undo(i)
        boxes.pop(bi)
        if i == self.current:
            self.selected = -1
        self._changed(i)

    def set_selected_class(self, cls: int | None = None):
        i = self._index(None)
        if not 0 <= self.selected < len(self.images[i]["boxes"]):
            raise ValueError("Önce bir kutu seçin")
        self._push_undo(i)
        self.images[i]["boxes"][self.selected][4] = self.active_class if cls is None else int(cls)
        self._changed(i)

    def clear_current(self):
        if self.current >= 0 and self.images[self.current]["boxes"]:
            self._push_undo(self.current)
            self.images[self.current]["boxes"] = []
            self.selected = -1
            self._changed(self.current)

    def undo(self) -> bool:
        if not self._undo:
            return False
        i, boxes = self._undo.pop()
        if 0 <= i < len(self.images):
            self.images[i]["boxes"] = boxes
            self._refresh_image_names()
            self.set_current(i)
        return True

    def _drawn(self, x, y, w, h):
        if self.current < 0:
            return
        if w < fd.MIN_BOX or h < fd.MIN_BOX:                 # a click, not a box: just deselect
            self.selected = -1
            self.render()
            return
        self._ui(lambda: self.add_box(x, y, w, h))

    def _select(self, i):
        self.selected = i
        if self.current >= 0 and 0 <= i < len(self.images[self.current]["boxes"]):
            self.set_class(self.images[self.current]["boxes"][i][4])

    def accept_detections(self, which=None) -> int:
        """Turn the model's finds on the shown picture into boxes (all, or the given numbers), skipping those that
        overlap a box already there. One undo step; returns how many were added."""
        i = self._index(None)
        dets = self.detections(i) or []
        pick = [dets[k] for k in (range(len(dets)) if which is None else which) if 0 <= k < len(dets)]
        boxes = self.images[i]["boxes"]
        new = []
        for d in pick:
            if any(fd.box_iou(d["box"], b[:4]) > 0.3 for b in boxes + new):
                continue
            try:
                new.append(self._clip(i, *d["box"]) + [int(d["class_id"])])
            except ValueError:
                continue
        if new:
            self._push_undo(i)
            boxes.extend(new)
            self._changed(i)
        self.status.setText(f"{len(new)} kutu eklendi." if new else "Eklenecek yeni bulunan yok.")
        return len(new)

    # --- training -------------------------------------------------------------------------------------------------
    def train(self) -> fd.DetModel:
        """Train on every picture (unboxed areas count as background, so box every object in a picture you use).
        Raises ValueError (Turkish) when nothing usable is drawn."""
        if not self.images:
            raise ValueError("Eğitim için önce resim ekleyin")
        used = [d for d in self.images if d["boxes"]]
        if not used:
            raise ValueError("Hiç kutu çizilmedi: öğretmek için resimlerde nesnelerin etrafına kutu çizin")
        model = fd.DetModel.train([d["img"] for d in used], [d["boxes"] for d in used],
                                  [c[0] for c in self.classes], [c[1] for c in self.classes],
                                  angle_range=self.angle_spin.value(),
                                  scale_range=(self.scale_lo.value(), self.scale_hi.value()),
                                  progress=self._progress)
        self.model = model
        self._dets = {}
        self.render()
        return model

    def _progress(self, frac: float):
        self.status.setText(f"Eğitiliyor… %{int(100 * frac)}")
        QApplication.processEvents()

    def train_ui(self):
        if getattr(self, "_training", False):                 # progress updates let clicks in: no second run
            return
        self._training = True
        self.btn_train.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            m = self.train()
        except ValueError as exc:
            self.status.setText(f"<span style='color:#ef5350'>{exc}</span>")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_train.setEnabled(True)
            self._training = False
        per = ", ".join(f"{k}: {v['examples']} örnek" for k, v in m.info.get("classes", {}).items())
        missing = [c[0] for i, c in enumerate(self.classes) if i not in m.trained_classes]
        self.status.setText(f"Eğitildi ({m.info.get('images', 0)} resim, {m.info.get('seconds', 0)} sn; {per})."
                            + (f" Kutusu olmayan sınıflar: {', '.join(missing)}." if missing else "")
                            + " Diğer resimlere bakın: kaçırılanlara kutu çizip yeniden eğitin.")

    def _score_changed(self, v):
        self.score_label.setText(f"{v / 100:.2f}")
        self._dets = {}
        self.render()

    def detections(self, i: int | None = None):
        """The current model's finds on a picture (cached), None before training."""
        i = self.current if i is None else i
        if self.model is None or not 0 <= i < len(self.images):
            return None
        if i not in self._dets:
            self._dets[i] = self.model.detect(self.images[i]["img"], score=self.score_slider.value() / 100.0)
        return self._dets[i]

    # --- saving ---------------------------------------------------------------------------------------------------
    def save_model(self, path: str | None = None) -> str:
        if self.model is None:
            raise ValueError("Kaydedilecek model yok: önce 'Eğit'e basın")
        path = (path or self.path_edit.text()).strip()
        if not path:
            raise ValueError("Model dosyasının yolunu yazın")
        n = len(self.model.classes)
        self.model.classes = [c[0] for c in self.classes][:n]
        self.model.colors = [tuple(c[1]) for c in self.classes][:n]
        target, setting = save_target(path, fd.EXT)      # next to the flow, where the block looks for it
        saved = self.model.save(target)
        self.path_edit.setText(setting)
        self.applied = self.apply is not None and self.apply("model", setting) is not False
        return saved

    def _browse_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Model dosyası", self.path_edit.text(),
                                              "Nesne bulma modeli (*.vfdet)")
        if path:
            self.path_edit.setText(path)

    def save_ui(self):
        try:
            saved = self.save_model()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Modeli kaydet", str(exc))
            return
        if self.applied:
            note = " — blok bu modeli kullanıyor."
        elif self.apply is not None:
            note = " — blok şu an düzenleyicide açık değil: model yolunu bloğun ayarına yazın."
        else:
            note = ""
        self.status.setText(f"Kaydedildi: {saved}{note}")

    # --- drawing --------------------------------------------------------------------------------------------------
    def render(self):
        self.btn_undo.setEnabled(bool(self._undo))
        colors = [c[1] for c in self.classes]
        names = [c[0] for c in self.classes]
        if self.current < 0:
            self.canvas.set_content(None, (1, 1), [], [], colors, names)
            return
        d = self.images[self.current]
        img = d["img"]
        dets = []
        if self.show_pred.isChecked() and self.model is not None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                dets = self.detections(self.current) or []
            finally:
                QApplication.restoreOverrideCursor()
        pix = d.get("pix")
        if pix is None:
            pix = d["pix"] = QPixmap.fromImage(_qimage(np.ascontiguousarray(img)))
        self.canvas.set_content(pix, (img.shape[1], img.shape[0]), d["boxes"], dets, colors, names, self.selected)
