"""Etiketle ve Eğit (Az Örnekle Segmentasyon): paint a few pictures with a brush (one colour per class), press Eğit,
look at the prediction drawn over the picture, paint corrections where it is wrong and train again; then save the
model for the block. A rectangle drawn in GrabCut mode gives a quick first mask of an object."""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut, QIcon
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QColorDialog, QDialog, QFileDialog,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QRadioButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from .. import fewshot_seg as fs
from ..core.node import resolve_path, save_target

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_SIDE = 1280            # pictures are kept at most this big in the window (training works smaller anyway)
MAX_UNDO = 40


def _read(path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), np.uint8)             # also works with non-ASCII paths on Windows
    except OSError:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    return img


def _fit(img: np.ndarray) -> np.ndarray:
    bgr = fs.to_bgr8(img)
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(bgr)


def _qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888).copy()


class Canvas(QWidget):
    """Shows the picture with the paint and the prediction over it; turns mouse drags into strokes / rectangles in
    picture pixels."""
    stroke_started = Signal(float, float)
    stroke_moved = Signal(float, float)
    stroke_ended = Signal()
    rect_drawn = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.pix: QPixmap | None = None
        self.img_size = (1, 1)
        self.mode = "brush"
        self.brush = 8
        self._drag = False
        self._rect_start: QPointF | None = None
        self._rect_end: QPointF | None = None
        self._cursor: QPointF | None = None

    def set_pixmap(self, pix: QPixmap | None, size):
        self.pix, self.img_size = pix, size
        self.update()

    def _target(self) -> QRectF:
        w, h = self.img_size
        s = min(self.width() / max(1, w), self.height() / max(1, h))
        return QRectF((self.width() - w * s) / 2, (self.height() - h * s) / 2, w * s, h * s)

    def to_image(self, pos) -> tuple[float, float]:
        r = self._target()
        s = r.width() / max(1, self.img_size[0])
        return (pos.x() - r.x()) / s, (pos.y() - r.y()) / s

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        if self.pix is None:
            p.setPen(QColor("#aaa"))
            p.drawText(self.rect(), Qt.AlignCenter, "Resim ekleyin: 'Klasör…', 'Dosya…' veya 'Akıştan al'")
            return
        r = self._target()
        p.drawPixmap(r, self.pix, QRectF(self.pix.rect()))
        s = r.width() / max(1, self.img_size[0])
        if self.mode == "rect" and self._rect_start is not None and self._rect_end is not None:
            p.setPen(QPen(QColor("#ffeb3b"), 2, Qt.DashLine))
            p.drawRect(QRectF(self._rect_start, self._rect_end).normalized())
        elif self._cursor is not None and self.mode in ("brush", "erase"):
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawEllipse(self._cursor, self.brush * s, self.brush * s)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or self.pix is None:
            return
        self._drag = True
        if self.mode == "rect":
            self._rect_start = self._rect_end = e.position()
        else:
            self.stroke_started.emit(*self.to_image(e.position()))

    def mouseMoveEvent(self, e):
        self._cursor = e.position()
        if self._drag:
            if self.mode == "rect":
                self._rect_end = e.position()
            else:
                self.stroke_moved.emit(*self.to_image(e.position()))
        self.update()

    def mouseReleaseEvent(self, e):
        if not self._drag:
            return
        self._drag = False
        if self.mode == "rect" and self._rect_start is not None:
            (x0, y0), (x1, y1) = self.to_image(self._rect_start), self.to_image(e.position())
            self._rect_start = self._rect_end = None
            self.rect_drawn.emit(int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))
        else:
            self.stroke_ended.emit()
        self.update()

    def leaveEvent(self, _e):
        self._cursor = None
        self.update()


class SegTeachDialog(QDialog):
    """The painting / training window. ``node`` is the Az Örnekle Segmentasyon block (or None), ``apply(name, value)``
    sets one of its settings (used to point it at the saved model)."""

    def __init__(self, parent=None, node=None, apply=None):
        super().__init__(parent)
        self.node, self.apply = node, apply
        self.setWindowTitle("Az Örnekle Segmentasyon — Etiketle ve Eğit")
        self.resize(1100, 720)
        self.images: list[dict] = []            # {"name", "img" (BGR8), "labels" (uint8, 0 = not painted)}
        self.current = -1
        self.classes: list[list] = [["arka plan", fs.default_color(0)], ["nesne", fs.default_color(1)]]
        self.active_class = 1
        self.model: fs.SegModel | None = None
        self._pred: dict[int, np.ndarray] = {}  # image index → predicted class map (for the current model)
        self._undo: list[tuple[int, np.ndarray]] = []
        self._last_pt: tuple[float, float] | None = None
        self._build()
        self._load_existing_model()
        self._refresh_classes()
        self.render()

    # --- layout ---------------------------------------------------------------------------------------------------
    def _build(self):
        self.canvas = Canvas()
        self.canvas.stroke_started.connect(self._stroke_start)
        self.canvas.stroke_moved.connect(self._stroke_move)
        self.canvas.stroke_ended.connect(self._stroke_end)
        self.canvas.rect_drawn.connect(lambda x, y, w, h: self._grabcut_ui((x, y, w, h)))

        def button(text, slot, tip=None):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False: slot())
            if tip:
                b.setToolTip(tip)
            return b

        # pictures
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.set_current)
        self.btn_folder = button("Klasör…", self._add_folder_ui)
        self.btn_files = button("Dosya…", self._add_files_ui)
        self.btn_flow = button("Akıştan al", self.add_from_flow, "Bloğun son çalıştırmada gördüğü resmi ekler")
        self.btn_remove = button("Kaldır", self.remove_current)
        img_btns = QHBoxLayout()
        for b in (self.btn_folder, self.btn_files, self.btn_flow, self.btn_remove):
            img_btns.addWidget(b)

        # classes
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._class_picked)
        cls_btns = QHBoxLayout()
        for b in (button("Sınıf ekle", self._add_class_ui), button("Yeniden adlandır", self._rename_class_ui),
                  button("Renk…", self._recolor_ui)):
            cls_btns.addWidget(b)

        # tools
        self.tool_group = QButtonGroup(self)
        tools = QHBoxLayout()
        self.tool_buttons = {}
        for key, text in (("brush", "Fırça (B)"), ("erase", "Silgi (E)"), ("rect", "GrabCut kutusu (G)")):
            rb = QRadioButton(text)
            rb.toggled.connect(lambda on, k=key: on and self.set_tool(k))
            self.tool_group.addButton(rb)
            self.tool_buttons[key] = rb
            tools.addWidget(rb)
        self.tool_buttons["brush"].setChecked(True)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 60)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 60)
        self.size_slider.valueChanged.connect(self.size_spin.setValue)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.size_spin.valueChanged.connect(self._set_brush)
        self.size_spin.setValue(8)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Fırça boyu:"))
        size_row.addWidget(self.size_slider, 1)
        size_row.addWidget(self.size_spin)

        self.btn_undo = button("Geri al (Ctrl+Z)", self.undo)
        self.btn_clear = button("Boyamayı temizle", self.clear_current)
        self.btn_accept = button("Tahmini boyamaya çevir", self.prediction_to_labels,
                                 "Modelin bu resim için tahminini boyama olarak alır; sonra yalnızca hataları düzeltin")
        edit_row = QHBoxLayout()
        for b in (self.btn_undo, self.btn_clear):
            edit_row.addWidget(b)

        self.btn_train = button("Eğit", self.train_ui)
        self.btn_train.setStyleSheet("font-weight:bold; padding:6px")
        self.show_pred = QCheckBox("Tahmini göster")
        self.show_pred.setChecked(True)
        self.show_pred.toggled.connect(lambda _: self.render())
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 90)
        self.opacity.setValue(45)
        self.opacity.valueChanged.connect(lambda _: self.render())
        self.use_pos = QCheckBox("Konumu da kullan (nesneler hep aynı yerdeyse)")
        pred_row = QHBoxLayout()
        pred_row.addWidget(self.show_pred)
        pred_row.addWidget(QLabel("Saydamlık:"))
        pred_row.addWidget(self.opacity, 1)

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
        side.addWidget(QLabel("<b>Sınıflar</b> (boyanacak sınıfı seçin, 1–9 tuşları)"))
        side.addWidget(self.class_list, 1)
        side.addLayout(cls_btns)
        side.addWidget(QLabel("<b>Araç</b>"))
        side.addLayout(tools)
        side.addLayout(size_row)
        side.addLayout(edit_row)
        side.addWidget(self.btn_accept)
        side.addWidget(QLabel("<b>Eğitim</b>"))
        side.addWidget(self.use_pos)
        side.addWidget(self.btn_train)
        side.addLayout(pred_row)
        side.addWidget(QLabel("<b>Model dosyası</b>"))
        side.addLayout(save_row)
        side.addWidget(self.btn_save)
        side.addWidget(self.status)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(340)

        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(side_w)

        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("B"), self, lambda: self.tool_buttons["brush"].setChecked(True))
        QShortcut(QKeySequence("E"), self, lambda: self.tool_buttons["erase"].setChecked(True))
        QShortcut(QKeySequence("G"), self, lambda: self.tool_buttons["rect"].setChecked(True))
        QShortcut(QKeySequence("T"), self, self.train_ui)
        for i in range(9):
            QShortcut(QKeySequence(str(i + 1)), self, lambda i=i: self.set_class(i))
        self.status.setText("1) Resim ekleyin  2) Her sınıftan birkaç çizgi boyayın (arka planı da)  3) Eğit  "
                            "4) Hatalı yerleri boyayıp yeniden eğitin  5) Kaydedin.")

    def _default_model_path(self) -> str:
        cur = str(self.node.values.get("model", "")) if self.node is not None else ""
        return cur or "modeller/segmentasyon.vfseg"

    def _load_existing_model(self):
        """The block's current model: its classes and colours, so a retrain keeps the same names."""
        path = resolve_path(self.path_edit.text().strip())
        if not path or not pathlib.Path(path).is_file():
            return
        try:
            m = fs.SegModel.load(path)
        except ValueError:
            return
        self.model = m
        self.classes = [[n, tuple(c)] for n, c in zip(m.classes, m.colors)]
        self.use_pos.setChecked(m.use_position)

    # --- pictures -------------------------------------------------------------------------------------------------
    def add_image(self, img, name: str = "resim") -> int:
        """Add a picture (any depth / channels); returns its index. Raises ValueError for a non-picture."""
        bgr = _fit(img)
        self.images.append({"name": name, "img": bgr, "labels": np.zeros(bgr.shape[:2], np.uint8), "qimg": None})
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
        self._pred = {(k if k < i else k - 1): v for k, v in self._pred.items() if k != i}
        self._undo = [(k if k < i else k - 1, v) for k, v in self._undo if k != i]
        self.image_list.blockSignals(True)
        self.image_list.takeItem(i)
        self.image_list.blockSignals(False)
        self.current = -1
        self.set_current(min(i, len(self.images) - 1))

    def set_current(self, i: int):
        if i < 0 or i >= len(self.images):
            self.current = -1
            self.render()
            return
        self.current = i
        if self.image_list.currentRow() != i:
            self.image_list.blockSignals(True)
            self.image_list.setCurrentRow(i)
            self.image_list.blockSignals(False)
        if self.model is not None and i not in self._pred:
            self._predict(i)
        self.render()

    # --- classes --------------------------------------------------------------------------------------------------
    def _refresh_classes(self):
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for i, (name, col) in enumerate(self.classes):
            pm = QPixmap(14, 14)
            pm.fill(QColor(int(col[2]), int(col[1]), int(col[0])))
            it = QListWidgetItem(QIcon(pm), f"{i + 1}  {name}" + ("  (arka plan)" if i == 0 and name != "arka plan"
                                                                   else ""))
            self.class_list.addItem(it)
        self.class_list.setCurrentRow(min(self.active_class, len(self.classes) - 1))
        self.class_list.blockSignals(False)

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
        if any(c[0].casefold() == name.casefold() for i, c in enumerate(self.classes) if i != skip):
            raise ValueError(f"'{name}' adında bir sınıf zaten var")
        return name

    def add_class(self, name: str, color=None) -> int:
        if len(self.classes) >= fs.MAX_CLASSES:
            raise ValueError(f"En çok {fs.MAX_CLASSES} sınıf olabilir")
        name = self._clean_name(name)
        self.classes.append([name, tuple(color) if color else fs.default_color(len(self.classes))])
        self.active_class = len(self.classes) - 1
        self._refresh_classes()
        return self.active_class

    def rename_class(self, i: int, name: str):
        if not 0 <= i < len(self.classes):
            raise ValueError("Sınıf seçilmedi")
        self.classes[i][0] = self._clean_name(name, skip=i)
        if self.model is not None and i < len(self.model.classes):
            self.model.classes[i] = self.classes[i][0]
        self._refresh_classes()

    def set_class_color(self, i: int, bgr):
        self.classes[i][1] = tuple(int(v) for v in bgr)
        if self.model is not None and i < len(self.model.colors):
            self.model.colors[i] = self.classes[i][1]
        self._refresh_classes()
        self.render()

    def _add_class_ui(self):
        name, ok = QInputDialog.getText(self, "Sınıf ekle", "Sınıf adı (ör. çizik, leke, yaprak):")
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

    def _recolor_ui(self):
        b, g, r = self.classes[self.active_class][1]
        col = QColorDialog.getColor(QColor(int(r), int(g), int(b)), self, "Sınıf rengi")
        if col.isValid():
            self.set_class_color(self.active_class, (col.blue(), col.green(), col.red()))

    # --- painting -------------------------------------------------------------------------------------------------
    def set_tool(self, key: str):
        self.canvas.mode = key
        if not self.tool_buttons[key].isChecked():
            self.tool_buttons[key].setChecked(True)
        self.canvas.update()

    def _set_brush(self, v):
        self.canvas.brush = int(v)
        self.canvas.update()

    def _push_undo(self, i):
        self._undo.append((i, self.images[i]["labels"].copy()))
        del self._undo[:-MAX_UNDO]

    def paint_stroke(self, points, cls: int | None = None, radius: int | None = None, erase: bool = False,
                     index: int | None = None):
        """Paint (or erase) a polyline of picture-pixel points on a picture: one undo step."""
        i = self.current if index is None else index
        if not 0 <= i < len(self.images):
            raise ValueError("Önce bir resim ekleyin")
        cls = self.active_class if cls is None else int(cls)
        if not erase and not 0 <= cls < len(self.classes):
            raise ValueError("Geçersiz sınıf")
        self._push_undo(i)
        self._draw(i, points, 0 if erase else cls + 1, self.canvas.brush if radius is None else int(radius))
        if i == self.current:
            self.render()

    def _draw(self, i, points, value, radius):
        lab = self.images[i]["labels"]
        pts = [(int(round(x)), int(round(y))) for x, y in points]
        if len(pts) == 1:
            cv2.circle(lab, pts[0], radius, int(value), -1)
        for a, b in zip(pts, pts[1:]):
            cv2.line(lab, a, b, int(value), max(1, 2 * radius), cv2.LINE_8)
            cv2.circle(lab, b, radius, int(value), -1)

    def _stroke_value(self):
        return 0 if self.canvas.mode == "erase" else self.active_class + 1

    def _stroke_start(self, x, y):
        if self.current < 0:
            return
        self._push_undo(self.current)
        self._last_pt = (x, y)
        self._draw(self.current, [(x, y)], self._stroke_value(), self.canvas.brush)
        self.render()

    def _stroke_move(self, x, y):
        if self.current < 0 or self._last_pt is None:
            return
        self._draw(self.current, [self._last_pt, (x, y)], self._stroke_value(), self.canvas.brush)
        self._last_pt = (x, y)
        self.render()

    def _stroke_end(self):
        self._last_pt = None

    def undo(self):
        if not self._undo:
            return False
        i, lab = self._undo.pop()
        if 0 <= i < len(self.images):
            self.images[i]["labels"] = lab
            self.set_current(i)
        return True

    def clear_current(self):
        if self.current >= 0 and self.images[self.current]["labels"].any():
            self._push_undo(self.current)
            self.images[self.current]["labels"][:] = 0
            self.render()

    def grabcut(self, rect, cls: int | None = None, index: int | None = None):
        """GrabCut inside a rectangle: the found object is painted with ``cls`` (default: the chosen class) and a
        thin ring just outside the rectangle as the background. One undo step."""
        i = self.current if index is None else index
        if not 0 <= i < len(self.images):
            raise ValueError("Önce bir resim ekleyin")
        cls = self.active_class if cls is None else int(cls)
        if cls == 0:
            raise ValueError("GrabCut için arka plan dışında bir sınıf seçin")
        fg = fs.grabcut_rect(self.images[i]["img"], rect)
        if not fg.any():
            raise ValueError("GrabCut bu kutuda bir nesne bulamadı")
        self._push_undo(i)
        lab = self.images[i]["labels"]
        x, y, w, h = (int(v) for v in rect)
        ring = np.zeros(lab.shape, np.uint8)
        cv2.rectangle(ring, (x - 6, y - 6), (x + w + 6, y + h + 6), 1, 6)
        lab[(ring > 0) & (lab == 0)] = 1                         # background ring, never over the user's paint
        lab[fg] = cls + 1
        if i == self.current:
            self.render()
        return int(fg.sum())

    def _grabcut_ui(self, rect):
        try:
            n = self.grabcut(rect)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(f"GrabCut: {n} piksel '{self.classes[self.active_class][0]}' olarak boyandı. "
                            "Hatalı yerleri fırça/silgi ile düzeltin.")

    def prediction_to_labels(self):
        i = self.current
        if i < 0 or i not in self._pred:
            self.status.setText("Önce eğitin: bu resim için tahmin yok.")
            return
        self._push_undo(i)
        self.images[i]["labels"] = (self._pred[i].astype(np.uint8) + 1)
        self.render()

    # --- training -------------------------------------------------------------------------------------------------
    def train(self) -> fs.SegModel:
        """Train on every painted picture; predictions are refreshed for the shown picture. ValueError (Turkish)
        when nothing usable is painted."""
        if not self.images:
            raise ValueError("Eğitim için önce resim ekleyin")
        model = fs.SegModel.train([d["img"] for d in self.images], [d["labels"] for d in self.images],
                                  [c[0] for c in self.classes], [c[1] for c in self.classes],
                                  use_position=self.use_pos.isChecked())
        self.model = model
        self._pred = {}
        if self.current >= 0:
            self._predict(self.current)
        self.render()
        return model

    def train_ui(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            m = self.train()
        except ValueError as exc:
            self.status.setText(f"<span style='color:#ef5350'>{exc}</span>")
            return
        finally:
            QApplication.restoreOverrideCursor()
        px = ", ".join(f"{k}: {v}" for k, v in m.info.get("pixels", {}).items())
        self.status.setText(f"Eğitildi ({m.info.get('images', 0)} resim; boyanan piksel {px}; eğitim doğruluğu "
                            f"%{100 * m.info.get('train_accuracy', 0):.1f}). Hatalı yerleri boyayıp yeniden eğitin.")

    def _predict(self, i):
        if self.model is None:
            return None
        self._pred[i] = self.model.predict(self.images[i]["img"], smooth=1)
        return self._pred[i]

    def prediction(self, i: int | None = None):
        i = self.current if i is None else i
        if i < 0 or self.model is None:
            return None
        return self._pred.get(i) if i in self._pred else self._predict(i)

    # --- saving ---------------------------------------------------------------------------------------------------
    def save_model(self, path: str | None = None) -> str:
        if self.model is None:
            raise ValueError("Kaydedilecek model yok: önce 'Eğit'e basın")
        path = (path or self.path_edit.text()).strip()
        if not path:
            raise ValueError("Model dosyasının yolunu yazın")
        self.model.classes = [c[0] for c in self.classes][:len(self.model.classes)]
        self.model.colors = [tuple(c[1]) for c in self.classes][:len(self.model.colors)]
        target, setting = save_target(path, fs.EXT)      # next to the flow, where the block looks for it
        saved = self.model.save(target)
        self.path_edit.setText(setting)
        self.applied = self.apply is not None and self.apply("model", setting) is not False
        return saved

    def _browse_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Model dosyası", self.path_edit.text(),
                                              "Segmentasyon modeli (*.vfseg)")
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
        if self.current < 0:
            self.canvas.set_pixmap(None, (1, 1))
            self.btn_undo.setEnabled(bool(self._undo))
            return
        d = self.images[self.current]
        img = d["img"]
        alpha = self.opacity.value() / 100.0
        colors = [c[1] for c in self.classes]
        pred = self._pred.get(self.current) if self.show_pred.isChecked() else None
        out = img
        if pred is not None and pred.shape == img.shape[:2]:
            out = fs.overlay(img, np.minimum(pred, len(colors) - 1), colors, alpha * 0.8)
        lab = d["labels"]
        if lab.any():
            out = out.copy() if out is img else out
            m = lab > 0
            col = fs.colorize(np.minimum(lab - 1, len(colors) - 1).astype(np.uint8) * m, colors)
            out[m] = (out[m] * 0.25 + col[m] * 0.75).astype(np.uint8)    # paint: strong, prediction: soft
        self.canvas.set_pixmap(QPixmap.fromImage(_qimage(np.ascontiguousarray(out))), (img.shape[1], img.shape[0]))
        self.btn_undo.setEnabled(bool(self._undo))
