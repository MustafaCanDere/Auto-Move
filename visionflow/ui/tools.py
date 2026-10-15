"""Tool windows: image labelling (YOLO boxes) and YOLO model training."""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
from PySide6.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QSplitter, QVBoxLayout, QWidget)

from ..datasets import write_data_yaml, yaml_names  # noqa: F401  (yaml_names: used to live here)
from ..nodes.learning import IMAGE_EXT, _imread
from ..train import BASE_MODELS
from .worker import to_qimage

PALETTE = ["#ff5252", "#4fc3f7", "#aed581", "#ffb74d", "#ba68c8", "#f06292", "#4db6ac", "#fff176", "#90a4ae"]


class Dataset:
    """A folder of images with YOLO txt labels. Supports root/images + root/labels or a flat folder."""

    def __init__(self, folder: pathlib.Path):
        folder = pathlib.Path(folder)
        if (folder / "images").is_dir():
            self.root, self.img_dir, self.lbl_dir, self.images_rel = folder, folder / "images", folder / "labels", "images"
        elif folder.name.lower() == "images" and (folder.parent / "labels").is_dir():   # opened images/ itself
            self.root, self.img_dir, self.lbl_dir, self.images_rel = folder.parent, folder, folder.parent / "labels", \
                "images"
        elif folder.parent.name.lower() == "images" and (folder.parent.parent / "labels").is_dir():  # images/train
            self.root, self.img_dir, self.images_rel = folder.parent.parent, folder, "images"
            self.lbl_dir = self.root / "labels" / folder.name
        else:
            self.root = self.img_dir = self.lbl_dir = folder
            self.images_rel = "."
        # Roboflow layout (set/train/images): the class list is one level up
        if not any((self.root / f).exists() for f in ("classes.txt", "data.yaml")) and \
                any((self.root.parent / f).exists() for f in ("classes.txt", "data.yaml")) and \
                self.root.name.lower() in ("train", "valid", "val", "test"):
            self.root = self.root.parent
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.images = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
        cls_file = self.root / "classes.txt"
        self.classes = [c.strip() for c in cls_file.read_text(encoding="utf-8").splitlines() if c.strip()] \
            if cls_file.exists() else yaml_names(self.root / "data.yaml")
        self._other: dict[str, list[str]] = {}       # label lines the tool cannot edit (polygons), kept as they are

    def label_path(self, img: pathlib.Path) -> pathlib.Path:
        return self.lbl_dir / (img.stem + ".txt")

    def load(self, img: pathlib.Path, w: int, h: int) -> list[list]:
        p = self.label_path(img)
        boxes = []
        if p.exists():
            other = []
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                try:
                    if len(parts) in (5, 6):            # box, or box + confidence from an automatic labeller
                        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                        boxes.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                                      (cy + bh / 2) * h])
                        continue
                except ValueError:
                    pass
                if line.strip():
                    other.append(line)                  # segmentation polygon etc.: never thrown away
            self._other[img.name] = other
        return boxes

    def save(self, img: pathlib.Path, boxes: list[list], w: int, h: int):
        lines = []
        for c, x1, y1, x2, y2 in boxes:
            # clamp both ends: a box drawn partly outside the picture must stay inside 0..1
            x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
            y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        lines += self._other.get(img.name, [])
        used = [int(b[0]) for b in boxes] + [int(ln.split()[0]) for ln in self._other.get(img.name, [])
                                             if ln.split()[0].isdigit()]
        while used and len(self.classes) <= max(used):     # never write fewer names than the ids in use
            self.classes.append(f"sinif{len(self.classes)}")
        self.label_path(img).write_text("\n".join(lines), encoding="utf-8")
        self.save_meta()

    def save_meta(self):
        (self.root / "classes.txt").write_text("\n".join(self.classes), encoding="utf-8")
        write_data_yaml(self.root, self.classes, self.images_rel)     # keeps a train/val/test split

    def labelled_count(self) -> int:
        return sum(1 for i in self.images if self.label_path(i).exists() and self.label_path(i).stat().st_size)


class LabelCanvas(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pixmap: QPixmap | None = None
        self.boxes: list[list] = []
        self.current_class = 0
        self.classes: list[str] = []
        self._drag: QPointF | None = None
        self._mouse = QPointF()

    def set_image(self, img: np.ndarray | None, boxes: list[list]):
        self.pixmap = QPixmap.fromImage(to_qimage(img)) if img is not None else None
        self.boxes = boxes
        self.update()

    # image <-> widget coordinates
    def _xf(self):
        pw, ph = self.pixmap.width(), self.pixmap.height()
        s = min(self.width() / pw, self.height() / ph)
        return s, (self.width() - pw * s) / 2, (self.height() - ph * s) / 2

    def to_img(self, p: QPointF) -> QPointF:
        s, ox, oy = self._xf()
        return QPointF((p.x() - ox) / s, (p.y() - oy) / s)

    def to_widget(self, x, y) -> QPointF:
        s, ox, oy = self._xf()
        return QPointF(x * s + ox, y * s + oy)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#18191b"))
        if not self.pixmap:
            p.setPen(QColor("#8b8e94"))
            p.drawText(self.rect(), Qt.AlignCenter, "Etiketlemek için bir klasör açın")
            return
        s, ox, oy = self._xf()
        p.drawPixmap(QRectF(ox, oy, self.pixmap.width() * s, self.pixmap.height() * s), self.pixmap,
                     QRectF(self.pixmap.rect()))
        f = QFont()
        f.setBold(True)
        p.setFont(f)
        for c, x1, y1, x2, y2 in self.boxes:
            col = QColor(PALETTE[c % len(PALETTE)])
            a, b = self.to_widget(x1, y1), self.to_widget(x2, y2)
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(a, b))
            name = self.classes[c] if c < len(self.classes) else str(c)
            p.fillRect(QRectF(a.x(), a.y() - 18, p.fontMetrics().horizontalAdvance(name) + 8, 18), col)
            p.setPen(QColor("black"))
            p.drawText(QPointF(a.x() + 4, a.y() - 5), name)
        if self._drag is not None:
            p.setPen(QPen(QColor(PALETTE[self.current_class % len(PALETTE)]), 2, Qt.DashLine))
            p.drawRect(QRectF(self._drag, self._mouse))
        # crosshair helps aligning box edges
        p.setPen(QPen(QColor(255, 255, 255, 70), 1))
        p.drawLine(QPointF(0, self._mouse.y()), QPointF(self.width(), self._mouse.y()))
        p.drawLine(QPointF(self._mouse.x(), 0), QPointF(self._mouse.x(), self.height()))

    def mousePressEvent(self, e):
        if not self.pixmap:
            return
        if e.button() == Qt.LeftButton:
            self._drag = e.position()
        elif e.button() == Qt.RightButton:
            pt = self.to_img(e.position())
            for i in range(len(self.boxes) - 1, -1, -1):
                _, x1, y1, x2, y2 = self.boxes[i]
                if min(x1, x2) <= pt.x() <= max(x1, x2) and min(y1, y2) <= pt.y() <= max(y1, y2):
                    del self.boxes[i]
                    self.changed.emit()
                    break
            self.update()

    def mouseMoveEvent(self, e):
        self._mouse = e.position()
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is None or e.button() != Qt.LeftButton:
            return
        a, b = self.to_img(self._drag), self.to_img(e.position())
        self._drag = None
        if abs(a.x() - b.x()) > 3 and abs(a.y() - b.y()) > 3:
            self.boxes.append([self.current_class, min(a.x(), b.x()), min(a.y(), b.y()),
                               max(a.x(), b.x()), max(a.y(), b.y())])
            self.changed.emit()
        self.update()


class LabelingDialog(QDialog):
    def __init__(self, parent=None, folder: str | None = None, image: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Etiketleme Aracı (YOLO)")
        self.resize(1300, 800)
        self.ds: Dataset | None = None
        self.index = -1
        self.img_shape = (0, 0)
        self.dirty = False

        self.files = QListWidget()
        self.canvas = LabelCanvas()
        self.class_list = QListWidget()
        self.info = QLabel("Sol tık sürükle: kutu çiz · Sağ tık: kutu sil · A/D: önceki/sonraki · 1-9: sınıf · "
                           "Ctrl+Z: son kutuyu geri al")
        self.info.setObjectName("hint")
        self.progress = QLabel()

        open_btn = QPushButton("Klasör aç…")
        add_cls = QPushButton("+ Sınıf")
        self.detector = None                  # img -> detections, from "Modelle öner"
        self.model_path = ""
        self.conf = 0.35
        model_btn = QPushButton("Model seç…")
        model_btn.setToolTip("Kutuları önerecek YOLO ONNX modeli (ör. daha önce eğittiğiniz model)")
        suggest_btn = QPushButton("Modelle öner (M)")
        all_btn = QPushButton("Etiketsizleri modelle etiketle…")
        copy_btn = QPushButton("Önceki resmin kutuları (C)")
        self.model_label = QLabel("Model seçilmedi")
        self.model_label.setWordWrap(True)
        prev_btn, next_btn = QPushButton("◀ Önceki (A)"), QPushButton("Sonraki (D) ▶")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Sınıflar"))
        rl.addWidget(self.class_list)
        rl.addWidget(add_cls)
        rl.addWidget(self.progress)
        rl.addSpacing(12)
        rl.addWidget(QLabel("Otomatik etiketleme"))
        for w in (model_btn, self.model_label, suggest_btn, all_btn, copy_btn):
            rl.addWidget(w)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(open_btn)
        ll.addWidget(self.files)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.addWidget(self.canvas, 1)
        nav = QHBoxLayout()
        nav.addWidget(prev_btn)
        nav.addWidget(self.info, 1)
        nav.addWidget(next_btn)
        cl.addLayout(nav)
        split = QSplitter()
        for w in (left, center, right):
            split.addWidget(w)
        split.setSizes([220, 900, 200])
        QVBoxLayout(self).addWidget(split)

        open_btn.clicked.connect(lambda: self.open_folder())
        add_cls.clicked.connect(self.add_class)
        prev_btn.clicked.connect(lambda: self.go(self.index - 1))
        next_btn.clicked.connect(lambda: self.go(self.index + 1))
        self.files.currentRowChanged.connect(self.go)
        self.class_list.currentRowChanged.connect(self._set_class)
        self.canvas.changed.connect(self._changed)
        model_btn.clicked.connect(lambda: self.choose_model())
        suggest_btn.clicked.connect(self.suggest)
        all_btn.clicked.connect(self.label_all)
        copy_btn.clicked.connect(self.copy_previous)
        QShortcut(QKeySequence("M"), self, self.suggest)
        QShortcut(QKeySequence("C"), self, self.copy_previous)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_box)
        QShortcut(QKeySequence("A"), self, lambda: self.go(self.index - 1))
        QShortcut(QKeySequence("D"), self, lambda: self.go(self.index + 1))
        QShortcut(QKeySequence("Ctrl+S"), self, self.save)
        for i in range(9):
            QShortcut(QKeySequence(str(i + 1)), self, lambda i=i: self.class_list.setCurrentRow(i))
        if folder:
            self.open_folder(folder)
            if image:
                self.show_image(image)

    def show_image(self, image) -> bool:
        """Jump to a picture of the open folder (by path or file name)."""
        if not self.ds:
            return False
        name = pathlib.Path(str(image)).name
        for i, p in enumerate(self.ds.images):
            if p.name == name:
                self.go(i)
                return True
        return False

    def open_folder(self, folder: str | None = None):
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Resim klasörü seç")
            if not folder:
                return
        self.save()
        self.ds = Dataset(pathlib.Path(folder))
        if not self.ds.classes:
            self.ds.classes = ["nesne"]
        self.files.blockSignals(True)
        self.files.clear()
        for p in self.ds.images:
            self.files.addItem(p.name)
        self.files.blockSignals(False)
        self._refresh_classes()
        self.index = -1
        if self.ds.images:
            self.files.setCurrentRow(0)
        else:
            QMessageBox.information(self, "Boş klasör", "Bu klasörde resim yok.")

    def _refresh_classes(self):
        row = max(0, self.class_list.currentRow())
        self.class_list.clear()
        for i, c in enumerate(self.ds.classes):
            self.class_list.addItem(f"{i + 1}. {c}")
            self.class_list.item(i).setForeground(QColor(PALETTE[i % len(PALETTE)]))
        self.class_list.setCurrentRow(min(row, len(self.ds.classes) - 1))
        self.canvas.classes = self.ds.classes

    def add_class(self, name: str | None = None):
        if not self.ds:
            return
        if not name:
            name, ok = QInputDialog.getText(self, "Yeni sınıf", "Sınıf adı:")
            if not ok or not name.strip():
                return
        self.ds.classes.append(name.strip())
        self.ds.save_meta()
        self._refresh_classes()
        self.class_list.setCurrentRow(len(self.ds.classes) - 1)

    def _set_class(self, row):
        if row >= 0:
            self.canvas.current_class = row

    def _changed(self):
        self.dirty = True
        self.save()

    def save(self):
        if self.ds and 0 <= self.index < len(self.ds.images) and self.dirty:
            h, w = self.img_shape
            self.ds.save(self.ds.images[self.index], self.canvas.boxes, w, h)
            self.dirty = False
        self._update_progress()

    def _update_progress(self):
        if self.ds:
            self.progress.setText(f"Etiketli: {self.ds.labelled_count()} / {len(self.ds.images)}")

    def go(self, index: int):
        if not self.ds or not (0 <= index < len(self.ds.images)) or index == self.index:
            return
        self.save()
        self.index = index
        self.files.blockSignals(True)
        self.files.setCurrentRow(index)
        self.files.blockSignals(False)
        path = self.ds.images[index]
        img = _imread(path)
        self.img_shape = img.shape[:2] if img is not None else (1, 1)
        self.canvas.set_image(img, self.ds.load(path, self.img_shape[1], self.img_shape[0]))
        self.setWindowTitle(f"Etiketleme Aracı — {path.name}")
        self._update_progress()

    def closeEvent(self, e):
        self.save()
        super().closeEvent(e)

    # --- automatic labelling ------------------------------------------------------------------------------------
    def undo_box(self):
        if self.canvas.boxes:
            self.canvas.boxes.pop()
            self.canvas.update()
            self._changed()

    def choose_model(self, path: str | None = None) -> bool:
        if not path:
            start = str(pathlib.Path("models").resolve()) if pathlib.Path("models").is_dir() else ""
            path, _ = QFileDialog.getOpenFileName(self, "YOLO modeli (ONNX)", start, "ONNX modeli (*.onnx)")
            if not path:
                return False
        from ..core import create
        node = create("YoloDetector")
        node.values.update(model=path, conf=self.conf)
        if not node.safe_setup():
            QMessageBox.warning(self, "Model açılamadı", node.setup_error or "")
            return False
        self.detector = lambda img: node.process({"image": img})["detections"]
        self.model_path = path
        self.model_label.setText(f"Model: {pathlib.Path(path).name}")
        return True

    def _class_index(self, name: str) -> int:
        """Dataset class for a model label; a label the dataset does not have yet becomes a new class."""
        name = " ".join(str(name).split()) or "nesne"   # one line: classes.txt has one name per line
        if name not in self.ds.classes:
            if self.ds.classes == ["nesne"] and not self.ds.labelled_count():
                self.ds.classes = []                 # the placeholder of an empty dataset is replaced
            self.ds.classes.append(name)
            self.ds.save_meta()
            self._refresh_classes()
        return self.ds.classes.index(name)

    @staticmethod
    def _iou(a, b) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    def _suggested_boxes(self, img, existing: list[list]) -> list[list]:
        new = []
        ih, iw = img.shape[:2]
        for d in self.detector(img) or []:
            try:
                x, y, w, h = (float(v) for v in d["box"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite([x, y, w, h])):
                continue
            x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(iw), x + w), min(float(ih), y + h)
            if x2 - x1 < 2 or y2 - y1 < 2:                       # empty, or outside the picture
                continue
            box = [self._class_index(d.get("label", "nesne")), x1, y1, x2, y2]
            # a box the user (or an earlier suggestion) already drew for the same thing is not added twice
            if any(b[0] == box[0] and self._iou(b[1:], box[1:]) > 0.5 for b in existing + new):
                continue
            new.append(box)
        return new

    def suggest(self):
        if not self.ds or not (0 <= self.index < len(self.ds.images)):
            return
        if self.detector is None and not self.choose_model():
            return
        img = _imread(self.ds.images[self.index])
        if img is None:
            return
        try:
            new = self._suggested_boxes(img, self.canvas.boxes)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Model çalışmadı", str(exc))
            return
        self.canvas.boxes.extend(new)
        self.canvas.update()
        self.info.setText(f"Model {len(new)} kutu önerdi. Yanlış olanı sağ tıkla silin, eksik olanı çizin.")
        if new:
            self._changed()

    def label_all(self):
        """Run the model on every image that has no label yet. Already labelled images are never touched."""
        if not self.ds or not self.ds.images:
            return
        if self.detector is None and not self.choose_model():
            return
        self.save()
        todo = [p for p in self.ds.images
                if not (self.ds.label_path(p).exists() and self.ds.label_path(p).stat().st_size)]
        if not todo:
            QMessageBox.information(self, "Etiketleme", "Etiketsiz resim yok.")
            return
        from PySide6.QtWidgets import QApplication, QProgressDialog
        prog = QProgressDialog("Model etiketliyor…", "Durdur", 0, len(todo), self)
        prog.setWindowModality(Qt.WindowModal)       # no drawing, browsing or folder change while it runs
        prog.setMinimumDuration(300)
        ds = self.ds
        done = boxes = 0
        failed = False
        written: set = set()
        for i, path in enumerate(todo):
            if prog.wasCanceled() or self.ds is not ds:
                break
            lp = ds.label_path(path)
            if lp in written or (lp.exists() and lp.stat().st_size):
                continue            # labelled meanwhile, or a.jpg + a.png sharing a.txt in a flat folder
            prog.setValue(i)
            QApplication.processEvents()
            img = _imread(path)
            if img is None:
                continue
            try:
                new = self._suggested_boxes(img, [])
            except Exception as exc:  # noqa: BLE001
                prog.setValue(len(todo))
                QMessageBox.warning(self, "Model çalışmadı", f"{path.name}: {exc}")
                failed = True
                break
            if new:
                ds.load(path, img.shape[1], img.shape[0])             # keeps polygon lines of this image
                ds.save(path, new, img.shape[1], img.shape[0])
                written.add(lp)
                done += 1
                boxes += len(new)
        prog.setValue(len(todo))
        cur, self.index = self.index, -1
        self.go(max(cur, 0))                                          # show the current image's new boxes
        self._update_progress()
        if failed:
            return
        QMessageBox.information(self, "Etiketleme", f"{done} resme {boxes} kutu eklendi. Lütfen kontrol edin: "
                                "model yanılabilir, yanlış kutuları silin.")

    def copy_previous(self):
        """Boxes of the previous image (video frames, a product that does not move)."""
        if not self.ds or self.index <= 0:
            return
        prev = self.ds.images[self.index - 1]
        img = _imread(prev)
        if img is None:
            return
        old = self.ds.load(prev, img.shape[1], img.shape[0])
        h, w = self.img_shape
        sx, sy = w / img.shape[1], h / img.shape[0]                   # resized frames: scale the boxes
        added = 0
        for c, x1, y1, x2, y2 in old:
            box = [c, x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            if not any(b[0] == c and self._iou(b[1:], box[1:]) > 0.5 for b in self.canvas.boxes):
                self.canvas.boxes.append(box)
                added += 1
        self.canvas.update()
        if added:
            self._changed()


class TrainDialog(QDialog):
    """Runs visionflow.train in a child process and streams its output."""
    finished_model = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Modeli Eğit")
        self.resize(760, 560)
        self.proc: QProcess | None = None
        self.data = QLineEdit()
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.data)
        row.addWidget(browse)
        self.model = QComboBox()
        for k, v in BASE_MODELS.items():
            self.model.addItem(f"{k} — {v}", k)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 2000)
        self.epochs.setValue(50)
        self.imgsz = QComboBox()
        self.imgsz.addItems(["320", "480", "640", "800", "1024"])
        self.imgsz.setCurrentText("640")
        self.device = QComboBox()
        self.device.addItems(["otomatik", "cpu", "0"])
        self.name = QLineEdit("benim_modelim")
        form = QFormLayout()
        form.addRow("Veri seti (data.yaml)", row)
        form.addRow("Başlangıç modeli", self.model)
        form.addRow("Tur (epoch)", self.epochs)
        form.addRow("Görüntü boyutu", self.imgsz)
        form.addRow("Cihaz", self.device)
        form.addRow("Model adı", self.name)
        self.out = QPlainTextEdit(readOnly=True)
        self.out.setFont(QFont("Consolas", 9))
        self.start_btn = QPushButton("Eğitimi başlat")
        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.setEnabled(False)
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.start_btn)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        hint = QLabel("Önce Araçlar → Etiketleme Aracı ile resimleri etiketleyin. Eğitim bitince model "
                      "models/ klasörüne ONNX olarak kaydedilir ve YOLO bloğunda seçilebilir. "
                      "Gerekli paket: pip install ultralytics")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        lay.addWidget(hint)
        lay.addWidget(self.out, 1)
        lay.addLayout(btns)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "data.yaml seç", "", "YAML (*.yaml *.yml)")
        if p:
            self.data.setText(p)

    @staticmethod
    def program() -> tuple[str | None, list[str]]:
        """(python, args that start the trainer). The packaged app has no Python of its own for heavy
        training (ultralytics/torch are not bundled), so it uses an installed Python and a copy of train.py."""
        if not getattr(sys, "frozen", False):
            return sys.executable, ["-u", "-m", "visionflow.train"]
        import shutil
        script = pathlib.Path(getattr(sys, "_MEIPASS", ".")) / "train.py"
        for name in ("py", "python", "python3"):
            exe = shutil.which(name)
            if exe:
                return exe, (["-3"] if name == "py" else []) + ["-u", str(script)]
        return None, []

    def command(self) -> list[str]:
        dev = self.device.currentText()
        args = self.program()[1] + ["--data", self.data.text(), "--model", self.model.currentData(),
                "--epochs", str(self.epochs.value()), "--imgsz", self.imgsz.currentText(),
                "--name", self.name.text().strip() or "model"]
        if dev != "otomatik":
            args += ["--device", dev]
        return args

    def start(self):
        if not pathlib.Path(self.data.text()).exists():
            QMessageBox.warning(self, "Veri seti yok", "Geçerli bir data.yaml seçin.")
            return
        exe, _ = self.program()
        if exe is None:
            QMessageBox.warning(self, "Python gerekli",
                                "Model eğitimi için bilgisayarda Python kurulu olmalı (python.org), ardından: "
                                "pip install ultralytics")
            return
        self.out.clear()
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")              # Windows: child output in UTF-8, so "BİTTİ:" is recognised
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(pathlib.Path.cwd()))
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.start(exe, self.command())
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _read(self):
        # output arrives in arbitrary pieces: a "İ" or a line can be split between two reads
        if getattr(self, "_decoder", None) is None:
            import codecs
            self._decoder, self._partial = codecs.getincrementaldecoder("utf-8")(errors="replace"), ""
        text = self._partial + self._decoder.decode(bytes(self.proc.readAllStandardOutput()))
        text = text.replace("\r", "\n")
        lines = text.split("\n")
        self._partial = lines.pop()               # unfinished last line: wait for the rest
        for line in lines:
            if line.strip():
                self.out.appendPlainText(line)
                if line.startswith("BİTTİ: "):
                    self.finished_model.emit(line[len("BİTTİ: "):].strip())

    def _done(self, code, _status):
        rest = getattr(self, "_partial", "")
        if rest.strip():
            self.out.appendPlainText(rest)
            if rest.startswith("BİTTİ: "):
                self.finished_model.emit(rest[len("BİTTİ: "):].strip())
        self._decoder, self._partial = None, ""
        self.out.appendPlainText(f"\n— işlem bitti (kod {code}) —")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def stop(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)
