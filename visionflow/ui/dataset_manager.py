"""Araçlar → Veri seti yöneticisi: statistics, problems, thumbnails, class edits, split, export, augmentation and
model-assisted labelling with a review queue (least sure first). The work itself is in visionflow/datasets.py."""
from __future__ import annotations

import pathlib

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListView,
                               QListWidget, QListWidgetItem, QMessageBox, QProgressDialog, QPushButton, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .. import datasets as D
from .tools import PALETTE
from .worker import to_qimage

THUMB = 132
MAX_THUMBS = 2000


def _draw_boxes(img: np.ndarray, boxes, dashed=()) -> np.ndarray:
    """A copy of the picture with label boxes (solid) and suggestions (dashed, with confidence)."""
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    t = max(1, int(round(max(w, h) / 400)))
    for group, dash in ((boxes, False), (dashed, True)):
        for b in group:
            c, x1, y1, x2, y2 = D.to_xyxy(b, w, h)
            col = QColor(PALETTE[c % len(PALETTE)])
            bgr = (col.blue(), col.green(), col.red())
            p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
            if dash:
                n = max(2, int((abs(x2 - x1) + abs(y2 - y1)) / 12))
                for k in range(0, n, 2):
                    a, e = k / n, min(1.0, (k + 1) / n)
                    for (xa, ya), (xb, yb) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)),
                                               ((x1, y2), (x1, y1))):
                        cv2.line(out, (int(xa + (xb - xa) * a), int(ya + (yb - ya) * a)),
                                 (int(xa + (xb - xa) * e), int(ya + (yb - ya) * e)), bgr, t)
                if len(b) > 5:
                    cv2.putText(out, f"{b[5]:.2f}", (p1[0] + 2, max(12, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4 * t, bgr, t)
            else:
                cv2.rectangle(out, p1, p2, bgr, t)
    return out


class ImageView(QWidget):
    """A picture scaled to fit (with its boxes already drawn in)."""

    def __init__(self, text: str = "Resim seçin"):
        super().__init__()
        self.setMinimumSize(240, 180)
        self.pixmap: QPixmap | None = None
        self.text = text

    def set_image(self, img: np.ndarray | None):
        self.pixmap = QPixmap.fromImage(to_qimage(img)) if img is not None else None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#18191b"))
        if not self.pixmap or self.pixmap.isNull():
            p.setPen(QColor("#8b8e94"))
            p.drawText(self.rect(), Qt.AlignCenter, self.text)
            return
        pw, ph = self.pixmap.width(), self.pixmap.height()
        s = min(self.width() / pw, self.height() / ph)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(QRectF((self.width() - pw * s) / 2, (self.height() - ph * s) / 2, pw * s, ph * s),
                     self.pixmap, QRectF(self.pixmap.rect()))


class ClassBalanceChart(QWidget):
    """Horizontal bars: boxes per class (drawn with QPainter)."""

    def __init__(self):
        super().__init__()
        self.names: list[str] = []
        self.counts: list[int] = []
        self.setMinimumHeight(120)

    def set_data(self, names, counts):
        self.names, self.counts = list(names), list(counts)
        self.setMinimumHeight(max(120, 26 * len(self.names) + 20))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1f22"))
        if not self.names:
            p.setPen(QColor("#8b8e94"))
            p.drawText(self.rect(), Qt.AlignCenter, "Sınıf yok")
            return
        fm = p.fontMetrics()
        label_w = min(self.width() // 3, max(fm.horizontalAdvance(n) for n in self.names) + 16)
        top = max(self.counts) if self.counts and max(self.counts) > 0 else 1
        bar_h = max(10, min(22, (self.height() - 20) // max(1, len(self.names)) - 6))
        y = 10
        avail = self.width() - label_w - 70
        for i, (name, n) in enumerate(zip(self.names, self.counts)):
            p.setPen(QColor("#d6d7da"))
            p.drawText(QRectF(4, y, label_w - 8, bar_h), Qt.AlignRight | Qt.AlignVCenter, name)
            w = avail * n / top
            col = QColor(PALETTE[i % len(PALETTE)])
            p.fillRect(QRectF(label_w, y, max(1.0 if n else 0.0, w), bar_h), col)
            p.setPen(QColor("#d6d7da") if n else QColor("#ff8a80"))
            p.drawText(QRectF(label_w + w + 6, y, 64, bar_h), Qt.AlignLeft | Qt.AlignVCenter, str(n))
            y += bar_h + 6


class DatasetManager(QDialog):
    """The window. Every action is a method with explicit arguments (tests call them without dialogs); with
    ``quiet`` set, messages go to ``last_message`` instead of a message box and confirmations are answered yes."""

    def __init__(self, parent=None, folder: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Veri Seti Yöneticisi")
        self.resize(1250, 820)
        self.ds: D.Dataset | None = None
        self.st: dict | None = None
        self.quiet = False
        self.last_message = ""
        self._thumb_queue: list = []
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setInterval(0)
        self._thumb_timer.timeout.connect(self._load_some_thumbs)
        self._aug_rng = np.random.default_rng(0)
        self._labeler = None

        # top bar
        self.path_label = QLabel("Veri seti açılmadı")
        self.path_label.setObjectName("hint")
        top = QHBoxLayout()
        for text, fn in (("Aç…", lambda: self.open_dataset()), ("Yeni…", lambda: self.new_dataset()),
                         ("Resim ekle…", lambda: self.add_images()),
                         ("COCO içe aktar…", lambda: self.import_dataset("coco")),
                         ("VOC içe aktar…", lambda: self.import_dataset("voc")), ("Yenile", self.refresh)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            top.addWidget(b)
        top.addWidget(self.path_label, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_overview(), "Özet")
        self.tabs.addTab(self._build_images(), "Resimler")
        self.tabs.addTab(self._build_classes(), "Sınıflar")
        self.tabs.addTab(self._build_split(), "Bölme ve dışa aktarma")
        self.tabs.addTab(self._build_augment(), "Çoğaltma")
        self.tabs.addTab(self._build_autolabel(), "Otomatik etiketle")
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.tabs, 1)
        for b in self.findChildren(QPushButton):     # Enter in a text box must not press a button
            b.setAutoDefault(False)
            b.setDefault(False)
        if folder:
            self.open_dataset(folder)

    # ---- messages ----
    def _msg(self, title: str, text: str, warn: bool = False):
        self.last_message = text
        if self.quiet:
            return
        (QMessageBox.warning if warn else QMessageBox.information)(self, title, text)

    def _confirm(self, title: str, text: str) -> bool:
        if self.quiet:
            return True
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def _need(self) -> bool:
        if self.ds is None:
            self._msg("Veri seti", "Önce bir veri seti klasörü açın.", True)
            return False
        return True

    def _run(self, title: str, fn, *args, **kw):
        """Run fn(..., progress=cb) with a progress dialog; ValueError/OSError → Turkish message, returns None."""
        prog = None
        if not self.quiet:
            prog = QProgressDialog(title, "Durdur", 0, 1000, self)
            prog.setWindowModality(Qt.WindowModal)
            prog.setMinimumDuration(300)

        def cb(i, n=None):
            frac = i if n is None else (i / max(1, n))
            if prog is not None:
                prog.setValue(int(1000 * max(0.0, min(1.0, frac))))
                QApplication.processEvents()
                return not prog.wasCanceled()
            return True
        try:
            return fn(*args, progress=cb, **kw)
        except (ValueError, OSError) as exc:
            self._msg(title, str(exc), True)
            return None
        except Exception as exc:  # noqa: BLE001 - a tool window reports, it does not crash the editor
            self._msg(title, f"Beklenmeyen hata: {type(exc).__name__}: {exc}", True)
            return None
        finally:
            if prog is not None:
                prog.setValue(1000)
                prog.deleteLater()

    # ---- building the tabs ----
    def _build_overview(self) -> QWidget:
        w = QWidget()
        self.summary = QLabel("Bir veri seti klasörü açın (images/ + labels/, Roboflow dışa aktarımı veya resim + "
                              "txt aynı klasörde) ya da COCO / VOC içe aktarın.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.chart = ClassBalanceChart()
        self.problems = QListWidget()
        self.problems.itemDoubleClicked.connect(lambda it: self.open_in_labeler(it.data(Qt.UserRole)))
        left = QVBoxLayout()
        left.addWidget(self.summary)
        left.addWidget(QLabel("Sınıf dengesi (kutu sayısı)"))
        left.addWidget(self.chart, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("Sorunlar (çift tık: etiketleme aracında aç)"))
        right.addWidget(self.problems, 1)
        lay = QHBoxLayout(w)
        lay.addLayout(left, 3)
        lay.addLayout(right, 2)
        return w

    def _build_images(self) -> QWidget:
        w = QWidget()
        self.filter_combo = QComboBox()
        self.filter_combo.currentIndexChanged.connect(lambda _i: self.fill_grid())
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setIconSize(QSize(THUMB, THUMB))
        self.grid.setGridSize(QSize(THUMB + 16, THUMB + 34))
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.itemDoubleClicked.connect(lambda it: self.open_in_labeler(it.data(Qt.UserRole)))
        self.grid_info = QLabel()
        open_btn = QPushButton("Etiketleme aracında aç")
        open_btn.clicked.connect(lambda: self.open_in_labeler(self._grid_rel()))
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Göster:"))
        bar.addWidget(self.filter_combo)
        bar.addWidget(self.grid_info, 1)
        bar.addWidget(open_btn)
        lay = QVBoxLayout(w)
        lay.addLayout(bar)
        lay.addWidget(self.grid, 1)
        return w

    def _build_classes(self) -> QWidget:
        w = QWidget()
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.ExtendedSelection)
        btns = QVBoxLayout()
        for text, fn in (("Yeniden adlandır…", lambda: self.rename_class()),
                         ("Seçilenleri birleştir…", lambda: self.merge_classes()),
                         ("Sil (kutularıyla)…", lambda: self.delete_class()),
                         ("Sınıf ekle…", lambda: self.add_class()),
                         ("Yedekten geri yükle…", lambda: self.restore_backup())):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        hint = QLabel("Etiket dosyaları değiştirilmeden önce .vf_yedek/ klasörüne yedeklenir.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        btns.addWidget(hint)
        lay = QHBoxLayout(w)
        lay.addWidget(self.class_list, 1)
        lay.addLayout(btns)
        return w

    def _build_split(self) -> QWidget:
        w = QWidget()
        box = QGroupBox("Eğitim / doğrulama / test bölmesi (sınıf dengeli)")
        f = QFormLayout(box)
        self.r_train, self.r_val, self.r_test = (QDoubleSpinBox() for _ in range(3))
        for sp, v in ((self.r_train, 0.7), (self.r_val, 0.2), (self.r_test, 0.1)):
            sp.setRange(0, 1)
            sp.setSingleStep(0.05)
            sp.setDecimals(2)
            sp.setValue(v)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.labelled_only = QCheckBox("Yalnız etiketli resimler")
        split_btn = QPushButton("Böl")
        split_btn.clicked.connect(lambda: self.do_split())
        self.split_info = QLabel()
        f.addRow("Eğitim", self.r_train)
        f.addRow("Doğrulama", self.r_val)
        f.addRow("Test", self.r_test)
        f.addRow("Tohum (aynı tohum = aynı bölme)", self.seed)
        f.addRow(self.labelled_only)
        f.addRow(split_btn, self.split_info)
        ex = QGroupBox("Dışa aktar")
        g = QFormLayout(ex)
        self.export_fmt = QComboBox()
        self.export_fmt.addItems(["YOLO (images/ + labels/ + data.yaml)", "COCO (annotations.json)",
                                  "Pascal VOC (Annotations/*.xml)"])
        exp_btn = QPushButton("Boş bir klasöre dışa aktar…")
        exp_btn.clicked.connect(lambda: self.export())
        train_btn = QPushButton("YOLO modeli eğit…")
        train_btn.setToolTip("YOLO Modeli Eğit penceresini bu veri setinin data.yaml dosyasıyla açar")
        train_btn.clicked.connect(self.open_trainer)
        g.addRow("Biçim", self.export_fmt)
        g.addRow(exp_btn)
        g.addRow(train_btn)
        lay = QVBoxLayout(w)
        lay.addWidget(box)
        lay.addWidget(ex)
        lay.addStretch(1)
        return w

    def _build_augment(self) -> QWidget:
        w = QWidget()
        self.aug_checks = {}
        ops = QGroupBox("İşlemler (kutular da dönüştürülür)")
        grid = QGridLayout(ops)
        for i, op in enumerate(D.AUG_OPS):
            cb = QCheckBox(D.AUG_NAMES[op])
            cb.setChecked(op in ("flip_h", "rotate", "brightness"))
            self.aug_checks[op] = cb
            grid.addWidget(cb, i // 4, i % 4)
        lim = QFormLayout()
        self.max_angle = QDoubleSpinBox()
        self.max_angle.setRange(0, 45)
        self.max_angle.setValue(10)
        self.max_bright = QDoubleSpinBox()
        self.max_bright.setRange(0, 120)
        self.max_bright.setValue(40)
        self.max_noise = QDoubleSpinBox()
        self.max_noise.setRange(2, 50)
        self.max_noise.setValue(12)
        self.copies = QSpinBox()
        self.copies.setRange(1, 50)
        self.copies.setValue(3)
        self.aug_train_only = QCheckBox("Yalnız eğitim (train) resimleri")
        self.aug_train_only.setChecked(True)
        lim.addRow("En büyük açı (°)", self.max_angle)
        lim.addRow("En büyük parlaklık farkı", self.max_bright)
        lim.addRow("En büyük gürültü", self.max_noise)
        lim.addRow("Resim başına kopya (N)", self.copies)
        lim.addRow(self.aug_train_only)
        prev_btn = QPushButton("Yeni önizleme")
        prev_btn.clicked.connect(lambda: self.preview_augment())
        save_btn = QPushButton("Çoğalt ve kaydet")
        save_btn.clicked.connect(lambda: self.augment_and_save())
        self.aug_before, self.aug_after = ImageView("Önizleme için etiketli resim gerekli"), ImageView("")
        views = QHBoxLayout()
        views.addWidget(self.aug_before)
        views.addWidget(self.aug_after)
        self.aug_info = QLabel()
        bar = QHBoxLayout()
        bar.addWidget(prev_btn)
        bar.addWidget(save_btn)
        bar.addWidget(self.aug_info, 1)
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(ops, 2)
        top.addLayout(lim, 1)
        lay.addLayout(top)
        lay.addLayout(views, 1)
        lay.addLayout(bar)
        return w

    def _build_autolabel(self) -> QWidget:
        w = QWidget()
        model_box = QGroupBox("Model")
        f = QFormLayout(model_box)
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(".vfdet (Az Örnekle Nesne Bul) veya .onnx (YOLO)")
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse_model)
        row = QHBoxLayout()
        row.addWidget(self.model_edit, 1)
        row.addWidget(browse)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 1.0)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.5)
        run_btn = QPushButton("Etiketsizleri otomatik etiketle")
        run_btn.clicked.connect(lambda: self.run_autolabel())
        self.fs_images = QSpinBox()
        self.fs_images.setRange(1, 200)
        self.fs_images.setValue(12)
        self.fs_angle = QDoubleSpinBox()
        self.fs_angle.setRange(0, 180)
        self.fs_angle.setValue(30)
        fs_btn = QPushButton("Mevcut etiketlerden az örnekle model eğit ve etiketle")
        fs_btn.setToolTip("Birkaç etiketli resimden .vfdet modeli öğrenir, kalanları onunla önerir. Önerileri kabul "
                          "edip tekrar eğiterek (aktif öğrenme) hızla büyütün.")
        fs_btn.clicked.connect(lambda: self.train_and_label())
        f.addRow("Model dosyası", row)
        f.addRow("Güven eşiği", self.conf)
        f.addRow(run_btn)
        f.addRow("Az örnekle: en çok resim", self.fs_images)
        f.addRow("Az örnekle: açı aralığı ±°", self.fs_angle)
        f.addRow(fs_btn)
        self.al_info = QLabel()
        self.al_info.setWordWrap(True)
        f.addRow(self.al_info)

        self.review = QListWidget()
        self.review.currentRowChanged.connect(self._show_review)
        self.review_view = ImageView("Onay bekleyen öneri yok")
        self.accept_min = QDoubleSpinBox()
        self.accept_min.setRange(0, 1)
        self.accept_min.setSingleStep(0.05)
        btns = QHBoxLayout()
        for text, fn in (("Kabul (Enter)", lambda: self.accept_current()), ("Reddet (Del)", lambda: self.reject_current()),
                         ("Tümünü kabul", lambda: self.accept_all()), ("Tümünü reddet", lambda: self.reject_all()),
                         ("Etiketleme aracında aç", lambda: self.open_in_labeler(self._review_rel()))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addWidget(QLabel("Tümünü kabulde en düşük güven:"))
        btns.addWidget(self.accept_min)
        for key, fn in ((Qt.Key_Return, self.accept_current), (Qt.Key_Enter, self.accept_current),
                        (Qt.Key_Delete, self.reject_current)):
            sc = QShortcut(QKeySequence(key), self.review, fn)
            sc.setContext(Qt.WidgetWithChildrenShortcut)          # only while the review list has the focus
        rv = QWidget()
        rl = QVBoxLayout(rv)
        rl.addWidget(QLabel("İnceleme (en emin olunmayan önce; kesikli kutular öneridir)"))
        split = QSplitter()
        split.addWidget(self.review)
        split.addWidget(self.review_view)
        split.setSizes([300, 700])
        rl.addWidget(split, 1)
        rl.addLayout(btns)
        lay = QVBoxLayout(w)
        lay.addWidget(model_box)
        lay.addWidget(rv, 1)
        return w

    # ---- opening ----
    def open_dataset(self, folder: str | None = None) -> bool:
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Veri seti klasörü seç")
            if not folder:
                return False
        try:
            self.ds = D.Dataset(folder)
        except (ValueError, OSError) as exc:
            self._msg("Veri seti", str(exc), True)
            return False
        self.path_label.setText(f"{self.ds.root}  ·  düzen: {self.ds.layout}")
        self.setWindowTitle(f"Veri Seti Yöneticisi — {self.ds.root.name}")
        self.refresh()
        return True

    def new_dataset(self, folder: str | None = None, classes: list[str] | None = None) -> bool:
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Yeni veri seti için (boş) klasör seç")
            if not folder:
                return False
        if classes is None and not self.quiet:
            text, ok = QInputDialog.getText(self, "Sınıflar", "Sınıf adları (virgülle):", text="nesne")
            if not ok:
                return False
            classes = [c for c in (D.clean_name(t) for t in text.split(",")) if c]
        ds = D.Dataset(folder, create=True)
        if classes:
            ds.classes = list(dict.fromkeys(classes))
            ds.save_meta()
        return self.open_dataset(folder)

    def add_images(self, paths=None) -> int:
        if not self._need():
            return 0
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Resim ekle", "", "Resimler (*.png *.jpg *.jpeg *.bmp "
                                                    "*.tif *.tiff *.webp)")
            if not paths:
                return 0
        try:
            n = self.ds.add_images(paths)
        except (ValueError, OSError) as exc:
            self._msg("Resim ekle", str(exc), True)
            return 0
        self.refresh()
        return n

    def import_dataset(self, kind: str, src: str | None = None, out: str | None = None) -> bool:
        if not src:
            if kind == "coco":
                src, _ = QFileDialog.getOpenFileName(self, "COCO json dosyası", "", "COCO (*.json)")
            else:
                src = QFileDialog.getExistingDirectory(self, "Pascal VOC klasörü (Annotations/ + JPEGImages/)")
            if not src:
                return False
        if not out:
            out = QFileDialog.getExistingDirectory(self, "Yeni YOLO veri seti için boş klasör")
            if not out:
                return False
        fn = D.import_coco if kind == "coco" else D.import_voc
        ds = self._run("İçe aktarılıyor…", fn, src, out)
        if ds is None:
            return False
        if getattr(ds, "missing", 0):
            self._msg("İçe aktarma", f"{ds.missing} resim bulunamadı ve atlandı.", True)
        return self.open_dataset(str(ds.root))

    # ---- overview / grid ----
    def refresh(self):
        if self.ds is None:
            return
        self.ds.reload()
        self.st = self._run("Veri seti inceleniyor…", self.ds.stats) or self.ds.stats()
        st = self.st
        probs: dict[str, int] = {}
        for p in st["problems"]:
            probs[p["kind"]] = probs.get(p["kind"], 0) + 1
        size_txt = ", ".join(f"{k} ({v})" for k, v in list(st["sizes"].items())[:4]) or "—"
        bal = "—" if st["balance"] is None else f"{st['balance']:.2f}"
        lines = [f"<b>{st['images']}</b> resim · <b>{st['labelled']}</b> etiketli · <b>{st['unlabelled']}</b> "
                 f"etiketsiz · <b>{st['empty']}</b> boş (nesnesiz) · <b>{st['boxes']}</b> kutu",
                 f"Sınıf dengesi (en az / en çok): {bal}", f"Resim boyutları: {size_txt}"]
        if any(st["splits"].values()):
            lines.append("Bölme: " + ", ".join(f"{k} {v}" for k, v in st["splits"].items()))
        if st["suggested"]:
            lines.append(f"<b>{st['suggested']}</b> resimde onay bekleyen öneri var (Otomatik etiketle sekmesi)")
        lines.append("Sorunlar: " + (", ".join(f"{D.PROBLEM_NAMES.get(k, k)} {v}" for k, v in probs.items())
                                     or "yok"))
        self.summary.setText("<br>".join(lines))
        names = list(st["classes"]) + [f"(tanımsız {c})" for c in sorted(st["unknown_classes"])]
        counts = list(st["per_class"]) + [st["unknown_classes"][c] for c in sorted(st["unknown_classes"])]
        self.chart.set_data(names, counts)
        self.problems.clear()
        for p in st["problems"]:
            it = QListWidgetItem(f"{D.PROBLEM_NAMES.get(p['kind'], p['kind'])}: {p['image']} — {p['detail']}")
            it.setData(Qt.UserRole, p["image"])
            self.problems.addItem(it)
        # filters
        cur = self.filter_combo.currentData()
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        for text, key in (("Tümü", "all"), ("Etiketsiz", "unlabelled"), ("Sorunlu", "problems"),
                          ("Boş (nesnesiz)", "empty"), ("Önerili (onay bekleyen)", "suggested")):
            self.filter_combo.addItem(text, key)
        for i, n in enumerate(self.ds.classes):
            self.filter_combo.addItem(f"Sınıf: {n}", f"class:{i}")
        k = self.filter_combo.findData(cur)
        self.filter_combo.setCurrentIndex(max(0, k))
        self.filter_combo.blockSignals(False)
        self.fill_grid()
        self.fill_classes()
        self.fill_review()

    def fill_grid(self):
        self._thumb_timer.stop()
        self.grid.clear()
        if self.ds is None:
            return
        key = self.filter_combo.currentData() or "all"
        items = self.ds.filter(key, self.st)
        shown = items[:MAX_THUMBS]
        bad = D.Dataset.problem_images(self.ds, self.st) if self.st else set()
        ph = QPixmap(THUMB, THUMB)
        ph.fill(QColor("#2a2b2e"))
        for it in shown:
            li = QListWidgetItem(QIcon(ph), it.image.name)
            li.setData(Qt.UserRole, it.rel)
            tip = it.rel + ("" if it.label.exists() else "\n(etiketsiz)") + ("\n⚠ sorunlu" if it.rel in bad else "")
            li.setToolTip(tip)
            if it.rel in bad:
                li.setForeground(QColor("#ffb74d"))
            elif not it.label.exists():
                li.setForeground(QColor("#8b8e94"))
            self.grid.addItem(li)
        self.grid_info.setText(f"{len(items)} resim" + (f" (ilk {MAX_THUMBS} gösteriliyor)" if len(items) > len(shown)
                                                       else ""))
        self._thumb_queue = list(range(len(shown)))
        if shown:
            self._thumb_timer.start()

    def _load_some_thumbs(self):
        import cv2
        for _ in range(12):
            if not self._thumb_queue or self.ds is None:
                self._thumb_timer.stop()
                return
            row = self._thumb_queue.pop(0)
            li = self.grid.item(row)
            if li is None:
                continue
            it = self.ds.find(li.data(Qt.UserRole))
            img = D.imread(it.image) if it else None
            if img is None:
                continue
            s = THUMB / max(img.shape[:2])
            small = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                               interpolation=cv2.INTER_AREA) if s < 1 else img
            small = _draw_boxes(small, self.ds.boxes(it), self.ds.suggestions(it))
            li.setIcon(QIcon(QPixmap.fromImage(to_qimage(small))))

    def _grid_rel(self):
        it = self.grid.currentItem()
        return it.data(Qt.UserRole) if it else None

    def open_in_labeler(self, rel):
        """Quick open: the labelling tool on the picture's folder, at that picture."""
        if not rel or self.ds is None:
            return None
        it = self.ds.find(rel)
        if it is None:
            return None
        from .tools import LabelingDialog
        folder = self.ds.root if (self.ds.layout == "yolo" and it.image.parent == self.ds.root / "images") \
            else it.image.parent
        dlg = LabelingDialog(self, str(folder), str(it.image))
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.finished.connect(lambda _r: self.refresh())
        dlg.show()
        self._labeler = dlg
        return dlg

    # ---- classes ----
    def fill_classes(self):
        self.class_list.clear()
        if self.ds is None:
            return
        st = self.st or {}
        for i, n in enumerate(self.ds.classes):
            boxes = st.get("per_class", [0] * len(self.ds.classes))
            imgs = st.get("images_per_class", [0] * len(self.ds.classes))
            li = QListWidgetItem(f"{i}. {n}  —  {boxes[i] if i < len(boxes) else 0} kutu, "
                                 f"{imgs[i] if i < len(imgs) else 0} resim")
            li.setForeground(QColor(PALETTE[i % len(PALETTE)]))
            li.setData(Qt.UserRole, i)
            self.class_list.addItem(li)

    def _selected_classes(self) -> list[int]:
        return sorted(li.data(Qt.UserRole) for li in self.class_list.selectedItems())

    def _class_edit(self, title, fn, *args):
        try:
            bk = fn(*args)
        except (ValueError, OSError) as exc:
            self._msg(title, str(exc), True)
            return None
        self.refresh()
        self.last_message = f"Yedek: {bk}"
        return bk

    def rename_class(self, index: int | None = None, name: str | None = None):
        if not self._need():
            return None
        if index is None:
            sel = self._selected_classes()
            if len(sel) != 1:
                self._msg("Yeniden adlandır", "Bir sınıf seçin.", True)
                return None
            index = sel[0]
        if name is None:
            name, ok = QInputDialog.getText(self, "Yeniden adlandır", "Yeni ad:", text=self.ds.classes[index])
            if not ok:
                return None
        return self._class_edit("Yeniden adlandır", self.ds.rename_class, index, name)

    def merge_classes(self, sources: list[int] | None = None, target: int | None = None):
        if not self._need():
            return None
        if sources is None:
            sources = self._selected_classes()
            if len(sources) < 2:
                self._msg("Birleştir", "Birleştirilecek en az iki sınıf seçin (Ctrl ile).", True)
                return None
        if target is None:
            names = [self.ds.classes[i] for i in sources]
            name, ok = QInputDialog.getItem(self, "Birleştir", "Hangi sınıfta toplansın?", names, 0, False)
            if not ok:
                return None
            target = sources[names.index(name)]
        if not self._confirm("Birleştir", f"{len(sources)} sınıfın kutuları '{self.ds.classes[target]}' sınıfına "
                             "taşınacak. Devam edilsin mi? (Önce yedek alınır.)"):
            return None
        return self._class_edit("Birleştir", self.ds.merge_classes, sources, target)

    def delete_class(self, index: int | None = None):
        if not self._need():
            return None
        if index is None:
            sel = self._selected_classes()
            if len(sel) != 1:
                self._msg("Sil", "Bir sınıf seçin.", True)
                return None
            index = sel[0]
        n = (self.st or {}).get("per_class", [0] * (index + 1))
        if not self._confirm("Sınıfı sil", f"'{self.ds.classes[index]}' sınıfı ve {n[index] if index < len(n) else 0}"
                             " kutusu silinecek; üstteki sınıf numaraları bir azalır. Devam edilsin mi? "
                             "(Önce yedek alınır.)"):
            return None
        return self._class_edit("Sil", self.ds.delete_class, index)

    def add_class(self, name: str | None = None):
        if not self._need():
            return
        if name is None:
            name, ok = QInputDialog.getText(self, "Sınıf ekle", "Sınıf adı:")
            if not ok:
                return
        name = D.clean_name(name)
        if not name:
            return
        self.ds.class_index(name)
        self.refresh()

    def restore_backup(self, backup: str | None = None):
        if not self._need():
            return 0
        if backup is None:
            bks = self.ds.backups()
            if not bks:
                self._msg("Yedek", "Yedek yok.")
                return 0
            name, ok = QInputDialog.getItem(self, "Yedekten geri yükle", "Yedek:", [b.name for b in bks], 0, False)
            if not ok:
                return 0
            backup = str(next(b for b in bks if b.name == name))
        if not self._confirm("Yedekten geri yükle", "Etiketler ve sınıf listesi bu yedekteki hâline döner. "
                             "Devam edilsin mi?"):
            return 0
        self.ds.backup("geri_yukleme_oncesi")
        n = self.ds.restore(pathlib.Path(backup))
        self.refresh()
        return n

    # ---- split / export ----
    def do_split(self, ratios=None, seed: int | None = None):
        if not self._need():
            return None
        ratios = ratios or (self.r_train.value(), self.r_val.value(), self.r_test.value())
        seed = self.seed.value() if seed is None else seed
        try:
            out = self.ds.split(ratios, seed, self.labelled_only.isChecked())
        except ValueError as exc:
            self._msg("Bölme", str(exc), True)
            return None
        self.split_info.setText("Bölündü: " + ", ".join(f"{k} {len(v)}" for k, v in out.items())
                                + " · train.txt / val.txt / test.txt ve data.yaml yazıldı")
        self.refresh()
        return out

    def export(self, fmt: str | None = None, out: str | None = None):
        if not self._need():
            return None
        fmt = fmt or ("yolo", "coco", "voc")[self.export_fmt.currentIndex()]
        if not out:
            out = QFileDialog.getExistingDirectory(self, "Dışa aktarma için boş klasör")
            if not out:
                return None
        fn = {"yolo": D.export_yolo, "coco": D.export_coco, "voc": D.export_voc}[fmt]
        res = self._run("Dışa aktarılıyor…", fn, self.ds, out)
        if res is not None:
            self._msg("Dışa aktar", f"Dışa aktarıldı: {res}")
        return res

    def open_trainer(self):
        if not self._need():
            return None
        self.ds.save_meta()
        from .tools import TrainDialog
        dlg = TrainDialog(self)
        dlg.data.setText(str(self.ds.root / "data.yaml"))
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.show()
        return dlg

    # ---- augmentation ----
    def _aug_enabled(self) -> list[str]:
        return [k for k, cb in self.aug_checks.items() if cb.isChecked()]

    def _aug_limits(self) -> dict:
        return {"max_angle": self.max_angle.value(), "max_brightness": self.max_bright.value(),
                "max_noise": self.max_noise.value()}

    def preview_augment(self, rel: str | None = None):
        if not self._need():
            return None
        it = self.ds.find(rel or self._grid_rel() or "") if (rel or self._grid_rel()) else None
        if it is None or not it.label.exists():
            it = next((i for i in self.ds.items if i.label.exists()), None)
        if it is None:
            self.aug_info.setText("Önizleme için etiketli resim yok.")
            return None
        img = D.imread(it.image)
        if img is None:
            return None
        h, w = img.shape[:2]
        boxes = self.ds.boxes(it)
        try:
            ops = D.random_ops(self._aug_rng, self._aug_enabled(), **self._aug_limits())
            aimg, abx = D.augment(img, [D.to_xyxy(b, w, h) for b in boxes], ops, self._aug_rng)
        except ValueError as exc:
            self.aug_info.setText(str(exc))
            return None
        ah, aw = aimg.shape[:2]
        self.aug_before.set_image(_draw_boxes(img, boxes))
        self.aug_after.set_image(_draw_boxes(aimg, [D.to_yolo(b, aw, ah) for b in abx]))
        desc = ", ".join(f"{D.AUG_NAMES[k]}" + ("" if v is True else f" {v:.1f}" if isinstance(v, float) else f" {v}")
                         for k, v in ops.items())
        self.aug_info.setText(f"{it.image.name}: {desc}")
        return ops, abx

    def augment_and_save(self, copies: int | None = None, seed: int | None = None):
        if not self._need():
            return 0
        copies = copies or self.copies.value()
        items = [it for it in self.ds.items if it.label.exists() and "_aug" not in it.image.stem]
        if self.aug_train_only.isChecked() and any(it.split for it in items):
            items = [it for it in items if it.split in ("train", "")]
        if not self._confirm("Çoğalt", f"{len(items)} etiketli resmin her birinden {copies} kopya kaydedilecek. "
                             "Devam edilsin mi?"):
            return 0
        n = self._run("Çoğaltılıyor…", D.augment_dataset, self.ds, items, copies, self._aug_enabled(),
                      self.seed.value() if seed is None else seed, **self._aug_limits())
        if n is None:
            return 0
        self.aug_info.setText(f"{n} yeni resim kaydedildi.")
        self.refresh()
        return n

    # ---- auto-label ----
    def _browse_model(self):
        start = str(self.ds.root / D.MODEL_DIR) if self.ds and (self.ds.root / D.MODEL_DIR).is_dir() else \
            ("models" if pathlib.Path("models").is_dir() else "")
        path, _ = QFileDialog.getOpenFileName(self, "Model seç", start,
                                              "Modeller (*.vfdet *.onnx);;Az örnekle (*.vfdet);;YOLO ONNX (*.onnx)")
        if path:
            self.model_edit.setText(path)

    def run_autolabel(self, model: str | None = None, conf: float | None = None):
        if not self._need():
            return None
        model = model or self.model_edit.text().strip()
        if not model:
            self._msg("Otomatik etiketle", "Önce bir model seçin (.vfdet veya .onnx) ya da az örnekle model eğitin.",
                      True)
            return None
        conf = self.conf.value() if conf is None else conf
        try:
            pred = D.Predictor(model, conf)
        except ValueError as exc:
            self._msg("Otomatik etiketle", str(exc), True)
            return None
        try:
            res = self._run("Model etiketliyor…", D.autolabel, self.ds, pred, conf)
        finally:
            pred.close()
        if res is None:
            return None
        self.al_info.setText(f"{res['images']} resme {res['boxes']} kutu önerildi (cihaz: {res['device']}). "
                             "Aşağıda en emin olunmayandan başlayarak inceleyin.")
        self.refresh()
        self.tabs.setCurrentIndex(5)
        return res

    def train_and_label(self, conf: float | None = None):
        if not self._need():
            return None
        path = self._run("Az örnekle model eğitiliyor…", D.train_fewshot, self.ds, None, self.fs_images.value(),
                         self.fs_angle.value())
        if path is None:
            return None
        self.model_edit.setText(str(path))
        return self.run_autolabel(str(path), conf)

    def fill_review(self):
        cur = self._review_rel()
        self.review.blockSignals(True)
        self.review.clear()
        if self.ds is not None:
            for it, bx, lo in self.ds.review_queue():
                li = QListWidgetItem(f"{it.image.name} — {len(bx)} kutu" +
                                     (f", en düşük güven {lo:.2f}" if bx else " (model bir şey bulmadı)"))
                li.setData(Qt.UserRole, it.rel)
                self.review.addItem(li)
        self.review.blockSignals(False)
        row = 0
        for i in range(self.review.count()):
            if self.review.item(i).data(Qt.UserRole) == cur:
                row = i
        if self.review.count():
            self.review.setCurrentRow(row)
            self._show_review(row)
        else:
            self.review_view.set_image(None)

    def _review_rel(self):
        it = self.review.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _show_review(self, row):
        li = self.review.item(row) if row is not None and row >= 0 else None
        it = self.ds.find(li.data(Qt.UserRole)) if (li and self.ds) else None
        img = D.imread(it.image) if it else None
        self.review_view.set_image(None if img is None else _draw_boxes(img, self.ds.boxes(it),
                                                                        self.ds.suggestions(it)))

    def _after_review(self, row):
        self.st = self.ds.stats()
        self.fill_review()
        if self.review.count():
            self.review.setCurrentRow(min(row, self.review.count() - 1))
        self.refresh_summary_only()

    def refresh_summary_only(self):
        if self.st:
            self.chart.set_data(self.st["classes"], self.st["per_class"])

    def accept_current(self):
        rel = self._review_rel()
        if not rel or self.ds is None:
            return 0
        row = self.review.currentRow()
        n = self.ds.accept(self.ds.find(rel))
        self._after_review(row)
        return n

    def reject_current(self):
        rel = self._review_rel()
        if not rel or self.ds is None:
            return
        row = self.review.currentRow()
        self.ds.reject(self.ds.find(rel))
        self._after_review(row)

    def accept_all(self, min_conf: float | None = None):
        if not self._need():
            return None
        min_conf = self.accept_min.value() if min_conf is None else min_conf
        if not self._confirm("Tümünü kabul", f"Onay bekleyen bütün öneriler (güven ≥ {min_conf:.2f}) etikete "
                             "çevrilecek. Devam edilsin mi?"):
            return None
        res = self.ds.accept_all(min_conf)
        self.refresh()
        return res

    def reject_all(self):
        if not self._need():
            return 0
        if not self._confirm("Tümünü reddet", "Onay bekleyen bütün öneriler silinecek. Devam edilsin mi?"):
            return 0
        n = self.ds.reject_all()
        self.refresh()
        return n

    def closeEvent(self, e):
        self._thumb_timer.stop()
        self._thumb_queue = []
        super().closeEvent(e)
