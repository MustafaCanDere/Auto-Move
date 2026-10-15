"""Side panels: block palette, properties editor, live preview with pixel inspector, log."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
                               QSizePolicy, QSlider, QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import Node, Param, by_category
from .scene import MIME
from .theme import CATEGORY_COLORS, DEFAULT_CATEGORY_COLOR
from .worker import to_qimage


_FOLD = str.maketrans("İIıŞşĞğÜüÖöÇç", "iiissgguuoocc")


def search_fold(text: str) -> str:
    """Search key that ignores case and Turkish letters: "izle" finds "Klasörü İzle", "kirp" finds "Kırp"."""
    return text.translate(_FOLD).lower()


class _Tree(QTreeWidget):
    def mimeData(self, items):
        md = QMimeData()
        tn = items[0].data(0, Qt.UserRole)
        if tn:
            md.setData(MIME, tn.encode())
        return md


class Palette(QWidget):
    add_requested = Signal(str)
    create_custom_requested = Signal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.search = QLineEdit(placeholderText="Blok ara…  (Ctrl+F)")
        self.btn_create_custom = QPushButton("🎨 + Özel Blok Oluştur…")
        self.btn_create_custom.setToolTip("Kendi özel OpenCV / Python bloğunuzu hazırlayın (Ctrl+Shift+B)")
        self.btn_create_custom.setStyleSheet("padding: 5px; font-weight: bold; color: #4fc3f7;")
        self.btn_create_custom.clicked.connect(self.create_custom_requested.emit)
        self.tree = _Tree()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.itemDoubleClicked.connect(self._double)
        hint = QLabel("Sürükleyip bırakın veya çift tıklayın")
        hint.setObjectName("hint")
        lay.addWidget(self.search)
        lay.addWidget(self.btn_create_custom)
        lay.addWidget(self.tree)
        lay.addWidget(hint)
        self.search.textChanged.connect(self._filter)
        self.rebuild()


    def rebuild(self):
        self.tree.clear()
        for cat, classes in by_category().items():
            top = QTreeWidgetItem([cat])
            top.setForeground(0, QColor(CATEGORY_COLORS.get(cat, DEFAULT_CATEGORY_COLOR)).lighter(160))
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            top.setFlags(Qt.ItemIsEnabled)
            for cls in classes:
                it = QTreeWidgetItem([cls.title])
                it.setData(0, Qt.UserRole, cls.type_name)
                it.setToolTip(0, cls.description or cls.title)
                top.addChild(it)
            self.tree.addTopLevelItem(top)
        self.tree.expandAll()

    def _filter(self, text):
        text = search_fold(text.strip())
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            visible = 0
            for j in range(top.childCount()):
                ch = top.child(j)
                hit = not text or text in search_fold(ch.text(0)) or text in search_fold(ch.toolTip(0))
                ch.setHidden(not hit)
                visible += hit
            top.setHidden(visible == 0)

    def _double(self, item, _col):
        tn = item.data(0, Qt.UserRole)
        if tn:
            self.add_requested.emit(tn)


class Properties(QWidget):
    param_changed = Signal(str, str, object)   # node_id, name, value
    expose_toggled = Signal(str, str, bool)    # node_id, setting, show on the sub-flow block (inside a sub-flow)
    dialog_requested = Signal(str)             # node_id: the block's own window (Node.dialog) was asked for

    def __init__(self):
        super().__init__()
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(8, 8, 8, 8)
        self.title = QLabel()
        f = self.title.font()
        f.setBold(True)
        f.setPointSize(11)
        self.title.setFont(f)
        self.desc = QLabel(wordWrap=True)
        self.desc.setObjectName("hint")
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 6, 0, 0)
        self.outer.addWidget(self.title)
        self.outer.addWidget(self.desc)
        self.outer.addWidget(self.form_host)
        self.outer.addStretch(1)
        self.node: Node | None = None
        # set by the editor inside a sub-flow: (node, setting) → is it shown on the sub-flow block (None: cannot be)
        self.expose_state = None
        self.show_node(None)

    def show_node(self, node: Node | None):
        self.node = node
        while self.form.rowCount():
            self.form.removeRow(0)
        if node is None:
            self.title.setText("Blok seçilmedi")
            self.desc.setText("Ayarlarını görmek için tuvalde bir blok seçin.")
            return
        self.title.setText(node.title)
        self.desc.setText(node.description)
        if not node.params:
            self.form.addRow(QLabel("Bu bloğun ayarı yok."))
        for p in node.params:
            label = self._label(node, p)
            if p.kind == "code":                    # long text: label above, editor full width
                self.form.addRow(label)
                self.form.addRow(self._editor(node, p))
            else:
                self.form.addRow(label, self._editor(node, p))
        dialog = getattr(node, "dialog", None)      # ("button text", "module.Class" in visionflow.ui)
        if dialog:
            btn = QPushButton(dialog[0])
            btn.setObjectName("block_dialog")
            btn.clicked.connect(lambda _=False, nid=node.id: self.dialog_requested.emit(nid))
            self.form.addRow(btn)

    def _exposed(self, node, name):
        try:
            return self.expose_state(node, name) if self.expose_state is not None else None
        except Exception:  # noqa: BLE001  (never break the settings panel)
            return None

    def _label(self, node: Node, p: Param) -> QLabel:
        """The setting's name; inside a sub-flow right-clicking it offers to show it on the sub-flow block."""
        state = self._exposed(node, p.name)
        lab = QLabel(("📌 " if state else "") + p.display)
        lab.setObjectName(f"label_{p.name}")
        if "/" in p.name and node.type_name == "SubFlow":
            lab.setToolTip("Alt akışın içindeki bir bloğun ayarı (alt akışın içinde sağ tıkla gizlenebilir)")
        if state is not None:
            lab.setToolTip("Sağ tık: bu ayarı alt akış bloğunda göster / gizle" + (" (gösteriliyor)" if state else ""))
            lab.setContextMenuPolicy(Qt.CustomContextMenu)
            lab.customContextMenuRequested.connect(
                lambda pos, n=node, name=p.name, w=lab: self.expose_menu(n, name).exec(w.mapToGlobal(pos)))
        return lab

    def expose_menu(self, node: Node, name: str):
        """Right-click menu of a setting's label inside a sub-flow."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        a = menu.addAction("Bu ayarı alt akış bloğunda göster")
        a.setCheckable(True)
        a.setChecked(bool(self._exposed(node, name)))
        a.toggled.connect(lambda on, nid=node.id: self.expose_toggled.emit(nid, name, on))
        self._expose_menu = menu                # tests read it
        return menu

    def _emit(self, node, name, value):
        if node.values.get(name) == value:
            return   # e.g. focus left a text box without an edit: don't reload models for nothing
        node.values[name] = value
        self.param_changed.emit(node.id, name, value)

    def _editor(self, node: Node, p: Param) -> QWidget:
        val = node.values[p.name]
        emit = lambda v, n=p.name: self._emit(node, n, v)  # noqa: E731
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(val))
            w.toggled.connect(emit)
            return w
        if p.kind == "choice":
            w = QComboBox()
            w.addItems(p.choices)
            if str(val) not in p.choices:     # e.g. a flow saved by a newer version: show it, keep it selectable
                w.addItem(str(val))
            w.setCurrentText(str(val))
            w.currentTextChanged.connect(emit)
            return w
        if p.kind == "int":
            box = QSpinBox()
            box.setRange(int(p.min if p.min is not None else -10**9), int(p.max if p.max is not None else 10**9))
            box.setSingleStep(int(p.step or 1))
            box.setValue(int(val))
            box.valueChanged.connect(emit)
            if p.min is not None and p.max is not None and p.max - p.min <= 1000:
                # slider + spinbox: drag for live tuning, type for precision
                sl = QSlider(Qt.Horizontal)
                sl.setRange(int(p.min), int(p.max))
                sl.setValue(int(val))
                sl.valueChanged.connect(box.setValue)
                box.valueChanged.connect(sl.setValue)
                return self._row(sl, box)
            return box
        if p.kind == "float":
            box = QDoubleSpinBox()
            box.setDecimals(3)
            box.setRange(p.min if p.min is not None else -1e9, p.max if p.max is not None else 1e9)
            box.setSingleStep(p.step or 0.1)
            box.setValue(float(val))
            box.valueChanged.connect(emit)
            if p.min is not None and p.max is not None and p.max - p.min <= 10:
                sl = QSlider(Qt.Horizontal)
                sl.setRange(int(p.min * 100), int(p.max * 100))
                sl.setValue(round(val * 100))

                def from_slider(v):
                    if abs(box.value() - v / 100) >= 0.005:   # the slider is coarser: never round a typed value
                        box.setValue(v / 100)

                def to_slider(v):
                    sl.blockSignals(True)
                    sl.setValue(round(v * 100))
                    sl.blockSignals(False)
                sl.valueChanged.connect(from_slider)
                box.valueChanged.connect(to_slider)
                return self._row(sl, box)
            return box
        if p.kind in ("path", "folder"):
            le = QLineEdit(str(val))
            le.editingFinished.connect(lambda: emit(le.text()))
            btn = QPushButton("…")
            btn.setFixedWidth(30)

            def browse():
                if p.kind == "folder":
                    path = QFileDialog.getExistingDirectory(self, p.display, le.text())
                else:
                    path, _ = QFileDialog.getOpenFileName(self, p.display, le.text())
                if path:
                    le.setText(path)
                    emit(path)
            btn.clicked.connect(browse)
            return self._row(le, btn)
        if p.kind == "code":
            ed = QPlainTextEdit(str(val))
            if p.name == "code":                    # Python: monospace; prompts and notes: normal text
                ed.setFont(QFont("Consolas", 10))
            ed.setMinimumHeight(180 if p.name == "code" else 110)
            ed.textChanged.connect(lambda: emit(ed.toPlainText()))
            return ed
        le = QLineEdit(str(val))
        le.editingFinished.connect(lambda: emit(le.text()))
        return le

    @staticmethod
    def _row(*widgets):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            h.addWidget(x)
        return w


class Preview(QWidget):
    """Shows the selected block's image output; hover shows pixel BGR/HSV for color tuning. For blocks with a
    region (Kırp, Renk Tanı, Çizgi Geçiş Sayacı, Bölgedeki Nesneler, Perspektif Düzelt) it shows the block's input
    image instead, and the region can be drawn on it with the mouse."""

    roi_drawn = Signal(dict)            # settings of the drawn region, e.g. {"roi": "10,20,100,50"}

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        # Quick overlay controls
        tools_layout = QHBoxLayout()
        tools_layout.setContentsMargins(0, 0, 0, 4)
        tools_layout.setSpacing(4)

        self.btn_crosshair = QPushButton("✛ Nişangah")
        self.btn_crosshair.setCheckable(True)
        self.btn_crosshair.setToolTip("Görüntü merkezine hassas optik nişangah çizer")
        self.btn_crosshair.toggled.connect(lambda _: self._render())
        tools_layout.addWidget(self.btn_crosshair)

        self.btn_grid = QPushButton("▦ Izgara")
        self.btn_grid.setCheckable(True)
        self.btn_grid.setToolTip("Hizalama ve kalibrasyon için optik ızgara bindirir")
        self.btn_grid.toggled.connect(lambda _: self._render())
        tools_layout.addWidget(self.btn_grid)

        self.btn_snap = QPushButton("📷 Kaydet")
        self.btn_snap.setToolTip("Bu kareyi doğrudan PNG dosyası olarak masaüstüne kaydeder")
        self.btn_snap.clicked.connect(self._save_quick_snapshot)
        tools_layout.addWidget(self.btn_snap)

        tools_layout.addStretch()
        lay.addLayout(tools_layout)

        self.label = QLabel("Çalıştırınca seçili bloğun görüntüsü burada görünür")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(200, 150)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setMouseTracking(True)
        self.label.mouseMoveEvent = self._mouse_move
        self.label.mousePressEvent = self._mouse_press
        self.label.mouseReleaseEvent = self._mouse_release
        self.label.mouseDoubleClickEvent = self._mouse_double
        self.info = QLabel(" ")
        self.info.setObjectName("hint")
        lay.addWidget(self.label, 1)
        lay.addWidget(self.info)
        self.img: np.ndarray | None = None
        self.full_size: tuple[int, int] | None = None     # (w, h) of the image before it was shrunk for display
        self._pm_rect = None
        self.roi_spec = None
        self.roi_values: dict = {}
        self._drawing: list[tuple[float, float]] = []
        self._dragging = False

    def _save_quick_snapshot(self):
        if self.img is None:
            return
        import datetime as dt
        import pathlib
        home_desktop = pathlib.Path.home() / "Desktop"
        save_dir = home_desktop if home_desktop.exists() else pathlib.Path.cwd()
        filename = f"VisionFlow_Kare_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), self.img)
        self.info.setText(f"✓ Kare kaydedildi: {filename}")

    # --- image ---------------------------------------------------------------------------------------------
    def set_image(self, img: np.ndarray | None, full_size: tuple[int, int] | None = None):
        self.img = img
        if img is None:
            self.full_size = None
            self.label.setPixmap(QPixmap())
            return
        self.full_size = tuple(full_size) if full_size else (img.shape[1], img.shape[0])
        self._render()

    def set_roi(self, spec, values: dict | None = None):
        """Turn region drawing on for a block (spec from Node.roi) or off (None)."""
        from .roi import HINTS
        self.roi_spec, self.roi_values = spec, dict(values or {})
        self._drawing, self._dragging = [], False
        self.info.setText(HINTS[spec[0]] if spec else " ")
        if self.img is not None:
            self._render()

    def _render(self):
        pm = QPixmap.fromImage(to_qimage(self.img)).scaled(self.label.size(), Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation)
        ox = (self.label.width() - pm.width()) / 2
        oy = (self.label.height() - pm.height()) / 2
        self._pm_rect = (ox, oy, pm.width(), pm.height())

        from PySide6.QtGui import QPainter, QPen
        if self.btn_grid.isChecked():
            p = QPainter(pm)
            p.setPen(QPen(QColor(80, 180, 255, 60), 1, Qt.DashLine))
            step_x = max(10, pm.width() // 8)
            step_y = max(10, pm.height() // 8)
            for gx in range(step_x, pm.width(), step_x):
                p.drawLine(gx, 0, gx, pm.height())
            for gy in range(step_y, pm.height(), step_y):
                p.drawLine(0, gy, pm.width(), gy)
            p.end()

        if self.btn_crosshair.isChecked():
            p = QPainter(pm)
            p.setPen(QPen(QColor(255, 80, 80, 200), 1, Qt.SolidLine))
            cx, cy = pm.width() // 2, pm.height() // 2
            p.drawLine(0, cy, pm.width(), cy)
            p.drawLine(cx, 0, cx, pm.height())
            p.drawEllipse(cx - 15, cy - 15, 30, 30)
            p.end()

        if self.roi_spec:
            self._paint_roi(pm)
        self.label.setPixmap(pm)

    def _paint_roi(self, pm):
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QPainter, QPen, QPolygonF
        from .roi import shape_from_values
        fw, fh = self.full_size
        sx, sy = pm.width() / fw, pm.height() / fh
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        def draw(kind, pts, color, closed):
            p.setPen(QPen(QColor(color), 2))
            if kind == "rect":
                x, y, w, h = pts
                p.drawRect(QRectF(x * sx, y * sy, w * sx, h * sy))
                return
            poly = QPolygonF([QPointF(x * sx, y * sy) for x, y in pts])
            p.drawPolygon(poly) if closed and len(pts) > 2 else p.drawPolyline(poly)
            for pt in poly:
                p.drawEllipse(pt, 3, 3)
        shape = shape_from_values(self.roi_spec, self.roi_values)
        if shape and not self._drawing:
            draw(shape[0], shape[1], "#ffd54f", shape[0] != "line")
        if self._drawing:
            kind = self.roi_spec[0]
            if kind == "rect" and len(self._drawing) >= 2:
                (x0, y0), (x1, y1) = self._drawing[0], self._drawing[-1]
                draw("rect", (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)), "#4fc3f7", True)
            elif kind == "line" and len(self._drawing) >= 2:
                draw("line", [self._drawing[0], self._drawing[-1]], "#4fc3f7", False)
            elif kind in ("poly", "quad"):
                draw(kind, self._drawing, "#4fc3f7", False)
        p.end()

    # --- mouse -----------------------------------------------------------------------------------------------
    def _to_image(self, pos):
        """Label position → (x, y) in full-resolution image pixels, or None outside the picture."""
        if self.img is None or not self._pm_rect:
            return None
        ox, oy, pw, ph = self._pm_rect
        fw, fh = self.full_size
        x, y = (pos.x() - ox) * fw / pw, (pos.y() - oy) * fh / ph
        return (x, y) if 0 <= x < fw and 0 <= y < fh else None

    def _mouse_move(self, e):
        pt = self._to_image(e.position())
        if pt is None:
            return
        if self._dragging:
            self._drawing[-1:] = [pt] if len(self._drawing) > 1 else [self._drawing[0], pt]
            self._render()
        h, w = self.img.shape[:2]
        x, y = min(int(pt[0] * w / self.full_size[0]), w - 1), min(int(pt[1] * h / self.full_size[1]), h - 1)
        px = self.img[y, x]
        X, Y = int(pt[0]), int(pt[1])
        if self.img.ndim == 2:
            self.info.setText(f"x={X} y={Y}  değer={int(px)}")
        else:
            hsv = cv2.cvtColor(np.uint8([[px]]), cv2.COLOR_BGR2HSV)[0, 0]
            self.info.setText(f"x={X} y={Y}  BGR={tuple(int(v) for v in px)}  HSV={tuple(int(v) for v in hsv)}")

    def _mouse_press(self, e):
        if not self.roi_spec:
            return
        pt = self._to_image(e.position())
        kind = self.roi_spec[0]
        if e.button() == Qt.RightButton:
            if kind == "poly":
                self._finish()
            return
        if pt is None or e.button() != Qt.LeftButton:
            return
        if kind in ("rect", "line"):
            self._drawing, self._dragging = [pt], True
        else:
            self._drawing.append(pt)
            if kind == "quad" and len(self._drawing) == 4:
                self._finish()
                return
        self._render()

    def _mouse_release(self, e):
        if self._dragging and e.button() == Qt.LeftButton:
            self._dragging = False
            self._finish()

    def _mouse_double(self, e):
        if self.roi_spec and self.roi_spec[0] == "poly":
            self._finish()

    def _finish(self):
        from .roi import values_from_points
        values = values_from_points(self.roi_spec, self._drawing, self.full_size) if self.full_size else None
        self._drawing, self._dragging = [], False
        if values:
            self.roi_values.update(values)
            self.roi_drawn.emit(values)
        if self.img is not None:
            self._render()


class LogPanel(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setFont(QFont("Consolas", 9))

    def log(self, text: str):
        text = text.rstrip()
        if text:
            self.appendPlainText(text)


class QuickAdd(QWidget):
    """Popup search box (Tab on the canvas): type a few letters, Enter adds the block."""
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setMinimumWidth(300)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.edit = QLineEdit(placeholderText="Blok ara ve Enter'a bas…")
        self.list = QListWidget()
        self.list.setMinimumHeight(280)
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.entries = [(cls.title, cls.type_name, cat, cls.description or "")
                        for cat, classes in by_category().items() for cls in classes]
        self.allowed: set | None = None    # only these blocks (a dropped wire offers the ones that fit it)
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._accept)
        self.list.itemActivated.connect(lambda _: self._accept())
        self.edit.installEventFilter(self)
        self._filter("")

    def eventFilter(self, obj, e):
        if e.type() == QEvent.KeyPress and e.key() in (Qt.Key_Down, Qt.Key_Up):
            row = self.list.currentRow() + (1 if e.key() == Qt.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
            return True
        return super().eventFilter(obj, e)

    def _filter(self, text):
        text = search_fold(text.strip())
        self.list.clear()
        for title, tn, cat, desc in self.entries:
            if self.allowed is not None and tn not in self.allowed:
                continue
            if not text or text in search_fold(title) or text in search_fold(desc) or text in search_fold(tn):
                it = QListWidgetItem(f"{title}    ·  {cat.split('·')[-1].strip()}")
                it.setData(Qt.UserRole, tn)
                it.setToolTip(desc)
                self.list.addItem(it)
        self.list.setCurrentRow(0)

    def _accept(self):
        it = self.list.currentItem()
        if it:
            self.chosen.emit(it.data(Qt.UserRole))
        self.close()

    def popup(self, global_pos, allowed=None):
        self.allowed = set(allowed) if allowed is not None else None
        self.edit.setPlaceholderText("Bağlanacak bloğu ara ve Enter'a bas…" if allowed is not None
                                     else "Blok ara ve Enter'a bas…")
        self.edit.clear()
        self._filter("")
        self.move(global_pos)
        self.show()
        self.edit.setFocus()
