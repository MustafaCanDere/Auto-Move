"""Araçlar → Ekran tasarımcısı: operator (HMI) screens designed by drag and drop, without code.

The screen is kept in the flow's "operator" settings (core/screens.py) and can be used as the operator panel
("Panel düzeni: Tasarım ekranı"), also with ``--operator``. The designer works on its own copy with its own undo
history; "Uygula" puts it into the flow as one step of the editor's undo history."""
from __future__ import annotations

import copy
import json
import pathlib

from PySide6.QtCore import QMimeData, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QDrag, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget, QTabBar,
                               QToolBar, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import screens as S
from ..core.graph import walk_nodes
from .screen_items import BackgroundItem, ScreenItem

MIME = "application/x-visionflow-screen-widget"
HISTORY_MAX = 200
_CLIPBOARD: list[dict] = []          # copied widgets (shared by every designer window of the program)
ALIGN = [("sol", "Sola hizala"), ("sağ", "Sağa hizala"), ("üst", "Üste hizala"), ("alt", "Alta hizala"),
         ("yatay orta", "Yatay ortala"), ("dikey orta", "Dikey ortala"), ("yatay dağıt", "Yatay eşit dağıt"),
         ("dikey dağıt", "Dikey eşit dağıt"), ("aynı genişlik", "Aynı genişlik"), ("aynı yükseklik", "Aynı yükseklik")]


# --- choosing what a widget shows / writes ----------------------------------------------------------------------------

class SourcePicker(QDialog):
    """A tree of what a widget can be bound to: block outputs and settings (blocks inside sub-flows too, keyed
    "subflow id/inner id" like the operator panel's references), shared variables and the panel's counters."""

    def __init__(self, parent, graph, prop_kind: str = "source", current=None):
        super().__init__(parent)
        self.setWindowTitle("Bağlantı seç")
        self.resize(560, 560)
        self.graph, self.prop_kind = graph, prop_kind
        allowed = S.SOURCE_KINDS.get(prop_kind, ())
        lay = QVBoxLayout(self)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Ara…")
        self.filter.textChanged.connect(self._filter)
        lay.addWidget(self.filter)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(lambda it, _c: it.data(0, Qt.UserRole) and self.accept())
        lay.addWidget(self.tree, 1)
        from ..core.node import PortType
        if "port" in allowed:
            top = QTreeWidgetItem(self.tree, ["Blok çıkışları"])
            self._blocks(top, lambda node: [
                (f"{port}  ({t})", {"node": None, "port": port}) for port, t in node.outputs
                if prop_kind != "image_source" or t in (PortType.IMAGE, PortType.ANY)])
        if "setting" in allowed:
            top = QTreeWidgetItem(self.tree, ["Blok ayarları"])
            self._blocks(top, lambda node: [
                (f"{p.display}  ({p.name})", {"node": None, "setting": p.name}) for p in node.params
                if p.kind != "code" and "/" not in p.name])
        if "var" in allowed:
            top = QTreeWidgetItem(self.tree, ["Paylaşılan değişkenler"])
            from ..nodes.connect import VARS
            names = S.flow_variables(graph)
            names += [str(n) for n in list(VARS) if str(n) not in names]
            for n in names:
                self._leaf(top, n, {"var": n})
        if "counter" in allowed:
            top = QTreeWidgetItem(self.tree, ["Operatör paneli sayaçları"])
            for k in S.COUNTER_KEYS:
                self._leaf(top, S.COUNTER_NAMES[k], {"counter": k})
        self.tree.expandToDepth(0)
        if "var" in allowed:
            row = QHBoxLayout()
            self.var_edit = QLineEdit()
            self.var_edit.setPlaceholderText("veya yeni değişken adı yazın")
            row.addWidget(QLabel("Değişken:"))
            row.addWidget(self.var_edit, 1)
            lay.addLayout(row)
        else:
            self.var_edit = None
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if current:
            self.select_ref(current)

    def _leaf(self, parent, text, ref):
        it = QTreeWidgetItem(parent, [text])
        it.setData(0, Qt.UserRole, json.dumps(ref))
        return it

    def _blocks(self, top, leaves):
        items: dict[str, QTreeWidgetItem] = {}
        if self.graph is None:
            return
        for key, label, node in walk_nodes(self.graph):
            parent = items.get(key.rsplit("/", 1)[0], top) if "/" in key else top
            it = QTreeWidgetItem(parent, [f"{node.title}  ({key.rsplit('/', 1)[-1]})"])
            it.setToolTip(0, label)
            items[key] = it
            for text, ref in leaves(node):
                ref["node"] = key
                self._leaf(it, text, ref)

    def _all(self):
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            yield it
            stack += [it.child(i) for i in range(it.childCount())]

    def _filter(self, text):
        text = text.casefold().strip()
        for it in self._all():
            it.setHidden(False)
        if not text:
            return
        for it in self._all():
            if it.childCount() == 0:
                hit = text in it.text(0).casefold() or text in (it.parent().text(0).casefold() if it.parent() else "")
                it.setHidden(not hit)
        for _ in range(12):                 # hide groups with nothing visible left (inner levels first)
            for it in self._all():
                if it.childCount():
                    it.setHidden(all(it.child(i).isHidden() for i in range(it.childCount())))
        self.tree.expandAll()

    def select_ref(self, ref) -> bool:
        want = json.dumps(S.clean_binding(ref))
        for it in self._all():
            if it.data(0, Qt.UserRole) == want:
                self.tree.setCurrentItem(it)
                return True
        if S.binding_kind(ref) == "var" and self.var_edit is not None:
            self.var_edit.setText(ref["var"])
            return True
        return False

    def ref(self):
        if self.var_edit is not None and self.var_edit.text().strip():
            return {"var": self.var_edit.text().strip()}
        it = self.tree.currentItem()
        data = it.data(0, Qt.UserRole) if it is not None else None
        return json.loads(data) if data else None


# --- the canvas -------------------------------------------------------------------------------------------------------

class _Canvas(QGraphicsView):
    def __init__(self, scene, designer):
        super().__init__(scene)
        self.designer = designer
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setBackgroundBrush(QColor("#2b2d31"))
        self.zoom = 1.0

    def drawForeground(self, p, rect):
        d = self.designer
        if not d.show_grid or self.zoom * d.grid < 4:
            return
        w, h = d.screen["size"]
        p.setPen(QPen(QColor(255, 255, 255, 28), 0))
        g = d.grid
        x = 0
        while x <= w:
            p.drawLine(x, 0, x, h)
            x += g
        y = 0
        while y <= h:
            p.drawLine(0, y, w, y)
            y += g
        p.setPen(QPen(QColor("#29b6f6"), 0, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(0, 0, w, h))

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            wtype = bytes(e.mimeData().data(MIME)).decode()
            pos = self.mapToScene(e.position().toPoint())
            self.designer.add_widget(wtype, pos.x(), pos.y())
            e.acceptProposedAction()
        else:
            super().dropEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and self.designer.selected_ids():
            self.designer.keyPressEvent(e)          # arrows move the selected widgets, not the view
            return
        super().keyPressEvent(e)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
            self.zoom = max(0.1, min(8.0, self.zoom * f))
            self.scale(f, f)
            e.accept()
        else:
            super().wheelEvent(e)

    def fit(self):
        w, h = self.designer.screen["size"]
        self.fitInView(QRectF(-20, -20, w + 40, h + 40), Qt.KeepAspectRatio)
        self.zoom = self.transform().m11()


class _Palette(QListWidget):
    def __init__(self, designer):
        super().__init__()
        self.designer = designer
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        for wtype, spec in S.WIDGETS.items():
            it = QListWidgetItem(spec["title"])
            it.setData(Qt.UserRole, wtype)
            it.setToolTip("Tuvale sürükleyin veya çift tıklayın")
            self.addItem(it)
        self.itemDoubleClicked.connect(lambda it: designer.add_widget(it.data(Qt.UserRole)))

    def startDrag(self, actions):
        it = self.currentItem()
        if it is None:
            return
        mime = QMimeData()
        mime.setData(MIME, it.data(Qt.UserRole).encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class _ColorButton(QPushButton):
    def __init__(self, value: str, allow_empty: bool, on_change):
        super().__init__()
        self.allow_empty, self.on_change = allow_empty, on_change
        self.set(value)
        self.clicked.connect(self.pick)

    def set(self, value):
        self.value = value or ""
        self.setText(self.value or "(yok)")
        fg = "#000" if self.value and QColor(self.value).lightness() > 140 else "#fff"
        self.setStyleSheet(f"background: {self.value}; color: {fg};" if self.value else "")

    def pick(self):
        c = QColorDialog.getColor(QColor(self.value or "#ffffff"), self, "Renk seç")
        if c.isValid():
            self.set(c.name())
            self.on_change(self.value)


# --- the designer window ----------------------------------------------------------------------------------------------

class ScreenDesigner(QDialog):
    """``graph`` is the flow (the editor's root); ``editor`` the main window (for "Uygula" and the live preview)."""

    def __init__(self, parent=None, graph=None, editor=None, screen: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Ekran tasarımcısı")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(1400, 860)
        self._graph, self.editor = graph, editor
        op = (getattr(graph, "operator", None) or {}) if graph is not None else {}
        if screen is None:
            screen = op.get("screen")
        self.screen = S.clean_screen(copy.deepcopy(screen), graph, []) if screen else S.new_screen()
        self.use_as_panel = op.get("layout") == "screen" if screen else True
        self.page_index = 0
        self.items: list[ScreenItem] = []
        self.snap, self.show_grid = True, True
        self.blink_phase = 0
        self.preview = None
        self._building = False
        self.history = [self._dump()]
        self.hidx = 0
        self.applied = self.history[0] + str(self.use_as_panel)
        self._build_ui()
        self.rebuild()
        QTimer.singleShot(0, self.canvas.fit)

    @property
    def graph(self):
        """The flow the bindings are checked against: the editor's current flow (undo there makes a new one)."""
        root = getattr(self.editor, "root", None) if self.editor is not None else None
        return root if root is not None else self._graph

    @graph.setter
    def graph(self, value):
        self._graph = value

    @property
    def grid(self) -> int:
        return int(self.screen.get("grid", 10))

    @property
    def page(self) -> dict:
        return self.screen["pages"][self.page_index]

    # --- UI ------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        lay = QVBoxLayout(self)
        tb = QToolBar()
        self.toolbar = tb

        def act(text, slot, key=None, tip=None):
            a = QAction(text, self)
            a.triggered.connect(lambda _c=False: slot())
            if key:
                a.setShortcut(QKeySequence(key))
                a.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            if tip:
                a.setToolTip(tip)
            tb.addAction(a)
            self.addAction(a)
            return a
        self.undo_act = act("↶ Geri al", self.undo, "Ctrl+Z")
        self.redo_act = act("↷ Yinele", self.redo, "Ctrl+Y")
        tb.addSeparator()
        act("Kopyala", self.copy, "Ctrl+C")
        act("Yapıştır", self.paste, "Ctrl+V")
        act("Çoğalt", self.duplicate, "Ctrl+D")
        act("Sil", self.delete_selected, "Del")
        tb.addSeparator()
        act("⤒ Öne getir", lambda: self.z_order("front"), "Ctrl+Shift+Up")
        act("⤓ Arkaya gönder", lambda: self.z_order("back"), "Ctrl+Shift+Down")
        act("Bir öne", lambda: self.z_order("up"), "Ctrl+Up")
        act("Bir arkaya", lambda: self.z_order("down"), "Ctrl+Down")
        align_btn = QToolButton()
        align_btn.setText("Hizala ▾")
        align_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(align_btn)
        for mode, text in ALIGN:
            menu.addAction(text, lambda m=mode: self.align(m))
        align_btn.setMenu(menu)
        tb.addWidget(align_btn)
        tb.addSeparator()
        self.snap_check = QCheckBox("Izgaraya yapış")
        self.snap_check.setChecked(True)
        self.snap_check.toggled.connect(self._set_snap)
        tb.addWidget(self.snap_check)
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(1, 200)
        self.grid_spin.setValue(self.grid)
        self.grid_spin.setSuffix(" px")
        self.grid_spin.valueChanged.connect(self._set_grid)
        tb.addWidget(self.grid_spin)
        tb.addSeparator()
        act("+ Sayfa", self.add_page)
        act("Sayfayı yeniden adlandır", self.rename_page)
        act("− Sayfa", self.remove_page)
        tb.addSeparator()
        self.preview_act = act("▶ Önizleme", self.toggle_preview, "F5", "Ekranı çalışırken olduğu gibi gösterir")
        self.preview_act.setCheckable(True)
        act("Sığdır", lambda: self.canvas.fit(), "F")
        lay.addWidget(tb)

        split = QSplitter()
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Öğeler (sürükleyin)"))
        self.palette = _Palette(self)
        ll.addWidget(self.palette)
        split.addWidget(left)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        self.page_tabs = QTabBar()
        self.page_tabs.currentChanged.connect(self._tab_changed)
        cl.addWidget(self.page_tabs)
        self.stack = QStackedWidget()
        self.scene = QGraphicsScene(self)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.canvas = _Canvas(self.scene, self)
        self.stack.addWidget(self.canvas)
        cl.addWidget(self.stack, 1)
        split.addWidget(center)

        self.prop_area = QScrollArea()
        self.prop_area.setWidgetResizable(True)
        self.prop_area.setMinimumWidth(320)
        split.addWidget(self.prop_area)
        split.setSizes([180, 900, 340])
        lay.addWidget(split, 1)

        bottom = QHBoxLayout()
        self.use_check = QCheckBox("Operatör panelinde bu ekranı kullan (Panel düzeni: Tasarım ekranı)")
        self.use_check.setChecked(self.use_as_panel)
        bottom.addWidget(self.use_check)
        self.info = QLabel("")
        self.info.setStyleSheet("color: #ffb74d;")
        bottom.addWidget(self.info, 1)
        bb = QDialogButtonBox()
        self.apply_btn = bb.addButton("Uygula", QDialogButtonBox.ApplyRole)
        bb.addButton("Tamam", QDialogButtonBox.AcceptRole)
        bb.addButton("Kapat", QDialogButtonBox.RejectRole)
        for b in bb.buttons():          # Enter in a property field must not close the window
            b.setAutoDefault(False)
            b.setDefault(False)
        self.apply_btn.clicked.connect(self.apply)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bottom.addWidget(bb)
        lay.addLayout(bottom)
        self._prop_timer = QTimer(self, singleShot=True, interval=0, timeout=self._build_props)
        self._update_actions()

    def _set_snap(self, on):
        self.snap = bool(on)

    def _set_grid(self, v):
        if v != self.grid:
            self.screen["grid"] = int(v)
            self.canvas.viewport().update()
            self.push()

    # --- model ↔ scene -------------------------------------------------------------------------------------------
    def _dump(self) -> str:
        return json.dumps(self.screen, sort_keys=True, ensure_ascii=False)

    def rebuild(self, select: list[str] | None = None):
        """Draw the current page from the model again (after undo, page change, paste …)."""
        self._building = True
        try:
            self.page_index = max(0, min(self.page_index, len(self.screen["pages"]) - 1))
            self.page_tabs.blockSignals(True)
            while self.page_tabs.count():
                self.page_tabs.removeTab(0)
            for p in self.screen["pages"]:
                self.page_tabs.addTab(p["name"])
            self.page_tabs.setCurrentIndex(self.page_index)
            self.page_tabs.blockSignals(False)
            self.scene.clear()
            self.items = []
            w, h = self.screen["size"]
            self.scene.setSceneRect(QRectF(-200, -200, w + 400, h + 400))
            self.scene.addItem(BackgroundItem(self.page, (w, h)))
            for z, wd in enumerate(self.page["widgets"]):
                it = ScreenItem(wd, self, S.widget_problem(wd, self.graph), editing=True)
                it.setZValue(z)
                self.scene.addItem(it)
                self.items.append(it)
                if select and wd["id"] in select:
                    it.setSelected(True)
            self.grid_spin.blockSignals(True)
            self.grid_spin.setValue(self.grid)
            self.grid_spin.blockSignals(False)
        finally:
            self._building = False
        self._update_info()
        self._prop_timer.start()

    def _update_info(self):
        problems = sum(1 for p in self.screen["pages"] for wd in p["widgets"] if S.widget_problem(wd, self.graph))
        self.info.setText(f"⚠ {problems} öğe bağlantısız / tanınmıyor (yer tutucu olarak gösterilir)" if problems
                          else "")

    def push(self):
        """Record the model as an undo step (when it changed)."""
        s = self._dump()
        if s == self.history[self.hidx]:
            return
        del self.history[self.hidx + 1:]
        self.history.append(s)
        if len(self.history) > HISTORY_MAX:
            del self.history[0]
        self.hidx = len(self.history) - 1
        self._update_actions()
        self._update_info()

    def _update_actions(self):
        self.undo_act.setEnabled(self.hidx > 0)
        self.redo_act.setEnabled(self.hidx < len(self.history) - 1)

    def _restore(self, s: str):
        sel = self.selected_ids()
        self.screen = json.loads(s)
        self.rebuild(sel)
        self._update_actions()

    def undo(self):
        if self.preview is not None or self.hidx <= 0:
            return
        self.hidx -= 1
        self._restore(self.history[self.hidx])

    def redo(self):
        if self.preview is not None or self.hidx >= len(self.history) - 1:
            return
        self.hidx += 1
        self._restore(self.history[self.hidx])

    def items_edited(self):
        """Items were moved / resized with the mouse: keep their places in the model."""
        changed = False
        for it in self.items:
            r = it.rect_list()
            if it.wd["rect"] != r:
                it.wd["rect"] = r
                changed = True
        if changed:
            self.push()
            self._prop_timer.start()

    def selected_ids(self) -> list[str]:
        try:
            return [it.wd["id"] for it in self.items if it.isSelected()]
        except RuntimeError:
            return []

    def selected(self) -> list[dict]:
        ids = set(self.selected_ids())
        return [wd for wd in self.page["widgets"] if wd["id"] in ids]

    def select(self, ids):
        self.scene.clearSelection()
        for it in self.items:
            if it.wd["id"] in ids:
                it.setSelected(True)

    def _used_ids(self) -> set:
        return {wd["id"] for p in self.screen["pages"] for wd in p["widgets"]}

    # --- editing -------------------------------------------------------------------------------------------------
    def add_widget(self, wtype: str, x: float | None = None, y: float | None = None, **props) -> dict | None:
        if wtype not in S.WIDGETS or self.preview is not None:
            return None
        if len(self.page["widgets"]) >= S.MAX_WIDGETS:
            self.info.setText(f"Bir sayfada en çok {S.MAX_WIDGETS} öğe olabilir")
            return None
        w, h = self.screen["size"]
        sw, sh = S.WIDGETS[wtype]["size"]
        if x is None:
            x, y = (w - sw) / 2, (h - sh) / 2
        g = self.grid if self.snap else 1
        x, y = round(x / g) * g, round(y / g) * g
        wd = S.new_widget(wtype, x, y, S.unique_id(self._used_ids()), **props)
        self.page["widgets"].append(wd)
        self.push()
        self.rebuild([wd["id"]])
        return wd

    def delete_selected(self):
        ids = set(self.selected_ids())
        if not ids or self.preview is not None:
            return
        self.page["widgets"] = [wd for wd in self.page["widgets"] if wd["id"] not in ids]
        self.push()
        self.rebuild()

    def copy(self) -> int:
        sel = self.selected()
        if sel:
            _CLIPBOARD[:] = copy.deepcopy(sel)
        return len(sel)

    def paste(self, offset: float | None = None) -> list[str]:
        if not _CLIPBOARD or self.preview is not None:
            return []
        off = self.grid * 2 if offset is None else offset
        used = self._used_ids()
        new = []
        for wd in copy.deepcopy(_CLIPBOARD):
            if len(self.page["widgets"]) >= S.MAX_WIDGETS:
                self.info.setText(f"Bir sayfada en çok {S.MAX_WIDGETS} öğe olabilir")
                break
            wd = S.clean_widget(wd, self.graph, [], set(used) | set())
            if wd is None:
                continue
            wd["id"] = S.unique_id(used)
            used.add(wd["id"])
            wd["rect"][0] += off
            wd["rect"][1] += off
            self.page["widgets"].append(wd)
            new.append(wd["id"])
        _CLIPBOARD[:] = [copy.deepcopy(wd) for wd in self.page["widgets"] if wd["id"] in new]   # next paste: further
        self.push()
        self.rebuild(new)
        return new

    def duplicate(self):
        if self.copy():
            self.paste()

    def move_selected(self, dx: float, dy: float):
        for wd in self.selected():
            wd["rect"][0] += dx
            wd["rect"][1] += dy
        self.push()
        self.rebuild(self.selected_ids())

    def z_order(self, how: str):
        ws = self.page["widgets"]
        ids = set(self.selected_ids())
        if not ids:
            return
        sel = [wd for wd in ws if wd["id"] in ids]
        rest = [wd for wd in ws if wd["id"] not in ids]
        if how == "front":
            ws[:] = rest + sel
        elif how == "back":
            ws[:] = sel + rest
        elif how == "up":
            for i in range(len(ws) - 2, -1, -1):
                if ws[i]["id"] in ids and ws[i + 1]["id"] not in ids:
                    ws[i], ws[i + 1] = ws[i + 1], ws[i]
        elif how == "down":
            for i in range(1, len(ws)):
                if ws[i]["id"] in ids and ws[i - 1]["id"] not in ids:
                    ws[i], ws[i - 1] = ws[i - 1], ws[i]
        self.push()
        self.rebuild(list(ids))

    def align(self, mode: str):
        sel = self.selected()
        if len(sel) < 2 or len(sel) < 3 and mode.endswith("dağıt"):
            self.info.setText("Hizalamak için en az iki öğe seçin (eşit dağıtmak için üç)")
            return
        rs = [wd["rect"] for wd in sel]
        left, top = min(r[0] for r in rs), min(r[1] for r in rs)
        right, bottom = max(r[0] + r[2] for r in rs), max(r[1] + r[3] for r in rs)
        if mode == "sol":
            for r in rs:
                r[0] = left
        elif mode == "sağ":
            for r in rs:
                r[0] = right - r[2]
        elif mode == "üst":
            for r in rs:
                r[1] = top
        elif mode == "alt":
            for r in rs:
                r[1] = bottom - r[3]
        elif mode == "yatay orta":
            cx = (left + right) / 2
            for r in rs:
                r[0] = cx - r[2] / 2
        elif mode == "dikey orta":
            cy = (top + bottom) / 2
            for r in rs:
                r[1] = cy - r[3] / 2
        elif mode == "yatay dağıt":
            rs.sort(key=lambda r: r[0])
            gap = (right - left - sum(r[2] for r in rs)) / (len(rs) - 1)
            x = left
            for r in rs:
                r[0] = x
                x += r[2] + gap
        elif mode == "dikey dağıt":
            rs.sort(key=lambda r: r[1])
            gap = (bottom - top - sum(r[3] for r in rs)) / (len(rs) - 1)
            y = top
            for r in rs:
                r[1] = y
                y += r[3] + gap
        elif mode == "aynı genişlik":
            for r in rs:
                r[2] = rs[0][2]
        elif mode == "aynı yükseklik":
            for r in rs:
                r[3] = rs[0][3]
        for r in rs:
            r[:] = [round(v, 2) for v in r]
        self.push()
        self.rebuild([wd["id"] for wd in sel])

    def set_prop(self, wid: str, name: str, value) -> bool:
        """Change one property of a widget on the current page (one undo step)."""
        wd = next((w for w in self.page["widgets"] if w["id"] == wid), None)
        if wd is None:
            return False
        if name == "rect":
            wd["rect"] = [float(v) for v in value]
        else:
            if wd.get("props", {}).get(name) == value:
                return False
            wd.setdefault("props", {})[name] = copy.deepcopy(value)
        it = next((i for i in self.items if i.wd is wd), None)
        if it is not None:
            it.problem = S.widget_problem(wd, self.graph)
            it.setToolTip(it.problem or "")
            if name == "rect":
                it.set_geometry(*wd["rect"])
            it.update()
        self.push()
        return True

    def set_screen(self, name: str, value):
        """A setting of the whole screen (size, show_bar, show_tabs, rate) or of the page (bg, bg_image)."""
        if name in ("bg", "bg_image"):
            self.page[name] = value
        elif name == "size":
            self.screen["size"] = [max(S.SIZE_MIN[0], min(S.SIZE_MAX[0], int(value[0]))),
                                   max(S.SIZE_MIN[1], min(S.SIZE_MAX[1], int(value[1])))]
        else:
            self.screen[name] = value
        self.push()
        self.rebuild(self.selected_ids())

    # --- pages ---------------------------------------------------------------------------------------------------
    def _tab_changed(self, i):
        if not self._building and i >= 0 and self.preview is None:
            self.page_index = i
            self.rebuild()

    def add_page(self, name: str | None = None) -> int:
        if len(self.screen["pages"]) >= S.MAX_PAGES:
            self.info.setText(f"En çok {S.MAX_PAGES} sayfa olabilir")
            return self.page_index
        names = {p["name"] for p in self.screen["pages"]}
        if not name:
            n = len(self.screen["pages"]) + 1
            while f"Sayfa {n}" in names:
                n += 1
            name = f"Sayfa {n}"
        page = S.new_page(name[:60])
        page["bg"] = self.page["bg"]
        self.screen["pages"].append(page)
        self.page_index = len(self.screen["pages"]) - 1
        self.push()
        self.rebuild()
        return self.page_index

    def rename_page(self, name: str | None = None):
        if name is None:
            name, ok = QInputDialog.getText(self, "Sayfa adı", "Yeni ad:", QLineEdit.Normal, self.page["name"])
            if not ok:
                return
        name = str(name).strip()[:60]
        if name:
            self.page["name"] = name
            self.push()
            self.rebuild(self.selected_ids())

    def remove_page(self):
        if len(self.screen["pages"]) <= 1:
            self.info.setText("Son sayfa silinemez")
            return
        del self.screen["pages"][self.page_index]
        self.page_index = max(0, self.page_index - 1)
        self.push()
        self.rebuild()

    # --- property editor -----------------------------------------------------------------------------------------
    def _selection_changed(self):
        if not self._building:
            self._prop_timer.start()

    def _build_props(self):
        try:
            sel = self.selected()
        except RuntimeError:
            return
        box = QWidget()
        form = QFormLayout(box)
        self.prop_widgets = {}
        if len(sel) == 1:
            self._widget_props(form, sel[0])
        elif sel:
            form.addRow(QLabel(f"{len(sel)} öğe seçili. Hizala menüsüyle düzenleyin."))
        else:
            self._page_props(form)
        old = self.prop_area.takeWidget()
        if old is not None:
            # a focused field would report "editing finished" while it is destroyed, into a half-deleted panel
            for child in old.findChildren(QWidget):
                child.blockSignals(True)
            old.hide()
            old.deleteLater()
        self.prop_area.setWidget(box)

    def _widget_props(self, form, wd):
        wid = wd["id"]
        spec = S.WIDGETS.get(wd["type"])
        head = QLabel(f"<b>{spec['title'] if spec else wd['type']}</b>  ({wid})")
        form.addRow(head)
        problem = S.widget_problem(wd, self.graph)
        if problem:
            warn = QLabel(f"⚠ {problem}")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #ffb74d;")
            form.addRow(warn)
        spins = []
        rows = [QHBoxLayout(), QHBoxLayout()]
        for i, v in enumerate(wd["rect"]):
            sp = QDoubleSpinBox()
            sp.setRange(-20000, 20000) if i < 2 else sp.setRange(4, 20000)
            sp.setDecimals(0)
            sp.setValue(v)
            spins.append(sp)
            rows[i // 2].addWidget(sp)

        def geo_changed():
            self.set_prop(wid, "rect", [s.value() for s in spins])
        for sp in spins:
            sp.editingFinished.connect(geo_changed)
        form.addRow("Konum (x, y)", rows[0])
        form.addRow("Boyut (gen., yük.)", rows[1])
        self.prop_widgets["rect"] = spins
        if spec is None:
            form.addRow(QLabel("Bu öğe türü bu sürümde tanınmıyor; kaydedilir ama gösterilemez."))
            return
        props = wd["props"]
        for name, kind, _default, label, extra in S.prop_specs(wd["type"]):
            w = self._editor_for(wid, name, kind, props.get(name), extra)
            self.prop_widgets[name] = w
            form.addRow(label, w)

    def _editor_for(self, wid, name, kind, value, extra):
        def setter(v, n=name):
            self.set_prop(wid, n, v)
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda c: setter(bool(c)))
        elif kind == "int":
            w = QSpinBox()
            w.setRange(int(extra.get("min", -1_000_000)), int(extra.get("max", 1_000_000)))
            w.setValue(int(value or 0))
            w.editingFinished.connect(lambda: setter(w.value()))
        elif kind == "float":
            w = QDoubleSpinBox()
            w.setRange(float(extra.get("min", -1e9)), float(extra.get("max", 1e9)))
            w.setDecimals(3)
            w.setValue(float(value or 0))
            w.editingFinished.connect(lambda: setter(w.value()))
        elif kind == "choice":
            w = QComboBox()
            w.addItems(extra.get("choices", []))
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(setter)
        elif kind in ("color", "color0"):
            w = _ColorButton(value, kind == "color0", setter)
            if kind == "color0":
                row = QWidget()
                hl = QHBoxLayout(row)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(w, 1)
                clear = QPushButton("Yok")
                clear.clicked.connect(lambda: (w.set(""), setter("")))
                hl.addWidget(clear)
                row.button = w
                return row
        elif kind == "text":
            w = QPlainTextEdit(str(value or ""))
            w.setFixedHeight(70)
            timer = QTimer(w, singleShot=True, interval=500, timeout=lambda: setter(w.toPlainText()))
            w.textChanged.connect(timer.start)
        elif kind == "thresholds":
            w = QLineEdit(S.thresholds_text(value))
            w.setPlaceholderText("50=#ff9800; 80=#c62828")

            def th_done():
                try:
                    setter(S.parse_thresholds(w.text()))
                    w.setStyleSheet("")
                    w.setToolTip("")
                except ValueError as exc:
                    w.setStyleSheet("border: 1px solid #ef5350;")
                    w.setToolTip(str(exc))
            w.editingFinished.connect(th_done)
        elif kind in S.SOURCE_KINDS:
            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            btn = QPushButton(S.binding_label(value, self.graph) if value else "(seçilmedi) …")
            btn.setToolTip(S.binding_problem(value, self.graph, kind) or "" if value else "Bağlamak için tıklayın")
            btn.clicked.connect(lambda: self.pick_binding(wid, name, kind))
            hl.addWidget(btn, 1)
            clear = QPushButton("×")
            clear.setFixedWidth(28)
            clear.clicked.connect(lambda: (setter(None), self._prop_timer.start()))
            hl.addWidget(clear)
            w.button = btn
        elif kind == "file":
            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(str(value or ""))
            edit.editingFinished.connect(lambda: setter(edit.text().strip()))
            hl.addWidget(edit, 1)
            b = QPushButton("…")
            b.setFixedWidth(28)
            b.clicked.connect(lambda: self._pick_file(edit, setter))
            hl.addWidget(b)
        else:
            w = QLineEdit(str(value or ""))
            w.editingFinished.connect(lambda: setter(w.text()))
        return w

    def _pick_file(self, edit, setter):
        path, _ = QFileDialog.getOpenFileName(self, "Resim seç", edit.text(), "Resimler (*.png *.jpg *.jpeg *.bmp)")
        if path:
            path = relative_to_flow(path)
            edit.setText(path)
            setter(path)

    def pick_binding(self, wid, name, kind):
        wd = next((w for w in self.page["widgets"] if w["id"] == wid), None)
        if self.editor is not None and hasattr(self.editor, "_sync"):
            self.editor._sync()              # blocks just added inside an open sub-flow are offered too
        dlg = SourcePicker(self, self.graph, kind, wd["props"].get(name) if wd else None)
        if dlg.exec() == QDialog.Accepted and dlg.ref() is not None:
            self.set_prop(wid, name, dlg.ref())
            self._prop_timer.start()

    def _page_props(self, form):
        form.addRow(QLabel(f"<b>Sayfa: {self.page['name']}</b>"))
        bg = _ColorButton(self.page["bg"], False, lambda v: self.set_screen("bg", v))
        form.addRow("Arka plan rengi", bg)
        img_row = QWidget()
        hl = QHBoxLayout(img_row)
        hl.setContentsMargins(0, 0, 0, 0)
        img = QLineEdit(self.page.get("bg_image", ""))
        img.editingFinished.connect(lambda: self.set_screen("bg_image", img.text().strip()))
        hl.addWidget(img, 1)
        b = QPushButton("…")
        b.setFixedWidth(28)
        b.clicked.connect(lambda: self._pick_file(img, lambda v: self.set_screen("bg_image", v)))
        hl.addWidget(b)
        form.addRow("Arka plan resmi", img_row)
        form.addRow(QLabel("<b>Ekran</b>"))
        size_row = QWidget()
        sl = QHBoxLayout(size_row)
        sl.setContentsMargins(0, 0, 0, 0)
        ws, hs = QSpinBox(), QSpinBox()
        ws.setRange(S.SIZE_MIN[0], S.SIZE_MAX[0])
        hs.setRange(S.SIZE_MIN[1], S.SIZE_MAX[1])
        ws.setValue(self.screen["size"][0])
        hs.setValue(self.screen["size"][1])
        for sp in (ws, hs):
            sp.editingFinished.connect(lambda: self.set_screen("size", (ws.value(), hs.value())))
            sl.addWidget(sp)
        form.addRow("Tasarım çözünürlüğü", size_row)
        hint = QLabel("Ekran her çözünürlükte oranı korunarak sığdırılır.")
        hint.setWordWrap(True)
        form.addRow(hint)
        for key, text in (("show_bar", "Alt düğme çubuğunu göster (Başlat / Durdur / Düzenleyici)"),
                          ("show_tabs", "Sayfa sekmelerini göster")):
            c = QCheckBox(text)
            c.setChecked(bool(self.screen.get(key, True)))
            c.toggled.connect(lambda v, k=key: self.set_screen(k, bool(v)))
            form.addRow(c)
        rate = QSpinBox()
        rate.setRange(1, 30)
        rate.setValue(int(self.screen.get("rate", 15)))
        rate.setSuffix(" Hz")
        rate.editingFinished.connect(lambda: self.set_screen("rate", rate.value()))
        form.addRow("Yenileme hızı", rate)
        self.prop_widgets = {"size": (ws, hs), "bg": bg, "rate": rate}

    # --- preview -------------------------------------------------------------------------------------------------
    def toggle_preview(self, on: bool | None = None):
        on = self.preview is None if on is None else on
        if on and self.preview is None:
            from ..core import Graph
            from .screen_runtime import ScreenHost, ScreenView
            graph = Graph.from_dict(self.graph.to_dict()) if self.graph is not None else None
            editor = self.editor
            host = ScreenHost(graph, None, message=lambda t: self.info.setText(f"Önizleme: {t}"))
            view = ScreenView(copy.deepcopy(self.screen), graph, host)
            view.show_page(self.page_index)
            self.preview = view
            self.stack.addWidget(view)
            self.stack.setCurrentWidget(view)
            worker = getattr(editor, "worker", None)
            if worker is not None:            # the editor runs the flow: the preview shows its live values
                worker.screen = view.feed
            self.info.setText("Önizleme: düğmeler ve girişler çalışır; değişen ayarlar akışa yazılmaz"
                              + ("" if worker is not None else " (canlı değerler için akışı çalıştırın)"))
        elif not on and self.preview is not None:
            view, self.preview = self.preview, None
            worker = getattr(self.editor, "worker", None)
            if worker is not None and getattr(worker, "screen", None) is view.feed:
                worker.screen = None
            view.stop_timer()
            self.stack.removeWidget(view)
            view.deleteLater()
            self.stack.setCurrentWidget(self.canvas)
            self._update_info()
        self.preview_act.setChecked(self.preview is not None)
        return self.preview

    # --- saving into the flow --------------------------------------------------------------------------------------
    def result_operator(self) -> dict:
        """The flow's operator settings with this screen in them."""
        from ..core.operator import clean_operator
        from ..production import auto_config
        graph = self.graph
        base = dict(getattr(graph, "operator", None) or auto_config(graph))
        base["screen"] = copy.deepcopy(self.screen)
        base["layout"] = "screen" if self.use_check.isChecked() else "auto"
        return clean_operator(base, graph, [])

    def dirty(self) -> bool:
        return self._dump() + str(self.use_check.isChecked()) != self.applied

    def apply(self) -> bool:
        """Put the screen into the flow (one undo step in the editor)."""
        if self.preview is not None:
            self.toggle_preview(False)
        if self.editor is not None and hasattr(self.editor, "_sync"):
            self.editor._sync()
        graph = self.graph
        if graph is None:
            return False
        graph.operator = self.result_operator()
        self.graph = graph
        if self.editor is not None:
            self.editor._snapshot()
            self.editor.statusBar().showMessage("Ekran tasarımı akışa kaydedildi (dosyayı kaydetmeyi unutmayın)",
                                                5000)
        self.applied = self._dump() + str(self.use_check.isChecked())
        self.info.setText("Uygulandı")
        return True

    def ask_unsaved(self) -> str:
        """'apply', 'discard' or 'cancel' for unapplied changes. Tests replace this method."""
        r = QMessageBox.question(self, "Ekran tasarımcısı", "Değişiklikler akışa uygulansın mı?",
                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        return {QMessageBox.Yes: "apply", QMessageBox.No: "discard"}.get(r, "cancel")

    def accept(self):
        self.apply()
        self._cleanup()
        super().accept()

    def reject(self):
        if self.dirty():
            ans = self.ask_unsaved()
            if ans == "cancel":
                return
            if ans == "apply":
                self.apply()
        self._cleanup()
        super().reject()

    def closeEvent(self, e):
        if self.dirty():
            ans = self.ask_unsaved()
            if ans == "cancel":
                e.ignore()
                return
            if ans == "apply":
                self.apply()
        self._cleanup()
        super().closeEvent(e)

    def _cleanup(self):
        self.toggle_preview(False)

    def keyPressEvent(self, e):
        if self.preview is None and self.selected_ids() and e.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up,
                                                                       Qt.Key_Down):
            step = self.grid if self.snap else 1
            if e.modifiers() & Qt.ShiftModifier:
                step *= 5
            dx = {Qt.Key_Left: -step, Qt.Key_Right: step}.get(e.key(), 0)
            dy = {Qt.Key_Up: -step, Qt.Key_Down: step}.get(e.key(), 0)
            self.move_selected(dx, dy)
            return
        if e.key() == Qt.Key_Escape and self.preview is not None:
            self.toggle_preview(False)
            return
        super().keyPressEvent(e)


def relative_to_flow(path: str) -> str:
    """A picked file inside the open flow's folder is kept relative to it (the flow can be moved with its folder)."""
    from ..core import node as _node
    try:
        if _node.FLOW_DIR:
            return str(pathlib.Path(path).resolve().relative_to(pathlib.Path(_node.FLOW_DIR).resolve()))
    except ValueError:
        pass
    return path
