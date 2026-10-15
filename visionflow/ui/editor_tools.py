"""Editor helpers (2.0): align / distribute, snap to grid, mini map, export the canvas as an image, dropping a wire
on empty canvas, inserting a block into a wire, searching the flow's blocks, jumping to failing blocks, light theme
and block colour labels. Most of it is plain functions over a NodeScene so it can be tested without the window;
``install(win)`` adds the menus and actions to the main window."""
from __future__ import annotations

import pathlib

from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, \
    QToolBar, QVBoxLayout, QWidget

from ..core import GraphError, PortType, by_category, registry
from ..core.node import BLOCK_COLORS, clean_color
from . import theme
from .scene import HEADER_H, NODE_W, ROW_H, EdgeItem, FrameItem, NodeItem, SocketItem

GRID = 20


# --- snap to grid -------------------------------------------------------------------------------------------------

def snap_value(v: float, grid: float = GRID) -> float:
    return float(round(v / grid) * grid) if grid > 0 else float(v)


def snap_point(p: QPointF, grid: float = GRID) -> QPointF:
    return QPointF(snap_value(p.x(), grid), snap_value(p.y(), grid))


# --- align / distribute -------------------------------------------------------------------------------------------

ALIGN_MODES = {
    "left": "Sola hizala", "right": "Sağa hizala", "top": "Üste hizala", "bottom": "Alta hizala",
    "hcenter": "Ortala (dikey eksende)", "vcenter": "Ortala (yatay eksende)",
}
DISTRIBUTE_MODES = {"horizontal": "Yatay dağıt (eşit aralık)", "vertical": "Dikey dağıt (eşit aralık)"}


def align_positions(rects: list[tuple], mode: str) -> list[tuple[float, float]]:
    """New top-left corners for rects (x, y, w, h) aligned by `mode` (see ALIGN_MODES)."""
    if mode not in ALIGN_MODES:
        raise ValueError(f"Bilinmeyen hizalama: {mode}")
    if not rects:
        return []
    if mode == "left":
        v = min(r[0] for r in rects)
        return [(v, r[1]) for r in rects]
    if mode == "right":
        v = max(r[0] + r[2] for r in rects)
        return [(v - r[2], r[1]) for r in rects]
    if mode == "top":
        v = min(r[1] for r in rects)
        return [(r[0], v) for r in rects]
    if mode == "bottom":
        v = max(r[1] + r[3] for r in rects)
        return [(r[0], v - r[3]) for r in rects]
    if mode == "hcenter":                       # same centre x (a column)
        v = (min(r[0] for r in rects) + max(r[0] + r[2] for r in rects)) / 2
        return [(v - r[2] / 2, r[1]) for r in rects]
    v = (min(r[1] for r in rects) + max(r[1] + r[3] for r in rects)) / 2   # vcenter: same centre y (a row)
    return [(r[0], v - r[3] / 2) for r in rects]


def distribute_positions(rects: list[tuple], axis: str) -> list[tuple[float, float]]:
    """Equal gaps between neighbours along `axis` ("horizontal"/"vertical"); the outermost two stay put. When the
    blocks overlap too much for that, their centres are spread evenly instead. Order of the result = order given."""
    if axis not in DISTRIBUTE_MODES:
        raise ValueError(f"Bilinmeyen dağıtma yönü: {axis}")
    out = [(r[0], r[1]) for r in rects]
    if len(rects) < 3:
        return out
    a, size = (0, 2) if axis == "horizontal" else (1, 3)
    order = sorted(range(len(rects)), key=lambda i: (rects[i][a], i))
    first, last = rects[order[0]], rects[order[-1]]
    span = last[a] + last[size] - first[a]
    gap = (span - sum(r[size] for r in rects)) / (len(rects) - 1)
    pos = first[a]
    c0, c1 = first[a] + first[size] / 2, last[a] + last[size] / 2
    for k, i in enumerate(order):
        r = rects[i]
        if gap >= 0:
            v = pos
            pos += r[size] + gap
        else:
            v = c0 + (c1 - c0) * k / (len(rects) - 1) - r[size] / 2
        out[i] = (v, r[1]) if a == 0 else (r[0], v)
    return out


def selected_blocks(scene) -> list[NodeItem]:
    return [it for it in scene.selectedItems() if isinstance(it, NodeItem)]


def _rect(it: NodeItem) -> tuple:
    return it.pos().x(), it.pos().y(), NODE_W, it.height


def arrange_selected(scene, mode: str) -> int:
    """Align (ALIGN_MODES) or distribute (DISTRIBUTE_MODES) the selected blocks as one undoable step.
    Returns how many blocks moved."""
    items = selected_blocks(scene)
    need = 3 if mode in DISTRIBUTE_MODES else 2
    if len(items) < need:
        scene.message.emit(f"Bunun için en az {need} blok seçin")
        return 0
    rects = [_rect(it) for it in items]
    new = distribute_positions(rects, mode) if mode in DISTRIBUTE_MODES else align_positions(rects, mode)
    moved = 0
    for it, (x, y), r in zip(items, new, rects):
        if (round(x, 3), round(y, 3)) != (round(r[0], 3), round(r[1], 3)):
            it.setPos(x, y)
            moved += 1
    if moved:
        scene.edited.emit()
    return moved


# --- block colour labels --------------------------------------------------------------------------------------------

def set_block_color(scene, color: str | None, items: list[NodeItem] | None = None) -> int:
    """Colour label on the given (default: selected) blocks; None removes it. One undoable step."""
    if color is not None and clean_color(color) is None:
        raise ValueError(f"Geçersiz renk: {color!r} (#rrggbb olmalı)")
    color = clean_color(color)
    items = selected_blocks(scene) if items is None else items
    changed = 0
    for it in items:
        if it.node.color != color:
            it.node.color = color
            it.update()
            changed += 1
    if changed:
        scene.edited.emit()
    return changed


# --- wires: drop on empty canvas, insert a block into a wire ---------------------------------------------------------

def best_port(ports, ptype: str, ports_are_inputs: bool) -> str | None:
    """The port among `ports` that a wire of type `ptype` fits best: the same type first, then a compatible one."""
    def fits(t):
        return PortType.compatible(ptype, t) if ports_are_inputs else PortType.compatible(t, ptype)
    exact = next((n for n, t in ports if t == ptype), None)
    if exact is not None:
        return exact
    typed = next((n for n, t in ports if fits(t) and PortType.ANY not in (t, ptype)), None)
    return typed or next((n for n, t in ports if fits(t)), None)


def compatible_blocks(ptype: str, need_input: bool) -> dict[str, str]:
    """{type_name: port} of every block that can take (need_input) or give a wire of type `ptype`."""
    out = {}
    for _, classes in by_category().items():
        for cls in classes:
            port = best_port(cls.inputs if need_input else cls.outputs, ptype, need_input)
            if port is not None:
                out[cls.type_name] = port
    return out


def add_for_wire(scene, pending: tuple, type_name: str, pos: QPointF) -> NodeItem:
    """Add `type_name` where a wire was dropped and connect it to the wire's socket.
    pending = (node id, port name, the port is an output)."""
    nid, port, is_out = pending
    src = scene.items_by_id.get(nid)
    sock = src.sockets.get((port, is_out)) if src is not None else None
    cls = registry.NODE_REGISTRY.get(type_name)
    ports = (cls.inputs if is_out else cls.outputs) if cls is not None else []
    pname = best_port(ports, sock.ptype, is_out) if sock is not None else None
    row = next((i for i, (n, _) in enumerate(ports) if n == pname), 0)
    dy = HEADER_H + ROW_H * row + ROW_H / 2 + 3
    at = QPointF(pos.x() + 20, pos.y() - dy) if is_out else QPointF(pos.x() - NODE_W - 20, pos.y() - dy)
    item = scene.add_node(type_name, at)
    if sock is None or sock.scene() is not scene:
        return item
    node = item.node                           # instance ports (a sub-flow's differ from its class)
    pname = best_port(node.inputs if is_out else node.outputs, sock.ptype, is_out)
    new_sock = item.sockets.get((pname, not is_out)) if pname else None
    if new_sock is None:
        scene.message.emit(f"'{node.title}' bu bağlantıya uygun bir uç içermiyor")
        return item
    out, inp = (sock, new_sock) if is_out else (new_sock, sock)
    scene.try_connect(out, inp)
    return item


def on_wire_dropped(win, sock, pos: QPointF) -> bool:
    """A wire let go on empty canvas: offer the blocks that fit it; the chosen one is added and connected."""
    allowed = compatible_blocks(sock.ptype, need_input=sock.is_output)
    if not allowed:
        win.statusBar().showMessage("Bu bağlantıya uyan blok yok", 4000)
        return False
    win._pending_wire = (sock.node_item.node.id, sock.name, sock.is_output)
    win._quick_pos = QPointF(pos)
    gp = win.view.mapToGlobal(win.view.mapFromScene(pos))
    win.quick_add.popup(gp, allowed=set(allowed))
    return True


def insert_ports(node, src_type: str, dst_type: str) -> tuple[str, str] | None:
    """(input, output) of `node` that a wire src_type → dst_type can be routed through, or None."""
    i = best_port(node.inputs, src_type, True)
    o = best_port(node.outputs, dst_type, False)
    return (i, o) if i is not None and o is not None else None


def _can_insert(scene, item) -> bool:
    return (isinstance(item, NodeItem) and item.scene() is scene and not item.is_note and not item.edges
            and selected_blocks(scene) == [item])


def update_insert_target(scene, item) -> EdgeItem | None:
    """While a single unconnected block is dragged: highlight the wire it would be inserted into."""
    edge = None
    if _can_insert(scene, item):
        r = QRectF(item.pos().x(), item.pos().y(), NODE_W, item.height)
        area = QPainterPath()
        area.addRect(r)
        for it in scene.items(r):
            if isinstance(it, EdgeItem) and it.dst is not None and it.conn is not None \
                    and it.path().intersects(area) and insert_ports(item.node, it.src.ptype, it.dst.ptype):
                edge = it
                break
    if edge is not scene.insert_edge:
        if scene.insert_edge is not None and scene.insert_edge.scene() is scene:
            scene.insert_edge.highlight = False
            scene.insert_edge.update()
        scene.insert_edge = edge
        if edge is not None:
            edge.highlight = True
            edge.update()
    return edge


def insert_into_edge(scene, item: NodeItem, edge: EdgeItem) -> bool:
    """Route `edge` through `item`: src → item input, item output → dst (one structure change)."""
    ports = insert_ports(item.node, edge.src.ptype, edge.dst.ptype) if edge.conn else None
    if ports is None:
        return False
    c = edge.conn
    scene._remove_edge(edge)
    added = []
    try:
        added.append(scene.graph.connect(c.src, c.src_port, item.node.id, ports[0]))
        added.append(scene.graph.connect(item.node.id, ports[1], c.dst, c.dst_port))
    except GraphError as exc:
        for a in added:
            scene.graph.disconnect(a)
        scene._add_edge_item(scene.graph.connect(c.src, c.src_port, c.dst, c.dst_port))
        scene.message.emit(f"Bağlantıya eklenemedi: {exc}")
        return False
    for a in added:
        scene._add_edge_item(a)
    scene.structure_changed.emit()
    scene.message.emit(f"'{item.node.title}' bağlantının arasına eklendi")
    return True


def finish_insert(scene, item) -> bool:
    edge, scene.insert_edge = scene.insert_edge, None
    if edge is None:
        return False
    edge.highlight = False
    edge.update()
    if item is None or not _can_insert(scene, item) or edge.scene() is not scene:
        return False
    return insert_into_edge(scene, item, edge)


# --- errors ---------------------------------------------------------------------------------------------------------

def error_items(scene) -> list[NodeItem]:
    """Blocks showing an error of their own, top to bottom then left to right. Blocks that only failed because a
    block before them did are left out (the cause is the one to look at)."""
    return sorted((it for it in scene.items_by_id.values() if it.error and it.error != "Önceki blok hata verdi"),
                  key=lambda it: (round(it.pos().y()), it.pos().x(), it.node.id))


def next_error(scene, current_id: str | None) -> NodeItem | None:
    items = error_items(scene)
    if not items:
        return None
    ids = [it.node.id for it in items]
    if current_id in ids:
        return items[(ids.index(current_id) + 1) % len(items)]
    return items[0]


def goto_next_error(win) -> NodeItem | None:
    cur = win.scene.selected_node()
    it = next_error(win.scene, cur.id if cur is not None else None)
    if it is None:
        win.statusBar().showMessage("Hata gösteren blok yok", 3000)
        return None
    win.focus_node(it.node.id)
    n = len(error_items(win.scene))
    win.statusBar().showMessage(f"Hatalı blok ({n} tane): {it.node.title}: {it.error}", 6000)
    return it


# --- search the flow --------------------------------------------------------------------------------------------------

def fuzzy_score(query: str, text: str) -> int | None:
    """Lower is better; None = no match. Every word of the query must appear in `text`, as a substring (best) or
    as letters in order (fuzzy, e.g. "esk" finds "Eşik"). Turkish letters and case are ignored."""
    from .panels import search_fold
    t = search_fold(text)
    total = 0
    for word in search_fold(query).split():
        i = t.find(word)
        if i >= 0:
            total += i
            continue
        pos, gaps, start = -1, 0, None
        for ch in word:
            j = t.find(ch, pos + 1)
            if j < 0:
                return None
            if start is None:
                start = j
            elif j > pos + 1:
                gaps += j - pos - 1
            pos = j
        total += 1000 + gaps * 10 + start
    return total


class FlowSearch(QWidget):
    """Ctrl+P: the blocks of the canvas (this level of the flow), filtered as you type; Enter jumps to one."""
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.edit = QLineEdit(placeholderText="Akışta ara: blok adı veya türü…  (Enter: git)")
        self.list = QListWidget()
        self.list.setMinimumHeight(300)
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.entries: list[tuple[str, str, str, str]] = []
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._accept)
        self.list.itemActivated.connect(lambda _: self._accept())
        self.edit.installEventFilter(self)

    def eventFilter(self, obj, e):
        if e.type() == QEvent.KeyPress and e.key() in (Qt.Key_Down, Qt.Key_Up):
            row = self.list.currentRow() + (1 if e.key() == Qt.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
            return True
        return super().eventFilter(obj, e)

    def load(self, scene):
        self.entries = sorted(((it.node.id, it.node.title, it.node.type_name, it.error or "")
                               for it in scene.items_by_id.values()), key=lambda e: (search_key(e[1]), e[0]))

    def _filter(self, text):
        self.list.clear()
        scored = []
        for k, (nid, title, tn, err) in enumerate(self.entries):
            s = fuzzy_score(text, f"{title} {tn}")
            if s is not None:
                scored.append((s, k, nid, title, tn, err))
        for _, _, nid, title, tn, err in sorted(scored):
            it = QListWidgetItem(f"{'⛔ ' if err else ''}{title}    ·  {tn}" + (f"   — {err}" if err else ""))
            it.setData(Qt.UserRole, nid)
            if err:
                it.setForeground(QColor("#e53935"))
                it.setToolTip(err)
            self.list.addItem(it)
        self.list.setCurrentRow(0)

    def _accept(self):
        it = self.list.currentItem()
        self.close()
        if it:
            self.chosen.emit(it.data(Qt.UserRole))

    def popup(self, scene, global_pos):
        self.load(scene)
        self.edit.blockSignals(True)
        self.edit.clear()
        self.edit.blockSignals(False)
        self._filter("")
        self.move(global_pos)
        self.show()
        self.edit.setFocus()


def search_key(text: str) -> str:
    from .panels import search_fold
    return search_fold(text)


def show_flow_search(win):
    if not win.scene.items_by_id:
        win.statusBar().showMessage("Akışta blok yok", 3000)
        return None
    v = win.view
    gp = v.mapToGlobal(v.viewport().rect().center()) - QPointF(210, 170).toPoint()
    win.flow_search.popup(win.scene, gp)
    return win.flow_search


# --- mini map ----------------------------------------------------------------------------------------------------------

class MiniMap(QWidget):
    """Overview of the whole canvas in the bottom-right corner of the view; click or drag to move the view."""
    W, H, MARGIN = 220, 150, 12

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.setFixedSize(self.W, self.H)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Mini harita: tıklayın veya sürükleyin")
        self._drag_map = None
        view.installEventFilter(self)
        view.horizontalScrollBar().valueChanged.connect(self.update)
        view.verticalScrollBar().valueChanged.connect(self.update)
        view.scene().changed.connect(self._scene_changed)
        self._timer = QTimer(self, interval=500, timeout=self.update)   # zoom changes move no scroll bar
        self._place()

    def _scene_changed(self, *_):
        if self.isVisible():
            self.update()

    def showEvent(self, e):
        self._place()
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def eventFilter(self, obj, e):
        if obj is self.view and e.type() == QEvent.Resize:
            self._place()
        return False

    def _place(self):
        self.move(max(0, self.view.width() - self.W - self.MARGIN), max(0, self.view.height() - self.H - self.MARGIN))
        self.raise_()

    def visible_rect(self) -> QRectF:
        return self.view.mapToScene(self.view.viewport().rect()).boundingRect()

    def mapping(self) -> tuple[QRectF, float, float, float]:
        """(world rect, scale, x offset, y offset) from scene to widget coordinates."""
        world = blocks_rect(self.view.scene())
        vis = self.visible_rect()
        world = vis if world.isNull() else world.united(vis)
        world = world.adjusted(-40, -40, 40, 40)
        s = min((self.W - 8) / max(world.width(), 1), (self.H - 8) / max(world.height(), 1))
        ox = (self.W - world.width() * s) / 2
        oy = (self.H - world.height() * s) / 2
        return world, s, ox, oy

    def to_scene(self, p, m=None) -> QPointF:
        world, s, ox, oy = m or self.mapping()
        return QPointF(world.left() + (p.x() - ox) / s, world.top() + (p.y() - oy) / s)

    def paintEvent(self, e):
        c = theme.CANVAS
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg = QColor(c["minimap_bg"])
        bg.setAlpha(225)
        p.setPen(QPen(QColor(c["minimap_border"]), 1))
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(0.5, 0.5, self.W - 1, self.H - 1), 6, 6)
        world, s, ox, oy = self._drag_map or self.mapping()

        def tr(r: QRectF) -> QRectF:
            return QRectF(ox + (r.left() - world.left()) * s, oy + (r.top() - world.top()) * s,
                          max(2.0, r.width() * s), max(2.0, r.height() * s))
        sc = self.view.scene()
        p.setPen(Qt.NoPen)
        for it in sc.items():
            if isinstance(it, FrameItem):
                col = QColor(it.frame["color"])
                col.setAlpha(80)
                p.setBrush(col)
                p.drawRect(tr(it.rect_in_scene()))
        for it in sc.items_by_id.values():
            col = QColor("#ff5252") if it.error else (QColor(it.node.color) if it.node.color else it.color)
            p.setBrush(col)
            p.drawRect(tr(QRectF(it.pos().x(), it.pos().y(), NODE_W, it.height)))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(c["selected"]), 1.5))
        p.drawRect(tr(self.visible_rect()))
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_map = self.mapping()          # fixed while dragging, else the map shifts under the mouse
            self.view.centerOn(self.to_scene(e.position(), self._drag_map))
            self.update()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_map is not None:
            self.view.centerOn(self.to_scene(e.position(), self._drag_map))
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag_map = None
        self.update()


# --- export as an image ------------------------------------------------------------------------------------------------

def blocks_rect(scene) -> QRectF:
    r = QRectF()
    for it in scene.items():
        if isinstance(it, (NodeItem, FrameItem)) and it.parentItem() is None:
            r = r.united(it.sceneBoundingRect())
    return r


MAX_SIDE = 8000
MAX_PIXELS = 40_000_000          # ~160 MB of RGBA at most, however big the flow


def export_image(scene, path, scale: float = 2.0, margin: float = 40) -> pathlib.Path:
    """Save the whole canvas (all blocks, wires and frames) as PNG (any raster format Qt writes by extension) or
    SVG. Raises ValueError with a Turkish message for an empty flow or an unsupported file type."""
    path = pathlib.Path(path)
    ext = path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".svg"):
        raise ValueError("Desteklenmeyen dosya türü: .png veya .svg seçin")
    rect = blocks_rect(scene)
    if rect.isNull() or rect.isEmpty():
        raise ValueError("Akışta dışa aktarılacak blok yok")
    rect = rect.adjusted(-margin, -margin, margin, margin)
    selected = scene.selectedItems()
    scene.blockSignals(True)                     # deselect for a clean picture without telling the window
    try:
        for it in selected:
            it.setSelected(False)
        if ext == ".svg":
            try:
                from PySide6.QtSvg import QSvgGenerator
            except ImportError as exc:
                raise ValueError("SVG kaydı için PySide6 QtSvg modülü gerekli; PNG olarak kaydedin") from exc
            w, h = int(rect.width()), int(rect.height())
            gen = QSvgGenerator()
            gen.setFileName(str(path))
            gen.setSize(QSize(w, h))
            gen.setViewBox(QRect(0, 0, w, h))
            gen.setTitle("VisionFlow akışı")
            p = QPainter(gen)
            scene.render(p, QRectF(0, 0, w, h), rect)
            p.end()
        else:
            s = min(scale, MAX_SIDE / max(rect.width(), rect.height()),
                    (MAX_PIXELS / (rect.width() * rect.height())) ** 0.5)
            s = max(0.05, s)
            img = QImage(max(1, int(rect.width() * s)), max(1, int(rect.height() * s)), QImage.Format_ARGB32)
            img.fill(QColor(theme.CANVAS["bg"]))
            p = QPainter(img)
            p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
            scene.render(p, QRectF(img.rect()), rect)
            p.end()
            if not img.save(str(path)):
                raise OSError(f"Resim kaydedilemedi: {path}")
    finally:
        for it in selected:
            if it.scene() is scene:
                it.setSelected(True)
        scene.blockSignals(False)
        scene.update()
    if not path.exists() or path.stat().st_size == 0:
        raise OSError(f"Resim kaydedilemedi: {path}")
    return path


def export_dialog(win, path: str | None = None):
    if not win.scene.items_by_id and not win.scene.frames():
        win.statusBar().showMessage("Akışta dışa aktarılacak blok yok", 4000)
        return None
    if not path:
        base = pathlib.Path(win.path).stem if win.path else "akis"
        path, _ = QFileDialog.getSaveFileName(win, "Akışı resim olarak kaydet", f"{base}.png",
                                              "PNG resmi (*.png);;SVG çizimi (*.svg)")
        if not path:
            return None
        if not pathlib.Path(path).suffix:
            path += ".png"
    try:
        out = export_image(win.scene, path)
    except (ValueError, OSError) as exc:
        QMessageBox.warning(win, "Dışa aktarılamadı", str(exc))
        return None
    win.statusBar().showMessage(f"Akış resmi kaydedildi: {out}", 5000)
    return out


# --- theme -------------------------------------------------------------------------------------------------------------

def apply_theme(win, name: str, save: bool = True) -> str:
    name = theme.set_theme(name, QApplication.instance())
    for it in win.scene.items():
        if isinstance(it, SocketItem):                                  # sockets keep their pen
            it.setPen(QPen(QColor(theme.CANVAS["socket_pen"]), 2))
    win.scene.update()
    win.view.viewport().update()
    if getattr(win, "minimap", None) is not None:
        win.minimap.update()
    for key, act in getattr(win, "theme_acts", {}).items():
        act.setChecked(key == name)
    if save:
        win.settings.setValue("theme", name)
    return name


# --- window glue ---------------------------------------------------------------------------------------------------

def _menu(win, title: str):
    for a in win.menuBar().actions():
        if a.menu() is not None and a.text().replace("&", "") == title:
            return a.menu()
    return win.menuBar().addMenu(title)


def _add_align_menu(win, menu):
    sub = menu.addMenu("Hizala")
    for mode, label in ALIGN_MODES.items():
        sub.addAction(label, lambda m=mode: arrange_selected(win.scene, m))
    sub.addSeparator()
    for mode, label in DISTRIBUTE_MODES.items():
        sub.addAction(label, lambda m=mode: arrange_selected(win.scene, m))
    return sub


def _add_color_menu(win, menu):
    sub = menu.addMenu("Blok rengi")
    sub.addAction("Renksiz (kategori rengi)", lambda: set_block_color(win.scene, None))
    for name, hexc in BLOCK_COLORS.items():
        from PySide6.QtGui import QIcon, QPixmap
        pm = QPixmap(14, 14)
        pm.fill(QColor(hexc))
        sub.addAction(QIcon(pm), name, lambda c=hexc: set_block_color(win.scene, c))
    return sub


def extend_context_menu(win, menu):
    """Right-click entries for the selected blocks: Hizala (2+) and Blok rengi (1+)."""
    n = len(selected_blocks(win.scene))
    if n >= 2:
        _add_align_menu(win, menu)
    if n >= 1:
        _add_color_menu(win, menu)
    if n:
        menu.addSeparator()


def _quick_chosen(win, type_name: str):
    pending, win._pending_wire = getattr(win, "_pending_wire", None), None
    if pending:
        return add_for_wire(win.scene, pending, type_name, win._quick_pos)
    return win.scene.add_node(type_name, win._quick_pos)


def set_snap(win, on: bool):
    win.scene.snap_grid = GRID if on else 0
    win.settings.setValue("snap_grid", bool(on))
    win.snap_act.setChecked(bool(on))


def set_minimap(win, on: bool):
    win.minimap.setVisible(bool(on))
    win.settings.setValue("minimap", bool(on))
    win.minimap_act.setChecked(bool(on))


def install(win):
    """Add the 2.0 editor tools to the main window's menus, tool bar and canvas."""
    s = win.settings
    win._pending_wire = None
    win.flow_search = FlowSearch(win)
    win.flow_search.chosen.connect(win.focus_node)
    win.minimap = MiniMap(win.view)
    win.scene.wire_dropped.connect(lambda sock, pos: on_wire_dropped(win, sock, pos))

    def act(menu, text, slot, shortcut=None, checkable=False, canvas_only=False):
        a = QAction(text, win, checkable=checkable)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if canvas_only:
            a.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            win.view.addAction(a)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    f = _menu(win, "Dosya")
    quit_act = next((a for a in f.actions() if a.text() == "Çıkış"), None)
    exp = QAction("Akışı resim olarak kaydet…", win)
    exp.triggered.connect(lambda: export_dialog(win))
    f.insertAction(quit_act, exp) if quit_act else f.addAction(exp)
    win.export_act = exp

    e = _menu(win, "Düzen")
    e.addSeparator()
    _add_align_menu(win, e)
    _add_color_menu(win, e)
    win.search_flow_act = act(e, "Akışta ara…", lambda: show_flow_search(win), "Ctrl+P")
    win.goto_error_act = act(e, "⛔ Hatalı bloğa git", lambda: goto_next_error(win), "F8")
    win.goto_error_act.setToolTip("Hata gösteren bloklar arasında gez (F8)")
    tb = win.findChild(QToolBar, "Ana")
    if tb is not None:
        tb.addAction(win.goto_error_act)

    v = _menu(win, "Görünüm")
    v.addSeparator()
    win.snap_act = act(v, "Izgaraya yapıştır (20 px)", lambda on: set_snap(win, on), checkable=True)
    win.minimap_act = act(v, "Mini harita", lambda on: set_minimap(win, on), checkable=True)
    tm = v.addMenu("Tema")
    group = QActionGroup(win)
    group.setExclusive(True)
    win.theme_acts = {}
    for key, label in theme.THEMES.items():
        a = QAction(label, win, checkable=True)
        a.triggered.connect(lambda _=False, k=key: apply_theme(win, k))
        group.addAction(a)
        tm.addAction(a)
        win.theme_acts[key] = a

    set_snap(win, s.value("snap_grid", False, type=bool))
    set_minimap(win, s.value("minimap", True, type=bool))
    saved = str(s.value("theme", "dark") or "dark")
    saved = saved if saved in theme.CANVAS_THEMES else "dark"
    if saved != theme.current_theme:
        apply_theme(win, saved, save=False)
    else:                                         # already in use (the dark stylesheet is set by __main__)
        win.theme_acts[saved].setChecked(True)
