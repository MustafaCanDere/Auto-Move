"""Node editor canvas: node items, sockets, bezier edges, the scene and the zoomable view."""
from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsScene,
                               QGraphicsView)

from ..core import Graph, GraphError, Node, PortType, registry
from ..core.node import clean_color
from .theme import CANVAS, CATEGORY_COLORS, DEFAULT_CATEGORY_COLOR

NODE_W = 190
HEADER_H = 26
ROW_H = 22
STATUS_H = 20
THUMB_H = 108
SOCKET_R = 6
MIME = "application/x-visionflow-node"


class SocketItem(QGraphicsEllipseItem):
    def __init__(self, parent: "NodeItem", name: str, ptype: str, is_output: bool, y: float):
        super().__init__(-SOCKET_R, -SOCKET_R, 2 * SOCKET_R, 2 * SOCKET_R, parent)
        self.node_item, self.name, self.ptype, self.is_output = parent, name, ptype, is_output
        self.setPos(NODE_W if is_output else 0, y)
        self.setBrush(QColor(PortType.COLORS.get(ptype, "#bdbdbd")))
        self.setPen(QPen(QColor(CANVAS["socket_pen"]), 2))
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip(f"{name} ({ptype})")
        self.setZValue(2)

    def hoverEnterEvent(self, e):
        self.setScale(1.4)

    def hoverLeaveEvent(self, e):
        self.setScale(1.0)

    def center(self) -> QPointF:
        return self.scenePos()


class EdgeItem(QGraphicsPathItem):
    highlight = False      # a dragged block would be inserted into this wire on release

    def __init__(self, src: SocketItem, dst: SocketItem | None, conn=None):
        super().__init__()
        self.src, self.dst, self.conn = src, dst, conn
        self.end_pos = src.center()
        self.color = QColor(PortType.COLORS.get(src.ptype, "#bdbdbd"))
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.ItemIsSelectable, dst is not None)
        self.update_path()

    def update_path(self):
        a = self.src.center()
        b = self.dst.center() if self.dst else self.end_pos
        dx = max(abs(b.x() - a.x()) * 0.5, 40)
        path = QPainterPath(a)
        path.cubicTo(a + QPointF(dx, 0), b - QPointF(dx, 0), b)
        self.setPath(path)

    def paint(self, painter, option, widget=None):
        pen = QPen(QColor(CANVAS["edge_selected"]) if self.isSelected() else self.color,
                   3 if self.isSelected() else 2.2)
        if self.highlight:
            pen = QPen(QColor("#ffd54f"), 5)
        if self.dst is None:
            pen.setStyle(Qt.DashLine)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pen)
        painter.drawPath(self.path())

    def shape(self):
        stroker = QPainterPath(self.path())
        from PySide6.QtGui import QPainterPathStroker
        s = QPainterPathStroker()
        s.setWidth(10)
        return s.createStroke(stroker)


class NodeItem(QGraphicsItem):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        self.edges: list[EdgeItem] = []
        self.status_text = ""
        self.error: str | None = None
        self.ms: float | None = None
        self.thumb: QPixmap | None = None
        self.has_image_out = any(t == PortType.IMAGE for _, t in node.outputs)
        self.show_thumb = True
        self.color = QColor(CATEGORY_COLORS.get(node.category, DEFAULT_CATEGORY_COLOR))
        rows = max(len(node.inputs), len(node.outputs), 1)
        self.body_h = HEADER_H + rows * ROW_H + 6
        self.sockets: dict[tuple[str, bool], SocketItem] = {}
        for i, (name, t) in enumerate(node.inputs):
            self.sockets[(name, False)] = SocketItem(self, name, t, False, HEADER_H + ROW_H * i + ROW_H / 2 + 3)
        for i, (name, t) in enumerate(node.outputs):
            self.sockets[(name, True)] = SocketItem(self, name, t, True, HEADER_H + ROW_H * i + ROW_H / 2 + 3)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.setPos(*node.pos)
        self.setToolTip(node.description or node.title)

    @property
    def is_note(self) -> bool:
        return self.node.type_name == "Note"

    @property
    def is_subflow(self) -> bool:
        return self.node.type_name == "SubFlow"

    def _note_lines(self) -> list[str]:
        return str(self.node.values.get("text", "")).splitlines() or [""]

    @property
    def height(self) -> float:
        if self.is_note:
            f = QFont()
            f.setPointSize(9)
            r = QFontMetrics(f).boundingRect(0, 0, NODE_W - 20, 10000, Qt.TextWordWrap,
                                             "\n".join(self._note_lines()))
            return max(60, r.height() + 22)
        h = self.body_h + STATUS_H
        if self.has_image_out and self.show_thumb:
            h += THUMB_H
        return h

    def boundingRect(self) -> QRectF:
        return QRectF(-2, -2, NODE_W + 10, self.height + 10)

    def set_thumb_visible(self, on: bool):
        self.prepareGeometryChange()
        self.show_thumb = on
        self.update()

    def refresh(self):
        """Call after values change (note text, enabled flag) so size and look update."""
        self.prepareGeometryChange()
        for e in self.edges:
            e.update_path()
        self.update()

    def _paint_note(self, p: QPainter):
        rect = QRectF(0, 0, NODE_W, self.height)
        p.setPen(QPen(QColor("#4fc3f7") if self.isSelected() else QColor("#c9a227"), 1.5))
        p.setBrush(QColor("#f6e58d"))
        p.drawRoundedRect(rect, 4, 4)
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(QColor("#3d3500"))
        p.drawText(QRectF(10, 10, NODE_W - 20, self.height - 16), Qt.TextWordWrap | Qt.AlignTop,
                   "\n".join(self._note_lines()))

    def paint(self, p: QPainter, option, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        if self.is_note:
            self._paint_note(p)
            return
        if not self.node.enabled:
            p.setOpacity(0.45)
        rect = QRectF(0, 0, NODE_W, self.height)
        if self.is_subflow:                    # a stack of cards: there are more blocks inside
            p.setPen(QPen(QColor(CANVAS["border"]), 1.5))
            p.setBrush(QColor(CANVAS["subflow_back"]))
            p.drawRoundedRect(rect.translated(6, 6), 8, 8)
        # body
        is_sel = self.isSelected()
        border_col = QColor("#ef4444") if self.error else (QColor(CANVAS["selected"]) if is_sel else QColor(CANVAS["border"]))
        border_w = 2.5 if is_sel else (2.0 if self.error else 1.2)
        p.setPen(QPen(border_col, border_w))
        p.setBrush(QColor(CANVAS["body"]))
        p.drawRoundedRect(rect, 8, 8)
        # header
        hp = QPainterPath()
        hp.addRoundedRect(QRectF(0, 0, NODE_W, HEADER_H), 8, 8)
        hp.addRect(QRectF(0, HEADER_H - 8, NODE_W, 8))
        head = QColor(self.node.color) if getattr(self.node, "color", None) else self.color
        p.fillPath(hp.simplified(), head)
        if head is not self.color:             # colour label: keep a stripe of the category colour
            p.fillRect(QRectF(0, HEADER_H - 4, NODE_W, 4), self.color)
        light = 0.299 * head.red() + 0.587 * head.green() + 0.114 * head.blue() > 160
        p.setPen(QColor("#0f172a") if light else QColor("white"))
        f = QFont()
        f.setBold(True)
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(QRectF(10, 0, NODE_W - 20, HEADER_H), Qt.AlignVCenter,
                   p.fontMetrics().elidedText(self.node.title, Qt.ElideRight, NODE_W - 20))
        # port labels
        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor(CANVAS["port_text"]))
        for (name, is_out), s in self.sockets.items():
            y = s.pos().y()
            r = QRectF(12, y - 9, NODE_W - 24, 18)
            p.drawText(r, Qt.AlignVCenter | (Qt.AlignRight if is_out else Qt.AlignLeft), name)
        y = self.body_h
        # thumbnail
        if self.has_image_out and self.show_thumb:
            area = QRectF(8, y, NODE_W - 16, THUMB_H - 6)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(CANVAS["thumb_bg"]))
            p.drawRoundedRect(area, 6, 6)
            if self.thumb is not None:
                pm = self.thumb.scaled(area.size().toSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                p.drawPixmap(int(area.center().x() - pm.width() / 2), int(area.center().y() - pm.height() / 2), pm)
            y += THUMB_H
        # status line / latency pill
        status_rect = QRectF(8, y - 1, NODE_W - 16, STATUS_H - 2)
        if self.error:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(239, 68, 68, 40))
            p.drawRoundedRect(status_rect, 4, 4)
            p.setPen(QColor("#f87171"))
        elif self.ms is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(14, 165, 233, 25))
            p.drawRoundedRect(status_rect, 4, 4)
            p.setPen(QColor(CANVAS["status"]))
        else:
            p.setPen(QColor(CANVAS["status"]))
        text = self.error or self.status_text or ("çift tıklayıp içini açın" if self.is_subflow else "")
        if not self.node.enabled:
            text = "devre dışı (atlanıyor)"
        elif self.ms is not None and not self.error:
            text = f"⚡ {self.ms:.1f} ms  {text}"
        metrics = p.fontMetrics()
        p.drawText(QRectF(12, y - 2, NODE_W - 24, STATUS_H), Qt.AlignVCenter,
                   metrics.elidedText(text, Qt.ElideRight, NODE_W - 24))

    def set_result(self, ms, error, text, qimg: QImage | None):
        self.ms, self.error, self.status_text = ms, error, text or ""
        if qimg is not None:
            self.thumb = QPixmap.fromImage(qimg)
        self.setToolTip(error or self.node.description or self.node.title)
        self.update()

    def clear_result(self):
        self.set_result(None, None, "", None)
        self.thumb = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        if self.is_subflow and self.scene() is not None:
            self.scene().subflow_open.emit(self.node)
            return
        super().mouseDoubleClickEvent(e)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            sc = self.scene()
            # snap only while the user drags blocks (not on load / undo / align)
            if sc is not None and getattr(sc, "snap_grid", 0) and isinstance(sc.mouseGrabberItem(), NodeItem):
                from .editor_tools import snap_point
                return snap_point(value, sc.snap_grid)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node.pos = (self.pos().x(), self.pos().y())
            for e in self.edges:
                e.update_path()
        return super().itemChange(change, value)


FRAME_TITLE_H = 28
FRAME_HANDLE = 16


class FrameItem(QGraphicsItem):
    """A comment frame behind the blocks: a titled, coloured area that groups and explains part of a flow.
    Drag it by its title bar and the blocks inside move with it; drag the bottom-right corner to resize."""

    def __init__(self, frame: dict):
        super().__init__()
        self.frame = frame                     # the graph's own dict: edits land straight in the saved flow
        x, y, w, h = frame["rect"]
        self.w, self.h = w, h
        self.setPos(x, y)
        self.setZValue(-10)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self._carry: list[QGraphicsItem] = []
        self._resizing = False
        self._start: tuple | None = None

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.w, self.h)

    def shape(self) -> QPainterPath:
        """Only the title bar and the resize grip are "the frame": a rubber band drawn around blocks inside it must
        not select (and then delete or drag) the frame too."""
        path = QPainterPath()
        path.addRect(QRectF(0, 0, self.w, FRAME_TITLE_H))
        path.addRect(QRectF(self.w - FRAME_HANDLE, self.h - FRAME_HANDLE, FRAME_HANDLE, FRAME_HANDLE))
        return path

    def paint(self, p: QPainter, option, widget=None):
        c = QColor(self.frame["color"])
        body = QColor(c)
        body.setAlpha(45)
        p.setPen(QPen(QColor(CANVAS["edge_selected"]) if self.isSelected() else c.lighter(130),
                      2 if self.isSelected() else 1.2))
        p.setBrush(body)
        p.drawRoundedRect(self.boundingRect(), 8, 8)
        head = QColor(c)
        head.setAlpha(170)
        p.setPen(Qt.NoPen)
        p.setBrush(head)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.w, FRAME_TITLE_H), 8, 8)
        path.addRect(QRectF(0, FRAME_TITLE_H / 2, self.w, FRAME_TITLE_H / 2))
        p.drawPath(path.simplified())
        p.setPen(QColor(CANVAS["frame_title"]))
        f = QFont()
        f.setBold(True)
        f.setPointSizeF(10.5)
        p.setFont(f)
        lines = (self.frame["title"] or "Yorum (başlığa çift tıklayıp yazın)").split("\n")
        p.drawText(QRectF(10, 0, self.w - 20, FRAME_TITLE_H), Qt.AlignVCenter,
                   QFontMetrics(f).elidedText(lines[0], Qt.ElideRight, int(self.w - 20)))
        if len(lines) > 1:                     # the rest of the comment is written inside the frame
            f.setBold(False)
            f.setPointSizeF(9.5)
            p.setFont(f)
            p.setPen(QColor(CANVAS["frame_text"]))
            p.drawText(QRectF(10, FRAME_TITLE_H + 6, self.w - 20, self.h - FRAME_TITLE_H - 12),
                       Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, "\n".join(lines[1:]))
        p.setPen(QPen(c.lighter(150), 1.5))
        for k in (5, 10):                      # resize grip
            p.drawLine(QPointF(self.w - k, self.h - 3), QPointF(self.w - 3, self.h - k))

    def rect_in_scene(self) -> QRectF:
        return QRectF(self.pos().x(), self.pos().y(), self.w, self.h)

    def _store(self):
        self.frame["rect"] = [round(self.pos().x(), 1), round(self.pos().y(), 1), round(self.w, 1), round(self.h, 1)]

    def _in_handle(self, pos) -> bool:
        return pos.x() >= self.w - FRAME_HANDLE and pos.y() >= self.h - FRAME_HANDLE

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        pos = e.pos()
        if not (pos.y() <= FRAME_TITLE_H or self._in_handle(pos)):
            e.ignore()                         # the body is free for rubber-band selection and the canvas
            return
        self._start = (self.pos().x(), self.pos().y(), self.w, self.h)
        if self._in_handle(pos):
            self._resizing = True
            self.setSelected(True)
            e.accept()
            return
        area = self.rect_in_scene()
        sc = self.scene()
        # blocks (and smaller frames) inside move with the frame, unless they are selected (they move anyway)
        self._carry = [it for it in sc.items(area, Qt.ContainsItemBoundingRect)
                       if it is not self and isinstance(it, (NodeItem, FrameItem)) and it.parentItem() is None]
        self._last = self.pos()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._resizing:
            self.prepareGeometryChange()
            self.w = max(120.0, e.pos().x())
            self.h = max(70.0, e.pos().y())
            self.update()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._resizing = False
        self._carry = []
        super().mouseReleaseEvent(e)
        self._store()
        if self._start and self._start != (self.pos().x(), self.pos().y(), self.w, self.h) and self.scene():
            self.scene().edited.emit()
        self._start = None

    def mouseDoubleClickEvent(self, e):
        if e.pos().y() <= FRAME_TITLE_H and self.scene():
            self.scene().rename_frame(self)
            return
        super().mouseDoubleClickEvent(e)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            last = getattr(self, "_last", None)
            if last is not None and self._carry:
                d = self.pos() - last
                for it in self._carry:
                    if not it.isSelected() and it.scene() is self.scene():
                        it.moveBy(d.x(), d.y())
            self._last = self.pos()
            self._store()
        return super().itemChange(change, value)


class NodeScene(QGraphicsScene):
    structure_changed = Signal()   # nodes/connections added or removed
    edited = Signal()              # anything undoable changed (incl. moves)
    message = Signal(str)
    subflow_open = Signal(object)  # a sub-flow block was double-clicked: show what is inside
    wire_dropped = Signal(object, object)  # a wire let go on empty canvas: (its SocketItem, scene QPointF)

    def __init__(self, graph: Graph | None = None):
        super().__init__()
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.graph = graph or Graph()
        self.items_by_id: dict[str, NodeItem] = {}
        self.drag_edge: EdgeItem | None = None
        self.show_thumbs = True
        self._press_positions: dict[str, tuple] = {}
        self.snap_grid = 0            # > 0: dragged blocks snap to this grid (px)
        self.insert_edge: EdgeItem | None = None   # wire a dragged block would be inserted into
        self.structure_changed.connect(self.edited)

    # graph <-> items -----------------------------------------------------------
    def load_graph(self, graph: Graph):
        self.clear()
        self.items_by_id.clear()
        self.insert_edge = None
        self.graph = graph
        for node in graph.nodes.values():
            self._add_item(node)
        for c in graph.connections:
            self._add_edge_item(c)
        for f in graph.frames:
            self.addItem(FrameItem(f))
        self.structure_changed.emit()

    # comment frames ------------------------------------------------------------
    def frames(self) -> list[FrameItem]:
        return [it for it in self.items() if isinstance(it, FrameItem)]

    def frame_at(self, pos) -> FrameItem | None:
        """The innermost frame whose area (not only its title) holds `pos`."""
        hits = [f for f in self.frames() if f.rect_in_scene().contains(pos)]
        return min(hits, key=lambda f: f.w * f.h) if hits else None

    def add_frame(self, rect: QRectF | None = None, title: str = "", at: QPointF | None = None) -> FrameItem:
        """A new frame around the selected blocks, or an empty one at `at`."""
        from ..core.graph import FRAME_COLORS, clean_frame
        if rect is None:
            sel = [it for it in self.selectedItems() if isinstance(it, NodeItem)]
            if sel:
                r = sel[0].sceneBoundingRect()
                for it in sel[1:]:
                    r = r.united(it.sceneBoundingRect())
                rect = r.adjusted(-24, -24 - FRAME_TITLE_H, 24, 24)
            else:
                at = at or QPointF(0, 0)
                rect = QRectF(at.x(), at.y(), 420, 260)
        color = FRAME_COLORS[len(self.graph.frames) % len(FRAME_COLORS)]
        ids = {f["id"] for f in self.graph.frames}
        n = len(ids) + 1
        while f"c{n}" in ids:
            n += 1
        frame = clean_frame({"id": f"c{n}", "title": title, "color": color,
                             "rect": [rect.x(), rect.y(), rect.width(), rect.height()]})
        self.graph.frames.append(frame)
        item = FrameItem(frame)
        self.addItem(item)
        self.edited.emit()
        return item

    def remove_frame(self, item: FrameItem):
        if item.frame in self.graph.frames:
            self.graph.frames.remove(item.frame)
        self.removeItem(item)
        self.edited.emit()

    def rename_frame(self, item: FrameItem, title: str | None = None):
        if title is None:
            from PySide6.QtWidgets import QInputDialog
            views = self.views()
            title, ok = QInputDialog.getMultiLineText(views[0] if views else None, "Yorum çerçevesi", "Başlık:",
                                                      item.frame["title"])
            if not ok:
                return
        item.frame["title"] = title.strip()[:200]
        item.update()
        self.edited.emit()

    def recolor_frame(self, item: FrameItem):
        from ..core.graph import FRAME_COLORS
        i = FRAME_COLORS.index(item.frame["color"]) if item.frame["color"] in FRAME_COLORS else -1
        item.frame["color"] = FRAME_COLORS[(i + 1) % len(FRAME_COLORS)]
        item.update()
        self.edited.emit()

    def _add_item(self, node: Node) -> NodeItem:
        item = NodeItem(node)
        item.show_thumb = self.show_thumbs
        self.addItem(item)
        self.items_by_id[node.id] = item
        return item

    def _add_edge_item(self, c):
        s = self.items_by_id[c.src].sockets[(c.src_port, True)]
        d = self.items_by_id[c.dst].sockets[(c.dst_port, False)]
        e = EdgeItem(s, d, c)
        self.addItem(e)
        s.node_item.edges.append(e)
        d.node_item.edges.append(e)
        return e

    def add_node(self, type_name: str, pos: QPointF, values: dict | None = None) -> NodeItem:
        node = registry.create(type_name)
        node.pos = (pos.x(), pos.y())
        if values:
            node.values.update(values)
        self.graph.add(node)
        item = self._add_item(node)
        self.clearSelection()
        item.setSelected(True)
        self.structure_changed.emit()
        return item

    def _remove_edge(self, e: EdgeItem):
        if e.conn:
            self.graph.disconnect(e.conn)
        for it in (e.src.node_item, e.dst.node_item if e.dst else None):
            if it and e in it.edges:
                it.edges.remove(e)
        self.removeItem(e)

    def delete_selected(self):
        changed = False
        for it in list(self.selectedItems()):
            if isinstance(it, FrameItem):
                if it.frame in self.graph.frames:
                    self.graph.frames.remove(it.frame)
                self.removeItem(it)
                changed = True
        for it in list(self.selectedItems()):
            if isinstance(it, EdgeItem) and it.scene() is self:
                self._remove_edge(it)
                changed = True
        for it in list(self.selectedItems()):
            if isinstance(it, NodeItem):
                for e in list(it.edges):
                    self._remove_edge(e)
                self.graph.remove(it.node.id)
                del self.items_by_id[it.node.id]
                self.removeItem(it)
                changed = True
        if changed:
            self.structure_changed.emit()

    # sub-flows -----------------------------------------------------------------------
    def group_selected(self, name: str = "Alt Akış") -> NodeItem | None:
        """Pack the selected blocks into one sub-flow block (one undoable step)."""
        from ..nodes import subflow
        ids = [it.node.id for it in self.selectedItems() if isinstance(it, NodeItem)]
        try:
            sub = subflow.group(self.graph, ids, name)
        except GraphError as exc:
            self.message.emit(str(exc))
            return None
        self.load_graph(self.graph)
        item = self.items_by_id[sub.id]
        item.setSelected(True)
        return item

    def ungroup(self, node: Node) -> list[NodeItem]:
        """Put a sub-flow's blocks back on the canvas in its place."""
        from ..nodes import subflow
        before = len(self.graph.warnings)
        try:
            ids = subflow.ungroup(self.graph, node.id)
        except GraphError as exc:
            self.message.emit(str(exc))
            return []
        for w in self.graph.warnings[before:]:
            self.message.emit(w)
        del self.graph.warnings[before:]
        self.load_graph(self.graph)
        items = [self.items_by_id[i] for i in ids if i in self.items_by_id]
        for it in items:
            it.setSelected(True)
        return items

    def duplicate_selected(self):
        clip = self.copy_selected()
        if clip:
            self.paste(clip)

    def copy_selected(self) -> dict | None:
        ids = {it.node.id for it in self.selectedItems() if isinstance(it, NodeItem)}
        if not ids:
            return None
        return {
            "format": "visionflow-clip",
            "nodes": [self.graph.nodes[i].to_dict() for i in ids],
            "connections": [c.to_dict() for c in self.graph.connections if c.src in ids and c.dst in ids],
        }

    def paste(self, clip: dict, at: QPointF | None = None):
        nodes = [n for n in clip.get("nodes", []) if isinstance(n, dict) and isinstance(n.get("type"), str)]
        if not nodes:
            return

        def pos(nd):
            try:
                return float(nd["pos"][0]), float(nd["pos"][1])
            except (KeyError, TypeError, ValueError, IndexError):
                return 0.0, 0.0
        minx = min(pos(n)[0] for n in nodes)
        miny = min(pos(n)[1] for n in nodes)
        offset = (at - QPointF(minx, miny)) if at is not None else QPointF(40, 40)
        id_map = {}
        self.clearSelection()
        for nd in nodes:
            try:
                node = registry.create(nd["type"])
            except KeyError as exc:
                self.message.emit(str(exc))
                continue
            params = {p.name: p for p in node.params}
            values = nd.get("values")
            for k, v in (values.items() if isinstance(values, dict) else []):
                if k in params:
                    try:
                        node.values[k] = params[k].coerce(v, clamp=True)
                    except (TypeError, ValueError):
                        pass
            node.enabled = not nd.get("disabled", False)
            node.color = clean_color(nd.get("color"))
            warnings: list[str] = []
            node.load_extra(nd, warnings)
            for w in warnings:
                self.message.emit(w)
            x, y = pos(nd)
            node.pos = (x + offset.x(), y + offset.y())
            if "id" in nd:
                id_map[nd["id"]] = node.id
            self.graph.add(node)
            self._add_item(node).setSelected(True)
        for cd in clip.get("connections", []) or []:
            try:
                if cd["src"] in id_map and cd["dst"] in id_map:
                    c = self.graph.connect(id_map[cd["src"]], cd["src_port"], id_map[cd["dst"]], cd["dst_port"])
                    self._add_edge_item(c)
            except (GraphError, KeyError, TypeError) as exc:     # a bad link must not leave half a paste
                self.message.emit(f"Bağlantı atlandı: {exc}")
        self.structure_changed.emit()

    def toggle_enabled_selected(self):
        items = [it for it in self.selectedItems() if isinstance(it, NodeItem) and not it.is_note]
        if not items:
            return
        on = not all(it.node.enabled for it in items)
        for it in items:
            it.node.enabled = on
            it.refresh()
        self.structure_changed.emit()

    def auto_layout(self):
        """Tidy the canvas: columns by flow depth, keeping the current top-left corner."""
        movable = [it for it in self.items_by_id.values() if not it.is_note]
        if not movable:
            return
        origin = (min(it.pos().x() for it in movable), min(it.pos().y() for it in movable))
        self.graph.auto_layout(origin=origin, heights={i: it.height for i, it in self.items_by_id.items()})
        for it in movable:
            it.setPos(*it.node.pos)
        self.edited.emit()

    def set_thumbs_visible(self, on: bool):
        self.show_thumbs = on
        for it in self.items_by_id.values():
            it.set_thumb_visible(on)
            for e in it.edges:
                e.update_path()

    def selected_node(self) -> Node | None:
        sel = [it for it in self.selectedItems() if isinstance(it, NodeItem)]
        return sel[0].node if len(sel) == 1 else None

    # edge dragging ---------------------------------------------------------------
    def _socket_at(self, pos) -> SocketItem | None:
        for it in self.items(pos):
            if isinstance(it, SocketItem):
                return it
        return None

    def mousePressEvent(self, event):
        sock = self._socket_at(event.scenePos()) if event.button() == Qt.LeftButton else None
        if sock:
            if not sock.is_output:
                # grabbing a connected input detaches the wire so it can be re-routed
                existing = next((e for e in sock.node_item.edges if e.dst is sock), None)
                if existing:
                    src = existing.src
                    self._remove_edge(existing)
                    self.structure_changed.emit()
                    sock = src
            self.drag_edge = EdgeItem(sock, None)
            self.drag_edge.end_pos = event.scenePos()
            self.drag_edge.update_path()
            self.addItem(self.drag_edge)
            event.accept()
            return
        self._press_positions = {nid: it.node.pos for nid, it in self.items_by_id.items()}
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_edge:
            self.drag_edge.end_pos = event.scenePos()
            self.drag_edge.update_path()
            return
        super().mouseMoveEvent(event)
        if event.buttons() & Qt.LeftButton and isinstance(self.mouseGrabberItem(), NodeItem):
            from .editor_tools import update_insert_target
            update_insert_target(self, self.mouseGrabberItem())

    def mouseReleaseEvent(self, event):
        if self.drag_edge:
            start = self.drag_edge.src
            self.removeItem(self.drag_edge)
            self.drag_edge = None
            target = self._socket_at(event.scenePos())
            if target and target.is_output != start.is_output and target.node_item is not start.node_item:
                out, inp = (start, target) if start.is_output else (target, start)
                self.try_connect(out, inp)
            elif target is None and not any(isinstance(it, NodeItem) for it in self.items(event.scenePos())):
                self.wire_dropped.emit(start, event.scenePos())
            return
        grabbed = self.mouseGrabberItem()
        super().mouseReleaseEvent(event)
        if self.insert_edge is not None:
            from .editor_tools import finish_insert
            finish_insert(self, grabbed if isinstance(grabbed, NodeItem) else None)
        if any(self._press_positions.get(nid) != it.node.pos for nid, it in self.items_by_id.items()):
            self.edited.emit()
        self._press_positions = {}

    def try_connect(self, out: SocketItem, inp: SocketItem):
        src, dst = out.node_item.node.id, inp.node_item.node.id
        if any((c.src, c.src_port, c.dst, c.dst_port) == (src, out.name, dst, inp.name) for c in self.graph.connections):
            return          # already connected: dropping the same wire again must not draw a second one
        try:
            conn = self.graph.connect(out.node_item.node.id, out.name, inp.node_item.node.id, inp.name)
        except GraphError as exc:
            self.message.emit(str(exc))
            return
        # drop the item of any connection the graph replaced on that input
        for e in list(inp.node_item.edges):
            if e.dst is inp and e.conn not in self.graph.connections:
                self._remove_edge(e)
        self._add_edge_item(conn)
        self.structure_changed.emit()

    # background grid ---------------------------------------------------------------
    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, QColor(CANVAS["bg"]))
        for step, color in ((20, CANVAS["grid"]), (100, CANVAS["grid_major"])):
            painter.setPen(QPen(QColor(color), 1))
            left = math.floor(rect.left() / step) * step
            top = math.floor(rect.top() / step) * step
            x = left
            while x < rect.right():
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
                x += step
            y = top
            while y < rect.bottom():
                painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
                y += step


class NodeView(QGraphicsView):
    context_requested = Signal(object, object)   # global QPoint, scene QPointF
    quick_add_requested = Signal(object, object)  # global QPoint, scene QPointF

    def event(self, e):
        # Tab normally moves keyboard focus; on the canvas it opens the quick-add search instead
        if e.type() == QEvent.KeyPress and e.key() == Qt.Key_Tab:
            pos = self.mapFromGlobal(QCursor.pos())
            if not self.viewport().rect().contains(pos):
                pos = self.viewport().rect().center()
            self.quick_add_requested.emit(self.mapToGlobal(pos), self.mapToScene(pos))
            return True
        return super().event(e)

    def __init__(self, scene: NodeScene):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAcceptDrops(True)
        self._pan = None
        self._pan_start = None
        self.zoom = 1.0

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        # also allow steps back into range: "fit all" on a big flow can zoom out further than 0.2
        if (f > 1 and self.zoom * f < 3) or (f < 1 and self.zoom * f > 0.1):
            self.zoom *= f
            self.scale(f, f)

    def mousePressEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan = self._pan_start = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            moved = (e.position() - self._pan_start).manhattanLength()
            self._pan = None
            self.setCursor(Qt.ArrowCursor)
            if e.button() == Qt.RightButton and moved < 5:
                # a right click without dragging opens the context menu
                pos = e.position().toPoint()
                item = self.itemAt(pos)
                while item is not None and not isinstance(item, NodeItem):
                    item = item.parentItem()
                if isinstance(item, NodeItem) and not item.isSelected():
                    self.scene().clearSelection()
                    item.setSelected(True)
                self.context_requested.emit(self.mapToGlobal(pos), self.mapToScene(pos))
            return
        super().mouseReleaseEvent(e)

    def fit_all(self):
        r = self.scene().itemsBoundingRect()
        if r.isNull():
            return
        self.fitInView(r.adjusted(-60, -60, 60, 60), Qt.KeepAspectRatio)
        self.zoom = self.transform().m11()
        if self.zoom > 1.2:
            self.resetTransform()
            self.zoom = 1.0
            self.centerOn(r.center())

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()

    def dropEvent(self, e):
        type_name = bytes(e.mimeData().data(MIME)).decode()
        self.scene().add_node(type_name, self.mapToScene(e.position().toPoint()))
        e.acceptProposedAction()
