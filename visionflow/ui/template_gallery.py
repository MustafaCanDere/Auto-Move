"""'Yeni akış' wizard: a gallery of ready flows (templates and examples) as cards with a picture of the flow,
category filter and search. 'Kullan' asks the template's few questions and opens the flow with the answers in its
blocks; 'Boş akış' starts empty. Opens from Dosya → Yeni, and on the first start when there are no recent files
(unless 'Program açılınca gösterme' is ticked)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from .. import templates as T
from ..core import registry
from ..core.graph import Graph
from .theme import CATEGORY_COLORS, DEFAULT_CATEGORY_COLOR

HIDE_KEY = "galeri/acilista_gosterme"      # QSettings: do not show the gallery when the program starts
THUMB = QSize(240, 140)


# --- thumbnail: the flow drawn small ------------------------------------------------------------------------------

def render_thumbnail(data: dict, size: QSize = THUMB) -> QPixmap:
    """A small picture of the flow: blocks as coloured boxes (category colour) with their titles and the links."""
    pm = QPixmap(size)
    pm.fill(QColor("#1f2125"))
    nodes = {}
    for n in data.get("nodes", []) if isinstance(data, dict) else []:
        if not isinstance(n, dict) or n.get("type") not in registry.NODE_REGISTRY:
            continue
        cls = registry.NODE_REGISTRY[n["type"]]
        if not (cls.inputs or cls.outputs):
            continue                                    # notes are not part of the picture
        try:
            x, y = float(n["pos"][0]), float(n["pos"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            x, y = 0.0, 0.0
        h = 44.0 + 14.0 * max(len(cls.inputs), len(cls.outputs))
        nodes[str(n.get("id"))] = (QRectF(x, y, 200.0, h), cls, n.get("disabled", False))
    if not nodes:
        p = QPainter(pm)
        p.setPen(QColor("#8b8e94"))
        p.drawText(pm.rect(), Qt.AlignCenter, "Boş akış")
        p.end()
        return pm
    box = QRectF()
    for r, _, _ in nodes.values():
        box = box.united(r)
    margin = 10.0
    s = min((size.width() - 2 * margin) / max(box.width(), 1.0), (size.height() - 2 * margin) / max(box.height(), 1.0),
            0.6)
    ox = margin + (size.width() - 2 * margin - box.width() * s) / 2 - box.left() * s
    oy = margin + (size.height() - 2 * margin - box.height() * s) / 2 - box.top() * s

    def tr(r: QRectF) -> QRectF:
        return QRectF(ox + r.left() * s, oy + r.top() * s, r.width() * s, r.height() * s)

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor("#9aa0a6"), 1.2))
    for c in data.get("connections", []) or []:
        if not isinstance(c, dict) or c.get("src") not in nodes or c.get("dst") not in nodes:
            continue
        a, b = tr(nodes[c["src"]][0]), tr(nodes[c["dst"]][0])
        p0, p1 = QPointF(a.right(), a.center().y()), QPointF(b.left(), b.center().y())
        dx = max(12.0, abs(p1.x() - p0.x()) / 2)
        path = QPainterPath(p0)
        path.cubicTo(p0 + QPointF(dx, 0), p1 - QPointF(dx, 0), p1)
        p.drawPath(path)
    font = QFont()
    font.setPixelSize(max(7, min(12, int(26 * s))))
    p.setFont(font)
    for r0, cls, disabled in nodes.values():
        r = tr(r0)
        col = QColor(CATEGORY_COLORS.get(cls.category, DEFAULT_CATEGORY_COLOR))
        p.setPen(QPen(QColor("#111214"), 1))
        p.setBrush(QBrush(QColor("#2b2d30")))
        p.drawRoundedRect(r, 3, 3)
        head = QRectF(r.left(), r.top(), r.width(), min(r.height(), max(6.0, 22 * s)))
        p.setBrush(QBrush(col.darker(160) if disabled else col))
        p.drawRoundedRect(head, 3, 3)
        text_box = head if head.height() >= 12 else r.adjusted(0, head.height(), 0, 0)
        if text_box.height() >= 7:
            p.setPen(QColor("#ffffff") if text_box is head else QColor("#c4c7cc"))
            p.drawText(text_box.adjusted(3, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft,
                       p.fontMetrics().elidedText(cls.title, Qt.ElideRight, int(text_box.width() - 5)))
    p.end()
    return pm


# --- the questions of a template ----------------------------------------------------------------------------------

class QuestionsDialog(QDialog):
    """The few questions of a template ('Kamera nereden?', 'PLC IP?', 'Tolerans?') with their default answers."""

    def __init__(self, template: dict, parent=None):
        super().__init__(parent)
        self.template = template
        self.setWindowTitle(f"{template['title']} — ayarlar")
        self.widgets: dict[str, tuple] = {}
        lay = QVBoxLayout(self)
        intro = QLabel("Birkaç soruyu cevaplayın; cevaplar bloklara yazılır. Hepsini sonra blok ayarlarından "
                       "değiştirebilirsiniz.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        lay.addLayout(form)
        for q in template["questions"]:
            w = self._widget(q)
            form.addRow(q["label"], w)
            if q.get("help"):
                hint = QLabel(str(q["help"]))
                hint.setObjectName("hint")
                hint.setWordWrap(True)
                form.addRow("", hint)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #e53935;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Akışı oluştur")
        bb.button(QDialogButtonBox.Cancel).setText("Vazgeç")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _widget(self, q: dict) -> QWidget:
        kind, default = q["kind"], q.get("default")
        if kind == "source":
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            combo = QComboBox()
            combo.addItems(T.SOURCES)
            addr = QLineEdit()
            h.addWidget(combo)
            h.addWidget(addr, 1)

            def changed(text, addr=addr):
                addr.setEnabled(text != T.SAMPLE)
                addr.setPlaceholderText(T.SOURCE_HINT.get(text, ""))
                if text != T.SAMPLE and not addr.text().strip():
                    addr.setText(T.SOURCE_DEFAULT.get(text, ""))
            combo.currentTextChanged.connect(changed)
            combo.setCurrentText(default if default in T.SOURCES else T.SAMPLE)
            changed(combo.currentText())
            self.widgets[q["id"]] = ("source", combo, addr)
            return box
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
        elif kind == "int":
            w = QSpinBox()
            w.setRange(int(q.get("min", -10 ** 9)), int(q.get("max", 10 ** 9)))
            w.setValue(int(default or 0))
        elif kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(float(q.get("min", -1e9)), float(q.get("max", 1e9)))
            w.setValue(float(default or 0))
        elif kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in q["choices"]])
            if default is not None:
                w.setCurrentText(str(default))
        else:
            w = QLineEdit("" if default is None else str(default))
        self.widgets[q["id"]] = (kind, w)
        return w

    def answers(self) -> dict:
        out = {}
        for qid, entry in self.widgets.items():
            kind, w = entry[0], entry[1]
            if kind == "source":
                out[qid] = {"type": w.currentText(), "address": entry[2].text()}
            elif kind == "bool":
                out[qid] = w.isChecked()
            elif kind in ("int", "float"):
                out[qid] = w.value()
            elif kind == "choice":
                out[qid] = w.currentText()
            else:
                out[qid] = w.text()
        return out

    def accept(self):
        try:
            self.graph = T.apply_answers(self.template, self.answers())
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()


# --- the gallery --------------------------------------------------------------------------------------------------

class TemplateGallery(QDialog):
    """Cards of the templates. After exec(): `graph` is the new flow (None when cancelled), `chosen` its entry."""

    def __init__(self, parent=None, settings=None, folder=None):
        super().__init__(parent)
        self.setWindowTitle("Yeni akış — şablon galerisi")
        self.resize(1000, 640)
        self.settings = settings
        self.graph: Graph | None = None
        self.chosen: dict | None = None
        self.items = T.list_templates(folder)
        self._thumbs: dict[str, QPixmap] = {}

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Ara: etiket, renk, PLC, sayma…")
        self.search.setClearButtonEnabled(True)
        self.category = QComboBox()
        self.category.addItem("Tüm kategoriler", "")
        for c in T.CATEGORIES:
            self.category.addItem(c, c)
        top.addWidget(QLabel("Ne yapmak istiyorsunuz?"))
        top.addWidget(self.search, 1)
        top.addWidget(self.category)
        lay.addLayout(top)

        body = QHBoxLayout()
        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode)
        self.list.setIconSize(THUMB)
        self.list.setGridSize(QSize(THUMB.width() + 20, THUMB.height() + 52))
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setWordWrap(True)
        self.list.setSpacing(4)
        body.addWidget(self.list, 3)

        side = QVBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 13pt; font-weight: bold;")
        self.title.setWordWrap(True)
        self.cats = QLabel()
        self.cats.setObjectName("hint")
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.asks = QLabel()
        self.asks.setObjectName("hint")
        self.asks.setWordWrap(True)
        self.big = QLabel()
        self.big.setAlignment(Qt.AlignCenter)
        self.use_btn = QPushButton("Kullan")
        self.use_btn.setDefault(True)
        side.addWidget(self.big)
        side.addWidget(self.title)
        side.addWidget(self.cats)
        side.addWidget(self.desc)
        side.addWidget(self.asks)
        side.addStretch(1)
        side.addWidget(self.use_btn)
        sidew = QWidget()
        sidew.setLayout(side)
        sidew.setMinimumWidth(310)
        body.addWidget(sidew, 1)
        lay.addLayout(body, 1)

        bottom = QHBoxLayout()
        self.hide_on_start = QCheckBox("Program açılınca gösterme")
        self.hide_on_start.setChecked(self._hidden())
        self.hide_on_start.toggled.connect(self._set_hidden)
        blank = QPushButton("Boş akış")
        close = QPushButton("Kapat")
        bottom.addWidget(self.hide_on_start)
        bottom.addStretch(1)
        bottom.addWidget(blank)
        bottom.addWidget(close)
        lay.addLayout(bottom)

        self.search.textChanged.connect(self.refresh)
        self.category.currentIndexChanged.connect(self.refresh)
        self.list.currentItemChanged.connect(self._show)
        self.list.itemDoubleClicked.connect(lambda _it: self.use())
        self.use_btn.clicked.connect(lambda: self.use())
        blank.clicked.connect(self.blank)
        close.clicked.connect(self.reject)
        self.refresh()

    # settings ----------------------------------------------------------------------------------------------------
    def _hidden(self) -> bool:
        return show_hidden(self.settings)

    def _set_hidden(self, on: bool):
        if self.settings is not None:
            self.settings.setValue(HIDE_KEY, bool(on))

    # listing -----------------------------------------------------------------------------------------------------
    def thumb(self, t: dict) -> QPixmap:
        if t["path"] not in self._thumbs:
            self._thumbs[t["path"]] = render_thumbnail(t["data"])
        return self._thumbs[t["path"]]

    def visible(self) -> list[dict]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]

    def refresh(self):
        cat, text = self.category.currentData() or "", self.search.text()
        current = self.selected()
        self.list.clear()
        for t in self.items:
            if not T.matches(t, cat, text):
                continue
            it = QListWidgetItem(QIcon(self.thumb(t)), ("★ " if t["new"] else "") + t["title"])
            it.setData(Qt.UserRole, t)
            it.setToolTip(t["description"])
            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.list.addItem(it)
            if current is not None and current["path"] == t["path"]:
                self.list.setCurrentItem(it)
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)
        self._show()

    def selected(self) -> dict | None:
        it = self.list.currentItem() if hasattr(self, "list") else None
        return it.data(Qt.UserRole) if it is not None else None

    def select(self, name: str) -> bool:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole)["name"] == name:
                self.list.setCurrentRow(i)
                return True
        return False

    def _show(self, *_):
        t = self.selected()
        self.use_btn.setEnabled(t is not None)
        if t is None:
            self.title.setText("Bulunamadı")
            self.cats.setText("")
            self.desc.setText("Aramaya uyan şablon yok. Başka bir kelime deneyin veya 'Boş akış' ile başlayın.")
            self.asks.setText("")
            self.big.clear()
            return
        self.big.setPixmap(render_thumbnail(t["data"], QSize(300, 190)))
        self.title.setText(t["title"])
        self.cats.setText(" · ".join(t["categories"]))
        self.desc.setText(t["description"] or "")
        qs = [q["label"] for q in t["questions"]]
        self.asks.setText("Sorulacaklar: " + ", ".join(qs) if qs else "Soru yok: olduğu gibi açılır.")

    # choosing ----------------------------------------------------------------------------------------------------
    def use(self, answers: dict | None = None) -> bool:
        """Make the flow from the selected template. Without `answers` the questions are asked in a dialog."""
        t = self.selected()
        if t is None:
            return False
        if answers is None and t["questions"]:
            dlg = QuestionsDialog(t, self)
            if dlg.exec() != int(QDialog.DialogCode.Accepted.value):
                return False
            graph = dlg.graph
        else:
            try:
                graph = T.apply_answers(t, answers)
            except ValueError as exc:
                self.asks.setText(f"Hata: {exc}")
                return False
        self.graph, self.chosen = graph, t
        self.accept()
        return True

    def blank(self):
        self.graph, self.chosen = Graph(), None
        self.accept()


def show_hidden(settings) -> bool:
    if settings is None:
        return False
    v = settings.value(HIDE_KEY, False)
    return v is True or str(v).lower() in ("true", "1")


def load_into(win, graph: Graph, title: str = "") -> None:
    """Put a new flow from the gallery on the editor's canvas: unsaved (no file yet), so closing asks to save."""
    from .main_window import _set_flow_dir
    win.stop()
    _set_flow_dir(None)
    win._set_root(graph)
    win._loaded(None)
    if graph.nodes:
        win.saved_snap = ""            # made from a template but not saved anywhere yet
        win._title()
        QTimer.singleShot(60, win.view.fit_all)
        win.statusBar().showMessage(f"Şablondan yeni akış: {title} ({len(graph.nodes)} blok). Kaydetmek için Ctrl+S",
                                    6000)
        for w in graph.warnings:
            win.logp.log(f"Uyarı: {w}")


def new_from_gallery(win, exec_dialog=None) -> bool:
    """Dosya → Yeni: the gallery; a template or an empty flow replaces the current one (after asking to save)."""
    if not win._confirm_discard():
        return False
    dlg = TemplateGallery(win, win.settings)
    if not (exec_dialog or (lambda d: d.exec()))(dlg) or dlg.graph is None:
        return False
    load_into(win, dlg.graph, dlg.chosen["title"] if dlg.chosen else "")
    return True


def should_show_on_start(win) -> bool:
    """The first start (no recent files, nothing on the canvas) and the user did not tick the gallery off."""
    recent = win.settings.value("recent", [], list) or []
    return not recent and not show_hidden(win.settings) and not win.root.nodes


def maybe_show_on_start(win) -> bool:
    """On the first start the gallery greets the user (see should_show_on_start)."""
    if not should_show_on_start(win):
        return False
    QTimer.singleShot(200, lambda: new_from_gallery(win))
    return True
