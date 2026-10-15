"""The widgets of a designed operator screen, drawn with QPainter as graphics items. The same item is used by the
designer (movable, resizable) and at runtime (shows live values, buttons and inputs react to the mouse / touch)."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem

from ..core import screens as S

HANDLE = 8.0            # size of the resize handles in the designer (scene units)
_PICTURES: dict[str, QImage] = {}     # static pictures by path (small cache: a screen shows the same file often)


def picture(path: str) -> QImage:
    """A picture file (relative paths found like block files: next to the flow, then the program folder)."""
    from ..core.node import resolve_path
    if not path:
        return QImage()
    real = resolve_path(path)
    img = _PICTURES.get(real)
    if img is None:
        img = QImage(real)
        if len(_PICTURES) > 64:
            _PICTURES.clear()
        _PICTURES[real] = img
    return img


def fit_rect(src_w: float, src_h: float, r: QRectF, fit: str) -> QRectF:
    """Where a src_w×src_h picture goes in r: 'sığdır' keeps it whole, 'doldur' covers r, 'uzat' stretches."""
    if fit == "uzat" or src_w <= 0 or src_h <= 0:
        return QRectF(r)
    s = (min if fit == "sığdır" else max)(r.width() / src_w, r.height() / src_h)
    w, h = src_w * s, src_h * s
    return QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)


def _font(px: float, bold=False) -> QFont:
    f = QFont()
    f.setPixelSize(max(6, int(px)))
    f.setBold(bold)
    return f


def _fit(text: str, r: QRectF, px: float, bold=False) -> QFont:
    """The font of ``px`` pixels, made smaller until one line of ``text`` fits the width of ``r``."""
    from PySide6.QtGui import QFontMetricsF
    px = max(6.0, min(px, r.height() * 0.9 if r.height() > 0 else px))
    f = _font(px, bold)
    w = QFontMetricsF(f).horizontalAdvance(text or " ")
    if w > r.width() * 0.94 and w > 0:
        f = _font(max(6.0, px * r.width() * 0.94 / w), bold)
    return f


def _qc(c: str, fallback="#000000") -> QColor:
    q = QColor(c or fallback)
    return q if q.isValid() else QColor(fallback)


class ScreenItem(QGraphicsItem):
    """One widget. ``wd`` is its dict in the screen (id, type, rect, props); ``ctx`` is the designer or the runtime
    view (gives the blink phase, alarms, counters, recipe and receives the presses)."""

    def __init__(self, wd: dict, ctx=None, problem: str | None = None, editing: bool = False):
        super().__init__()
        self.wd, self.ctx, self.problem, self.editing = wd, ctx, problem, editing
        x, y, w, h = wd["rect"]
        self.w, self.h = float(w), float(h)
        self.setPos(x, y)
        self.value = None              # what it shows now
        self.error = ""                # the bound block's error
        self.qimage: QImage | None = None
        self.img_seq = None
        self.trend: list[float] = []
        self.pressed = False
        self.wheel_zoom = 1.0
        self._resize = None            # designer: (handle, start rect, start mouse pos)
        if editing:
            self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable |
                          QGraphicsItem.ItemSendsGeometryChanges)
            self.setAcceptHoverEvents(True)
        if problem:
            self.setToolTip(problem)

    # --- geometry ------------------------------------------------------------------------------------------------
    @property
    def props(self) -> dict:
        return self.wd.get("props", {})

    @property
    def wtype(self) -> str:
        return self.wd.get("type", "")

    def boundingRect(self) -> QRectF:
        m = HANDLE if self.editing else 0.0
        return QRectF(-m, -m, self.w + 2 * m, self.h + 2 * m)

    def rect_list(self) -> list[float]:
        return [round(self.pos().x(), 2), round(self.pos().y(), 2), round(self.w, 2), round(self.h, 2)]

    def set_geometry(self, x, y, w, h):
        self.prepareGeometryChange()
        self.w, self.h = max(4.0, w), max(4.0, h)
        self.setPos(x, y)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.editing and self.ctx is not None \
                and getattr(self.ctx, "snap", False) and self._resize is None:
            g = max(1, int(getattr(self.ctx, "grid", 10)))
            return QPointF(round(value.x() / g) * g, round(value.y() / g) * g)
        return super().itemChange(change, value)

    # --- state ---------------------------------------------------------------------------------------------------
    def set_value(self, value, error: str = "") -> bool:
        """New value to show; True (and repainted) when it changed."""
        try:
            same = value == self.value and type(value) is type(self.value) and error == self.error
        except Exception:  # noqa: BLE001
            same = False
        if same:
            return False
        self.value, self.error = value, error
        self.update()
        return True

    def set_image(self, seq, img) -> bool:
        if seq == self.img_seq:
            return False
        from .worker import to_qimage
        self.img_seq = seq
        self.qimage = to_qimage(img) if img is not None else None
        self.update()
        return True

    def lamp_state(self) -> bool | None:
        b = S.to_bool(self.value) if not self.error else None
        if b is not None and self.props.get("invert"):
            b = not b
        return b

    def button_active(self) -> bool:
        """A toggle button is drawn 'on' when its target holds the on value (or while a momentary one is pressed)."""
        if self.pressed:
            return True
        if self.props.get("mode") != "aç/kapa" or self.value is None:
            return False
        return values_match(self.value, self.props.get("on_value", "1"))

    # --- drawing -------------------------------------------------------------------------------------------------
    def paint(self, p: QPainter, option, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(0, 0, self.w, self.h)
        try:
            if self.problem:
                self._placeholder(p, r, self.problem)
            else:
                getattr(self, f"_paint_{self.wtype}", lambda p_, r_: self._placeholder(p_, r_, "Çizilemiyor"))(p, r)
        except Exception as exc:  # noqa: BLE001  (a strange value must not stop the screen)
            self._placeholder(p, r, f"Çizim hatası: {exc}")
        if self.editing and self.isSelected():
            p.setPen(QPen(QColor("#29b6f6"), 1.5, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            p.setPen(QPen(QColor("#0277bd"), 1))
            p.setBrush(QColor("#e1f5fe"))
            for hr in self.handles().values():
                p.drawRect(hr)

    def _auto_px(self, frac=0.5, cap=None) -> float:
        px = self.props.get("font") or 0
        if px:
            return px
        px = self.h * frac
        return min(px, cap) if cap else px

    def _box(self, p, r, bg, radius=8.0):
        if bg:
            p.setPen(Qt.NoPen)
            p.setBrush(_qc(bg))
            p.drawRoundedRect(r, radius, radius)

    def _title(self, p, r, text, color="#b0bec5") -> QRectF:
        """Draws a small title on top; returns the rest of the rect."""
        if not text:
            return r
        th = max(10.0, min(r.height() * 0.22, 28.0))
        p.setPen(_qc(color))
        p.setFont(_font(th * 0.75))
        p.drawText(QRectF(r.x() + 6, r.y() + 2, r.width() - 12, th), Qt.AlignLeft | Qt.AlignVCenter, text)
        return QRectF(r.x(), r.y() + th + 2, r.width(), max(1.0, r.height() - th - 2))

    def _placeholder(self, p, r, text):
        p.setPen(QPen(QColor("#ffb300"), 2, Qt.DashLine))
        p.setBrush(QBrush(QColor(255, 179, 0, 40), Qt.BDiagPattern))
        p.drawRect(r.adjusted(1, 1, -1, -1))
        p.setPen(QColor("#ffcc80"))
        p.setFont(_font(max(9.0, min(16.0, r.height() / 5))))
        p.drawText(r.adjusted(6, 4, -6, -4), Qt.AlignCenter | Qt.TextWordWrap, f"⚠ {text}")

    def _err_text(self, p, r):
        p.setPen(QColor("#ff9800"))
        p.setFont(_font(max(9.0, min(r.height() * 0.3, 40.0)), True))
        p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, "HATA")
        self.setToolTip(self.error)

    def _paint_image(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"), 0)
        img = self.qimage
        if img is not None and not img.isNull():
            zoom = max(1.0, float(pr.get("zoom", 1.0)) * self.wheel_zoom)
            sw, sh = img.width() / zoom, img.height() / zoom
            src = QRectF((img.width() - sw) / 2, (img.height() - sh) / 2, sw, sh)
            p.save()
            p.setClipRect(r)
            p.drawImage(fit_rect(sw, sh, r, pr.get("fit", "sığdır")), img, src)
            p.restore()
        else:
            p.setPen(QColor("#888888"))
            p.setFont(_font(max(10.0, min(r.height() / 10, 24.0))))
            msg = "HATA: " + self.error if self.error else ("Görüntü: " + self._src_label() if self.editing
                                                             else "Görüntü bekleniyor")
            p.drawText(r.adjusted(6, 6, -6, -6), Qt.AlignCenter | Qt.TextWordWrap, msg)
        if pr.get("title"):
            p.setPen(QColor("#ffffff"))
            p.setFont(_font(max(10.0, min(r.height() / 14, 22.0)), True))
            p.drawText(r.adjusted(8, 4, -8, -4), Qt.AlignLeft | Qt.AlignTop, pr["title"])
        if pr.get("border"):
            p.setPen(QPen(_qc(pr["border"]), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r.adjusted(1, 1, -1, -1))

    def _src_label(self) -> str:
        ref = self.props.get("src") or self.props.get("target")
        graph = getattr(self.ctx, "graph", None)
        return S.binding_label(ref, graph) if ref else "(bağlanmadı)"

    def _paint_number(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"))
        rest = self._title(p, r, pr.get("title"))
        if self.error:
            return self._err_text(p, rest)
        color = S.pick_color(self.value, pr.get("thresholds"), pr.get("color", "#e8e8e8"))
        text = S.format_number(self.value, int(pr.get("decimals", 1)), pr.get("unit", ""), pr.get("thousands"))
        p.setPen(_qc(color))
        p.setFont(_fit(text, rest, pr.get("font") or rest.height() * 0.7, True))
        p.drawText(rest, Qt.AlignCenter, text)

    def _aligned(self):
        return {"sol": Qt.AlignLeft, "orta": Qt.AlignHCenter, "sağ": Qt.AlignRight}.get(self.props.get("align"),
                                                                                      Qt.AlignLeft)

    def _paint_value(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"), 4)
        text = "HATA: " + self.error if self.error else S.format_text(pr.get("text", "{}"), self.value)
        p.setPen(_qc(pr.get("color"), "#e8e8e8"))
        p.setFont(_font(self._auto_px(0.5, 200), pr.get("bold")))
        p.drawText(r.adjusted(6, 0, -6, 0), self._aligned() | Qt.AlignVCenter | Qt.TextWordWrap, text)

    def _paint_text(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"), 4)
        p.setPen(_qc(pr.get("color"), "#e8e8e8"))
        lines = max(1, str(pr.get("text", "")).count("\n") + 1)
        p.setFont(_font(self._auto_px(0.5 / lines, 200), pr.get("bold")))
        p.drawText(r.adjusted(6, 0, -6, 0), self._aligned() | Qt.AlignVCenter | Qt.TextWordWrap, pr.get("text", ""))

    def blink_off(self, state) -> bool:
        mode = self.props.get("blink", "yok")
        phase = getattr(self.ctx, "blink_phase", 0)
        return bool(phase) and (mode == "doğruyken" and state is True or mode == "yanlışken" and state is False)

    def _paint_lamp(self, p, r):
        pr = self.props
        rest = self._title(p, r, pr.get("title"), "#cfd8dc")
        state = self.lamp_state()
        if self.error:
            color, text = "#ef6c00", "HATA"
            self.setToolTip(self.error)
        elif state is None:
            color, text = pr.get("none_color", "#455a64"), ""
        else:
            color = pr.get("on_color") if state else pr.get("off_color")
            text = pr.get("on_text") if state else pr.get("off_text")
        c = _qc(color)
        if self.blink_off(state):
            c = c.darker(260)
        p.setPen(QPen(c.darker(150), 2))
        p.setBrush(c)
        if pr.get("shape") == "dikdörtgen":
            p.drawRoundedRect(rest.adjusted(2, 2, -2, -2), 10, 10)
            box = rest
        else:
            d = min(rest.width(), rest.height()) - 4
            box = QRectF(rest.center().x() - d / 2, rest.center().y() - d / 2, d, d)
            p.drawEllipse(box)
        if text:
            p.setPen(QColor("#ffffff"))
            inner = box.adjusted(box.width() * 0.12, 0, -box.width() * 0.12, 0)
            p.setFont(_fit(text, inner, box.height() * 0.3, True))
            p.drawText(box, Qt.AlignCenter, text)

    def _fraction(self) -> float | None:
        v = S.to_number(self.value)
        lo, hi = float(self.props.get("min", 0)), float(self.props.get("max", 100))
        if v is None or hi == lo:
            return None
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    def _paint_bar(self, p, r):
        pr = self.props
        rest = self._title(p, r, pr.get("title"))
        if self.error:
            return self._err_text(p, rest)
        frac = self._fraction()
        color = _qc(S.pick_color(self.value, pr.get("thresholds"), pr.get("color", "#42a5f5")))
        text = S.format_number(self.value, int(pr.get("decimals", 1)), pr.get("unit", ""))
        style = pr.get("style", "yatay")
        if style == "gösterge":
            d = min(rest.width(), rest.height() * 1.6) - 8
            box = QRectF(rest.center().x() - d / 2, rest.y() + 4, d, d)
            p.setPen(QPen(QColor("#37474f"), max(4.0, d * 0.09), Qt.SolidLine, Qt.FlatCap))
            p.drawArc(box, 180 * 16, -180 * 16)
            if frac is not None:
                p.setPen(QPen(color, max(4.0, d * 0.09), Qt.SolidLine, Qt.FlatCap))
                p.drawArc(box, 180 * 16, int(-180 * 16 * frac))
                a = math.pi * (1 - frac)
                c = box.center()
                tip = QPointF(c.x() + math.cos(a) * d * 0.42, c.y() - math.sin(a) * d * 0.42)
                p.setPen(QPen(QColor("#eceff1"), max(2.0, d * 0.02)))
                p.drawLine(c, tip)
            if pr.get("show_value", True):
                p.setPen(QColor("#eceff1"))
                p.setFont(_font(max(9.0, d * 0.12), True))
                p.drawText(QRectF(box.x(), box.center().y() + 2, box.width(), d * 0.2), Qt.AlignCenter, text)
            return
        track = rest.adjusted(2, 2, -2, -2)
        if pr.get("show_value", True) and style == "yatay":
            tw = min(track.width() * 0.35, 160.0)
            p.setPen(QColor("#eceff1"))
            vr = QRectF(track.right() - tw, track.y(), tw, track.height())
            p.setFont(_fit(text, vr, min(track.height() * 0.6, 40.0), True))
            p.drawText(vr, Qt.AlignRight | Qt.AlignVCenter, text)
            track.setWidth(track.width() - tw - 6)
        elif pr.get("show_value", True):
            th = min(track.height() * 0.15, 40.0)
            p.setPen(QColor("#eceff1"))
            p.setFont(_font(max(9.0, th * 0.8), True))
            p.drawText(QRectF(track.x(), track.bottom() - th, track.width(), th), Qt.AlignCenter, text)
            track.setHeight(track.height() - th - 4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#263238"))
        p.drawRoundedRect(track, 4, 4)
        if frac:
            fill = QRectF(track)
            if style == "dikey":
                fill.setTop(track.bottom() - track.height() * frac)
            else:
                fill.setWidth(track.width() * frac)
            p.setBrush(color)
            p.drawRoundedRect(fill, 4, 4)

    def _paint_trend(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"), 4)
        rest = self._title(p, r, pr.get("title"))
        plot = rest.adjusted(48, 4, -8, -6)
        if plot.width() < 10 or plot.height() < 10:
            return
        vals = self.trend[-int(pr.get("points", 100)):]
        lo, hi = float(pr.get("min", 0)), float(pr.get("max", 0))
        if lo >= hi:
            lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
            if hi - lo < 1e-9:
                lo, hi = lo - 1, hi + 1
        p.setFont(_font(max(8.0, min(14.0, plot.height() / 8))))
        if pr.get("grid", True):
            p.setPen(QPen(QColor("#37474f"), 1))
            for i in range(5):
                y = plot.top() + plot.height() * i / 4
                p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                p.setPen(QColor("#90a4ae"))
                p.drawText(QRectF(r.x(), y - 8, 44, 16), Qt.AlignRight | Qt.AlignVCenter, f"{hi - (hi - lo) * i / 4:.4g}")
                p.setPen(QPen(QColor("#37474f"), 1))

        def ymap(v):
            return plot.bottom() - (min(hi, max(lo, v)) - lo) / (hi - lo) * plot.height()
        for limit, c in S.parse_thresholds(pr.get("thresholds")):
            if lo <= limit <= hi:
                p.setPen(QPen(_qc(c), 1.5, Qt.DashLine))
                p.drawLine(QPointF(plot.left(), ymap(limit)), QPointF(plot.right(), ymap(limit)))
        if self.error:
            p.setPen(QColor("#ff9800"))
            p.drawText(plot, Qt.AlignCenter, "HATA")
        if len(vals) >= 2:
            n = int(pr.get("points", 100))
            step = plot.width() / max(1, n - 1)
            x0 = plot.right() - step * (len(vals) - 1)
            poly = QPolygonF([QPointF(x0 + i * step, ymap(v)) for i, v in enumerate(vals)])
            p.setPen(QPen(_qc(pr.get("color"), "#66bb6a"), 2))
            p.drawPolyline(poly)

    def _paint_button(self, p, r):
        pr = self.props
        active = self.button_active()
        c = _qc(pr.get("active_color") if active else pr.get("color"), "#33363c")
        p.setPen(QPen(c.lighter(140), 2))
        p.setBrush(c.darker(120) if self.pressed else c)
        p.drawRoundedRect(r.adjusted(2, 2, -2, -2), 10, 10)
        p.setPen(_qc(pr.get("text_color"), "#ffffff"))
        text = pr.get("text", "")
        tr = r.adjusted(8, 0, -8, 0)
        p.setFont(_fit(text, tr, pr.get("font") or r.height() * 0.4, True) if "\n" not in text else
                  _font(pr.get("font") or r.height() * 0.25, True))
        p.drawText(tr, Qt.AlignCenter | Qt.TextWordWrap, text)

    def _paint_input(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"))
        p.setPen(QPen(QColor("#546e7a"), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 8, 8)
        rest = self._title(p, r, pr.get("title") or (self._src_label() if self.editing else ""))
        if self.error:
            return self._err_text(p, rest)
        v = self.value
        if isinstance(v, bool):
            text = "Evet" if v else "Hayır"
        elif isinstance(v, (int, float)):
            text = S.format_number(v, int(pr.get("decimals", 2)) if isinstance(v, float) else 0, pr.get("unit", ""))
        else:
            text = "–" if v is None else str(v) + (f" {pr['unit']}" if pr.get("unit") else "")
        p.setPen(_qc(pr.get("color"), "#e8e8e8"))
        tr = rest.adjusted(10, 0, -34, 0)
        p.setFont(_fit(text, tr, pr.get("font") or rest.height() * 0.6, True))
        p.drawText(tr, Qt.AlignLeft | Qt.AlignVCenter, text)
        p.setPen(QColor("#90a4ae"))
        p.setFont(_font(min(rest.height() * 0.5, 28.0)))
        p.drawText(rest.adjusted(0, 0, -8, 0), Qt.AlignRight | Qt.AlignVCenter, "✎")

    def _paint_recipe(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"))
        rest = self._title(p, r, pr.get("title"))
        name = self.value or "(reçete yok)"
        p.setPen(_qc(pr.get("color"), "#e8e8e8"))
        tr = rest.adjusted(10, 0, -34, 0)
        p.setFont(_fit(str(name), tr, pr.get("font") or rest.height() * 0.6, True))
        p.drawText(tr, Qt.AlignLeft | Qt.AlignVCenter, str(name))
        p.setPen(QColor("#90a4ae"))
        p.drawText(rest.adjusted(0, 0, -10, 0), Qt.AlignRight | Qt.AlignVCenter, "▾")

    def _paint_alarms(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"), 4)
        rest = self._title(p, r, pr.get("title"), "#ffcc80")
        rows = int(pr.get("rows", 6))
        alarms = list(self.value or [])[:rows]
        rh = rest.height() / rows
        p.setFont(_font(max(8.0, rh * 0.55)))
        if not alarms:
            p.setPen(QColor("#78909c"))
            p.drawText(rest, Qt.AlignCenter, "Alarm yok")
            return
        for i, (ts, level, text) in enumerate(alarms):
            p.setPen(QColor("#ef5350") if level == "hata" else _qc(pr.get("color"), "#ffb74d"))
            line = (time.strftime("%H:%M:%S  ", time.localtime(ts)) if pr.get("show_time", True) else "") + text
            p.drawText(QRectF(rest.x() + 6, rest.y() + i * rh, rest.width() - 12, rh),
                       Qt.AlignLeft | Qt.AlignVCenter, line)

    def _paint_counters(self, p, r):
        pr = self.props
        self._box(p, r, pr.get("bg"), 4)
        rest = self._title(p, r, pr.get("title"))
        snap = self.value or {}
        rows = [(k, S.COUNTER_NAMES[k]) for k in S.COUNTER_KEYS if pr.get(f"show_{k}", True)]
        if not rows:
            return
        rh = rest.height() / len(rows)
        colors = {"good": "#66bb6a", "bad": "#ef5350", "total": pr.get("color", "#e8e8e8"), "yield": "#64b5f6"}
        for i, (k, name) in enumerate(rows):
            box = QRectF(rest.x() + 10, rest.y() + i * rh, rest.width() - 20, rh)
            p.setFont(_font(max(8.0, rh * 0.45)))
            p.setPen(_qc(pr.get("color"), "#e8e8e8"))
            p.drawText(box, Qt.AlignLeft | Qt.AlignVCenter, name)
            v = snap.get(k)
            text = ("–" if v is None else f"%{v:g}") if k == "yield" else str(v if v is not None else 0)
            p.setFont(_font(max(8.0, rh * 0.6), True))
            p.setPen(_qc(colors[k]))
            p.drawText(box, Qt.AlignRight | Qt.AlignVCenter, text)

    def _paint_picture(self, p, r):
        img = picture(self.props.get("file", ""))
        if img.isNull():
            p.setPen(QPen(QColor("#607d8b"), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            p.setPen(QColor("#90a4ae"))
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap,
                       "Resim yok" if not self.props.get("file") else f"Resim açılamadı: {self.props['file']}")
            return
        p.save()
        p.setClipRect(r)
        p.drawImage(fit_rect(img.width(), img.height(), r, self.props.get("fit", "sığdır")), img)
        p.restore()

    def _paint_rect(self, p, r):
        pr = self.props
        w = int(pr.get("width", 2))
        p.setPen(QPen(_qc(pr["border"]), w) if pr.get("border") and w else Qt.NoPen)
        p.setBrush(_qc(pr["fill"]) if pr.get("fill") else Qt.NoBrush)
        rad = float(pr.get("radius", 0))
        p.drawRoundedRect(r.adjusted(w / 2, w / 2, -w / 2, -w / 2), rad, rad)

    # --- designer: resize handles --------------------------------------------------------------------------------
    def handles(self) -> dict[str, QRectF]:
        w, h, s = self.w, self.h, HANDLE
        pts = {"tl": (0, 0), "t": (w / 2, 0), "tr": (w, 0), "r": (w, h / 2), "br": (w, h), "b": (w / 2, h),
               "bl": (0, h), "l": (0, h / 2)}
        return {k: QRectF(x - s / 2, y - s / 2, s, s) for k, (x, y) in pts.items()}

    def handle_at(self, pos) -> str | None:
        if not (self.editing and self.isSelected()):
            return None
        for k, hr in self.handles().items():
            if hr.adjusted(-2, -2, 2, 2).contains(pos):
                return k
        return None

    def hoverMoveEvent(self, e):
        h = self.handle_at(e.pos())
        cursors = {"tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor, "tr": Qt.SizeBDiagCursor,
                   "bl": Qt.SizeBDiagCursor, "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor, "l": Qt.SizeHorCursor,
                   "r": Qt.SizeHorCursor}
        self.setCursor(cursors[h] if h else Qt.SizeAllCursor)
        super().hoverMoveEvent(e)

    def resize_by(self, handle: str, dx: float, dy: float, start: tuple, grid: int = 0):
        """Resize from ``start`` (x, y, w, h) by dragging ``handle``; edges snap to the grid when given."""
        x, y, w, h = start
        x2, y2 = x + w, y + h

        def snap(v):
            return round(v / grid) * grid if grid else v
        if "l" in handle:
            x = min(snap(x + dx), x2 - 4)
        if "r" in handle:
            x2 = max(snap(x2 + dx), x + 4)
        if handle.startswith("t"):
            y = min(snap(y + dy), y2 - 4)
        if handle.startswith("b"):
            y2 = max(snap(y2 + dy), y + 4)
        self.set_geometry(x, y, x2 - x, y2 - y)

    def mousePressEvent(self, e):
        if self.editing:
            h = self.handle_at(e.pos())
            if h and e.button() == Qt.LeftButton:
                self._resize = (h, (self.pos().x(), self.pos().y(), self.w, self.h), e.scenePos())
                e.accept()
                return
            return super().mousePressEvent(e)
        if self.ctx is not None and e.button() == Qt.LeftButton and hasattr(self.ctx, "press") \
                and self.wtype in ("button", "input", "recipe"):
            self.pressed = True
            self.update()
            self.ctx.press(self)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.editing and self._resize is not None:
            h, start, p0 = self._resize
            d = e.scenePos() - p0
            grid = int(getattr(self.ctx, "grid", 0)) if getattr(self.ctx, "snap", False) else 0
            self.resize_by(h, d.x(), d.y(), start, grid)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.editing:
            was = self._resize
            self._resize = None
            if was is None:
                super().mouseReleaseEvent(e)
            if self.ctx is not None and hasattr(self.ctx, "items_edited"):
                self.ctx.items_edited()
            return
        if self.pressed:
            self.pressed = False
            self.update()
            if self.ctx is not None and hasattr(self.ctx, "release"):
                inside = QRectF(0, 0, self.w, self.h).contains(e.pos())
                self.ctx.release(self, inside)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        if not self.editing and self.wtype == "image" and self.props.get("wheel_zoom", True):
            step = 1.25 if e.delta() > 0 else 0.8
            self.wheel_zoom = max(1.0, min(16.0, self.wheel_zoom * step))
            self.update()
            e.accept()
            return
        super().wheelEvent(e)

    def mouseDoubleClickEvent(self, e):
        if not self.editing and self.wtype == "image":
            self.wheel_zoom = 1.0                 # double click: back to the designed zoom
            self.update()
            e.accept()
            return
        super().mouseDoubleClickEvent(e)


def values_match(current, text) -> bool:
    """Does a variable / setting hold the value written by a button (text from its properties)?"""
    if current is None:
        return False
    a, b = S.to_number(current), S.to_number(text)
    if a is not None and b is not None and not isinstance(current, str):
        return a == b
    if isinstance(current, bool):
        return S.to_bool(text) is current
    return str(current) == str(text)


class BackgroundItem(QGraphicsItem):
    """A page's background colour and picture, under every widget."""

    def __init__(self, page: dict, size):
        super().__init__()
        self.page, self.size = page, size
        self.setZValue(-1e6)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.size[0], self.size[1])

    def paint(self, p, option, widget=None):
        r = self.boundingRect()
        p.fillRect(r, _qc(self.page.get("bg"), "#15161a"))
        img = picture(self.page.get("bg_image", ""))
        if not img.isNull():
            p.setClipRect(r)
            p.drawImage(fit_rect(img.width(), img.height(), r, "doldur"), img)

