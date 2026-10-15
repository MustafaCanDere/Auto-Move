"""Runtime of a designed operator screen: ``ScreenFeed`` picks the bound values out of every frame on the flow
thread (cheap, never blocks the flow), ``ScreenView`` shows the pages scaled to the window and refreshes them at a
limited rate (default 15 Hz) from the GUI thread, ``ScreenHost`` carries out what buttons and inputs ask for
(shared variables, block settings, recipes, start / stop / counters) and asks for the edit-lock password."""
from __future__ import annotations

import collections
import threading
import time

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QGraphicsScene, QGraphicsView, QInputDialog, QLineEdit, QMessageBox, QTabBar,
                               QVBoxLayout, QWidget)

from ..core import screens as S
from ..core.graph import find_node, inner_error, node_at
from .screen_items import BackgroundItem, ScreenItem, values_match

IMAGE_MAX_W = 1600
IMAGE_INTERVAL = 1 / 20           # a bound picture is copied at most 20 times a second
ALARMS_MAX = 200
UNLOCK_S = 120.0                  # after the right password, settings can be changed for two minutes


def _plain(v):
    """A block output as something small and safe to hand to the GUI thread."""
    if v is None or isinstance(v, (bool, int, str)):
        return v[:500] if isinstance(v, str) else v
    if isinstance(v, float):
        return v if v == v else None
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return len(v)                              # detections, contours …: how many
    if isinstance(v, np.ndarray):
        return v.reshape(-1)[0].item() if v.size == 1 else f"<dizi {'x'.join(str(s) for s in v.shape)}>"
    return str(v)[:200]


class ScreenFeed:
    """Collects what the screen shows from the running flow. ``feed`` runs on the flow thread with every frame's
    results (only the bound outputs are touched); ``snapshot`` is read by the GUI thread."""

    def __init__(self, screen: dict | None):
        self._lock = threading.Lock()
        self.ports: dict[tuple, set] = {}          # (node, port) → {"image", "value"}: what widgets show of it
        self.trend_len: dict[tuple, int] = {}
        self.settings: set[tuple] = set()
        for wd, name, ref in S.refs(screen or {}):
            kind = S.binding_kind(ref)
            if kind == "port":
                key = (ref["node"], ref["port"])
                # an image widget and a value widget may show the same (ANY) output: each gets its own
                self.ports.setdefault(key, set()).add("image" if wd.get("type") == "image" else "value")
                if wd.get("type") == "trend":
                    self.trend_len[key] = max(self.trend_len.get(key, 2), int(wd["props"].get("points", 100)))
            elif kind == "setting":
                self.settings.add((ref["node"], ref["setting"]))
        self.reset()

    def reset(self):
        with self._lock:
            self.values: dict[tuple, tuple] = {}
            self.images: dict[tuple, tuple] = {}
            self.image_errors: dict[tuple, str] = {}
            self.trends = {k: collections.deque(maxlen=n) for k, n in self.trend_len.items()}
            self.setting_values: dict[tuple, object] = {}
            self.alarms: collections.deque = collections.deque(maxlen=ALARMS_MAX)
            self.recipe = None
            self._img_t: dict[tuple, float] = {}
            self._seq = 0
            self._last_err: dict[str, str] = {}
            self.frames = 0

    def feed(self, results: dict, graph=None) -> None:
        now = time.monotonic()
        vals, imgs, img_errs, trend_add = {}, {}, {}, []
        for key, kinds in self.ports.items():
            r = results.get(key[0])
            if r is None or r.waiting:
                continue
            if r.error:
                vals[key] = (None, str(r.error))
                if "image" in kinds:
                    img_errs[key] = str(r.error)
                continue
            v = r.outputs.get(key[1])
            if "image" in kinds:
                if isinstance(v, np.ndarray):
                    if now - self._img_t.get(key, -1.0) >= IMAGE_INTERVAL:
                        from .worker import displayable, shrink
                        img = displayable(v)
                        if img is not None:
                            self._img_t[key] = now
                            imgs[key] = np.ascontiguousarray(shrink(img, IMAGE_MAX_W)).copy()
                    img_errs[key] = ""
                else:
                    img_errs[key] = "" if v is None else "Görüntü değil"
                if "value" not in kinds:
                    vals[key] = (None, img_errs[key])
                    continue
            pv = _plain(v)
            vals[key] = (pv, "")
            if key in self.trend_len:
                n = S.to_number(pv)
                if n is not None:
                    trend_add.append((key, n))
        settings = {}
        if graph is not None:
            for key in self.settings:
                node = node_at(graph, key[0])
                if node is not None and key[1] in node.values:
                    settings[key] = node.values[key[1]]
        alarms = []
        if graph is not None:
            for nid, r in results.items():
                err = r.error
                if err and err != "Önceki blok hata verdi":
                    if self._last_err.get(nid) != err:
                        self._last_err[nid] = err
                        node = node_at(graph, nid)
                        if node is None or not inner_error(node, err):
                            alarms.append((time.time(), "hata", f"{node.title if node else nid}: {err}"))
                elif not r.waiting:
                    self._last_err.pop(nid, None)        # fixed: the same error later is a new alarm
        with self._lock:
            self.values.update(vals)
            self.image_errors.update(img_errs)
            for key, img in imgs.items():
                self._seq += 1
                self.images[key] = (self._seq, img)
            for key, n in trend_add:
                self.trends[key].append(n)
            self.setting_values.update(settings)
            self.alarms.extend(alarms)
            if graph is not None:
                self.recipe = getattr(graph, "active_recipe", None) or self.recipe
            self.frames += 1

    def new_run(self) -> None:
        """A new start of the flow: the same block error is an alarm again."""
        self._last_err = {}
        self.frames = 0

    def alarm(self, text: str, level: str = "hata") -> None:
        with self._lock:
            self.alarms.append((time.time(), level, str(text)[:300]))

    def snapshot(self) -> dict:
        with self._lock:
            return {"values": dict(self.values), "images": dict(self.images), "image_errors": dict(self.image_errors),
                    "trends": {k: list(d) for k, d in self.trends.items()},
                    "settings": dict(self.setting_values), "alarms": list(self.alarms), "recipe": self.recipe,
                    "frames": self.frames}


def ask_value(parent, title: str, label: str, current: str, items: list[str] | None = None) -> tuple[str, bool]:
    """Asks the operator for a value (a list to pick from when ``items``). Tests replace this function."""
    if items is not None:
        idx = items.index(current) if current in items else 0
        return QInputDialog.getItem(parent, title, label, items, idx, False)
    return QInputDialog.getText(parent, title, label, QLineEdit.Normal, current)


class ScreenHost:
    """What a screen's buttons and inputs act on. ``graph`` is the flow (a copy owned by the panel: written
    settings are kept there for its next start); ``worker`` gives the running FlowWorker (or None)."""

    def __init__(self, graph, lock=None, worker=lambda: None, tally=None, start=None, stop=None, reset=None,
                 recipe=None, after_write=None, message=None):
        self.graph, self.lock = graph, lock
        self.worker, self.tally = worker, tally
        self._start, self._stop, self._reset, self._recipe = start, stop, reset, recipe
        self.after_write = after_write
        self.message = message or (lambda _t: None)
        self.unlocked_until = 0.0
        self.recipe_name = ""
        self._settings: dict[tuple, object] = {}

    # --- reading -------------------------------------------------------------------------------------------------
    def running(self) -> bool:
        return self.worker() is not None

    def counters(self) -> dict:
        return self.tally.snapshot() if self.tally is not None else {}

    def setting_value(self, key: str, name: str):
        """The setting in the panel's copy of the flow (cached: blocks inside a stopped sub-flow are rebuilt to be
        read)."""
        ck = (key, name)
        if ck not in self._settings:
            node = find_node(self.graph, key) if self.graph is not None else None
            self._settings[ck] = node.values.get(name) if node is not None else None
        return self._settings[ck]

    def param(self, key: str, name: str):
        node = find_node(self.graph, key) if self.graph is not None else None
        if node is None or name not in node.values:
            return None
        return next((p for p in node.params if p.name == name), None)

    def recipes(self) -> list[str]:
        from ..core import recipes
        return recipes.names(self.graph) if self.graph is not None else []

    # --- lock ----------------------------------------------------------------------------------------------------
    def unlock(self, parent=None) -> bool:
        """No lock, unlocked a moment ago, or the right password now (the operator panel's prompt)."""
        if not self.lock or time.monotonic() < self.unlocked_until:
            return True
        from . import operator as op
        if op.ask_password(parent, self.lock, "Bu değeri değiştirmek için şifreyi girin:"):
            self.unlocked_until = time.monotonic() + UNLOCK_S
            return True
        return False

    # --- writing -------------------------------------------------------------------------------------------------
    def write_var(self, name: str, value) -> None:
        from ..nodes import connect
        connect.VARS[name] = value                  # one assignment: atomic for the flow thread reading it

    def write_setting(self, key: str, name: str, value):
        """Checks and sets a block setting in the flow copy and in the running flow. ValueError (Turkish)."""
        if self.graph is None:
            raise ValueError("Akış yok")
        try:
            v = S.write_setting(self.graph, key, name, value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        self._settings[(key, name)] = v
        w = self.worker()
        if w is not None:
            w.set_param(key, name, v)
        if self.after_write is not None:
            self.after_write(key, name, v)
        return v

    def apply_recipe(self, name: str) -> list[str]:
        if self._recipe is not None:
            errors = self._recipe(name)
        else:
            from ..core import recipes
            r = recipes.get(self.graph, name)
            if r is None:
                return [f"'{name}' adlı reçete yok"]
            errors = recipes.apply(self.graph, r)
            w = self.worker()
            if w is not None:
                w.request_recipe(r["name"])
            name = r["name"]
        self._settings.clear()
        self.recipe_name = name
        return list(errors or [])

    def start(self):
        if self._start is not None:
            self._start()
        else:
            self.message("Önizlemede akış buradan başlatılamaz")

    def stop(self):
        if self._stop is not None:
            self._stop()
        else:
            self.message("Önizlemede akış buradan durdurulamaz")

    def reset(self):
        if self._reset is not None:
            self._reset()
        elif self.tally is not None:
            self.tally.reset()


class _View(QGraphicsView):
    """Shows the page's design area scaled to fit (letterboxed), at any window size."""

    def __init__(self, scene, owner):
        super().__init__(scene)
        self.owner = owner
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#000000"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fit()

    def showEvent(self, e):
        super().showEvent(e)
        self.fit()


class ScreenView(QWidget):
    """A designed screen, live. ``host`` (ScreenHost) carries out the actions; the view polls ``feed`` (attach it
    to a FlowWorker as ``worker.screen``) ``rate`` times a second."""

    def __init__(self, screen: dict, graph=None, host: ScreenHost | None = None, parent=None):
        super().__init__(parent)
        self.screen = screen or S.new_screen()
        self.graph = graph
        self.host = host or ScreenHost(graph)
        self.feed = ScreenFeed(self.screen)
        self.items: list[ScreenItem] = []
        self.page_index = 0
        self.blink_phase = 0
        self.last_message = ""
        self.size_ = tuple(self.screen.get("size", S.DEFAULT_SIZE))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setStyleSheet("QTabBar::tab { font-size: 18px; padding: 8px 22px; }")
        for page in self.screen["pages"]:
            self.tabs.addTab(page["name"])
        self.tabs.setVisible(bool(self.screen.get("show_tabs", True)) and len(self.screen["pages"]) > 1)
        self.tabs.currentChanged.connect(self.show_page)
        lay.addWidget(self.tabs)
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(QRectF(0, 0, *self.size_))
        self.view = _View(self.scene, self)
        lay.addWidget(self.view, 1)
        self.show_page(0)
        self.timer = QTimer(self, interval=max(33, int(1000 / int(self.screen.get("rate", 15)))), timeout=self.poll)
        self.timer.start()

    # --- pages ---------------------------------------------------------------------------------------------------
    def show_page(self, index: int):
        index = max(0, min(len(self.screen["pages"]) - 1, int(index)))
        self.page_index = index
        if self.tabs.currentIndex() != index:
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(index)
            self.tabs.blockSignals(False)
        self.scene.clear()
        self.items = []
        page = self.screen["pages"][index]
        self.scene.addItem(BackgroundItem(page, self.size_))
        for z, wd in enumerate(page["widgets"]):
            it = ScreenItem(wd, self, S.widget_problem(wd, self.graph))
            it.setZValue(z)
            self.scene.addItem(it)
            self.items.append(it)
        self.poll()

    def go_page(self, spec: str) -> bool:
        """'+' next, '-' previous, a number (1 = first) or a page name."""
        pages = self.screen["pages"]
        spec = str(spec or "").strip()
        if spec in ("+", ">"):
            idx = (self.page_index + 1) % len(pages)
        elif spec in ("-", "<"):
            idx = (self.page_index - 1) % len(pages)
        elif spec.isdigit() and 1 <= int(spec) <= len(pages):
            idx = int(spec) - 1
        else:
            idx = next((i for i, p in enumerate(pages) if p["name"].casefold() == spec.casefold()), None)
            if idx is None:
                self.say(f"Sayfa bulunamadı: {spec}")
                return False
        self.show_page(idx)
        return True

    def item(self, wid: str) -> ScreenItem | None:
        return next((it for it in self.items if it.wd.get("id") == wid), None)

    # --- refreshing ----------------------------------------------------------------------------------------------
    def value_of(self, ref, snap) -> tuple:
        kind = S.binding_kind(ref)
        if kind == "port":
            return snap["values"].get((ref["node"], ref["port"]), (None, ""))
        if kind == "setting":
            key = (ref["node"], ref["setting"])
            if self.host.running() and key in snap["settings"]:
                return snap["settings"][key], ""
            return self.host.setting_value(*key), ""
        if kind == "var":
            from ..nodes.connect import VARS
            return VARS.get(ref["var"]), ""
        if kind == "counter":
            return self.host.counters().get(ref["counter"]), ""
        return None, ""

    def poll(self):
        snap = self.feed.snapshot()
        phase = int(time.monotonic() * 2.5) % 2
        blink_changed = phase != self.blink_phase
        self.blink_phase = phase
        if snap["recipe"] and self.host.running():
            self.host.recipe_name = snap["recipe"]
        for it in self.items:
            if it.problem:
                continue
            try:
                self._update(it, snap)
            except Exception as exc:  # noqa: BLE001  (one odd value must not freeze the screen)
                it.set_value(None, f"{exc}")
            if blink_changed and it.wtype == "lamp" and it.props.get("blink", "yok") != "yok":
                it.update()
        return snap

    def _update(self, it: ScreenItem, snap):
        t, pr = it.wtype, it.props
        if t == "image":
            ref = pr.get("src")
            if ref:
                key = (ref["node"], ref["port"])
                entry = snap["images"].get(key)
                if entry is not None:
                    it.set_image(*entry)
                err = snap["image_errors"].get(key)
                it.set_value(None, snap["values"].get(key, (None, ""))[1] if err is None else err)
        elif t == "trend":
            ref = pr.get("src")
            if not ref:
                return
            n = int(pr.get("points", 100))
            if S.binding_kind(ref) == "port":
                key = (ref["node"], ref["port"])
                vals = snap["trends"].get(key, [])[-n:]
                v, err = snap["values"].get(key, (None, ""))
                if vals != it.trend or err != it.error:
                    it.trend = vals
                    it.error = err
                    it.update()
            else:
                v, err = self.value_of(ref, snap)
                num = S.to_number(v)
                if num is not None:
                    it.trend = (it.trend + [num])[-n:]
                    it.update()
        elif t == "alarms":
            it.set_value(tuple(reversed(snap["alarms"][-int(pr.get("rows", 6)):])))
        elif t == "counters":
            it.set_value(self.host.counters())
        elif t == "recipe":
            it.set_value(self.host.recipe_name)
        elif t in ("button", "input"):
            ref = pr.get("target")
            if ref:
                it.set_value(*self.value_of(ref, snap))
        elif pr.get("src"):
            it.set_value(*self.value_of(pr["src"], snap))

    def say(self, text: str):
        self.last_message = text
        self.host.message(text)

    # --- actions (runtime presses) -------------------------------------------------------------------------------
    @staticmethod
    def _text_value(text: str, param=None):
        """A button's on / off text as the value to write."""
        if param is not None:
            return text
        t = str(text).strip()
        if t.lower() in ("true", "doğru", "evet"):
            return True
        if t.lower() in ("false", "yanlış", "hayır"):
            return False
        return S.parse_input(t, None)

    def _write(self, it: ScreenItem, text) -> bool:
        ref = it.props.get("target")
        kind = S.binding_kind(ref)
        try:
            if kind == "var":
                self.host.write_var(ref["var"], self._text_value(text))
            elif kind == "setting":
                param = self.host.param(ref["node"], ref["setting"])
                if param is None:
                    raise ValueError(f"Ayar bulunamadı: {ref['node']} → {ref['setting']}")
                self.host.write_setting(ref["node"], ref["setting"], S.parse_input(text, param))
            else:
                self.say("Düğmenin yazacağı yer seçilmemiş")
                return False
        except ValueError as exc:
            self.say(f"Yazılamadı: {exc}")
            return False
        self.poll()
        return True

    def _needs_unlock(self, it: ScreenItem) -> bool:
        if not it.props.get("lock", True):
            return False
        if it.wtype == "button":
            act = it.props.get("action")
            return act == "reçete" or act == "ayar" and S.binding_kind(it.props.get("target")) == "setting"
        return it.wtype in ("input", "recipe") and (it.wtype == "recipe" or
                                                    S.binding_kind(it.props.get("target")) == "setting")

    def press(self, it: ScreenItem):
        pr = it.props
        if it.wtype == "button" and pr.get("action") in ("değişken", "ayar") and pr.get("mode", "anlık") == "anlık":
            if self._needs_unlock(it) and not self.host.unlock(self):
                it.pressed = False
                it.update()
                self.say("Şifre yanlış: değer değiştirilmedi")
                return
            it._momentary = self._write(it, pr.get("on_value", "1"))

    def release(self, it: ScreenItem, inside: bool = True):
        pr = it.props
        if it.wtype == "button":
            act = pr.get("action")
            if act in ("değişken", "ayar"):
                if pr.get("mode", "anlık") == "anlık":
                    if getattr(it, "_momentary", False):
                        it._momentary = False
                        self._write(it, pr.get("off_value", "0"))
                    return
                if not inside:
                    return
                if self._needs_unlock(it) and not self.host.unlock(self):
                    self.say("Şifre yanlış: değer değiştirilmedi")
                    return
                on = values_match(it.value, pr.get("on_value", "1"))
                self._write(it, pr.get("off_value", "0") if on else pr.get("on_value", "1"))
                return
            if not inside:
                return
            if act == "başlat":
                self.host.start()
            elif act == "durdur":
                self.host.stop()
            elif act == "sayaçları sıfırla":
                self.host.reset()
                self.poll()
            elif act == "reçete":
                self.choose_recipe(it, pr.get("recipe") or None)
            elif act == "sayfa":
                self.go_page(pr.get("page", "+"))
        elif it.wtype == "input" and inside:
            self.edit_input(it)
        elif it.wtype == "recipe" and inside:
            self.choose_recipe(it)

    def choose_recipe(self, it: ScreenItem, name: str | None = None) -> bool:
        names = self.host.recipes()
        if not names:
            self.say("Bu akışta reçete yok")
            return False
        if name is None:
            text, ok = ask_value(self, "Reçete", "Ürün tipini seçin:", self.host.recipe_name, names)
            if not ok:
                return False
            name = text
        if self._needs_unlock(it) and not self.host.unlock(self):
            self.say("Şifre yanlış: reçete değiştirilmedi")
            return False
        errors = self.host.apply_recipe(name)
        for e in errors:
            self.feed.alarm(f"Reçete '{name}': {e}", "uyarı")
        if errors and errors[0].endswith("adlı reçete yok"):
            self.say(errors[0])
            return False
        self.say(f"Reçete: {name}")
        self.poll()
        return True

    def edit_input(self, it: ScreenItem, text: str | None = None) -> bool:
        """Asks for (or takes) a new value for the input's setting / variable; checks it before writing."""
        pr = it.props
        ref = pr.get("target")
        kind = S.binding_kind(ref)
        if kind not in ("var", "setting"):
            self.say("Girişin ayarı seçilmemiş")
            return False
        param = self.host.param(ref["node"], ref["setting"]) if kind == "setting" else None
        if kind == "setting" and param is None:
            self.say(f"Ayar bulunamadı: {ref['node']} → {ref['setting']}")
            return False
        if self._needs_unlock(it) and not self.host.unlock(self):
            self.say("Şifre yanlış: değer değiştirilmedi")
            return False
        if text is None:
            cur = it.value
            items = None
            if param is not None and param.kind == "choice":
                items = list(param.choices)
            elif param is not None and param.kind == "bool":
                items = ["Evet", "Hayır"]
                cur = "Evet" if cur else "Hayır"
            lo, hi = float(pr.get("min", 0)), float(pr.get("max", 0))
            rng = f" ({lo:g} … {hi:g})" if lo < hi else (
                f" ({param.min:g} … {param.max:g})" if param is not None and param.min is not None
                and param.max is not None else "")
            text, ok = ask_value(self, pr.get("title") or "Değer", f"Yeni değer{rng}:", "" if cur is None else
                                 str(cur), items)
            if not ok:
                return False
        try:
            value = S.parse_input(text, param, float(pr.get("min", 0)), float(pr.get("max", 0)))
            if kind == "var":
                self.host.write_var(ref["var"], value)
            else:
                self.host.write_setting(ref["node"], ref["setting"], value)
        except ValueError as exc:
            self.say(f"Geçersiz değer: {exc}")
            if self.isVisible():
                QMessageBox.warning(self, "Geçersiz değer", str(exc))
            return False
        self.poll()
        return True

    def stop_timer(self):
        self.timer.stop()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_PageDown:
            self.go_page("+")
        elif e.key() == Qt.Key_PageUp:
            self.go_page("-")
        else:
            super().keyPressEvent(e)
