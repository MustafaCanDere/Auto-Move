"""Background thread that runs the flow while the editor stays responsive."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from ..core import FrameLimiter, Graph, PortType, StopFlow
from ..core.graph import flatten, inner_error, node_at, sync_running_parents  # noqa: F401  (kept importable here)
from ..evolve import error_log

THUMB_W = 240
WATCH_MAX_W = 1280
EMIT_INTERVAL = 1 / 20   # UI refresh cap (results still computed at full speed)


def displayable(img) -> np.ndarray | None:
    """Any array a block may put on an image port (bool mask, float, int64, 1/2/4 channels, NaN) → uint8
    gray or BGR that Qt can show. None when there is nothing to show (empty image)."""
    if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        return None
    if img.ndim == 3:
        c = img.shape[2]
        img = img[:, :, 0] if c in (1, 2) else img[:, :, :3] if c >= 4 else img
    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        from ..api_server import stretch_u8
        img = stretch_u8(img)          # NaN holes (depth maps) are black and keep the contrast of the rest
    return np.ascontiguousarray(img)


def to_qimage(img: np.ndarray) -> QImage:
    img = displayable(img)
    if img is None:
        return QImage()
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, _ = img.shape
    return QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()


def shrink(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_AREA)


def summarize(node, outputs: dict) -> str:
    parts = []
    for name, t in node.outputs:
        if t in (PortType.IMAGE, PortType.CONTOURS) or name not in outputs:
            continue
        v = outputs[name]
        if isinstance(v, list):
            v = f"{len(v)}"
        elif isinstance(v, float):
            v = f"{v:.3g}"
        elif isinstance(v, bool):
            v = "✔" if v else "✘"
        elif v is None:
            continue
        parts.append(f"{name}={v}")
    return "  ".join(parts)


class FlowWorker(QThread):
    frame = Signal(dict, float)        # {node_id: (ms, error, text, QImage|None)}, fps
    watch = Signal(str, object, object)   # node_id, image of the watched block (shrunk), its real (w, h)
    failed = Signal(str)
    recipe = Signal(str, list)         # a recipe was applied while running ('Reçete Seç'), its problems

    def __init__(self, graph_data: dict):
        super().__init__()
        self.graph_data = graph_data
        self.watch_id: str | None = None
        self.thumbs = True
        self.tally = None                # operator panel: counts every frame (production.Tally)
        self.screen = None               # designed operator screen: picks its values (screen_runtime.ScreenFeed)
        self.limiter = FrameLimiter(30)
        self._stop = threading.Event()
        self._pending: list[tuple[str, str, object]] = []
        self._lock = threading.Lock()
        self._recipe_req = None          # a recipe to switch to, applied on the flow thread between frames
        self.api = None                  # remote control API (api_server.FlowState): gets every frame's results
        self.recipe_name = ""            # the recipe the flow starts with; follows 'Reçete Seç' / API switches

    def stop(self):
        self._stop.set()

    def set_param(self, node_id: str, name: str, value):
        with self._lock:
            self._pending.append((node_id, name, value))

    def request_recipe(self, name: str) -> None:
        """Switch the running flow to a recipe (like a 'Reçete Seç' block does): applied between frames, reported
        through the ``recipe`` signal and to the remote control API."""
        with self._lock:
            self._recipe_req = name

    def _apply_pending(self, graph: Graph) -> bool:
        with self._lock:
            pending, self._pending = self._pending, []
            recipe, self._recipe_req = self._recipe_req, None
        if recipe is not None:
            try:
                graph.request_recipe(recipe)
            except ValueError as exc:
                self.failed.emit(str(exc))
        restart = {}                     # restart each block once, however many of its settings changed
        for nid, name, value in pending:
            node = node_at(graph, nid)
            if node is None or name not in node.values:
                continue
            node.values[name] = value
            if "/" in nid:          # the running sub-flows around it keep the value too (for their own restart)
                sync_running_parents(graph, nid, name, value)
            if not node.enabled:
                continue
            # settings that need re-opening the camera/model, or a retry after a failed start
            # (not every setting of a source: moving a slider must not restart a batch or empty a mail queue)
            if node.param(name).kind in ("path", "folder") or name in node.setup_params or node.setup_error:
                restart[nid] = node
        for node in restart.values():
            if not node.setup_error:
                node.teardown()
            if not node.safe_setup():
                self.failed.emit(f"{node.title}: {node.setup_error}")
        return bool(pending) or recipe is not None

    def run(self):
        try:
            graph = Graph.from_dict(self.graph_data)
            for nid, err in graph.setup().items():
                self.failed.emit(f"{graph.nodes[nid].title}: {err}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Başlatılamadı: {exc}")
            return
        graph.active_recipe = self.recipe_name or ""     # 'Parçayı Arşivle' records it from the first frame
        attached = self.api
        if attached is not None:
            attached.attach(self.recipe_name)
        dirty, last_emit, frames, t0 = True, 0.0, 0, time.perf_counter()
        unsent: dict = {}                # results not shown yet: a triggered frame must not be lost between emits
        fps = 0.0
        try:
            while not self._stop.is_set():
                try:
                    if self._apply_pending(graph):
                        dirty = True
                    has_source = graph.has_source   # may change, e.g. toggling "animate"
                    if not has_source and not dirty and not graph.wants_frames():
                        time.sleep(0.03)   # static flow: recompute only when a setting changes
                        continue
                    if has_source:
                        self.limiter.wait()
                    elif not dirty:
                        time.sleep(0.03)   # only waiting for an AI answer etc.: check a few times a second
                    self._emit_recipes(graph)
                    results = graph.run_once()
                    self._emit_recipes(graph)
                except StopFlow:
                    break
                except Exception as exc:  # noqa: BLE001  (e.g. re-setup of a node failed)
                    self.failed.emit(str(exc))
                    time.sleep(0.2)
                    continue
                dirty = False
                if not graph.idle(results):
                    frames += 1
                flat = flatten(graph, results)
                api = self.api           # may be switched on while the flow runs
                if api is not None and not graph.idle(results):
                    try:      # the API snapshot must never stop the flow
                        api.publish(graph, flat, fps if has_source else None)
                    except Exception as exc:  # noqa: BLE001
                        self.failed.emit(f"API: {exc}")
                if self.tally is not None:
                    try:      # the operator panel's counting must never stop the flow
                        self.tally.feed(flat, graph)
                    except Exception as exc:  # noqa: BLE001
                        self.failed.emit(f"Sayım hatası: {exc}")
                screen = self.screen     # may be attached while the flow runs (designer preview)
                if screen is not None:
                    try:      # a designed screen must never stop the flow
                        screen.feed(flat, graph)
                    except Exception as exc:  # noqa: BLE001
                        self.failed.emit(f"Ekran hatası: {exc}")
                for nid, r in flat.items():
                    if not r.waiting or nid not in unsent or unsent[nid].waiting:
                        unsent[nid] = r
                now = time.perf_counter()
                if now - t0 >= 0.5:
                    fps, frames, t0 = frames / (now - t0), 0, now
                if now - last_emit >= EMIT_INTERVAL or not has_source:
                    last_emit = now
                    self._emit(graph, unsent, fps if has_source else 0.0)
                    unsent = {}
        finally:
            graph.teardown()
            if self.api is not None:
                self.api.detach()

    def _emit_recipes(self, graph: Graph):
        while graph.recipe_events:
            name, errors = graph.recipe_events.pop(0)
            self.recipe_name = name
            if self.api is not None:
                self.api.set_recipe(name)
            self.recipe.emit(name, list(errors))

    def _emit(self, graph: Graph, results, fps):
        payload = {}
        for nid, r in results.items():
            node = node_at(graph, nid)
            if node is None:
                continue
            if r.waiting:                        # nothing new: keep showing the last picture and values
                if r.waiting != "önceki blok bekliyor":
                    payload[nid] = (r.ms, None, r.waiting, None)
                continue
            qimg, img = None, None
            try:     # a strange array on an image port must not kill the worker thread
                img = displayable(next((r.outputs.get(n) for n, t in node.outputs if t in (PortType.IMAGE, PortType.ANY)
                                        and isinstance(r.outputs.get(n), np.ndarray)), None))
                if img is not None and self.thumbs:
                    qimg = to_qimage(shrink(img, THUMB_W))
                text = summarize(node, r.outputs)
            except Exception as exc:  # noqa: BLE001
                img, text = None, f"önizleme hatası: {exc}"
            payload[nid] = (r.ms, r.error, text, qimg)
            if r.error and not inner_error(node, r.error):
                try:     # remembered for "Kendini Geliştir"; never allowed to stop the flow
                    error_log().record(node.type_name, r.error, getattr(r, "where", ""))
                except Exception:  # noqa: BLE001
                    pass
            if nid == self.watch_id and img is not None:
                self.watch.emit(nid, shrink(img, WATCH_MAX_W).copy(), (img.shape[1], img.shape[0]))
        self.frame.emit(payload, fps)
