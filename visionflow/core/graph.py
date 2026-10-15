"""Flow graph: nodes + connections, validation, execution and JSON I/O."""
from __future__ import annotations

import copy
import json
import math
import re
import time
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import registry
from .node import NoNewFrame, Node, PortType, StopFlow, clean_color
from .node import where as node_where

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Connection:
    src: str
    src_port: str
    dst: str
    dst_port: str

    def to_dict(self) -> dict:
        return {"src": self.src, "src_port": self.src_port, "dst": self.dst, "dst_port": self.dst_port}


class FrameLimiter:
    """Sleeps just enough to keep the loop at or below ``max_fps`` (0 = unlimited)."""

    def __init__(self, max_fps: float = 0):
        self.max_fps = max_fps
        self.next_t = 0.0

    def wait(self) -> None:
        if self.max_fps <= 0:
            return
        now = time.perf_counter()
        if self.next_t > now:
            time.sleep(self.next_t - now)
            now = self.next_t
        self.next_t = max(self.next_t + 1.0 / self.max_fps, now)


class GraphError(Exception):
    pass


_where = node_where      # kept under the old name


@dataclass
class NodeResult:
    outputs: dict[str, Any]
    ms: float
    error: str | None = None
    where: str = ""          # code location of the error (file:line), for the self-improvement report
    waiting: str = ""        # skipped this frame: a source before it had nothing new (NoNewFrame)


def node_at(graph: Graph, key: str):
    """The block named by "id" or, inside sub-flows, "subflow id/inner id/..." (None when gone)."""
    parts = key.split("/")
    node = graph.nodes.get(parts[0])
    for part in parts[1:]:
        inner = node.state.get("graph") if node is not None else None
        node = inner.nodes.get(part) if inner is not None else None
    return node


def sync_running_parents(graph: Graph, key: str, name: str, value) -> None:
    """Setting `name` of the running block "sub/.../inner" changes: the running sub-flows on the way keep the new
    value in their own inner flow (and exposed setting) too, so restarting one of them does not bring back the old
    value."""
    parts = key.split("/")
    for i in range(1, len(parts)):
        sub = node_at(graph, "/".join(parts[:i]))
        if sub is None or sub.type_name != "SubFlow":
            return
        rest = "/".join(parts[i:] + [name])
        if rest in sub.values:
            sub.set_exposed(rest, value)
        else:
            from ..nodes.subflow import _set_in
            _set_in(sub.inner, rest, value)


def _inner_of(node):
    """The blocks inside a sub-flow block: the running ones while it runs, else built from its saved inner flow."""
    if node is None or node.type_name != "SubFlow":
        return None
    g = node.state.get("graph") if isinstance(getattr(node, "state", None), dict) else None
    if g is None:
        try:
            g = node.inner_graph()
        except Exception:  # noqa: BLE001  (a broken inner flow simply has no blocks to offer)
            return None
    return g


def find_node(graph: Graph, key: str):
    """Like node_at, but also finds blocks inside sub-flows that are not running (for checking settings that name
    "subflow id/inner id" blocks, e.g. the operator panel's outputs)."""
    node = node_at(graph, key)
    if node is not None or "/" not in key:
        return node
    parts = key.split("/")
    node = graph.nodes.get(parts[0])
    for part in parts[1:]:
        inner = _inner_of(node)
        node = inner.nodes.get(part) if inner is not None else None
    return node


def walk_nodes(graph: Graph, prefix: str = "", label: str = "", depth: int = 0):
    """(key, label, block) of every block, those inside sub-flows too: keys "subflow id/inner id" (like node_at),
    labels "Alt akış › Blok". The sub-flows' own input/output blocks are left out."""
    for nid, node in graph.nodes.items():
        if prefix and node.type_name in ("SubflowInput", "SubflowOutput"):
            continue
        yield prefix + nid, label + node.title, node
        if node.type_name == "SubFlow" and depth < 12:
            inner = _inner_of(node)
            if inner is not None:
                yield from walk_nodes(inner, f"{prefix}{nid}/", f"{label}{node.title} › ", depth + 1)


def flatten(graph: Graph, results: dict, prefix: str = "") -> dict:
    """The results of a frame with those of the blocks inside sub-flows, under "subflow id/inner id" keys, so the
    editor can show the inside of an open sub-flow. Each sub-flow's inner results are taken once."""
    out = {}
    for nid, r in results.items():
        out[prefix + nid] = r
        node = graph.nodes.get(nid)
        inner_results = node.state.pop("results", None) if node is not None and node.type_name == "SubFlow" else None
        inner = node.state.get("graph") if inner_results is not None else None
        if inner is not None:
            out.update(flatten(inner, inner_results, f"{prefix}{nid}/"))
    return out


def inner_error(node, error: str) -> bool:
    """A sub-flow's error that only repeats the error of a block inside it (that block's error is logged itself)."""
    return node is not None and node.type_name == "SubFlow" and str(error).startswith("InnerError:")


FRAME_COLORS = ["#3d5a80", "#5a3d80", "#3d8064", "#80673d", "#803d4a", "#4a4f57"]


def clean_frame(fd) -> dict | None:
    """A comment frame from a file, checked: anything odd is dropped rather than breaking the editor."""
    try:
        x, y, w, h = (float(v) for v in fd["rect"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    color = fd.get("color") if isinstance(fd.get("color"), str) and re.fullmatch(r"#[0-9a-fA-F]{6}",
                                                                                  fd.get("color")) else FRAME_COLORS[0]
    return {"id": str(fd.get("id") or f"f{abs(hash((x, y, w, h))) % 10 ** 8}"), "title": str(fd.get("title", ""))[:200],
            "rect": [x, y, max(80.0, min(w, 20000.0)), max(60.0, min(h, 20000.0))], "color": color}


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.frames: list[dict] = []     # comment frames of the editor: {id, title, rect: [x, y, w, h], color}
        self.connections: list[Connection] = []
        self.warnings: list[str] = []
        self.operator: dict | None = None  # operator panel settings + edit lock (see core/operator.py)
        self.recipes: list[dict] = []    # product types: {name, values: {block id: {setting: value}}} (core/recipes.py)
        self.accel: dict = {}            # inference device of the flow: {"device", "fp16"} (visionflow/accel.py)
        self.parent: Graph | None = None  # while running inside a sub-flow: the flow around it
        self._pending_recipe: str | None = None
        self.recipe_events: list[tuple[str, list[str]]] = []   # (recipe applied while running, problems)
        self.active_recipe = ""          # the recipe applied last while running (read by 'Parçayı Arşivle')

    # editing -----------------------------------------------------------------
    def add(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def add_new(self, type_name: str, pos=(0.0, 0.0), **values) -> Node:
        node = registry.create(type_name)
        node.pos = tuple(pos)
        node.values.update(values)
        return self.add(node)

    def remove(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.connections = [c for c in self.connections if node_id not in (c.src, c.dst)]

    def connect(self, src: str, src_port: str, dst: str, dst_port: str) -> Connection:
        a, b = self.nodes[src], self.nodes[dst]
        if src == dst:
            raise GraphError("Bir blok kendisine bağlanamaz")
        if src_port not in dict(a.outputs) or dst_port not in dict(b.inputs):
            raise GraphError(f"Port yok: {src_port} → {dst_port}")
        st, dt = a.output_type(src_port), b.input_type(dst_port)
        if not PortType.compatible(st, dt):
            raise GraphError(f"Tip uyumsuz: {st} → {dt}")
        # an input accepts exactly one connection
        previous = list(self.connections)
        self.connections = [c for c in self.connections if not (c.dst == dst and c.dst_port == dst_port)]
        conn = Connection(src, src_port, dst, dst_port)
        self.connections.append(conn)
        try:
            self.order()
        except GraphError:
            self.connections = previous
            raise
        return conn

    def disconnect(self, conn: Connection) -> None:
        if conn in self.connections:
            self.connections.remove(conn)

    # execution ---------------------------------------------------------------
    def is_feedback(self, c: Connection) -> bool:
        """A link into a feedback input carries last frame's value and is left out of the ordering."""
        node = self.nodes.get(c.dst)
        return node is not None and c.dst_port in node.feedback_inputs

    def order(self) -> list[str]:
        """Topological order (Kahn). Raises GraphError on cycles."""
        indeg = {nid: 0 for nid in self.nodes}
        out: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for c in self.connections:
            if self.is_feedback(c):
                continue
            indeg[c.dst] += 1
            out[c.src].append(c.dst)
        ready = sorted(n for n, d in indeg.items() if d == 0)
        result = []
        while ready:
            n = ready.pop(0)
            result.append(n)
            for m in out[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        if len(result) != len(self.nodes):
            raise GraphError("Akışta döngü var")
        return result

    def auto_layout(self, dx: float = 260.0, gap: float = 40.0, origin=(0.0, 0.0),
                    heights: dict[str, float] | None = None) -> None:
        """Arrange blocks left-to-right in columns by depth (longest path from a source). Inside a column
        blocks keep their previous vertical order, which keeps the result close to the user's own layout.
        Note blocks (no ports) are left where they are."""
        movable = [nid for nid, n in self.nodes.items() if n.inputs or n.outputs]
        depth = {nid: 0 for nid in movable}
        for nid in self.order():
            for c in self.connections:
                if c.src == nid and c.dst in depth and not self.is_feedback(c):
                    depth[c.dst] = max(depth[c.dst], depth.get(nid, 0) + 1)
        columns: dict[int, list[str]] = {}
        for nid in movable:
            columns.setdefault(depth[nid], []).append(nid)
        h = lambda nid: (heights or {}).get(nid, 110.0)  # noqa: E731
        col_h = {d: sum(h(i) + gap for i in ids) - gap for d, ids in columns.items()}
        tallest = max(col_h.values(), default=0.0)
        for d, ids in columns.items():
            ids.sort(key=lambda i: (self.nodes[i].pos[1], self.nodes[i].pos[0]))
            y = origin[1] + (tallest - col_h[d]) / 2       # center shorter columns
            for nid in ids:
                self.nodes[nid].pos = (origin[0] + d * dx, y)
                y += h(nid) + gap

    def setup(self) -> dict[str, str]:
        """Prepare every block. A block that fails (missing model, camera...) is marked with an error and
        the rest of the flow still runs. Returns {node_id: error}."""
        self._previous = {}
        for node in self.nodes.values():
            node._owner = weakref.ref(self)    # blocks like "Reçete Seç" reach their flow through this
            if not node.safe_setup():
                print(f"[{node.title}] {node.setup_error}")
        self._apply_pending_recipe()           # a recipe chosen by a block's setup: before the first frame
        return {n.id: n.setup_error for n in self.nodes.values() if n.setup_error}

    def teardown(self) -> None:
        for node in self.nodes.values():
            if not node.enabled or node.setup_error:
                continue
            try:
                node.teardown()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def run_once(self) -> dict[str, NodeResult]:
        """Evaluate every node once. Raises StopFlow when a source is exhausted."""
        if self._pending_recipe is not None:
            self._apply_pending_recipe()      # between frames, never in the middle of one
        results: dict[str, NodeResult] = {}
        incoming: dict[str, list[Connection]] = {}
        for c in self.connections:
            incoming.setdefault(c.dst, []).append(c)
        previous = getattr(self, "_previous", {})
        for nid in self.order():
            node = self.nodes[nid]
            inputs: dict[str, Any] = {name: None for name, _ in node.inputs}
            upstream_failed, upstream_waiting = False, False
            for c in incoming.get(nid, []):
                if self.is_feedback(c):
                    # last frame's value; False (not None) until there is one, so "connected" stays visible
                    prev = previous.get(c.src)
                    value = prev.outputs.get(c.src_port) if prev is not None and not prev.error else None
                    inputs[c.dst_port] = False if value is None else value
                    continue
                src_res = results.get(c.src)
                if src_res is None or src_res.error:
                    upstream_failed = True
                    continue
                if src_res.waiting:
                    upstream_waiting = True
                    continue
                inputs[c.dst_port] = src_res.outputs.get(c.src_port)
            if upstream_failed:
                results[nid] = NodeResult({}, 0.0, "Önceki blok hata verdi")
                continue
            if upstream_waiting:
                results[nid] = NodeResult({}, 0.0, waiting="önceki blok bekliyor")
                continue
            if not node.enabled:
                results[nid] = NodeResult(node.bypass(inputs), 0.0)
                continue
            if node.setup_error:
                results[nid] = NodeResult({}, 0.0, node.setup_error, getattr(node, "setup_where", ""))
                continue
            empty = next((p for p, t in node.inputs if t == PortType.IMAGE and isinstance(inputs.get(p), np.ndarray)
                          and inputs[p].size == 0), None)
            if empty:
                results[nid] = NodeResult({}, 0.0, f"Boş görüntü geldi ({empty}: 0 piksel)")
                continue
            t0 = time.perf_counter()
            try:
                outputs = node.process(inputs) or {}
                err = None
            except StopFlow:
                raise
            except NoNewFrame as exc:
                results[nid] = NodeResult({}, (time.perf_counter() - t0) * 1000, waiting=str(exc) or "bekliyor")
                continue
            except Exception as exc:  # noqa: BLE001
                outputs, err = {}, f"{type(exc).__name__}: {exc}"
                where = _where(exc)
            results[nid] = NodeResult(outputs, (time.perf_counter() - t0) * 1000, err, where if err else "")
        # feedback links read the last real values: a block skipped while waiting keeps what it had before
        self._previous = {nid: previous[nid] if r.waiting and nid in previous else r for nid, r in results.items()}
        return results

    def idle(self, results: dict[str, NodeResult]) -> bool:
        """Nothing new happened: every source that ran is waiting (e.g. for its hardware trigger)."""
        waiting = [r.waiting for nid, r in results.items() if self.nodes[nid].is_source and not r.error]
        return bool(waiting) and all(waiting)

    @property
    def has_source(self) -> bool:
        return any(n.is_source and n.enabled and not n.setup_error for n in self.nodes.values())

    def wants_frames(self) -> bool:
        """A block still has work that only shows on a later frame (e.g. an AI answer being written)."""
        return any(n.enabled and not n.setup_error and n.wants_frames() for n in self.nodes.values())

    def run(self, max_frames: int | None = None, on_frame: Callable[[int, dict], bool | None] | None = None,
            should_stop: Callable[[], bool] = lambda: False, max_fps: float = 0) -> int:
        """Run the frame loop. Without a source node the flow runs until no block has pending work."""
        frames = 0
        limiter = FrameLimiter(max_fps)
        self.setup()
        has_source = self.has_source       # after setup: a camera that failed to open is no source
        try:
            while not should_stop():
                if has_source:
                    limiter.wait()
                elif frames:
                    time.sleep(0.02)       # only waiting for background work: no need to spin
                try:
                    results = self.run_once()
                except StopFlow:
                    break
                if self.idle(results):     # only waiting for a trigger: not a frame
                    continue
                frames += 1
                if on_frame and on_frame(frames, results) is False:
                    break
                if (not has_source and not self.wants_frames()) or (max_frames and frames >= max_frames):
                    break
        finally:
            self.teardown()
        return frames

    # recipes -----------------------------------------------------------------
    @property
    def root(self) -> "Graph":
        g = self
        while g.parent is not None:
            g = g.parent
        return g

    def request_recipe(self, wanted, base: int = 1) -> dict:
        """While running: apply a recipe of the whole flow before the next frame. Raises ValueError when there is
        no such recipe."""
        from . import recipes
        root = self.root
        r = recipes.find(root, wanted, base)
        root._pending_recipe = r["name"]
        return r

    def apply_recipe(self, wanted, live: bool = False) -> list[str]:
        """Put a recipe's values into the blocks now (not running, or between frames). Returns the problems."""
        from . import recipes
        return recipes.apply(self, recipes.find(self, wanted), live)

    def _apply_pending_recipe(self) -> None:
        from . import recipes
        name, self._pending_recipe = self._pending_recipe, None
        r = recipes.get(self, name) if name is not None else None
        if r is not None:
            self.active_recipe = r["name"]
            self.recipe_events.append((r["name"], recipes.apply(self, r, live=True)))
            del self.recipe_events[:-20]       # nobody reading them (tests, scripts): keep only the last ones

    # serialization -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": "visionflow",
            "version": FORMAT_VERSION,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "connections": [c.to_dict() for c in self.connections],
            **({"frames": [dict(f) for f in self.frames]} if self.frames else {}),
            **({"operator": copy.deepcopy(self.operator)} if self.operator else {}),
            **({"recipes": self._export_recipes()} if self.recipes else {}),
            **({"accel": dict(self.accel)} if self.accel else {}),
        }

    def _export_recipes(self) -> list[dict]:
        from . import recipes
        return recipes.export(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        """Build a graph. Unknown blocks (e.g. a plugin that is not installed) and broken links are
        skipped and listed in ``graph.warnings`` instead of failing the whole file."""
        g = cls()
        if not isinstance(data, dict) or not isinstance(data.get("nodes", []), list):
            raise GraphError("Bu bir VisionFlow akış dosyası değil")
        if isinstance(data.get("version"), int) and data["version"] > FORMAT_VERSION:
            g.warnings.append("Dosya daha yeni bir VisionFlow sürümüyle kaydedilmiş; bazı ayarlar eksik olabilir")
        for i, nd in enumerate(data.get("nodes", [])):
            if not isinstance(nd, dict) or not isinstance(nd.get("type"), str):
                g.warnings.append(f"Bozuk blok kaydı atlandı (sıra {i + 1})")
                continue
            if nd["type"] not in registry.NODE_REGISTRY:
                g.warnings.append(f"Bilinmeyen blok atlandı: {nd['type']}")
                continue
            nid = str(nd["id"]) if nd.get("id") not in (None, "") else None
            if nid in g.nodes:
                g.warnings.append(f"Aynı kimlikli iki blok var ({nid}); ikincisine yeni kimlik verildi")
                nid = None
            node = registry.create(nd["type"], nid)
            pos = nd.get("pos")
            try:
                node.pos = (float(pos[0]), float(pos[1]))
            except (TypeError, ValueError, IndexError, KeyError):
                node.pos = (0.0, 0.0)
            node.enabled = not nd.get("disabled", False)
            node.color = clean_color(nd.get("color"))
            values = nd.get("values", {})
            params = {p.name: p for p in node.params}
            for k, val in (values.items() if isinstance(values, dict) else []):
                if k not in params:
                    continue
                try:
                    node.values[k] = params[k].coerce(val, clamp=True)
                except (TypeError, ValueError) as exc:
                    if params[k].kind == "choice" and isinstance(val, str):
                        node.values[k] = val          # unknown option (newer version): keep it, the editor shows it
                    else:
                        g.warnings.append(f"{node.title}: '{params[k].display}' ayarı geçersiz ({val!r}: {exc}), "
                                          f"varsayılan kullanıldı")
            node.load_extra(nd, g.warnings)
            g.add(node)
        for cd in data.get("connections", []) or []:
            if not isinstance(cd, dict) or not all(isinstance(cd.get(k), str)
                                                   for k in ("src", "src_port", "dst", "dst_port")):
                g.warnings.append("Bozuk bağlantı kaydı atlandı")
                continue
            if cd["src"] not in g.nodes or cd["dst"] not in g.nodes:
                continue
            try:
                g.connect(cd["src"], cd["src_port"], cd["dst"], cd["dst_port"])
            except GraphError as exc:
                g.warnings.append(f"Bağlantı atlandı ({cd['src_port']} → {cd['dst_port']}): {exc}")
        for fd in data.get("frames", []) or []:
            frame = clean_frame(fd) if isinstance(fd, dict) else None
            if frame is None:
                g.warnings.append("Bozuk çerçeve kaydı atlandı")
                continue
            ids = {f["id"] for f in g.frames}
            if frame["id"] in ids:
                n = len(ids) + 1
                while f"c{n}" in ids:
                    n += 1
                frame["id"] = f"c{n}"
            g.frames.append(frame)
        if data.get("operator") is not None:          # older versions ignore this key
            from .operator import clean_operator
            g.operator = clean_operator(data["operator"], g, g.warnings)
        from . import recipes
        g.recipes = recipes.clean(data.get("recipes"), g, g.warnings)
        from ..accel import clean_graph_accel
        g.accel = clean_graph_accel(data.get("accel"), g.warnings)
        return g

    def save(self, path: str) -> None:
        """Write to a temporary file first, so a failed save never destroys the existing file."""
        import os
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @classmethod
    def load(cls, path: str) -> "Graph":
        import pathlib
        from . import node as _node
        with open(path, encoding="utf-8") as f:
            g = cls.from_dict(json.load(f))
        _node.FLOW_DIR = pathlib.Path(path).resolve().parent     # relative file settings are found from here too
        _node.FLOW_FILE = pathlib.Path(path).resolve()
        return g
