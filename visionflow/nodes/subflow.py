"""Sub-flows: part of a flow packed into one block ("Alt Akış"), with its own inputs and outputs.

Inside a sub-flow, "Alt Akış Girişi" blocks hand on what comes into the sub-flow block and "Alt Akış Çıkışı" blocks
give its outputs: their names are the port names of the sub-flow block, their vertical order is the port order.
The blocks inside are saved with the sub-flow block itself (its "inner" flow), so a sub-flow can be copied, pasted,
saved to the library and nested in another sub-flow.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import pathlib
import re
import uuid

from ..core import Graph, GraphError, Node, NoNewFrame, Param, PortType, register, registry
from ..core.node import clean_color

CAT = "4b · Alt Akışlar"
TYPES = {"Otomatik": None, "görüntü": PortType.IMAGE, "sayı": PortType.NUMBER, "açık/kapalı": PortType.BOOL,
         "yazı": PortType.TEXT, "tespitler": PortType.DETECTIONS, "konturlar": PortType.CONTOURS,
         "herhangi": PortType.ANY}
BOUNDARY = ("SubflowInput", "SubflowOutput")
MAX_DEPTH = 12                    # sub-flows inside sub-flows: more than this is a broken (or hostile) file
_depth = 0


def empty_flow() -> dict:
    return {"format": "visionflow", "version": 1, "nodes": [], "connections": []}


def _pos(nd: dict) -> tuple[float, float]:
    try:
        return float(nd["pos"][1]), float(nd["pos"][0])
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0, 0.0


def _port_type(nd, port: str, output: bool, depth: int, memo: dict) -> str | None:
    """Type of a port of a block saved as a dict (None when unknown)."""
    if not isinstance(nd, dict) or nd.get("type") in BOUNDARY:
        return None
    if nd.get("type") == "SubFlow":
        if depth > MAX_DEPTH:
            return None
        inner = nd.get("inner")
        if id(inner) not in memo:              # each nested flow is worked out once, however many ports use it
            memo[id(inner)] = (inner, boundary(inner, depth + 1, memo))
        ins, outs = memo[id(inner)][1]
        return next((t for p, _, t in (outs if output else ins) if p == port), None)
    cls = registry.NODE_REGISTRY.get(nd.get("type"))
    if cls is None:
        return None
    return dict(cls.outputs if output else cls.inputs).get(port)


def boundary(inner, depth: int = 0, memo: dict | None = None) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """([(port, block id, type)] of the inputs, the same of the outputs) of a sub-flow's inner flow. Ports are in
    the top-to-bottom order of their blocks; a name used twice gets "_2", "_3"; the "Otomatik" type is taken from
    what the boundary block is wired to inside."""
    if not isinstance(inner, dict):
        return [], []
    memo = {} if memo is None else memo
    nodes = [n for n in inner.get("nodes", []) or [] if isinstance(n, dict)]
    conns = [c for c in inner.get("connections", []) or [] if isinstance(c, dict)]
    by_id = {str(n.get("id")): n for n in nodes}
    result = []
    for kind, default in (("SubflowInput", "giriş"), ("SubflowOutput", "çıkış")):
        ports, used = [], set()
        for n in sorted((n for n in nodes if n.get("type") == kind), key=_pos):
            vals = n.get("values") if isinstance(n.get("values"), dict) else {}
            base = re.sub(r"\s+", " ", str(vals.get("name") or "")).strip()[:40] or default
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            nid = str(n.get("id"))
            t = TYPES.get(vals.get("type"))
            if t is None:
                for c in conns:
                    if kind == "SubflowInput" and str(c.get("src")) == nid:
                        t = _port_type(by_id.get(str(c.get("dst"))), c.get("dst_port"), False, depth, memo)
                    elif kind == "SubflowOutput" and str(c.get("dst")) == nid:
                        t = _port_type(by_id.get(str(c.get("src"))), c.get("src_port"), True, depth, memo)
                    if t and t != PortType.ANY:
                        break
            ports.append((name, nid, t or PortType.ANY))
        result.append(ports)
    return result[0], result[1]


def _has_source(inner, depth: int = 0) -> bool:
    """Does a sub-flow's inner flow hold a block that drives the frame loop (camera, video, timer)?"""
    if not isinstance(inner, dict) or depth > MAX_DEPTH:
        return False
    for nd in inner.get("nodes", []) or []:
        if not isinstance(nd, dict) or nd.get("disabled"):
            continue
        if nd.get("type") == "SubFlow":
            if _has_source(nd.get("inner"), depth + 1):
                return True
            continue
        cls = registry.NODE_REGISTRY.get(nd.get("type"))
        if cls is None:
            continue
        flag = cls.__dict__.get("is_source", getattr(cls, "is_source", False))
        if isinstance(flag, property):          # depends on a setting (e.g. "animate" of the test picture)
            node = cls()
            vals = nd.get("values") if isinstance(nd.get("values"), dict) else {}
            node.values.update({k: v for k, v in vals.items() if k in node.values})
            try:
                flag = node.is_source
            except Exception:  # noqa: BLE001
                flag = False
        if flag:
            return True
    return False


# exposed settings ("Dışarı açılan ayarlar") -----------------------------------------------------------------------
# A sub-flow can show settings of blocks inside it on its own block: the inner flow keeps them as
# inner["exposed"] = [{"node": inner block id, "param": setting name}, ...]. On the sub-flow block such a setting is
# named "inner id/setting" (a sub-flow inside it adds its own part: "inner id/its inner id/setting"), which is also
# the path of the running block ("sub-flow id/inner id") plus the setting. The value lives in the inner block itself,
# so copies of a sub-flow keep their own values and the library, undo and saving need nothing extra.

def _title_of(nd: dict) -> str:
    vals = nd.get("values") if isinstance(nd.get("values"), dict) else {}
    if nd.get("type") == "SubFlow":
        return (re.sub(r"\s+", " ", str(vals.get("name") or "")).strip() or "Alt Akış")[:60]
    cls = registry.NODE_REGISTRY.get(nd.get("type"))
    return (cls.title or cls.type_name) if cls is not None else str(nd.get("type"))


def node_settings(nd, depth: int = 0) -> list[tuple[Param, object]]:
    """(setting, value) of a block saved as a dict; for a sub-flow including its exposed settings."""
    cls = registry.NODE_REGISTRY.get(nd.get("type")) if isinstance(nd, dict) else None
    if cls is None or depth > MAX_DEPTH:
        return []
    vals = nd.get("values") if isinstance(nd.get("values"), dict) else {}
    out = []
    for p in cls.params:
        v = vals.get(p.name, p.default)
        try:
            v = p.coerce(v, clamp=True)
        except (TypeError, ValueError):
            v = v if p.kind == "choice" and isinstance(v, str) else p.default
        out.append((p, v))
    if nd.get("type") == "SubFlow":
        out += exposed_settings(nd.get("inner"), depth + 1)
    return out


def exposed_settings(inner, depth: int = 0) -> list[tuple[Param, object]]:
    """The exposed settings of a sub-flow's inner flow as they appear on the sub-flow block: (setting named
    "inner id/setting" and labelled "<block title>: <setting label>", current value). Entries whose block or
    setting is gone are left out."""
    if not isinstance(inner, dict) or depth > MAX_DEPTH or not isinstance(inner.get("exposed"), list):
        return []
    by_id = {str(n.get("id")): n for n in inner.get("nodes", []) or [] if isinstance(n, dict)}
    out, seen = [], set()
    for e in inner["exposed"]:
        if not isinstance(e, dict) or not isinstance(e.get("node"), str) or not isinstance(e.get("param"), str):
            continue
        key = f"{e['node']}/{e['param']}"
        nd = by_id.get(e["node"])
        if key in seen or nd is None or nd.get("type") in BOUNDARY:
            continue
        found = next(((p, v) for p, v in node_settings(nd, depth) if p.name == e["param"]), None)
        if found is None:
            continue
        seen.add(key)
        p, v = found
        out.append((dataclasses.replace(p, name=key, label=f"{_title_of(nd)}: {p.display}"), v))
    return out


def _set_in(inner, key: str, value, depth: int = 0) -> bool:
    """Write the value of exposed setting `key` into the inner flow (dict) where it lives."""
    nid, _, name = key.partition("/")
    if not isinstance(inner, dict) or not name or depth > MAX_DEPTH:
        return False
    nd = next((n for n in inner.get("nodes", []) or [] if isinstance(n, dict) and str(n.get("id")) == nid), None)
    if nd is None:
        return False
    if not isinstance(nd.get("values"), dict):
        nd["values"] = {}
    nd["values"][name] = value
    if "/" in name and nd.get("type") == "SubFlow":        # a setting the sub-flow inside exposes in its turn
        return _set_in(nd.get("inner"), name, value, depth + 1)
    return True


class InnerError(ValueError):
    """A block inside a sub-flow failed: the error is that block's (logged for it, not for the sub-flow)."""


@register
class SubflowInput(Node):
    title = "Alt Akış Girişi"
    category = CAT
    description = ("Alt akışın içinde kullanılır: alt akış bloğuna dışarıdan bağlanan değeri buradan verir. Adı, alt "
                   "akış bloğundaki girişin adı olur; yukarıdan aşağı sırası girişlerin sırasıdır.")
    outputs = [("value", PortType.ANY)]
    params = [Param("name", "str", "giriş", "Giriş adı"),
              Param("type", "choice", "Otomatik", "Tür (Otomatik: bağlandığı yere göre)", choices=list(TYPES))]

    def process(self, inputs):
        return {"value": self.state.get("value")}


@register
class SubflowOutput(Node):
    title = "Alt Akış Çıkışı"
    category = CAT
    description = ("Alt akışın içinde kullanılır: buraya bağlanan değer alt akış bloğunun çıkışı olur. Adı, çıkışın "
                   "adıdır; yukarıdan aşağı sırası çıkışların sırasıdır.")
    inputs = [("value", PortType.ANY)]
    params = [Param("name", "str", "çıkış", "Çıkış adı"),
              Param("type", "choice", "Otomatik", "Tür (Otomatik: bağlandığı yere göre)", choices=list(TYPES))]

    def process(self, inputs):
        return {}


@register
class SubFlow(Node):
    title = "Alt Akış"
    category = CAT
    description = ("Birden çok bloğu tek blokta toplar: akış sadeleşir, aynı işlem (ör. 'parçayı bul ve ölç') kopyalanıp "
                   "başka yerde veya başka akışta tekrar kullanılır. Bloklar seçilip sağ tık → 'Alt akış yap' ile "
                   "oluşturulur; çift tıklayınca içi açılır. Girişleri ve çıkışları içerideki 'Alt Akış Girişi / "
                   "Çıkışı' bloklarıdır.")
    params = [Param("name", "str", "Alt Akış", "Adı")]

    def __init__(self, node_id: str | None = None):
        super().__init__(node_id)
        self.inner: dict = empty_flow()
        self.refresh()

    # what is inside ------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        """Ports, title, source flag and exposed settings from the inner flow and the name setting."""
        ins, outs = boundary(self.inner)
        self.inputs = [(p, t) for p, _, t in ins]
        self.outputs = [(p, t) for p, _, t in outs]
        self.title = (re.sub(r"\s+", " ", str(self.values.get("name") or "")).strip() or "Alt Akış")[:60]
        self._source = _has_source(self.inner)
        exposed = exposed_settings(self.inner)
        self.params = list(type(self).params) + [p for p, _ in exposed]   # this block's own list of settings
        own = {p.name for p in type(self).params}
        for k in [k for k in self.values if k not in own]:
            del self.values[k]
        for p, v in exposed:
            self.values[p.name] = v
        if isinstance(self.inner, dict) and "exposed" in self.inner:     # forget settings that are gone
            keep = [{"node": p.name.split("/", 1)[0], "param": p.name.split("/", 1)[1]} for p, _ in exposed]
            if keep != self.inner["exposed"]:
                self.inner = {k: v for k, v in self.inner.items() if k != "exposed"}
                if keep:
                    self.inner["exposed"] = keep

    def set_inner(self, data: dict) -> None:
        """New inner flow (already checked: the editor's own graph.to_dict()). The exposed settings stay."""
        exposed = self.inner.get("exposed") if isinstance(self.inner, dict) else None
        self.inner = copy.deepcopy(data)
        if "exposed" not in self.inner and exposed:
            self.inner["exposed"] = copy.deepcopy(exposed)
        self.refresh()

    # exposed settings ------------------------------------------------------------------------------------------
    def exposed_keys(self) -> list[str]:
        return [p.name for p in self.params if "/" in p.name]

    def is_exposed(self, inner_id: str, name: str) -> bool:
        return f"{inner_id}/{name}" in self.values and any(p.name == f"{inner_id}/{name}" for p in self.params)

    def expose(self, inner_id: str, name: str, on: bool = True) -> bool:
        """Show (or stop showing) setting `name` of the block `inner_id` inside on this block. Returns False when
        there is no such block or setting inside."""
        entries = [e for e in (self.inner.get("exposed") or []) if isinstance(e, dict)
                   and not (e.get("node") == inner_id and e.get("param") == name)]
        if on:
            nd = next((n for n in self.inner.get("nodes", []) or [] if isinstance(n, dict)
                       and str(n.get("id")) == inner_id), None)
            if nd is None or nd.get("type") in BOUNDARY or not any(p.name == name for p, _ in node_settings(nd)):
                return False
            entries.append({"node": inner_id, "param": name})
        self.inner = dict(self.inner)
        if entries:
            self.inner["exposed"] = entries
        else:
            self.inner.pop("exposed", None)
        self.refresh()
        return True

    def set_exposed(self, key: str, value) -> None:
        """New value of an exposed setting: kept on this block and in the block inside (where it is used)."""
        self.values[key] = value
        _set_in(self.inner, key, value)

    def inner_graph(self, warnings: list[str] | None = None) -> Graph:
        global _depth
        _depth += 1
        try:
            if _depth > MAX_DEPTH:
                raise GraphError("Alt akışlar çok derin iç içe")
            g = Graph.from_dict(self.inner)
        finally:
            _depth -= 1
        if warnings is not None:
            warnings += [f"Alt akış '{self.title}': {w}" for w in g.warnings]
        return g

    def extra(self) -> dict:
        return {"inner": copy.deepcopy(self.inner)}

    def load_extra(self, data: dict, warnings: list[str]) -> None:
        inner = data.get("inner")
        if inner is None:
            self.inner = empty_flow()
        elif not isinstance(inner, dict):
            warnings.append(f"Alt akış '{self.values.get('name')}': içeriği bozuk, boş açıldı")
            self.inner = empty_flow()
        else:
            self.inner = inner
            self.refresh()
            exposed = self.inner.get("exposed")
            try:          # checked like any flow: unknown blocks and broken links are dropped with a warning
                g = self.inner_graph(warnings)
                self.inner = g.to_dict()
                if exposed:
                    self.inner["exposed"] = copy.deepcopy(exposed)
            except (GraphError, RecursionError) as exc:
                warnings.append(f"Alt akış '{self.title}': {exc}; boş açıldı")
                self.inner = empty_flow()
        self.refresh()

    # running ---------------------------------------------------------------------------------------------------
    @property
    def is_source(self):
        g = self.state.get("graph") if isinstance(getattr(self, "state", None), dict) else None
        return g.has_source if g is not None else getattr(self, "_source", False)

    def wants_frames(self) -> bool:
        g = self.state.get("graph")
        return g is not None and g.wants_frames()

    def setup(self):
        g = self.inner_graph()
        ins, outs = boundary(g.to_dict())
        feeds = {}
        for p, nid, _ in outs:
            c = next((c for c in g.connections if c.dst == nid and c.dst_port == "value"), None)
            feeds[p] = (nid, c.src, c.src_port) if c else (nid, None, None)
        self.state.update(graph=g, ins={p: nid for p, nid, _ in ins}, outs=feeds)
        owner = getattr(self, "_owner", None)
        g.parent = owner() if owner is not None else None     # e.g. a "Reçete Seç" inside reaches the whole flow
        g.setup()          # a block inside that cannot start shows its error inside and on this block

    def process(self, inputs):
        g: Graph = self.state["graph"]
        for p, nid in self.state["ins"].items():
            node = g.nodes.get(nid)
            if node is not None:
                node.state["value"] = inputs.get(p)
        results = g.run_once()               # StopFlow (end of a video inside) ends the whole flow, as outside
        self.state["results"] = results      # the editor shows the blocks inside (taken by the flow worker)
        if g.idle(results):
            raise NoNewFrame("içerideki kaynak bekliyor")
        out = {}
        for p, (nid, src, port) in self.state["outs"].items():
            r = results.get(src) if src else None
            if r is not None and r.waiting:
                raise NoNewFrame("içerideki blok bekliyor")
            enabled = g.nodes[nid].enabled if nid in g.nodes else False
            out[p] = r.outputs.get(port) if r is not None and not r.error and enabled else None
        failed = next(((nid, r) for nid, r in results.items() if r.error and r.error != "Önceki blok hata verdi"),
                      None)
        if failed:
            raise InnerError(f"içerideki '{g.nodes[failed[0]].title}' bloğu: {failed[1].error}")
        return out

    def teardown(self):
        g = self.state.get("graph")
        if g is not None:
            g.teardown()


# grouping ------------------------------------------------------------------------------------------------------

def _new_id(taken) -> str:
    while True:
        nid = uuid.uuid4().hex[:8]
        if nid not in taken:
            return nid


def group(graph: Graph, ids, name: str = "Alt Akış") -> SubFlow:
    """Replace the blocks `ids` with one sub-flow block holding them. Links into the group become inputs (one per
    outside output, however many blocks inside it feeds), links out of it become outputs. Raises GraphError, and
    leaves the flow as it was, when the group cannot be made."""
    ids = [i for i in dict.fromkeys(ids) if i in graph.nodes]
    if not ids:
        raise GraphError("Alt akış yapmak için önce blokları seçin")
    if any(graph.nodes[i].type_name in BOUNDARY for i in ids):
        raise GraphError("'Alt Akış Girişi / Çıkışı' blokları gruba alınamaz: onlar bu alt akışın kendi girişleri ve "
                         "çıkışlarıdır. Seçimden çıkarın.")
    inside = set(ids)
    for c in graph.connections:
        if (c.src in inside) != (c.dst in inside) and graph.is_feedback(c):
            raise GraphError("Bir geri besleme bağlantısı (önceki kareden gelen) grubun sınırından geçiyor: iki "
                             "ucundaki blokları birlikte seçin")
    nodes = [graph.nodes[i] for i in ids]
    minx = min(n.pos[0] for n in nodes)
    maxx = max(n.pos[0] for n in nodes)
    pos = lambda nid: graph.nodes[nid].pos  # noqa: E731
    inner = empty_flow()
    inner["nodes"] = [n.to_dict() for n in nodes]
    inner["connections"] = [c.to_dict() for c in graph.connections if c.src in inside and c.dst in inside]
    taken = set(graph.nodes)
    incoming: dict[tuple, list] = {}
    for c in sorted((c for c in graph.connections if c.dst in inside and c.src not in inside),
                    key=lambda c: (pos(c.dst)[1], pos(c.dst)[0])):
        incoming.setdefault((c.src, c.src_port), []).append(c)
    outgoing: dict[tuple, list] = {}
    for c in sorted((c for c in graph.connections if c.src in inside and c.dst not in inside),
                    key=lambda c: (pos(c.src)[1], pos(c.src)[0])):
        outgoing.setdefault((c.src, c.src_port), []).append(c)
    in_ids, out_ids = {}, {}
    for k, (key, conns) in enumerate(incoming.items()):
        bid = in_ids[key] = _new_id(taken)
        taken.add(bid)
        y = pos(conns[0].dst)[1] + k * 0.01          # keeps the order of ports even for blocks at the same height
        inner["nodes"].append({"id": bid, "type": "SubflowInput", "pos": [minx - 260, y],
                               "values": {"name": conns[0].dst_port, "type": "Otomatik"}})
        inner["connections"] += [{"src": bid, "src_port": "value", "dst": c.dst, "dst_port": c.dst_port}
                                 for c in conns]
    for k, (key, conns) in enumerate(outgoing.items()):
        bid = out_ids[key] = _new_id(taken)
        taken.add(bid)
        inner["nodes"].append({"id": bid, "type": "SubflowOutput", "pos": [maxx + 260, pos(key[0])[1] + k * 0.01],
                               "values": {"name": key[1], "type": "Otomatik"}})
        inner["connections"].append({"src": key[0], "src_port": key[1], "dst": bid, "dst_port": "value"})
    sub = registry.create("SubFlow")
    sub.values["name"] = (str(name).strip() or "Alt Akış")[:60]
    sub.pos = (minx, sum(n.pos[1] for n in nodes) / len(nodes))
    warnings: list[str] = []
    sub.load_extra({"inner": inner}, warnings)
    ins, outs = boundary(sub.inner)
    port_in = {nid: p for p, nid, _ in ins}
    port_out = {nid: p for p, nid, _ in outs}
    before_nodes, before_conns = dict(graph.nodes), list(graph.connections)
    try:
        for i in ids:
            graph.remove(i)
        graph.add(sub)
        for (src, sp), bid in in_ids.items():
            graph.connect(src, sp, sub.id, port_in[bid])
        for key, bid in out_ids.items():
            for c in outgoing[key]:
                graph.connect(sub.id, port_out[bid], c.dst, c.dst_port)
    except (GraphError, KeyError) as exc:
        graph.nodes, graph.connections = before_nodes, before_conns
        if "döngü" in str(exc):
            raise GraphError("Bu bloklar gruplanamaz: aralarındaki bir yol seçilmeyen bir bloktan geçiyor (döngü "
                             "olurdu). O bloğu da seçin.") from None
        raise GraphError(f"Alt akış yapılamadı: {exc}") from None
    _grouped_refs(graph, sub, inside)
    return sub


def _remap_operator(graph: Graph, remap) -> None:
    """Rewrite the block ids the operator panel settings name (`remap(id) -> new id, or None when gone`)."""
    cfg = graph.operator
    if not cfg:
        return
    cfg = dict(cfg)
    if cfg.get("image"):
        cfg["image"] = remap(cfg["image"], None) or ""
    for key in ("lamp", "good", "bad", "total"):
        ref = cfg.get(key)
        if isinstance(ref, dict) and isinstance(ref.get("node"), str):
            new = remap(ref["node"], ref.get("port"))
            cfg[key] = new if isinstance(new, dict) else ({"node": new, "port": ref.get("port")} if new else None)
    if isinstance(cfg.get("screen"), dict):          # the designed screen's bindings too
        from ..core.screens import remap_screen
        cfg["screen"] = remap_screen(cfg["screen"], remap)
    from ..core.operator import clean_operator
    graph.operator = clean_operator(cfg, graph, graph.warnings)


def _grouped_refs(graph: Graph, sub: SubFlow, inside: set) -> None:
    """After grouping: recipe values of the moved blocks become exposed settings of the new sub-flow block (so they
    are kept and still applied), and operator panel outputs of them are named "sub-flow id/block id"."""
    for r in graph.recipes:
        values = r.get("values", {})
        moved = {nid: pv for nid, pv in values.items() if nid in inside}
        if not moved:
            continue
        for nid in moved:
            del values[nid]
        for nid, pv in moved.items():
            for name, v in pv.items():
                key = f"{nid}/{name}"
                if key not in sub.values and not sub.expose(nid, name):
                    continue
                values.setdefault(sub.id, {})[key] = v

    def remap(ref, _port):
        return f"{sub.id}/{ref}" if ref.split("/", 1)[0] in inside else ref
    _remap_operator(graph, remap)


def ungroup(graph: Graph, sub_id: str) -> list[str]:
    """Put the blocks of a sub-flow back in its place (the opposite of group); returns the ids of those blocks."""
    sub = graph.nodes.get(sub_id)
    if not isinstance(sub, SubFlow):
        raise GraphError("Bu bir alt akış bloğu değil")
    warnings: list[str] = []
    inner = sub.inner_graph(warnings)
    ins, outs = boundary(inner.to_dict())
    in_port = {nid: p for p, nid, _ in ins}
    out_port = {nid: p for p, nid, _ in outs}
    feeds = {c.dst_port: (c.src, c.src_port) for c in graph.connections if c.dst == sub_id}
    takes: dict[str, list] = {}
    for c in graph.connections:
        if c.src == sub_id:
            takes.setdefault(c.src_port, []).append((c.dst, c.dst_port))
    real = [n for n in inner.nodes.values() if n.id not in in_port and n.id not in out_port]
    if real:
        dx = sub.pos[0] - min(n.pos[0] for n in real)
        dy = sub.pos[1] - min(n.pos[1] for n in real)
    else:
        dx = dy = 0.0
    graph.remove(sub_id)
    idmap = {}
    for n in real:
        idmap[n.id] = n.id if n.id not in graph.nodes else _new_id(set(graph.nodes) | {x.id for x in real})
        n.id = idmap[n.id]
        n.pos = (n.pos[0] + dx, n.pos[1] + dy)
        graph.add(n)
    links = []
    for c in inner.connections:
        if c.src in in_port:
            if c.src_port != "value" or in_port[c.src] not in feeds:
                continue
            srcs = [feeds[in_port[c.src]]]
        elif c.src in idmap:
            srcs = [(idmap[c.src], c.src_port)]
        else:
            continue
        if c.dst in out_port:
            dsts = takes.get(out_port[c.dst], [])
        elif c.dst in idmap:
            dsts = [(idmap[c.dst], c.dst_port)]
        else:
            continue
        links += [(s, sp, d, dp) for s, sp in srcs for d, dp in dsts]
    for s, sp, d, dp in links:
        try:
            graph.connect(s, sp, d, dp)
        except (GraphError, KeyError) as exc:
            graph.warnings.append(f"Bağlantı atlandı ({sp} → {dp}): {exc}")
    _ungrouped_refs(graph, sub_id, idmap, inner, out_port)
    for f in inner.frames:                      # comment frames inside come out too, beside their blocks
        ids = {x["id"] for x in graph.frames}
        f = dict(f, rect=[f["rect"][0] + dx, f["rect"][1] + dy, f["rect"][2], f["rect"][3]])
        while f["id"] in ids:
            f["id"] = f"c{_new_id(ids)}"
        graph.frames.append(f)
    return [idmap[n] for n in idmap]


def _ungrouped_refs(graph: Graph, sub_id: str, idmap: dict, inner: Graph, out_port: dict) -> None:
    """After ungrouping: recipe values of the sub-flow's exposed settings go back to the blocks themselves, and
    operator panel outputs inside it (or of its outputs) name those blocks again."""
    for r in graph.recipes:
        values = r.get("values", {})
        pv = values.pop(sub_id, None)
        if not isinstance(pv, dict):
            continue
        for key, v in pv.items():
            nid, _, name = key.partition("/")
            if name and nid in idmap:
                values.setdefault(idmap[nid], {})[name] = v
    feeds = {}                                  # sub-flow output port -> the block output inside that gives it
    for c in inner.connections:
        if c.dst in out_port and c.dst_port == "value" and c.src in idmap:
            feeds[out_port[c.dst]] = {"node": idmap[c.src], "port": c.src_port}

    def remap(ref, port):
        head, _, rest = ref.partition("/")
        if head != sub_id:
            return ref
        if not rest:                            # an output of the sub-flow block itself: the block giving it
            if port is None:                    # the live picture: the block giving its first output
                return next((f["node"] for f in feeds.values()), None)
            return feeds.get(port)
        nid, _, deeper = rest.partition("/")
        if nid not in idmap:
            return None
        return idmap[nid] + (f"/{deeper}" if deeper else "")
    _remap_operator(graph, remap)


def apply_inner(parent: Graph, node: "SubFlow", data: dict) -> bool:
    """Give a sub-flow block its edited inner flow. Links of the outer flow follow renamed inputs/outputs (matched by
    their boundary block); links to removed ports, or ports whose type no longer fits, are dropped. Returns True when
    the block's ports changed."""
    import dataclasses
    old_in, old_out = boundary(node.inner)
    new_in, new_out = boundary(data)
    ren_in = {p: q for p, bid, _ in old_in for q, bid2, _ in new_in if bid == bid2}
    ren_out = {p: q for p, bid, _ in old_out for q, bid2, _ in new_out if bid == bid2}
    before = (list(node.inputs), list(node.outputs))
    conns = []
    for c in parent.connections:
        if c.dst == node.id and c.dst_port in ren_in:
            c = dataclasses.replace(c, dst_port=ren_in[c.dst_port])
        elif c.src == node.id and c.src_port in ren_out:
            c = dataclasses.replace(c, src_port=ren_out[c.src_port])
        conns.append(c)
    parent.connections = conns
    node.set_inner(data)
    drop_stale_links(parent, node)
    return before != (list(node.inputs), list(node.outputs))


def drop_stale_links(graph: Graph, node: Node) -> int:
    """Remove the links of `node` to ports it no longer has (or whose type changed): after its sub-flow's inner
    inputs and outputs were edited. Returns how many were removed."""
    ins, outs = dict(node.inputs), dict(node.outputs)
    keep = []
    for c in graph.connections:
        ok = True
        if c.dst == node.id:
            src = graph.nodes.get(c.src)
            ok = c.dst_port in ins and src is not None and c.src_port in dict(src.outputs) and \
                PortType.compatible(src.output_type(c.src_port), ins[c.dst_port])
        elif c.src == node.id:
            dst = graph.nodes.get(c.dst)
            ok = c.src_port in outs and dst is not None and c.dst_port in dict(dst.inputs) and \
                PortType.compatible(outs[c.src_port], dst.input_type(c.dst_port))
        if ok:
            keep.append(c)
    removed = len(graph.connections) - len(keep)
    graph.connections = keep
    return removed


# library -------------------------------------------------------------------------------------------------------

def library_dir():
    from ..ai.config import home
    return home() / "alt_akislar"


def _safe_name(name: str) -> str:
    s = re.sub(r"[^\w\- ]+", "_", name, flags=re.UNICODE).strip(" ._") or "alt_akis"
    return s[:60]


def save_to_library(sub: SubFlow, folder=None) -> pathlib.Path:
    """Keep a sub-flow for other flows: <home>/alt_akislar/<name>.json (an existing one is replaced)."""
    folder = pathlib.Path(folder) if folder else library_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_safe_name(sub.title)}.json"
    tmp = path.with_suffix(".tmp")
    data = {"format": "visionflow-subflow", "name": sub.title, "inner": sub.inner}
    if sub.color:
        data["color"] = sub.color
    tmp.write_text(json.dumps(data,
                              ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def library(folder=None) -> list[tuple[str, pathlib.Path]]:
    folder = pathlib.Path(folder) if folder else library_dir()
    out = []
    for p in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("format") == "visionflow-subflow":
            out.append((str(data.get("name") or p.stem), p))
    return out


def load_from_library(path) -> tuple[SubFlow, list[str]]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "visionflow-subflow":
        raise ValueError(f"Bu bir alt akış dosyası değil: {path}")
    sub = registry.create("SubFlow")
    sub.values["name"] = str(data.get("name") or "Alt Akış")[:60]
    sub.color = clean_color(data.get("color"))
    warnings: list[str] = []
    sub.load_extra(data, warnings)
    return sub, warnings
