"""Recipes (product types, "Reçeteler"): named sets of setting values for chosen blocks of a flow.

A recipe is ``{"name": "Ürün A", "values": {block id: {setting: value}}}``; the flow keeps its recipes in
``Graph.recipes`` and saves them under the "recipes" key of the flow file (older versions ignore it). Settings of
blocks inside a sub-flow are reached through the sub-flow block's exposed settings ("inner id/setting").

Applying a recipe while the flow runs is never done in the middle of a frame: ``Graph.request_recipe`` only marks it,
and the graph applies it before the next frame (blocks whose changed setting is read in setup() are restarted).
"""
from __future__ import annotations

import copy
import numbers
import re
import traceback

MAX_RECIPES = 500
MAX_NAME = 80
SKIP_TYPES = ("RecipeSelect", "Note", "SubflowInput", "SubflowOutput")


def clean_name(name) -> str:
    return re.sub(r"\s+", " ", str(name if name is not None else "")).strip()[:MAX_NAME]


def _fold(name: str) -> str:
    return clean_name(name).casefold()


def coerce_value(node, name: str, value):
    """`value` checked and converted for setting `name` of `node` (numbers out of range are pulled into range).
    Raises ValueError/TypeError with a Turkish reason."""
    p = next((p for p in node.params if p.name == name), None)
    if p is None:
        raise ValueError(f"'{name}' diye bir ayar yok")
    if isinstance(value, bool) or isinstance(value, (str, numbers.Real)):
        try:
            return p.coerce(value, clamp=True)
        except (TypeError, ValueError):
            if p.kind == "choice" and isinstance(value, str):
                return value                    # unknown option (newer version): kept like in Graph.from_dict
            raise
    raise TypeError("değer sayı, yazı veya doğru/yanlış olmalı")


def clean(raw, graph, warnings: list[str]) -> list[dict]:
    """The recipes of a flow file, checked against the flow: anything broken is dropped with a warning instead of
    breaking the flow; values of blocks that are not in the flow (e.g. an uninstalled plugin) are left out."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append("Reçete kayıtları bozuk, atlandı")
        return []
    out, names = [], set()
    for i, r in enumerate(raw):
        if len(out) >= MAX_RECIPES:
            warnings.append(f"En fazla {MAX_RECIPES} reçete okunur; fazlası atlandı")
            break
        name = clean_name(r.get("name")) if isinstance(r, dict) and isinstance(r.get("name"), (str, int)) else ""
        if not name:
            warnings.append(f"Bozuk reçete kaydı atlandı (sıra {i + 1})")
            continue
        if _fold(name) in names:
            warnings.append(f"Aynı adlı iki reçete var ('{name}'); ikincisi atlandı")
            continue
        names.add(_fold(name))
        values = r.get("values") if isinstance(r.get("values"), dict) else {}
        kept: dict[str, dict] = {}
        for nid, pv in values.items():
            node = graph.nodes.get(str(nid))
            if node is None or not isinstance(pv, dict) or node.type_name in SKIP_TYPES:
                continue
            for pname, v in pv.items():
                if not isinstance(pname, str) or pname not in node.values:
                    continue
                try:
                    kept.setdefault(node.id, {})[pname] = coerce_value(node, pname, v)
                except (TypeError, ValueError) as exc:
                    label = node.param(pname).display
                    warnings.append(f"Reçete '{name}': {node.title} '{label}' değeri geçersiz ({v!r}: {exc}), atlandı")
        out.append({"name": name, "values": kept})
    return out


def export(graph) -> list[dict]:
    """The recipes to save: values of blocks or settings no longer in the flow are left out."""
    out = []
    for r in graph.recipes:
        vals = {}
        for nid, pv in r.get("values", {}).items():
            node = graph.nodes.get(nid)
            if node is None:
                continue
            kept = {k: v for k, v in pv.items() if k in node.values}
            if kept:
                vals[nid] = copy.deepcopy(kept)
        out.append({"name": r["name"], "values": vals})
    return out


def names(graph) -> list[str]:
    return [r["name"] for r in graph.recipes]


def get(graph, name: str) -> dict | None:
    key = _fold(name)
    return next((r for r in graph.recipes if _fold(r["name"]) == key), None)


def find(graph, wanted, base: int = 1) -> dict:
    """The recipe named `wanted` (case and extra spaces do not matter), or the `wanted`-th one (a number, or a text
    that is only digits and not a name), counting from `base`. Raises ValueError with a Turkish message."""
    recipes = graph.recipes
    if not recipes:
        raise ValueError("Akışta hiç reçete yok: Araçlar → Reçeteler… ile oluşturun")
    if wanted is None or isinstance(wanted, bool) or not isinstance(wanted, (str, numbers.Real)):
        raise ValueError(f"Reçete adı (yazı) veya numarası (sayı) bekleniyor, gelen: {type(wanted).__name__}")
    if isinstance(wanted, str):
        r = get(graph, wanted)
        if r is not None:
            return r
        s = clean_name(wanted)
        if not re.fullmatch(r"[+-]?\d+", s):
            raise ValueError(f"'{s}' adlı reçete yok. Var olanlar: {', '.join(names(graph))}")
        idx = int(s)
    else:
        f = float(wanted)
        if f != f or f in (float("inf"), float("-inf")) or f != int(f):
            raise ValueError(f"Reçete numarası tam sayı olmalı (gelen: {wanted})")
        idx = int(f)
    i = idx - base
    if not 0 <= i < len(recipes):
        raise ValueError(f"{idx} numaralı reçete yok ({base}…{len(recipes) - 1 + base} arası)")
    return recipes[i]


def capture(graph, node_ids) -> dict[str, dict]:
    """Current values of the given blocks (every setting; sub-flows with their exposed settings)."""
    out = {}
    for nid in node_ids:
        node = graph.nodes.get(nid)
        if node is None or node.type_name in SKIP_TYPES:
            continue
        vals = {k: copy.deepcopy(v) for k, v in node.values.items()
                if isinstance(v, (str, int, float, bool)) and any(p.name == k for p in node.params)}
        if vals:
            out[nid] = vals
    return out


def _needs_restart(node, name: str) -> bool:
    if not node.enabled:
        return False
    p = next((p for p in node.params if p.name == name), None)
    return p is not None and (p.kind in ("path", "folder") or name in node.setup_params or bool(node.setup_error))


def _restart(node) -> str | None:
    if not node.setup_error:
        try:
            node.teardown()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    if not node.safe_setup():
        return f"{node.title}: {node.setup_error}"
    return None


def set_value(graph, nid: str, name: str, value, live: bool = False, restarts: dict | None = None) -> str | None:
    """Set one setting of block `nid` (a sub-flow's exposed setting goes into the block inside it). With `live`
    (the graph is running) the running block inside a sub-flow is changed too, and blocks that read the setting in
    setup() are collected in `restarts` (or restarted now). Returns an error message, or None."""
    from .graph import node_at, sync_running_parents
    node = graph.nodes.get(nid)
    if node is None or name not in node.values:
        return None                                   # block or setting gone: nothing to do
    try:
        v = coerce_value(node, name, value)
    except (TypeError, ValueError) as exc:
        p = next((p for p in node.params if p.name == name), None)
        return f"{node.title}: '{p.display if p else name}' = {value!r} geçersiz ({exc})"
    if node.type_name == "SubFlow" and "/" in name:
        node.set_exposed(name, v)
    else:
        node.values[name] = v
        if node.type_name == "SubFlow" and name == "name":
            node.refresh()                            # its title follows its name
    if not live:
        return None
    if "/" in name:
        path, _, leaf = name.rpartition("/")
        sync_running_parents(graph, f"{nid}/{path}", leaf, v)
        target = node_at(graph, f"{nid}/{path}")
        if target is None:
            return None
        if leaf not in target.values:
            return None
        target.values[leaf] = v
    else:
        target, leaf = node, name
    if _needs_restart(target, leaf):
        if restarts is None:
            return _restart(target)
        restarts[id(target)] = target
    return None


def apply(graph, recipe: dict, live: bool = False) -> list[str]:
    """Put a recipe's values into the blocks of `graph`. Returns the problems (bad values; blocks that could not
    restart), the rest is applied anyway."""
    errors, restarts = [], {}
    for nid, pv in recipe.get("values", {}).items():
        for name, value in pv.items():
            err = set_value(graph, nid, name, value, live, restarts)
            if err:
                errors.append(err)
    for node in restarts.values():
        err = _restart(node)
        if err:
            errors.append(err)
    return errors


def rename_selectors(graph, old: str, new: str) -> int:
    """'Reçete Seç' blocks set to recipe `old` follow its new name. Returns how many changed."""
    n = 0
    for node in graph.nodes.values():
        if node.type_name == "RecipeSelect" and _fold(node.values.get("recipe", "")) == _fold(old):
            node.values["recipe"] = new
            n += 1
    return n
