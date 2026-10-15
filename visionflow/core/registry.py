"""Global block registry. Blocks register themselves with @register."""
from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

from .node import Node

NODE_REGISTRY: dict[str, type[Node]] = {}


def register(cls: type[Node]) -> type[Node]:
    # look only at the class itself: a subclass must not inherit (and overwrite) its parent's id
    if not cls.__dict__.get("type_name"):
        cls.type_name = cls.__name__
    if not cls.__dict__.get("title"):
        cls.title = cls.__name__
    old = NODE_REGISTRY.get(cls.type_name)
    if old is not None and old.__qualname__ != cls.__qualname__:
        print(f"Uyarı: '{cls.type_name}' blok kimliği iki kez kullanılmış, sonuncusu geçerli")
    NODE_REGISTRY[cls.type_name] = cls
    return cls


def create(type_name: str, node_id: str | None = None) -> Node:
    if type_name not in NODE_REGISTRY:
        raise KeyError(f"Bilinmeyen blok: {type_name}")
    return NODE_REGISTRY[type_name](node_id)


def by_category() -> dict[str, list[type[Node]]]:
    cats: dict[str, list[type[Node]]] = {}
    for cls in NODE_REGISTRY.values():
        cats.setdefault(cls.category, []).append(cls)
    for items in cats.values():
        items.sort(key=lambda c: c.title)
    return dict(sorted(cats.items()))


def load_builtin() -> None:
    from .. import nodes  # noqa: F401  (imports register everything)


PLUGIN_ERRORS: dict[str, str] = {}      # plugin file → why it did not load (shown in the log at start)


def unregister(type_name: str) -> bool:
    """Remove a block from NODE_REGISTRY if present."""
    if type_name in NODE_REGISTRY:
        del NODE_REGISTRY[type_name]
        return True
    return False


def load_single_plugin(path: str | pathlib.Path) -> tuple[bool, str]:
    """Import a single .py plugin file dynamically into NODE_REGISTRY."""
    path = pathlib.Path(path)
    if not path.is_file():
        return False, f"Dosya bulunamadı: {path}"
    name = f"visionflow_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if not (spec and spec.loader):
        return False, "Modül yükleyici oluşturulamadı"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - one broken plugin must not stop the program from starting
        sys.modules.pop(name, None)
        PLUGIN_ERRORS[path.name] = f"{type(exc).__name__}: {exc}"
        print(f"[Eklenti] {path.name} yüklenemedi: {type(exc).__name__}: {exc}")
        return False, f"{type(exc).__name__}: {exc}"
    PLUGIN_ERRORS.pop(path.name, None)
    return True, f"'{path.stem}' başarıyla yüklendi"


def load_plugins(folder: str | pathlib.Path) -> list[str]:
    """Import every *.py in ``folder`` so users can add their own blocks."""
    loaded = []
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        return loaded
    for path in sorted(folder.glob("*.py")):
        ok, _msg = load_single_plugin(path)
        if ok:
            loaded.append(path.stem)
    return loaded

