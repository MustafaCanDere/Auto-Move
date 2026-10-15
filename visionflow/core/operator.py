"""Operator panel settings saved in the flow file ("operator" key) and the optional edit lock (salted password hash).

The lock lives in the flow file itself, so it travels with the flow to the line PC and holds for every Windows
account. It only guards against accidental or casual changes on the shop floor: anyone who can edit the JSON file
with a text editor can remove it.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets

COUNTERS = ("good", "bad", "total")
HISTORY_DEFAULT, HISTORY_MAX = 20, 500
LOCK_ITERATIONS = 200_000
_ITER_RANGE = (1_000, 10_000_000)
_HEX = re.compile(r"[0-9a-f]+")


def _ref_ok(graph, ref, kinds) -> bool:
    """{"node": id, "port": name} that names an existing block output of one of the port types ``kinds``."""
    from .graph import find_node
    if not isinstance(ref, dict) or not isinstance(ref.get("node"), str) or not isinstance(ref.get("port"), str):
        return False
    node = find_node(graph, ref["node"]) if graph is not None else None
    if node is None:
        return graph is None
    return dict(node.outputs).get(ref["port"]) in kinds


def clean_operator(data, graph=None, warnings: list | None = None) -> dict | None:
    """The "operator" section of a flow file, checked against the graph's blocks. Unknown or broken entries are
    dropped (with a warning) instead of breaking the file; a broken lock is dropped too, because it could never be
    unlocked."""
    from .node import PortType
    warn = warnings.append if warnings is not None else (lambda _m: None)
    if not isinstance(data, dict):
        warn("Bozuk operatör paneli ayarı atlandı")
        return None
    out: dict = {"title": str(data.get("title", ""))[:120]}
    img = data.get("image", "")
    if img and isinstance(img, str):
        from .graph import find_node
        node = find_node(graph, img) if graph is not None else None
        if graph is not None and (node is None or not any(t in (PortType.IMAGE, PortType.ANY) for _, t in node.outputs)):
            warn("Operatör paneli: görüntü bloğu bulunamadı, otomatik seçilecek")
            img = ""
        out["image"] = img
    else:
        out["image"] = ""
    for key, kinds, label in [("lamp", (PortType.BOOL, PortType.ANY), "lamba"),
                              *[(c, (PortType.NUMBER, PortType.BOOL, PortType.ANY), "sayaç") for c in COUNTERS]]:
        ref = data.get(key)
        if ref in (None, "", {}):
            out[key] = None
        elif _ref_ok(graph, ref, kinds):
            out[key] = {"node": ref["node"], "port": ref["port"]}
        else:
            warn(f"Operatör paneli: {label} bağlantısı geçersiz, kaldırıldı")
            out[key] = None
    try:
        out["history"] = max(1, min(HISTORY_MAX, int(data.get("history", HISTORY_DEFAULT))))
    except (TypeError, ValueError):
        out["history"] = HISTORY_DEFAULT
    if data.get("screen") is not None:            # a designed screen (Araçlar → Ekran tasarımcısı, core/screens.py)
        from .screens import clean_screen
        screen = clean_screen(data["screen"], graph, warnings)
        if screen is not None:
            out["screen"] = screen
            out["layout"] = "screen" if data.get("layout") == "screen" else "auto"
    lock = data.get("lock")
    if lock is not None:
        if lock_ok(lock):
            out["lock"] = {"salt": lock["salt"], "hash": lock["hash"], "iterations": int(lock["iterations"])}
        else:
            warn("Operatör paneli: düzenleme kilidi bozuk, kaldırıldı")
    return out


def lock_ok(lock) -> bool:
    return (isinstance(lock, dict) and isinstance(lock.get("salt"), str) and isinstance(lock.get("hash"), str)
            and 16 <= len(lock["salt"]) <= 128 and len(lock["hash"]) == 64
            and bool(_HEX.fullmatch(lock["salt"])) and bool(_HEX.fullmatch(lock["hash"]))
            and isinstance(lock.get("iterations"), int) and not isinstance(lock.get("iterations"), bool)
            and _ITER_RANGE[0] <= lock["iterations"] <= _ITER_RANGE[1])


def _derive(password: str, salt: str, iterations: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations).hex()


def make_lock(password: str, iterations: int = LOCK_ITERATIONS) -> dict:
    """A salted PBKDF2-SHA256 hash of the password (the password itself is never saved)."""
    if not isinstance(password, str) or len(password) < 4:
        raise ValueError("Şifre en az 4 karakter olmalı")
    salt = secrets.token_hex(16)
    return {"salt": salt, "hash": _derive(password, salt, iterations), "iterations": iterations}


def check_password(lock, password) -> bool:
    """True when there is no (valid) lock, or the password matches it."""
    if not lock:
        return True
    if not lock_ok(lock) or not isinstance(password, str):
        return False
    return hmac.compare_digest(_derive(password, lock["salt"], lock["iterations"]), lock["hash"])


def locked(graph) -> dict | None:
    """The flow's edit lock, if it has one."""
    cfg = getattr(graph, "operator", None) or {}
    return cfg.get("lock") or None
