"""Draw a block's region (rectangle, line, polygon) with the mouse on the preview: conversion between what was drawn
and the block's settings. Pure functions, so they are tested without a window."""
from __future__ import annotations

HINTS = {"rect": "Bölge çizmek için önizlemede fareyle sürükleyin",
         "line": "Çizgi çizmek için önizlemede fareyle sürükleyin",
         "poly": "Köşelere tıklayın, bitirmek için çift tıklayın veya sağ tıklayın",
         "quad": "Dört köşeye sırayla tıklayın (sol üst, sağ üst, sağ alt, sol alt)"}


def _ints(text: str) -> list[int] | None:
    try:
        return [int(round(float(v))) for v in str(text).split(",") if v.strip()]
    except ValueError:
        return None


def shape_from_values(spec, values: dict):
    """The block's current region as ("rect", (x, y, w, h)) or ("line"/"poly"/"quad", [(x, y), …]); None if unset
    or unreadable."""
    kind, names = spec
    if kind == "rect":
        if isinstance(names, tuple):
            try:
                x, y, w, h = (int(values[n]) for n in names)
            except (KeyError, TypeError, ValueError):
                return None
        else:
            v = _ints(values.get(names, ""))
            if not v or len(v) != 4:
                return None
            x, y, w, h = v
        return ("rect", (x, y, w, h)) if w > 0 and h > 0 else None
    v = _ints(values.get(names, ""))
    if not v or len(v) % 2 or len(v) < 4:
        return None
    return (kind, list(zip(v[0::2], v[1::2])))


def values_from_points(spec, pts: list[tuple[float, float]], size: tuple[int, int]) -> dict | None:
    """Settings for what was drawn (points in full-resolution image pixels, clamped to ``size`` = (w, h)).
    None when the drawing is too small to mean anything (a click instead of a drag)."""
    kind, names = spec
    w, h = size
    pts = [(min(max(int(round(x)), 0), w - 1), min(max(int(round(y)), 0), h - 1)) for x, y in pts]
    if kind == "rect":
        if len(pts) < 2:
            return None
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        x, y, rw, rh = min(x0, x1), min(y0, y1), abs(x1 - x0) + 1, abs(y1 - y0) + 1
        if rw < 3 or rh < 3:
            return None
        if isinstance(names, tuple):
            return dict(zip(names, (x, y, rw, rh)))
        return {names: f"{x},{y},{rw},{rh}"}
    if kind == "line":
        if len(pts) < 2 or abs(pts[0][0] - pts[-1][0]) + abs(pts[0][1] - pts[-1][1]) < 3:
            return None
        pts = [pts[0], pts[-1]]
    elif kind == "quad":
        if len(pts) != 4:
            return None
    elif len(pts) < 3:
        return None
    return {names: ",".join(f"{x},{y}" for x, y in pts)}
