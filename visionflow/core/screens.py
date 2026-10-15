"""Designed operator screens ("Ekran Tasarımcısı"): the data model kept in the flow file under
``"operator": {"layout": "screen", "screen": {...}}``, its checking, and the small helpers the designer and the
runtime share (value formatting, colour thresholds, input validation, writing a block setting).

Nothing here imports Qt, so the flow file can be checked on a PC without a screen.

A screen::

    {"size": [1280, 800], "show_bar": true, "show_tabs": true, "rate": 15, "grid": 10,
     "pages": [{"name": "Ana sayfa", "bg": "#15161a", "bg_image": "",
                "widgets": [{"id": "w1", "type": "number", "rect": [x, y, w, h], "props": {...}}]}]}

The order of a page's widgets is their z-order (the last one is drawn on top). Bindings (``src`` / ``target``
properties) name what a widget shows or writes:

    {"node": "id" or "subflow id/inner id", "port": "ok"}      a block output (like the operator panel's refs)
    {"node": ..., "setting": "thresh"}                          a block setting
    {"var": "sayac"}                                            a shared variable ('Değişken Yaz' / 'Değişken Oku')
    {"counter": "good" | "bad" | "total" | "yield"}             the operator panel's counters
"""
from __future__ import annotations

import copy
import math
import re

SCREEN_VERSION = 1
MAX_PAGES, MAX_WIDGETS = 30, 500
SIZE_MIN, SIZE_MAX = (160, 120), (7680, 4320)
DEFAULT_SIZE = (1280, 800)
COUNTER_KEYS = ("good", "bad", "total", "yield")
COUNTER_NAMES = {"good": "Sağlam", "bad": "Hatalı", "total": "Toplam", "yield": "Verim"}
_COLOR = re.compile(r"#[0-9a-fA-F]{6}")

# property kinds: str, text, int, float, bool, color, color0 (may be empty = none), choice, source, image_source,
# target, thresholds, file
_TITLE = ("title", "str", "", "Başlık")
_FONT = ("font", "int", 0, "Yazı boyutu (0 = otomatik)", {"min": 0, "max": 400})
_LOCK = ("lock", "bool", True, "Kilitli akışta şifre iste")

WIDGETS: dict[str, dict] = {
    "image": {"title": "Canlı görüntü", "size": (480, 360), "props": [
        ("src", "image_source", None, "Görüntü (blok çıkışı)"),
        _TITLE,
        ("fit", "choice", "sığdır", "Yerleşim", {"choices": ["sığdır", "doldur", "uzat"]}),
        ("zoom", "float", 1.0, "Yakınlaştırma", {"min": 1.0, "max": 16.0}),
        ("wheel_zoom", "bool", True, "Fare tekerleğiyle yakınlaştırma"),
        ("bg", "color0", "#000000", "Arka plan"),
        ("border", "color0", "", "Çerçeve rengi")]},
    "number": {"title": "Büyük sayı", "size": (240, 140), "props": [
        ("src", "source", None, "Değer"),
        _TITLE,
        ("decimals", "int", 1, "Ondalık basamak", {"min": 0, "max": 8}),
        ("unit", "str", "", "Birim"),
        ("thousands", "bool", False, "Binlik ayırıcı"),
        ("color", "color", "#e8e8e8", "Yazı rengi"),
        ("bg", "color0", "#22252b", "Arka plan"),
        ("thresholds", "thresholds", [], "Renk eşikleri"),
        _FONT]},
    "value": {"title": "Metin değeri", "size": (300, 60), "props": [
        ("src", "source", None, "Değer"),
        ("text", "str", "{}", "Şablon ({} = değer)"),
        ("color", "color", "#e8e8e8", "Yazı rengi"),
        ("bg", "color0", "", "Arka plan"),
        ("align", "choice", "sol", "Hizalama", {"choices": ["sol", "orta", "sağ"]}),
        ("bold", "bool", False, "Kalın"),
        _FONT]},
    "text": {"title": "Sabit yazı", "size": (300, 60), "props": [
        ("text", "text", "Yazı", "Yazı"),
        ("color", "color", "#e8e8e8", "Yazı rengi"),
        ("bg", "color0", "", "Arka plan"),
        ("align", "choice", "sol", "Hizalama", {"choices": ["sol", "orta", "sağ"]}),
        ("bold", "bool", False, "Kalın"),
        _FONT]},
    "lamp": {"title": "Lamba", "size": (160, 160), "props": [
        ("src", "source", None, "Durum (doğru/yanlış)"),
        _TITLE,
        ("on_text", "str", "SAĞLAM", "Doğruyken yazı"),
        ("off_text", "str", "HATALI", "Yanlışken yazı"),
        ("on_color", "color", "#2e7d32", "Doğruyken renk"),
        ("off_color", "color", "#c62828", "Yanlışken renk"),
        ("none_color", "color", "#455a64", "Değer yokken renk"),
        ("blink", "choice", "yok", "Yanıp sönme", {"choices": ["yok", "doğruyken", "yanlışken"]}),
        ("shape", "choice", "daire", "Şekil", {"choices": ["daire", "dikdörtgen"]}),
        ("invert", "bool", False, "Ters çevir")]},
    "bar": {"title": "Çubuk / gösterge", "size": (320, 80), "props": [
        ("src", "source", None, "Değer"),
        _TITLE,
        ("style", "choice", "yatay", "Biçim", {"choices": ["yatay", "dikey", "gösterge"]}),
        ("min", "float", 0.0, "En az"),
        ("max", "float", 100.0, "En çok"),
        ("decimals", "int", 1, "Ondalık basamak", {"min": 0, "max": 8}),
        ("unit", "str", "", "Birim"),
        ("color", "color", "#42a5f5", "Çubuk rengi"),
        ("thresholds", "thresholds", [], "Renk eşikleri"),
        ("show_value", "bool", True, "Değeri yaz")]},
    "trend": {"title": "Trend grafiği", "size": (480, 240), "props": [
        ("src", "source", None, "Değer"),
        _TITLE,
        ("points", "int", 100, "Son kaç değer", {"min": 2, "max": 5000}),
        ("min", "float", 0.0, "En az (en az = en çok: otomatik)"),
        ("max", "float", 0.0, "En çok"),
        ("color", "color", "#66bb6a", "Çizgi rengi"),
        ("bg", "color0", "#1b1d22", "Arka plan"),
        ("grid", "bool", True, "Izgara"),
        ("thresholds", "thresholds", [], "Sınır çizgileri")]},
    "button": {"title": "Düğme", "size": (200, 80), "props": [
        ("text", "str", "Düğme", "Yazı"),
        ("action", "choice", "değişken", "İşlev",
         {"choices": ["değişken", "ayar", "başlat", "durdur", "sayaçları sıfırla", "reçete", "sayfa"]}),
        ("mode", "choice", "anlık", "Tür", {"choices": ["anlık", "aç/kapa"]}),
        ("target", "target", None, "Yazılacak yer (değişken / ayar)"),
        ("on_value", "str", "1", "Basılıyken / açıkken değer"),
        ("off_value", "str", "0", "Bırakınca / kapalıyken değer"),
        ("recipe", "str", "", "Reçete adı"),
        ("page", "str", "", "Sayfa (ad veya numara; + / - = sonraki / önceki)"),
        ("color", "color", "#33363c", "Renk"),
        ("active_color", "color", "#1e88e5", "Açıkken renk"),
        ("text_color", "color", "#ffffff", "Yazı rengi"),
        _LOCK,
        _FONT]},
    "input": {"title": "Sayı / metin girişi", "size": (280, 90), "props": [
        ("target", "target", None, "Ayar / değişken"),
        _TITLE,
        ("min", "float", 0.0, "En az (en az = en çok: ayarınki)"),
        ("max", "float", 0.0, "En çok"),
        ("decimals", "int", 2, "Ondalık basamak", {"min": 0, "max": 8}),
        ("unit", "str", "", "Birim"),
        ("color", "color", "#e8e8e8", "Yazı rengi"),
        ("bg", "color0", "#2a2d33", "Arka plan"),
        _LOCK,
        _FONT]},
    "recipe": {"title": "Reçete seçici", "size": (300, 90), "props": [
        ("title", "str", "Reçete", "Başlık"),
        ("color", "color", "#e8e8e8", "Yazı rengi"),
        ("bg", "color0", "#2a2d33", "Arka plan"),
        _LOCK,
        _FONT]},
    "alarms": {"title": "Alarm listesi", "size": (480, 220), "props": [
        ("title", "str", "Alarmlar", "Başlık"),
        ("rows", "int", 6, "Satır sayısı", {"min": 1, "max": 50}),
        ("show_time", "bool", True, "Saati göster"),
        ("color", "color", "#ffb74d", "Yazı rengi"),
        ("bg", "color0", "#1b1d22", "Arka plan")]},
    "counters": {"title": "Sayaç tablosu", "size": (360, 200), "props": [
        ("title", "str", "Üretim", "Başlık"),
        ("show_good", "bool", True, "Sağlam"),
        ("show_bad", "bool", True, "Hatalı"),
        ("show_total", "bool", True, "Toplam"),
        ("show_yield", "bool", True, "Verim"),
        ("color", "color", "#e8e8e8", "Yazı rengi"),
        ("bg", "color0", "#1b1d22", "Arka plan")]},
    "picture": {"title": "Sabit resim", "size": (200, 150), "props": [
        ("file", "file", "", "Resim dosyası"),
        ("fit", "choice", "sığdır", "Yerleşim", {"choices": ["sığdır", "doldur", "uzat"]})]},
    "rect": {"title": "Dikdörtgen", "size": (300, 200), "props": [
        ("fill", "color0", "#263238", "Dolgu"),
        ("border", "color0", "#546e7a", "Çerçeve"),
        ("width", "int", 2, "Çerçeve kalınlığı", {"min": 0, "max": 50}),
        ("radius", "int", 8, "Köşe yuvarlaklığı", {"min": 0, "max": 500})]},
}

# what a binding may name, per property kind
SOURCE_KINDS = {"source": ("port", "setting", "var", "counter"), "image_source": ("port",),
                "target": ("setting", "var")}


def prop_specs(wtype: str) -> list[tuple]:
    """[(name, kind, default, label, extra)] of a widget type (empty for an unknown type)."""
    out = []
    for p in WIDGETS.get(wtype, {}).get("props", []):
        out.append((*p[:4], p[4] if len(p) > 4 else {}))
    return out


def defaults(wtype: str) -> dict:
    return {name: copy.deepcopy(default) for name, _k, default, _l, _e in prop_specs(wtype)}


def new_screen() -> dict:
    return {"version": SCREEN_VERSION, "size": list(DEFAULT_SIZE), "show_bar": True, "show_tabs": True, "rate": 15,
            "grid": 10, "pages": [new_page("Ana sayfa")]}


def new_page(name: str = "Sayfa") -> dict:
    return {"name": name, "bg": "#15161a", "bg_image": "", "widgets": []}


def new_widget(wtype: str, x: float = 0, y: float = 0, wid: str = "", **props) -> dict:
    """A widget of a known type with its default properties (``props`` override them)."""
    if wtype not in WIDGETS:
        raise ValueError(f"Bilinmeyen ekran öğesi: {wtype}")
    w, h = WIDGETS[wtype]["size"]
    p = defaults(wtype)
    p.update(props)
    return {"id": wid or "w1", "type": wtype, "rect": [float(x), float(y), float(w), float(h)], "props": p}


def unique_id(used: set) -> str:
    n = len(used) + 1
    while f"w{n}" in used:
        n += 1
    return f"w{n}"


# --- bindings -------------------------------------------------------------------------------------------------------

def binding_kind(ref) -> str | None:
    """'port', 'setting', 'var', 'counter' for a well-formed binding, else None."""
    if not isinstance(ref, dict):
        return None
    if isinstance(ref.get("node"), str) and ref["node"]:
        if isinstance(ref.get("port"), str) and ref["port"] and "setting" not in ref:
            return "port"
        if isinstance(ref.get("setting"), str) and ref["setting"] and "port" not in ref:
            return "setting"
        return None
    if isinstance(ref.get("var"), str) and ref["var"].strip():
        return "var"
    if ref.get("counter") in COUNTER_KEYS:
        return "counter"
    return None


def clean_binding(ref) -> dict | None:
    kind = binding_kind(ref)
    if kind == "port":
        return {"node": ref["node"], "port": ref["port"]}
    if kind == "setting":
        return {"node": ref["node"], "setting": ref["setting"]}
    if kind == "var":
        return {"var": ref["var"].strip()[:100]}
    if kind == "counter":
        return {"counter": ref["counter"]}
    return None


def binding_label(ref, graph=None) -> str:
    """Human readable text of a binding: "Eşik (n3) → ok", "Değişken: sayac", "Sayaç: Sağlam"."""
    kind = binding_kind(ref)
    if kind is None:
        return "(seçilmedi)"
    if kind == "var":
        return f"Değişken: {ref['var']}"
    if kind == "counter":
        return f"Sayaç: {COUNTER_NAMES[ref['counter']]}"
    title = ref["node"]
    if graph is not None:
        from .graph import find_node
        node = find_node(graph, ref["node"])
        if node is not None:
            title = f"{node.title} ({ref['node']})"
    if kind == "port":
        return f"{title} → {ref['port']}"
    return f"{title} → ayar: {ref['setting']}"


def binding_problem(ref, graph, prop_kind: str = "source") -> str | None:
    """Why a binding cannot work in this flow (Turkish), or None. Without a graph only its form is checked."""
    kind = binding_kind(ref)
    if kind is None:
        return "Bağlantı bozuk"
    if kind not in SOURCE_KINDS.get(prop_kind, ()):
        return "Bu öğe bu tür bağlantıyı kullanamaz"
    if kind in ("var", "counter") or graph is None:
        return None
    from .graph import find_node
    from .node import PortType
    node = find_node(graph, ref["node"])
    if node is None:
        return f"Blok bulunamadı: {ref['node']}"
    if kind == "port":
        ports = dict(node.outputs)
        if ref["port"] not in ports:
            return f"'{node.title}' bloğunda '{ref['port']}' çıkışı yok"
        if prop_kind == "image_source" and ports[ref["port"]] not in (PortType.IMAGE, PortType.ANY):
            return f"'{node.title}' → {ref['port']} bir görüntü çıkışı değil"
        return None
    if ref["setting"] not in node.values:
        return f"'{node.title}' bloğunda '{ref['setting']}' ayarı yok"
    return None


def widget_problem(wd: dict, graph=None) -> str | None:
    """Why a widget is shown as a placeholder (Turkish), or None when it can work."""
    wtype = wd.get("type")
    if wtype not in WIDGETS:
        return f"Bilinmeyen öğe türü: {wtype}"
    props = wd.get("props", {})
    for name, kind, _d, label, _e in prop_specs(wtype):
        if kind in SOURCE_KINDS and props.get(name) is not None:
            err = binding_problem(props[name], graph, kind)
            if err:
                return f"{label}: {err}"
    return None


# --- checking a screen from a file ----------------------------------------------------------------------------------

def _num(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def clean_color(v, allow_empty=False, default="#000000") -> str:
    if isinstance(v, str) and _COLOR.fullmatch(v):
        return v.lower()
    if allow_empty and v in ("", None):
        return ""
    return default


def parse_thresholds(value) -> list[list]:
    """[[limit, "#rrggbb"], ...] sorted by limit, from a list or from text "50=#ff0000; 80=#00ff00".
    Raises ValueError (Turkish) for text it cannot read."""
    out = []
    if isinstance(value, str):
        for part in re.split(r"[;\n]+", value):
            part = part.strip()
            if not part:
                continue
            m = re.fullmatch(r"(-?[0-9]+(?:[.,][0-9]+)?)\s*[=:]\s*(#[0-9a-fA-F]{6})", part)
            if not m:
                raise ValueError(f"Eşik okunamadı: '{part}' (örnek: 50=#ff0000; 80=#00ff00)")
            out.append([float(m.group(1).replace(",", ".")), m.group(2).lower()])
    elif isinstance(value, (list, tuple)):
        for item in value[:50]:
            if (isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item[0], bool)
                    and math.isfinite(_num(item[0], math.nan)) and isinstance(item[1], str)
                    and _COLOR.fullmatch(item[1])):
                out.append([float(item[0]), item[1].lower()])
    out.sort(key=lambda t: t[0])
    return out


def thresholds_text(th) -> str:
    return "; ".join(f"{v:g}={c}" for v, c in parse_thresholds(th))


def pick_color(value, thresholds, base: str) -> str:
    """The colour of the highest threshold the value reached (base colour below all, or without a number)."""
    v = to_number(value)
    color = base
    if v is None:
        return color
    for limit, c in parse_thresholds(thresholds):
        if v >= limit:
            color = c
    return color


def _coerce_prop(kind, value, default, extra):
    if kind in ("str", "text", "file"):
        return value[:2000] if isinstance(value, str) else default
    if kind == "int":
        if isinstance(value, bool):
            return default
        v = _num(value, None)
        if v is None:
            return default
        v = int(round(v))
        return max(extra.get("min", v), min(extra.get("max", v), v))
    if kind == "float":
        if isinstance(value, bool):
            return default
        v = _num(value, None)
        if v is None:
            return default
        return float(max(extra.get("min", v), min(extra.get("max", v), v)))
    if kind == "bool":
        return value if isinstance(value, bool) else default
    if kind in ("color", "color0"):
        return clean_color(value, kind == "color0", default)
    if kind == "choice":
        return value if value in extra.get("choices", ()) else default
    if kind == "thresholds":
        try:
            return parse_thresholds(value)
        except ValueError:
            return []
    return default


def clean_widget(wd, graph=None, warnings: list | None = None, used: set | None = None, where: str = "") -> dict | None:
    """One widget from a file, checked. Broken entries give None; an unknown type (a newer version) is kept as it is
    (shown as a placeholder); bindings to blocks that are not in the flow are kept with a warning (placeholder)."""
    warn = warnings.append if warnings is not None else (lambda _m: None)
    used = used if used is not None else set()
    if not isinstance(wd, dict) or not isinstance(wd.get("type"), str) or not wd["type"]:
        warn(f"Ekran tasarımı{where}: bozuk öğe atlandı")
        return None
    wtype = wd["type"][:60]
    size = WIDGETS.get(wtype, {}).get("size", (200, 100))
    try:
        x, y, w, h = (float(v) for v in wd["rect"])
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        x, y, (w, h) = 0.0, 0.0, size
    rect = [max(-20000.0, min(20000.0, x)), max(-20000.0, min(20000.0, y)),
            max(4.0, min(20000.0, w)), max(4.0, min(20000.0, h))]
    wid = wd.get("id") if isinstance(wd.get("id"), str) and wd.get("id") else ""
    if not wid or wid in used or len(wid) > 40:
        wid = unique_id(used)
    used.add(wid)
    raw = wd.get("props") if isinstance(wd.get("props"), dict) else {}
    if wtype not in WIDGETS:
        warn(f"Ekran tasarımı{where}: bilinmeyen öğe türü '{wtype}' (yer tutucu gösterilecek)")
        return {"id": wid, "type": wtype, "rect": rect, "props": copy.deepcopy(raw)}
    props = {}
    for name, kind, default, label, extra in prop_specs(wtype):
        if name not in raw:
            props[name] = copy.deepcopy(default)
        elif kind in SOURCE_KINDS:
            ref = raw[name]
            if ref in (None, "", {}):
                props[name] = None
                continue
            ref = clean_binding(ref)
            if ref is None or binding_kind(ref) not in SOURCE_KINDS[kind]:
                warn(f"Ekran tasarımı{where}: '{WIDGETS[wtype]['title']}' öğesinin bağlantısı bozuk, kaldırıldı")
                props[name] = None
                continue
            err = binding_problem(ref, graph, kind)
            if err:
                warn(f"Ekran tasarımı{where}: '{WIDGETS[wtype]['title']}' öğesi: {err} (yer tutucu gösterilecek)")
            props[name] = ref
        else:
            props[name] = _coerce_prop(kind, raw[name], copy.deepcopy(default), extra)
    return {"id": wid, "type": wtype, "rect": rect, "props": props}


def clean_page(pd, graph=None, warnings: list | None = None, index: int = 0) -> dict | None:
    warn = warnings.append if warnings is not None else (lambda _m: None)
    if not isinstance(pd, dict):
        warn(f"Ekran tasarımı: bozuk sayfa atlandı (sıra {index + 1})")
        return None
    name = str(pd.get("name") or f"Sayfa {index + 1}")[:60]
    page = {"name": name, "bg": clean_color(pd.get("bg"), False, "#15161a"),
            "bg_image": pd.get("bg_image")[:2000] if isinstance(pd.get("bg_image"), str) else "", "widgets": []}
    used: set = set()
    widgets = pd.get("widgets") if isinstance(pd.get("widgets"), list) else []
    if len(widgets) > MAX_WIDGETS:
        warn(f"Ekran tasarımı ({name}): {len(widgets) - MAX_WIDGETS} öğe sınırı aştığı için atlandı")
    for wd in widgets[:MAX_WIDGETS]:
        w = clean_widget(wd, graph, warnings, used, f" ({name})")
        if w is not None:
            page["widgets"].append(w)
    return page


def clean_screen(data, graph=None, warnings: list | None = None) -> dict | None:
    """A designed screen from a flow file, checked against the flow's blocks (None when it is not a screen at all)."""
    warn = warnings.append if warnings is not None else (lambda _m: None)
    if not isinstance(data, dict):
        warn("Bozuk ekran tasarımı atlandı")
        return None
    out = new_screen()
    size = data.get("size")
    try:
        w, h = int(float(size[0])), int(float(size[1]))
    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
        w, h = DEFAULT_SIZE
    out["size"] = [max(SIZE_MIN[0], min(SIZE_MAX[0], w)), max(SIZE_MIN[1], min(SIZE_MAX[1], h))]
    for key in ("show_bar", "show_tabs"):
        if isinstance(data.get(key), bool):
            out[key] = data[key]
    out["rate"] = int(max(1, min(30, _num(data.get("rate"), 15))))
    out["grid"] = int(max(1, min(200, _num(data.get("grid"), 10))))
    pages = data.get("pages") if isinstance(data.get("pages"), list) else []
    out["pages"] = []
    for i, pd in enumerate(pages[:MAX_PAGES]):
        page = clean_page(pd, graph, warnings, i)
        if page is not None:
            out["pages"].append(page)
    if not out["pages"]:
        out["pages"] = [new_page("Ana sayfa")]
    return out


def remap_screen(screen: dict, remap) -> dict:
    """A copy of the screen with the block ids of its bindings rewritten (grouping blocks into a sub-flow and back):
    ``remap(id, port) -> new id | {"node", "port"} | None`` (None: the block is gone, the binding is removed)."""
    screen = copy.deepcopy(screen)
    for page in screen.get("pages", []) if isinstance(screen.get("pages"), list) else []:
        for wd in page.get("widgets", []) if isinstance(page, dict) and isinstance(page.get("widgets"), list) else []:
            props = wd.get("props") if isinstance(wd, dict) else None
            if not isinstance(props, dict):
                continue
            for name, ref in list(props.items()):
                kind = binding_kind(ref)
                if kind not in ("port", "setting"):
                    continue
                new = remap(ref["node"], ref["port"] if kind == "port" else "")
                if isinstance(new, dict) and kind == "port":
                    props[name] = {"node": new.get("node"), "port": new.get("port")}
                elif isinstance(new, str) and new:
                    props[name] = {**ref, "node": new}
                else:
                    props[name] = None
    return screen


def refs(screen: dict) -> list[tuple[dict, str, dict]]:
    """(widget, property name, binding) of every binding on every page."""
    out = []
    for page in (screen or {}).get("pages", []):
        for wd in page.get("widgets", []):
            for name, kind, _d, _l, _e in prop_specs(wd.get("type")):
                ref = wd.get("props", {}).get(name) if kind in SOURCE_KINDS else None
                if binding_kind(ref):
                    out.append((wd, name, ref))
    return out


# --- values ---------------------------------------------------------------------------------------------------------

def to_number(v) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        v = v.strip().replace(",", ".")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def to_bool(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "evet", "on", "açık", "sağlam", "ok"):
            return True
        if s in ("0", "false", "hayır", "hayir", "off", "kapalı", "", "none"):
            return False
        return None
    n = to_number(v)
    return None if n is None else n != 0


def format_number(v, decimals: int = 1, unit: str = "", thousands: bool = False) -> str:
    """'12.3 mm', '1.234,5' style text of a value; text stays text; nothing → '–'."""
    if v is None:
        return "–"
    n = to_number(v) if not isinstance(v, str) else None
    if n is None:
        if isinstance(v, str):
            n = to_number(v)
        if n is None:
            return f"{v}" + (f" {unit}" if unit else "")
    text = f"{n:,.{decimals}f}" if thousands else f"{n:.{decimals}f}"
    if thousands:          # Turkish style: 1.234,5
        text = text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return text + (f" {unit}" if unit else "")


def format_text(template: str, v) -> str:
    shown = "–" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
    template = template or "{}"
    return template.replace("{}", shown) if "{}" in template else f"{template} {shown}"


def parse_input(text, param=None, lo: float = 0.0, hi: float = 0.0):
    """What an operator typed for a block setting (``param``, a core.node.Param) or a shared variable (None),
    checked: numbers within lo…hi (when lo < hi, else the setting's own range). Raises ValueError (Turkish)."""
    text = "" if text is None else str(text).strip()
    if param is None:                         # a shared variable: number when it looks like one, else text
        n = to_number(text)
        if n is not None and lo < hi and not lo <= n <= hi:
            raise ValueError(f"Değer {lo:g} … {hi:g} aralığında olmalı")
        if n is None:
            return text
        return int(n) if n.is_integer() and re.fullmatch(r"-?\d+", text) else n
    kind = param.kind
    if kind in ("int", "float"):
        n = to_number(text)
        if n is None:
            raise ValueError("Sayı girin")
        a, b = (lo, hi) if lo < hi else (param.min, param.max)
        if a is not None and n < a or b is not None and n > b:
            lo_t = "-∞" if a is None else f"{a:g}"
            hi_t = "∞" if b is None else f"{b:g}"
            raise ValueError(f"Değer {lo_t} … {hi_t} aralığında olmalı")
        if kind == "int":
            if not float(n).is_integer():
                raise ValueError("Tam sayı girin")
            return int(n)
        return float(n)
    if kind == "bool":
        b = to_bool(text)
        if b is None:
            raise ValueError("Evet / hayır (1 / 0) girin")
        return b
    if kind == "choice":
        if text not in param.choices:
            raise ValueError(f"Seçenekler: {', '.join(param.choices)}")
        return text
    if kind == "code":
        raise ValueError("Kod ayarları operatör ekranından değiştirilemez")
    if len(text) > 1000:
        raise ValueError("Metin çok uzun (en çok 1000 karakter)")
    return text


def write_setting(graph, key: str, name: str, value):
    """Set setting ``name`` of block ``key`` ("id" or "subflow id/inner id") in a flow that is not running (the
    panel's copy, kept for its next start). Returns the value as stored; ValueError (Turkish) when it is not there."""
    from .graph import find_node
    from .recipes import coerce_value
    node = find_node(graph, key)
    if node is None or name not in node.values:
        raise ValueError(f"Ayar bulunamadı: {key} → {name}")
    v = coerce_value(node, name, value)
    if "/" not in key:
        if node.type_name == "SubFlow" and "/" in name:
            node.set_exposed(name, v)
        else:
            node.values[name] = v
        return v
    parts = key.split("/")
    top = graph.nodes.get(parts[0])
    if top is None or top.type_name != "SubFlow":
        raise ValueError(f"Ayar bulunamadı: {key} → {name}")
    rest = "/".join(parts[1:] + [name])
    if rest in top.values:
        top.set_exposed(rest, v)
    else:
        from ..nodes.subflow import _set_in
        _set_in(top.inner, rest, v)
    return v


def flow_variables(graph) -> list[str]:
    """Names of the shared variables the flow's 'Değişken Yaz' / 'Değişken Oku' blocks use."""
    from .graph import walk_nodes
    names = []
    if graph is None:
        return names
    for _key, _label, node in walk_nodes(graph):
        if node.type_name in ("SetVariable", "GetVariable"):
            n = str(node.values.get("name", "")).strip() or "degisken"
            if n not in names:
                names.append(n)
    return names
