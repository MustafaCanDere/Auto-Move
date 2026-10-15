"""Node base class, port types and parameter definitions."""
from __future__ import annotations

import pathlib
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any


def where(exc: BaseException) -> str:
    """'visionflow/nodes/more.py:152 process ← ...': the innermost lines of the program's own code (or of a plugin)
    that raised, for the self-improvement report. User scripts ("<Python Kodu>") and library code are skipped."""
    import traceback
    root = pathlib.Path(__file__).resolve().parents[2]          # folder that holds visionflow/
    out = []
    for fr in reversed(traceback.extract_tb(exc.__traceback__)):
        name = fr.filename or ""
        if not name or name.startswith("<"):
            continue
        path = pathlib.Path(name)
        if path.parent.name == "plugins":
            rel = f"plugins/{path.name}"
        else:
            try:
                rel = path.resolve().relative_to(root).as_posix()
            except (ValueError, OSError):
                continue                                          # numpy / OpenCV internals: not ours to fix
            if not rel.startswith("visionflow/"):
                continue
        out.append(f"{rel}:{fr.lineno} {fr.name}")
        if len(out) == 3:
            break
    return " ← ".join(out)


# colour labels a block can carry in the editor (right-click → Blok rengi); saved as an optional "color" key
BLOCK_COLORS = {
    "Kırmızı": "#e53935", "Turuncu": "#fb8c00", "Sarı": "#fdd835", "Yeşil": "#43a047",
    "Mavi": "#1e88e5", "Mor": "#8e24aa", "Pembe": "#d81b60", "Gri": "#757575",
}


def clean_color(value) -> str | None:
    """A saved block colour: '#rrggbb' (lower case) or None for anything else."""
    if isinstance(value, str) and len(value) == 7 and value[0] == "#" and \
            all(c in "0123456789abcdefABCDEF" for c in value[1:]):
        return value.lower()
    return None


class PortType:
    IMAGE = "image"
    NUMBER = "number"
    BOOL = "bool"
    TEXT = "text"
    DETECTIONS = "detections"
    CONTOURS = "contours"
    ANY = "any"

    COLORS = {
        IMAGE: "#4fc3f7",
        NUMBER: "#aed581",
        BOOL: "#ffb74d",
        TEXT: "#e0e0e0",
        DETECTIONS: "#f06292",
        CONTOURS: "#ba68c8",
        ANY: "#bdbdbd",
    }

    @staticmethod
    def compatible(src: str, dst: str) -> bool:
        if src == dst or PortType.ANY in (src, dst):
            return True
        # numbers and bools convert freely
        return {src, dst} == {PortType.NUMBER, PortType.BOOL}


@dataclass
class Param:
    """A user-editable node setting. kind: int | float | bool | str | choice | path | folder | code"""
    name: str
    kind: str
    default: Any
    label: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] = field(default_factory=list)

    @property
    def display(self) -> str:
        return self.label or self.name

    def coerce(self, value, clamp: bool = False):
        """Value converted to this setting's type ("640" → 640, "false" → False). Out-of-range numbers raise,
        or are pulled into range with ``clamp``; an unknown choice raises. ValueError/TypeError on failure."""
        kind = self.kind
        if kind == "int":
            if isinstance(value, bool):
                raise TypeError("sayı bekleniyor")
            value = int(round(float(value)))
        elif kind == "float":
            if isinstance(value, bool):
                raise TypeError("sayı bekleniyor")
            value = float(value)
            if value != value:
                raise ValueError("sayı değil (NaN)")
        elif kind == "bool":
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            if s in ("1", "true", "evet", "yes", "on"):
                return True
            if s in ("0", "false", "hayır", "hayir", "no", "off", "", "none"):
                return False
            raise ValueError("doğru/yanlış bekleniyor")
        elif kind == "choice":
            value = str(value)
            if value not in self.choices:
                raise ValueError(f"seçenekler: {', '.join(self.choices)}")
            return value
        else:
            if isinstance(value, (dict, list)):
                raise TypeError("metin bekleniyor")
            return "" if value is None else str(value)
        lo, hi = self.min, self.max
        if clamp:
            value = max(lo, value) if lo is not None else value
            value = min(hi, value) if hi is not None else value
            return type(value)(value) if kind == "float" else int(value)
        if lo is not None and value < lo or hi is not None and value > hi:
            raise ValueError(f"aralık {lo if lo is not None else '-∞'}…{hi if hi is not None else '∞'}")
        return value


# where relative file settings ("examples/ornek_veri", "models/yolov8n.onnx") are looked for when they do not exist
# from the current folder: the folder of the open flow, then the program folder (works for the packaged .exe and
# for shortcuts / file associations that start the program from another folder)
FLOW_DIR: pathlib.Path | None = None
FLOW_FILE: pathlib.Path | None = None    # the open flow file itself (its name is the flow's name in archives)


def search_dirs() -> list[pathlib.Path]:
    dirs = [FLOW_DIR] if FLOW_DIR else []
    if getattr(sys, "frozen", False):
        dirs += [pathlib.Path(sys.executable).parent, pathlib.Path(getattr(sys, "_MEIPASS", "."))]
    dirs.append(pathlib.Path(__file__).resolve().parents[2])
    return dirs


def resolve_path(value: str) -> str:
    """A relative path that is missing from the current folder but exists in one of search_dirs() → that one.
    Anything else (absolute, already there, or an output that does not exist yet) is returned unchanged."""
    if not value or not isinstance(value, str):
        return value
    p = pathlib.Path(value)
    if p.is_absolute() or p.exists():
        return value
    for d in search_dirs():
        cand = d / p
        if cand.exists():
            return str(cand)
    return value


def save_target(value: str, ext: str = "") -> tuple[str, str]:
    """A file a window writes for a block's path setting (a trained model): (where to write it, what the setting
    keeps). A relative path is written where resolve_path() will find it again — the existing file it already
    names, else next to the open flow file (the setting stays relative, the flow can be moved with its folder) —
    and not into whatever folder the program was started from. Without a flow file it becomes absolute."""
    value = str(value or "").strip()
    if ext and not value.lower().endswith(ext):
        value += ext
    p = pathlib.Path(value)
    if not value or p.is_absolute():
        return value, value
    found = resolve_path(value)
    if found != value or p.exists():                 # the file it names now: overwrite that one
        return str(pathlib.Path(found).resolve()), value
    if FLOW_DIR:
        return str(pathlib.Path(FLOW_DIR) / p), value
    target = str(p.resolve())
    return target, target


class StopFlow(Exception):
    """Raised by a source node when it has no more data (end of video etc.)."""


class NoNewFrame(Exception):
    """Raised by a source that has nothing new this time (e.g. a camera waiting for its hardware trigger): it and
    the blocks after it are skipped for this frame, without an error and without running on an old picture."""


class Node:
    """Base class of every block.

    Subclasses declare ``inputs``/``outputs`` as (name, PortType) tuples and
    ``params`` as Param objects, then implement ``process``.
    """

    type_name: str = ""
    title: str = ""
    category: str = "Diğer"
    description: str = ""
    inputs: list[tuple[str, str]] = []
    outputs: list[tuple[str, str]] = []
    params: list[Param] = []
    is_source = False   # drives the frame loop (camera, video)
    # inputs that may be wired from a later block ("done" back to a source's "next"); they read the value
    # from the previous frame, so such a link does not count as a loop
    feedback_inputs: tuple[str, ...] = ()
    # settings that are read in setup(); the editor restarts the block when one of them changes during a run
    setup_params: tuple[str, ...] = ()
    # a region that can be drawn with the mouse on the preview of the block's input image:
    # ("rect", "roi") one "x,y,w,h" text setting, ("rect", ("x", "y", "w", "h")) four number settings,
    # ("line", "line") "x1,y1,x2,y2", ("poly", "zone") "x1,y1,x2,y2,…", ("quad", "points") exactly 4 corners
    roi: tuple | None = None

    def __init__(self, node_id: str | None = None):
        self.id = node_id or uuid.uuid4().hex[:8]
        self.values: dict[str, Any] = {p.name: p.default for p in self.params}
        self.pos: tuple[float, float] = (0.0, 0.0)
        self.state: dict[str, Any] = {}
        self.enabled = True   # disabled blocks pass their inputs straight through
        self.color: str | None = None   # editor colour label ('#rrggbb'), see BLOCK_COLORS
        self.setup_error: str | None = None
        self.setup_where = ""

    # lifecycle -----------------------------------------------------------
    def setup(self) -> None:
        """Called once before a run starts (open camera, load model...)."""

    def safe_setup(self) -> bool:
        """setup() that records a failure on the block instead of aborting the whole flow."""
        self.state = {}
        self.setup_error = None
        self.setup_where = ""
        if not self.enabled:
            return True
        try:
            self.setup()
            return True
        except Exception as exc:  # noqa: BLE001
            self.setup_error = f"Başlatılamadı: {exc}"
            self.setup_where = where(exc)
            return False

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def teardown(self) -> None:
        """Called once after a run ends."""

    # helpers ---------------------------------------------------------------
    def v(self, name: str) -> Any:
        value = self.values[name]
        kinds = type(self).__dict__.get("_kinds")
        if kinds is None:
            kinds = {p.name: p.kind for p in self.params}
            type(self)._kinds = kinds
        if kinds.get(name) in ("path", "folder"):
            return resolve_path(value)
        return value

    def wants_frames(self) -> bool:
        """True while the block waits for background work whose result appears on a later frame. A flow
        without a camera/timer keeps running until this is False everywhere."""
        return False

    def param(self, name: str) -> Param:
        return next(p for p in self.params if p.name == name)

    def input_type(self, name: str) -> str:
        return dict(self.inputs)[name]

    def output_type(self, name: str) -> str:
        return dict(self.outputs)[name]

    def bypass(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Outputs of a disabled block: each output takes the same-named, else first same-typed, input."""
        out = {}
        for name, t in self.outputs:
            if name in inputs and PortType.compatible(dict(self.inputs)[name], t):
                out[name] = inputs[name]
                continue
            out[name] = next((inputs[n] for n, it in self.inputs if it == t and inputs.get(n) is not None), None)
        return out

    def extra(self) -> dict:
        """More of the block to save besides its settings (e.g. the blocks inside a sub-flow)."""
        return {}

    def load_extra(self, data: dict, warnings: list[str]) -> None:
        """Read back what extra() saved; called after the settings are set, before any connection is made."""

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type_name, "pos": list(self.pos), "values": dict(self.values)}
        if not self.enabled:
            d["disabled"] = True
        if self.color:
            d["color"] = self.color
        d.update(self.extra())
        return d
