"""Labelled image datasets ("Veri Seti Yöneticisi"): YOLO folders, statistics and problem finding, class
rename / merge / delete (with a backup), stratified train/val/test split, COCO / Pascal VOC / YOLO conversion,
box-aware augmentation and model-assisted labelling with suggestions the user accepts or rejects.

No Qt here: the window (ui/dataset_manager.py) and the command line (``python -m visionflow veriseti ...``) both use
this module.

Layouts understood (``Dataset(root)``):

* ``root/images/**`` + ``root/labels/**`` (YOLO / Ultralytics, also with ``images/train`` ... sub folders),
* ``root/train/images`` + ``root/train/labels`` (Roboflow export: train, valid/val, test),
* a flat folder with the pictures and their ``.txt`` labels side by side.

Class names come from ``classes.txt`` or ``data.yaml``. Model suggestions are kept apart from the real labels in
``root/oneriler/<image path>.txt`` (YOLO lines + a confidence column) until the user accepts them, so a training run
never sees an unchecked suggestion.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import re
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                 "test": "test"}
SUGGEST_DIR = "oneriler"
BACKUP_DIR = ".vf_yedek"
CACHE_FILE = ".vf_veriseti.json"
MODEL_DIR = "modeller"
PROBLEM_NAMES = {"okunamadi": "Resim okunamıyor", "kucuk": "Çok küçük kutu", "disarida": "Kutu resmin dışında",
                 "bozuk": "Bozuk etiket satırı", "sinif": "Tanımsız sınıf numarası", "kopya": "Kopya resim"}
AUG_OPS = ("flip_h", "flip_v", "rot90", "rotate", "brightness", "blur", "noise")
AUG_NAMES = {"flip_h": "Yatay çevir", "flip_v": "Dikey çevir", "rot90": "90° döndür", "rotate": "Küçük açı döndür",
             "brightness": "Parlaklık", "blur": "Bulanıklık", "noise": "Gürültü"}


# --- small helpers --------------------------------------------------------------------------------------------------

def imread(path) -> np.ndarray | None:
    """BGR uint8 picture (unicode paths too) or None."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if not data.size:
        return None
    try:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def imwrite(path, img) -> bool:
    path = pathlib.Path(path)
    ext = path.suffix.lower() if path.suffix.lower() in IMAGE_EXT else ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
    return bool(ok)


def _write_text(path: pathlib.Path, text: str) -> None:
    """Atomic write: a crash never leaves a half-written label file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def clean_name(name) -> str:
    """A class name on one line (classes.txt has one name per line)."""
    return " ".join(str(name).split())


def yaml_names(path: pathlib.Path) -> list[str]:
    """Class names from an Ultralytics/Roboflow data.yaml ("names: [a, b]" or "names:\n  0: a"), [] if none."""
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        import yaml
        names = (yaml.safe_load(text) or {}).get("names")
    except ImportError:
        names = None
        m = re.search(r"(?m)^names:\s*\[(.*)\]\s*$", text)
        if m:
            names = [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
        else:
            m = re.search(r"(?ms)^names:\s*\n((?:[ \t]+\d+\s*:.*\n?)+)", text)
            if m:
                pairs = re.findall(r"(\d+)\s*:\s*(.+)", m.group(1))
                names = {int(k): v.strip().strip("'\"") for k, v in pairs}
    except Exception:  # noqa: BLE001 - a broken data.yaml just means no names
        return []
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    return [str(n) for n in names] if isinstance(names, list) else []


def _yaml_value(text: str, key: str) -> str | None:
    m = re.search(rf"(?m)^{key}:\s*(.+?)\s*$", text)
    if not m:
        return None
    v = m.group(1)
    try:
        return str(json.loads(v)) if v.startswith('"') else v.strip("'")
    except ValueError:
        return v.strip('"')


def write_data_yaml(root, classes: list[str], images: str = "images", splits: dict | None = None,
                    keep_splits: bool = True) -> pathlib.Path:
    """data.yaml for Ultralytics. ``splits`` = {"train": "train.txt", ...}; without it the train/val/test entries an
    earlier split wrote are kept (``keep_splits``), else train = val = ``images``."""
    root = pathlib.Path(root)
    path = root / "data.yaml"
    if splits is None and keep_splits and path.exists():
        try:
            old = path.read_text(encoding="utf-8")
        except OSError:
            old = ""
        found = {k: _yaml_value(old, k) for k in SPLITS}
        if found["train"] and found["val"] and (found["train"] != images or found["val"] != images
                                                 or found["test"]):
            splits = {k: v for k, v in found.items() if v}
    if not splits:
        splits = {"train": images, "val": images}
    q = lambda v: json.dumps(str(v), ensure_ascii=False)  # noqa: E731
    lines = [f"path: {q(root.resolve().as_posix())}"]
    for k in SPLITS:
        if splits.get(k):
            lines.append(f"{k}: {q(splits[k])}")
    lines.append("names:")
    lines += [f"  {i}: {q(n)}" for i, n in enumerate(classes)]
    _write_text(path, "\n".join(lines) + "\n")
    return path


def box_iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_xyxy(box, w: float, h: float) -> list:
    """YOLO (c, cx, cy, bw, bh[, conf]) normalised → [c, x1, y1, x2, y2] pixels."""
    c, cx, cy, bw, bh = box[:5]
    return [int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


def to_yolo(box, w: float, h: float) -> tuple:
    """[c, x1, y1, x2, y2] pixels → (c, cx, cy, bw, bh) normalised (no clipping)."""
    c, x1, y1, x2, y2 = box[:5]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return int(c), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


def format_line(box, conf: float | None = None) -> str:
    c, cx, cy, bw, bh = box[:5]
    s = f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    return s if conf is None else f"{s} {conf:.4f}"


def parse_labels(text: str) -> tuple[list[tuple], list[str], list[str]]:
    """YOLO label text → (boxes (c, cx, cy, w, h, conf|None), other valid lines (polygons), broken lines)."""
    boxes, other, bad = [], [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            c = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            bad.append(line)
            continue
        if c < 0 or not all(math.isfinite(v) for v in vals):
            bad.append(line)
        elif len(vals) in (4, 5):
            if vals[2] <= 0 or vals[3] <= 0:
                bad.append(line)
            else:
                boxes.append((c, *vals[:4], vals[4] if len(vals) == 5 else None))
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            other.append(line)                     # segmentation polygon: kept as it is
        else:
            bad.append(line)
    return boxes, other, bad


def phash(img: np.ndarray) -> int:
    """64-bit perceptual hash (DCT of a 32x32 grey picture): near-identical pictures differ in a few bits."""
    g = img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def duplicate_groups(hashes: dict, max_dist: int = 4) -> list[list]:
    """Groups of keys whose hashes differ in at most ``max_dist`` bits (sorted, each group ≥ 2)."""
    keys = sorted(hashes)
    if len(keys) < 2:
        return []
    arr = np.array([[(hashes[k] >> (8 * i)) & 0xFF for i in range(8)] for k in keys], np.uint8)
    pop = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(keys) - 1):
        dist = pop[arr[i + 1:] ^ arr[i]].sum(axis=1)
        for j in np.nonzero(dist <= max_dist)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    groups: dict[int, list] = {}
    for i, k in enumerate(keys):
        groups.setdefault(find(i), []).append(k)
    return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: g[0])


# --- the dataset ----------------------------------------------------------------------------------------------------

class Item:
    """One picture of the dataset."""
    __slots__ = ("image", "label", "rel", "split")

    def __init__(self, image: pathlib.Path, label: pathlib.Path, rel: str, split: str = ""):
        self.image, self.label, self.rel, self.split = image, label, rel, split

    def __repr__(self):
        return f"Item({self.rel!r}, {self.split!r})"


def _images_under(folder: pathlib.Path, recursive: bool) -> list[pathlib.Path]:
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.suffix.lower() in IMAGE_EXT and p.is_file()
                  and not any(part.startswith(".") for part in p.relative_to(folder).parts))


class Dataset:
    """A labelled picture folder (see the module help for the layouts)."""

    def __init__(self, root, create: bool = False):
        root = pathlib.Path(root)
        if create:
            (root / "images").mkdir(parents=True, exist_ok=True)
            (root / "labels").mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ValueError(f"Veri seti klasörü yok: {root}")
        self.root = root
        self.classes: list[str] = []
        self.items: list[Item] = []
        self.layout = "flat"
        self._cache: dict | None = None
        self.reload()
        if create and not (root / "data.yaml").exists():
            self.save_meta()

    # ---- discovery ----
    def reload(self) -> None:
        root = self.root
        items: list[Item] = []
        if (root / "images").is_dir():
            self.layout = "yolo"
            base = root / "images"
            for p in _images_under(base, True):
                rel = p.relative_to(base)
                split = SPLIT_ALIASES.get(rel.parts[0].lower(), "") if len(rel.parts) > 1 else ""
                items.append(Item(p, (root / "labels" / rel).with_suffix(".txt"), p.relative_to(root).as_posix(),
                                  split))
        else:
            subs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.lower() in SPLIT_ALIASES
                    and (d / "images").is_dir()] if root.is_dir() else []
            if subs:
                self.layout = "roboflow"
                for d in subs:
                    for p in _images_under(d / "images", True):
                        rel = p.relative_to(d / "images")
                        items.append(Item(p, (d / "labels" / rel).with_suffix(".txt"),
                                          p.relative_to(root).as_posix(), SPLIT_ALIASES[d.name.lower()]))
            else:
                self.layout = "flat"
                for p in _images_under(root, False):
                    items.append(Item(p, p.with_suffix(".txt"), p.name, ""))
        self.items = items
        # split lists written by split(): they win over the folder names
        lists = {s: root / f"{s}.txt" for s in SPLITS}
        if any(f.exists() for f in lists.values()):
            where = {}
            for s, f in lists.items():
                if f.exists():
                    for line in f.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line:
                            where[self._key(line)] = s
            for it in self.items:
                it.split = where.get(it.rel, it.split)
        cls_file = root / "classes.txt"
        if cls_file.exists():
            self.classes = [clean_name(c) for c in cls_file.read_text(encoding="utf-8").splitlines() if c.strip()]
        else:
            self.classes = [clean_name(c) for c in yaml_names(root / "data.yaml")]

    def _key(self, line: str) -> str:
        p = pathlib.Path(line[2:] if line.startswith("./") else line)
        if p.is_absolute():
            try:
                p = p.relative_to(self.root)
            except ValueError:
                pass
        return p.as_posix()

    @property
    def images_rel(self) -> str:
        return "images" if self.layout == "yolo" else "."

    def find(self, rel_or_path) -> Item | None:
        s = str(rel_or_path)
        for test in (lambda it: it.rel == s, lambda it: str(it.image) == s, lambda it: it.image.name == s):
            for it in self.items:                 # exact path first: the same name may be in train and val
                if test(it):
                    return it
        return None

    def save_meta(self, splits: dict | None = None) -> None:
        _write_text(self.root / "classes.txt", "\n".join(self.classes))
        if self.layout != "roboflow" or splits:
            write_data_yaml(self.root, self.classes, self.images_rel, splits)

    def class_index(self, name, add: bool = True) -> int:
        """Index of a class name (case-insensitive); a new name is added (``add``) or -1."""
        name = clean_name(name) or "nesne"
        low = [c.casefold() for c in self.classes]
        if name.casefold() in low:
            return low.index(name.casefold())
        if not add:
            return -1
        self.classes.append(name)
        self.save_meta()
        return len(self.classes) - 1

    # ---- labels ----
    def read(self, item: Item) -> tuple[list[tuple], list[str], list[str]]:
        if not item.label.exists():
            return [], [], []
        try:
            return parse_labels(item.label.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return [], [], []

    def boxes(self, item: Item) -> list[tuple]:
        return [b[:5] for b in self.read(item)[0]]

    def write(self, item: Item, boxes, other=()) -> None:
        _write_text(item.label, "\n".join([format_line(b) for b in boxes] + list(other)))

    def is_labelled(self, item: Item) -> bool:
        return item.label.exists()

    # ---- image facts (cached by file size + time) ----
    def _load_cache(self) -> dict:
        if self._cache is None:
            try:
                self._cache = json.loads((self.root / CACHE_FILE).read_text(encoding="utf-8"))
                if not isinstance(self._cache, dict):
                    self._cache = {}
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    def save_cache(self) -> None:
        if self._cache is not None:
            try:
                (self.root / CACHE_FILE).write_text(json.dumps(self._cache), encoding="utf-8")
            except OSError:
                pass

    def info(self, item: Item) -> tuple[int, int, int] | None:
        """(width, height, perceptual hash) or None for an unreadable picture."""
        cache = self._load_cache()
        try:
            st = item.image.stat()
        except OSError:
            return None
        stamp = [st.st_mtime_ns, st.st_size]
        hit = cache.get(item.rel)
        if isinstance(hit, list) and len(hit) == 5 and hit[:2] == stamp:
            return None if hit[2] is None else (int(hit[2]), int(hit[3]), int(hit[4], 16))
        img = imread(item.image)
        if img is None:
            cache[item.rel] = stamp + [None, None, None]
            return None
        h, w = img.shape[:2]
        ph = phash(img)
        cache[item.rel] = stamp + [w, h, f"{ph:016x}"]
        return w, h, ph

    def size(self, item: Item) -> tuple[int, int] | None:
        i = self.info(item)
        return None if i is None else (i[0], i[1])

    # ---- statistics and problems ----
    def stats(self, tiny_px: float = 4.0, dup_dist: int = 4, progress=None) -> dict:
        """Counts, class balance, picture sizes and problems. ``progress(i, n)`` may return False to stop."""
        n = len(self.items)
        per_class = [0] * len(self.classes)
        img_per_class = [0] * len(self.classes)
        unknown: dict[int, int] = {}
        sizes: dict[str, int] = {}
        problems: list[dict] = []
        hashes: dict[str, int] = {}
        labelled = empty = boxes_total = 0
        widths, heights = [], []
        for i, it in enumerate(self.items):
            if progress is not None and progress(i, n) is False:
                break
            inf = self.info(it)
            if inf is None:
                problems.append({"image": it.rel, "kind": "okunamadi", "detail": "resim açılamadı"})
                w = h = None
            else:
                w, h, hashes[it.rel] = inf
                sizes[f"{w}x{h}"] = sizes.get(f"{w}x{h}", 0) + 1
                widths.append(w)
                heights.append(h)
            if not it.label.exists():
                continue
            labelled += 1
            bx, _other, bad = self.read(it)
            if not bx and not _other:
                empty += 1
            for line in bad:
                problems.append({"image": it.rel, "kind": "bozuk", "detail": line[:80]})
            seen = set()
            for k, (c, cx, cy, bw, bh, _conf) in enumerate(bx):
                boxes_total += 1
                if c < len(self.classes):
                    per_class[c] += 1
                    seen.add(c)
                else:
                    unknown[c] = unknown.get(c, 0) + 1
                    problems.append({"image": it.rel, "kind": "sinif", "box": k,
                                     "detail": f"sınıf {c} tanımlı değil ({len(self.classes)} sınıf var)"})
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                eps = 1e-3
                if x1 < -eps or y1 < -eps or x2 > 1 + eps or y2 > 1 + eps:
                    inside = max(0.0, min(x2, 1) - max(x1, 0)) * max(0.0, min(y2, 1) - max(y1, 0)) / (bw * bh)
                    problems.append({"image": it.rel, "kind": "disarida", "box": k,
                                     "detail": f"kutunun %{100 * (1 - inside):.0f} kadarı resmin dışında"})
                if w is not None and (bw * w < tiny_px or bh * h < tiny_px):
                    problems.append({"image": it.rel, "kind": "kucuk", "box": k,
                                     "detail": f"{bw * w:.1f}x{bh * h:.1f} piksel"})
            for c in seen:
                img_per_class[c] += 1
        self.save_cache()
        dups = duplicate_groups(hashes, dup_dist)
        for g in dups:
            for rel in g[1:]:
                problems.append({"image": rel, "kind": "kopya", "detail": f"{g[0]} ile aynı/çok benzer"})
        counts = [c for c in per_class if c > 0]
        return {
            "images": n, "labelled": labelled, "unlabelled": n - labelled, "empty": empty, "boxes": boxes_total,
            "classes": list(self.classes), "per_class": per_class, "images_per_class": img_per_class,
            "unknown_classes": unknown, "sizes": dict(sorted(sizes.items(), key=lambda kv: -kv[1])),
            "min_size": (min(widths), min(heights)) if widths else None,
            "max_size": (max(widths), max(heights)) if widths else None,
            "balance": (min(counts) / max(counts)) if counts else None,
            "problems": problems, "duplicates": dups,
            "splits": {s: sum(1 for it in self.items if it.split == s) for s in SPLITS},
            "suggested": sum(1 for it in self.items if self.suggest_path(it).exists()),
        }

    def problem_images(self, stats: dict | None = None) -> set[str]:
        return {p["image"] for p in (stats or self.stats())["problems"]}

    def filter(self, kind: str = "all", stats: dict | None = None) -> list[Item]:
        """kind: "all", "unlabelled", "labelled", "empty", "problems", "suggested", "class:<index>"."""
        if kind == "unlabelled":
            return [it for it in self.items if not it.label.exists()]
        if kind == "labelled":
            return [it for it in self.items if it.label.exists()]
        if kind == "empty":
            return [it for it in self.items if it.label.exists() and not self.read(it)[0]]
        if kind == "problems":
            bad = self.problem_images(stats)
            return [it for it in self.items if it.rel in bad]
        if kind == "suggested":
            return [it for it in self.items if self.suggest_path(it).exists()]
        if kind.startswith("class:"):
            c = int(kind.split(":", 1)[1])
            return [it for it in self.items if any(b[0] == c for b in self.read(it)[0])]
        return list(self.items)

    # ---- class edits (with backup) ----
    def label_files(self) -> list[pathlib.Path]:
        return [it.label for it in self.items if it.label.exists()]

    def backup(self, reason: str = "") -> pathlib.Path:
        """Copy every label file, suggestion and classes.txt / data.yaml into root/.vf_yedek/<time>/."""
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dest = self.root / BACKUP_DIR / (stamp + (f"_{reason}" if reason else ""))
        files = self.label_files() + [self.root / "classes.txt", self.root / "data.yaml"]
        sdir = self.root / SUGGEST_DIR
        if sdir.is_dir():
            files += [p for p in sdir.rglob("*.txt")]
        for f in files:
            if f.exists():
                target = dest / f.relative_to(self.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
        return dest

    def backups(self) -> list[pathlib.Path]:
        d = self.root / BACKUP_DIR
        return sorted((p for p in d.iterdir() if p.is_dir()), reverse=True) if d.is_dir() else []

    def restore(self, backup: pathlib.Path) -> int:
        """Put a backup back (label files, suggestions, class list). Returns the number of files restored."""
        backup = pathlib.Path(backup)
        n = 0
        for f in backup.rglob("*"):
            if f.is_file():
                target = self.root / f.relative_to(backup)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                n += 1
        self.reload()
        return n

    def remap(self, mapping: dict, new_classes: list[str], reason: str = "") -> pathlib.Path:
        """Rewrite every label (and suggestion) file: class ``old`` → ``mapping[old]`` (None = drop the box).
        Classes missing from ``mapping`` keep their number. Takes a backup first; returns its folder."""
        bk = self.backup(reason)
        for it in self.items:
            for path in (it.label, self.suggest_path(it)):
                if not path.exists():
                    continue
                boxes, other, bad = parse_labels(path.read_text(encoding="utf-8", errors="replace"))
                out = []
                for b in boxes:
                    c = mapping.get(b[0], b[0])
                    if c is not None:
                        out.append(format_line((c, *b[1:5]), b[5]))
                for line in other:                          # polygons: same class mapping
                    parts = line.split()
                    c = mapping.get(int(parts[0]), int(parts[0]))
                    if c is not None:
                        out.append(" ".join([str(c)] + parts[1:]))
                _write_text(path, "\n".join(out + bad))
        self.classes = list(new_classes)
        self.save_meta()
        return bk

    def rename_class(self, index: int, name: str) -> pathlib.Path:
        name = clean_name(name)
        if not 0 <= index < len(self.classes):
            raise ValueError(f"Sınıf numarası geçersiz: {index}")
        if not name:
            raise ValueError("Sınıf adı boş olamaz")
        if any(c.casefold() == name.casefold() for i, c in enumerate(self.classes) if i != index):
            raise ValueError(f"'{name}' adında bir sınıf zaten var; birleştirmek için 'Birleştir'i kullanın")
        bk = self.backup("ad")
        self.classes[index] = name
        self.save_meta()
        return bk

    def merge_classes(self, sources, target: int) -> pathlib.Path:
        """Boxes of ``sources`` become ``target``; the source classes disappear (numbers above shift down)."""
        sources = sorted({int(s) for s in sources} - {int(target)})
        n = len(self.classes)
        if not sources:
            raise ValueError("Birleştirilecek başka bir sınıf seçin")
        if not 0 <= target < n or any(not 0 <= s < n for s in sources):
            raise ValueError("Sınıf numarası geçersiz")
        keep = [i for i in range(n) if i not in sources]
        new_index = {old: k for k, old in enumerate(keep)}
        mapping = {i: new_index[target] if i in sources else new_index[i] for i in range(n)}
        return self.remap(mapping, [self.classes[i] for i in keep], "birlestir")

    def delete_class(self, index: int) -> pathlib.Path:
        """Remove a class and every box of it; the numbers above shift down."""
        n = len(self.classes)
        if not 0 <= index < n:
            raise ValueError(f"Sınıf numarası geçersiz: {index}")
        mapping = {i: (None if i == index else i - (i > index)) for i in range(n)}
        return self.remap(mapping, [c for i, c in enumerate(self.classes) if i != index], "sil")

    # ---- split ----
    def split(self, ratios=(0.7, 0.2, 0.1), seed: int = 0, labelled_only: bool = False) -> dict[str, list[Item]]:
        """Stratified train/val/test split: pictures are grouped by their rarest class (empty / unlabelled pictures
        in their own groups) and every group is dealt to the splits in the given ratios. Same seed = same split.
        Writes train.txt / val.txt / test.txt and data.yaml."""
        try:
            r = [float(v) for v in ratios]
        except (TypeError, ValueError):
            raise ValueError("Oranlar sayı olmalı (ör. 0.7, 0.2, 0.1)") from None
        r = (r + [0.0, 0.0, 0.0])[:3]
        if any(v < 0 or not math.isfinite(v) for v in r) or sum(r) <= 0 or r[0] <= 0:
            raise ValueError("Oranlar geçersiz: eğitim oranı 0'dan büyük, hiçbiri negatif olmamalı")
        r = [v / sum(r) for v in r]
        items = [it for it in self.items if it.label.exists()] if labelled_only else list(self.items)
        if not items:
            raise ValueError("Bölünecek resim yok")
        totals: dict[int, int] = {}
        boxes_of = {}
        for it in items:
            cl = sorted({b[0] for b in self.read(it)[0]})
            boxes_of[it.rel] = cl
            for c in cl:
                totals[c] = totals.get(c, 0) + 1
        groups: dict[tuple, list[Item]] = {}
        for it in items:
            cl = boxes_of[it.rel]
            key = (0, min(cl, key=lambda c: (totals[c], c))) if cl else ((1, 0) if it.label.exists() else (2, 0))
            groups.setdefault(key, []).append(it)
        rng = np.random.default_rng(int(seed))
        out = {s: [] for s in SPLITS}
        counts = [0, 0, 0]
        total = 0
        for key in sorted(groups):
            g = sorted(groups[key], key=lambda it: it.rel)
            order = rng.permutation(len(g))
            local = [0, 0, 0]
            for j, idx in enumerate(order):
                # deal to the split furthest behind its share, inside the group first, then overall
                need = [r[s] * (j + 1) - local[s] + 1e-6 * (r[s] * (total + 1) - counts[s]) for s in range(3)]
                s = int(np.argmax(need))
                local[s] += 1
                counts[s] += 1
                total += 1
                out[SPLITS[s]].append(g[idx])
        for s in SPLITS:
            out[s].sort(key=lambda it: it.rel)
            _write_text(self.root / f"{s}.txt", "\n".join(f"./{it.rel}" for it in out[s]) + ("\n" if out[s] else ""))
            for it in out[s]:
                it.split = s
        splits = {"train": "train.txt", "val": "val.txt" if out["val"] else "train.txt"}
        if out["test"]:
            splits["test"] = "test.txt"
        self.save_meta(splits)
        return out

    # ---- suggestions (model-assisted labelling) ----
    def suggest_path(self, item: Item) -> pathlib.Path:
        return (self.root / SUGGEST_DIR / item.rel).with_suffix(".txt")

    def suggestions(self, item: Item) -> list[tuple]:
        p = self.suggest_path(item)
        if not p.exists():
            return []
        return [(b[0], b[1], b[2], b[3], b[4], 0.0 if b[5] is None else b[5])
                for b in parse_labels(p.read_text(encoding="utf-8", errors="replace"))[0]]

    def write_suggestions(self, item: Item, boxes) -> None:
        """boxes: (c, cx, cy, w, h, conf)."""
        _write_text(self.suggest_path(item), "\n".join(format_line(b[:5], b[5]) for b in boxes))

    def review_queue(self) -> list[tuple[Item, list[tuple], float]]:
        """Pictures with suggestions, least sure first (no detection at all counts as 0: the model may have missed
        something). [(item, boxes, lowest confidence)]."""
        out = []
        for it in self.items:
            if self.suggest_path(it).exists():
                bx = self.suggestions(it)
                out.append((it, bx, min((b[5] for b in bx), default=0.0)))
        out.sort(key=lambda t: (t[2], t[0].rel))
        return out

    def accept(self, item: Item, min_conf: float = 0.0, keep=None) -> int:
        """Suggestions → labels (added to the existing boxes, duplicates of a box already there skipped). ``keep``:
        indices of the suggestions to take (None = all ≥ min_conf). Returns the number of boxes added."""
        sug = self.suggestions(item)
        take = [b for i, b in enumerate(sug) if (keep is None or i in keep) and b[5] >= min_conf]
        boxes, other, bad = self.read(item)
        cur = [b[:5] for b in boxes]
        added = 0
        for b in take:
            nb = to_xyxy(b, 1, 1)
            if any(c[0] == b[0] and box_iou(to_xyxy(c, 1, 1)[1:], nb[1:]) > 0.5 for c in cur):
                continue
            cur.append(tuple(b[:5]))
            added += 1
        self.write(item, cur, other + bad)
        self.suggest_path(item).unlink(missing_ok=True)
        return added

    def reject(self, item: Item) -> None:
        self.suggest_path(item).unlink(missing_ok=True)

    def accept_all(self, min_conf: float = 0.0) -> tuple[int, int]:
        imgs = boxes = 0
        for it, _bx, _lo in self.review_queue():
            boxes += self.accept(it, min_conf)
            imgs += 1
        return imgs, boxes

    def reject_all(self) -> int:
        n = 0
        for it in self.items:
            if self.suggest_path(it).exists():
                self.reject(it)
                n += 1
        return n

    # ---- adding pictures ----
    def add_images(self, paths) -> int:
        """Copy pictures into images/ (a name already there gets a number)."""
        if self.layout == "roboflow":
            raise ValueError("Bu düzende (train/images…) resim eklenemez; YOLO düzenine dönüştürün")
        dest = self.root / "images" if self.layout == "yolo" else self.root
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in paths:
            p = pathlib.Path(p)
            if p.suffix.lower() not in IMAGE_EXT or not p.is_file():
                continue
            t = dest / p.name
            k = 1
            while t.exists() or t.with_suffix(".txt").exists() and self.layout == "flat":
                t = dest / f"{p.stem}_{k}{p.suffix}"
                k += 1
            shutil.copy2(p, t)
            n += 1
        self.reload()
        return n


# --- conversion -----------------------------------------------------------------------------------------------------

def _unique_name(name: str, used: set) -> str:
    stem, ext = pathlib.Path(name).stem, pathlib.Path(name).suffix
    out, k = name, 1
    while out.lower() in used or pathlib.Path(out).stem.lower() in {pathlib.Path(u).stem for u in used}:
        out = f"{stem}_{k}{ext}"
        k += 1
    used.add(out.lower())
    return out


def _empty_out(out: pathlib.Path, src_root: pathlib.Path | None = None) -> pathlib.Path:
    out = pathlib.Path(out)
    if src_root is not None and out.resolve() == pathlib.Path(src_root).resolve():
        raise ValueError("Hedef klasör kaynakla aynı olamaz")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Hedef klasör boş değil: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_yolo(ds: Dataset, out, use_splits: bool = True, progress=None) -> pathlib.Path:
    """Copy into a clean YOLO folder: images/<split>/, labels/<split>/, data.yaml, classes.txt."""
    out = _empty_out(out, ds.root)
    split_mode = use_splits and any(it.split for it in ds.items)
    used: dict[str, set] = {}
    lists: dict[str, list[str]] = {s: [] for s in SPLITS}
    for i, it in enumerate(ds.items):
        if progress is not None and progress(i, len(ds.items)) is False:
            break
        sub = (it.split or "train") if split_mode else ""
        name = _unique_name(it.image.name, used.setdefault(sub, set()))
        img_dst = out / "images" / sub / name
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(it.image, img_dst)
        if it.label.exists():
            lbl = (out / "labels" / sub / name).with_suffix(".txt")
            lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(it.label, lbl)
        if sub:
            lists[sub].append(name)
    (out / "labels").mkdir(exist_ok=True)
    _write_text(out / "classes.txt", "\n".join(ds.classes))
    if split_mode:
        sp = {"train": "images/train", "val": "images/val" if lists["val"] else "images/train"}
        if lists["test"]:
            sp["test"] = "images/test"
        write_data_yaml(out, ds.classes, "images", sp)
    else:
        write_data_yaml(out, ds.classes, "images", keep_splits=False)
    return out


def _coco_doc(ds: Dataset, items: list[Item], names: dict[str, str]) -> dict:
    images, anns = [], []
    for it in items:
        sz = ds.size(it)
        if sz is None:
            continue
        w, h = sz
        iid = len(images) + 1
        images.append({"id": iid, "file_name": names[it.rel], "width": w, "height": h})
        for b in ds.boxes(it):
            c, x1, y1, x2, y2 = to_xyxy(b, w, h)
            if c >= len(ds.classes):
                continue
            bw, bh = x2 - x1, y2 - y1
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": c + 1,
                         "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                         "area": round(bw * bh, 2), "iscrowd": 0})
    return {"info": {"description": "VisionFlow Studio veri seti"}, "images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": n, "supercategory": ""} for i, n in enumerate(ds.classes)]}


def export_coco(ds: Dataset, out, progress=None) -> pathlib.Path:
    """images/ + annotations.json (and instances_<split>.json when the dataset is split). Returns the json."""
    out = _empty_out(out, ds.root)
    used: set = set()
    names = {}
    for i, it in enumerate(ds.items):
        if progress is not None and progress(i, len(ds.items)) is False:
            break
        names[it.rel] = _unique_name(it.image.name, used)
        (out / "images").mkdir(exist_ok=True)
        shutil.copy2(it.image, out / "images" / names[it.rel])
    items = [it for it in ds.items if it.rel in names]
    path = out / "annotations.json"
    path.write_text(json.dumps(_coco_doc(ds, items, names), ensure_ascii=False, indent=1), encoding="utf-8")
    for s in SPLITS:
        part = [it for it in items if it.split == s]
        if part:
            (out / f"instances_{s}.json").write_text(json.dumps(_coco_doc(ds, part, names), ensure_ascii=False),
                                                     encoding="utf-8")
    return path


def export_voc(ds: Dataset, out, progress=None) -> pathlib.Path:
    """Pascal VOC: JPEGImages/, Annotations/*.xml (1-based pixel corners), ImageSets/Main/<split>.txt."""
    out = _empty_out(out, ds.root)
    (out / "JPEGImages").mkdir()
    (out / "Annotations").mkdir()
    used: set = set()
    sets: dict[str, list[str]] = {s: [] for s in SPLITS}
    for i, it in enumerate(ds.items):
        if progress is not None and progress(i, len(ds.items)) is False:
            break
        sz = ds.size(it)
        if sz is None:
            continue
        w, h = sz
        name = _unique_name(it.image.name, used)
        shutil.copy2(it.image, out / "JPEGImages" / name)
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "JPEGImages"
        ET.SubElement(root, "filename").text = name
        size = ET.SubElement(root, "size")
        for k, v in (("width", w), ("height", h), ("depth", 3)):
            ET.SubElement(size, k).text = str(v)
        ET.SubElement(root, "segmented").text = "0"
        for b in ds.boxes(it):
            c, x1, y1, x2, y2 = to_xyxy(b, w, h)
            if c >= len(ds.classes):
                continue
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = ds.classes[c]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bb = ET.SubElement(obj, "bndbox")
            x1, y1 = min(max(x1, 0), w), min(max(y1, 0), h)
            x2, y2 = min(max(x2, 0), w), min(max(y2, 0), h)
            for k, v in (("xmin", x1 + 1), ("ymin", y1 + 1), ("xmax", x2), ("ymax", y2)):
                ET.SubElement(bb, k).text = str(int(round(v)))
        ET.indent(root) if hasattr(ET, "indent") else None
        ET.ElementTree(root).write(out / "Annotations" / (pathlib.Path(name).stem + ".xml"), encoding="utf-8",
                                   xml_declaration=True)
        if it.split:
            sets[it.split].append(pathlib.Path(name).stem)
    _write_text(out / "classes.txt", "\n".join(ds.classes))
    if any(sets.values()):
        (out / "ImageSets" / "Main").mkdir(parents=True)
        for s, stems in sets.items():
            if stems:
                (out / "ImageSets" / "Main" / f"{s}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    return out


def _find_image(name: str, dirs: list[pathlib.Path]) -> pathlib.Path | None:
    for d in dirs:
        p = d / name
        if p.is_file():
            return p
    base = pathlib.Path(name).name
    for d in dirs:
        p = d / base
        if p.is_file():
            return p
    stem = pathlib.Path(name).stem
    for d in dirs:
        if d.is_dir():
            for ext in IMAGE_EXT:
                p = d / (stem + ext)
                if p.is_file():
                    return p
    return None


def import_coco(json_path, out, image_dir=None, progress=None) -> Dataset:
    """COCO detection json → a new YOLO dataset in ``out`` (pictures copied)."""
    json_path = pathlib.Path(json_path)
    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        images, anns, cats = doc["images"], doc.get("annotations", []), doc.get("categories", [])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"COCO dosyası okunamadı ({json_path.name}): {exc}") from None
    if not isinstance(images, list):
        raise ValueError("COCO dosyasında 'images' listesi yok")
    cats = sorted(cats, key=lambda c: c.get("id", 0))
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    dirs = [pathlib.Path(image_dir)] if image_dir else []
    dirs += [json_path.parent, json_path.parent / "images", json_path.parent.parent / "images",
             json_path.parent.parent]
    out = _empty_out(out)
    ds = Dataset(out, create=True)
    ds.classes = [clean_name(c.get("name", f"sinif{i}")) or f"sinif{i}" for i, c in enumerate(cats)]
    by_image: dict = {}
    for a in anns:
        by_image.setdefault(a.get("image_id"), []).append(a)
    used: set = set()
    missing = 0
    for i, im in enumerate(images):
        if progress is not None and progress(i, len(images)) is False:
            break
        src = _find_image(str(im.get("file_name", "")), dirs)
        if src is None:
            missing += 1
            continue
        name = _unique_name(src.name, used)
        dst = out / "images" / name
        shutil.copy2(src, dst)
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            img = imread(dst)
            if img is None:
                continue
            h, w = img.shape[:2]
        lines = []
        for a in by_image.get(im.get("id"), []):
            bb = a.get("bbox")
            if not bb or len(bb) < 4 or a.get("category_id") not in cat_index:
                continue
            x, y, bw, bh = (float(v) for v in bb[:4])
            if bw <= 0 or bh <= 0:
                continue
            lines.append(format_line(to_yolo([cat_index[a["category_id"]], x, y, x + bw, y + bh], w, h)))
        _write_text((out / "labels" / name).with_suffix(".txt"), "\n".join(lines))
    ds.save_meta()
    ds.reload()
    ds.missing = missing
    return ds


def import_voc(folder, out, progress=None) -> Dataset:
    """Pascal VOC (a folder with Annotations/*.xml and JPEGImages/, or xml files next to the pictures) → YOLO."""
    folder = pathlib.Path(folder)
    ann_dir = folder / "Annotations" if (folder / "Annotations").is_dir() else folder
    xmls = sorted(ann_dir.rglob("*.xml"))
    if not xmls:
        raise ValueError(f"Pascal VOC xml dosyası bulunamadı: {folder}")
    parsed = []
    names: list[str] = []
    for cand in (folder / "classes.txt", folder / "labels.txt", ann_dir / "classes.txt"):
        if cand.is_file():                        # the class order an export wrote (VOC itself has none)
            names = [clean_name(n) for n in cand.read_text(encoding="utf-8").splitlines() if n.strip()]
            break
    for x in xmls:
        try:
            root = ET.parse(x).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"VOC dosyası okunamadı ({x.name}): {exc}") from None
        objs = []
        for o in root.iter("object"):
            n = clean_name(o.findtext("name", ""))
            bb = o.find("bndbox")
            if not n or bb is None:
                continue
            try:
                v = [float(bb.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")]
            except ValueError:
                continue
            if not all(math.isfinite(t) for t in v):
                continue
            objs.append((n, v))
            if n not in names:
                names.append(n)
        parsed.append((x, root, objs))
    out = _empty_out(out)
    ds = Dataset(out, create=True)
    ds.classes = names
    used: set = set()
    missing = 0
    for i, (x, root, objs) in enumerate(parsed):
        if progress is not None and progress(i, len(parsed)) is False:
            break
        fname = root.findtext("filename") or x.stem
        src = _find_image(fname, [folder / "JPEGImages", x.parent, folder, x.parent.parent / "JPEGImages"])
        if src is None:
            src = _find_image(x.stem, [folder / "JPEGImages", x.parent, folder])
        if src is None:
            missing += 1
            continue
        name = _unique_name(src.name, used)
        dst = out / "images" / name
        shutil.copy2(src, dst)
        try:
            w = int(float(root.findtext("size/width", "0")))
            h = int(float(root.findtext("size/height", "0")))
        except ValueError:
            w = h = 0
        if w <= 0 or h <= 0:
            img = imread(dst)
            if img is None:
                continue
            h, w = img.shape[:2]
        lines = [format_line(to_yolo([ds.classes.index(n), v[0] - 1, v[1] - 1, v[2], v[3]], w, h))
                 for n, v in objs if v[2] > v[0] - 1 and v[3] > v[1] - 1]
        _write_text((out / "labels" / name).with_suffix(".txt"), "\n".join(lines))
    ds.save_meta()
    ds.reload()
    ds.missing = missing
    return ds


def detect_format(src) -> str:
    """"coco", "voc" or "yolo" for a source path."""
    src = pathlib.Path(src)
    if src.is_file():
        if src.suffix.lower() == ".json":
            return "coco"
        if src.suffix.lower() == ".xml":
            return "voc"
        raise ValueError(f"Tanınmayan dosya: {src.name}")
    if not src.is_dir():
        raise ValueError(f"Kaynak bulunamadı: {src}")
    if (src / "Annotations").is_dir() or (any(src.glob("*.xml")) and not (src / "labels").is_dir()):
        return "voc"
    if not (src / "labels").is_dir() and not any(src.glob("*.txt")) and \
            (any(src.glob("*.json")) or any(src.glob("annotations/*.json"))):
        return "coco"
    return "yolo"


def _coco_json_in(src: pathlib.Path) -> pathlib.Path:
    if src.is_file():
        return src
    for cand in ("annotations.json", "_annotations.coco.json", "instances.json"):
        if (src / cand).is_file():
            return src / cand
    js = sorted(src.glob("*.json")) + sorted(src.glob("annotations/*.json"))
    js = [j for j in js if j.name != CACHE_FILE]
    if not js:
        raise ValueError(f"COCO json dosyası bulunamadı: {src}")
    return js[0]


def convert(src, out, fmt: str, src_format: str = "auto", progress=None) -> pathlib.Path:
    """Convert any supported dataset to ``fmt`` ("yolo", "coco", "voc") in ``out``."""
    fmt = fmt.lower()
    if fmt not in ("yolo", "coco", "voc"):
        raise ValueError("Biçim yolo, coco veya voc olmalı")
    src = pathlib.Path(src)
    kind = detect_format(src) if src_format == "auto" else src_format.lower()
    if kind == "yolo":
        ds = Dataset(src)
        if fmt == "yolo":
            return export_yolo(ds, out, progress=progress)
        return export_coco(ds, out, progress) if fmt == "coco" else export_voc(ds, out, progress)
    if fmt == "yolo":
        ds = import_coco(_coco_json_in(src), out, progress=progress) if kind == "coco" else \
            import_voc(src.parent if src.is_file() else src, out, progress)
        return ds.root
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vf_ds_") as tmp:
        tmp = pathlib.Path(tmp) / "yolo"
        ds = import_coco(_coco_json_in(src), tmp) if kind == "coco" else \
            import_voc(src.parent if src.is_file() else src, tmp)
        res = export_coco(ds, out, progress) if fmt == "coco" else export_voc(ds, out, progress)
        return pathlib.Path(out) if fmt == "voc" else res


# --- augmentation ---------------------------------------------------------------------------------------------------

def augment(img: np.ndarray, boxes, ops: dict, rng=None) -> tuple[np.ndarray, list]:
    """Apply ``ops`` to a picture and its pixel boxes [c, x1, y1, x2, y2] (box edges, 0..W).

    ops: flip_h / flip_v (bool), rot90 (quarter turns clockwise), rotate (degrees, counter-clockwise, same canvas),
    brightness (added to every pixel, -255..255), blur (Gaussian kernel, odd), noise (Gaussian sigma)."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Çoğaltma için geçerli bir resim gerekli")
    rng = rng if rng is not None else np.random.default_rng()
    out = img.copy()
    bx = [list(map(float, b[1:5])) for b in boxes]
    cls = [int(b[0]) for b in boxes]
    h, w = out.shape[:2]
    if ops.get("flip_h"):
        out = cv2.flip(out, 1)
        bx = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in bx]
    if ops.get("flip_v"):
        out = cv2.flip(out, 0)
        bx = [[x1, h - y2, x2, h - y1] for x1, y1, x2, y2 in bx]
    k = int(ops.get("rot90", 0) or 0) % 4
    for _ in range(k):                        # clockwise: (x, y) → (H - y, x)
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        bx = [[h - y2, x1, h - y1, x2] for x1, y1, x2, y2 in bx]
        h, w = w, h
    ang = float(ops.get("rotate", 0) or 0)
    if ang:
        M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), ang, 1.0)
        # plain border (the mean colour): a mirrored border would copy objects without labels
        fill = [float(v) for v in np.atleast_1d(out.reshape(-1, *out.shape[2:]).mean(axis=0))][:4]
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=fill + [0.0] * (4 - len(fill)))
        nb = []
        for x1, y1, x2, y2 in bx:
            # box edges → pixel centre coordinates (−0.5), rotate, back (+0.5)
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64) - 0.5
            r = pts @ M[:, :2].T + M[:, 2] + 0.5
            nb.append([r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()])
        bx = nb
    b = float(ops.get("brightness", 0) or 0)
    if b:
        out = np.clip(out.astype(np.int16) + int(round(b)), 0, 255).astype(np.uint8) if out.dtype == np.uint8 \
            else np.clip(out.astype(np.float32) + b, 0, np.iinfo(out.dtype).max if out.dtype.kind in "ui" else None
                         ).astype(out.dtype)
    kb = int(ops.get("blur", 0) or 0)
    if kb > 1:
        kb += (kb % 2 == 0)
        out = cv2.GaussianBlur(out, (kb, kb), 0)
    s = float(ops.get("noise", 0) or 0)
    if s > 0:
        noise = rng.normal(0, s, out.shape)
        top = 255 if out.dtype == np.uint8 else (np.iinfo(out.dtype).max if out.dtype.kind in "ui" else None)
        out = np.clip(out.astype(np.float32) + noise, 0, top).astype(out.dtype)
    res = []
    for c, (x1, y1, x2, y2) in zip(cls, bx):
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 - x1 >= 2 and y2 - y1 >= 2:     # a box rotated out of the picture is dropped
            res.append([c, x1, y1, x2, y2])
    return out, res


def random_ops(rng, enabled, max_angle: float = 10.0, max_brightness: float = 40.0, max_blur: int = 5,
               max_noise: float = 12.0) -> dict:
    """Random settings for the enabled operations (each one used with probability 1/2, at least one)."""
    enabled = [e for e in enabled if e in AUG_OPS]
    if not enabled:
        raise ValueError("En az bir çoğaltma işlemi seçin")
    ops: dict = {}
    while not ops:
        for e in enabled:
            if rng.random() >= 0.5:
                continue
            if e in ("flip_h", "flip_v"):
                ops[e] = True
            elif e == "rot90":
                ops[e] = int(rng.integers(1, 4))
            elif e == "rotate":
                ops[e] = float(rng.uniform(-max_angle, max_angle))
            elif e == "brightness":
                ops[e] = float(rng.uniform(-max_brightness, max_brightness))
            elif e == "blur":
                ops[e] = int(rng.choice([3, 5, 7][:max(1, (max_blur - 1) // 2)]))
            elif e == "noise":
                ops[e] = float(rng.uniform(2.0, max_noise))
    return ops


def augment_dataset(ds: Dataset, items, copies: int, enabled, seed: int = 0, progress=None, **limits) -> int:
    """Save ``copies`` augmented copies of every labelled picture in ``items`` next to it (name_augK) with
    transformed labels; a split list the source is in gets the copies too. Returns the number written."""
    if copies < 1:
        raise ValueError("Kopya sayısı en az 1 olmalı")
    rng = np.random.default_rng(int(seed))
    items = [it for it in items if it.label.exists()]
    if not items:
        raise ValueError("Çoğaltılacak etiketli resim yok")
    written = 0
    new_rel: dict[str, list[str]] = {}
    for i, it in enumerate(items):
        if progress is not None and progress(i, len(items)) is False:
            break
        img = imread(it.image)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes, other, _bad = ds.read(it)
        px = [to_xyxy(b, w, h) for b in boxes]
        for k in range(copies):
            ops = random_ops(rng, enabled, **limits)
            if other and any(o in ops for o in ("rotate",)):
                ops.pop("rotate")                         # polygons are not rotated: keep them consistent
                ops = ops or {"brightness": 20.0}
            aimg, abx = augment(img, px, ops, rng)
            n = 1
            while True:
                dst = it.image.with_name(f"{it.image.stem}_aug{n}{it.image.suffix}")
                if not dst.exists():
                    break
                n += 1
            ah, aw = aimg.shape[:2]
            if not imwrite(dst, aimg):
                continue
            lbl = it.label.with_name(f"{it.image.stem}_aug{n}.txt")
            poly = _transform_polys(other, ops, w, h) if other else []
            _write_text(lbl, "\n".join([format_line(to_yolo(b, aw, ah)) for b in abx] + poly))
            written += 1
            if it.split and (ds.root / f"{it.split}.txt").exists():
                new_rel.setdefault(it.split, []).append(dst.relative_to(ds.root).as_posix())
    for s, rels in new_rel.items():
        f = ds.root / f"{s}.txt"
        text = f.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            text += "\n"
        _write_text(f, text + "".join(f"./{r}\n" for r in rels))
    ds.reload()
    return written


def _transform_polys(lines, ops, w, h) -> list[str]:
    """Normalised polygon lines through flips and quarter turns (rotate is never used together with them)."""
    out = []
    for line in lines:
        parts = line.split()
        xy = np.array([float(v) for v in parts[1:]]).reshape(-1, 2)
        if ops.get("flip_h"):
            xy[:, 0] = 1 - xy[:, 0]
        if ops.get("flip_v"):
            xy[:, 1] = 1 - xy[:, 1]
        for _ in range(int(ops.get("rot90", 0) or 0) % 4):
            xy = np.stack([1 - xy[:, 1], xy[:, 0]], axis=1)
        out.append(" ".join([parts[0]] + [f"{v:.6f}" for v in xy.flatten()]))
    return out


# --- model-assisted labelling ---------------------------------------------------------------------------------------

class Predictor:
    """A .vfdet few-shot model or a YOLO ONNX model (run through the YOLO block, so the device chosen in
    Araçlar → Hızlandırma is used). ``predict(img)`` → [(label, x, y, w, h, score)]."""

    def __init__(self, path, conf: float = 0.5):
        path = pathlib.Path(str(path).strip())
        if not path.is_file():
            raise ValueError(f"Model dosyası yok: {path}")
        self.path, self.conf, self.node, self.model = path, float(conf), None, None
        ext = path.suffix.lower()
        if ext == ".vfdet":
            from .fewshot_det import DetModel
            self.model = DetModel.load(path)
            self.kind, self.device = "fewshot", "CPU"
        elif ext == ".onnx":
            from .core import create
            from .core.registry import NODE_REGISTRY
            if "YoloDetector" not in NODE_REGISTRY:
                from .core import load_builtin
                load_builtin()
            node = create("YoloDetector")
            node.values.update(model=str(path), conf=self.conf)
            if not node.safe_setup():
                raise ValueError(f"YOLO modeli açılamadı: {node.setup_error or path.name}")
            self.node, self.kind = node, "yolo"
            self.device = node.state.get("device", "CPU")
        else:
            raise ValueError("Model .vfdet (Az Örnekle Nesne Bul) veya .onnx (YOLO) olmalı")

    def predict(self, img) -> list[tuple]:
        if self.model is not None:
            dets = self.model.detect(img, score=self.conf)
        else:
            dets = self.node.process({"image": img})["detections"]
        out = []
        for d in dets or []:
            try:
                x, y, w, h = (float(v) for v in d["box"])
                s = float(d.get("score", 1.0))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0 and s >= self.conf:
                out.append((str(d.get("label", "nesne")).split(" #")[0], x, y, w, h, s))
        return out

    def close(self) -> None:
        if self.node is not None:
            try:
                self.node.teardown()
            except Exception:  # noqa: BLE001
                pass
            self.node = None


def autolabel(ds: Dataset, model, conf: float = 0.5, items=None, progress=None) -> dict:
    """Run a model over the unlabelled pictures (or ``items``) and write suggestions (never real labels).
    ``model``: a Predictor or a model path. Returns {"images", "boxes", "device"}."""
    own = not isinstance(model, Predictor)
    pred = Predictor(model, conf) if own else model
    try:
        todo = [it for it in (ds.items if items is None else items) if not it.label.exists()]
        imgs = boxes = 0
        for i, it in enumerate(todo):
            if progress is not None and progress(i, len(todo)) is False:
                break
            img = imread(it.image)
            if img is None:
                continue
            h, w = img.shape[:2]
            sug = []
            for name, x, y, bw, bh, s in pred.predict(img):
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                c = ds.class_index(name)
                nb = (*to_yolo([c, x1, y1, x2, y2], w, h), s)
                if any(o[0] == c and box_iou(to_xyxy(o, 1, 1)[1:], to_xyxy(nb, 1, 1)[1:]) > 0.5 for o in sug):
                    continue
                sug.append(nb)
            ds.write_suggestions(it, sug)
            imgs += 1
            boxes += len(sug)
        return {"images": imgs, "boxes": boxes, "device": pred.device}
    finally:
        if own:
            pred.close()


def train_fewshot(ds: Dataset, out=None, max_images: int = 12, angle_range: float = 30.0, scale_range=(0.8, 1.2),
                  progress=None) -> pathlib.Path:
    """Teach a few-shot detector (.vfdet) from the boxes already drawn: a handful of labelled pictures chosen so
    every class is seen. Saved to ``out`` (default root/modeller/veriseti.vfdet)."""
    from .fewshot_det import MAX_CLASSES, MIN_BOX, DetModel
    cand = []
    for it in ds.items:
        if not it.label.exists():
            continue
        bx = [b for b in ds.boxes(it) if b[0] < len(ds.classes)]
        if bx:
            cand.append((it, bx))
    if not cand:
        raise ValueError("Önce birkaç resmi etiketleyin: az örnekle model eğitmek için kutulu resim gerekli")
    used_classes = sorted({b[0] for _it, bx in cand for b in bx})
    if len(used_classes) > MAX_CLASSES:
        raise ValueError(f"Az örnekle model en çok {MAX_CLASSES} sınıf öğrenebilir (veri setinde "
                         f"{len(used_classes)} sınıf kutulu)")
    # greedy: pictures that add the least seen class first, then the ones with most boxes
    seen = {c: 0 for c in used_classes}
    chosen = []
    pool = sorted(cand, key=lambda t: (-len(t[1]), t[0].rel))
    while pool and len(chosen) < max(1, int(max_images)):
        best = max(pool, key=lambda t: (sum(1.0 / (1 + seen[b[0]]) for b in t[1]), -pool.index(t)))
        pool.remove(best)
        chosen.append(best)
        for b in best[1]:
            seen[b[0]] += 1
    # the model's classes are the dataset's classes that have boxes (their names keep the mapping)
    names = [ds.classes[c] for c in used_classes]
    remap = {c: i for i, c in enumerate(used_classes)}
    images, boxes = [], []
    for it, bx in chosen:
        img = imread(it.image)
        if img is None:
            continue
        h, w = img.shape[:2]
        row = []
        for b in bx:
            c, x1, y1, x2, y2 = to_xyxy(b, w, h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 >= MIN_BOX and y2 - y1 >= MIN_BOX:
                row.append((x1, y1, x2 - x1, y2 - y1, remap[c]))
        if row:
            images.append(img)
            boxes.append(row)
    if not images:
        raise ValueError(f"Kutular çok küçük (en az {MIN_BOX}x{MIN_BOX} piksel olmalı)")
    model = DetModel.train(images, boxes, names, angle_range=angle_range, scale_range=scale_range,
                           progress=progress)
    out = pathlib.Path(out) if out else ds.root / MODEL_DIR / "veriseti.vfdet"
    return pathlib.Path(model.save(out))


# --- command line ---------------------------------------------------------------------------------------------------

CLI_HELP = """Veri seti araçları — kullanım:

  python -m visionflow veriseti istatistik KLASÖR [--json]
        Resim/kutu sayıları, sınıf dengesi, boyutlar ve sorunlar (küçük kutu, dışarı taşan kutu, kopya resim…)
  python -m visionflow veriseti bol KLASÖR [--oran 0.7,0.2,0.1] [--tohum 0]
        Sınıf dengeli eğitim/doğrulama/test bölmesi (train.txt, val.txt, test.txt, data.yaml)
  python -m visionflow veriseti donustur KAYNAK HEDEF --bicim yolo|coco|voc [--kaynak-bicim auto|yolo|coco|voc]
        YOLO / COCO json / Pascal VOC xml arasında dönüştürür (HEDEF boş bir klasör olmalı)
  python -m visionflow veriseti etiketle KLASÖR --model MODEL.vfdet|MODEL.onnx [--esik 0.5] [--kabul]
  python -m visionflow veriseti etiketle KLASÖR --egit [--esik 0.5] [--kabul]
        Etiketsiz resimlere model önerisi yazar (oneriler/ klasörü). --egit: mevcut etiketlerden az örnekle
        model eğitip onu kullanır. --kabul: önerileri doğrudan etikete çevirir."""


def _fmt_stats(st: dict) -> str:
    lines = [f"Resim: {st['images']} · etiketli: {st['labelled']} · etiketsiz: {st['unlabelled']} · "
             f"boş (nesnesiz): {st['empty']} · kutu: {st['boxes']}"]
    if st["suggested"]:
        lines.append(f"Onay bekleyen öneri: {st['suggested']} resim")
    lines.append("Sınıflar:")
    for i, name in enumerate(st["classes"]):
        lines.append(f"  {i:>3} {name:<24} {st['per_class'][i]:>6} kutu  {st['images_per_class'][i]:>6} resim")
    for c, n in sorted(st["unknown_classes"].items()):
        lines.append(f"  {c:>3} {'(tanımsız)':<24} {n:>6} kutu")
    if st["balance"] is not None:
        lines.append(f"Sınıf dengesi (en az / en çok): {st['balance']:.2f}")
    if st["sizes"]:
        top = ", ".join(f"{k} ({v})" for k, v in list(st["sizes"].items())[:5])
        lines.append(f"Resim boyutları: {top}")
    if any(st["splits"].values()):
        lines.append("Bölme: " + ", ".join(f"{k}={v}" for k, v in st["splits"].items()))
    probs: dict[str, int] = {}
    for p in st["problems"]:
        probs[p["kind"]] = probs.get(p["kind"], 0) + 1
    if probs:
        lines.append("Sorunlar:")
        for k, n in probs.items():
            lines.append(f"  {PROBLEM_NAMES.get(k, k)}: {n}")
        for p in st["problems"][:30]:
            lines.append(f"    {p['image']}: {PROBLEM_NAMES.get(p['kind'], p['kind'])} — {p['detail']}")
    else:
        lines.append("Sorun bulunmadı.")
    return "\n".join(lines)


def cli(argv=None) -> int:
    import argparse
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "yardim"):
        print(CLI_HELP)
        return 0
    cmd, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"python -m visionflow veriseti {cmd}")
    try:
        if cmd == "istatistik":
            ap.add_argument("klasor")
            ap.add_argument("--json", action="store_true")
            ap.add_argument("--kucuk", type=float, default=4.0, help="küçük kutu sınırı (piksel)")
            a = ap.parse_args(rest)
            st = Dataset(a.klasor).stats(tiny_px=a.kucuk)
            print(json.dumps(st, ensure_ascii=False, indent=1) if a.json else _fmt_stats(st))
            return 0
        if cmd == "bol":
            ap.add_argument("klasor")
            ap.add_argument("--oran", default="0.7,0.2,0.1")
            ap.add_argument("--tohum", type=int, default=0)
            ap.add_argument("--etiketliler", action="store_true", help="yalnız etiketli resimler")
            a = ap.parse_args(rest)
            try:
                ratios = [float(v) for v in a.oran.replace(";", ",").split(",") if v.strip()]
            except ValueError:
                raise ValueError("--oran sayılardan oluşmalı (ör. 0.7,0.2,0.1)") from None
            out = Dataset(a.klasor).split(ratios, a.tohum, a.etiketliler)
            print("Bölündü: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
            return 0
        if cmd == "donustur":
            ap.add_argument("kaynak")
            ap.add_argument("hedef")
            ap.add_argument("--bicim", required=True, choices=["yolo", "coco", "voc"])
            ap.add_argument("--kaynak-bicim", default="auto", choices=["auto", "yolo", "coco", "voc"])
            a = ap.parse_args(rest)
            res = convert(a.kaynak, a.hedef, a.bicim, a.kaynak_bicim)
            print(f"Dönüştürüldü: {res}")
            return 0
        if cmd == "etiketle":
            ap.add_argument("klasor")
            ap.add_argument("--model", default="")
            ap.add_argument("--egit", action="store_true", help="mevcut etiketlerden az örnekle model eğit")
            ap.add_argument("--esik", type=float, default=0.5)
            ap.add_argument("--kabul", action="store_true", help="önerileri doğrudan etiket yap")
            a = ap.parse_args(rest)
            ds = Dataset(a.klasor)
            model = a.model
            if a.egit:
                model = str(train_fewshot(ds))
                print(f"Az örnekle model eğitildi: {model}")
            if not model:
                raise ValueError("--model veya --egit verin")
            res = autolabel(ds, model, a.esik)
            print(f"{res['images']} resme {res['boxes']} kutu önerildi (cihaz: {res['device']})")
            if a.kabul:
                n, b = ds.accept_all()
                print(f"Kabul edildi: {n} resim, {b} kutu")
            else:
                print("Önerileri Araçlar → Veri seti yöneticisi → Otomatik etiketle sekmesinde inceleyin.")
            return 0
    except ValueError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:                 # argparse: wrong arguments
        return int(exc.code or 0)
    print(f"Bilinmeyen komut: {cmd}\n\n{CLI_HELP}", file=sys.stderr)
    return 2
