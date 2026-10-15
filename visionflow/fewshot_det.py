"""Few-shot object detection ("Az Örnekle Nesne Bul"): the user draws boxes around an object on a few pictures and
gets a detector that runs on the CPU in tens of milliseconds, with no network, no torch and no scikit-learn.

How it works (two stages, one per class):

1. **Rotation- and scale-tolerant template search.** The first boxed example of a class becomes a *prototype*: its
   grey picture in a canonical frame, an object mask (GrabCut inside the box, so the background corners of a rotated
   object do not count) and the object's outline. The other examples are matched against the prototype to find their
   own angle and scale; well matching ones are averaged into the prototype (less noise), clearly different ones
   (another side of the part, another look) become extra prototypes. At run time every prototype is rotated over the
   angle range and scaled over the scale range, all at a reduced "work" resolution where the object is about 24 px,
   and correlated with the picture (zero-mean masked template, normalised by the local contrast). Local maxima are the
   candidates; each is refined by an exact masked normalised cross-correlation over a few angles, scales and shifts.

2. **A verifier trained on the user's own pictures.** Every candidate is warped into the prototype's frame (so the
   features do not depend on the angle) and described by a gradient-orientation histogram (HOG-like, inside the
   object mask), a normalised 8x8 thumbnail, colour statistics and its correlation scores. A ridge-regularised
   logistic regression (plain numpy, so a saved model gives exactly the same result with any OpenCV build) is trained
   with positives = the boxed objects (jittered in angle, scale, shift and brightness) plus the candidates that hit
   them, and negatives = the candidates that hit nothing (hard negatives: look-alike shapes in the pictures), random
   patches of the unboxed area and the other classes' objects. The verifier's probability is the detection score.

Each detection is a dict like the YOLO blocks' (``box`` x, y, w, h; ``score``; ``class_id``; ``label``) plus
``angle`` (degrees, counter-clockwise, relative to the class's first example), ``poly`` (the rotated example box),
``center`` and ``size`` (as the YOLO OBB block), so every detection block can use it.

A model is one ``.vfdet`` file (numpy .npz: arrays + a JSON header)."""
from __future__ import annotations

import io
import json
import math
import pathlib
import time

import cv2
import numpy as np

from .fewshot_seg import to_bgr8

FORMAT = "visionflow-fewshot-det"
VERSION = 1
EXT = ".vfdet"
MAX_CLASSES = 16
MAX_PROTOS = 3              # prototypes per class
WORK_SIZE = 20.0
CANON_MAX = 96              # longest side of a prototype's canonical frame
FEAT_AREA = 40 * 40         # feature patch area
HOG_CELLS = 5
HOG_BINS = 9
COARSE_MIN = 0.2            # candidates below this coarse correlation are not looked at
MIN_BOX = 6                 # smallest box side the user may draw (px)
MAX_ANGLES = 48

DEFAULT_COLORS = [(0, 200, 0), (0, 0, 230), (230, 120, 0), (0, 200, 230), (200, 0, 200), (230, 230, 0),
                  (0, 128, 255), (128, 0, 255), (255, 0, 128), (128, 255, 0), (90, 160, 160), (160, 90, 160),
                  (160, 160, 90), (255, 255, 255), (60, 60, 180), (40, 40, 40)]


def default_color(i: int) -> tuple[int, int, int]:
    return DEFAULT_COLORS[i % len(DEFAULT_COLORS)]


# --- small helpers -------------------------------------------------------------------------------------------------

def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _rot(theta_deg: float, k: float) -> np.ndarray:
    """2x2 matrix of a counter-clockwise (on screen) rotation by theta and scaling by k; same convention as
    cv2.getRotationMatrix2D."""
    t = math.radians(theta_deg)
    c, s = math.cos(t) * k, math.sin(t) * k
    return np.array([[c, s], [-s, c]], np.float64)


def _affine(theta: float, k: float, src_c, dst_c) -> np.ndarray:
    """2x3 matrix mapping a point u of a canonical frame (centre src_c) to dst = R(theta)*k*(u - src_c) + dst_c."""
    a = _rot(theta, k)
    t = np.asarray(dst_c, np.float64) - a @ np.asarray(src_c, np.float64)
    return np.hstack([a, t[:, None]])


def _warp_to_canon(img, theta, k, center, size, src_c, interp=cv2.INTER_LINEAR):
    """Cut the rotated, scaled region around ``center`` out of ``img`` into an upright canonical patch of ``size``
    (w, h). ``k`` = picture pixels per canonical unit."""
    m = _affine(theta, k, src_c, center)
    return cv2.warpAffine(img, m, (int(size[0]), int(size[1])), flags=interp | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _norm_angle(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _masked_ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    """Normalised cross-correlation of two equal-size float pictures inside a 0/1 mask."""
    w = m > 0
    if w.sum() < 4:
        return 0.0
    x, y = a[w], b[w]
    x = x - x.mean()
    y = y - y.mean()
    d = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-6 else 0.0


# --- prototypes ----------------------------------------------------------------------------------------------------

class Proto:
    """One look of a class: canonical grey template, object mask and outline (canonical units), and how many picture
    pixels one canonical unit is at scale 1 (``unit``)."""

    def __init__(self, tmpl, mask, hull, unit, angle0=0.0, margin=0.0):
        self.tmpl = np.asarray(tmpl, np.float32)
        self.mask = (np.asarray(mask) > 0).astype(np.uint8)
        self.hull = np.asarray(hull, np.float32).reshape(-1, 2)
        self.unit = float(unit)
        self.angle0 = float(angle0)
        self.margin = float(margin)          # user box margin around the object (canonical units)
        self._sum = self.tmpl.astype(np.float64)
        self._n = 1

    @property
    def size(self):
        return self.tmpl.shape[1], self.tmpl.shape[0]

    @property
    def center(self):
        w, h = self.size
        return (w - 1) / 2.0, (h - 1) / 2.0

    def add_view(self, canon_gray: np.ndarray):
        """Average another aligned example into the template."""
        self._sum += canon_gray.astype(np.float64)
        self._n += 1
        self.tmpl = (self._sum / self._n).astype(np.float32)


def _object_mask(bgr, box) -> np.ndarray:
    """Object mask inside the user's box (bool, box size): GrabCut, filled convex hull; the whole box when GrabCut
    finds nothing sensible."""
    x, y, w, h = box
    full = np.ones((h, w), bool)
    H, W = bgr.shape[:2]
    pad = max(8, int(0.25 * max(w, h)))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    if x0 == x and y0 == y and x1 == x + w and y1 == y + h:
        return full                                       # no room outside the box for the background model
    crop = bgr[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    try:
        cv2.grabCut(crop, mask, (x - x0, y - y0, w, h), np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return full
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)[y - y0:y - y0 + h, x - x0:x - x0 + w]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return full
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = cv2.findNonZero((lab == big).astype(np.uint8))
    hull = cv2.convexHull(pts)
    out = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    frac = out.sum() / float(w * h)
    if frac < 0.25 or frac > 0.97:
        return full
    return out > 0


def _make_proto(bgr, box) -> Proto:
    x, y, w, h = box
    unit = max(1.0, max(w, h) / float(CANON_MAX))         # picture px per canonical unit
    cw, ch = max(4, int(round(w / unit))), max(4, int(round(h / unit)))
    crop = bgr[y:y + h, x:x + w]
    tmpl = cv2.resize(_gray(crop), (cw, ch), interpolation=cv2.INTER_AREA)
    obj = _object_mask(bgr, box).astype(np.uint8)
    m = cv2.resize(obj, (cw, ch), interpolation=cv2.INTER_NEAREST)
    pts = cv2.findNonZero(m)
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32) if pts is not None else \
        np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
    bx, by, bw, bh = cv2.boundingRect(m)
    margin = float(np.mean([bx, by, cw - bx - bw, ch - by - bh]))
    if margin < 0.04 * min(cw, ch):
        margin = 0.0
    # outer half pixel of the hull so its bounding box covers whole pixels
    c = hull.mean(0)
    d = hull - c
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    hull = hull + 0.5 * d / np.maximum(nrm, 1e-6)
    return Proto(tmpl, m, hull, unit, 0.0, margin)


# --- the search (stage 1) ------------------------------------------------------------------------------------------

class _Bank:
    """Rotated/scaled zero-mean templates of one prototype at one work resolution."""

    def __init__(self, proto: Proto, f: float, angles, scales):
        self.proto, self.f = proto, f
        self.angles = np.asarray(angles, np.float64)
        self.scales = np.asarray(scales, np.float64)
        w, h = proto.size
        # canonical template at roughly the work resolution (area-averaged once, then warped)
        k0 = proto.unit * f                                     # work px per canonical unit at scale 1
        sw, sh = max(3, int(round(w * k0))), max(3, int(round(h * k0)))
        self.small = cv2.GaussianBlur(cv2.resize(proto.tmpl, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), 0.6)
        self.small_mask = cv2.resize(proto.mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        self.kx, self.ky = sw / float(w), sh / float(h)
        self.items = []                                         # per scale: (D, [(T', norm, n) per angle])
        diag = math.hypot(sw, sh)
        sc = ((sw - 1) / 2.0, (sh - 1) / 2.0)
        for s in self.scales:
            D = int(math.ceil(diag * s)) + 2
            D += 1 - D % 2
            dc = ((D - 1) / 2.0, (D - 1) / 2.0)
            per = []
            for a in self.angles:
                m = _affine(a, s, sc, dc)
                t = cv2.warpAffine(self.small, m, (D, D), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                mk = cv2.warpAffine(self.small_mask.astype(np.float32), m, (D, D), flags=cv2.INTER_LINEAR) > 0.5
                n = int(mk.sum())
                if n < 6:
                    per.append(None)
                    continue
                t = np.where(mk, t - t[mk].mean(), 0).astype(np.float32)
                nr = float(np.sqrt((t * t).sum()))
                # pre-normalised: correlation / (|t| * sqrt(n)) only needs dividing by the local std afterwards
                per.append(np.ascontiguousarray(t / (nr * math.sqrt(n)), np.float32) if nr > 1e-3 else None)
            self.items.append((D, per))
        self.win = max(3, int(round(math.sqrt(max(9.0, float(self.small_mask.sum()))))))


def _scale_grid(lo: float, hi: float, ratio: float = 1.12) -> list[float]:
    lo, hi = max(0.05, min(lo, hi)), max(lo, hi)
    if hi / lo <= ratio ** 0.5:
        return [math.sqrt(lo * hi)]
    n = int(math.ceil(math.log(hi / lo) / math.log(ratio))) + 1
    return [float(v) for v in np.geomspace(lo, hi, n)]


def _angle_grid(rng: float, step: float) -> list[float]:
    rng = float(max(0.0, min(180.0, rng)))
    if rng <= 0:
        return [0.0]
    if rng >= 180.0:
        n = max(4, min(MAX_ANGLES, int(math.ceil(360.0 / step))))
        return [float(v) for v in np.linspace(-180.0, 180.0, n, endpoint=False)]
    n = max(1, int(math.ceil(rng / step)))
    return [float(v) for v in np.linspace(-rng, rng, 2 * n + 1)]


class _Work:
    """A picture prepared for the search at one work resolution."""

    def __init__(self, gray: np.ndarray, f: float):
        self.f = f
        g = gray if f >= 0.999 else cv2.resize(gray, (max(1, int(round(gray.shape[1] * f))),
                                                        max(1, int(round(gray.shape[0] * f)))),
                                               interpolation=cv2.INTER_AREA)
        self.img = cv2.GaussianBlur(g, (0, 0), 0.6)
        self._pad = {}
        self._std = {}

    def padded(self, c: int):
        """The picture with ``c`` replicated pixels on every side, so templates can be centred up to the edges."""
        if c not in self._pad:
            self._pad[c] = cv2.copyMakeBorder(self.img, c, c, c, c, cv2.BORDER_REPLICATE)
        return self._pad[c]

    def std(self, win: int):
        """Local standard deviation over a win x win window (+ a noise floor, so flat areas do not explode)."""
        if win not in self._std:
            m = cv2.blur(self.img, (win, win), borderType=cv2.BORDER_REPLICATE)
            m2 = cv2.blur(self.img * self.img, (win, win), borderType=cv2.BORDER_REPLICATE)
            self._std[win] = np.sqrt(np.maximum(m2 - m * m, 0) + 16.0)
        return self._std[win]


_POOL = None


def _pool_map(fn, items):
    """Map over a few worker threads (OpenCV releases the GIL in matchTemplate)."""
    global _POOL
    items = list(items)
    if len(items) < 2:
        return [fn(i) for i in items]
    if _POOL is None:
        import concurrent.futures
        import os
        _POOL = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2))),
                                                      thread_name_prefix="vf-fewshot-det")
    return list(_POOL.map(fn, items))


def _search_scale(bank: "_Bank", top_k: int, s: float, per, img, std):
    rs = []
    sm = None
    for item in per:
        if item is None:
            continue
        r = cv2.matchTemplate(img, item, cv2.TM_CCORR)
        rs.append(r)
        sm = r.copy() if sm is None else np.maximum(sm, r, out=sm)
    if sm is None:
        return []
    sm /= std
    k = max(3, int(round(bank.win * s * 0.6)))
    k += 1 - k % 2
    dil = cv2.dilate(sm, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((sm >= dil) & (sm >= COARSE_MIN))
    if not len(xs):
        return []
    sc = sm[ys, xs]
    order = np.argsort(-sc)[:top_k]
    ys, xs, sc = ys[order], xs[order], sc[order]
    ai = np.argmax(np.stack([r[ys, xs] for r in rs]), axis=0)          # the angle of each peak
    valid = [a for a, item in zip(bank.angles, per) if item is not None]
    return [(float(sc[i]), float(xs[i]), float(ys[i]), float(valid[ai[i]]), s) for i in range(len(xs))]


def _search(work: _Work, bank: _Bank, top_k: int):
    """Coarse correlation over all angles and scales; returns local maxima of every scale as
    (score, cx, cy, angle, scale) in work coordinates, best first. Peaks are taken per scale so that a part of the
    object matched at a smaller scale cannot hide the whole object."""
    jobs = []
    for si, (D, per) in enumerate(bank.items):
        s = float(bank.scales[si])
        c = (D - 1) // 2
        # caches filled here, not in the worker threads
        jobs.append((s, per, work.padded(c), work.std(max(3, int(round(bank.win * s))))))
    found = []
    for part in _pool_map(lambda j: _search_scale(bank, top_k, *j), jobs):
        found.extend(part)
    found.sort(key=lambda t: -t[0])
    # the same object peaks at neighbouring scales: keep one candidate per place and size
    size = math.sqrt(float(bank.small_mask.size))
    out = []
    for f in found:
        if all(math.hypot(f[1] - o[1], f[2] - o[2]) > 0.15 * size * min(f[4], o[4])
               or max(f[4], o[4]) / min(f[4], o[4]) > 1.3 for o in out):
            out.append(f)
            if len(out) >= top_k:
                break
    return out


def _dedupe(poses: list[dict], size_px: float) -> list[dict]:
    """Refined candidates that ended on the same object: keep the best correlation."""
    out = []
    for p in sorted(poses, key=lambda p: -p["ncc"]):
        if all(math.hypot(p["cx"] - o["cx"], p["cy"] - o["cy"]) > 0.2 * size_px * min(p["scale"], o["scale"])
               or max(p["scale"], o["scale"]) / min(p["scale"], o["scale"]) > 1.15 for o in out):
            out.append(p)
    return out


def _refine(work: _Work, bank: _Bank, cand, astep: float):
    """Exact masked NCC around a coarse candidate: a few angles and scales, each with a small shift search.
    Returns (ncc, cx, cy, angle, scale) in work coordinates."""
    _, cx, cy, a0, s0 = cand
    tm, mk = bank.small, bank.small_mask
    sh, sw = tm.shape
    sc = ((sw - 1) / 2.0, (sh - 1) / 2.0)
    m = 2
    big = (sw + 2 * m, sh + 2 * m)
    bc = (sc[0] + m, sc[1] + m)
    best = (-2.0, cx, cy, a0, s0)
    tried = {}

    def at(a, s):
        key = (round(a, 3), round(s, 4))
        if key in tried:
            return tried[key]
        reg = _warp_to_canon(work.img, a, s, (cx, cy), big, bc)
        r = cv2.matchTemplate(reg, tm, cv2.TM_CCOEFF_NORMED, mask=mk)
        cv2.patchNaNs(r, -1.0)
        _, mv, _, ml = cv2.minMaxLoc(r)
        du, dv = ml[0] - m, ml[1] - m                          # shift in canonical (work-scaled) units
        d = _rot(a, s) @ np.array([du, dv], np.float64)
        tried[key] = (float(mv), cx + d[0], cy + d[1], a, s)
        return tried[key]

    h = astep / 2.0
    vals = [at(a0 + da, s0) for da in (-h, 0.0, h)]
    i = int(np.argmax([v[0] for v in vals]))
    best = vals[i]
    if i == 1:                                                 # parabola through the three angle scores
        y0, y1, y2 = vals[0][0], vals[1][0], vals[2][0]
        den = y0 - 2 * y1 + y2
        if den < -1e-6:
            off = 0.5 * (y0 - y2) / den
            best = (best[0], best[1], best[2], a0 + float(np.clip(off, -1, 1)) * h, s0)
    a1 = best[3]
    for s in (s0 / 1.06, s0 * 1.06):
        v = at(a1, s)
        if v[0] > best[0]:
            best = v
    return best


# --- features (stage 2) --------------------------------------------------------------------------------------------

def _feat_size(proto: Proto):
    w, h = proto.size
    k = math.sqrt(FEAT_AREA / float(w * h))
    return max(16, int(round(w * k))), max(16, int(round(h * k)))


class _FeatCtx:
    def __init__(self, proto: Proto):
        self.proto = proto
        fw, fh = self.size = _feat_size(proto)
        m = cv2.resize(proto.mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        self.mask = m.astype(bool)
        self.mask_dil = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(np.float32)
        ys, xs = np.mgrid[0:fh, 0:fw]
        cell = (np.minimum(ys * HOG_CELLS // fh, HOG_CELLS - 1) * HOG_CELLS
                + np.minimum(xs * HOG_CELLS // fw, HOG_CELLS - 1))
        self.cell = cell.ravel()
        self.tmpl = cv2.resize(proto.tmpl, (fw, fh), interpolation=cv2.INTER_AREA)
        self.kx = proto.size[0] / float(fw)                    # canonical units per feature pixel


N_FEAT = HOG_CELLS * HOG_CELLS * HOG_BINS + 64 + 6 + 1
_H = HOG_CELLS * HOG_CELLS * HOG_BINS
GROUPS = ((0, _H), (_H, _H + 64), (_H + 64, _H + 70))      # shape (HOG), look (thumbnail), colour


class _Typical:
    """How far a candidate is from the taught examples, per feature group (shape, look, colour), in units of the
    examples' own spread. Unlike the verifier this needs no negatives, so it also rejects things that looked like
    nothing in the training pictures (e.g. the right outline in the wrong colour)."""

    def __init__(self, mu, sd, thr):
        self.mu, self.sd, self.thr = (np.asarray(v, np.float32) for v in (mu, sd, thr))

    @classmethod
    def fit(cls, pos: np.ndarray) -> "_Typical":
        mu = pos.mean(0)
        sd = pos.std(0)
        for a, b in GROUPS:
            sd[a:b] = np.maximum(sd[a:b], 0.3 * float(sd[a:b].mean()) + 1e-4)
        t = cls(mu, sd, np.ones(len(GROUPS)))
        d = t.dist(pos)
        t.thr = np.maximum(np.percentile(d, 98, axis=0), 1e-3).astype(np.float32)
        return t

    def dist(self, x: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(x) - self.mu) / self.sd
        return np.stack([np.sqrt((z[:, a:b] ** 2).mean(1)) for a, b in GROUPS], 1)

    def extra(self, x: np.ndarray) -> np.ndarray:
        """Features appended for the verifier: log(1 + distance / typical distance)."""
        return np.log1p(self.dist(x) / self.thr).astype(np.float32)

    def gate(self, x: np.ndarray) -> float:
        """1 while the candidate is about as typical as the examples, falling to 0 when twice as far as the
        typical distance."""
        ratio = float((self.dist(x)[0] / self.thr).max())
        return float(np.clip(2.5 - ratio, 0.0, 1.0))


def _patch(bgr, ctx: _FeatCtx, cx, cy, angle, scale):
    """Canonical BGR patch of a pose given in picture coordinates."""
    k = ctx.proto.unit * scale * ctx.kx                        # picture px per feature px
    fw, fh = ctx.size
    return _warp_to_canon(bgr, angle, k, (cx, cy), ctx.size, ((fw - 1) / 2.0, (fh - 1) / 2.0))


def _features(patch_bgr: np.ndarray, ctx: _FeatCtx) -> np.ndarray:
    g = _gray(patch_bgr)
    ncc = _masked_ncc(g, ctx.tmpl, ctx.mask)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) * ctx.mask_dil
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0)
    b = np.minimum((ang * (HOG_BINS / 180.0)).astype(np.int32), HOG_BINS - 1)
    hog = np.bincount(ctx.cell * HOG_BINS + b.ravel(), weights=mag.ravel(),
                      minlength=HOG_CELLS * HOG_CELLS * HOG_BINS)
    hog = hog / (np.linalg.norm(hog) + 1e-6)
    hog = np.minimum(hog, 0.2)
    hog = np.sqrt(hog / (np.linalg.norm(hog) + 1e-6))
    m = ctx.mask
    vals = g[m]
    mu, sd = float(vals.mean()), float(vals.std()) + 1.0
    th = np.where(m, (g - mu) / sd, 0).astype(np.float32)
    thumb = cv2.resize(th, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)[m] / 255.0
    col = np.concatenate([lab.mean(0), lab.std(0)])
    return np.concatenate([hog, thumb, col, [ncc]]).astype(np.float32)


# --- the verifier: ridge logistic regression -----------------------------------------------------------------------

def _fit_logistic(x: np.ndarray, y: np.ndarray, lam: float = 3.0, iters: int = 25):
    """Class-balanced, L2-regularised logistic regression by Newton steps. Returns (mean, std, w, b)."""
    mean = x.mean(0)
    std = x.std(0) + 1e-3
    z = (x - mean) / std
    n, d = z.shape
    npos, nneg = max(1, int(y.sum())), max(1, int(n - y.sum()))
    sw = np.where(y > 0, 0.5 * n / npos, 0.5 * n / nneg)
    za = np.hstack([z, np.ones((n, 1))]).astype(np.float64)
    w = np.zeros(d + 1)
    reg = np.full(d + 1, lam)
    reg[-1] = 1e-6
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(za @ w, -30, 30)))
        g = za.T @ (sw * (p - y)) + reg * w
        hdiag = sw * p * (1 - p)
        hm = (za * hdiag[:, None]).T @ za + np.diag(reg)
        step = np.linalg.solve(hm, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return mean.astype(np.float32), std.astype(np.float32), w[:-1].astype(np.float32), float(w[-1])


# --- the model -----------------------------------------------------------------------------------------------------

class DetModel:
    """A trained few-shot detector. Build it with :meth:`train`, use :meth:`detect`, keep it with :meth:`save` /
    :meth:`load`."""

    def __init__(self, classes, colors=None, angle_range=180.0, scale_range=(0.8, 1.2)):
        self.classes = [str(c) for c in classes]
        self.colors = [tuple(int(v) for v in c) for c in (colors or [default_color(i)
                                                                    for i in range(len(self.classes))])]
        self.angle_range = float(angle_range)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.protos: dict[int, list[Proto]] = {}
        self.search_scales: dict[int, tuple[float, float]] = {}
        self.clf: dict[int, tuple] = {}                        # class → (mean, std, w, b)
        self.typical: dict[int, _Typical] = {}
        self.info: dict = {}
        self._rt = None

    # ---- runtime ----
    @property
    def trained_classes(self) -> list[int]:
        return sorted(c for c in self.protos if c in self.clf)

    def _runtime(self):
        if self._rt is None:
            rt = {}
            for c, plist in self.protos.items():
                p0 = plist[0]
                w, h = p0.size
                f = min(1.0, WORK_SIZE / math.sqrt(w * h * p0.unit * p0.unit))
                lo, hi = self.search_scales.get(c, self.scale_range)
                banks, feats = [], []
                for p in plist:
                    pw, ph = p.size
                    r = 0.5 * math.hypot(pw, ph) * p.unit * f
                    step = float(np.clip(math.degrees(2.8 / max(r, 1.0)), 6.0, 20.0))
                    banks.append((_Bank(p, f, _angle_grid(self.angle_range, step), _scale_grid(lo, hi)), step))
                    feats.append(_FeatCtx(p))
                rt[c] = (f, banks, feats)
            self._rt = rt
        return self._rt

    def _candidates(self, bgr, gray, c: int, top_k: int, works: dict):
        """Refined, featurised candidates of one class: list of (feature vector, pose dict)."""
        f, banks, feats = self._runtime()[c]
        key = round(f, 4)
        if key not in works:
            works[key] = _Work(gray, f)
        work = works[key]
        poses = []
        for pi, (bank, step) in enumerate(banks):
            for cand in _search(work, bank, top_k):
                ncc, wx, wy, a, s = _refine(work, bank, cand, step)
                poses.append({"proto": pi, "cx": (wx + 0.5) / f - 0.5, "cy": (wy + 0.5) / f - 0.5, "angle": a, "scale": s, "ncc": ncc,
                              "coarse": cand[0]})
        p0 = self.protos[c][0]
        out = []
        for pose in _dedupe(poses, math.sqrt(p0.size[0] * p0.size[1]) * p0.unit)[:top_k]:
            ctx = feats[pose["proto"]]
            patch = _patch(bgr, ctx, pose["cx"], pose["cy"], pose["angle"], pose["scale"])
            out.append((_features(patch, ctx), pose))
        return out

    def _pose_box(self, c, pose, shape, with_hull: bool = False):
        p = self.protos[c][pose["proto"]]
        k = p.unit * pose["scale"]
        m = _affine(pose["angle"], k, p.center, (pose["cx"], pose["cy"]))
        hull = p.hull @ m[:, :2].T + m[:, 2]
        x0, y0 = hull.min(0)
        x1, y1 = hull.max(0)
        if p.margin > 0:
            e = p.margin * k
            x0, y0, x1, y1 = x0 - e, y0 - e, x1 + e, y1 + e
        H, W = shape[:2]
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(W), x1), min(float(H), y1)
        w, h = p.size
        corners = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
        poly = corners @ m[:, :2].T + m[:, 2]
        box = (int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))
        return (box, poly, hull) if with_hull else (box, poly)

    def _score(self, c, feat):
        mean, std, w, b = self.clf[c]
        typ = self.typical[c]
        x = np.concatenate([feat, typ.extra(feat)[0]])
        z = float(((x - mean) / std) @ w + b)
        return typ.gate(feat) / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def detect(self, img, score: float = 0.5, max_objects: int = 100, classes=None, iou: float = 0.45) -> list[dict]:
        """Find the taught objects in a picture (any depth / channels). ``classes`` = names or numbers to look for
        (None = all). Returns detections sorted by score (highest first)."""
        self._check()
        bgr = to_bgr8(img)
        gray = _gray(bgr)
        want = self._class_filter(classes)
        max_objects = max(0, int(max_objects))
        if not want or max_objects == 0:
            return []
        works, dets = {}, []
        H, W = bgr.shape[:2]
        for c in want:
            p0 = self.protos[c][0]
            room = H * W / max(1.0, p0.size[0] * p0.size[1] * p0.unit * p0.unit * self.search_scales[c][0] ** 2)
            top_k = int(min(2 * max_objects + 10, 0.5 * room + 10, 200))
            for feat, pose in self._candidates(bgr, gray, c, top_k, works):
                if pose["ncc"] < 0.1:
                    continue
                sc = self._score(c, feat)
                if sc < score:
                    continue
                box, poly, hull = self._pose_box(c, pose, bgr.shape, True)
                p = self.protos[c][pose["proto"]]
                w, h = p.size
                k = p.unit * pose["scale"]
                dets.append({"box": box, "score": round(sc, 4), "class_id": c, "label": self.classes[c],
                             "angle": round(_norm_angle(pose["angle"] + p.angle0), 2),
                             "poly": [(float(px), float(py)) for px, py in poly],
                             "center": (float(pose["cx"]), float(pose["cy"])),
                             "size": (float(w * k), float(h * k)), "match": round(pose["ncc"], 4),
                             "_hull": hull.astype(np.float32)})
        out = nms(dets, iou)[:max_objects]
        for d in out:
            d.pop("_hull", None)
        return out

    def _class_filter(self, classes) -> list[int]:
        trained = self.trained_classes
        if classes is None or (isinstance(classes, str) and not classes.strip()):
            return trained
        items = [s.strip() for s in classes.split(",")] if isinstance(classes, str) else list(classes)
        low = [n.casefold() for n in self.classes]
        out = []
        for it in items:
            if isinstance(it, (int, np.integer)) or (isinstance(it, str) and it.isdigit()
                                                     and it.casefold() not in low):
                i = int(it)
            elif str(it).casefold() in low:
                i = low.index(str(it).casefold())
            elif str(it) == "":
                continue
            else:
                raise ValueError(f"'{it}' sınıfı modelde yok. Sınıflar: {', '.join(self.classes)}")
            if not 0 <= i < len(self.classes):
                raise ValueError(f"Sınıf numarası {i} geçersiz (0–{len(self.classes) - 1})")
            if i in trained and i not in out:
                out.append(i)
        return out

    # ---- training ----
    @classmethod
    def train(cls, images, boxes, classes, colors=None, angle_range: float = 180.0, scale_range=(0.8, 1.2),
              seed: int = 0, progress=None) -> "DetModel":
        """``images``: pictures; ``boxes``: per picture a list of (x, y, w, h, class index); ``classes``: names.
        Raises ValueError (Turkish) for unusable input."""
        t0 = time.perf_counter()
        if not classes:
            raise ValueError("En az bir sınıf olmalı")
        if len(classes) > MAX_CLASSES:
            raise ValueError(f"En çok {MAX_CLASSES} sınıf olabilir")
        if images is None or len(images) == 0:
            raise ValueError("Eğitim için önce resim ekleyin")
        if boxes is None or len(boxes) != len(images):
            raise ValueError("Her resim için bir kutu listesi gerekli")
        lo, hi = float(scale_range[0]), float(scale_range[1])
        if not (0.1 <= lo <= 1.0 <= hi <= 10.0):
            raise ValueError("Ölçek aralığı 1'i içermeli (ör. 0.8–1.2)")
        bgrs = [to_bgr8(im, "Eğitim resmi") for im in images]
        gt: list[list[tuple]] = []
        for bi, (bgr, bl) in enumerate(zip(bgrs, boxes)):
            H, W = bgr.shape[:2]
            row = []
            for b in bl or []:
                if len(b) < 5:
                    raise ValueError("Kutu (x, y, w, h, sınıf) olmalı")
                x, y, w, h = (int(round(float(v))) for v in b[:4])
                c = int(b[4])
                if not 0 <= c < len(classes):
                    raise ValueError(f"Geçersiz sınıf numarası: {c}")
                x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
                    raise ValueError(f"{bi + 1}. resimdeki kutu çok küçük ({w}x{h}); en az {MIN_BOX}x{MIN_BOX} "
                                     "piksel olmalı")
                row.append((x0, y0, x1 - x0, y1 - y0, c))
            gt.append(row)
        if not any(gt):
            raise ValueError("Hiç kutu çizilmedi: öğretmek için resimlerde nesnelerin etrafına kutu çizin")
        model = cls(classes, colors, angle_range, (lo, hi))
        rng = np.random.default_rng(seed)
        present = sorted({b[4] for row in gt for b in row})
        stats = {}
        # 1) prototypes and aligned example poses
        poses: dict[int, list] = {}
        for c in present:
            ex = [(i, b[:4]) for i, row in enumerate(gt) for b in row if b[4] == c]
            i0, b0 = ex[0]
            plist = [_make_proto(bgrs[i0], b0)]
            model.protos[c] = plist
            ex_poses = [(i0, b0, 0, b0[0] + (b0[2] - 1) / 2.0, b0[1] + (b0[3] - 1) / 2.0, 0.0, 1.0, 1.0)]
            ex_scales = [1.0]
            for i, b in ex[1:]:
                pose = model._align(c, bgrs[i], b)
                if pose is None or pose[4] < 0.5:
                    if len(plist) < MAX_PROTOS:
                        p = _make_proto(bgrs[i], b)
                        if pose is not None and pose[4] >= 0.3 and pose[0] == 0:
                            p.angle0 = pose[3]
                        plist.append(p)
                        model._rt = None
                        ex_poses.append((i, b, len(plist) - 1, b[0] + (b[2] - 1) / 2.0, b[1] + (b[3] - 1) / 2.0,
                                         0.0, 1.0, 1.0))
                        continue
                    if pose is None:
                        continue
                pi, cx, cy, a, ncc, s = pose
                ex_poses.append((i, b, pi, cx, cy, a, s, ncc))
                if pi == 0:
                    ex_scales.append(s)
                if ncc >= 0.5:
                    p = plist[pi]
                    p.add_view(_warp_to_canon(_gray(bgrs[i]), a, p.unit * s, (cx, cy), p.size, p.center))
                    model._rt = None
            # scale 1 = the examples' typical size; the range covers the examples and the user's range around it
            gm = float(np.exp(np.mean(np.log(ex_scales))))
            plist[0].unit *= gm
            ex_poses = [e if e[2] != 0 else (*e[:6], e[6] / gm, e[7]) for e in ex_poses]
            rel = [v / gm for v in ex_scales]
            model.search_scales[c] = (min(min(rel) * 0.95, lo), max(max(rel) * 1.05, hi))
            poses[c] = ex_poses
            model._rt = None
        # 2) verifier training data
        if progress:
            progress(0.4)
        xs = {c: [] for c in present}
        ys = {c: [] for c in present}
        for i, bgr in enumerate(bgrs):
            gray = _gray(bgr)
            works = {}
            for c in present:
                f, banks, feats = model._runtime()[c]
                mine = [b[:4] for b in gt[i] if b[4] == c]
                others = [b[:4] for b in gt[i] if b[4] != c]
                for feat, pose in model._candidates(bgr, gray, c, 60, works):
                    box, _ = model._pose_box(c, pose, bgr.shape)
                    best = max([box_iou(box, g) for g in mine], default=0.0)
                    if best >= 0.6:
                        xs[c].append(feat)
                        ys[c].append(1)
                    elif best < 0.5:                         # also parts of an object matched at a smaller size
                        xs[c].append(feat)
                        ys[c].append(0)
                # random patches away from this class's objects, and the other classes' objects
                H, W = bgr.shape[:2]
                p0 = model.protos[c][0]
                ctx0 = feats[0]
                lo_s, hi_s = model.search_scales[c]
                for _ in range(40):
                    s = float(rng.uniform(lo_s, hi_s))
                    a = float(rng.uniform(-model.angle_range, model.angle_range))
                    cx, cy = float(rng.uniform(0, W)), float(rng.uniform(0, H))
                    pose = {"proto": 0, "cx": cx, "cy": cy, "angle": a, "scale": s}
                    box, _ = model._pose_box(c, pose, bgr.shape)
                    if max([box_iou(box, g) for g in mine], default=0.0) > 0.2:
                        continue
                    xs[c].append(_features(_patch(bgr, ctx0, cx, cy, a, s), ctx0))
                    ys[c].append(0)
                for g in others:
                    for _ in range(4):
                        cx = g[0] + g[2] / 2.0 + rng.uniform(-0.1, 0.1) * g[2]
                        cy = g[1] + g[3] / 2.0 + rng.uniform(-0.1, 0.1) * g[3]
                        s = math.sqrt(g[2] * g[3]) / math.sqrt(p0.size[0] * p0.size[1]) / p0.unit
                        a = float(rng.uniform(-model.angle_range, model.angle_range))
                        xs[c].append(_features(_patch(bgr, ctx0, cx, cy, a, s), ctx0))
                        ys[c].append(0)
            if progress:
                progress(0.4 + 0.5 * (i + 1) / len(bgrs))
        # jittered positives from the examples' own poses
        for c in present:
            _, banks, feats = model._runtime()[c]
            for (i, b, pi, cx, cy, a, s, ncc0) in poses[c]:
                ctx = feats[pi]
                step = banks[pi][1]
                for j in range(24):
                    if j == 0:
                        da, ds, dx, dy = 0.0, 1.0, 0.0, 0.0
                    else:                                       # about the search's own pose error
                        da = float(rng.uniform(-0.35, 0.35) * step)
                        ds = float(np.exp(rng.uniform(-0.04, 0.04)))
                        dx, dy = (float(v) for v in rng.uniform(-0.03, 0.03, 2) * math.sqrt(b[2] * b[3]))
                    patch = _patch(bgrs[i], ctx, cx + dx, cy + dy, a + da, s * ds).astype(np.float32)
                    if j:
                        patch = patch * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-20, 20))
                        patch = patch + rng.normal(0, 3, patch.shape).astype(np.float32)
                    xs[c].append(_features(np.clip(patch, 0, 255).astype(np.uint8), ctx))
                    ys[c].append(1)
                # half-off poses (part of the object + its surroundings) are the commonest false alarm
                size = math.sqrt(b[2] * b[3])
                for _ in range(8):
                    t = float(rng.uniform(0, 2 * math.pi))
                    r = float(rng.uniform(0.45, 0.8)) * size
                    pose = {"proto": pi, "cx": cx + r * math.cos(t), "cy": cy + r * math.sin(t),
                            "angle": a + float(rng.uniform(-30, 30)), "scale": s * float(np.exp(rng.uniform(-.2, .2)))}
                    box, _ = model._pose_box(c, pose, bgrs[i].shape)
                    if box_iou(box, b) < 0.4:
                        xs[c].append(_features(_patch(bgrs[i], ctx, pose["cx"], pose["cy"], pose["angle"],
                                                      pose["scale"]), ctx))
                        ys[c].append(0)
        for c in present:
            x = np.asarray(xs[c], np.float32)
            y = np.asarray(ys[c], np.float64)
            if y.min() == y.max():                             # no negatives at all: add plain noise patches
                ctx = model._runtime()[c][2][0]
                fw, fh = ctx.size
                extra = [_features(rng.integers(0, 256, (fh, fw, 3)).astype(np.uint8), ctx)
                         for _ in range(20)]
                x = np.vstack([x, np.asarray(extra, np.float32)])
                y = np.concatenate([y, np.zeros(len(extra))])
            model.typical[c] = _Typical.fit(x[y > 0])
            model.clf[c] = _fit_logistic(np.hstack([x, model.typical[c].extra(x)]), y)
            pred = np.array([model._score(c, f) for f in x]) >= 0.5
            stats[model.classes[c]] = {"examples": len(poses[c]), "prototypes": len(model.protos[c]),
                                       "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
                                       "train_accuracy": round(float((pred == (y > 0)).mean()), 4)}
        model.info = {"images": len(bgrs), "boxes": sum(len(r) for r in gt), "classes": stats,
                      "seconds": round(time.perf_counter() - t0, 2)}
        if progress:
            progress(1.0)
        return model

    def _align(self, c, bgr, box):
        """Find an example box with the class's current prototypes: (proto, cx, cy, angle, ncc, scale) or None."""
        x, y, w, h = box
        H, W = bgr.shape[:2]
        pad = int(0.5 * max(w, h)) + 4
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        crop = bgr[y0:y1, x0:x1]
        gray = _gray(crop)
        lo, hi = self.scale_range
        saved = self.search_scales.get(c)
        self.search_scales[c] = (lo * 0.85, hi * 1.15)         # examples may differ in size more than test parts
        self._rt = None
        try:
            f, banks, _ = self._runtime()[c]
            work = _Work(gray, f)
            gcx, gcy = x + (w - 1) / 2.0 - x0, y + (h - 1) / 2.0 - y0
            best = None
            for pi, (bank, step) in enumerate(banks):
                for cand in _search(work, bank, 10):
                    ncc, wx, wy, a, s = _refine(work, bank, cand, step)
                    cx, cy = (wx + 0.5) / f - 0.5, (wy + 0.5) / f - 0.5
                    if abs(cx - gcx) > 0.3 * w or abs(cy - gcy) > 0.3 * h:
                        continue
                    if best is None or ncc > best[4]:
                        best = (pi, cx + x0, cy + y0, a, ncc, s)
        finally:
            if saved is None:
                self.search_scales.pop(c, None)
            else:
                self.search_scales[c] = saved
            self._rt = None
        return best

    # ---- persistence ----
    def _check(self):
        if not self.trained_classes:
            raise ValueError("Model eğitilmemiş")

    def header(self) -> dict:
        protos = {str(c): [{"unit": p.unit, "angle0": p.angle0, "margin": p.margin, "hull": p.hull.tolist()}
                           for p in plist] for c, plist in self.protos.items()}
        return {"format": FORMAT, "version": VERSION, "classes": self.classes,
                "colors": [list(c) for c in self.colors], "angle_range": self.angle_range,
                "scale_range": list(self.scale_range),
                "search_scales": {str(c): list(v) for c, v in self.search_scales.items()},
                "protos": protos, "clf_bias": {str(c): v[3] for c, v in self.clf.items()}, "info": self.info,
                "n_features": N_FEAT}

    def save(self, path) -> str:
        self._check()
        path = pathlib.Path(path)
        if path.suffix.lower() != EXT:
            path = path.with_name(path.name + EXT)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"header": np.frombuffer(json.dumps(self.header(), ensure_ascii=False).encode(), np.uint8)}
        for c, plist in self.protos.items():
            for i, p in enumerate(plist):
                arrays[f"t_{c}_{i}"] = p.tmpl
                arrays[f"m_{c}_{i}"] = p.mask
        for c, (mean, std, w, _b) in self.clf.items():
            arrays[f"mean_{c}"], arrays[f"std_{c}"], arrays[f"w_{c}"] = mean, std, w
            t = self.typical[c]
            arrays[f"tmu_{c}"], arrays[f"tsd_{c}"], arrays[f"tthr_{c}"] = t.mu, t.sd, t.thr
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)                     # a running block never reads a half-written model
        return str(path)

    @classmethod
    def load(cls, path) -> "DetModel":
        path = pathlib.Path(path)
        if not path.is_file():
            raise ValueError(f"Model dosyası yok: {path}")
        try:
            with np.load(path, allow_pickle=False) as z:
                head = json.loads(bytes(z["header"]).decode()) if "header" in z.files else {}
                if not isinstance(head, dict) or head.get("format") != FORMAT:
                    raise TypeError("biçim")
                arrays = {k: z[k] for k in z.files if k != "header"}
        except TypeError:
            raise ValueError(f"Bu dosya bir Az Örnekle Nesne Bul modeli değil: {path.name}") from None
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Model dosyası okunamadı ({path.name}): {exc}") from exc
        if int(head.get("version", 0)) > VERSION:
            raise ValueError("Model daha yeni bir sürümle kaydedilmiş; programı güncelleyin")
        try:
            m = cls(head["classes"], head.get("colors"), head.get("angle_range", 180.0),
                    tuple(head.get("scale_range", (0.8, 1.2))))
            m.search_scales = {int(c): (float(v[0]), float(v[1])) for c, v in head.get("search_scales", {}).items()}
            for c, plist in head["protos"].items():
                c = int(c)
                m.protos[c] = [Proto(arrays[f"t_{c}_{i}"], arrays[f"m_{c}_{i}"], p["hull"], p["unit"],
                                     p.get("angle0", 0.0), p.get("margin", 0.0)) for i, p in enumerate(plist)]
            for c, b in head.get("clf_bias", {}).items():
                c = int(c)
                m.clf[c] = (arrays[f"mean_{c}"].astype(np.float32), arrays[f"std_{c}"].astype(np.float32),
                            arrays[f"w_{c}"].astype(np.float32), float(b))
                m.typical[c] = _Typical(arrays[f"tmu_{c}"], arrays[f"tsd_{c}"], arrays[f"tthr_{c}"])
                if m.clf[c][2].shape != (N_FEAT + len(GROUPS),) or m.typical[c].mu.shape != (N_FEAT,):
                    raise ValueError("özellik sayısı")
            m.info = head.get("info", {})
            if not m.trained_classes or not all(0 <= c < len(m.classes) for c in m.protos):
                raise ValueError("sınıf")
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"Model dosyası bozuk: {path.name}") from exc
        return m


def _inside(a, b) -> float:
    """Share of box a that lies inside box b."""
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / max(1.0, float(a[2] * a[3]))


def _hull_overlap(a: dict, b: dict) -> float:
    """Overlap of two detections' object outlines, as a share of the smaller one (0 when unknown). Two parts cannot
    lie on top of each other, so a candidate covering much of a better one is a false alarm (e.g. half a part plus
    its neighbour)."""
    ha, hb = a.get("_hull"), b.get("_hull")
    if ha is None or hb is None:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(ha, hb)
    except cv2.error:
        return 0.0
    small = min(abs(cv2.contourArea(ha)), abs(cv2.contourArea(hb)))
    return float(inter) / small if small > 0 else 0.0


def nms(dets: list[dict], iou: float = 0.45) -> list[dict]:
    """Greedy class-agnostic non-maximum suppression on the axis-aligned boxes. Ranked by score and, among (nearly)
    sure detections, by the template match, so a whole object wins over a part of it matched at a smaller size; a box
    lying mostly inside a better one is dropped too."""
    out = []
    for d in sorted(dets, key=lambda d: -(round(d["score"], 2) + 0.05 * d.get("match", 0.0))):
        if all(box_iou(d["box"], o["box"]) < iou and _inside(d["box"], o["box"]) < 0.7
               and _hull_overlap(d, o) < 0.3 for o in out):
            out.append(d)
    return sorted(out, key=lambda d: -d["score"])


def draw(img, dets, colors=None, thickness: int = 2) -> np.ndarray:
    """BGR picture with the detections drawn (rotated box + label)."""
    out = to_bgr8(img).copy()
    for d in dets:
        c = int(d.get("class_id", 0))
        col = tuple(int(v) for v in (colors[c] if colors and c < len(colors) else default_color(c)))
        if d.get("poly"):
            cv2.polylines(out, [np.round(np.asarray(d["poly"])).astype(np.int32)], True, col, thickness)
        x, y, w, h = d["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 1)
        cv2.putText(out, f"{d['label']} {d['score']:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
    return out
