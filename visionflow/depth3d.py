"""3D / depth vision helpers (numpy + OpenCV only): depth maps, camera intrinsics, point clouds, ground plane fit,
heights, volume, a synthetic test scene, point cloud files (.ply / .npy / 16-bit PNG), depth camera drivers
(Intel RealSense through pyrealsense2, Orbbec through pyorbbecsdk; both optional and imported only when used) and
a small software point cloud renderer used by the 3D viewer window.

Conventions
- A depth map is float32, millimetres, NaN where the camera has no measurement (0 and negative values are invalid).
- Intrinsics are a dict {"fx", "fy", "cx", "cy", "width", "height"} in pixels (pinhole camera, no distortion).
- Camera frame: x right, y down, z forward (the viewing direction), millimetres.
- A plane is a dict {"normal": [nx, ny, nz], "d": d, ...} with n·X + d = 0, |n| = 1 and d > 0, so that
  n·X + d is the signed height above the plane, positive towards the camera.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import pathlib
import threading

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------- depth and intrinsics


class Intrinsics(dict):
    """Camera intrinsics: a plain dict (saved / sent as JSON as it is) that prints short on the block."""

    def __str__(self):
        return f"fx={self['fx']:.1f} fy={self['fy']:.1f} cx={self['cx']:.1f} cy={self['cy']:.1f}"


class Plane(dict):
    """A fitted plane: a plain dict that prints short on the block."""

    def __str__(self):
        return f"eğim {self['tilt']:.2f}°, uzaklık {self['d']:.1f} mm"


def as_depth(img, what: str = "Derinlik") -> np.ndarray:
    """Any depth-like array (uint16 mm, float mm, HxWx1) → float32 mm with NaN for invalid; Turkish error for
    anything that is not a one-channel picture."""
    if img is None:
        raise ValueError(f"{what} girişi bağlı değil")
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{what} girişine derinlik haritası bağlanmalı (gelen: {type(img).__name__})")
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim != 2:
        raise ValueError(f"{what} tek kanallı olmalı (gelen boyut {img.shape}; renkli resim mi bağlandı?)")
    if img.size == 0 or min(img.shape) < 2:
        raise ValueError(f"{what} boş veya çok küçük: {img.shape}")
    if img.dtype == np.bool_ or img.dtype.kind not in "uif":
        raise ValueError(f"{what} sayı dizisi olmalı (gelen tür {img.dtype})")
    d = img.astype(np.float32, copy=True)
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d


def make_intrinsics(width: int, height: int, hfov_deg: float = 60.0) -> dict:
    """A pinhole camera with the given horizontal field of view and the principal point in the middle."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(fx=fx, fy=fx, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0, width=int(width),
                      height=int(height))


def intrinsics(obj, shape=None) -> dict:
    """Checked intrinsics from a dict (fx, fy, cx, cy; RealSense names ppx/ppy too) or a list of 4 numbers.
    With ``shape`` and a known width/height of another size (the depth map was resized) the values are scaled."""
    if obj is None:
        raise ValueError("Kamera iç parametreleri (intrinsics) bağlı değil: Derinlik Kamerası veya Derinlik Test "
                         "Sahnesi bloğunun 'intrinsics' çıkışını bağlayın")
    try:
        if isinstance(obj, dict):
            fx, fy = float(obj["fx"]), float(obj["fy"])
            cx = float(obj["cx"] if "cx" in obj else obj["ppx"])
            cy = float(obj["cy"] if "cy" in obj else obj["ppy"])
            w, h = int(obj.get("width") or 0), int(obj.get("height") or 0)
        elif isinstance(obj, (list, tuple)) and len(obj) in (4, 6):
            fx, fy, cx, cy = (float(v) for v in obj[:4])
            w, h = (int(obj[4]), int(obj[5])) if len(obj) == 6 else (0, 0)
        else:
            raise TypeError
    except (KeyError, TypeError, ValueError):
        raise ValueError("Kamera iç parametreleri anlaşılmadı: fx, fy, cx, cy gerekli") from None
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError(f"Kamera iç parametreleri geçersiz: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    out = Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
    if shape is not None:
        H, W = shape[:2]
        if w > 0 and h > 0 and (w, h) != (W, H):
            sx, sy = W / w, H / h
            out.update(fx=fx * sx, fy=fy * sy, cx=(cx + 0.5) * sx - 0.5, cy=(cy + 0.5) * sy - 0.5, width=W, height=H)
        elif not w or not h:
            out.update(width=W, height=H)
    return out


def rays(intr: dict, shape) -> np.ndarray:
    """HxWx3 ray directions (x/z, y/z, 1) of every pixel."""
    H, W = shape[:2]
    u = (np.arange(W, dtype=np.float32) - intr["cx"]) / intr["fx"]
    v = (np.arange(H, dtype=np.float32) - intr["cy"]) / intr["fy"]
    r = np.empty((H, W, 3), np.float32)
    r[:, :, 0] = u[None, :]
    r[:, :, 1] = v[:, None]
    r[:, :, 2] = 1.0
    return r


def depth_to_xyz(depth: np.ndarray, intr: dict) -> np.ndarray:
    """Organized point cloud HxWx3 (mm, NaN where the depth is invalid)."""
    return rays(intr, depth.shape) * depth[:, :, None]


def depth_to_points(depth: np.ndarray, intr: dict, mask=None, step: int = 1):
    """Valid points as (N, 3) float32 and their pixel (rows, cols)."""
    step = max(1, int(step))
    valid = np.isfinite(depth)
    if mask is not None:
        valid &= mask
    if step > 1:
        sub = np.zeros_like(valid)
        sub[::step, ::step] = True
        valid &= sub
    vs, us = np.nonzero(valid)
    z = depth[vs, us]
    pts = np.stack([(us - intr["cx"]) / intr["fx"] * z, (vs - intr["cy"]) / intr["fy"] * z, z], 1)
    return pts.astype(np.float32), (vs, us)


def points_to_depth(points: np.ndarray, intr: dict, width: int, height: int, colors=None):
    """Project a point cloud into a depth map (nearest point wins) and, with colours, a BGR picture."""
    depth = np.full((height, width), np.nan, np.float32)
    img = np.zeros((height, width, 3), np.uint8) if colors is not None else None
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    ok = np.isfinite(pts).all(1) & (pts[:, 2] > 0)
    pts = pts[ok]
    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3)[ok]
    if not len(pts):
        return depth, img
    u = np.round(pts[:, 0] / pts[:, 2] * intr["fx"] + intr["cx"]).astype(np.int64)
    v = np.round(pts[:, 1] / pts[:, 2] * intr["fy"] + intr["cy"]).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], pts[inside, 2]
    order = np.argsort(z, kind="stable")                    # nearest first: np.unique keeps the first one
    idx = v[order] * width + u[order]
    _, first = np.unique(idx, return_index=True)
    sel = order[first]
    depth[v[sel], u[sel]] = z[sel]
    if img is not None:
        img[v[sel], u[sel]] = colors[inside][sel]
    return depth, img


# ------------------------------------------------------------------------------------------------------------ planes


def plane_dict(n, d, **extra) -> dict:
    n = np.asarray(n, np.float64)
    norm = float(np.linalg.norm(n))
    n, d = n / norm, float(d) / norm
    if d < 0:                                                # the camera side positive
        n, d = -n, -d
    out = Plane(normal=[float(v) for v in n], d=d, tilt=tilt_deg(n))
    out.update(extra)
    return out


def as_plane(obj, what: str = "Düzlem") -> tuple[np.ndarray, float]:
    """(unit normal, d) of a plane dict; Turkish error for anything else."""
    if obj is None:
        raise ValueError(f"{what} girişi bağlı değil: Zemin Düzlemi Bul bloğunun 'plane' çıkışını bağlayın")
    try:
        n = np.asarray(obj["normal"], np.float64).reshape(3)
        d = float(obj["d"])
    except (KeyError, TypeError, ValueError, IndexError):
        raise ValueError(f"{what} girişi anlaşılmadı (Zemin Düzlemi Bul çıkışı bekleniyor)") from None
    norm = float(np.linalg.norm(n))
    if not np.isfinite(n).all() or not math.isfinite(d) or norm < 1e-9:
        raise ValueError(f"{what} geçersiz")
    n, d = n / norm, d / norm
    if d < 0:
        n, d = -n, -d
    return n, d


def tilt_deg(n) -> float:
    """Angle between the plane normal and the camera's viewing axis (0° = the camera looks straight at it)."""
    n = np.asarray(n, np.float64)
    c = abs(float(n[2])) / max(float(np.linalg.norm(n)), 1e-12)
    return math.degrees(math.acos(min(1.0, c)))


def fit_plane_lsq(pts: np.ndarray):
    """Least-squares plane through points: (unit normal, d)."""
    pts = np.asarray(pts, np.float64)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    d = -float(n @ c)
    if d < 0:
        n, d = -n, -d
    return n, d


def ransac_plane(pts: np.ndarray, threshold: float = 3.0, iterations: int = 200, seed: int = 0,
                 max_points: int = 20000):
    """Dominant plane of a point cloud: RANSAC on a sample, then least squares over every inlier (twice).
    Returns (normal, d, inlier mask over pts)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 3:
        raise ValueError("Düzlem için yeterli geçerli nokta yok (en az 3)")
    rng = np.random.default_rng(seed)
    sample = pts if len(pts) <= max_points else pts[rng.choice(len(pts), max_points, replace=False)]
    iterations = max(1, int(iterations))
    tri = rng.integers(0, len(sample), size=(iterations, 3))
    a, b, c = sample[tri[:, 0]], sample[tri[:, 1]], sample[tri[:, 2]]
    normals = np.cross(b - a, c - a)
    lens = np.linalg.norm(normals, axis=1)
    good = lens > 1e-9
    if not good.any():
        raise ValueError("Noktalar bir düzlem belirtmiyor (hepsi bir doğru üzerinde)")
    normals = normals[good] / lens[good, None]
    ds = -np.einsum("ij,ij->i", normals, a[good])
    best, best_count = 0, -1
    for start in range(0, len(normals), 64):                 # score in chunks: (64, N) at a time
        dist = np.abs(sample @ normals[start:start + 64].T + ds[start:start + 64])
        counts = (dist <= threshold).sum(0)
        k = int(counts.argmax())
        if counts[k] > best_count:
            best, best_count = start + k, int(counts[k])
    n, d = normals[best], float(ds[best])
    inl = np.abs(pts @ n + d) <= threshold
    for _ in range(2):
        if inl.sum() < 3:
            break
        n, d = fit_plane_lsq(pts[inl])
        inl = np.abs(pts @ n + d) <= threshold
    if d < 0:
        n, d = -n, -d
    return n, float(d), inl


def height_map(depth: np.ndarray, intr: dict, plane) -> np.ndarray:
    """Signed distance of every pixel's 3D point to the plane (mm, + towards the camera), NaN where invalid."""
    n, d = as_plane(plane) if isinstance(plane, dict) else plane
    r = rays(intr, depth.shape)
    h = (r @ np.asarray(n, np.float32)) * depth + np.float32(d)
    return h.astype(np.float32)


def plane_basis(n) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors in the plane (e1 roughly along the camera x axis)."""
    n = np.asarray(n, np.float64)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def pixel_areas(depth: np.ndarray, intr: dict, plane) -> np.ndarray:
    """Area (mm²) each pixel's surface patch covers when projected onto the plane.

    The patch is spanned by the differences to the right/left and lower/upper neighbour, taking for each
    direction the neighbour on the same surface (smaller depth step), so occlusion edges do not produce huge
    patches and walls standing on the plane project to ~0. Where no neighbour is valid the patch is assumed
    parallel to the plane. Summing height × area over pixels integrates the volume above the plane."""
    n, d = as_plane(plane) if isinstance(plane, dict) else plane
    e1, e2 = plane_basis(n)
    xyz = depth_to_xyz(depth, intr).astype(np.float64)
    s = xyz @ e1
    t = xyz @ e2
    z = depth.astype(np.float64)
    jump = np.maximum(3.0, 4.0 * z / math.sqrt(intr["fx"] * intr["fy"]))    # mm: a step bigger than this is an edge
    jump = np.where(np.isfinite(jump), jump, 3.0)

    def pick(arr, axis):
        fwd = np.full_like(arr, np.nan)
        bwd = np.full_like(arr, np.nan)
        zf = np.full_like(z, np.inf)
        zb = np.full_like(z, np.inf)
        if axis == 1:
            fwd[:, :-1] = arr[:, 1:] - arr[:, :-1]
            bwd[:, 1:] = arr[:, 1:] - arr[:, :-1]
            zf[:, :-1] = np.abs(z[:, 1:] - z[:, :-1])
            zb[:, 1:] = np.abs(z[:, 1:] - z[:, :-1])
        else:
            fwd[:-1] = arr[1:] - arr[:-1]
            bwd[1:] = arr[1:] - arr[:-1]
            zf[:-1] = np.abs(z[1:] - z[:-1])
            zb[1:] = np.abs(z[1:] - z[:-1])
        zf = np.where(np.isnan(zf), np.inf, zf)
        zb = np.where(np.isnan(zb), np.inf, zb)
        # the forward neighbour unless it is across a depth jump and the backward one is not (choosing the
        # smaller step every time would favour noise that shrinks the patch and bias the area low)
        use_bwd = (zf > jump) & (zb < zf)
        return np.where(use_bwd, bwd, fwd)

    su, tu = pick(s, 1), pick(t, 1)
    sv, tv = pick(s, 0), pick(t, 0)
    area = np.abs(su * tv - tu * sv)
    # no usable neighbour: the footprint of a patch parallel to the plane
    r = rays(intr, depth.shape).astype(np.float64)
    rn = np.linalg.norm(r, axis=2)
    cos_plane = np.abs(r @ n) / rn
    fallback = z * z / (intr["fx"] * intr["fy"]) * (1.0 / rn) / np.maximum(cos_plane, 0.05)
    bad = ~np.isfinite(area)
    area[bad] = fallback[bad]
    area[~np.isfinite(z)] = np.nan
    area[_under_higher(xyz @ n + d, s, t, z, intr)] = 0.0
    return area.astype(np.float32)


def _under_higher(h: np.ndarray, s: np.ndarray, t: np.ndarray, z: np.ndarray, intr: dict) -> np.ndarray:
    """Pixels whose point lies clearly below a higher surface seen at (about) the same place on the plane: walls of
    boxes and the like. Looking from above they are hidden, so they must not add area (the top already does)."""
    ok = np.isfinite(h) & np.isfinite(s) & np.isfinite(t)
    out = np.zeros(h.shape, bool)
    if ok.sum() < 9:
        return out
    hv, sv, tv = h[ok], s[ok], t[ok]
    cell = 0.75 * float(np.median(z[ok])) / math.sqrt(intr["fx"] * intr["fy"])
    i = np.floor((sv - sv.min()) / cell).astype(np.int64)
    j = np.floor((tv - tv.min()) / cell).astype(np.int64)
    gw, gh = int(i.max()) + 1, int(j.max()) + 1
    if gw * gh > 40_000_000:                                   # absurd spread (wrong intrinsics): skip
        return out
    grid = np.full(gw * gh, -1e9, np.float32)
    np.maximum.at(grid, j * gw + i, hv.astype(np.float32))
    top = cv2.dilate(grid.reshape(gh, gw), np.ones((3, 3), np.uint8)).reshape(-1)
    # tolerance from the measurement noise: the spread of heights around their 3x3 mean
    hf = np.where(ok, h, 0).astype(np.float32)
    cnt = cv2.blur(ok.astype(np.float32), (3, 3))
    mean = cv2.blur(hf, (3, 3)) / np.maximum(cnt, 1e-6)
    resid = np.abs(h - mean)[ok & (cnt > 0.99)]
    sigma = 1.4826 * float(np.median(resid)) * 1.06 if resid.size else 0.0
    tol = 2.0 + 4.0 * sigma
    under = hv < top[j * gw + i] - tol
    out[ok] = under
    return out


def volume_above(depth: np.ndarray, intr: dict, plane, mask=None, min_height: float = 0.0) -> dict:
    """Volume (mm³) of material above the plane, its footprint area (mm²) and the highest point, over the
    pixels of ``mask`` (None = all) whose height is above ``min_height``."""
    h = height_map(depth, intr, plane)
    a = pixel_areas(depth, intr, plane)
    sel = np.isfinite(h) & np.isfinite(a) & (h > float(min_height))
    if mask is not None:
        sel &= mask
    hv, av = h[sel].astype(np.float64), a[sel].astype(np.float64)
    return {"volume": float((hv * av).sum()), "area": float(av.sum()),
            "max_height": float(hv.max()) if hv.size else 0.0, "pixels": int(sel.sum())}


def fill_holes(depth: np.ndarray, max_area: int) -> np.ndarray:
    """Fill invalid regions up to ``max_area`` pixels from their valid neighbours (larger ones stay invalid)."""
    out = depth.copy()
    bad = ~np.isfinite(out)
    if not bad.any() or max_area <= 0:
        return out
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bad.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    todo = small[lab] & bad
    for _ in range(int(math.sqrt(max_area)) + 2):
        if not todo.any():
            break
        ok = np.isfinite(out)
        val = np.where(ok, out, 0).astype(np.float32)
        s = cv2.blur(val, (3, 3), borderType=cv2.BORDER_CONSTANT)
        c = cv2.blur(ok.astype(np.float32), (3, 3), borderType=cv2.BORDER_CONSTANT)
        can = todo & (c > 1e-6)
        out[can] = s[can] / c[can]
        todo &= ~can
    return out


# -------------------------------------------------------------------------------------------- synthetic test scene

SCENES = ["kutular", "kutular + silindir + delik", "tek kutu", "boş zemin"]


def scene_objects(scene: str) -> list[dict]:
    """Objects on the plane, positions in mm from the point where the optical axis meets the plane."""
    if scene == SCENES[2]:
        return [{"kind": "box", "x": 0.0, "y": 0.0, "sx": 100.0, "sy": 70.0, "h": 40.0, "angle": 15.0}]
    if scene == SCENES[3]:
        return []
    boxes = [{"kind": "box", "x": -120.0, "y": -60.0, "sx": 80.0, "sy": 60.0, "h": 40.0, "angle": 0.0},
             {"kind": "box", "x": 30.0, "y": -70.0, "sx": 70.0, "sy": 70.0, "h": 60.0, "angle": 20.0},
             {"kind": "box", "x": 140.0, "y": 50.0, "sx": 90.0, "sy": 50.0, "h": 25.0, "angle": -10.0}]
    if scene == SCENES[1]:
        boxes += [{"kind": "cylinder", "x": -90.0, "y": 80.0, "r": 35.0, "h": 50.0},
                  {"kind": "hole", "x": 40.0, "y": 90.0, "r": 25.0, "h": 30.0}]
    return boxes


def object_volume(o: dict) -> float:
    """True volume (mm³) of a scene object; a hole has none (no material above the plane)."""
    if o["kind"] == "box":
        return o["sx"] * o["sy"] * o["h"]
    if o["kind"] == "cylinder":
        return math.pi * o["r"] ** 2 * o["h"]
    return 0.0


def _rot(ax: float, ay: float) -> np.ndarray:
    """Plane frame → camera frame: columns e1, e2, n (n towards the camera) for a plane tilted ax° about the
    camera x axis and ay° about its y axis."""
    a, b = math.radians(ax), math.radians(ay)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    base = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])   # e1 = x, e2 = y, n = -z (facing the camera)
    return ry @ rx @ base


def synthetic_scene(width: int = 640, height: int = 480, distance: float = 800.0, tilt_x: float = 0.0,
                    tilt_y: float = 0.0, noise: float = 0.0, dropout: float = 0.0, objects=None,
                    scene: str = SCENES[0], hfov: float = 60.0, seed: int = 0) -> dict:
    """Ray-cast depth map of a plane with boxes, cylinders and holes (walls and occlusions included).
    Returns {"depth", "image", "intrinsics", "plane", "objects"}."""
    intr = make_intrinsics(width, height, hfov)
    objs = [dict(o) for o in (scene_objects(scene) if objects is None else objects)]
    R = _rot(tilt_x, tilt_y)
    n = R[:, 2]
    origin = np.array([0.0, 0.0, float(distance)])
    d0 = -float(n @ origin)                       # plane: n·X + d0 = 0
    r = rays(intr, (height, width)).reshape(-1, 3).astype(np.float64)
    # the ray X = λ r (λ = camera z) in plane coordinates: C + λ q
    C = R.T @ (-origin)
    q = r @ R                                     # (N, 3): components along e1, e2, n
    N = len(q)
    best = np.full(N, np.inf)
    label = np.zeros(N, np.int32)                 # 0 = ground, k = object k, -k = inside hole k
    shade = np.ones(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_ground = -C[2] / q[:, 2]
    lam_ground[~(lam_ground > 0)] = np.inf
    gpt = C[None, :] + lam_ground[:, None] * q
    ground_ok = np.isfinite(lam_ground)
    holes = []
    for k, o in enumerate(objs, 1):
        if o["kind"] == "hole":
            inside = ground_ok & ((gpt[:, 0] - o["x"]) ** 2 + (gpt[:, 1] - o["y"]) ** 2 < o["r"] ** 2)
            holes.append((k, o, inside))
    ground_lam = lam_ground.copy()
    for k, o, inside in holes:
        ground_lam[inside] = np.inf
    best = np.minimum(best, ground_lam)
    shade[:] = 1.0

    def take(lam, k, s):
        nonlocal best
        better = lam < best
        best = np.where(better, lam, best)
        label[better] = k
        shade[better] = s[better] if isinstance(s, np.ndarray) else s

    with np.errstate(divide="ignore", invalid="ignore"):
        for k, o in enumerate(objs, 1):
            if o["kind"] == "box":
                ang = math.radians(o.get("angle", 0.0))
                ca, sa = math.cos(ang), math.sin(ang)
                # into the box frame: rotate about the plane normal, centre at (x, y)
                cl = np.array([ca * (C[0] - o["x"]) + sa * (C[1] - o["y"]),
                               -sa * (C[0] - o["x"]) + ca * (C[1] - o["y"]), C[2]])
                ql = np.stack([ca * q[:, 0] + sa * q[:, 1], -sa * q[:, 0] + ca * q[:, 1], q[:, 2]], 1)
                lo = np.array([-o["sx"] / 2, -o["sy"] / 2, 0.0])
                hi = np.array([o["sx"] / 2, o["sy"] / 2, o["h"]])
                t1 = (lo[None, :] - cl[None, :]) / ql
                t2 = (hi[None, :] - cl[None, :]) / ql
                tmin = np.nanmax(np.minimum(t1, t2), axis=1)
                tmax = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tmax >= tmin) & (tmin > 0)
                face = np.nanargmax(np.minimum(t1, t2), axis=1)      # 2 = top, 0/1 = a wall
                s = np.where(face == 2, 1.0, 0.7)
                take(np.where(hit, tmin, np.inf), k, s)
            elif o["kind"] == "cylinder":
                ox, oy = C[0] - o["x"], C[1] - o["y"]
                a = q[:, 0] ** 2 + q[:, 1] ** 2
                b = 2 * (ox * q[:, 0] + oy * q[:, 1])
                c = ox ** 2 + oy ** 2 - o["r"] ** 2
                disc = b * b - 4 * a * c
                lam_side = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
                w_side = C[2] + lam_side * q[:, 2]
                side = (disc >= 0) & (lam_side > 0) & (w_side >= 0) & (w_side <= o["h"])
                lam_top = (o["h"] - C[2]) / q[:, 2]
                tp = C[None, :2] + lam_top[:, None] * q[:, :2]
                top = (lam_top > 0) & ((tp[:, 0] - o["x"]) ** 2 + (tp[:, 1] - o["y"]) ** 2 <= o["r"] ** 2)
                lam = np.where(top, lam_top, np.where(side, lam_side, np.inf))
                take(lam, k, np.where(top, 1.0, 0.75))
        for k, o, inside in holes:
            ox, oy = C[0] - o["x"], C[1] - o["y"]
            a = q[:, 0] ** 2 + q[:, 1] ** 2
            b = 2 * (ox * q[:, 0] + oy * q[:, 1])
            c = ox ** 2 + oy ** 2 - o["r"] ** 2
            disc = b * b - 4 * a * c
            lam_exit = (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a)
            lam_bottom = (-o["h"] - C[2]) / q[:, 2]
            lam = np.where(lam_exit < lam_bottom, lam_exit, lam_bottom)
            lam = np.where(inside, lam, np.inf)
            take(lam, -k, np.where(lam_exit < lam_bottom, 0.6, 0.5))
    depth = best.reshape(height, width).astype(np.float32)
    depth[~np.isfinite(depth)] = np.nan
    rng = np.random.default_rng(seed)
    if noise > 0:
        depth = depth + rng.normal(0, noise, depth.shape).astype(np.float32)
    if dropout > 0:
        depth[rng.random(depth.shape) < dropout] = np.nan
    # colour picture: checker floor, one colour per object, simple shading
    lab = label.reshape(height, width)
    pts = (C[None, :] + np.where(np.isfinite(best), best, 0)[:, None] * q).reshape(height, width, 3)
    checker = ((np.floor(pts[:, :, 0] / 40) + np.floor(pts[:, :, 1] / 40)) % 2).astype(np.float32)
    img = np.empty((height, width, 3), np.float32)
    img[:] = (110, 110, 105)
    img += checker[:, :, None] * 25
    palette = [(60, 140, 230), (80, 200, 120), (220, 120, 70), (200, 90, 200), (70, 210, 220), (150, 150, 240)]
    for k in range(1, len(objs) + 1):
        img[lab == k] = palette[(k - 1) % len(palette)]
        img[lab == -k] = (45, 45, 50)
    img *= shade.reshape(height, width, 1)
    img[~np.isfinite(best.reshape(height, width))] = 0
    plane = plane_dict(n, d0)
    return {"depth": depth, "image": np.clip(img, 0, 255).astype(np.uint8), "intrinsics": intr, "plane": plane,
            "objects": objs}


# ------------------------------------------------------------------------------------------------------- files

PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
             "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def write_ply(path: str, points: np.ndarray, colors=None, binary: bool = True, intr: dict | None = None,
              size: tuple[int, int] | None = None) -> str:
    """Point cloud (mm) to a .ply file; colours are BGR (as OpenCV) and written as red/green/blue. Intrinsics and
    picture size go into comments so the file can be read back as the same depth map."""
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    ok = np.isfinite(pts).all(1)
    pts = pts[ok]
    cols = None
    if colors is not None:
        cols = np.asarray(colors, np.uint8).reshape(-1, 3)[ok][:, ::-1]     # BGR → RGB
    head = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0", "comment VisionFlow nokta bulutu (mm)"]
    if intr is not None:
        head.append(f"comment intrinsics {intr['fx']:.9g} {intr['fy']:.9g} {intr['cx']:.9g} {intr['cy']:.9g}")
    if size is not None:
        head.append(f"comment size {int(size[0])} {int(size[1])}")
    head += [f"element vertex {len(pts)}", "property float x", "property float y", "property float z"]
    if cols is not None:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    head.append("end_header")
    p = pathlib.Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        if binary:
            dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")]
                                                              if cols is not None else [])
            rec = np.empty(len(pts), dtype=dt)
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            if cols is not None:
                rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
            f.write(rec.tobytes())
        else:
            if cols is not None:
                lines = (f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}" for (x, y, z), (r, g, b) in zip(pts.tolist(),
                                                                                              cols.tolist()))
            else:
                lines = (f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in pts.tolist())
            f.write(("\n".join(lines) + ("\n" if len(pts) else "")).encode("ascii"))
    return str(p)


def read_ply(path: str) -> dict:
    """{"points" (N,3) float32, "colors" (N,3) BGR uint8 or None, "intrinsics" or None, "size" or None}.
    ASCII and binary (little/big endian) vertex lists with x, y, z (and red, green, blue) are read."""
    data = pathlib.Path(path).read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError(f"PLY dosyası değil: {path}")
    nl = data.find(b"\n", end)
    header = data[:end].decode("ascii", "replace").splitlines()
    body = data[nl + 1:] if nl >= 0 else b""
    fmt, elements, intr, size = "ascii", [], None, None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "comment" and len(parts) >= 6 and parts[1] == "intrinsics":
            try:
                fx, fy, cx, cy = (float(v) for v in parts[2:6])
                intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
            except ValueError:
                pass
        elif parts[0] == "comment" and len(parts) >= 4 and parts[1] == "size":
            try:
                size = (int(parts[2]), int(parts[3]))
            except ValueError:
                pass
        elif parts[0] == "element":
            elements.append([parts[1], int(parts[2]), []])
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1][2].append((parts[-1], None))
            else:
                elements[-1][2].append((parts[2], parts[1]))
    vert = next((e for e in elements if e[0] == "vertex"), None)
    if vert is None:
        raise ValueError("PLY dosyasında nokta (vertex) yok")
    names = [n for n, _ in vert[2]]
    if not {"x", "y", "z"} <= set(names):
        raise ValueError("PLY noktalarında x, y, z yok")
    count = vert[1]
    if fmt == "ascii":
        lines = body.decode("ascii", "replace").splitlines()
        skip = sum(e[1] for e in elements[:elements.index(vert)])
        rows = [ln.split() for ln in lines[skip:skip + count]]
        if len(rows) < count:
            raise ValueError("PLY dosyası eksik (nokta sayısı tutmuyor)")
        try:
            arr = np.array([[float(v) for v in r[:len(names)]] for r in rows], np.float64).reshape(count, len(names))
        except ValueError:
            raise ValueError("PLY dosyasında sayı olmayan değer var") from None
        cols = {n: arr[:, i] for i, n in enumerate(names)}
    elif fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if fmt.endswith("little_endian") else ">"
        offset = 0
        for e in elements:
            if any(t is None for _, t in e[2]):
                if e is vert or elements.index(e) < elements.index(vert):
                    raise ValueError("Bu PLY dosyası okunamıyor (noktalardan önce liste var)")
                continue
            dt = np.dtype([(n, endian + PLY_TYPES[t]) for n, t in e[2]])
            if e is vert:
                if len(body) < offset + dt.itemsize * count:
                    raise ValueError("PLY dosyası eksik (nokta sayısı tutmuyor)")
                rec = np.frombuffer(body, dt, count, offset)
                cols = {n: rec[n].astype(np.float64) for n in names}
                break
            offset += dt.itemsize * e[1]
    else:
        raise ValueError(f"Bilinmeyen PLY biçimi: {fmt}")
    pts = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float32)
    colors = None
    rgb = [next((k for k in (c, c[0]) if k in cols), None) for c in ("red", "green", "blue")]
    if all(rgb):
        colors = np.stack([cols[rgb[2]], cols[rgb[1]], cols[rgb[0]]], 1).clip(0, 255).astype(np.uint8)
    return {"points": pts, "colors": colors, "intrinsics": intr, "size": size}


DEPTH_EXT = (".png", ".npy", ".ply", ".tif", ".tiff")
COLOR_SUFFIX = ("_renk", "_color")


def _imread(path: str, flags):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)       # unicode-safe on Windows
    if img is None:
        raise ValueError(f"Dosya okunamadı: {path}")
    return img


def _sidecar(path: str) -> dict | None:
    """Intrinsics saved next to a depth file: <ad>.json or <ad>_intrinsics.json."""
    p = pathlib.Path(path)
    for cand in (p.with_suffix(".json"), p.with_name(p.stem + "_intrinsics.json")):
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                return intrinsics(data.get("intrinsics", data))
            except (OSError, ValueError, AttributeError):
                return None
    return None


def find_color(path: str) -> str | None:
    p = pathlib.Path(path)
    for suf in COLOR_SUFFIX:
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            cand = p.with_name(p.stem + suf + ext)
            if cand.is_file():
                return str(cand)
    return None


def _estimate_intrinsics(xyz: np.ndarray) -> dict | None:
    """fx, cx, fy, cy of an organized HxWx3 cloud from x = (u - cx) / fx · z (least squares)."""
    H, W = xyz.shape[:2]
    ok = np.isfinite(xyz).all(2) & (xyz[:, :, 2] > 0)
    if ok.sum() < 10:
        return None
    vs, us = np.nonzero(ok)
    a = xyz[vs, us, 0] / xyz[vs, us, 2]
    b = xyz[vs, us, 1] / xyz[vs, us, 2]
    ka, ca = np.polyfit(a, us, 1)
    kb, cb = np.polyfit(b, vs, 1)
    if not (ka > 0 and kb > 0):
        return None
    return {"fx": float(ka), "fy": float(kb), "cx": float(ca), "cy": float(cb), "width": W, "height": H}


def load_depth_file(path: str, unit: str = "mm", hfov: float = 60.0, size=(640, 480), intr: dict | None = None):
    """A depth file → (depth mm, colour or None, intrinsics). 16-bit PNG / TIFF (mm), .npy (HxW depth, HxWx3
    organized XYZ or Nx3 cloud) and .ply (point cloud; written by VisionFlow it has its own intrinsics)."""
    scale = 1000.0 if unit == "m" else 0.1 if unit == "0.1 mm" else 1.0
    ext = pathlib.Path(path).suffix.lower()
    if not os.path.isfile(path):
        raise ValueError(f"Dosya bulunamadı: {path}")
    color = None
    known = intr or _sidecar(path)
    if ext in (".png", ".tif", ".tiff"):
        raw = _imread(path, cv2.IMREAD_UNCHANGED)
        if raw.ndim == 3:
            raise ValueError(f"Derinlik dosyası tek kanallı olmalı (renkli resim: {path})")
        if raw.dtype == np.uint8:
            raise ValueError(f"Derinlik PNG'si 16 bit olmalı (mm); bu dosya 8 bit: {path}")
        depth = as_depth(raw.astype(np.float32) * scale)
    elif ext == ".npy":
        try:
            arr = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f".npy okunamadı: {exc}") from None
        if arr.dtype.kind not in "uif":
            raise ValueError(f".npy sayı dizisi değil: {arr.dtype}")
        arr = arr.astype(np.float32) * scale
        if arr.ndim == 2 and arr.shape[1] != 3:
            depth = as_depth(arr)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            depth = as_depth(arr[:, :, 2])
            known = known or _estimate_intrinsics(arr)
        elif arr.ndim == 2 and arr.shape[1] == 3:
            w, h = size
            known = intrinsics(known, (h, w)) if known else make_intrinsics(w, h, hfov)
            depth, _ = points_to_depth(arr, known, known["width"], known["height"])
            depth = as_depth(depth)
        else:
            raise ValueError(f".npy boyutu anlaşılmadı: {arr.shape} (HxW derinlik, HxWx3 veya Nx3 nokta)")
    elif ext == ".ply":
        ply = read_ply(path)
        pts = ply["points"] * scale
        w, h = ply["size"] or size
        known = known or ply["intrinsics"] or make_intrinsics(w, h, hfov)
        known = dict(known, width=w, height=h)
        depth, color = points_to_depth(pts, known, w, h, ply["colors"])
        if np.isfinite(depth).sum() < 3:
            raise ValueError("PLY noktaları kameranın önünde değil (z > 0 olmalı)")
        depth = as_depth(depth)
    else:
        raise ValueError(f"Desteklenmeyen derinlik dosyası: {ext} (.png, .tif, .npy, .ply)")
    H, W = depth.shape
    known = intrinsics(known, (H, W)) if known else make_intrinsics(W, H, hfov)
    cpath = find_color(path)
    if cpath:
        color = _imread(cpath, cv2.IMREAD_COLOR)
    if color is not None and color.shape[:2] != (H, W):
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_AREA)
    return depth, color, known


def save_depth_png(path: str, depth: np.ndarray, intr: dict | None = None) -> str:
    """16-bit PNG in mm (0 = invalid) plus <ad>.json with the intrinsics."""
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(np.round(d), 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", d)
    if not ok:
        raise ValueError("PNG yazılamadı")
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(p))
    if intr is not None:
        p.with_suffix(".json").write_text(json.dumps({k: intr[k] for k in ("fx", "fy", "cx", "cy", "width", "height")
                                                      if k in intr}), encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------------------------------------- depth cameras

DRIVERS = ["Intel RealSense", "Orbbec", "Dosyadan"]


def _module(name: str, hint: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(hint) from exc


class RealSenseCamera:
    """Intel RealSense D400/L500 series through pyrealsense2: depth (z16) + colour (bgr8), aligned to colour."""

    def __init__(self, serial: str = "", width: int = 640, height: int = 480, fps: int = 30, align: bool = True):
        rs = self.rs = _module("pyrealsense2", "Intel RealSense için 'pip install pyrealsense2' kurun (ve kamerayı "
                                               "USB 3 porta takın)")
        ctx = rs.context()
        devs = list(ctx.query_devices())
        sn = lambda d: str(d.get_info(rs.camera_info.serial_number))   # noqa: E731
        dev = _pick_device(devs, serial, sn, "RealSense")
        self.serial = sn(dev)
        try:
            self.name = f"{dev.get_info(rs.camera_info.name)} ({self.serial})"
        except Exception:  # noqa: BLE001
            self.name = f"RealSense ({self.serial})"
        self.pipe = rs.pipeline(ctx)
        self.has_color = True
        try:
            profile = self.pipe.start(self._config(width, height, fps, True))
        except Exception:  # noqa: BLE001  (no colour sensor, or that colour mode is not offered)
            self.has_color = False
            try:
                profile = self.pipe.start(self._config(width, height, fps, False))
            except Exception as exc:
                raise RuntimeError(f"RealSense başlatılamadı ({width}x{height} @ {fps}): {exc}") from exc
        try:
            self.scale = float(profile.get_device().first_depth_sensor().get_depth_scale()) * 1000.0
        except Exception:  # noqa: BLE001
            self.scale = 1.0
        self.align = rs.align(rs.stream.color) if align and self.has_color else None

    def _config(self, w, h, fps, color):
        rs = self.rs
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.depth, int(w), int(h), rs.format.z16, int(fps))
        if color:
            cfg.enable_stream(rs.stream.color, int(w), int(h), rs.format.bgr8, int(fps))
        return cfg

    def grab(self, timeout_ms: int = 1000):
        try:
            frames = self.pipe.wait_for_frames(int(timeout_ms))
        except RuntimeError:
            return None                                   # time-out: no frame this time
        if self.align is not None:
            frames = self.align.process(frames)
        d = frames.get_depth_frame()
        if not d:
            return None
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.scale
        depth[depth <= 0] = np.nan
        c = frames.get_color_frame() if self.has_color else None
        color = np.array(np.asanyarray(c.get_data()), np.uint8, copy=True) if c else None
        i = d.profile.as_video_stream_profile().intrinsics
        intr = {"fx": float(i.fx), "fy": float(i.fy), "cx": float(i.ppx), "cy": float(i.ppy),
                "width": int(i.width), "height": int(i.height)}
        if color is not None and color.shape[:2] != depth.shape:     # not aligned: same size for the blocks
            color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        return depth, color, intrinsics(intr, depth.shape)

    def close(self):
        self.pipe.stop()


class OrbbecCamera:
    """Orbbec Femto / Astra / Gemini through pyorbbecsdk: depth (Y16) + colour, depth aligned to colour."""

    def __init__(self, serial: str = "", width: int = 640, height: int = 480, fps: int = 30, align: bool = True):
        ob = self.ob = _module("pyorbbecsdk", "Orbbec kameralar için 'pip install pyorbbecsdk' kurun")
        dl = ob.Context().query_devices()
        n = int(dl.get_count())
        devs = list(range(n))
        sn = lambda i: str(dl.get_device_serial_number_by_index(i))     # noqa: E731
        idx = _pick_device(devs, serial, sn, "Orbbec")
        dev = dl.get_device_by_index(idx)
        self.serial = sn(idx)
        self.name = f"Orbbec ({self.serial})"
        self.pipe = ob.Pipeline(dev)
        cfg = ob.Config()
        dprof = self._profile(ob.OBSensorType.DEPTH_SENSOR, width, height, ob.OBFormat.Y16, fps)
        if dprof is None:
            raise RuntimeError("Orbbec kamerada derinlik akışı yok")
        cfg.enable_stream(dprof)
        cprof = None
        try:
            cprof = self._profile(ob.OBSensorType.COLOR_SENSOR, width, height, ob.OBFormat.RGB, fps)
        except Exception:  # noqa: BLE001  (a depth-only model)
            cprof = None
        self.has_color = cprof is not None
        if self.has_color:
            cfg.enable_stream(cprof)
        self.aligned = bool(align and self.has_color)
        if self.aligned:
            try:
                cfg.set_align_mode(ob.OBAlignMode.SW_MODE)
            except Exception:  # noqa: BLE001
                self.aligned = False
        try:
            self.pipe.start(cfg)
        except Exception as exc:
            raise RuntimeError(f"Orbbec başlatılamadı: {exc}") from exc

    def _profile(self, sensor, w, h, fmt, fps):
        plist = self.pipe.get_stream_profile_list(sensor)
        if plist is None:
            return None
        try:
            return plist.get_video_stream_profile(int(w), int(h), fmt, int(fps))
        except Exception:  # noqa: BLE001  (that mode is not offered: the camera's own default)
            return plist.get_default_video_stream_profile()

    def _color(self, cf):
        ob = self.ob
        w, h = int(cf.get_width()), int(cf.get_height())
        data = np.frombuffer(bytes(cf.get_data()), np.uint8)
        fmt = cf.get_format()
        if fmt == ob.OBFormat.RGB:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == ob.OBFormat.BGR:
            return data.reshape(h, w, 3).copy()
        if fmt == ob.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == ob.OBFormat.YUYV:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        return None

    def grab(self, timeout_ms: int = 1000):
        frames = self.pipe.wait_for_frames(int(timeout_ms))
        if frames is None:
            return None
        d = frames.get_depth_frame()
        if d is None:
            return None
        w, h = int(d.get_width()), int(d.get_height())
        raw = np.frombuffer(bytes(d.get_data()), np.uint16)
        if raw.size != w * h:
            return None                                   # a broken frame
        depth = raw.reshape(h, w).astype(np.float32) * float(d.get_depth_scale())
        depth[depth <= 0] = np.nan
        color = None
        if self.has_color:
            cf = frames.get_color_frame()
            color = self._color(cf) if cf is not None else None
        param = self.pipe.get_camera_param()
        i = param.rgb_intrinsic if self.aligned else param.depth_intrinsic
        intr = {"fx": float(i.fx), "fy": float(i.fy), "cx": float(i.cx), "cy": float(i.cy),
                "width": int(i.width), "height": int(i.height)}
        if color is not None and color.shape[:2] != depth.shape:
            color = cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA)
        return depth, color, intrinsics(intr, depth.shape)

    def close(self):
        self.pipe.stop()


def _pick_device(devices, serial: str, serial_of, kind: str):
    devices = list(devices)
    if not devices:
        raise RuntimeError(f"{kind} derinlik kamerası bulunamadı (USB 3 kablosu takılı mı, sürücü kurulu mu?)")
    serial = (serial or "").strip()
    if not serial:
        return devices[0]
    for d in devices:
        if serial_of(d) == serial:
            return d
    if serial.isdigit() and len(serial) <= 2 and int(serial) < len(devices):
        return devices[int(serial)]
    raise RuntimeError(f"Seri numarası {serial} olan {kind} kamera yok. Bulunanlar: "
                       + ", ".join(serial_of(d) for d in devices))


def list_depth_cameras(driver: str) -> list[str]:
    """Serial numbers of the connected cameras of a driver (needs its package)."""
    if driver == DRIVERS[0]:
        rs = _module("pyrealsense2", "Intel RealSense için 'pip install pyrealsense2' kurun")
        return [str(d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()]
    if driver == DRIVERS[1]:
        ob = _module("pyorbbecsdk", "Orbbec kameralar için 'pip install pyorbbecsdk' kurun")
        dl = ob.Context().query_devices()
        return [str(dl.get_device_serial_number_by_index(i)) for i in range(int(dl.get_count()))]
    return []


class SharedDevice:
    """One open depth camera per (driver, serial, mode), shared by reference count (the block and the 3D viewer
    window may use it at the same time); grabs are serialized with a lock."""
    _devices: dict = {}
    _guard = threading.Lock()

    def __init__(self, key, cam):
        self.key, self.cam, self.users, self.lock = key, cam, 0, threading.Lock()

    @classmethod
    def acquire(cls, driver: str, serial: str, width: int, height: int, fps: int, align: bool) -> "SharedDevice":
        key = (driver, (serial or "").strip(), int(width), int(height), int(fps), bool(align))
        with cls._guard:
            dev = cls._devices.get(key)
            if dev is None:
                camcls = RealSenseCamera if driver == DRIVERS[0] else OrbbecCamera
                dev = cls(key, camcls(serial, width, height, fps, align))
                cls._devices[key] = dev
            dev.users += 1
            return dev

    def grab(self, timeout_ms: int):
        with self.lock:
            return self.cam.grab(timeout_ms)

    @property
    def name(self) -> str:
        return getattr(self.cam, "name", str(self.key[0]))

    def release(self):
        with self._guard:
            self.users -= 1
            if self.users > 0:
                return
            if self._devices.get(self.key) is self:
                self._devices.pop(self.key, None)
        try:
            self.cam.close()
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def open_count(cls) -> int:
        with cls._guard:
            return len(cls._devices)


# ---------------------------------------------------------------------- last frame of a block, for the 3D viewer

_LIVE: dict = {}
_LIVE_LOCK = threading.Lock()


def publish(node_id: str, depth, color, intr, plane=None) -> None:
    """Remember a block's newest depth frame (the flow runs on a copy of the blocks; the viewer finds it by id)."""
    with _LIVE_LOCK:
        n = _LIVE.get(node_id, (0,))[0] + 1
        _LIVE[node_id] = (n, depth, color, intr, plane)


def latest(node_id: str):
    """(frame number, depth, colour, intrinsics, plane) or None."""
    with _LIVE_LOCK:
        return _LIVE.get(node_id)


def forget(node_id: str) -> None:
    with _LIVE_LOCK:
        _LIVE.pop(node_id, None)


# ----------------------------------------------------------------------------------------- colours and rendering

COLORMAPS = {"Turbo": "COLORMAP_TURBO", "Jet": "COLORMAP_JET", "Viridis": "COLORMAP_VIRIDIS",
             "Inferno": "COLORMAP_INFERNO", "Gri": None}


def colorize(values: np.ndarray, vmin: float | None = None, vmax: float | None = None, cmap: str = "Turbo",
             invalid=(0, 0, 0), reverse: bool = False) -> tuple[np.ndarray, float, float]:
    """Float map → BGR picture with the range vmin..vmax (None = 2nd..98th percentile of the valid values);
    NaN pixels get ``invalid``. Returns (picture, vmin, vmax)."""
    v = np.asarray(values, np.float32)
    ok = np.isfinite(v)
    if vmin is None or vmax is None:
        if ok.any():
            lo, hi = np.percentile(v[ok], [2, 98])
        else:
            lo, hi = 0.0, 1.0
        vmin = float(lo) if vmin is None else vmin
        vmax = float(hi) if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-3
    t = np.clip((np.nan_to_num(v, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    if reverse:
        t = 1 - t
    g = (t * 255 + 0.5).astype(np.uint8)
    name = COLORMAPS.get(cmap)
    code = getattr(cv2, name, cv2.COLORMAP_JET) if name else None
    img = cv2.applyColorMap(g, code) if code is not None else cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    img[~ok] = invalid
    return img, float(vmin), float(vmax)


def render_cloud(points: np.ndarray, colors: np.ndarray | None, size=(640, 480), yaw: float = 0.0,
                 pitch: float = 0.0, zoom: float = 1.0, center=None, pan=(0.0, 0.0), point_size: int = 2,
                 bg=(28, 26, 24), axes: bool = True) -> np.ndarray:
    """Software render of a point cloud (no OpenGL): orbit camera around ``center`` (default: the cloud's
    middle), yaw/pitch in degrees, nearest point wins per pixel. Returns a BGR picture."""
    W, H = int(size[0]), int(size[1])
    img = np.empty((H, W, 3), np.uint8)
    img[:] = bg
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    ok = np.isfinite(pts).all(1)
    pts = pts[ok]
    cols = None if colors is None else np.asarray(colors, np.uint8).reshape(-1, 3)[ok]
    if not len(pts):
        return img
    c = np.asarray(center, np.float64) if center is not None else np.median(pts, 0)
    extent = float(np.percentile(np.linalg.norm(pts - c, axis=1), 95)) or 1.0
    a, b = math.radians(yaw), math.radians(pitch)
    # view rotation: start from the depth camera's own view (x right, y down, z forward), then orbit
    ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
    rot = rx @ ry
    dist = extent * 2.6 / max(zoom, 1e-3)
    p = (pts - c) @ rot.T
    p[:, 0] -= pan[0] * extent
    p[:, 1] -= pan[1] * extent
    p[:, 2] += dist
    front = p[:, 2] > 1e-3
    p = p[front]
    if cols is not None:
        cols = cols[front]
    f = min(W, H) * 1.1
    u = np.round(p[:, 0] / p[:, 2] * f + W / 2).astype(np.int64)
    v = np.round(p[:, 1] / p[:, 2] * f + H / 2).astype(np.int64)
    if cols is None:
        cols = np.full((len(p), 3), 220, np.uint8)
    order = np.argsort(p[:, 2], kind="stable")
    s = max(1, int(point_size))
    for dy in range(s):
        for dx in range(s):
            uu, vv = u + dx, v + dy
            inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            o = order[inside[order]]
            idx = vv[o] * W + uu[o]
            _, first = np.unique(idx, return_index=True)
            sel = o[first]
            # farther splat offsets must not cover nearer points: draw only where the pixel is still background
            flat = img.reshape(-1, 3)
            target = vv[sel] * W + uu[sel]
            if dx or dy:
                free = (flat[target] == np.asarray(bg, np.uint8)).all(1)
                sel, target = sel[free], target[free]
            flat[target] = cols[sel]
    if axes:
        L = extent * 0.4
        o3 = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], np.float64)
        q = o3 @ rot.T
        q[:, 0] -= pan[0] * extent
        q[:, 1] -= pan[1] * extent
        q[:, 2] += dist
        if (q[:, 2] > 1e-3).all():
            uv = np.stack([q[:, 0] / q[:, 2] * f + W / 2, q[:, 1] / q[:, 2] * f + H / 2], 1)
            o2 = tuple(int(round(x)) for x in uv[0])
            for k, col in ((1, (60, 60, 230)), (2, (60, 200, 60)), (3, (230, 140, 50))):
                cv2.line(img, o2, tuple(int(round(x)) for x in uv[k]), col, 1, cv2.LINE_AA)
    return img


def cloud_for_view(depth: np.ndarray, intr: dict, color=None, plane=None, mode: str = "yükseklik",
                   max_points: int = 60000):
    """Points and per-point BGR colours for the viewer: coloured by height above the plane (or by depth when no
    plane is known), by the camera picture, or plain."""
    depth = as_depth(depth)
    intr = intrinsics(intr, depth.shape)
    n_valid = int(np.isfinite(depth).sum())
    step = max(1, int(math.ceil(math.sqrt(n_valid / max(1, max_points))))) if n_valid else 1
    pts, (vs, us) = depth_to_points(depth, intr, step=step)
    if mode == "kamera rengi" and color is not None and isinstance(color, np.ndarray) and \
            color.shape[:2] == depth.shape:
        c = color if color.ndim == 3 else cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
        cols = c[vs, us, :3].astype(np.uint8)
    elif mode == "düz":
        cols = np.full((len(pts), 3), 220, np.uint8)
    else:
        if plane is not None:
            n, d = as_plane(plane)
            vals = pts.astype(np.float64) @ n + d
            img, _, _ = colorize(vals[None, :].astype(np.float32), cmap="Turbo")
        else:
            img, _, _ = colorize(pts[None, :, 2], cmap="Turbo", reverse=True)
        cols = img[0]
    return pts, cols
