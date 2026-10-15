"""Camera calibration and real-world (mm) scale.

- Lens calibration: 10-20 photos of a printed chessboard -> camera matrix and lens distortion, saved as JSON and used
  by the "Lens Bozulmasını Düzelt" block.
- Scale: a chessboard square or an ArUco marker of known size in the picture gives pixels per millimetre.
- Plane: the four outer corners of a chessboard give a top-down (bird's-eye) view in which 1 mm is a fixed number of
  pixels everywhere, even when the camera looks at the table at an angle.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Görüntü girişi bağlı değil")
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 else img[:, :, 0]
    return img


def find_board(img: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Inner corners of a chessboard (cols x rows inner corners, e.g. 9x6 for a 10x7-square board), (N, 2) float32
    in row order, or None. The sector-based detector is tried first: it is more accurate and finds tilted boards."""
    g = gray(img)
    size = (int(cols), int(rows))
    try:
        ok, c = cv2.findChessboardCornersSB(g, size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    except (cv2.error, AttributeError):
        ok, c = False, None
    if not ok:
        ok, c = cv2.findChessboardCorners(g, size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
        c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    c = c.reshape(-1, 2).astype(np.float32)
    # always start at the top-left corner, so the plane view is never upside down or mirrored
    grid = c.reshape(rows, cols, 2)
    if grid[0, 0, 1] > grid[-1, -1, 1] + 1e-6 or (abs(grid[0, 0, 1] - grid[-1, -1, 1]) <= 1e-6
                                                  and grid[0, 0, 0] > grid[-1, -1, 0]):
        grid = grid[::-1, ::-1]
    return np.ascontiguousarray(grid.reshape(-1, 2))


def board_px_per_mm(corners: np.ndarray, cols: int, rows: int, square_mm: float) -> float:
    """Average distance between neighbouring corners divided by the square size."""
    g = corners.reshape(rows, cols, 2)
    d = np.concatenate([np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(),
                        np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()])
    return float(d.mean() / square_mm)


def aruco_px_per_mm(img: np.ndarray, marker_mm: float, dictionary: str = "4X4_50", marker_id: int = -1):
    """Pixels per mm from the side length of an ArUco marker; returns (px_per_mm, corners) or (None, None)."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{dictionary}"))
    corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray(img))
    if ids is None:
        return None, None
    for c, i in zip(corners, ids.flatten()):
        if marker_id < 0 or int(i) == marker_id:
            q = c.reshape(4, 2)
            side = np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1).mean()
            return float(side / marker_mm), q
    return None, None


def plane_homography(corners: np.ndarray, cols: int, rows: int, square_mm: float, px_per_mm: float,
                     margin_mm: float = 0.0):
    """Homography from the camera picture to a top-down view of the board plane (all corners used, least squares).
    The board's first inner corner lands at (margin, margin) mm."""
    obj = np.array([[(x * square_mm + margin_mm) * px_per_mm, (y * square_mm + margin_mm) * px_per_mm]
                    for y in range(rows) for x in range(cols)], np.float32)
    H, _ = cv2.findHomography(corners.reshape(-1, 2), obj, 0)
    if H is None:
        raise ValueError("Satranç tahtasından düzlem hesaplanamadı")
    return H


@dataclass
class Calibration:
    camera_matrix: list
    dist_coeffs: list
    image_size: list                    # [width, height] of the calibration photos
    rms: float                          # mean reprojection error in pixels (below ~0.5 is good)
    images_used: int
    board: list = field(default_factory=list)        # [cols, rows, square_mm]
    per_image: dict = field(default_factory=dict)    # file name -> error in pixels
    skipped: list = field(default_factory=list)      # photos left out (board not found, other resolution)
    created: str = ""

    @property
    def K(self) -> np.ndarray:
        return np.array(self.camera_matrix, np.float64)

    @property
    def D(self) -> np.ndarray:
        return np.array(self.dist_coeffs, np.float64)

    def K_for(self, w: int, h: int) -> np.ndarray:
        """Camera matrix for another resolution of the same camera (same aspect ratio expected)."""
        K = self.K.copy()
        sx, sy = w / self.image_size[0], h / self.image_size[1]
        K[0] *= sx
        K[1] *= sy
        return K

    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "Calibration":
        try:
            data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except FileNotFoundError:
            raise FileNotFoundError(f"Kalibrasyon dosyası yok: {path} (Araçlar → Kamera Kalibrasyonu)") from None
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Kalibrasyon dosyası okunamadı: {path} ({exc})") from exc


def calibrate(images: dict[str, np.ndarray], cols: int, rows: int, square_mm: float, on_step=None) -> Calibration:
    """Lens calibration from chessboard photos {name: image}. Photos where the board is not found are skipped."""
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts, names, skipped = [], [], [], []
    sizes = [(im.shape[1], im.shape[0]) for im in images.values() if im is not None]
    # the camera's resolution is the most common one: other pictures (e.g. the printed board itself) are left out
    size = max(set(sizes), key=sizes.count) if sizes else None
    for k, (name, img) in enumerate(images.items()):
        if on_step:
            on_step(f"Satranç tahtası aranıyor: {name} ({k + 1}/{len(images)})")
        if img is None or (img.shape[1], img.shape[0]) != size:
            skipped.append(name)
            continue
        c = find_board(img, cols, rows)
        if c is not None:
            obj_pts.append(obj)
            img_pts.append(c.reshape(-1, 1, 2))
            names.append(name)
        else:
            skipped.append(name)
    if len(names) < 3:
        raise ValueError(f"Satranç tahtası sadece {len(names)} fotoğrafta bulundu; en az 3 gerekli (10–20 önerilir). "
                         f"İç köşe sayısını kontrol edin: {cols}x{rows} (kare sayısının bir eksiği).")
    if on_step:
        on_step(f"{len(names)} fotoğrafla kalibre ediliyor…")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    per = {}
    for name, o, ip, r, t in zip(names, obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, D)
        per[name] = round(float(np.sqrt(((proj - ip) ** 2).sum(-1).mean())), 3)
    return Calibration(K.tolist(), D.ravel().tolist(), list(size), round(float(rms), 4), len(names),
                       [int(cols), int(rows), float(square_mm)], per, skipped,
                       dt.datetime.now().isoformat(timespec="seconds"))


def read_folder(folder) -> dict[str, np.ndarray]:
    folder = pathlib.Path(folder)
    out = {}
    for p in sorted(folder.iterdir()) if folder.is_dir() else []:
        if p.suffix.lower() in IMAGE_EXT:
            data = np.fromfile(str(p), np.uint8)
            out[p.name] = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return out


def default_path() -> pathlib.Path:
    from .ai.config import home
    return home() / "kalibrasyon" / "kamera.json"


def chessboard(cols: int = 9, rows: int = 6, square_px: int = 60, margin: int = 40) -> np.ndarray:
    """A printable chessboard with cols x rows INNER corners (one more square each way)."""
    w, h = (cols + 1) * square_px + 2 * margin, (rows + 1) * square_px + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for y in range(rows + 1):
        for x in range(cols + 1):
            if (x + y) % 2 == 0:
                img[margin + y * square_px:margin + (y + 1) * square_px,
                    margin + x * square_px:margin + (x + 1) * square_px] = 0
    return img
