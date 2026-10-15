"""3D point cloud viewer window of the depth blocks ("🧊 3B Görüntüle…"): drag to orbit, right/middle drag to pan,
wheel to zoom, double click to reset. Drawn with QPainter from the software renderer in depth3d (no OpenGL needed,
works on any graphics card and offscreen)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from .. import depth3d as D

COLOR_MODES = ["yükseklik", "kamera rengi", "düz"]


class CloudView(QWidget):
    """A point cloud drawn from an orbiting virtual camera."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.points = np.zeros((0, 3), np.float32)
        self.colors: np.ndarray | None = None
        self.center = None
        self.reset_view()
        self._drag: tuple[str, QPointF] | None = None
        self._cache: tuple | None = None

    def reset_view(self):
        self.yaw, self.pitch, self.zoom, self.pan = -30.0, 35.0, 1.0, [0.0, 0.0]
        self._cache = None
        self.update()

    def set_cloud(self, points: np.ndarray, colors: np.ndarray | None):
        self.points = np.asarray(points, np.float32).reshape(-1, 3)
        self.colors = colors
        self.center = np.median(self.points, 0) if len(self.points) else None
        self._cache = None
        self.update()

    def render_image(self, w: int, h: int) -> np.ndarray:
        return D.render_cloud(self.points, self.colors, (max(1, w), max(1, h)), self.yaw, self.pitch, self.zoom,
                              self.center, tuple(self.pan), point_size=2)

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        key = (w, h, self.yaw, self.pitch, self.zoom, tuple(self.pan), id(self.points))
        if self._cache is None or self._cache[0] != key:
            img = np.ascontiguousarray(self.render_image(w, h))
            self._cache = (key, QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy())
        p.drawImage(0, 0, self._cache[1])
        if not len(self.points):
            p.setPen(QColor("#bbb"))
            p.drawText(self.rect(), Qt.AlignCenter, "Nokta bulutu yok: akışı çalıştırın veya 'Kare al' deyin")
        p.end()

    # mouse: left = orbit, right / middle = pan, wheel = zoom, double click = reset
    def mousePressEvent(self, ev):
        mode = "orbit" if ev.button() == Qt.LeftButton else "pan"
        self._drag = (mode, ev.position())

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        mode, last = self._drag
        dx, dy = ev.position().x() - last.x(), ev.position().y() - last.y()
        if mode == "orbit":
            self.yaw = (self.yaw + dx * 0.4 + 180) % 360 - 180
            self.pitch = max(-89.0, min(89.0, self.pitch - dy * 0.4))
        else:
            s = 1.0 / max(1, min(self.width(), self.height())) / max(self.zoom, 1e-3)
            self.pan[0] -= dx * s * 2.4
            self.pan[1] -= dy * s * 2.4
        self._drag = (mode, ev.position())
        self.update()

    def mouseReleaseEvent(self, _ev):
        self._drag = None

    def mouseDoubleClickEvent(self, _ev):
        self.reset_view()

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120.0
        self.zoom = float(min(50.0, max(0.05, self.zoom * (1.15 ** steps))))
        self.update()


class DepthViewerDialog(QDialog):
    """Opened from a depth block: shows that block's newest frame (live while the flow runs)."""

    def __init__(self, parent=None, node=None, apply=None):
        super().__init__(parent)
        self.node, self.apply = node, apply
        self.setWindowTitle(f"3B Görüntüleyici — {getattr(node, 'title', '')}")
        self.resize(900, 650)
        self.frame = None                       # (n, depth, color, intr, plane) shown now
        self._shown_n = None
        self.view = CloudView()
        self.mode = QComboBox()
        self.mode.addItems(COLOR_MODES)
        self.mode.currentIndexChanged.connect(lambda _=0: self.refresh(force=True))
        self.live = QCheckBox("Canlı")
        self.live.setChecked(True)
        grab = QPushButton("Kare al")
        grab.setToolTip("Akış çalışmıyorken bloğun kaynağından bir kare okur")
        grab.clicked.connect(self.grab_once)
        reset = QPushButton("Görünümü sıfırla")
        reset.clicked.connect(self.view.reset_view)
        save = QPushButton("PLY kaydet…")
        save.clicked.connect(self._save_dialog)
        self.status = QLabel("")
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Renk:"))
        bar.addWidget(self.mode)
        bar.addWidget(self.live)
        bar.addWidget(grab)
        bar.addWidget(reset)
        bar.addWidget(save)
        bar.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.refresh(force=True)

    def _tick(self):
        if self.live.isChecked():
            self.refresh()

    def refresh(self, force: bool = False) -> bool:
        """Show the block's newest frame; True when something is shown."""
        got = D.latest(self.node.id) if self.node is not None else None
        if got is not None and (force or got[0] != self._shown_n):
            self.frame = got
        if self.frame is None:
            self.status.setText("Henüz kare yok: akışı çalıştırın veya 'Kare al' deyin. Fare: sol = döndür, "
                                "sağ = kaydır, tekerlek = yakınlaştır, çift tık = sıfırla")
            return False
        if not force and self.frame[0] == self._shown_n:
            return True
        n, depth, color, intr, plane = self.frame
        try:
            pts, cols = D.cloud_for_view(depth, intr, color, plane, self.mode.currentText())
        except ValueError as exc:
            self.status.setText(str(exc))
            return False
        self._shown_n = n
        self.view.set_cloud(pts, cols)
        valid = int(np.isfinite(depth).sum())
        self.status.setText(f"Kare {n}  ·  {valid} geçerli nokta ({len(pts)} gösteriliyor)  ·  "
                            "sol = döndür, sağ = kaydır, tekerlek = yakınlaştır, çift tık = sıfırla")
        return True

    def grab_once(self) -> bool:
        """Run a copy of the block once (a source block only) and show what it gives."""
        from ..core import create
        node = self.node
        if node is None or node.type_name not in ("DepthCamera", "DepthTestScene"):
            self.status.setText("Bu blok kendi başına kare üretmez: akışı çalıştırın")
            return False
        copy = create(node.type_name, node.id)
        copy.values = dict(node.values)
        if not copy.safe_setup():
            self.status.setText(copy.setup_error or "Başlatılamadı")
            return False
        try:
            copy.process({})
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Kare alınamadı: {exc}")
            return False
        finally:
            copy.teardown()
        return self.refresh(force=True)

    def save_ply(self, path: str) -> str:
        if self.frame is None:
            raise ValueError("Kaydedilecek kare yok")
        _, depth, color, intr, _ = self.frame
        intr = D.intrinsics(intr, depth.shape)
        pts, (vs, us) = D.depth_to_points(depth, intr)
        cols = color[vs, us, :3] if isinstance(color, np.ndarray) and color.ndim == 3 and \
            color.shape[:2] == depth.shape and color.dtype == np.uint8 else None
        return D.write_ply(path, pts, cols, intr=intr, size=(depth.shape[1], depth.shape[0]))

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Nokta bulutunu kaydet", "nokta_bulutu.ply", "PLY (*.ply)")
        if not path:
            return
        try:
            self.status.setText(f"Kaydedildi: {self.save_ply(path)}")
        except (OSError, ValueError) as exc:
            self.status.setText(f"Kaydedilemedi: {exc}")

    def closeEvent(self, ev):
        self.timer.stop()
        super().closeEvent(ev)
