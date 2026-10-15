"""More YOLO tasks: pose estimation (keypoints) and oriented / rotated boxes (OBB), both from Ultralytics ONNX models."""
from __future__ import annotations

import ast
import math

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from .ai import CAT, DET, YoloDetector, letterbox

# COCO person keypoints and the lines between them
KPT_NAMES = ("burun sol_göz sağ_göz sol_kulak sağ_kulak sol_omuz sağ_omuz sol_dirsek sağ_dirsek sol_bilek "
             "sağ_bilek sol_kalça sağ_kalça sol_diz sağ_diz sol_ayak_bileği sağ_ayak_bileği").split()
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
DOTA = ("plane ship storage_tank baseball_diamond tennis_court basketball_court ground_track_field harbor bridge "
        "large_vehicle small_vehicle helicopter roundabout soccer_ball_field swimming_pool").split()


def _blob(img):
    return img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def _bgr(img):
    """Any picture -> 8-bit, 3-channel BGR (grey, with alpha, 16-bit or float pictures all work)."""
    if img is None:
        raise ValueError("Görüntü girişi bağlı değil")
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(img[:, :, :3])


def _covariance(b):
    a, c = b[:, 2] ** 2 / 12, b[:, 3] ** 2 / 12
    cos, sin = np.cos(b[:, 4]), np.sin(b[:, 4])
    return a * cos ** 2 + c * sin ** 2, a * sin ** 2 + c * cos ** 2, (a - c) * cos * sin


def probiou(b1, b2, eps=1e-7):
    """Probabilistic IoU of rotated boxes (cx, cy, w, h, radians), as Ultralytics uses for OBB NMS."""
    x1, y1, x2, y2 = b1[:, 0:1], b1[:, 1:2], b2[None, :, 0], b2[None, :, 1]
    a1, b_1, c1 = (v[:, None] for v in _covariance(b1))
    a2, b_2, c2 = (v[None] for v in _covariance(b2))
    den = (a1 + a2) * (b_1 + b_2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b_1 + b_2) * (x1 - x2) ** 2) / den * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / den * 0.5
    t3 = np.log(den / (4 * np.sqrt(np.clip(a1 * b_1 - c1 ** 2, 0, None) * np.clip(a2 * b_2 - c2 ** 2, 0, None))
                       + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    return 1 - np.sqrt(1 - np.exp(-bd) + eps)


def nms_rotated(boxes, scores, iou):
    """Ultralytics' rotated NMS: a box stays when it overlaps no higher-scoring box by more than `iou`."""
    order = np.argsort(-scores)[:3000]
    b = boxes[order]
    ious = np.triu(probiou(b, b), 1)
    return order[ious.max(0) < iou] if len(b) else order


def kpt_shape(meta: str | None) -> tuple[int, int]:
    try:
        k, d = ast.literal_eval(meta or "")
        return int(k), int(d)
    except (ValueError, SyntaxError, TypeError):
        return 17, 3


def pose_postprocess(out, conf, iou, scale, pad, names, kshape=(17, 3)) -> list[dict]:
    """YOLOv8/11-pose (1, 4+nc+K*D, N) or end-to-end YOLO26-pose (1, N, 6+K*D) -> detections with keypoints."""
    out = np.squeeze(out, 0)
    k, d = kshape
    nk = k * d
    nc = max(1, len(names))
    if out.shape[0] == 4 + nc + nk:                      # v8 layout
        rows = out.T
        boxes, scores_all, kpts = rows[:, :4], rows[:, 4:4 + nc], rows[:, 4 + nc:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        xy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]], 1)
        e2e = False
    elif out.shape[1] == 6 + nk:                          # end-to-end: x1, y1, x2, y2, score, class, kpts
        rows = out
        xy = np.stack([rows[:, 0], rows[:, 1], rows[:, 2] - rows[:, 0], rows[:, 3] - rows[:, 1]], 1)
        scores, cls, kpts = rows[:, 4], rows[:, 5].astype(int), rows[:, 6:]
        e2e = True
    else:
        raise ValueError(f"Bu bir poz modeli değil (çıkış {tuple(out.shape)}); -pose.onnx kullanın")
    keep = scores >= conf
    xy, scores, cls, kpts = xy[keep], scores[keep], cls[keep], kpts[keep]
    if not len(xy):
        return []
    xy = xy.copy()
    xy[:, 0] -= pad[0]
    xy[:, 1] -= pad[1]
    xy /= scale
    idx = range(len(xy)) if e2e else np.array(
        cv2.dnn.NMSBoxesBatched(xy.tolist(), scores.tolist(), cls.tolist(), conf, iou)).flatten()
    dets = []
    for i in idx:
        p = kpts[i].reshape(k, d)
        pts = [(float((p[j, 0] - pad[0]) / scale), float((p[j, 1] - pad[1]) / scale),
                float(p[j, 2]) if d > 2 else 1.0) for j in range(k)]
        c = int(cls[i])
        dets.append({"box": tuple(int(round(v)) for v in xy[i]), "score": float(scores[i]), "class_id": c,
                     "label": names[c] if 0 <= c < len(names) else str(c), "keypoints": pts})
    return dets


def obb_postprocess(out, conf, iou, scale, pad, names) -> list[dict]:
    """YOLOv8/11-obb (1, 4+nc+1, N): cx, cy, w, h, class scores, angle (radians) -> rotated detections.
    Each detection keeps an axis-aligned "box" (so every other block works with it) plus "poly" (4 corners),
    "angle" (degrees), "center" and "size"."""
    out = np.squeeze(out, 0)
    nc = len(names)
    e2e = out.ndim == 2 and out.shape[1] == 7 and out.shape[0] != 5 + nc     # YOLO26: x, y, w, h, score, cls, angle
    if e2e:
        boxes, scores, cls, ang = out[:, :4], out[:, 4], out[:, 5].astype(int), out[:, 6]
    else:
        if nc and out.shape[0] != 5 + nc and out.shape[1] == 5 + nc:
            out = out.T
        if out.shape[0] < 6:
            raise ValueError(f"Bu bir OBB modeli değil (çıkış {tuple(out.shape)}); -obb.onnx kullanın")
        rows = out.T
        nc = out.shape[0] - 5
        boxes, scores_all, ang = rows[:, :4], rows[:, 4:4 + nc], rows[:, 4 + nc]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
    keep = scores >= conf
    boxes, scores, cls, ang = boxes[keep], scores[keep], cls[keep], ang[keep]
    if not len(boxes):
        return []
    cx, cy = (boxes[:, 0] - pad[0]) / scale, (boxes[:, 1] - pad[1]) / scale
    w, h = boxes[:, 2] / scale, boxes[:, 3] / scale
    deg = np.degrees(ang)
    if e2e:
        idx = np.argsort(-scores)
    else:                    # per class: shift each class far away so classes never suppress each other
        shifted = np.stack([cx + cls * 7680.0, cy + cls * 7680.0, w, h, ang], 1).astype(np.float64)
        idx = nms_rotated(shifted, scores, iou)
    dets = []
    for i in idx:
        poly = cv2.boxPoints(((float(cx[i]), float(cy[i])), (float(w[i]), float(h[i])), float(deg[i])))
        x, y, bw, bh = cv2.boundingRect(np.round(poly).astype(np.int32))
        c = int(cls[i])
        a = float(deg[i]) % 180.0
        dets.append({"box": (x, y, bw, bh), "score": float(scores[i]), "class_id": c,
                     "label": names[c] if c < len(names) else str(c),
                     "poly": [(float(px), float(py)) for px, py in poly], "angle": round(a, 2),
                     "center": (float(cx[i]), float(cy[i])), "size": (float(w[i]), float(h[i]))})
    return dets


@register
class YoloPose(YoloDetector):
    title = "YOLO Poz Tahmini"
    description = ("YOLOv8/v11 -pose ONNX modeli ile insanların iskeletini (17 eklem noktası: omuz, dirsek, bilek, "
                   "diz…) bulur. Ergonomi, güvenlik (yerde yatan kişi, kol kaldırma), hareket sayma için. Kendi "
                   "eğittiğiniz anahtar nokta modelleri de çalışır.")
    outputs = [("detections", DET), ("count", PortType.NUMBER), ("device", PortType.TEXT)]

    def setup(self):
        super().setup()
        meta = self.state["sess"].get_modelmeta().custom_metadata_map
        self.state["kshape"] = kpt_shape(meta.get("kpt_shape"))
        if not self.v("classes").strip() and not meta.get("names"):
            self.state["names"] = ["person"]

    def process(self, inputs):
        img = _bgr(inputs["image"])
        st = self.state
        lb, scale, pad = letterbox(img, st["size"])
        out = st["sess"].run(None, {st["input"]: _blob(lb)})[0]
        dets = pose_postprocess(out, self.v("conf"), self.v("iou"), scale, pad, st["names"], st["kshape"])
        return {"detections": dets, "count": len(dets), "device": st["device"]}


@register
class YoloOBB(YoloDetector):
    title = "YOLO Döndürülmüş Kutu (OBB)"
    description = ("YOLOv8/v11 -obb ONNX modeli ile nesneleri döndürülmüş kutuyla bulur ve açısını verir: bantta "
                   "yamuk duran kutu, eğik vida, hava fotoğrafında gemi/araç. Her tespit 'angle' (derece) taşır.")
    outputs = [("detections", DET), ("count", PortType.NUMBER), ("angle", PortType.NUMBER), ("device", PortType.TEXT)]

    def setup(self):
        super().setup()
        meta = self.state["sess"].get_modelmeta().custom_metadata_map
        if not self.v("classes").strip() and not meta.get("names"):
            self.state["names"] = DOTA

    def process(self, inputs):
        img = _bgr(inputs["image"])
        st = self.state
        lb, scale, pad = letterbox(img, st["size"])
        out = st["sess"].run(None, {st["input"]: _blob(lb)})[0]
        dets = obb_postprocess(out, self.v("conf"), self.v("iou"), scale, pad, st["names"])
        best = max(dets, key=lambda d: d["score"]) if dets else None
        return {"detections": dets, "count": len(dets), "angle": best["angle"] if best else 0.0,
                "device": st["device"]}


@register
class PoseAngle(Node):
    title = "Eklem Açısı"
    category = CAT
    description = ("Poz tahmininden üç eklem noktasının yaptığı açıyı ölçer (ör. omuz-dirsek-bilek = dirsek "
                   "açısı). En emin kişiyi kullanır. Ergonomi kontrolü veya tekrar (squat, kaldırma) sayımı için.")
    inputs = [("detections", DET)]
    outputs = [("angle", PortType.NUMBER), ("found", PortType.BOOL)]
    params = [Param("a", "choice", "sağ_omuz", "Nokta 1", choices=list(KPT_NAMES)),
              Param("b", "choice", "sağ_dirsek", "Köşe noktası", choices=list(KPT_NAMES)),
              Param("c", "choice", "sağ_bilek", "Nokta 3", choices=list(KPT_NAMES)),
              Param("min_vis", "float", 0.5, "Min görünürlük", 0, 1, 0.05)]

    def process(self, inputs):
        people = [d for d in inputs["detections"] or [] if d.get("keypoints")]
        if not people:
            return {"angle": 0.0, "found": False}
        kp = max(people, key=lambda d: d["score"])["keypoints"]
        ids = [KPT_NAMES.index(self.v(n)) for n in ("a", "b", "c")]
        if max(ids) >= len(kp) or any(kp[i][2] < self.v("min_vis") for i in ids):
            return {"angle": 0.0, "found": False}
        (ax, ay, _), (bx, by, _), (cx, cy, _) = (kp[i] for i in ids)
        v1, v2 = (ax - bx, ay - by), (cx - bx, cy - by)
        n = math.hypot(*v1) * math.hypot(*v2)
        if n == 0:
            return {"angle": 0.0, "found": False}
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
        return {"angle": round(math.degrees(math.acos(cos)), 1), "found": True}


def draw_extras(img, d, color, thickness, min_vis=0.5):
    """Rotated box and skeleton, used by Tespitleri Çiz."""
    if d.get("poly"):
        cv2.polylines(img, [np.round(np.array(d["poly"])).astype(np.int32)], True, color, thickness)
    kp = d.get("keypoints")
    if kp:
        vis = [k[2] >= min_vis for k in kp]
        if len(kp) == 17:
            for a, b in SKELETON:
                if vis[a] and vis[b]:
                    cv2.line(img, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), color, thickness)
        for (x, y, _), v in zip(kp, vis):
            if v:
                cv2.circle(img, (int(x), int(y)), thickness + 2, (255, 255, 255), -1)
                cv2.circle(img, (int(x), int(y)), thickness + 2, color, 1)
