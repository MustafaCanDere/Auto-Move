"""AI / deep learning blocks: YOLO (ONNX) detection, face detection, detection utilities."""
from __future__ import annotations

import ast

import cv2
import numpy as np

from ._text import put_text, text_size
from .. import accel
from ..core import Node, Param, PortType, register

CAT = "3 · Yapay Zeka"
DET = PortType.DETECTIONS

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()


def model_names(meta: str | None) -> list[str]:
    """Class names from ONNX metadata: Ultralytics writes a dict {0: 'person', ...}, some exports a list."""
    if not meta:
        return []
    try:
        names = ast.literal_eval(meta)
    except (ValueError, SyntaxError):
        return [s.strip() for s in meta.split(",") if s.strip()]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    return [str(n) for n in names] if isinstance(names, (list, tuple)) else []


def letterbox(img, size):
    """Resize keeping aspect ratio and pad to size x size, or to (height, width) for rectangular models.
    Returns (img, scale, (pad_x, pad_y))."""
    th, tw = (size, size) if isinstance(size, int) else size
    h, w = img.shape[:2]
    s = min(th / h, tw / w)
    nh, nw = round(h * s), round(w * s)
    out = np.full((th, tw, 3), 114, np.uint8)
    px, py = (tw - nw) // 2, (th - nh) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return out, s, (px, py)


def yolo_postprocess(out: np.ndarray, conf: float, iou: float, scale: float, pad, names) -> list[dict]:
    """Decode raw YOLO output: v8/v11 (1, 4+nc, N), v5/v7 (1, N, 5+nc) and end-to-end models such as
    YOLOv10 / YOLO26 (1, N, 6) = x1, y1, x2, y2, score, class (already filtered, no NMS needed)."""
    out = np.squeeze(out, 0)
    nc = len(names)
    rows, cols = out.shape
    if cols == 6 and rows != 4 + nc and (not (cols == 5 + nc or cols == 4 + nc) or _looks_end_to_end(out)):
        return _end_to_end(out, conf, scale, pad, names)
    # decide the layout from the class count when it is known; the old shape guess fails for small inputs
    if nc and rows == 4 + nc and cols != 5 + nc:
        v8 = True
    elif nc and cols == 5 + nc:
        v8 = False
    elif nc and cols == 4 + nc:
        out, v8 = out.T, True
    else:
        v8 = rows < cols
    if v8:                                    # v8 layout -> (N, 4+nc)
        out = out.T
        boxes, scores_all = out[:, :4], out[:, 4:]
    else:                                     # v5 layout -> objectness * class score
        boxes, scores_all = out[:, :4], out[:, 5:] * out[:, 4:5]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls)), cls]
    keep = scores >= conf
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    if not len(boxes):
        return []
    xywh = np.empty_like(boxes)
    xywh[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad[0]) / scale
    xywh[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad[1]) / scale
    xywh[:, 2] = boxes[:, 2] / scale
    xywh[:, 3] = boxes[:, 3] / scale
    idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), cls.tolist(), conf, iou)
    dets = []
    for i in np.array(idx).flatten():
        c = int(cls[i])
        dets.append({"box": tuple(int(v) for v in xywh[i]), "score": float(scores[i]), "class_id": c,
                     "label": names[c] if c < len(names) else str(c)})
    return dets


def _looks_end_to_end(out) -> bool:
    """(N, 6) is ambiguous for 1- and 2-class models (v5: 5+1, v8 transposed: 4+2). End-to-end rows are
    x1, y1, x2, y2, score, class: the last column holds whole class numbers and boxes are corner-ordered."""
    live = out[out[:, 4] > 0]
    if not len(live):
        return bool(np.all(out == 0))
    return bool(np.all(live[:, 5] == np.round(live[:, 5])) and np.all(live[:, 4] <= 1)
                and np.all(live[:, 2] >= live[:, 0]) and np.all(live[:, 3] >= live[:, 1]))


def _end_to_end(out, conf, scale, pad, names) -> list[dict]:
    dets = []
    for x1, y1, x2, y2, score, c in out:
        if score < conf:
            continue
        c = int(c)
        x, y = (x1 - pad[0]) / scale, (y1 - pad[1]) / scale
        dets.append({"box": (int(x), int(y), int((x2 - x1) / scale), int((y2 - y1) / scale)), "score": float(score),
                     "class_id": c, "label": names[c] if 0 <= c < len(names) else str(c)})
    return dets


@register
class YoloDetector(Node):
    setup_params = ("model", "classes", "size", "device")
    title = "YOLO Nesne Tespiti"
    category = CAT
    description = ("YOLOv5/v8/v11 ONNX modeli ile nesne tespiti. Kendi eğittiğiniz modeli de yükleyebilirsiniz. "
                   "GPU varsa CUDA/DirectML otomatik kullanılır (Araçlar → Hızlandırma).")
    inputs = [("image", PortType.IMAGE)]
    outputs = [("detections", DET), ("count", PortType.NUMBER), ("device", PortType.TEXT)]
    params = [
        Param("model", "path", "", "ONNX model"),
        Param("conf", "float", 0.4, "Güven eşiği", 0.01, 1, 0.01),
        Param("iou", "float", 0.45, "NMS IoU", 0.01, 1, 0.01),
        Param("size", "int", 640, "Giriş boyutu", 32, 2048, 32),
        Param("classes", "str", "", "Sınıf adları (virgülle, boşsa modelden)"),
        Param("device", "choice", accel.GLOBAL, "Cihaz", choices=accel.BLOCK_CHOICES),
    ]

    def setup(self):
        path = self.v("model")
        if not path:
            raise ValueError("ONNX model seçilmedi")
        sess = accel.node_session(self, path)
        names = [s.strip() for s in self.v("classes").split(",") if s.strip()]
        if not names:
            names = model_names(sess.get_modelmeta().custom_metadata_map.get("names")) or COCO
        shape = sess.get_inputs()[0].shape
        h = shape[2] if len(shape) > 3 and isinstance(shape[2], int) else self.v("size")
        w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else h
        # rectangular fixed-size models (e.g. 384x640) need a rectangular letterbox
        self.state.update(sess=sess, names=names, size=h if h == w else (h, w), input=sess.get_inputs()[0].name,
                          device=accel.device_of(sess))

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        st = self.state
        lb, scale, pad = letterbox(img, st["size"])
        blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = st["sess"].run(None, {st["input"]: blob})[0]
        dets = yolo_postprocess(out, self.v("conf"), self.v("iou"), scale, pad, st["names"])
        return {"detections": dets, "count": len(dets), "device": st["device"]}


@register
class FaceDetector(Node):
    title = "Yüz Tespiti"
    category = CAT
    description = "Model indirmeden çalışan hızlı yüz tespiti (Haar)."
    inputs = [("image", PortType.IMAGE)]
    outputs = [("detections", DET), ("count", PortType.NUMBER)]
    params = [Param("min_size", "int", 40, "Min yüz boyutu", 10, 1000)]

    def setup(self):
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError("Bu OpenCV sürümünde Haar yok; 'pip install \"opencv-python<5\"' kurun")
        self.state["cc"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        m = self.v("min_size")
        faces = self.state["cc"].detectMultiScale(g, 1.1, 5, minSize=(m, m))
        dets = [{"box": tuple(int(v) for v in f), "score": 1.0, "class_id": 0, "label": "yüz"} for f in faces]
        return {"detections": dets, "count": len(dets)}


@register
class FilterDetections(Node):
    title = "Tespit Filtrele"
    category = CAT
    description = "Sınıf adına, güvene ve bölgeye göre tespitleri süzer."
    inputs = [("detections", DET)]
    outputs = [("detections", DET), ("count", PortType.NUMBER), ("any", PortType.BOOL)]
    params = [
        Param("labels", "str", "", "Sınıflar (virgülle, boş = hepsi)"),
        Param("min_score", "float", 0.0, "Min güven", 0, 1, 0.01),
        Param("zone", "str", "", "Bölge x,y,w,h (boş = tüm görüntü)"),
    ]

    def process(self, inputs):
        dets = inputs["detections"] or []
        labels = {s.strip() for s in self.v("labels").split(",") if s.strip()}
        zone = [int(v) for v in self.v("zone").split(",")] if self.v("zone").strip() else None
        out = []
        for d in dets:
            if labels and d["label"] not in labels:
                continue
            if d["score"] < self.v("min_score"):
                continue
            if zone:
                x, y, w, h = d["box"]
                cx, cy = x + w / 2, y + h / 2
                zx, zy, zw, zh = zone
                if not (zx <= cx <= zx + zw and zy <= cy <= zy + zh):
                    continue
            out.append(d)
        return {"detections": out, "count": len(out), "any": bool(out)}


def _color(i: int) -> tuple[int, int, int]:
    hue = (i * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


@register
class DrawDetections(Node):
    title = "Tespitleri Çiz"
    category = CAT
    inputs = [("image", PortType.IMAGE), ("detections", DET)]
    outputs = [("image", PortType.IMAGE)]
    params = [Param("show_score", "bool", True, "Güveni göster"), Param("thickness", "int", 2, "Kalınlık", 1, 10)]

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        from .ai_more import draw_extras             # rotated boxes and skeletons (imports this module)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        for d in inputs["detections"] or []:
            x, y, w, h = d["box"]
            color = _color(d["class_id"])
            if d.get("poly") or d.get("keypoints"):
                draw_extras(img, d, color, self.v("thickness"))
            if not d.get("poly"):
                cv2.rectangle(img, (x, y), (x + w, y + h), color, self.v("thickness"))
            text = f'{d["label"]} {d["score"]:.2f}' if self.v("show_score") else d["label"]
            tw, th = text_size(text, 0.5)
            cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
            put_text(img, text, (x + 2, y - 4), 0.5, (0, 0, 0), 1)
        return {"image": img}


def yolo_seg_postprocess(out, protos, conf, iou, scale, pad, names, img_shape, size):
    """Decode YOLOv8/11-seg output (1, 4+nc+32, N) + prototypes (1, 32, mh, mw)."""
    out = np.squeeze(out, 0).T                      # (N, 4+nc+32)
    nm = protos.shape[1]
    nc = out.shape[1] - 4 - nm
    boxes, scores_all, coefs = out[:, :4], out[:, 4:4 + nc], out[:, 4 + nc:]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls)), cls]
    keep = scores >= conf
    boxes, scores, cls, coefs = boxes[keep], scores[keep], cls[keep], coefs[keep]
    h, w = img_shape[:2]
    full = np.zeros((h, w), np.uint8)
    if not len(boxes):
        return [], full
    xywh = np.empty_like(boxes)
    xywh[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad[0]) / scale
    xywh[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad[1]) / scale
    xywh[:, 2:] = boxes[:, 2:] / scale
    idx = np.array(cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), cls.tolist(), conf, iou)).flatten()
    p = np.squeeze(protos, 0)
    mh, mw = p.shape[1:]
    dets = []
    for k, i in enumerate(idx):
        m = 1 / (1 + np.exp(-(coefs[i] @ p.reshape(nm, -1)))).reshape(mh, mw)
        sh, sw = (size, size) if isinstance(size, int) else size
        m = cv2.resize(m, (sw, sh))
        py, px = int(pad[1]), int(pad[0])
        m = m[py:sh - py, px:sw - px]
        m = cv2.resize(m, (w, h)) > 0.5
        x, y, bw, bh = (int(v) for v in xywh[i])
        box_mask = np.zeros_like(m)
        box_mask[max(0, y):y + bh, max(0, x):x + bw] = True
        m &= box_mask
        full[m] = min(255, 40 + k * 20)
        c = int(cls[i])
        dets.append({"box": (x, y, bw, bh), "score": float(scores[i]), "class_id": c,
                     "label": names[c] if c < len(names) else str(c), "area": int(m.sum())})
    return dets, full


@register
class YoloSegment(YoloDetector):
    title = "YOLO Segmentasyon"
    description = ("YOLOv8/v11-seg ONNX modeli ile nesneleri piksel düzeyinde ayırır. Maske ve alan (px) "
                   "verir; kusur alanı ölçmek, şekilli parçaları ayırmak için.")
    outputs = [("detections", DET), ("mask", PortType.IMAGE), ("overlay", PortType.IMAGE), ("count", PortType.NUMBER),
               ("device", PortType.TEXT)]

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        st = self.state
        lb, scale, pad = letterbox(img, st["size"])
        blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outs = st["sess"].run(None, {st["input"]: blob})
        if len(outs) < 2:
            raise ValueError("Bu model segmentasyon modeli değil (-seg.onnx kullanın)")
        dets, mask = yolo_seg_postprocess(outs[0], outs[1], self.v("conf"), self.v("iou"), scale, pad,
                                          st["names"], img.shape, st["size"])
        overlay = img.copy()
        for k, d in enumerate(dets):
            sel = mask == min(255, 40 + k * 20)
            overlay[sel] = (0.5 * overlay[sel] + 0.5 * np.array(_color(d["class_id"]))).astype(np.uint8)
        return {"detections": dets, "mask": mask, "overlay": overlay, "count": len(dets), "device": st["device"]}
