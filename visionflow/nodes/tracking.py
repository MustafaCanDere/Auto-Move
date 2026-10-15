"""Object tracking, line/zone counting and image classification."""
from __future__ import annotations

import cv2
import numpy as np

from ._text import put_text
from .. import accel
from ..core import Node, Param, PortType, register
from .ai import model_names

CAT = "3 · Yapay Zeka"
DET, NUM, IMG = PortType.DETECTIONS, PortType.NUMBER, PortType.IMAGE


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


@register
class ObjectTracker(Node):
    title = "Nesne Takibi"
    category = CAT
    description = ("Tespitlere kareler arasında kalıcı kimlik (ID) verir. Aynı nesnenin tekrar sayılmasını "
                   "önler; çizgi geçiş sayımı için gereklidir.")
    inputs = [("detections", DET)]
    outputs = [("detections", DET), ("count", NUM), ("total", NUM)]
    params = [Param("max_lost", "int", 15, "Kaybolma toleransı (kare)", 0, 1000),
              Param("max_dist", "int", 120, "Max hareket (px)", 1, 5000)]

    def setup(self):
        self.state.update(tracks={}, next_id=1)

    def process(self, inputs):
        dets = [dict(d) for d in (inputs["detections"] or [])]
        tracks: dict[int, dict] = self.state["tracks"]
        # greedy matching: highest IoU first, then nearest center
        pairs = []
        for tid, t in tracks.items():
            tc = _center(t["box"])
            for j, d in enumerate(dets):
                if d["class_id"] != t["class_id"]:
                    continue
                dc = _center(d["box"])
                dist = float(np.hypot(tc[0] - dc[0], tc[1] - dc[1]))
                if dist <= self.v("max_dist"):
                    pairs.append((-_iou(t["box"], d["box"]), dist, tid, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, _, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            t = tracks[tid]
            t["prev"] = _center(t["box"])
            t.update(box=dets[j]["box"], lost=0, age=t["age"] + 1)
            dets[j]["track_id"] = tid
        for tid in list(tracks):
            if tid not in used_t:
                tracks[tid]["lost"] += 1
                tracks[tid]["prev"] = _center(tracks[tid]["box"])
                if tracks[tid]["lost"] > self.v("max_lost"):
                    del tracks[tid]
        for j, d in enumerate(dets):
            if j not in used_d:
                tid = self.state["next_id"]
                self.state["next_id"] += 1
                tracks[tid] = {"box": d["box"], "class_id": d["class_id"], "lost": 0, "age": 1,
                               "prev": _center(d["box"])}
                d["track_id"] = tid
        for d in dets:
            t = tracks[d["track_id"]]
            d["prev_center"] = t["prev"]
            d["label"] = f'{d["label"]} #{d["track_id"]}'
        return {"detections": dets, "count": len(dets), "total": self.state["next_id"] - 1}


def _side(p, a, b) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@register
class LineCounter(Node):
    roi = ("line", "line")
    title = "Çizgi Geçiş Sayacı"
    category = CAT
    description = ("Takip edilen nesnelerin bir çizgiyi hangi yönde geçtiğini sayar (giren / çıkan; yönü "
                   "çevirmek için çizginin uçlarını ters yazın). "
                   "Girişi 'Nesne Takibi' bloğundan alın.")
    inputs = [("detections", DET), ("image", IMG), ("reset", PortType.BOOL)]
    outputs = [("in", NUM), ("out", NUM), ("total", NUM), ("crossed", PortType.BOOL), ("image", IMG)]
    params = [Param("line", "str", "0,240,640,240", "Çizgi x1,y1,x2,y2")]

    FORGET_AFTER = 300      # frames: ids not seen this long are dropped (a conveyor makes thousands of ids a day)

    def setup(self):
        self.state.update(n_in=0, n_out=0, counted=set(), side={}, seen={}, frame=0)

    def _forget_old(self):
        frame, seen = self.state["frame"], self.state["seen"]
        old = {i for i, f in seen.items() if frame - f > self.FORGET_AFTER}
        if old:
            for i in old:
                del seen[i]
                self.state["side"].pop(i, None)
            self.state["counted"] = {k for k in self.state["counted"] if k[0] not in old}

    def process(self, inputs):
        if inputs["reset"]:
            self.state.update(n_in=0, n_out=0, counted=set(), side={}, seen={})
        self.state["frame"] += 1
        if self.state["frame"] % 50 == 0:
            self._forget_old()
        x1, y1, x2, y2 = (int(v) for v in self.v("line").split(","))
        a, b = (x1, y1), (x2, y2)
        crossed = False
        for d in inputs["detections"] or []:
            if "track_id" not in d or "prev_center" not in d:
                continue
            self.state["seen"][d["track_id"]] = self.state["frame"]
            prev, now = d["prev_center"], _center(d["box"])
            s0, s1 = _side(prev, a, b), _side(now, a, b)
            # a centre exactly on the line has no side: use the last side the object was seen on, so
            # 230 → 240 → 250 across y=240 still counts once
            last = self.state["side"]
            if s0 == 0:
                s0 = last.get(d["track_id"], 0)
            if s1 != 0 or s0 != 0:
                last[d["track_id"]] = s1 or s0
            if s0 == 0 or s1 == 0 or (s0 > 0) == (s1 > 0):
                continue
            # only count crossings that happen within the segment's extent
            seg = np.array(b, float) - np.array(a, float)
            t = np.dot(np.array(now) - np.array(a), seg) / max(np.dot(seg, seg), 1e-9)
            if not -0.05 <= t <= 1.05:
                continue
            key = (d["track_id"], s1 > 0)
            if key in self.state["counted"]:
                continue
            self.state["counted"].add(key)
            self.state["n_in" if s1 > 0 else "n_out"] += 1
            crossed = True
        img = inputs["image"]
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
            cv2.line(img, a, b, (0, 0, 255) if crossed else (0, 220, 255), 2)
            put_text(img, f"giren {self.state['n_in']}  çıkan {self.state['n_out']}", (10, 30), 0.8, (0, 255, 255), 2)
        n_in, n_out = self.state["n_in"], self.state["n_out"]
        return {"in": n_in, "out": n_out, "total": n_in + n_out, "crossed": crossed, "image": img}


@register
class ImageClassifier(Node):
    setup_params = ("model", "classes", "size", "device")
    title = "Görüntü Sınıflandır"
    category = CAT
    description = ("ONNX sınıflandırma modeli ile görüntüyü sınıflandırır (sağlam / hatalı, ürün tipi…). "
                   "YOLO-cls, ResNet, MobileNet, EfficientNet ve Teachable Machine modelleriyle çalışır.")
    inputs = [("image", IMG)]
    outputs = [("label", PortType.TEXT), ("score", NUM), ("class_id", NUM), ("device", PortType.TEXT)]
    params = [Param("model", "path", "", "ONNX model"),
              Param("classes", "str", "", "Sınıf adları (virgülle, boşsa modelden)"),
              Param("size", "int", 224, "Giriş boyutu", 16, 1024),
              Param("normalize", "choice", "0-1", "Normalizasyon", choices=["0-1", "imagenet", "-1..1"]),
              Param("rgb", "bool", True, "RGB sırası"),
              Param("device", "choice", accel.GLOBAL, "Cihaz", choices=accel.BLOCK_CHOICES)]

    def setup(self):
        if not self.v("model"):
            raise ValueError("ONNX model seçilmedi")
        sess = accel.node_session(self, self.v("model"))
        names = [s.strip() for s in self.v("classes").split(",") if s.strip()]
        if not names:
            names = model_names(sess.get_modelmeta().custom_metadata_map.get("names"))
        inp = sess.get_inputs()[0]
        nhwc = isinstance(inp.shape[-1], int) and inp.shape[-1] == 3
        size = inp.shape[1 if nhwc else 2]
        self.state.update(sess=sess, names=names, input=inp.name, nhwc=nhwc,
                          size=size if isinstance(size, int) else self.v("size"), device=accel.device_of(sess))

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        st = self.state
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        x = cv2.resize(img, (st["size"], st["size"])).astype(np.float32)
        if self.v("rgb"):
            x = x[:, :, ::-1]
        x = x / 255.0
        if self.v("normalize") == "imagenet":
            x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        elif self.v("normalize") == "-1..1":
            x = x * 2 - 1
        x = x.astype(np.float32)[None] if st["nhwc"] else x.transpose(2, 0, 1).astype(np.float32)[None]
        out = np.asarray(st["sess"].run(None, {st["input"]: x})[0]).reshape(-1)
        if out.min() < 0 or out.sum() > 1.01 or out.sum() < 0.99:   # logits -> softmax
            e = np.exp(out - out.max())
            out = e / e.sum()
        i = int(out.argmax())
        label = st["names"][i] if i < len(st["names"]) else str(i)
        return {"label": label, "score": round(float(out[i]), 4), "class_id": i, "device": st["device"]}
