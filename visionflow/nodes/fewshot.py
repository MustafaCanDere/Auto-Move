"""Az Örnekle Segmentasyon: a segmentation block trained by painting a few pictures (see visionflow/fewshot_seg.py
and the "Etiketle ve Eğit" window in ui/seg_teach_dialog.py)."""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .. import fewshot_seg as fs
from ..core import Node, Param, PortType, register

CAT = "3b · Öğrenme (Kendi Örneklerinle)"
IMG, NUM, TEXT, DET, ANY = PortType.IMAGE, PortType.NUMBER, PortType.TEXT, PortType.DETECTIONS, PortType.ANY

MASK_MODES = ["Seçili sınıf (ikili 0/255)", "Sınıf numarası (0,1,2…)"]

# the last picture each block saw (block id → BGR image), so the teaching window can paint on the flow's own input
_LAST_INPUTS: dict[str, np.ndarray] = {}
_LAST_LOCK = threading.Lock()


def last_input(node_id: str):
    with _LAST_LOCK:
        img = _LAST_INPUTS.get(node_id)
        return None if img is None else img.copy()


def _remember(node_id: str, img: np.ndarray):
    with _LAST_LOCK:
        _LAST_INPUTS[node_id] = img
        while len(_LAST_INPUTS) > 16:                       # bounded: deleted blocks do not pile up
            _LAST_INPUTS.pop(next(iter(_LAST_INPUTS)))


def target_index(model: fs.SegModel, target) -> int:
    """The class chosen in the settings: its name (case-insensitive), its number, or "" = the first class after the
    background."""
    t = str(target if target is not None else "").strip()
    if not t:
        return 1 if len(model.classes) > 1 else 0
    low = [c.casefold() for c in model.classes]
    if t.casefold() in low:
        return low.index(t.casefold())
    if t.lstrip("-").isdigit():
        i = int(t)
        if 0 <= i < len(model.classes):
            return i
    raise ValueError(f"Hedef sınıf '{t}' modelde yok. Sınıflar: {', '.join(model.classes)}")


@register
class FewShotSegment(Node):
    title = "Az Örnekle Segmentasyon"
    category = CAT
    description = ("Birkaç resmin üzerine fırçayla boyayarak (ör. 3–10 resim) kendi segmentasyon modelinizi "
                   "eğitin; GPU ve internet gerekmez. 'Etiketle ve Eğit…' düğmesiyle pencereyi açın, sınıfları "
                   "boyayın, Eğit'e basın, hatalı yerleri düzeltip yeniden eğitin ve modeli kaydedin. Blok her "
                   "pikselin sınıfını bulur: maske, renkli bindirme, sınıf alanları (%) ve hedef sınıfın nesne "
                   "sayısı/konturları.")
    inputs = [("image", IMG)]
    outputs = [("mask", IMG), ("overlay", IMG), ("area", NUM), ("areas", ANY), ("count", NUM),
               ("contours", PortType.CONTOURS), ("detections", DET)]
    params = [Param("model", "path", "", "Model dosyası (.vfseg)"),
              Param("target", "str", "", "Hedef sınıf (boş = arka plan dışındaki ilk sınıf)"),
              Param("mask_mode", "choice", MASK_MODES[0], "Maske çıkışı", choices=MASK_MODES),
              Param("min_area", "int", 50, "En küçük bölge (px, küçükleri silinir)", 0, 10_000_000),
              Param("smooth", "int", 1, "Yumuşatma", 0, 10),
              Param("alpha", "float", 0.45, "Bindirme saydamlığı", 0, 1, 0.05)]
    # the properties panel shows a button that opens this window (see MainWindow.open_block_dialog)
    dialog = ("🖌 Etiketle ve Eğit…", "seg_teach_dialog.SegTeachDialog")

    def setup(self):
        self.state.update(model=None, key=None)

    def _model(self) -> fs.SegModel:
        path = self.v("model")
        if not path:
            raise ValueError("Model dosyası seçilmedi: 'Etiketle ve Eğit…' düğmesiyle birkaç resmi boyayıp bir "
                             "model eğitin ve kaydedin")
        try:
            st = os.stat(path)
        except OSError:
            raise ValueError(f"Model dosyası yok: {path}") from None
        key = (path, st.st_mtime_ns, st.st_size)
        if self.state.get("key") != key:                   # a model saved again from the window: use the new one
            self.state["model"] = fs.SegModel.load(path)
            self.state["key"] = key
        return self.state["model"]

    def process(self, inputs):
        img = inputs["image"]
        bgr = fs.to_bgr8(img)
        _remember(self.id, bgr.copy() if bgr is img else bgr)
        model = self._model()
        ti = target_index(model, self.v("target"))
        min_area = int(self.v("min_area"))
        lab = model.predict(bgr, smooth=int(self.v("smooth")), min_area=min_area)
        binary = ((lab == ti) * 255).astype(np.uint8)
        mask = binary if self.v("mask_mode") == MASK_MODES[0] else lab
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= max(1, min_area)]
        name = model.classes[ti]
        dets = [{"box": tuple(int(v) for v in cv2.boundingRect(c)), "score": 1.0, "class_id": ti,
                 "label": f"{name} {int(cv2.contourArea(c))}px"} for c in cnts]
        pct = fs.class_areas(lab, len(model.classes))
        ov = fs.overlay(bgr, lab, model.colors, float(self.v("alpha")))
        if cnts:
            cv2.drawContours(ov, cnts, -1, tuple(int(v) for v in model.colors[ti]), 2)
        return {"mask": mask, "overlay": ov, "area": pct[ti], "areas": dict(zip(model.classes, pct)),
                "count": len(cnts), "contours": cnts, "detections": dets}
