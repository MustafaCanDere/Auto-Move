"""Az Örnekle Nesne Bul: an object detector taught by drawing boxes on a few pictures (see visionflow/fewshot_det.py
and the "Kutu Çiz ve Eğit" window in ui/det_teach_dialog.py). Its detections have the YOLO blocks' format, so
Tespitleri Çiz, Tespit Filtrele, Sınıfa Göre Say, Nesne Takibi … work with it."""
from __future__ import annotations

import os

from .. import fewshot_det as fd
from ..core import Node, Param, PortType, register
from .fewshot import CAT, _remember


@register
class FewShotDetect(Node):
    title = "Az Örnekle Nesne Bul"
    category = CAT
    description = ("Birkaç resimde nesnelerin etrafına kutu çizerek öğretilen nesne bulucu (internet ve GPU "
                   "gerekmez). Döndürülmüş ve biraz büyük/küçük görünen parçaları da bulur; YOLO bloklarıyla aynı "
                   "tespit çıkışını verir (Tespitleri Çiz, Sınıfa Göre Say, Nesne Takibi…). 'Kutu Çiz ve Eğit…' "
                   "düğmesiyle modeli eğitin.")
    inputs = [("image", PortType.IMAGE)]
    outputs = [("detections", PortType.DETECTIONS), ("count", PortType.NUMBER), ("image", PortType.IMAGE)]
    params = [Param("model", "path", "", "Model dosyası (.vfdet)"),
              Param("score", "float", 0.5, "Güven eşiği", 0.01, 1, 0.01),
              Param("max_objects", "int", 50, "En çok nesne", 1, 1000),
              Param("classes", "str", "", "Sınıflar (virgülle, boş = hepsi)")]
    # the properties panel shows a button that opens this window (see MainWindow.open_block_dialog)
    dialog = ("🔲 Kutu Çiz ve Eğit…", "det_teach_dialog.DetTeachDialog")

    def setup(self):
        self.state.update(model=None, key=None)

    def _model(self) -> fd.DetModel:
        path = self.v("model")
        if not path:
            raise ValueError("Model dosyası seçilmedi: 'Kutu Çiz ve Eğit…' düğmesiyle birkaç resimde nesnelerin "
                             "etrafına kutu çizip bir model eğitin ve kaydedin")
        try:
            st = os.stat(path)
        except OSError:
            raise ValueError(f"Model dosyası yok: {path}") from None
        key = (path, st.st_mtime_ns, st.st_size)
        if self.state.get("key") != key:                   # a model saved again from the window: use the new one
            self.state["model"] = fd.DetModel.load(path)
            self.state["key"] = key
        return self.state["model"]

    def process(self, inputs):
        img = inputs["image"]
        bgr = fd.to_bgr8(img)
        _remember(self.id, bgr.copy() if bgr is img else bgr)
        model = self._model()
        dets = model.detect(bgr, score=float(self.v("score")), max_objects=int(self.v("max_objects")),
                            classes=self.v("classes"))
        return {"detections": dets, "count": len(dets), "image": fd.draw(bgr, dets, model.colors)}
