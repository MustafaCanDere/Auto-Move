"""Örnek eklenti: bu klasöre koyduğunuz her .py dosyası açılışta yüklenir ve
bloklar paletinde görünür. Kendi bloğunuzu yazmak için bu dosyayı kopyalayın."""
import cv2

from visionflow.core import Node, Param, PortType, register


@register
class Negatif(Node):
    title = "Negatif (örnek eklenti)"
    category = "6 · Eklentiler"
    description = "Görüntünün renklerini ters çevirir."
    inputs = [("image", PortType.IMAGE)]
    outputs = [("image", PortType.IMAGE)]
    params = [Param("enabled", "bool", True, "Açık")]

    def process(self, inputs):
        img = inputs["image"]
        if img is None:
            raise ValueError("Görüntü girişi bağlı değil")
        return {"image": cv2.bitwise_not(img) if self.v("enabled") else img}
