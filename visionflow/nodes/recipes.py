"""'Reçete Seç': switch the flow to another product type (recipe) while it runs."""
from __future__ import annotations

import numpy as np

from ..core import Node, Param, PortType, register

BASES = {"1'den": 1, "0'dan": 0}


@register
class RecipeSelect(Node):
    title = "Reçete Seç"
    category = "4 · Mantık"
    description = ("Akışı bir ürün tipine (reçeteye) geçirir: reçetedeki ayar değerleri ilgili bloklara yazılır (ör. "
                   "A ürünü: eşik 120, tolerans 10-12 mm). Reçeteler Araçlar → Reçeteler… penceresinde hazırlanır. "
                   "Girişe reçetenin adı (yazı) veya numarası (sayı, ör. PLC'den) bağlanabilir; giriş boşsa ayardaki "
                   "reçete kullanılır. Değer değişince yeni reçete bir sonraki karede, çalışırken geçerli olur.")
    inputs = [("recipe", PortType.ANY)]
    outputs = [("name", PortType.TEXT), ("index", PortType.NUMBER), ("changed", PortType.BOOL)]
    params = [Param("recipe", "str", "", "Etkin reçete (adı veya numarası)"),
              Param("base", "choice", "1'den", "Numaralar kaçtan başlar", choices=list(BASES))]

    def _flow(self):
        ref = getattr(self, "_owner", None)
        g = ref() if ref is not None else None
        if g is None:
            raise ValueError("Reçete Seç bloğu bir akışın içinde çalışmalı")
        return g

    def _request(self, wanted):
        r = self._flow().request_recipe(wanted, BASES.get(self.v("base"), 1))
        root = self._flow().root
        self.state.update(last=wanted, name=r["name"], index=root.recipes.index(r) + BASES.get(self.v("base"), 1),
                          changed=True)

    def setup(self):
        self.state.update(last=None, name="", index=0, changed=False, error=None)
        want = str(self.v("recipe") or "").strip()
        if want:
            try:                         # applied before the first frame; a wrong name shows on the block
                self._request(want)
            except ValueError as exc:
                self.state["error"] = str(exc)

    def process(self, inputs):
        value = inputs.get("recipe")
        if isinstance(value, np.generic):
            value = value.item()          # a numpy number from another block
        if isinstance(value, str):
            value = value.strip() or None
        if value is None:
            value = str(self.v("recipe") or "").strip() or None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int, float))):
            raise ValueError(f"Reçete adı (yazı) veya numarası (sayı) bekleniyor, gelen: {type(value).__name__}")
        if value is not None and value != self.state.get("last"):
            self._request(value)          # raises ValueError for an unknown recipe
        changed, self.state["changed"] = self.state.get("changed", False), False
        return {"name": self.state.get("name", ""), "index": self.state.get("index", 0), "changed": changed}
