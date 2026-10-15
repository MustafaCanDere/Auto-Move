"""Part traceability blocks: 'Parçayı Arşivle' writes every inspected part (picture, serial, verdict, values) to the
archive of visionflow/trace.py; 'Arşivde Ara' finds the last record of a serial number at a re-check station."""
from __future__ import annotations

import pathlib

import numpy as np

from ..core import Node, Param, PortType, register
from ..core import node as _node
from .. import trace

CAT = "5 · Çıkış ve Otomasyon"


def _folder(node: Node) -> pathlib.Path:
    raw = str(node.values.get("folder") or "").strip()
    return pathlib.Path(_node.resolve_path(raw)) if raw else trace.default_root()


def flow_archive(graph) -> pathlib.Path | None:
    """The archive folder the flow's 'Parçayı Arşivle' block (inside sub-flows too) writes to; None without one.
    The archive window opened from the editor / operator panel shows this one, not always the default folder."""
    if graph is None:
        return None
    from ..core.graph import walk_nodes
    found = None
    for _key, _label, node in walk_nodes(graph):
        if node.type_name == "ArchivePart":
            if node.enabled:
                return _folder(node)
            found = found or _folder(node)
    return found


def _edge(node: Node, value) -> bool:
    """Rising edge of the trigger; an unconnected trigger (None) means every frame is a part."""
    if value is None:
        return True
    now = bool(value)
    fire = now and not node.state.get("prev", False)
    node.state["prev"] = now
    return fire


@register
class ArchivePart(Node):
    title = "Parçayı Arşivle"
    category = CAT
    description = ("Her muayene edilen parçayı izlenebilirlik arşivine yazar: zaman, akış, istasyon, seri no / barkod, "
                   "lot, sağlam/hatalı kararı, sebep, ölçülen değerler, reçete ve görüntüler (orijinal + işaretli, "
                   "JPEG, gün gün klasörlerde). Yazma arka planda yapılır, yavaş disk hattı yavaşlatmaz (kuyruk dolarsa "
                   "kayıt atlanır ve sayılır). Eski kayıtlar gün veya GB sınırına göre kendiliğinden silinir. "
                   "'trigger' bağlıysa yükseldiği karede, bağlı değilse her karede bir parça kaydedilir. Kayıtlar "
                   "Araçlar → İzlenebilirlik arşivi… penceresinde aranır, dışa aktarılır.")
    inputs = [("image", PortType.IMAGE), ("annotated", PortType.IMAGE), ("serial", PortType.TEXT),
              ("ok", PortType.BOOL), ("reason", PortType.TEXT), ("values", PortType.ANY), ("lot", PortType.TEXT),
              ("trigger", PortType.BOOL)]
    outputs = [("record_id", PortType.TEXT), ("saved", PortType.BOOL), ("dropped", PortType.NUMBER)]
    params = [Param("folder", "folder", "", "Arşiv klasörü (boş = varsayılan)"),
              Param("flow_name", "str", "", "Akış adı (boş = akış dosyasının adı)"),
              Param("station", "str", "İstasyon 1", "İstasyon"),
              Param("recipe", "str", "", "Reçete (boş = Reçete Seç bloğundan)"),
              Param("images", "choice", trace.IMAGE_POLICIES[0], "Görüntü saklama", choices=list(trace.IMAGE_POLICIES)),
              Param("quality", "int", 85, "JPEG kalitesi", 10, 100),
              Param("keep_days", "int", 90, "Kaç gün saklansın (0 = sınırsız)", 0, 36500),
              Param("max_gb", "float", 20.0, "En fazla GB (0 = sınırsız)", 0, 100000, 0.5),
              Param("queue", "int", 256, "Kuyruk boyu (kayıt)", 1, 100000)]
    setup_params = ("folder", "keep_days", "max_gb", "queue")

    def setup(self):
        self.state.update(prev=False, last="", writer=None)
        self.state["writer"] = trace.open_writer(
            _folder(self), queue_size=self.v("queue"), jpeg_quality=self.v("quality"),
            keep_days=self.v("keep_days"), max_gb=self.v("max_gb"), images=self.v("images"))
        self.state["dropped0"] = self.state["writer"].dropped

    def teardown(self):
        w = self.state.pop("writer", None)
        if w is not None:
            if not trace.release_writer(w, timeout=30.0):
                print(f"[{self.title}] UYARI: {w.pending} kayıt 30 saniyede diske yazılamadı")

    def _flow_name(self) -> str:
        name = str(self.v("flow_name") or "").strip()
        if name:
            return name
        flow_file = getattr(_node, "FLOW_FILE", None)
        return flow_file.stem if flow_file is not None else ""

    def _recipe(self) -> str:
        name = str(self.v("recipe") or "").strip()
        if name:
            return name
        ref = getattr(self, "_owner", None)
        g = ref() if ref is not None else None
        if g is None:
            return ""
        if g.root.active_recipe:          # switched while running (Reçete Seç, remote control API, Reçeteler)
            return g.root.active_recipe
        from ..core.graph import walk_nodes
        for _key, _label, node in walk_nodes(g.root):
            if node.type_name == "RecipeSelect" and node.state.get("name"):
                return str(node.state["name"])
        return ""

    def process(self, inputs):
        w = self.state.get("writer")
        if w is None:
            raise ValueError("Arşiv açılamadı")
        dropped = w.dropped - self.state.get("dropped0", 0)
        if not _edge(self, inputs.get("trigger")):
            return {"record_id": self.state["last"], "saved": False, "dropped": dropped}
        ok = inputs.get("ok")
        if isinstance(ok, np.generic):
            ok = ok.item()
        if ok is not None and not isinstance(ok, (bool, int, float)):
            raise ValueError(f"'ok' girişi doğru/yanlış olmalı, gelen: {type(ok).__name__}")
        for port in ("image", "annotated"):
            v = inputs.get(port)
            if v is not None and not isinstance(v, np.ndarray):
                raise ValueError(f"'{port}' girişine görüntü bağlayın (gelen: {type(v).__name__})")
        rid = w.submit(serial=inputs.get("serial"), ok=ok, reason=inputs.get("reason"),
                       values=trace.values_dict(inputs.get("values")), image=inputs.get("image"),
                       annotated=inputs.get("annotated"), flow=self._flow_name(), station=self.v("station"),
                       lot=inputs.get("lot"), recipe=self._recipe(), images=self.v("images"),
                       jpeg_quality=self.v("quality"))
        if rid:
            self.state["last"] = rid
        return {"record_id": rid, "saved": bool(rid), "dropped": w.dropped - self.state.get("dropped0", 0)}


@register
class ArchiveLookup(Node):
    title = "Arşivde Ara"
    category = CAT
    description = ("Tekrar kontrol istasyonları için: seri numarasının izlenebilirlik arşivindeki son kaydını bulur; "
                   "kararı (sağlam/hatalı), sebebi, ölçülen değerleri ve görüntüsünü verir. 'found' kayıt yoksa "
                   "yanlıştır. Seri no değişince (veya 'trigger' yükselince) yeniden aranır.")
    inputs = [("serial", PortType.TEXT), ("trigger", PortType.BOOL)]
    outputs = [("found", PortType.BOOL), ("ok", PortType.BOOL), ("reason", PortType.TEXT), ("values", PortType.ANY),
               ("image", PortType.IMAGE), ("record_id", PortType.TEXT), ("time", PortType.TEXT),
               ("count", PortType.NUMBER)]
    params = [Param("folder", "folder", "", "Arşiv klasörü (boş = varsayılan)"),
              Param("station", "str", "", "Yalnız bu istasyonun kayıtları (boş = hepsi)"),
              Param("which", "choice", "İşaretli (yoksa orijinal)", "Görüntü",
                    choices=["İşaretli (yoksa orijinal)", "Orijinal"])]

    def setup(self):
        self.state.update(prev=False, serial=None, out=None)

    def _empty(self):
        return {"found": False, "ok": False, "reason": "", "values": {}, "image": None, "record_id": "", "time": "",
                "count": 0}

    def process(self, inputs):
        serial = trace.clean_text(inputs.get("serial"))
        trig = inputs.get("trigger")
        again = trig is not None and _edge(self, trig)
        if not serial:
            self.state.update(serial=None, out=None)
            return self._empty()
        if not again and serial == self.state.get("serial") and self.state.get("out") is not None:
            return self.state["out"]
        folder = _folder(self)
        arch = trace.Archive(folder)
        if not arch.exists():
            raise ValueError(f"Arşiv bulunamadı: {folder}")
        writer = trace.pending_writer(folder)
        if writer is not None:
            writer.flush(0.5)                  # a part archived a moment ago by this program is found too
        rec, n = arch.last_by_serial(serial, str(self.v("station") or "").strip())
        if rec is None:
            out = self._empty()
        else:
            path = rec["img_path"] if self.v("which") == "Orijinal" else (rec["ann_path"] or rec["img_path"])
            img = trace.read_image(path) if path else None
            out = {"found": True, "ok": bool(rec["ok"]), "reason": rec["reason"], "values": rec["values"],
                   "image": img, "record_id": rec["rid"], "time": f"{rec['time']:%Y-%m-%d %H:%M:%S}", "count": n}
        self.state.update(serial=serial, out=out)
        return out
