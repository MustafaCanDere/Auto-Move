"""Rapor Oluştur: writes the HTML (and PDF) production report from the counts the operator panel collected."""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import sqlite3

from ..core import Node, Param, PortType, register
from ..core import node as _node

CAT = "5 · Çıkış ve Otomasyon"


@register
class ProductionReport(Node):
    title = "Rapor Oluştur"
    category = CAT
    description = ("Operatör panelinin dakikada bir kaydettiği sağlam/hatalı sayılarından üretim raporu yazar: "
                   "toplamlar, verim, saatlik tablo, grafik ve son alarmlar (tek dosya HTML, isterseniz PDF). "
                   "'trigger' yanlış→doğru olunca son N saatin raporu yazılır (bağlı değilse akış başında bir kez). "
                   "Veri dosyası boşsa akışın klasöründeki uretim_verisi.sqlite kullanılır. Zamanlayıcı veya "
                   "Vardiya Raporu bloğunun 'report' çıkışıyla vardiya sonunda otomatik rapor alınır.")
    inputs = [("trigger", PortType.BOOL)]
    outputs = [("path", PortType.TEXT), ("done", PortType.BOOL)]
    params = [Param("hours", "int", 8, "Son kaç saat", 1, 24 * 31),
              Param("folder", "folder", "raporlar", "Rapor klasörü"),
              Param("db", "str", "", "Veri dosyası (boş = akış klasörü)"),
              Param("flow", "str", "", "Akış adı (boş = tümü)"),
              Param("report_title", "str", "Üretim raporu", "Rapor başlığı"),
              Param("pdf", "bool", False, "PDF de oluştur")]

    def setup(self):
        self.state.update(prev=False, fired_once=False, path="", note="")

    def _db(self) -> str:
        if self.v("db"):
            return _node.resolve_path(self.v("db"))
        from ..production import db_path
        if _node.FLOW_DIR is not None:
            return str(pathlib.Path(_node.FLOW_DIR) / "uretim_verisi.sqlite")
        return str(db_path(None))

    def process(self, inputs):
        st = self.state
        trig = inputs["trigger"]
        if trig is None:
            fire = not st["fired_once"]
        else:
            fire = bool(trig) and not st["prev"]
            st["prev"] = bool(trig)
        if not fire:
            return {"path": st["path"], "done": False}
        st["fired_once"] = True
        from ..production import make_report
        end = dt.datetime.now()
        start = end - dt.timedelta(hours=self.v("hours"))
        folder = self.v("folder") or "."
        out = pathlib.Path(folder) / f"uretim_raporu_{end:%Y%m%d_%H%M%S}.html"
        try:
            html_path, pdf_path, msg = make_report(self._db(), start, end, out, self.v("flow") or None,
                                                   self.v("report_title") or "Üretim raporu", self.v("pdf"))
        except (OSError, sqlite3.Error) as exc:
            raise ValueError(f"Rapor yazılamadı ({folder}): {exc}") from exc
        if msg:
            print(f"[{self.title}] {msg}")
        st["path"] = os.path.abspath(html_path)
        print(f"[{self.title}] Rapor yazıldı: {st['path']}" + (" (+ PDF)" if pdf_path else ""))
        return {"path": st["path"], "done": True}
