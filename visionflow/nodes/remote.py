"""'API Tetik': another program (MES, ERP, a script) fires the flow over the remote control REST API."""
from __future__ import annotations

import json
import re

from ..core import NoNewFrame, Node, Param, PortType, register

WAIT, FALSE = "bekle (sonraki bloklar çalışmaz)", "yanlış ver (akış devam eder)"


def clean_trigger_name(text) -> str:
    name = str(text or "").strip()
    if not name:
        raise ValueError("Tetik adı boş olamaz (API adresi: /api/v1/tetik/<ad>)")
    if not re.fullmatch(r"[\w.-]{1,64}", name):
        raise ValueError(f"Tetik adı sadece harf, rakam, '_', '-' ve '.' içerebilir (en fazla 64): {name!r}")
    return name


@register
class ApiTrigger(Node):
    is_source = True
    setup_params = ("name",)
    title = "API Tetik"
    category = "1 · Giriş"
    description = ("Uzaktan kontrol API'si (Araçlar → Uzaktan kontrol) üzerinden başka bir program akışı tetikler: "
                   "POST /api/v1/tetik/<ad> isteği gelince 'fired' doğru olur, istekle gönderilen JSON 'data' "
                   "çıkışına gelir (ör. seri no, ürün kodu). Her istek bir kez verilir; arka arkaya gelenler sıraya "
                   "girer. İstek yokken ya bekler (bu bloğa bağlı bloklar çalışmaz) ya da 'fired' yanlış verir.")
    outputs = [("fired", PortType.BOOL), ("data", PortType.ANY), ("text", PortType.TEXT),
               ("waiting", PortType.NUMBER), ("count", PortType.NUMBER)]
    params = [Param("name", "str", "tetik", "Tetik adı (adres: /api/v1/tetik/<ad>)"),
              Param("idle", "choice", WAIT, "İstek yokken", choices=[WAIT, FALSE])]

    def setup(self):
        from ..api_server import TRIGGERS
        name = clean_trigger_name(self.v("name"))
        TRIGGERS.register(name)
        self.state.update(name=name, count=0)

    def process(self, inputs):
        from ..api_server import TRIGGERS
        name = self.state["name"]
        fired, payload = TRIGGERS.pop(name)
        if not fired:
            if self.v("idle") == WAIT:
                raise NoNewFrame("API isteği bekleniyor")
            return {"fired": False, "data": None, "text": "", "waiting": 0, "count": self.state["count"]}
        self.state["count"] += 1
        if payload is None:
            text = ""
        elif isinstance(payload, str):
            text = payload
        elif isinstance(payload, dict) and any(k in payload for k in ("text", "mesaj", "metin")):
            text = str(next(payload[k] for k in ("text", "mesaj", "metin") if k in payload))
        else:
            text = json.dumps(payload, ensure_ascii=False)
        return {"fired": True, "data": payload, "text": text, "waiting": TRIGGERS.waiting(name),
                "count": self.state["count"]}

    def teardown(self):
        from ..api_server import TRIGGERS
        name = self.state.pop("name", None)
        if name:
            TRIGGERS.unregister(name)
