"""Thin layer over the Anthropic SDK used by the AI blocks, the AI centre window and the flow builder."""
from __future__ import annotations

import base64
import datetime as dt
import json
import sys
import threading
from types import SimpleNamespace

import cv2

from .config import AIConfig, home

MAX_CONTINUATIONS = 5
_test_client = None     # tests put a fake client here


class AIError(Exception):
    pass


def client_for(cfg: AIConfig | None = None, api_key: str | None = None):
    if _test_client is not None:
        return _test_client
    try:
        import anthropic
    except ImportError:
        raise AIError("Yapay zeka blokları için 'anthropic' paketi gerekli: pip install anthropic") from None
    key = api_key or (cfg or AIConfig.load()).key()
    if not key:
        raise AIError("Claude API anahtarı yok. Araçlar → AI Merkezi → Ayarlar'dan girin.")
    return anthropic.Anthropic(api_key=key)


def image_block(img) -> dict:
    h, w = img.shape[:2]
    if max(h, w) > 1568:                      # larger images are downscaled by the API anyway
        s = 1568 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    data = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


# $ per million tokens (input, output); cache reads are cheaper but counted as input here: an upper estimate
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
_usage_lock = threading.Lock()


def _usage_file():
    return home() / "kullanim.json"


def _write_atomic(path, text: str) -> None:
    """Write via a temporary file and rename, so a reader (or a crash) never sees a half-written file."""
    import os
    import time
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:             # Windows: another process has the file open for a moment
            time.sleep(0.05)
    try:                                    # still locked (antivirus, backup): write in place, never drop it
        path.write_text(text, encoding="utf-8")
    finally:
        tmp.unlink(missing_ok=True)


class _FileLock:
    """Lock shared by every VisionFlow process on this computer (editor + headless service + a second window),
    so two of them updating today's spending at the same moment cannot lose one of the updates."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        import time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+b")
        end = time.monotonic() + 10
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self.f.seek(0)
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() > end:      # a stuck process must not stop Claude calls for good
                    return self
                time.sleep(0.02)

    def __exit__(self, *exc):
        try:
            if sys.platform == "win32":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.f.close()


def usage_today() -> dict:
    """{'requests', 'input', 'output', 'usd'} for today, as recorded on this computer."""
    with _usage_lock:
        try:
            data = json.loads(_usage_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    return data.get(dt.date.today().isoformat(), {"requests": 0, "input": 0, "output": 0, "usd": 0.0})


def _record_usage(model: str, resp) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    inp = (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
        + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    out = getattr(u, "output_tokens", 0) or 0
    pi, po = PRICES.get(model, PRICES["claude-opus-5-5"])
    with _usage_lock, _FileLock(_usage_file().with_suffix(".lock")):
        try:
            data = json.loads(_usage_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        day = data.setdefault(dt.date.today().isoformat(), {"requests": 0, "input": 0, "output": 0, "usd": 0.0})
        day["requests"] += 1
        day["input"] += int(inp)
        day["output"] += int(out)
        day["usd"] = round(day["usd"] + inp * pi / 1e6 + out * po / 1e6, 4)
        data = dict(sorted(data.items())[-60:])          # keep two months
        _usage_file().parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(_usage_file(), json.dumps(data, indent=1))


def _check_budget(cfg: AIConfig | None):
    limit = (cfg or AIConfig.load()).daily_limit_usd
    if limit and usage_today()["usd"] >= limit:
        raise AIError(f"Günlük harcama sınırına ulaşıldı (~${limit:g}). AI Merkezi → Ayarlar'dan artırabilirsiniz; "
                      "yarın kendiliğinden açılır.")


def _create(client, cfg: AIConfig | None = None, **kw):
    _check_budget(cfg)
    resp = _create_raw(client, **kw)
    _record_usage(kw.get("model", ""), resp)
    return resp


def _create_raw(client, **kw):
    try:
        import anthropic
        errors = (anthropic.AuthenticationError, anthropic.RateLimitError, anthropic.APIStatusError,
                  anthropic.APIConnectionError)
    except ImportError:          # fake client in tests
        anthropic, errors = None, ()
    try:
        return client.beta.messages.create(
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
    except errors as exc:
        if isinstance(exc, anthropic.AuthenticationError):
            raise AIError("Claude API anahtarı geçersiz") from None
        if isinstance(exc, anthropic.RateLimitError):
            raise AIError("Çok fazla istek; biraz sonra tekrar deneyin") from None
        if isinstance(exc, anthropic.APIStatusError):
            raise AIError(f"Claude hatası ({exc.status_code}): {exc.message}") from None
        raise AIError("Claude'a bağlanılamadı (internet bağlantısını kontrol edin)") from None


def _check(resp):
    if resp.stop_reason == "refusal":
        raise AIError("Claude bu isteği yanıtlamadı (güvenlik). Farklı ifade etmeyi deneyin.")


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def ask(prompt: str, system: str | None = None, image=None, history: list[dict] | None = None,
        cfg: AIConfig | None = None, model: str | None = None, max_tokens: int = 4000,
        effort: str = "medium") -> str:
    """One question to Claude. ``history`` is earlier [{'role','content'}] turns (conversation memory)."""
    cfg = cfg or AIConfig.load()
    content = [image_block(image), {"type": "text", "text": prompt}] if image is not None else prompt
    messages = list(history or []) + [{"role": "user", "content": content}]
    resp = _create(client_for(cfg), cfg, model=model or cfg.model, max_tokens=max_tokens,
                   system=system if system is not None else cfg.persona, messages=messages,
                   output_config={"effort": effort})
    _check(resp)
    if resp.stop_reason == "max_tokens":
        return _text(resp) + " […]"
    return _text(resp)


def research(question: str, max_searches: int = 5, detailed: bool = False, cfg: AIConfig | None = None,
             model: str | None = None) -> tuple[str, list[dict]]:
    """Search the web with Claude and write a Turkish report. Returns (report, [{'title','url'}])."""
    cfg = cfg or AIConfig.load()
    client = client_for(cfg)
    style = ("Ayrıntılı, başlıklı bir rapor yaz; önemli rakamları ve tarihleri ver."
             if detailed else "Kısa ve net bir özet yaz (en fazla 8 madde).")
    system = (cfg.persona + "\n\nİnternette araştırma yapıp güncel ve doğrulanmış bilgi verirsin. Türkçe yaz. "
              + style + " Kaynakları metnin içinde değil, sonda ayrıca vereceğim; sen sadece raporu yaz.")
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}]
    messages = [{"role": "user", "content": question}]
    resp, content = None, []
    for _ in range(MAX_CONTINUATIONS):
        resp = _create(client, cfg, model=model or cfg.model, max_tokens=16000, system=system, messages=messages,
                       tools=tools, output_config={"effort": "high" if detailed else "medium"})
        _check(resp)
        content += list(resp.content)          # a paused turn continues: keep what came before
        if resp.stop_reason != "pause_turn":
            break
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": list(content)}]
    resp = SimpleNamespace(content=content, stop_reason=resp.stop_reason)
    sources, seen = [], set()
    for block in resp.content:
        if getattr(block, "type", "") == "web_search_tool_result" and isinstance(block.content, list):
            for r in block.content:
                if getattr(r, "type", "") == "web_search_result" and r.url not in seen:
                    seen.add(r.url)
                    sources.append({"title": r.title, "url": r.url})
    return _text(resp), sources


def extract(text: str, fields: list[str], hint: str = "", cfg: AIConfig | None = None,
            model: str | None = None) -> dict[str, str]:
    """Pull named fields out of free text (invoice, e-mail, form). Missing fields come back as ''."""
    cfg = cfg or AIConfig.load()
    keys = [f.strip() for f in fields if f.strip()]
    if not keys:
        raise AIError("Çıkarılacak alan listesi boş")
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in keys},
              "required": keys, "additionalProperties": False}
    system = ("Verilen metinden istenen alanları çıkar. Metinde olmayan alan için boş metin yaz, uydurma. "
              "Sayıları metindeki gibi yaz." + (f" Not: {hint}" if hint else ""))
    resp = _create(client_for(cfg), cfg, model=model or cfg.model, max_tokens=2000, system=system,
                   messages=[{"role": "user", "content": text[:50000]}],
                   output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
    _check(resp)
    try:
        data = json.loads(_text(resp))
    except ValueError:
        raise AIError("Claude geçerli JSON döndürmedi") from None
    return {k: str(data.get(k, "") or "") for k in keys}

