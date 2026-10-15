"""Flow checker: finds the usual mistakes before a flow is run ("Akışı Denetle", ``headless --check``).

It only reads the flow and the AI settings; it never opens cameras, files or network connections.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import Graph

OPTIONAL_INPUTS = {"trigger", "next", "context", "chat_id"}
# blocks whose inputs all have a setting to fall back on, so they work with nothing connected
STANDALONE = {"WebPage", "ApiGet", "TelegramSend", "ChatWebhook", "Webhook", "EmailSend", "Beep", "WebResearch"}
CLAUDE_BLOCKS = {"AskAI", "AIClassify", "WebResearch", "ExtractInfo"}
JOB_BLOCKS = CLAUDE_BLOCKS | {"EmailReply", "TextToSpeech", "MakeVideo", "YouTubeUpload"}
FEEDS = {"EmailRead", "TelegramReceive", "FolderWatch", "WebhookReceive", "RssFeed"}


@dataclass
class Issue:
    level: str          # "hata" stops the flow from doing its job, "uyarı" is worth a look, "bilgi" is a note
    node_id: str | None
    text: str

    def __str__(self):
        return f"[{self.level.upper()}] {self.text}"


def _downstream(g: Graph, start: str) -> set[str]:
    seen, todo = set(), [start]
    while todo:
        n = todo.pop()
        for c in g.connections:
            if c.src == n and c.dst not in seen and not g.is_feedback(c):
                seen.add(c.dst)
                todo.append(c.dst)
    return seen


def _all_nodes(g: Graph, depth: int = 0):
    """Every block of the flow, those inside sub-flows included."""
    for n in g.nodes.values():
        yield n
        if n.type_name == "SubFlow" and depth < 12:
            try:
                yield from _all_nodes(n.inner_graph(), depth + 1)
            except Exception:  # noqa: BLE001
                pass


def check(g: Graph, cfg=None, _inside: bool = False, _depth: int = 0, _written: set | None = None) -> list[Issue]:
    issues: list[Issue] = []
    connected_in = {(c.dst, c.dst_port) for c in g.connections}
    used_out = {c.src for c in g.connections}
    live = {nid: n for nid, n in g.nodes.items() if n.inputs or n.outputs}
    name = lambda n: f"'{n.title}'"  # noqa: E731

    if not live:
        return [Issue("uyarı", None, "Akış boş: soldan blok sürükleyin.")]
    if not any(n.is_source for n in live.values()):
        issues.append(Issue("bilgi", None, "Kaynak blok (kamera, video, zamanlayıcı, e-posta…) yok: akış bir kez "
                                           "hesaplanır, ayar değişince yeniden hesaplanır."))
    for w in g.warnings:
        issues.append(Issue("hata", None, w))
    if not _inside:
        everything = list(_all_nodes(g))
        if any(n.type_name in ("SubflowInput", "SubflowOutput") for n in live.values()):
            issues.append(Issue("uyarı", None, "'Alt Akış Girişi / Çıkışı' blokları sadece bir alt akışın içinde işe "
                                               "yarar."))
    else:
        everything = list(g.nodes.values())

    for nid, n in live.items():                # what is inside a sub-flow is checked too, under its name
        if n.type_name == "SubFlow" and _depth < 12:
            try:
                inner = n.inner_graph()
            except Exception as exc:  # noqa: BLE001
                issues.append(Issue("hata", nid, f"Alt akış {name(n)} açılamadı: {exc}"))
                continue
            if cfg is None:
                cfg = _cfg()
            written = {m.values.get("name", "").strip() for m in everything if m.type_name == "SetVariable"}
            for i in check(inner, cfg, True, _depth + 1, _written if _written is not None else written):
                if i.node_id is None and i.level == "bilgi":
                    continue
                issues.append(Issue(i.level, nid, f"[{n.title}] {i.text}"))

    for nid, n in live.items():
        if not n.enabled:
            issues.append(Issue("uyarı", nid, f"{name(n)} devre dışı (Ctrl+E ile açılır)."))
        needed = [p for p, _ in n.inputs if p not in OPTIONAL_INPUTS]
        if needed and n.type_name not in STANDALONE and not any((nid, p) in connected_in for p, _ in n.inputs):
            issues.append(Issue("hata", nid, f"{name(n)} bloğunun hiçbir girişi bağlı değil."))
        elif n.outputs and not n.inputs and nid not in used_out and not n.is_source:
            issues.append(Issue("uyarı", nid, f"{name(n)} bloğunun çıkışı hiçbir yere bağlı değil."))

    for nid, n in live.items():
        if n.type_name in FEEDS and (nid, "next") not in connected_in:
            jobs = [g.nodes[d] for d in _downstream(g, nid) if g.nodes[d].type_name in JOB_BLOCKS]
            if jobs:
                issues.append(Issue("uyarı", nid, f"{name(n)} → {name(jobs[0])}: iş bitmeden sıradaki öğe gelebilir. "
                                                  f"Son adımın 'done' çıkışını {name(n)} bloğunun 'next' girişine "
                                                  f"bağlayın."))

    written = _written if _written is not None else \
        {n.values.get("name", "").strip() for n in everything if n.type_name == "SetVariable"}
    for nid, n in live.items():
        if n.type_name == "GetVariable" and n.values.get("name", "").strip() not in written:
            issues.append(Issue("uyarı", nid, f"{name(n)}: '{n.values.get('name')}' adlı değişkeni yazan bir "
                                              f"'Değişken Yaz' bloğu yok (ad yanlış yazılmış olabilir)."))

    types = {n.type_name for n in everything if n.enabled}
    if cfg is None:
        cfg = _cfg()
    if types & CLAUDE_BLOCKS and not cfg.key() and not _inside:
        issues.append(Issue("hata", None, "Claude API anahtarı girilmemiş: Araçlar → AI Merkezi → Ayarlar."))
    for nid, n in live.items():
        if n.enabled and n.type_name in ("EmailRead", "EmailReply") and not (
                (n.values.get("user") or cfg.email_user) and (n.values.get("password") or cfg.email_password)):
            issues.append(Issue("hata", nid, f"{name(n)}: e-posta hesabı girilmemiş (blokta veya Araçlar → "
                                             f"AI Merkezi → Ayarlar)."))
    for nid, n in live.items():
        if not n.enabled:
            continue
        if n.type_name in ("TelegramReceive", "TelegramSend") and not (n.values.get("token") or cfg.telegram_token):
            issues.append(Issue("hata", nid, f"{name(n)}: Telegram bot token'ı yok (blokta veya AI Merkezi'nde)."))
        if n.type_name == "TelegramReceive" and not (n.values.get("allowed") or cfg.telegram_allowed):
            issues.append(Issue("uyarı", nid, f"{name(n)}: izinli chat ID yok, hiçbir mesaj işlenmez. İlk mesajı "
                                              f"gönderen kişinin ID'si kayıtta görünür."))
        if n.type_name == "YouTubeUpload" and not cfg.youtube_client_secret:
            issues.append(Issue("hata", nid, f"{name(n)}: YouTube client_secret.json seçilmemiş."))
        if n.type_name == "TelegramSend" and not n.values.get("chat_id") and (nid, "chat_id") not in connected_in:
            issues.append(Issue("hata", nid, f"{name(n)}: chat ID yok (ayara yazın veya girişe bağlayın)."))
        if n.type_name in ("ChatWebhook", "Webhook") and not n.values.get("url"):
            issues.append(Issue("hata", nid, f"{name(n)}: webhook adresi boş."))
    return issues


def _cfg():
    from .ai.config import AIConfig
    return AIConfig.load()
