"""Read incoming mail (IMAP) and answer it: save as draft, save for approval, or send."""
from __future__ import annotations

import datetime as dt
import html
import imaplib
import pathlib
import re
import smtplib
import time
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import make_msgid, parseaddr

from .config import AIConfig

REPLY_MODES = ["taslak (e-posta programında onaylarsınız)", "onay klasörüne kaydet", "hemen gönder"]
IMAP_TIMEOUT = 30          # seconds: a dropped Wi-Fi must not hang the mail check forever
IMAP_CLASS = imaplib.IMAP4_SSL      # tests swap in a fake server
SMTP_CLASS = smtplib.SMTP


class MailError(Exception):
    pass


@dataclass
class Mail:
    uid: str
    sender: str
    sender_addr: str
    subject: str
    body: str
    message_id: str
    date: str


def _plain(msg) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    text = part.get_content()
    if part.get_content_type() == "text/html":
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse(uid: str, raw: bytes) -> Mail:
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    name, addr = parseaddr(str(msg.get("from", "")))
    return Mail(uid=uid, sender=str(msg.get("from", "")), sender_addr=addr, subject=str(msg.get("subject", "")),
                body=_plain(msg)[:20000], message_id=str(msg.get("message-id", "")), date=str(msg.get("date", "")))


def _account(cfg: AIConfig, user, password, host):
    user, password, host = user or cfg.email_user, password or cfg.email_password, host or cfg.imap_host
    if not user or not password:
        raise MailError("E-posta hesabı girilmemiş (AI Merkezi → Ayarlar veya bloğun ayarları)")
    return user, password, host


def starttls_or_refuse(s, host: str) -> None:
    """Encrypt the SMTP connection before the password is sent. A server (or someone in between) that does not
    offer encryption is refused, except a relay on this computer."""
    try:
        s.starttls()
    except smtplib.SMTPNotSupportedError:
        if host.strip().lower() not in ("localhost", "127.0.0.1", "::1"):
            raise MailError(f"{host} şifreli bağlantı (STARTTLS) sunmuyor; şifre açık gönderilmez. Port 465 "
                            "(SSL) veya 587 deneyin.") from None


def mark_seen(uids, user="", password="", host="", folder="INBOX", cfg: AIConfig | None = None) -> None:
    """Mark mails read on the server (done when a mail is actually handed to the flow, not when downloaded)."""
    user, password, host = _account(cfg or AIConfig.load(), user, password, host)
    im = IMAP_CLASS(host, timeout=IMAP_TIMEOUT)
    try:
        im.login(user, password)
        im.select(folder)
        for uid in uids:
            im.uid("store", str(uid).encode() if isinstance(uid, str) else uid, "+FLAGS", "(\\Seen)")
    finally:
        try:
            im.logout()
        except Exception:  # noqa: BLE001
            pass


def fetch_unseen(user="", password="", host="", folder="INBOX", mark_seen=False, limit=20,
                 cfg: AIConfig | None = None, skip_uids=()) -> list[Mail]:
    """Unread mails, oldest first, at most ``limit`` per call. UIDs in ``skip_uids`` (already handled) are not
    downloaded and do not count against the limit, so older unread mail is never starved by newer."""
    user, password, host = _account(cfg or AIConfig.load(), user, password, host)
    try:
        im = IMAP_CLASS(host, timeout=IMAP_TIMEOUT)
        im.login(user, password)
    except (imaplib.IMAP4.error, OSError) as exc:
        raise MailError(f"Giriş yapılamadı ({host}): {exc}. Gmail için 'uygulama şifresi' gerekir.") from None
    try:
        im.select(folder)
        typ, data = im.uid("search", None, "UNSEEN")
        skip = {str(u) for u in skip_uids}
        uids = [u for u in (data[0] or b"").split() if u.decode() not in skip][:limit]
        out = []
        for uid in uids:
            typ, parts = im.uid("fetch", uid, "(BODY.PEEK[])")
            raw = next((p[1] for p in parts if isinstance(p, tuple)), None)
            if raw:
                out.append(parse(uid.decode(), raw))
            if mark_seen:
                im.uid("store", uid, "+FLAGS", "(\\Seen)")
        return out
    finally:
        try:
            im.logout()
        except Exception:  # noqa: BLE001
            pass


def build_reply(to: str, subject: str, body: str, sender: str, in_reply_to: str = "") -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject if subject.lower().startswith(("re:", "ynt:")) else f"Re: {subject}"
    msg["Date"] = dt.datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M:%S %z")
    msg["Message-ID"] = make_msgid()
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    msg.set_content(body)
    return msg


def _drafts_folder(im) -> str:
    """Find the drafts folder by its special-use flag (Gmail: '[Gmail]/Taslaklar' in Turkish)."""
    typ, rows = im.list()
    for row in rows or []:
        line = row.decode(errors="replace") if isinstance(row, bytes) else str(row)
        if "\\Drafts" in line:
            m = re.search(r'"([^"]+)"\s*$', line) or re.search(r"(\S+)\s*$", line)
            if m:
                return m.group(1)
    return "Drafts"


def deliver(msg: EmailMessage, mode: str, user="", password="", imap_host="", smtp_host="", smtp_port=0,
            folder: str = "onay_bekleyen", cfg: AIConfig | None = None) -> str:
    """Do what ``mode`` says with the reply. Returns a short description of where it went."""
    cfg = cfg or AIConfig.load()
    if mode.startswith("onay"):
        p = pathlib.Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        base = f"{dt.datetime.now():%Y%m%d_%H%M%S}_{re.sub(r'[^\w]+', '_', parseaddr(msg['To'])[1])[:40]}"
        for i in range(1000):                 # two replies to one sender in the same second get their own files
            f = p / (f"{base}.eml" if i == 0 else f"{base}_{i}.eml")
            try:
                with open(f, "xb") as fh:
                    fh.write(bytes(msg))
                break
            except FileExistsError:
                continue
        return f"onay için kaydedildi: {f}"
    user, password, imap_host = _account(cfg, user, password, imap_host)
    if mode.startswith("taslak"):
        im = IMAP_CLASS(imap_host, timeout=IMAP_TIMEOUT)
        try:
            im.login(user, password)
            box = _drafts_folder(im)
            typ, _ = im.append(f'"{box}"', "(\\Draft)", imaplib.Time2Internaldate(time.time()), bytes(msg))
            if typ != "OK":
                raise MailError(f"Taslak kaydedilemedi ({box})")
            return f"taslaklara kaydedildi ({box})"
        finally:
            try:
                im.logout()
            except Exception:  # noqa: BLE001
                pass
    host, port = smtp_host or cfg.smtp_host, smtp_port or cfg.smtp_port
    cls = smtplib.SMTP_SSL if port == 465 else SMTP_CLASS
    with cls(host, port, timeout=30) as s:
        if port != 465:
            starttls_or_refuse(s, host)
        s.login(user, password)
        s.send_message(msg)
    return f"gönderildi: {msg['To']}"
