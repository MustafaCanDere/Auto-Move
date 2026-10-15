"""Communication blocks: WhatsApp integration (Listen & Send) and Email (Send & Listen)."""
from __future__ import annotations

import collections
import datetime as dt
import email
from email import policy
from email.message import EmailMessage
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.parser import BytesParser
from email.utils import make_msgid, parseaddr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import imaplib
import json
import os
import pathlib
import re
import smtplib
import socket
import tempfile
import threading
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ..core import node as _node
from .processing import _bgr


def _rising(node: Node, trigger, key: str = "prev_trigger") -> bool:
    if trigger is None:
        once_key = f"{key}_once"
        if not node.state.get(once_key, False):
            node.state[once_key] = True
            return True
        return False
    now = bool(trigger)
    fired = now and not node.state.get(key, False)
    node.state[key] = now
    return fired


CAT_INPUT = "1 · Giriş"
CAT_OUT = "5 · Çıkış ve Otomasyon"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


def _save_temp_image(frame, prefix: str = "vf_media_") -> str | None:
    """Save an OpenCV numpy frame to a temporary image file and return its path."""
    if frame is None:
        return None
    try:
        bgr = _bgr(frame)
        tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".png", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, bgr)
        return tmp.name
    except Exception as exc:
        print(f"[Communication] Görsel kaydedilemedi: {exc}")
        return None


def _parse_whatsapp_payload(data: dict) -> dict:
    """Intelligently parse different WhatsApp webhook JSON payloads (Generic, Green API, UltraMsg, Evolution, Twilio)."""
    text = ""
    sender = ""
    msg_id = ""
    media_url = ""

    if not isinstance(data, dict):
        return {"text": str(data), "sender": "", "id": "", "media_url": ""}

    # 1. Green API format
    if "messageData" in data:
        md = data.get("messageData", {})
        sd = data.get("senderData", {})
        sender = sd.get("sender", "") or sd.get("chatId", "")
        msg_id = data.get("idMessage", "")
        tmd = md.get("textMessageData", {})
        if tmd and "textMessage" in tmd:
            text = tmd["textMessage"]
        elif md.get("typeMessage") == "extendedTextMessage":
            text = md.get("extendedTextMessageData", {}).get("text", "")
        elif "fileMessageData" in md:
            fmd = md["fileMessageData"]
            media_url = fmd.get("downloadUrl", "")
            text = fmd.get("caption", "") or fmd.get("fileName", "[Ek Dosya]")

    # 2. UltraMsg format
    elif "event_type" in data and "data" in data:
        sub = data.get("data", {})
        sender = sub.get("from", "")
        text = sub.get("body", "")
        msg_id = sub.get("id", "")
        media_url = sub.get("media", "")

    # 3. Evolution API format
    elif "data" in data and isinstance(data["data"], dict) and "message" in data["data"]:
        sub = data["data"]
        msg_id = sub.get("key", {}).get("id", "")
        sender = sub.get("key", {}).get("remoteJid", "")
        msg_obj = sub.get("message", {})
        text = msg_obj.get("conversation", "") or msg_obj.get("extendedTextMessage", {}).get("text", "")

    # 4. Standard / Generic formats
    else:
        for k in ("text", "message", "body", "content", "msg"):
            if k in data and isinstance(data[k], str):
                text = data[k]
                break
        for k in ("sender", "from", "phone", "chatId", "user", "number", "wa_id"):
            if k in data and isinstance(data[k], (str, int)):
                sender = str(data[k])
                break
        for k in ("id", "msg_id", "messageId", "message_id"):
            if k in data and isinstance(data[k], str):
                msg_id = data[k]
                break
        for k in ("media_url", "media", "attachment", "url", "file_url"):
            if k in data and isinstance(data[k], str):
                media_url = data[k]
                break

    # Clean up phone number
    sender = re.sub(r"@s\.whatsapp\.net|@c\.us", "", sender).strip()
    return {
        "text": text.strip(),
        "sender": sender,
        "id": msg_id or f"wa_{int(time.time()*1000)}",
        "media_url": media_url.strip(),
    }


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ==========================================================================================
# 1. WhatsApp Listen (WhatsApp Dinle)
# ==========================================================================================

@register
class WhatsAppListen(Node):
    setup_params = ("mode", "webhook_port", "webhook_path")
    title = "WhatsApp Dinle"
    category = CAT_INPUT
    description = (
        "WhatsApp'tan gelen mesajları canlı dinler (dahili Webhook sunucusu veya REST polling ile). "
        "Gelen mesaj metnini, gönderen telefon numarasını ve ek dosya/medya bilgisini akışa aktarır. "
        "Sunum ve rapor oluşturma bloklarına doğrudan bağlanabilir."
    )
    inputs = [
        ("next", BOOL),
        ("test_trigger", BOOL),
        ("test_message", TEXT),
        ("test_sender", TEXT),
    ]
    outputs = [
        ("new", BOOL),
        ("text", TEXT),
        ("sender", TEXT),
        ("message_id", TEXT),
        ("has_attachment", BOOL),
        ("media_url", TEXT),
        ("waiting", NUM),
        ("raw_json", TEXT),
    ]
    params = [
        Param("mode", "choice", "Dahili Webhook Sunucusu", "Çalışma Modu",
              choices=["Dahili Webhook Sunucusu", "REST Polling", "Simülasyon / Test"]),
        Param("webhook_port", "int", 5055, "Webhook Portu", 1024, 65535),
        Param("webhook_path", "str", "/webhook/whatsapp", "Webhook Yolu"),
        Param("polling_url", "str", "", "Polling URL (REST Modu için)"),
        Param("polling_interval", "float", 5.0, "Polling Aralığı (sn)", 1.0, 3600.0),
        Param("auth_token", "str", "", "Güvenlik Belirteci (Token - İsteğe bağlı)"),
        Param("filter_sender", "str", "", "Sadece Bu Numara (Boş = Tümü)"),
    ]

    def setup(self):
        # Shut down previous server instance if re-starting
        prev_server = self.state.get("server")
        if prev_server:
            try:
                prev_server.shutdown()
                prev_server.server_close()
            except Exception:
                pass

        queue: collections.deque = collections.deque(maxlen=200)
        lock = threading.Lock()
        server = None
        mode = self.v("mode")
        expected_token = self.v("auth_token").strip()
        expected_path = self.v("webhook_path").strip() or "/webhook/whatsapp"
        if not expected_path.startswith("/"):
            expected_path = "/" + expected_path

        if mode == "Dahili Webhook Sunucusu":
            port = int(self.v("webhook_port"))

            class WebhookHandler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass  # keep console clean

                def _send(self, code: int, body_dict: dict):
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(body_dict, ensure_ascii=False).encode("utf-8"))

                def do_GET(self):
                    # Health check endpoint
                    self._send(200, {
                        "status": "online",
                        "service": "VisionFlow WhatsApp Webhook",
                        "time": dt.datetime.now().isoformat(),
                        "path": expected_path,
                    })

                def do_POST(self):
                    req_path = self.path.split("?")[0]
                    if req_path != expected_path:
                        self._send(404, {"error": "Path not found"})
                        return

                    if expected_token:
                        auth_header = self.headers.get("Authorization", "")
                        token_header = self.headers.get("X-Webhook-Token", "")
                        if expected_token not in auth_header and expected_token != token_header:
                            self._send(401, {"error": "Unauthorized token"})
                            return

                    try:
                        content_len = int(self.headers.get("Content-Length", 0))
                        raw_body = self.rfile.read(content_len).decode("utf-8", errors="replace")
                        if self.headers.get("Content-Type", "").startswith("application/x-www-form-urlencoded"):
                            parsed_qs = urllib.parse.parse_qs(raw_body)
                            data = {k: v[0] if len(v) == 1 else v for k, v in parsed_qs.items()}
                        else:
                            data = json.loads(raw_body) if raw_body else {}

                        parsed = _parse_whatsapp_payload(data)
                        parsed["raw_json"] = raw_body

                        with lock:
                            queue.append(parsed)

                        self._send(200, {"success": True, "message_id": parsed.get("id")})
                    except Exception as exc:
                        self._send(400, {"error": f"Invalid payload: {exc}"})

            try:
                server = ReusableThreadingHTTPServer(("0.0.0.0", port), WebhookHandler)
                threading.Thread(target=server.serve_forever, daemon=True).start()
                print(f"[WhatsApp Dinle] Webhook sunucusu 0.0.0.0:{port}{expected_path} üzerinde dinliyor.")
            except Exception as exc:
                print(f"[WhatsApp Dinle] Sunucu başlatılamadı: {exc}")

        elif mode == "REST Polling":
            poll_url = self.v("polling_url").strip()
            interval = max(1.0, float(self.v("polling_interval")))

            def poll_worker():
                while not self.state.get("stopped", False):
                    if poll_url:
                        try:
                            req = urllib.request.Request(poll_url, headers={"User-Agent": "VisionFlow/WhatsApp"})
                            with urllib.request.urlopen(req, timeout=8) as resp:
                                body = resp.read().decode("utf-8", errors="replace")
                                data = json.loads(body)
                                items = data if isinstance(data, list) else [data]
                                for item in items:
                                    parsed = _parse_whatsapp_payload(item)
                                    parsed["raw_json"] = json.dumps(item)
                                    with lock:
                                        queue.append(parsed)
                        except Exception:
                            pass
                    time.sleep(interval)

            threading.Thread(target=poll_worker, daemon=True).start()

        self.state.update(
            queue=queue,
            lock=lock,
            server=server,
            stopped=False,
            last_text="",
            last_sender="",
            last_id="",
            last_media="",
            last_has_att=False,
            last_raw="",
            prev_next=False,
            prev_test=False,
        )

    def process(self, inputs):
        st = self.state
        lock: threading.Lock = st.get("lock", threading.Lock())
        queue: collections.deque = st.get("queue", collections.deque())

        # Check manual / test trigger
        test_trig = _rising(self, inputs.get("test_trigger"), "prev_test")
        if test_trig:
            t_msg = str(inputs.get("test_message") or "Test WhatsApp Mesajı: Hat 1 Verimi %98.4").strip()
            t_snd = str(inputs.get("test_sender") or "+905551234567").strip()
            item = {
                "text": t_msg,
                "sender": t_snd,
                "id": f"test_{int(time.time()*1000)}",
                "media_url": "",
                "raw_json": json.dumps({"text": t_msg, "sender": t_snd}),
            }
            with lock:
                queue.append(item)

        filter_sender = str(self.v("filter_sender")).strip().replace(" ", "").replace("-", "")

        is_new = False
        should_pop = False

        # If user explicitly fired 'next' port
        next_trig = _rising(self, inputs.get("next"), "prev_next")

        with lock:
            if queue and (next_trig or not st["last_text"]):
                # Pop the next matching message
                while queue:
                    cand = queue.popleft()
                    cand_sender = cand.get("sender", "").replace(" ", "").replace("-", "")
                    if not filter_sender or filter_sender in cand_sender:
                        st["last_text"] = cand.get("text", "")
                        st["last_sender"] = cand.get("sender", "")
                        st["last_id"] = cand.get("id", "")
                        st["last_media"] = cand.get("media_url", "")
                        st["last_has_att"] = bool(cand.get("media_url"))
                        st["last_raw"] = cand.get("raw_json", "")
                        is_new = True
                        break
            waiting_count = len(queue)

        return {
            "new": is_new,
            "text": st["last_text"],
            "sender": st["last_sender"],
            "message_id": st["last_id"],
            "has_attachment": st["last_has_att"],
            "media_url": st["last_media"],
            "waiting": waiting_count,
            "raw_json": st["last_raw"],
        }


# ==========================================================================================
# 2. WhatsApp Send (WhatsApp Gönder - Gelişmiş)
# ==========================================================================================

@register
class WhatsAppSend(Node):
    title = "WhatsApp Gönder"
    category = CAT_OUT
    description = (
        "Tetik gelince WhatsApp mesajı, görsel veya rapor/dosya gönderir. "
        "CallMeBot (ücretsiz), Green API, UltraMsg, Evolution API ve Özel Webhook ağ geçitlerini destekler."
    )
    inputs = [
        ("trigger", BOOL),
        ("text", ANY),
        ("value", ANY),
        ("media_path", TEXT),
        ("image", IMG),
    ]
    outputs = [
        ("sent", BOOL),
        ("count", NUM),
        ("status", TEXT),
    ]
    params = [
        Param("provider", "choice", "CallMeBot (Ücretsiz)", "Sağlayıcı",
              choices=["CallMeBot (Ücretsiz)", "Green API", "UltraMsg", "Evolution API / Webhook", "Özel HTTP POST"]),
        Param("phone", "str", "+905xxxxxxxxx", "Alıcı Numarası (+90...)"),
        Param("api_key_token", "str", "", "API Anahtarı / Token"),
        Param("instance_id", "str", "", "Instance ID (Green API / UltraMsg)"),
        Param("api_url", "str", "https://api.callmebot.com/whatsapp.php", "API URL"),
        Param("message_template", "code", "{v}", "Mesaj Şablonu ({v} yerine gelen değer geçer)"),
        Param("cooldown", "float", 5.0, "Bekleme Süresi (sn)", 0.0, 86400.0),
    ]

    def setup(self):
        self.state.update(last_sent=0.0, count=0, status="Hazır", prev_trig=False)

    def process(self, inputs):
        st = self.state
        st.setdefault("last_sent", 0.0)
        st.setdefault("count", 0)
        st.setdefault("status", "Hazır")

        if not _rising(self, inputs.get("trigger"), "prev_trig"):
            return {"sent": False, "count": st["count"], "status": st["status"]}

        now = time.monotonic()
        cooldown = float(self.v("cooldown"))
        if now - st["last_sent"] < cooldown:
            return {"sent": False, "count": st["count"], "status": f"Cooldown beklemesinde ({cooldown}s)"}

        phone = str(self.v("phone")).strip().replace(" ", "").replace("-", "")
        token = str(self.v("api_key_token")).strip()
        provider = self.v("provider")
        instance = str(self.v("instance_id")).strip()
        api_url = str(self.v("api_url")).strip()

        # Build message string
        in_val = inputs.get("text") if inputs.get("text") is not None else inputs.get("value")
        raw_msg = str(in_val) if in_val is not None else ""
        template = str(self.v("message_template"))
        final_text = template.replace("{v}", raw_msg) if "{v}" in template else (raw_msg or template)

        # Handle image frame or media file
        media_path = str(inputs.get("media_path") or "").strip()
        img_frame = inputs.get("image")
        if not media_path and img_frame is not None:
            media_path = _save_temp_image(img_frame) or ""

        st["last_sent"] = now

        def do_send():
            try:
                if provider == "CallMeBot (Ücretsiz)":
                    encoded_text = urllib.parse.quote(final_text)
                    url = f"{api_url}?phone={phone}&text={encoded_text}&apikey={token}"
                    req = urllib.request.Request(url, headers={"User-Agent": "VisionFlow/Studio"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        resp.read()

                elif provider == "Green API":
                    # Green API: https://api.green-api.com/waInstance{idInstance}/sendMessage/{apiTokenInstance}
                    base = api_url.rstrip("/") if api_url else "https://api.green-api.com"
                    chat_id = phone.replace("+", "") + "@c.us" if "@" not in phone else phone
                    url = f"{base}/waInstance{instance}/sendMessage/{token}"
                    payload = {"chatId": chat_id, "message": final_text}
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "User-Agent": "VisionFlow/Studio"},
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        resp.read()

                elif provider == "UltraMsg":
                    # UltraMsg: https://api.ultramsg.com/{instance}/messages/chat
                    base = api_url.rstrip("/") if api_url else f"https://api.ultramsg.com/{instance}/messages/chat"
                    payload = {"token": token, "to": phone, "body": final_text}
                    data_enc = urllib.parse.urlencode(payload).encode("utf-8")
                    req = urllib.request.Request(base, data=data_enc, headers={"User-Agent": "VisionFlow/Studio"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        resp.read()

                elif provider in ("Evolution API / Webhook", "Özel HTTP POST"):
                    payload = {
                        "number": phone,
                        "phone": phone,
                        "text": final_text,
                        "message": final_text,
                        "media_path": media_path,
                        "timestamp": dt.datetime.now().isoformat(),
                    }
                    headers = {"Content-Type": "application/json", "User-Agent": "VisionFlow/Studio"}
                    if token:
                        headers["apikey"] = token
                        headers["Authorization"] = f"Bearer {token}"
                    req = urllib.request.Request(api_url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        resp.read()

                st["count"] += 1
                st["status"] = f"Gönderildi ({provider}): {dt.datetime.now():%H:%M:%S}"
                print(f"[WhatsApp Gönder] {st['status']}")
            except Exception as exc:
                st["status"] = f"Hata ({provider}): {exc}"
                print(f"[WhatsApp Gönder] {st['status']}")

        threading.Thread(target=do_send, daemon=True).start()
        return {"sent": True, "count": st["count"], "status": "Gönderim başlatıldı"}


# ==========================================================================================
# 3. Email Send (E-posta Gönder)
# ==========================================================================================

@register
class EmailSend(Node):
    title = "E-posta Gönder"
    category = CAT_OUT
    description = (
        "Tetik gelince SMTP üzerinden zengin e-posta (HTML veya düz metin) ve dosya eki "
        "(PDF rapor, PowerPoint sunum .pptx, Word .docx, Excel .xlsx veya kamera görüntüsü) gönderir."
    )
    inputs = [
        ("trigger", BOOL),
        ("to", TEXT),
        ("subject", TEXT),
        ("body", ANY),
        ("attachment", TEXT),
        ("image", IMG),
        ("cc", TEXT),
    ]
    outputs = [
        ("sent", BOOL),
        ("count", NUM),
        ("status", TEXT),
    ]
    params = [
        Param("smtp_host", "str", "smtp.gmail.com", "SMTP Sunucu"),
        Param("smtp_port", "int", 587, "SMTP Portu", 1, 65535),
        Param("user", "str", "", "E-posta Hesabı (Kullanıcı Adı)"),
        Param("password", "str", "", "E-posta Şifresi / Uygulama Şifresi"),
        Param("security", "choice", "STARTTLS (587)", "Güvenlik Türü",
              choices=["STARTTLS (587)", "SSL/TLS (465)", "Şifresiz (25/Relay)"]),
        Param("sender_name", "str", "VisionFlow Raporlama", "Gönderen Adı"),
        Param("is_html", "bool", True, "HTML Formatı Kullan"),
        Param("signature", "code", "\n--\nVisionFlow Endüstriyel Yapay Zeka Sistemi", "İmza"),
        Param("cooldown", "float", 5.0, "Bekleme Süresi (sn)", 0.0, 86400.0),
    ]

    def setup(self):
        self.state.update(count=0, last_sent=0.0, status="Hazır", prev_trig=False)

    def process(self, inputs):
        st = self.state
        st.setdefault("count", 0)
        st.setdefault("last_sent", 0.0)
        st.setdefault("status", "Hazır")

        if not _rising(self, inputs.get("trigger"), "prev_trig"):
            return {"sent": False, "count": st["count"], "status": st["status"]}

        now = time.monotonic()
        cooldown = float(self.v("cooldown"))
        if now - st["last_sent"] < cooldown:
            return {"sent": False, "count": st["count"], "status": f"Cooldown beklemesinde ({cooldown}s)"}

        to_addr = str(inputs.get("to") or "").strip()
        if not to_addr:
            raise ValueError("Alıcı e-posta adresi ('to') boş olamaz")

        subject = str(inputs.get("subject") or "VisionFlow Rapor / Bildirim").strip()
        body_val = inputs.get("body")
        raw_body = str(body_val) if body_val is not None else ""
        sig = str(self.v("signature") or "")
        if sig:
            raw_body = raw_body + "\n\n" + sig

        user = str(self.v("user")).strip()
        passwd = str(self.v("password")).strip()
        host = str(self.v("smtp_host")).strip()
        port = int(self.v("smtp_port"))
        sec = self.v("security")
        sender_name = str(self.v("sender_name")).strip()
        from_header = f"{sender_name} <{user}>" if sender_name else user
        is_html = bool(self.v("is_html"))
        cc_addr = str(inputs.get("cc") or "").strip()

        att_path = str(inputs.get("attachment") or "").strip()
        img_frame = inputs.get("image")

        st["last_sent"] = now

        def do_email():
            try:
                msg = MIMEMultipart("mixed")
                msg["From"] = from_header
                msg["To"] = to_addr
                if cc_addr:
                    msg["Cc"] = cc_addr
                msg["Subject"] = subject
                msg["Date"] = dt.datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M:%S %z")
                msg["Message-ID"] = make_msgid()

                # Add body text
                subtype = "html" if is_html else "plain"
                body_part = MIMEText(raw_body, subtype, "utf-8")
                msg.attach(body_part)

                # Attach image if provided
                if img_frame is not None:
                    try:
                        bgr = _bgr(img_frame)
                        success, buf = cv2.imencode(".png", bgr)
                        if success:
                            img_part = MIMEImage(buf.tobytes(), name="kamera_denetim.png")
                            img_part.add_header("Content-Disposition", "attachment", filename="kamera_denetim.png")
                            msg.attach(img_part)
                    except Exception as img_exc:
                        print(f"[E-posta Gönder] Görsel eklenemedi: {img_exc}")

                # Attach file if provided
                if att_path:
                    resolved = _node.resolve_path(att_path)
                    p = pathlib.Path(resolved)
                    if p.is_file():
                        with open(p, "rb") as fh:
                            part = MIMEApplication(fh.read(), Name=p.name)
                        part.add_header("Content-Disposition", "attachment", filename=p.name)
                        msg.attach(part)
                    else:
                        print(f"[E-posta Gönder] Ek dosya bulunamadı: {att_path}")

                recipients = [a.strip() for a in to_addr.split(",") if a.strip()]
                if cc_addr:
                    recipients += [a.strip() for a in cc_addr.split(",") if a.strip()]

                if sec == "SSL/TLS (465)":
                    with smtplib.SMTP_SSL(host, port, timeout=25) as s:
                        if user and passwd:
                            s.login(user, passwd)
                        s.send_message(msg, from_addr=user or from_header, to_addrs=recipients)
                else:
                    with smtplib.SMTP(host, port, timeout=25) as s:
                        if sec.startswith("STARTTLS"):
                            s.starttls()
                        if user and passwd:
                            s.login(user, passwd)
                        s.send_message(msg, from_addr=user or from_header, to_addrs=recipients)

                st["count"] += 1
                st["status"] = f"Gönderildi ({to_addr}): {dt.datetime.now():%H:%M:%S}"
                print(f"[E-posta Gönder] {st['status']}")
            except Exception as exc:
                st["status"] = f"Hata: {exc}"
                print(f"[E-posta Gönder] Hata ({host}:{port}): {exc}")

        threading.Thread(target=do_email, daemon=True).start()
        return {"sent": True, "count": st["count"], "status": "Gönderim başlatıldı"}


# ==========================================================================================
# 4. Email Listen (E-posta Dinle)
# ==========================================================================================

@register
class EmailListen(Node):
    title = "E-posta Dinle"
    category = CAT_INPUT
    description = (
        "IMAP posta kutusunu dinler. Yeni gelen e-postaları başlık, gönderen, gövde ve ek dosya "
        "(görsel, CSV, rapor) olarak akışa aktarır. Sunum ve rapor oluşturma bloklarına girdi sağlar."
    )
    inputs = [("next", BOOL)]
    outputs = [
        ("new", BOOL),
        ("sender", TEXT),
        ("subject", TEXT),
        ("body", TEXT),
        ("attachment_path", TEXT),
        ("has_attachment", BOOL),
        ("date", TEXT),
        ("waiting", NUM),
        ("mail_full", TEXT),
    ]
    params = [
        Param("imap_host", "str", "imap.gmail.com", "IMAP Sunucu"),
        Param("user", "str", "", "E-posta Hesabı"),
        Param("password", "str", "", "E-posta Şifresi / Uygulama Şifresi"),
        Param("folder", "str", "INBOX", "Klasör"),
        Param("interval", "float", 30.0, "Kontrol Aralığı (sn)", 5.0, 86400.0),
        Param("subject_filter", "str", "", "Konu Filtresi (Boş = Tümü)"),
        Param("sender_filter", "str", "", "Gönderen Filtresi (Boş = Tümü)"),
        Param("save_attachments", "bool", True, "Ek Dosyaları İndir"),
        Param("attachment_folder", "folder", "gelen_ekler", "Ek Dosya Klasörü"),
        Param("mark_seen", "bool", False, "Okundu Olarak İşaretle"),
    ]

    def setup(self):
        self.state.update(
            queue=collections.deque(maxlen=100),
            lock=threading.Lock(),
            seen_uids=set(),
            last_check=0.0,
            cur_sender="",
            cur_subject="",
            cur_body="",
            cur_att="",
            cur_has_att=False,
            cur_date="",
            cur_full="",
            prev_next=False,
        )

    def _fetch_mails(self):
        host = self.v("imap_host").strip()
        user = self.v("user").strip()
        passwd = self.v("password").strip()
        folder = self.v("folder").strip() or "INBOX"
        subj_filter = self.v("subject_filter").strip().lower()
        sender_filter = self.v("sender_filter").strip().lower()
        save_att = bool(self.v("save_attachments"))
        att_folder = pathlib.Path(self.v("attachment_folder") or "gelen_ekler")
        att_folder.mkdir(parents=True, exist_ok=True)

        if not host or not user or not passwd:
            return

        try:
            with imaplib.IMAP4_SSL(host, timeout=20) as im:
                im.login(user, passwd)
                im.select(folder)
                typ, data = im.uid("search", None, "UNSEEN")
                if typ != "OK":
                    return

                uids = (data[0] or b"").split()
                for uid in uids[-15:]:  # Check up to 15 newest unread
                    uid_str = uid.decode()
                    if uid_str in self.state["seen_uids"]:
                        continue

                    typ, parts = im.uid("fetch", uid, "(BODY.PEEK[])")
                    raw = next((p[1] for p in parts if isinstance(p, tuple)), None)
                    if not raw:
                        continue

                    msg = BytesParser(policy=policy.default).parsebytes(raw)
                    sender_header = str(msg.get("from", ""))
                    name, sender_addr = parseaddr(sender_header)
                    subj = str(msg.get("subject", ""))
                    date_str = str(msg.get("date", ""))

                    if subj_filter and subj_filter not in subj.lower():
                        continue
                    if sender_filter and sender_filter not in sender_addr.lower():
                        continue

                    # Extract body text
                    body_text = ""
                    part = msg.get_body(preferencelist=("plain", "html"))
                    if part is not None:
                        body_text = part.get_content()
                        if part.get_content_type() == "text/html":
                            body_text = re.sub(r"(?is)<(script|style).*?</\1>", " ", body_text)
                            body_text = re.sub(r"<[^>]+>", " ", body_text)

                    # Extract attachment if present
                    saved_att_path = ""
                    if save_att:
                        for att in msg.iter_attachments():
                            fname = att.get_filename() or "ek_dosya"
                            clean_fname = re.sub(r'[^\w.\-_]', '_', fname)
                            target = att_folder / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{clean_fname}"
                            with open(target, "wb") as fh:
                                fh.write(att.get_payload(decode=True) or b"")
                            saved_att_path = str(target.resolve())
                            break

                    item = {
                        "sender": sender_addr or sender_header,
                        "subject": subj,
                        "body": body_text.strip(),
                        "attachment_path": saved_att_path,
                        "has_attachment": bool(saved_att_path),
                        "date": date_str,
                        "mail_full": f"Kimden: {sender_header}\nKonu: {subj}\nTarih: {date_str}\n\n{body_text.strip()}",
                    }

                    with self.state["lock"]:
                        self.state["seen_uids"].add(uid_str)
                        self.state["queue"].append(item)

                    if self.v("mark_seen"):
                        im.uid("store", uid, "+FLAGS", "(\\Seen)")
        except Exception as exc:
            print(f"[E-posta Dinle] Hata: {exc}")

    def process(self, inputs):
        st = self.state
        lock: threading.Lock = st["lock"]
        queue: collections.deque = st["queue"]

        # Background poll check
        now = time.monotonic()
        interval = max(5.0, float(self.v("interval")))
        if now - st["last_check"] >= interval:
            st["last_check"] = now
            threading.Thread(target=self._fetch_mails, daemon=True).start()

        next_trig = _rising(self, inputs.get("next"), "prev_next")
        is_new = False

        with lock:
            if queue and (next_trig or not st["cur_body"]):
                cand = queue.popleft()
                st["cur_sender"] = cand["sender"]
                st["cur_subject"] = cand["subject"]
                st["cur_body"] = cand["body"]
                st["cur_att"] = cand["attachment_path"]
                st["cur_has_att"] = cand["has_attachment"]
                st["cur_date"] = cand["date"]
                st["cur_full"] = cand["mail_full"]
                is_new = True
            waiting_cnt = len(queue)

        return {
            "new": is_new,
            "sender": st["cur_sender"],
            "subject": st["cur_subject"],
            "body": st["cur_body"],
            "attachment_path": st["cur_att"],
            "has_attachment": st["cur_has_att"],
            "date": st["cur_date"],
            "waiting": waiting_cnt,
            "mail_full": st["cur_full"],
        }
