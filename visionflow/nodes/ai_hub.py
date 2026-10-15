"""AI centre blocks: ask Claude, your own memory, web research, e-mail, voice-over, video, YouTube.

Slow jobs (anything that talks to the internet) run in the background so the flow keeps running. They start on
a rising 'trigger' or, when no trigger is connected, whenever their main text input changes. 'busy' is true
while working; 'done' is true for exactly one frame when a result is ready, so it can trigger the next step.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import os
import re
import threading

import numpy as np

from ..ai import claude, mail, media
from ..ai.config import AIConfig, home
from ..ai.memory import shared as shared_memory
from ..core import Node, Param, PortType, register
from .learning import IMAGE_EXT
from .outputs import _rising

CAT = "6 · AI Merkezi"
IMG, NUM, BOOL, TEXT, ANY = PortType.IMAGE, PortType.NUMBER, PortType.BOOL, PortType.TEXT, PortType.ANY


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)


class _Job(Node):
    """Background job base. Subclasses set ``key`` (input whose change fires the job when no trigger is
    connected), ``empty`` (outputs before the first result) and implement ``run(inputs) -> dict``."""
    key = "prompt"
    empty: dict = {}

    MAX_QUEUE = 50

    def wants_frames(self) -> bool:
        st = self.state
        return bool(st.get("busy") or st.get("completed") or st.get("pending") or st.get("pulse"))

    def setup(self):
        self.state.update(result=dict(self.empty), busy=False, error=None, last_key=None, runs=0,
                          pending=[], completed=[])
        self._lock = threading.Lock()

    def should_fire(self, inputs) -> bool:
        if inputs.get("trigger") is not None:
            return _rising(self, inputs["trigger"])
        k = inputs.get(self.key)
        if k is None or isinstance(k, str) and not k.strip():
            return False
        if isinstance(k, np.ndarray) or isinstance(k, float) and k != k:    # NaN never equals itself
            return False
        fire = k != self.state["last_key"]
        self.state["last_key"] = k
        return fire

    def _work(self, inputs):
        st = self.state
        while inputs is not None:
            try:
                st["completed"].append(self.run(inputs))
                st["runs"] += 1
                st["error"] = None                  # a later success clears an earlier failure
            except Exception as exc:  # noqa: BLE001
                st["error"] = str(exc)
                print(f"[{self.title}] {exc}")
            with self._lock:                        # so a request queued right now is not stranded
                inputs = st["pending"].pop(0) if st["pending"] else None
                if inputs is None:
                    st["busy"] = False

    def process(self, inputs):
        st = self.state
        if self.should_fire(inputs):
            snap = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inputs.items()}
            with self._lock:
                start = not st["busy"]
                if start:
                    st.update(busy=True, error=None)
                elif len(st["pending"]) < self.MAX_QUEUE:
                    # requests that arrive while working wait their turn instead of being dropped
                    st["pending"].append(snap)
                else:
                    print(f"[{self.title}] kuyruk dolu, istek atlandı")
            if start:
                threading.Thread(target=self._work, args=(snap,), daemon=True).start()
        done = False
        if st["completed"]:              # one finished result per frame, so every 'done' pulse is seen
            st["result"], done = st["completed"].pop(0), True
        st["pulse"] = done
        if st["error"] and not st["busy"] and not done:
            raise RuntimeError(st["error"])
        return {**st["result"], "busy": st["busy"] or bool(st["completed"]), "done": done}

    def run(self, inputs) -> dict:
        raise NotImplementedError


def _cfg() -> AIConfig:
    return AIConfig.load()


# --- text ----------------------------------------------------------------------------------------------

@register
class TextConst(Node):
    title = "Metin"
    category = CAT
    description = "Sabit bir metin verir: soru, konu, talimat, e-posta şablonu."
    outputs = [("text", TEXT)]
    params = [Param("text", "code", "Bugünün yapay zeka haberleri", "Metin")]

    def process(self, inputs):
        return {"text": self.v("text")}


@register
class SaveText(Node):
    title = "Metni Dosyaya Kaydet"
    category = CAT
    description = "Metni (rapor, cevap, özet) dosyaya yazar. {tarih} yerine tarih-saat gelir. Tetik yoksa metin değişince yazar."
    inputs = [("text", ANY), ("trigger", BOOL)]
    outputs = [("path", TEXT)]
    params = [Param("folder", "str", "raporlar", "Klasör"), Param("name", "str", "rapor_{tarih}.md", "Dosya adı"),
              Param("append", "bool", False, "Aynı dosyanın sonuna ekle")]

    def setup(self):
        self.state.update(last=None, path="")

    def process(self, inputs):
        text = _s(inputs["text"])
        fire = _rising(self, inputs["trigger"]) if inputs["trigger"] is not None else (text and text != self.state["last"])
        if fire and text:
            name = self.v("name").replace("{tarih}", f"{dt.datetime.now():%Y-%m-%d_%H%M%S}")
            p = pathlib.Path(self.v("folder")) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a" if self.v("append") else "w", encoding="utf-8") as f:
                f.write(text + ("\n\n" if self.v("append") else ""))
            self.state.update(last=text, path=str(p))
        return {"path": self.state["path"]}


# --- Claude --------------------------------------------------------------------------------------------

@register
class AskAI(_Job):
    title = "Yapay Zekaya Sor"
    category = CAT
    description = ("Claude'a soru sorar veya görev verir (yazı yaz, özetle, çevir, e-postaya cevap hazırla, resmi "
                   "yorumla). 'Hafızayı kullan' açıksa kendi belgelerinizden ilgili bilgileri de verir. "
                   "Şablonda {prompt} ve {context} kullanılabilir.")
    inputs = [("trigger", BOOL), ("prompt", ANY), ("context", ANY), ("image", IMG)]
    outputs = [("text", TEXT), ("busy", BOOL), ("done", BOOL)]
    params = [Param("template", "code", "{prompt}", "Görev şablonu"),
              Param("system", "code", "", "Kişilik / talimat (boş = AI Merkezi ayarı)"),
              Param("use_memory", "bool", True, "Hafızayı kullan"),
              Param("chat_memory", "bool", False, "Önceki konuşmaları hatırla"),
              Param("session", "str", "genel", "Konuşma adı"),
              Param("effort", "choice", "medium", "Düşünme", choices=["low", "medium", "high"]),
              Param("max_tokens", "int", 3000, "En uzun cevap (token)", 100, 64000)]
    empty = {"text": ""}

    def build_prompt(self, inputs) -> str:
        prompt, ctx = _s(inputs["prompt"]), _s(inputs["context"])
        t = self.v("template") or "{prompt}"
        text = t.replace("{prompt}", prompt).replace("{context}", ctx)
        if ctx and "{context}" not in t:
            text += f"\n\n{ctx}"
        return text.strip()

    def run(self, inputs):
        cfg = _cfg()
        prompt = self.build_prompt(inputs)
        if not prompt and inputs["image"] is None:
            raise ValueError("Soru boş")
        mem = shared_memory(cfg.memory_file())
        full = prompt
        if self.v("use_memory"):
            ctx = mem.context_for(prompt)
            if ctx:
                full = f"{ctx}\n\n---\n{prompt}"
        history = mem.history(self.v("session")) if self.v("chat_memory") else None
        answer = claude.ask(full, system=self.v("system") or None, image=inputs["image"], history=history,
                            cfg=cfg, max_tokens=self.v("max_tokens"), effort=self.v("effort"))
        if self.v("chat_memory"):
            mem.log(self.v("session"), "user", prompt)
            mem.log(self.v("session"), "assistant", answer)
        return {"text": answer}


@register
class AIClassify(_Job):
    title = "Yapay Zeka ile Sınıflandır"
    category = CAT
    description = ("Metni verdiğiniz kategorilerden birine ayırır (ör. e-posta: acil, soru, şikayet, reklam). "
                   "'Eşleşme' seçtiğiniz kategoriye düşünce doğru olur.")
    inputs = [("text", ANY), ("trigger", BOOL)]
    outputs = [("label", TEXT), ("match", BOOL), ("busy", BOOL), ("done", BOOL)]
    params = [Param("categories", "str", "acil,soru,şikayet,sipariş,reklam", "Kategoriler"),
              Param("target", "str", "acil", "Eşleşme kategorisi"),
              Param("hint", "str", "", "Ek açıklama (isteğe bağlı)")]
    key = "text"
    empty = {"label": "", "match": False}

    def run(self, inputs):
        cats = [c.strip() for c in self.v("categories").split(",") if c.strip()]
        if not cats:
            raise ValueError("Kategori listesi boş")
        system = ("Sen bir sınıflandırıcısın. Verilen metni kategorilerden birine ayır. Cevap olarak SADECE kategori "
                  "adını yaz, başka hiçbir şey yazma. Kategoriler: " + ", ".join(cats)
                  + (f"\nNot: {self.v('hint')}" if self.v("hint") else ""))
        ans = claude.ask(_s(inputs["text"])[:20000], system=system, max_tokens=50, effort="low").strip().lower()
        label = next((c for c in cats if c.lower() == ans), None) \
            or next((c for c in cats if c.lower() in ans), "diğer")
        return {"label": label, "match": label.lower() == self.v("target").strip().lower()}


@register
class WebResearch(_Job):
    title = "İnternette Araştır"
    category = CAT
    description = ("Claude internette arar, kaynakları okur ve Türkçe rapor yazar. İsterseniz sonucu hafızaya "
                   "da ekler. Konu girişi değişince veya tetik gelince çalışır.")
    inputs = [("topic", ANY), ("trigger", BOOL)]
    outputs = [("report", TEXT), ("sources", TEXT), ("busy", BOOL), ("done", BOOL)]
    params = [Param("searches", "int", 5, "En fazla arama", 1, 20),
              Param("detailed", "bool", False, "Ayrıntılı rapor"),
              Param("to_memory", "bool", True, "Hafızaya ekle")]
    key = "topic"
    empty = {"report": "", "sources": ""}

    def run(self, inputs):
        topic = _s(inputs["topic"]).strip()
        if not topic:
            raise ValueError("Araştırma konusu boş")
        cfg = _cfg()
        report, sources = claude.research(topic, self.v("searches"), self.v("detailed"), cfg=cfg)
        src = "\n".join(f"- {s['title']}: {s['url']}" for s in sources)
        if self.v("to_memory") and report:
            shared_memory(cfg.memory_file()).add_text(
                f"Araştırma: {topic}\n\n{report}\n\nKaynaklar:\n{src}",
                source=f"araştırma {dt.datetime.now():%Y-%m-%d %H:%M} – {topic[:40]}")
        return {"report": report, "sources": src}


# --- memory --------------------------------------------------------------------------------------------

@register
class MemoryAdd(Node):
    title = "Hafızaya Ekle"
    category = CAT
    description = ("Kendi yapay zekanızın hafızasına bilgi ekler: gelen metni (tetik veya değişince) ve/veya bir "
                   "klasördeki belgeleri (txt, md, pdf, docx, eml, csv). Hafıza bu bilgisayarda kalır.")
    inputs = [("text", ANY), ("trigger", BOOL)]
    outputs = [("passages", NUM)]
    params = [Param("source", "str", "notlar", "Kaynak adı"),
              Param("folder", "folder", "", "Belge klasörü (başlarken okunur)")]

    def setup(self):
        self.state.update(mem=shared_memory(_cfg().memory_file()), last=None, n=0)
        if self.v("folder"):
            folder = pathlib.Path(self.v("folder"))
            if not folder.is_dir():
                raise ValueError(f"Klasör yok: {folder}")
            res = self.state["mem"].add_folder(folder)
            self.state["n"] = sum(v for v in res.values() if isinstance(v, int))
            print(f"[Hafıza] {len(res)} belge, {self.state['n']} parça eklendi")

    def process(self, inputs):
        text = _s(inputs["text"]).strip()
        fire = _rising(self, inputs["trigger"]) if inputs["trigger"] is not None else (text and text != self.state["last"])
        if fire and text:
            self.state["n"] += self.state["mem"].add_text(text, source=self.v("source"))
            self.state["last"] = text
        return {"passages": self.state["n"]}


@register
class MemorySearch(Node):
    title = "Hafızada Ara"
    category = CAT
    description = "Hafızada soruyla en ilgili bilgileri bulur (internet ve yapay zeka kullanmaz, anında çalışır)."
    inputs = [("query", ANY)]
    outputs = [("results", TEXT), ("count", NUM)]
    params = [Param("k", "int", 4, "Kaç sonuç", 1, 20)]

    def setup(self):
        self.state.update(mem=shared_memory(_cfg().memory_file()), q=None, out={"results": "", "count": 0})

    def process(self, inputs):
        q = _s(inputs["query"]).strip()
        if q and q != self.state["q"]:
            hits = self.state["mem"].search(q, self.v("k"))
            self.state.update(q=q, out={"results": "\n\n".join(f"[{s}] {t}" for s, t in hits), "count": len(hits)})
        return self.state["out"]


# --- e-mail --------------------------------------------------------------------------------------------

class _Feed(Node):
    """Source that polls something slow (mailbox, Telegram, folder) in the background and hands out one item at
    a time. Without a 'next' input the next item comes on the next frame; with 'next' connected (e.g. from the
    last step's 'done') the item is held until that step finishes, so replies always go to the right sender."""
    is_source = True
    feedback_inputs = ("next",)
    interval_param = "interval"
    HOLD_TIMEOUT = 600.0   # if the last step never reports 'done' (e.g. it failed), move on after this many seconds

    def setup(self):
        self.state.update(queue=[], current=None, next_poll=0.0, polling=False, error=None,
                          seen=self._load_seen(), prev_next=False, held_since=0.0)

    # persistence of handed-out items, so a restart does not repeat them --------------------------------
    def _seen_file(self) -> pathlib.Path | None:
        return None

    MAX_SEEN = 5000

    def _load_seen(self) -> dict:
        """Handed-out keys in the order they were handed out (a dict is an ordered set), oldest first."""
        f = self._seen_file()
        try:
            return dict.fromkeys(json.loads(f.read_text(encoding="utf-8"))) if f else {}
        except (OSError, ValueError):
            return {}

    def _remember(self, key):
        seen = self.state["seen"]
        seen.pop(key, None)
        seen[key] = None
        while self.MAX_SEEN and len(seen) > self.MAX_SEEN:       # forget the oldest, never the newest
            seen.pop(next(iter(seen)))

    def _save_seen(self):
        f = self._seen_file()
        if f:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(list(self.state["seen"])), encoding="utf-8")
            os.replace(tmp, f)

    # subclass API -----------------------------------------------------------------------------------------
    def fetch(self) -> list:
        raise NotImplementedError

    def item_key(self, item) -> str:
        raise NotImplementedError

    def poll_interval(self) -> float:
        return self.v(self.interval_param)

    def extra_keys(self, item) -> list:
        """More keys to remember for an item (e.g. the mail's UID, so the server need not resend it)."""
        return []

    def outputs_for(self, item) -> dict:
        raise NotImplementedError

    def handed_out(self, item) -> None:
        """Called once when an item goes to the flow: confirm it to the server only now, so items still in the
        queue when the flow stops are fetched again next time instead of being lost."""

    def empty_outputs(self) -> dict:
        raise NotImplementedError

    def _poll(self):
        try:
            queued = {self.item_key(i) for i in self.state["queue"]}
            fresh = [i for i in self.fetch() if self.item_key(i) not in self.state["seen"]
                     and self.item_key(i) not in queued]
            self.state["queue"].extend(fresh)
            self.state["error"] = None
        except Exception as exc:  # noqa: BLE001
            self.state["error"] = str(exc)
        finally:
            self.state["polling"] = False

    def process(self, inputs):
        import time
        st = self.state
        if not st["polling"] and time.monotonic() >= st["next_poll"]:
            st.update(polling=True, next_poll=time.monotonic() + self.poll_interval())
            threading.Thread(target=self._poll, daemon=True).start()
        if st["error"] and not st["queue"]:
            # a failed poll does not hold back items that were already fetched; it shows when they run out
            raise RuntimeError(st["error"])
        nxt = inputs.get("next")
        if nxt is None:
            # one quiet frame between items, so 'new' is a separate rising edge for every item
            advance = not st.get("gave_last", False)
        else:
            stuck = (time.monotonic() - st["held_since"]) >= self.HOLD_TIMEOUT
            advance = st["current"] is None or (bool(nxt) and not st["prev_next"]) or stuck
            st["prev_next"] = bool(nxt)
        new = False
        if advance and st["queue"]:
            st["current"], new = st["queue"].pop(0), True
            st["held_since"] = time.monotonic()
            self._remember(self.item_key(st["current"]))
            for extra in self.extra_keys(st["current"]):
                self._remember(extra)
            self._save_seen()
            self.handed_out(st["current"])
        st["gave_last"] = new
        out = self.outputs_for(st["current"]) if st["current"] is not None else self.empty_outputs()
        return {**out, "new": new, "waiting": len(st["queue"])}


@register
class EmailRead(_Feed):
    setup_params = ("user", "host", "folder")     # which mailbox (and its 'already handled' list) to use
    title = "E-posta Oku"
    category = CAT
    description = ("Gelen kutusundaki okunmamış e-postaları belirli aralıklarla kontrol eder ve sırayla verir "
                   "('yeni' bir kare doğru olur). 'sonraki' girişine son adımın 'done' çıkışını bağlayın: bir e-posta "
                   "işlenmeden sıradakine geçmez. Aynı e-postayı iki kez vermez. Hesap boşsa AI Merkezi ayarları "
                   "kullanılır. Gmail için uygulama şifresi gerekir.")
    inputs = [("next", BOOL)]
    outputs = [("mail", TEXT), ("sender", TEXT), ("subject", TEXT), ("body", TEXT), ("message_id", TEXT),
               ("new", BOOL), ("waiting", NUM)]
    params = [Param("user", "str", "", "E-posta (boş = ayarlar)"), Param("password", "str", "", "Şifre"),
              Param("host", "str", "", "IMAP sunucu (boş = ayarlar)"), Param("folder", "str", "INBOX", "Klasör"),
              Param("interval", "float", 60.0, "Kontrol aralığı (sn)", 10, 86400),
              Param("mark_seen", "bool", False, "Okundu olarak işaretle"),
              Param("skip", "str", "noreply,no-reply,mailer-daemon", "Bu adreslere cevap verme (virgülle)")]

    def _seen_file(self):
        who = re.sub(r"[^\w.@-]", "_", (self.v("user") or _cfg().email_user or "hesap") + "_" + self.v("folder"))
        return home() / "eposta" / f"{who}.json"

    def fetch(self):
        done_uids = {k[4:] for k in list(self.state["seen"]) if k.startswith("uid:")}
        mails = mail.fetch_unseen(self.v("user"), self.v("password"), self.v("host"), self.v("folder"),
                                  False, skip_uids=done_uids)
        skip = [w.strip().lower() for w in self.v("skip").split(",") if w.strip()]
        keep = []
        for m in mails:
            if any(w in m.sender_addr.lower() for w in skip):
                self._remember(f"uid:{m.uid}")     # never download it again (and never let it fill the limit)
            else:
                keep.append(m)
        return keep

    def item_key(self, m):
        return m.message_id or f"uid:{m.uid}"     # UIDs can be renumbered by the server; Message-ID is stable

    def extra_keys(self, m):
        return [f"uid:{m.uid}"]                   # lets the next poll skip it without downloading it again

    def handed_out(self, m):
        if not self.v("mark_seen"):
            return
        args = (self.v("user"), self.v("password"), self.v("host"), self.v("folder"))

        def mark():
            try:
                mail.mark_seen([m.uid], *args)
            except Exception as exc:  # noqa: BLE001 - only the read flag on the server; the mail is handled anyway
                print(f"[E-posta Oku] okundu işaretlenemedi: {exc}")
        threading.Thread(target=mark, daemon=True).start()

    def empty_outputs(self):
        return {"mail": "", "sender": "", "subject": "", "body": "", "message_id": ""}

    def outputs_for(self, m):
        return {"mail": f"Kimden: {m.sender}\nKonu: {m.subject}\nTarih: {m.date}\n\n{m.body}",
                "sender": m.sender_addr, "subject": m.subject, "body": m.body, "message_id": m.message_id}


@register
class EmailReply(_Job):
    title = "E-postayı Cevapla"
    category = CAT
    description = ("Cevabı e-posta olarak hazırlar. Varsayılan: Taslaklar klasörüne kaydeder, siz kontrol edip "
                   "gönderirsiniz. 'Hemen gönder' seçilirse onaysız gönderir.")
    inputs = [("trigger", BOOL), ("to", TEXT), ("subject", TEXT), ("body", ANY), ("message_id", TEXT)]
    outputs = [("status", TEXT), ("count", NUM), ("busy", BOOL), ("done", BOOL)]
    params = [Param("mode", "choice", mail.REPLY_MODES[0], "Ne yapılsın", choices=mail.REPLY_MODES),
              Param("signature", "code", "", "İmza (isteğe bağlı)"),
              Param("folder", "str", "onay_bekleyen", "Onay klasörü"),
              Param("user", "str", "", "E-posta (boş = ayarlar)"), Param("password", "str", "", "Şifre"),
              Param("imap_host", "str", "", "IMAP (boş = ayarlar)"), Param("smtp_host", "str", "", "SMTP (boş = ayarlar)"),
              Param("smtp_port", "int", 0, "SMTP port (0 = ayarlar)", 0, 65535)]
    key = "body"
    empty = {"status": "", "count": 0}

    def run(self, inputs):
        to, body = _s(inputs["to"]).strip(), _s(inputs["body"]).strip()
        if not to or not body:
            raise ValueError("Alıcı veya mesaj boş")
        cfg = _cfg()
        if self.v("signature"):
            body += "\n\n" + self.v("signature")
        msg = mail.build_reply(to, _s(inputs["subject"]), body, self.v("user") or cfg.email_user,
                               _s(inputs["message_id"]))
        status = mail.deliver(msg, self.v("mode"), self.v("user"), self.v("password"), self.v("imap_host"),
                              self.v("smtp_host"), self.v("smtp_port"), self.v("folder"), cfg=cfg)
        self.state["sent"] = self.state.get("sent", 0) + 1     # counted here: results are handed out later
        return {"status": status, "count": self.state["sent"]}


# --- voice, video, YouTube ----------------------------------------------------------------------------

def _stamp(folder: str, prefix: str) -> pathlib.Path:
    return pathlib.Path(folder) / f"{prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}"


@register
class TextToSpeech(_Job):
    title = "Seslendir"
    category = CAT
    description = ("Metni sese çevirir. 'edge' ücretsizdir ve çok doğal Türkçe sesleri var (Emel, Ahmet; internet "
                   "gerekir). 'Bilgisayar sesi' internetsiz çalışır. ElevenLabs en doğal ama ücretlidir.")
    inputs = [("text", ANY), ("trigger", BOOL)]
    outputs = [("path", TEXT), ("busy", BOOL), ("done", BOOL)]
    params = [Param("engine", "choice", media.TTS_ENGINES[0], "Motor", choices=media.TTS_ENGINES),
              Param("voice", "str", "tr-TR-EmelNeural", "Ses (edge: tr-TR-AhmetNeural…, ElevenLabs: ses ID)"),
              Param("rate", "int", 0, "Hız (%)", -50, 100), Param("folder", "str", "sesler", "Klasör")]
    key = "text"
    empty = {"path": ""}

    def run(self, inputs):
        out = media.tts(_s(inputs["text"]), _stamp(self.v("folder"), "ses"), self.v("engine"), self.v("voice"),
                        self.v("rate"))
        return {"path": str(out)}


@register
class MakeVideo(_Job):
    title = "Video Üret"
    category = CAT
    description = ("Metin + seslendirme + resimlerden MP4 video yapar: resimler yavaşça yakınlaşır, cümleler alt "
                   "yazı olarak çıkar. Resim yoksa renkli arka planda büyük yazılarla anlatım videosu olur. "
                   "Ses girişi değişince (seslendirme bitince) veya tetikle çalışır.")
    inputs = [("script", ANY), ("audio", TEXT), ("image", IMG), ("trigger", BOOL)]
    outputs = [("path", TEXT), ("busy", BOOL), ("done", BOOL)]
    params = [Param("images", "folder", "", "Resim klasörü (isteğe bağlı)"),
              Param("size", "choice", list(media.SIZES)[0], "Boyut", choices=list(media.SIZES)),
              Param("subtitles", "bool", True, "Alt yazı"), Param("folder", "str", "videolar", "Klasör")]
    key = "audio"
    empty = {"path": ""}

    def run(self, inputs):
        pics: list = []
        if self.v("images"):
            folder = pathlib.Path(self.v("images"))
            pics = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT) if folder.is_dir() else []
        if inputs["image"] is not None:
            pics.append(inputs["image"])
        out = media.make_video(_s(inputs["script"]), _stamp(self.v("folder"), "video"), inputs["audio"] or None,
                               pics, media.SIZES[self.v("size")], self.v("subtitles"))
        return {"path": str(out)}


@register
class YouTubeUpload(_Job):
    title = "YouTube'a Yükle"
    category = CAT
    description = ("Videoyu YouTube kanalınıza yükler. İlk seferde tarayıcıda Google girişi ister. Güvenlik için "
                   "varsayılan 'private' (sadece siz görürsünüz); kontrol edip yayınlarsınız.")
    inputs = [("trigger", BOOL), ("video", TEXT), ("title", ANY), ("description", ANY)]
    outputs = [("url", TEXT), ("busy", BOOL), ("done", BOOL)]
    params = [Param("privacy", "choice", "private", "Görünürlük", choices=["private", "unlisted", "public"]),
              Param("tags", "str", "", "Etiketler (virgülle)"),
              Param("client_secret", "str", "", "client_secret.json (boş = ayarlar)")]
    key = "video"
    empty = {"url": ""}

    def run(self, inputs):
        title = _s(inputs["title"]).strip().splitlines()[0] if _s(inputs["title"]).strip() else "Yeni video"
        url = media.youtube_upload(_s(inputs["video"]), title.strip("#* "), _s(inputs["description"]),
                                   [t.strip() for t in self.v("tags").split(",") if t.strip()], self.v("privacy"),
                                   self.v("client_secret") or _cfg().youtube_client_secret)
        print(f"[YouTube] yüklendi: {url}")
        return {"url": url}




# --- more inputs: Telegram, web page, folder, documents, schedule ------------------------------------------

@register
class TelegramReceive(_Feed):
    title = "Telegram Mesaj Al"
    category = CAT
    description = ("Telegram botunuza yazılan mesajları sırayla verir: yapay zeka ile Telegram asistanı kurun. "
                   "Güvenlik: sadece izinli sohbetlerden gelen mesajlar alınır (AI Merkezi → Ayarlar); izinsiz "
                   "birinin yazdığı ID günlüğe yazılır. 'sonraki' girişine cevabın 'done' çıkışını bağlayın.")
    inputs = [("next", BOOL)]
    outputs = [("text", TEXT), ("chat_id", TEXT), ("user", TEXT), ("new", BOOL), ("waiting", NUM)]
    params = [Param("token", "str", "", "Bot token (boş = AI Merkezi)"),
              Param("allowed", "str", "", "İzinli chat ID'ler (boş = AI Merkezi)"),
              Param("interval", "float", 3.0, "Kontrol aralığı (sn)", 1, 3600),
              Param("api", "str", "https://api.telegram.org", "API adresi")]

    setup_params = ("token", "api")

    def setup(self):
        super().setup()
        self.state["offset"] = 0

    def _seen_file(self):
        import hashlib
        token = self.v("token") or _cfg().telegram_token or "bot"
        return home() / "telegram" / f"{hashlib.sha1(token.encode()).hexdigest()[:12]}.json"

    def _get(self, method: str, **params):
        import urllib.parse
        import urllib.request
        cfg = _cfg()
        token = self.v("token") or cfg.telegram_token
        if not token:
            raise ValueError("Telegram bot token yok (AI Merkezi → Ayarlar veya bloğun ayarı)")
        url = f"{self.v('api').rstrip('/')}/bot{token}/{method}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read().decode())
        if not data.get("ok"):
            raise RuntimeError(f"Telegram: {data.get('description', 'hata')}")
        return data["result"]

    def fetch(self):
        allowed = {a.strip() for a in (self.v("allowed") or _cfg().telegram_allowed).split(",") if a.strip()}
        items = []
        seen = self.state["seen"]
        pending = False                  # an earlier update is still waiting in the queue
        for u in self._get("getUpdates", offset=self.state["offset"], timeout=0):
            m = u.get("message") or u.get("edited_message")
            uid = str(u["update_id"])
            # the offset confirms updates to Telegram (they are then gone for good), so it only moves past
            # updates that were handed to the flow or ignored; queued ones come again after a restart
            done = uid in seen or not m or not m.get("text") or str(m["chat"]["id"]) not in allowed
            if done and not pending:
                self.state["offset"] = max(self.state["offset"], u["update_id"] + 1)
            pending = pending or not done
            if not m or not m.get("text") or uid in seen:
                continue
            chat = str(m["chat"]["id"])
            who = m.get("from", {}).get("first_name", "") or m["chat"].get("title", "")
            if chat not in allowed:
                print(f"[Telegram] izinsiz sohbetten mesaj yok sayıldı: {who} (chat ID {chat}). "
                      "İzin vermek için bu ID'yi AI Merkezi → Ayarlar'a ekleyin.")
                continue
            items.append({"id": str(u["update_id"]), "text": m["text"], "chat_id": chat, "user": who})
        return items

    def item_key(self, item):
        return item["id"]

    def empty_outputs(self):
        return {"text": "", "chat_id": "", "user": ""}

    def outputs_for(self, item):
        return {"text": item["text"], "chat_id": item["chat_id"], "user": item["user"]}


def html_to_text(raw: str) -> tuple[str, str]:
    import html as _html
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    title = _html.unescape(m.group(1)).strip() if m else ""
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)[^>]*>", "\n", raw)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return title, text.strip()


@register
class WebPage(_Job):
    title = "Web Sayfası Oku"
    category = CAT
    description = ("Bir web sayfasının yazılarını okur (yapay zeka kullanmaz). Aralık verilirse sayfayı düzenli "
                   "kontrol eder; içerik değişince 'changed' bir kare doğru olur: fiyat, duyuru, stok takibi.")
    inputs = [("url", TEXT), ("trigger", BOOL)]
    outputs = [("text", TEXT), ("title", TEXT), ("changed", BOOL), ("busy", BOOL), ("done", BOOL)]
    params = [Param("url", "str", "https://example.com", "Adres (giriş bağlı değilse)"),
              Param("interval", "float", 0.0, "Yenileme aralığı (sn, 0 = sadece tetik/değişince)", 0, 86400),
              Param("max_chars", "int", 20000, "En fazla karakter", 500, 500000)]
    key = "url"
    empty = {"text": "", "title": "", "changed": False}

    def setup(self):
        super().setup()
        self.state.update(next_t=0.0, last_hash=None)

    def should_fire(self, inputs):
        import time
        if inputs.get("url") is None:
            inputs["url"] = self.v("url")
        fire = super().should_fire(inputs)
        if self.v("interval") > 0 and time.monotonic() >= self.state["next_t"]:
            fire = True
        if fire:
            self.state["next_t"] = time.monotonic() + (self.v("interval") or 1e12)
        return fire

    def process(self, inputs):
        out = super().process(inputs)
        if not out["done"]:
            out["changed"] = False      # 'changed' is a one-frame pulse that rides on 'done'
        return out

    def run(self, inputs):
        import hashlib
        import urllib.request
        url = _s(inputs.get("url") or self.v("url")).strip()
        if not re.match(r"https?://", url):
            raise ValueError(f"Geçersiz adres: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (VisionFlow)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read(5_000_000)
            charset = r.headers.get_content_charset() or "utf-8"
        try:
            decoded = raw.decode(charset, "replace")
        except LookupError:                 # unknown charset name in the header, e.g. "utf8mb4"
            decoded = raw.decode("utf-8", "replace")
        title, text = html_to_text(decoded)
        text = text[: self.v("max_chars")]
        h = hashlib.sha1(text.encode()).hexdigest()
        changed = self.state["last_hash"] is not None and h != self.state["last_hash"]
        self.state["last_hash"] = h
        return {"text": text, "title": title, "changed": changed}


@register
class FolderWatch(_Feed):
    setup_params = ("existing", "folder", "pattern")   # read only when the block starts: changing them restarts it
    title = "Klasörü İzle"
    category = CAT
    description = ("Klasöre yeni dosya gelince (tarayıcıdan fatura, kameradan resim, indirilen PDF) yolunu verir. "
                   "Dosya tamamen yazılınca bildirilir. 'Belge Oku' ve 'Yapay Zekaya Sor' ile birleştirin.")
    inputs = [("next", BOOL)]
    outputs = [("path", TEXT), ("name", TEXT), ("new", BOOL), ("waiting", NUM)]
    params = [Param("folder", "folder", "gelen", "Klasör"),
              Param("pattern", "str", "*.pdf,*.docx,*.txt,*.md,*.jpg,*.png", "Dosya türleri"),
              Param("interval", "float", 2.0, "Kontrol aralığı (sn)", 0.5, 3600),
              Param("existing", "bool", False, "Başlarken mevcut dosyaları da ver")]

    def _files(self):
        folder = pathlib.Path(self.v("folder"))
        pats = [p.strip() for p in self.v("pattern").split(",") if p.strip()] or ["*"]
        out = {}
        for pat in pats:
            for f in folder.glob(pat):
                if f.is_file():
                    st = f.stat()
                    out[str(f)] = (st.st_size, st.st_mtime)
        return out

    def setup(self):
        super().setup()
        folder = pathlib.Path(self.v("folder"))
        if not folder.is_dir():
            raise ValueError(f"Klasör yok: {folder}")
        self.state["sizes"] = {}
        if not self.v("existing"):
            self.state["seen"] = dict.fromkeys(self._key(p, s) for p, s in self._files().items())

    @staticmethod
    def _key(path, stat):
        return f"{path}|{stat[1]}"

    MAX_SEEN = 0          # no age limit: what is remembered is exactly the files still in the folder (see fetch)

    def fetch(self):
        files = self._files()
        # forget files that left the folder; never forget ones still there, or they would come back as "new"
        self.state["seen"] = {k: None for k in list(self.state["seen"]) if k.rsplit("|", 1)[0] in files}
        ready = []
        for p, stat in sorted(files.items(), key=lambda kv: kv[1][1]):
            if self.state["sizes"].get(p) == stat:        # unchanged since last look: fully written
                ready.append({"path": p, "stat": stat})
        self.state["sizes"] = files
        return ready

    def item_key(self, item):
        return self._key(item["path"], item["stat"])

    def empty_outputs(self):
        return {"path": "", "name": ""}

    def outputs_for(self, item):
        return {"path": item["path"], "name": pathlib.Path(item["path"]).name}


@register
class ReadDocument(Node):
    title = "Belge Oku"
    category = CAT
    description = "PDF, Word, metin veya e-posta (.eml) dosyasının yazısını çıkarır (yapay zeka kullanmaz)."
    inputs = [("path", TEXT)]
    outputs = [("text", TEXT), ("chars", NUM)]
    params = [Param("max_chars", "int", 100000, "En fazla karakter", 100, 5_000_000)]

    def setup(self):
        self.state.update(path=None, out={"text": "", "chars": 0})

    def process(self, inputs):
        from ..ai.memory import read_document
        p = _s(inputs["path"]).strip()
        if p and p != self.state["path"]:
            try:
                text = read_document(pathlib.Path(p))[: self.v("max_chars")]
            except Exception:
                # never show the previous file's text for a file that could not be read; retry next frame
                self.state.update(path=None, out={"text": "", "chars": 0})
                raise
            self.state.update(path=p, out={"text": text, "chars": len(text)})
        return self.state["out"]


DAYS = ["her gün", "hafta içi", "hafta sonu"]


@register
class Scheduler(Node):
    setup_params = ("times",)   # read only when the block starts: changing them restarts it
    title = "Saatte Çalıştır"
    category = CAT
    description = ("Belirttiğiniz saatlerde (ör. 09:00, 18:30) bir kez tetik verir: günlük rapor, sabah e-posta "
                   "özeti, her akşam video. Akışı açık tutar.")
    outputs = [("tick", BOOL), ("next", TEXT)]
    params = [Param("times", "str", "09:00", "Saatler (virgülle)"),
              Param("days", "choice", DAYS[0], "Günler", choices=DAYS)]
    is_source = True

    def _times(self):
        out = []
        for t in self.v("times").split(","):
            m = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})\s*", t)
            if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
                if t.strip():
                    raise ValueError(f"Saat anlaşılmadı: '{t.strip()}' (ör. 09:00)")
                continue
            out.append((int(m.group(1)), int(m.group(2))))
        if not out:
            raise ValueError("En az bir saat girin (ör. 09:00)")
        return sorted(out)

    def _day_ok(self, d: dt.datetime) -> bool:
        wd = d.weekday()
        return {"hafta içi": wd < 5, "hafta sonu": wd >= 5}.get(self.v("days"), True)

    def setup(self):
        now = self.now()
        # times already past today count as done, so opening the flow at 10:00 does not fire the 09:00 job
        # (the current minute still fires: starting at 09:00:30 runs the 09:00 job)
        self.state["fired"] = {(h, m): now.date() for h, m in self._times() if (h, m) < (now.hour, now.minute)}

    def now(self) -> dt.datetime:
        return dt.datetime.now()

    def process(self, inputs):
        now, tick = self.now(), False
        if self._day_ok(now):
            for h, m in self._times():
                if (now.hour, now.minute) >= (h, m) and self.state["fired"].get((h, m)) != now.date():
                    self.state["fired"][(h, m)] = now.date()
                    tick = True
        return {"tick": tick, "next": self._next_text(now)}

    def _next_text(self, now):
        for add in range(8):
            day = now.date() + dt.timedelta(days=add)
            dtime = dt.datetime.combine(day, dt.time())
            if not self._day_ok(dtime):
                continue
            for h, m in self._times():
                if add or (h, m) > (now.hour, now.minute):
                    return f"sonraki: {day:%d.%m} {h:02d}:{m:02d}"
        return ""


@register
class ExtractInfo(_Job):
    title = "Bilgi Çıkar"
    category = CAT
    description = ("Metinden istediğiniz bilgileri ayıklar (fatura: firma, tutar, tarih; e-posta: ad, telefon, "
                   "sipariş no). İlk üç alan a, b, c çıkışlarına, hepsi 'data' çıkışına gelir; CSV/SQLite'a yazılabilir.")
    inputs = [("text", ANY), ("trigger", BOOL)]
    outputs = [("a", TEXT), ("b", TEXT), ("c", TEXT), ("data", ANY), ("summary", TEXT), ("busy", BOOL),
               ("done", BOOL)]
    params = [Param("fields", "str", "firma,tutar,tarih", "Alanlar (virgülle)"),
              Param("hint", "str", "", "Ek açıklama (isteğe bağlı)")]
    key = "text"
    empty = {"a": "", "b": "", "c": "", "data": {}, "summary": ""}

    def run(self, inputs):
        fields = [f.strip() for f in self.v("fields").split(",") if f.strip()]
        data = claude.extract(_s(inputs["text"]), fields, self.v("hint"))
        vals = [data.get(f, "") for f in fields] + ["", "", ""]
        return {"a": vals[0], "b": vals[1], "c": vals[2], "data": data,
                "summary": ", ".join(f"{k}: {v}" for k, v in data.items())}
