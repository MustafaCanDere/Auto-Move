"""AI centre: memory, Claude blocks (fake model), e-mail (fake IMAP/SMTP), voice, video, YouTube (fake)."""
import json
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from visionflow.ai import claude, mail, media
from visionflow.ai.config import AIConfig
from visionflow.ai.memory import Memory, split_passages
from visionflow.core import Graph, create, load_builtin

load_builtin()


@pytest.fixture(autouse=True)
def ai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_HOME", str(tmp_path / "home"))
    AIConfig(api_key="test").save()
    yield tmp_path


class FakeClaude:
    """Answers by a function of the request; records every request."""

    def __init__(self, fn=None):
        self.calls = []
        self.fn = fn or (lambda kw: "cevap")
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        out = self.fn(kw)
        if isinstance(out, SimpleNamespace):
            return out
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=out)])


@pytest.fixture
def fake_claude(monkeypatch):
    fake = FakeClaude()
    monkeypatch.setattr(claude, "_test_client", fake)
    return fake


def last_user_text(kw):
    c = kw["messages"][-1]["content"]
    return c if isinstance(c, str) else next(b["text"] for b in c if b["type"] == "text")


def pump(node, inputs, until, timeout=20.0):
    """Call process() like the flow loop does until ``until(out)`` holds."""
    end = time.time() + timeout
    while time.time() < end:
        out = node.process(dict(inputs))
        if until(out):
            return out
        time.sleep(0.01)
    raise AssertionError(f"zaman aşımı, son çıktı: {out}")


# --- memory --------------------------------------------------------------------------------------------

def test_memory_search_files_and_history(tmp_path):
    m = Memory(tmp_path / "h.db")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "iade.md").write_text("İade politikası: ürünler 14 gün içinde faturasıyla iade edilir.",
                                               encoding="utf-8")
    (tmp_path / "docs" / "saat.txt").write_bytes("Mağaza cumartesi 10:00-14:00 arası açıktır.".encode("cp1254"))
    (tmp_path / "docs" / "resim.png").write_bytes(b"x")
    res = m.add_folder(tmp_path / "docs")
    assert set(res) == {"iade.md", "saat.txt"}
    assert m.search("İade için fatura lazım mı?")[0][0] == "iade.md"
    assert m.search("cumartesi açık mı")[0][0] == "saat.txt"
    assert "iade.md" in m.context_for("faturasız iade")
    m.add_file(tmp_path / "docs" / "iade.md")                     # re-adding replaces, no duplicates
    assert [s for s in m.sources() if s[0] == "iade.md"][0][2] == 1
    m.remove("saat.txt")
    assert m.search("cumartesi") == []
    m.log("s", "assistant", "yetim cevap")                        # history must start with the user
    m.log("s", "user", "merhaba")
    m.log("s", "assistant", "selam")
    assert [h["role"] for h in m.history("s")] == ["user", "assistant"]


def test_split_long_text():
    text = ("Bu bir cümle. " * 400).strip()
    parts = split_passages(text)
    assert len(parts) > 3 and all(len(p) <= 950 for p in parts)


# --- Claude blocks ------------------------------------------------------------------------------------------

def test_ask_ai_uses_memory_persona_and_chat(fake_claude):
    from visionflow.ai.memory import shared
    shared(AIConfig.load().memory_file()).add_text("Şirketin kargo firması Aras Kargo'dur.", "bilgi.txt")
    fake_claude.fn = lambda kw: "Aras Kargo ile gönderiyoruz."
    node = create("AskAI")
    node.values.update(template="Müşteri soruyor: {prompt}", chat_memory=True, session="t")
    node.setup()
    out = pump(node, {"trigger": None, "prompt": "Hangi kargo ile gönderiyorsunuz?", "context": None, "image": None},
               lambda o: o["done"])
    assert out["text"] == "Aras Kargo ile gönderiyoruz." and not out["busy"]
    kw = fake_claude.calls[0]
    assert "Aras Kargo" in last_user_text(kw) and "Müşteri soruyor" in last_user_text(kw)
    assert kw["system"] == AIConfig.load().persona and kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default"
    # same prompt again: no new call; new prompt: history of the first exchange is sent along
    node.process({"trigger": None, "prompt": "Hangi kargo ile gönderiyorsunuz?", "context": None, "image": None})
    pump(node, {"trigger": None, "prompt": "Ücreti ne kadar?", "context": None, "image": None}, lambda o: o["done"])
    assert len(fake_claude.calls) == 2
    assert [m["role"] for m in fake_claude.calls[1]["messages"]] == ["user", "assistant", "user"]


def test_ask_ai_sends_image_and_reports_errors(fake_claude):
    node = create("AskAI")
    node.values["use_memory"] = False
    node.setup()
    img = np.zeros((40, 60, 3), np.uint8)
    pump(node, {"trigger": True, "prompt": "Bu resimde ne var?", "context": None, "image": img}, lambda o: o["done"])
    blocks = fake_claude.calls[0]["messages"][-1]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/jpeg"

    def boom(kw):
        raise claude.AIError("Claude API anahtarı geçersiz")
    fake_claude.fn = boom
    node.process({"trigger": False, "prompt": "x", "context": None, "image": None})
    with pytest.raises(RuntimeError, match="anahtar"):
        pump(node, {"trigger": True, "prompt": "x", "context": None, "image": None}, lambda o: False, timeout=2)


def test_refusal_becomes_error(fake_claude):
    fake_claude.fn = lambda kw: SimpleNamespace(stop_reason="refusal", content=[])
    with pytest.raises(claude.AIError, match="yanıtlamadı"):
        claude.ask("x")


def test_classify(fake_claude):
    fake_claude.fn = lambda kw: "Şikayet"
    node = create("AIClassify")
    node.setup()
    out = pump(node, {"text": "Ürün kırık geldi, çok kızgınım", "trigger": None}, lambda o: o["done"])
    assert out["label"] == "şikayet" and out["match"] is False
    assert "Kategoriler" in fake_claude.calls[0]["system"]


def test_research_handles_pause_and_collects_sources(fake_claude):
    src = SimpleNamespace(type="web_search_result", url="https://ornek.com/a", title="Örnek haber")
    responses = [
        SimpleNamespace(stop_reason="pause_turn", content=[SimpleNamespace(type="server_tool_use")]),
        SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="web_search_tool_result", content=[src, src]),
            SimpleNamespace(type="text", text="Yapay zekada bu hafta üç önemli gelişme oldu.")]),
    ]
    fake_claude.fn = lambda kw: responses.pop(0)
    node = create("WebResearch")
    node.setup()
    out = pump(node, {"topic": "yapay zeka haberleri", "trigger": None}, lambda o: o["done"])
    assert out["report"].startswith("Yapay zekada") and out["sources"].count("ornek.com") == 1
    assert fake_claude.calls[0]["tools"][0]["type"] == "web_search_20260209"
    assert fake_claude.calls[1]["messages"][-1]["role"] == "assistant"      # resumed, not a new question
    from visionflow.ai.memory import shared
    assert shared(AIConfig.load().memory_file()).search("gelişme")


# --- e-mail ------------------------------------------------------------------------------------------------

RAW = (b"From: Ali Veli <ali@ornek.com>\r\nTo: ben@ornek.com\r\nSubject: =?utf-8?q?Sipari=C5=9F?=\r\n"
       b"Message-ID: <abc@ornek.com>\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
       b"<html><body><p>Merhaba, sipari\xc5\x9fim nerede?</p><script>x()</script></body></html>\r\n")


class FakeIMAP:
    boxes = {}
    appended = []

    def __init__(self, host, timeout=None):
        self.host = host

    def login(self, user, pw):
        if pw != "dogru":
            import imaplib
            raise imaplib.IMAP4.error("AUTHENTICATIONFAILED")

    def select(self, folder):
        self.folder = folder
        return "OK", [b"1"]

    def uid(self, cmd, *args):
        if cmd == "search":
            return "OK", [b" ".join(self.boxes)]
        if cmd == "fetch":
            return "OK", [(b"1 (UID x BODY[] {n}", self.boxes[args[0]]), b")"]
        return "OK", [b""]

    def list(self):
        return "OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\Drafts \\HasNoChildren) "/" "[Gmail]/Taslaklar"']

    def append(self, box, flags, date, data):
        self.appended.append((box, flags, data))
        return "OK", [b""]

    def logout(self):
        pass


@pytest.fixture
def fake_imap(monkeypatch):
    FakeIMAP.boxes = {b"7": RAW}
    FakeIMAP.appended = []
    monkeypatch.setattr(mail, "IMAP_CLASS", FakeIMAP)
    c = AIConfig.load()
    c.email_user, c.email_password, c.imap_host = "ben@ornek.com", "dogru", "imap.ornek.com"
    c.save()
    return FakeIMAP


def test_fetch_and_parse_html_mail(fake_imap):
    [m] = mail.fetch_unseen()
    assert m.sender_addr == "ali@ornek.com" and m.subject == "Sipariş" and m.message_id == "<abc@ornek.com>"
    assert m.body == "Merhaba, siparişim nerede?"
    with pytest.raises(mail.MailError, match="uygulama şifresi"):
        mail.fetch_unseen(password="yanlis")


def test_email_read_gives_each_mail_once_across_restarts(fake_imap):
    node = create("EmailRead")
    node.setup()
    out = pump(node, {}, lambda o: o["new"])
    assert out["sender"] == "ali@ornek.com" and "siparişim" in out["mail"]
    assert node.process({})["new"] is False
    node2 = create("EmailRead")      # new run: same unread mail must not come again
    node2.values["interval"] = 10
    node2.setup()
    time.sleep(0.3)
    assert not any(node2.process({})["new"] for _ in range(20))


def test_reply_saved_as_draft_in_detected_folder(fake_imap):
    msg = mail.build_reply("ali@ornek.com", "Sipariş", "Kargoya verildi.", "ben@ornek.com", "<abc@ornek.com>")
    where = mail.deliver(msg, mail.REPLY_MODES[0])
    box, flags, data = fake_imap.appended[0]
    assert box == '"[Gmail]/Taslaklar"' and "Taslaklar" in where
    assert b"In-Reply-To: <abc@ornek.com>" in data and b"Subject: Re: " in data


def test_reply_for_approval_writes_eml(tmp_path):
    msg = mail.build_reply("a@b.com", "Re: Soru", "Cevap", "ben@ornek.com")
    where = mail.deliver(msg, "onay klasörüne kaydet", folder=str(tmp_path / "onay"))
    files = list((tmp_path / "onay").glob("*.eml"))
    assert len(files) == 1 and "onay" in where and b"Subject: Re: Soru" in files[0].read_bytes()


def test_email_assistant_flow_end_to_end(fake_imap, fake_claude, tmp_path):
    """E-posta Oku → Yapay Zekaya Sor → E-postayı Cevapla (taslak), run by the real flow engine."""
    fake_claude.fn = lambda kw: "Merhaba Ali Bey, siparişiniz bugün kargoya verildi."
    g = Graph()
    rd = g.add_new("EmailRead")
    ai = g.add_new("AskAI", template="Bu e-postaya kibar bir cevap yaz:\n{prompt}")
    rp = g.add_new("EmailReply", signature="Saygılarımla")
    for a, ap, b, bp in [(rd, "mail", ai, "prompt"), (rd, "new", ai, "trigger"), (ai, "done", rp, "trigger"),
                         (ai, "text", rp, "body"), (rd, "sender", rp, "to"), (rd, "subject", rp, "subject"),
                         (rd, "message_id", rp, "message_id")]:
        g.connect(a.id, ap, b.id, bp)
    g.run(max_frames=400, max_fps=100, on_frame=lambda i, r: not fake_imap.appended)
    assert len(fake_imap.appended) == 1
    data = fake_imap.appended[0][2]
    from email import message_from_bytes, policy
    sent = message_from_bytes(data, policy=policy.default)
    assert sent["To"] == "ali@ornek.com" and sent["Subject"] == "Re: Sipariş"
    assert sent.get_content().strip().endswith("Saygılarımla") and "kargoya verildi" in sent.get_content()
    assert "siparişim nerede" in last_user_text(fake_claude.calls[0])


# --- voice / video / YouTube -------------------------------------------------------------------------------

def _tone(path, seconds=2.0):
    subprocess.run([media.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=300:duration={seconds}", str(path)], check=True)


def test_edge_tts_with_sentence_timings(tmp_path, monkeypatch):
    edge_tts = pytest.importorskip("edge_tts")
    _tone(tmp_path / "t.mp3", 3)
    audio = (tmp_path / "t.mp3").read_bytes()

    class FakeCommunicate:
        def __init__(self, text, voice, rate="+0%"):
            assert voice == "tr-TR-AhmetNeural" and rate == "+10%"

        async def stream(self):
            yield {"type": "SentenceBoundary", "offset": 0, "duration": 1.2e7, "text": "Birinci cümle."}
            yield {"type": "audio", "data": audio}
            yield {"type": "SentenceBoundary", "offset": 1.5e7, "duration": 1.0e7, "text": "İkinci cümle."}
    monkeypatch.setattr(edge_tts, "Communicate", FakeCommunicate)
    out = media.tts("Birinci cümle. İkinci cümle.", tmp_path / "ses", voice="tr-TR-AhmetNeural", rate=10)
    assert out.suffix == ".mp3" and out.read_bytes() == audio
    segs = media.segments_for("", 3.0, out)
    assert [s[2] for s in segs] == ["Birinci cümle.", "İkinci cümle."]
    assert segs[0][0] == 0 and segs[1][0] == 1.5 and segs[1][1] == 3.0


def test_offline_tts_if_available(tmp_path):
    try:
        out = media.tts("Merhaba, bu bir deneme.", tmp_path / "s", engine=media.TTS_ENGINES[1])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"çevrimdışı ses motoru yok: {exc}")
    assert media.audio_duration(out) > 0.5


def test_make_video_with_audio_and_images(tmp_path):
    _tone(tmp_path / "a.wav", 2)
    img = np.full((300, 400, 3), (40, 120, 200), np.uint8)
    out = media.make_video("Birinci cümle. İkinci cümle burada.", tmp_path / "v", tmp_path / "a.wav", [img],
                           size=(320, 180))
    assert out.exists() and abs(media.audio_duration(out) - 2.0) < 0.3
    probe = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-i", str(out)], capture_output=True, text=True).stderr
    assert "Video: h264" in probe and "Audio: aac" in probe and "320x180" in probe


def test_video_block_without_audio_makes_title_cards(tmp_path):
    node = create("MakeVideo")
    node.values.update(folder=str(tmp_path), size="küçük 1280x720")
    node.setup()
    out = pump(node, {"script": "Kısa bir video.", "audio": None, "image": None, "trigger": True},
               lambda o: o["done"], timeout=60)
    assert out["path"].endswith(".mp4") and os.path.getsize(out["path"]) > 1000


def test_youtube_upload_with_fake_service(tmp_path, monkeypatch):
    calls = {}

    class Req:
        def next_chunk(self):
            return None, {"id": "abc123"}

    class Videos:
        def insert(self, part, body, media_body):
            calls.update(part=part, body=body)
            return Req()
    monkeypatch.setattr(media, "_test_youtube", SimpleNamespace(videos=lambda: Videos()))
    (tmp_path / "v.mp4").write_bytes(b"0" * 100)
    node = create("YouTubeUpload")
    node.values["tags"] = "yapay zeka, haber"
    node.setup()
    out = pump(node, {"trigger": True, "video": str(tmp_path / "v.mp4"), "title": "# Bugünün haberleri\nikinci satır",
                      "description": "açıklama"}, lambda o: o["done"])
    assert out["url"] == "https://youtu.be/abc123"
    assert calls["body"]["status"]["privacyStatus"] == "private"
    assert calls["body"]["snippet"]["title"] == "Bugünün haberleri"
    assert calls["body"]["snippet"]["tags"] == ["yapay zeka", "haber"]


def test_timer_is_a_source_and_can_fire_at_start():
    t = create("IntervalTimer")
    t.values.update(seconds=3600, at_start=True)
    t.setup()
    assert t.is_source and t.process({})["tick"] is True and t.process({})["tick"] is False


def test_save_text_and_memory_blocks(tmp_path):
    node = create("SaveText")
    node.values.update(folder=str(tmp_path), name="r_{tarih}.md")
    node.setup()
    p = node.process({"text": "rapor", "trigger": None})["path"]
    assert open(p, encoding="utf-8").read() == "rapor"
    add = create("MemoryAdd")
    add.values["source"] = "deneme"
    add.setup()
    assert add.process({"text": "Kedimin adı Pamuk.", "trigger": None})["passages"] == 1
    s = create("MemorySearch")
    s.setup()
    assert "Pamuk" in s.process({"query": "kedinin adı ne"})["results"]


def test_config_roundtrip_is_private():
    c = AIConfig.load()
    c.email_user = "x@y.z"
    c.save()
    assert AIConfig.load().email_user == "x@y.z"
    if os.name == "posix":
        assert oct(AIConfig.path().stat().st_mode & 0o777) == "0o600"
    assert json.loads(AIConfig.path().read_text(encoding="utf-8"))["api_key"] == "test"


# --- AI Merkezi window -------------------------------------------------------------------------------------

def test_ai_center_window(fake_claude, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from visionflow.ui.ai_center import AICenter
    (tmp_path / "fiyat.txt").write_text("Premium paket aylık 499 TL.", encoding="utf-8")
    w = AICenter()
    w.add_files([str(tmp_path / "fiyat.txt")])
    assert w.src_tree.topLevelItemCount() == 1
    w.mem_q.setText("premium fiyat")
    w.search_memory()
    assert "499" in w.mem_results.toPlainText()

    fake_claude.fn = lambda kw: "Premium paket aylık 499 TL'dir."
    w.chat_in.setPlainText("Premium ne kadar?")
    w.send()
    end = time.time() + 5
    while not w.last_answer and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    assert w.last_answer.startswith("Premium") and "499" in last_user_text(fake_claude.calls[0])
    assert "Premium paket aylık 499" in w.transcript.toPlainText()

    w.f_mail.setText("ben@ornek.com")
    w.f_model.setCurrentText("claude-sonnet-5-5")
    w.save_settings()
    cfg = AIConfig.load()
    assert cfg.email_user == "ben@ornek.com" and cfg.model == "claude-sonnet-5-5" and cfg.api_key == "test"
    w2 = AICenter()                                  # history survives reopening
    assert "Premium ne kadar?" in w2.transcript.toPlainText()
    w2.new_chat()
    assert w2.transcript.toPlainText() == ""
    w.close()
    w2.close()


def test_research_to_video_example_runs(fake_claude, tmp_path, monkeypatch):
    """The shipped example, end to end: research → script → voice → MP4 (fake Claude, offline voice)."""
    import pathlib
    ex = pathlib.Path(__file__).resolve().parents[1] / "examples" / "arastir_seslendir_video.json"
    g = Graph.load(str(ex))
    by = {n.type_name: n for n in g.nodes.values()}
    assert by["YouTubeUpload"].enabled is False

    def answer(kw):
        if kw.get("tools"):
            return "Bu hafta iki önemli yapay zeka gelişmesi oldu."
        return "Bu hafta yapay zekada iki önemli gelişme var. Birincisi yeni modeller. İkincisi yeni kurallar."
    fake_claude.fn = answer
    try:
        media.tts("deneme", tmp_path / "x", engine=media.TTS_ENGINES[1])
        by["TextToSpeech"].values["engine"] = media.TTS_ENGINES[1]
    except Exception:  # noqa: BLE001
        pytest.skip("çevrimdışı ses motoru yok")
    by["TextToSpeech"].values["folder"] = str(tmp_path / "ses")
    by["MakeVideo"].values.update(folder=str(tmp_path / "video"), size="küçük 1280x720")
    by["SaveText"].values["folder"] = str(tmp_path / "rapor")
    done = {}

    def frame(i, res):
        r = res.get(by["MakeVideo"].id)
        if r and r.outputs.get("path"):
            done["path"] = r.outputs["path"]
            return False          # stops the flow
    g.run(max_frames=6000, max_fps=100, on_frame=frame)
    assert done and media.audio_duration(done["path"]) > 1
    assert len(fake_claude.calls) == 2 and list((tmp_path / "rapor").glob("*.md"))
