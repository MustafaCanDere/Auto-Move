"""Voice-over (text to speech), video assembly and YouTube upload for the AI centre blocks."""
from __future__ import annotations

import asyncio
import json
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request

import cv2
import numpy as np

from .config import AIConfig, home


def _imread(path):
    """cv2.imread fails on Windows for paths with Turkish letters ("Masaüstü"); decode the bytes instead."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None

TTS_ENGINES = ["edge (ücretsiz, internet)", "bilgisayar sesi (çevrimdışı)", "ElevenLabs (en doğal, ücretli)"]
EDGE_VOICES = ["tr-TR-EmelNeural", "tr-TR-AhmetNeural", "en-US-AriaNeural", "en-US-GuyNeural", "de-DE-KatjaNeural"]
SIZES = {"yatay 1920x1080 (YouTube)": (1920, 1080), "dikey 1080x1920 (Shorts/Reels)": (1080, 1920),
         "kare 1080x1080": (1080, 1080), "küçük 1280x720": (1280, 720)}
FPS = 25


class MediaError(Exception):
    pass


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# --- text to speech ------------------------------------------------------------------------------------

def _timings_path(audio: pathlib.Path) -> pathlib.Path:
    return audio.with_suffix(".zaman.json")


def tts(text: str, out: str | pathlib.Path, engine: str = TTS_ENGINES[0], voice: str = "tr-TR-EmelNeural",
        rate: int = 0, cfg: AIConfig | None = None) -> pathlib.Path:
    """Speak ``text`` into an audio file. The edge engine also writes sentence timings for exact subtitles."""
    if not text.strip():
        raise MediaError("Seslendirilecek metin boş")
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if engine.startswith("edge"):
        try:
            import edge_tts
        except ImportError:
            raise MediaError("pip install edge-tts") from None
        out = out.with_suffix(".mp3")

        async def run():
            com = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%")
            marks = []
            with open(out, "wb") as f:
                async for chunk in com.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] in ("SentenceBoundary", "WordBoundary"):
                        marks.append({"start": chunk["offset"] / 1e7, "end": (chunk["offset"] + chunk["duration"]) / 1e7,
                                      "text": chunk["text"], "kind": chunk["type"]})
            return marks
        try:
            marks = asyncio.run(run())
        except Exception as exc:  # noqa: BLE001
            raise MediaError(f"Seslendirme başarısız (internet?): {exc}") from None
        sent = [m for m in marks if m["kind"] == "SentenceBoundary"]
        if sent:
            _timings_path(out).write_text(json.dumps(sent, ensure_ascii=False), encoding="utf-8")
        return out
    if engine.startswith("bilgisayar"):
        out = out.with_suffix(".wav")
        _offline_tts(text, out, voice, rate)
        if not out.exists() or out.stat().st_size < 100:
            raise MediaError("Bilgisayar sesi dosya üretemedi")
        return out
    # ElevenLabs
    cfg = cfg or AIConfig.load()
    if not cfg.elevenlabs_key:
        raise MediaError("ElevenLabs anahtarı yok (AI Merkezi → Ayarlar)")
    voice_id = voice if re.fullmatch(r"[A-Za-z0-9]{15,}", voice or "") else "21m00Tcm4TlvDq8nAiNo"
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        json.dumps({"text": text, "model_id": "eleven_multilingual_v2"}).encode(),
        {"xi-api-key": cfg.elevenlabs_key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    out = out.with_suffix(".mp3")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out.write_bytes(r.read())
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"ElevenLabs hatası: {exc}") from None
    return out


_tts_queue = None
_tts_thread = None


def stop_tts_worker(timeout: float = 2.0) -> None:
    global _tts_queue, _tts_thread
    if _tts_queue is not None and _tts_thread is not None and _tts_thread.is_alive():
        try:
            _tts_queue.put(None)
            _tts_thread.join(timeout)
        except Exception:
            pass
    _tts_queue = None
    _tts_thread = None


def _offline_tts(text: str, out: pathlib.Path, voice: str, rate: int) -> None:
    """pyttsx3 must live on one thread for the whole program (SAPI/COM on Windows, espeak callbacks on Linux),
    so all offline speech goes through a single worker thread that owns the engine."""
    global _tts_queue, _tts_thread
    import queue
    import threading
    if _tts_queue is None:
        _tts_queue = queue.Queue()

        def worker():
            if sys.platform == "win32":
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except ImportError:
                    pass
            engine = None
            try:
                while True:
                    item = _tts_queue.get()
                    if item is None:
                        break
                    job, reply = item
                    engine = None
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        base_rate = engine.getProperty("rate")
                        t, o, v, r = job
                        for vo in engine.getProperty("voices"):
                            info = f"{vo.name} {vo.id} {' '.join(str(x) for x in (getattr(vo, 'languages', []) or []))}"
                            if v and (v.lower() in info.lower() or v.lower().startswith("tr") and "turk" in info.lower()):
                                engine.setProperty("voice", vo.id)
                                break
                        engine.setProperty("rate", int(base_rate * (1 + r / 100)))
                        engine.save_to_file(t, str(o))
                        engine.runAndWait()
                        reply.put(None)
                    except Exception as exc:  # noqa: BLE001
                        reply.put(exc)
                    finally:
                        if engine is not None:
                            try:
                                engine.stop()
                            except Exception:
                                pass
                            del engine
            finally:
                if sys.platform == "win32":
                    try:
                        import pythoncom
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass
        _tts_thread = threading.Thread(target=worker, daemon=True, name="offline-tts")
        _tts_thread.start()
    reply = queue.Queue()
    _tts_queue.put(((text, out, voice, rate), reply))
    err = reply.get(timeout=300)
    if isinstance(err, ImportError):
        raise MediaError("pip install pyttsx3")
    if err is not None:
        raise MediaError(f"Bilgisayar sesi hatası: {err}")


# --- video ---------------------------------------------------------------------------------------------

def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise MediaError("ffmpeg bulunamadı: pip install imageio-ffmpeg") from None


def audio_duration(path: str | pathlib.Path) -> float:
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                       errors="replace", **_no_window())
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        raise MediaError(f"Ses dosyası okunamadı: {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _no_window() -> dict:
    return {"creationflags": 0x08000000} if sys.platform == "win32" else {}   # no console flash on Windows


def _font(size: int):
    from PIL import ImageFont
    for name in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if pathlib.Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size)


def _text_layer(text: str, w: int, h: int, big: bool) -> np.ndarray:
    """RGBA overlay with the caption: a bottom subtitle, or a large centred title when there are no images."""
    from PIL import Image, ImageDraw
    size = int(min(w, h) * (0.075 if big else 0.045))
    font = _font(size)
    chars = max(12, int(w * (0.8 if big else 0.86) / (size * 0.55)))
    lines = textwrap.wrap(text, chars)[: 6 if big else 3]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lh = int(size * 1.25)
    block_h = lh * len(lines)
    y = (h - block_h) // 2 if big else int(h * 0.9) - block_h
    if not big and lines:
        widest = max(d.textlength(t, font=font) for t in lines)
        pad = size // 2
        d.rounded_rectangle([(w - widest) / 2 - pad, y - pad / 2, (w + widest) / 2 + pad, y + block_h + pad / 2],
                            radius=size // 3, fill=(0, 0, 0, 150))
    for t in lines:
        tw = d.textlength(t, font=font)
        d.text(((w - tw) / 2, y), t, font=font, fill=(255, 255, 255, 255),
               stroke_width=max(2, size // 18) if big else 0, stroke_fill=(0, 0, 0, 200))
        y += lh
    return np.array(layer)[:, :, [2, 1, 0, 3]]     # RGBA -> BGRA


def _background(i: int, w: int, h: int) -> np.ndarray:
    palettes = [((48, 26, 92), (140, 60, 20)), ((20, 70, 40), (90, 30, 120)), ((80, 20, 20), (20, 60, 120)),
                ((30, 30, 30), (120, 90, 20))]
    a, b = (np.array(c, np.float32) for c in palettes[i % len(palettes)])
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return np.broadcast_to(a * (1 - t) + b * t, (h, w, 3)).astype(np.uint8)


def _cover(img: np.ndarray, w: int, h: int, zoom: float) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih) * zoom
    r = cv2.resize(img, (int(iw * s) + 1, int(ih * s) + 1), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def segments_for(script: str, duration: float, audio: pathlib.Path | None) -> list[tuple[float, float, str]]:
    """(start, end, sentence) list: exact timings from the voice engine when available, else by text length."""
    if audio is not None and _timings_path(audio).exists():
        marks = json.loads(_timings_path(audio).read_text(encoding="utf-8"))
        if marks:
            segs = [(m["start"], marks[i + 1]["start"] if i + 1 < len(marks) else duration, m["text"])
                    for i, m in enumerate(marks)]
            return [(0.0, segs[0][1], segs[0][2])] + segs[1:]
    sents = sentences(script) or [script.strip() or " "]
    total = sum(len(s) for s in sents) or 1
    out, t = [], 0.0
    for s in sents:
        d = duration * len(s) / total
        out.append((t, t + d, s))
        t += d
    return out


def make_video(script: str, out: str | pathlib.Path, audio: str | pathlib.Path | None = None,
               images: list | None = None, size: tuple[int, int] = (1920, 1080), subtitles: bool = True,
               progress=None) -> pathlib.Path:
    """Assemble an MP4: images (or coloured title cards) with a slow zoom, captions, and the voice-over."""
    w, h = size
    w, h = w - w % 2, h - h % 2
    out = pathlib.Path(out).with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    audio = pathlib.Path(audio) if audio else None
    if audio is not None and not audio.exists():
        raise MediaError(f"Ses dosyası yok: {audio}")
    duration = audio_duration(audio) if audio else max(3.0, len(script) / 14)
    segs = segments_for(script, duration, audio)
    pics = []
    for im in images or []:
        im = _imread(im) if isinstance(im, (str, pathlib.Path)) else im
        if im is not None:
            pics.append(im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **_no_window())
    total = int(duration * FPS) + 1
    try:
        frame_i = 0
        for si, (t0, t1, text) in enumerate(segs):
            n = max(1, int(round(t1 * FPS)) - int(round(t0 * FPS)))
            if si == len(segs) - 1:
                n = max(1, total - frame_i)
            base = pics[si * len(pics) // len(segs)] if pics else None
            overlay = _text_layer(text, w, h, big=not pics) if (subtitles or not pics) else None
            if overlay is not None:          # blend only the rows that carry text: much faster
                rows = np.where(overlay[:, :, 3].any(axis=1))[0]
                y0, y1 = (rows[0], rows[-1] + 1) if len(rows) else (0, 0)
                ov = overlay[y0:y1]
                alpha = ov[:, :, 3:4].astype(np.float32) / 255
                ov_rgb = ov[:, :, :3].astype(np.float32) * alpha
            for k in range(n):
                zoom = 1.0 + 0.08 * k / n
                frame = _cover(base, w, h, zoom) if base is not None else _cover(_background(si, w, h), w, h, zoom)
                if overlay is not None and y1 > y0:
                    frame = frame.copy()
                    frame[y0:y1] = (frame[y0:y1] * (1 - alpha) + ov_rgb).astype(np.uint8)
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                frame_i += 1
            if progress:
                progress(min(1.0, frame_i / total))
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise MediaError(f"Video oluşturulamadı: {err[-400:]}")
    except BrokenPipeError:
        proc.wait()
        raise MediaError(f"ffmpeg durdu: {proc.stderr.read().decode(errors='replace')[-400:]}") from None
    return out


# --- YouTube -------------------------------------------------------------------------------------------

YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_test_youtube = None     # tests inject a fake service here


def youtube_service(client_secret: str):
    if _test_youtube is not None:
        return _test_youtube
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise MediaError("pip install google-api-python-client google-auth-oauthlib") from None
    token = home() / "youtube_token.json"
    try:
        creds = Credentials.from_authorized_user_file(str(token), YT_SCOPES) if token.exists() else None
    except (ValueError, OSError):             # damaged token file: sign in again
        creds = None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token.write_text(creds.to_json(), encoding="utf-8")
        except Exception:  # noqa: BLE001 - RefreshError: revoked, or expired after 7 days in Google "testing" mode
            token.unlink(missing_ok=True)
            creds = None
    if not creds or not creds.valid:
        if not client_secret or not pathlib.Path(client_secret).exists():
            raise MediaError("YouTube için Google 'client_secret.json' dosyası gerekli (AI Merkezi → Ayarlar)")
        creds = InstalledAppFlow.from_client_secrets_file(client_secret, YT_SCOPES).run_local_server(port=0)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=creds)


def youtube_upload(video: str, title: str, description: str = "", tags: list[str] | None = None,
                   privacy: str = "private", client_secret: str = "", category: str = "22") -> str:
    """Upload and return the video URL. First run opens the browser once for Google sign-in."""
    if not pathlib.Path(video).exists():
        raise MediaError(f"Video dosyası yok: {video}")
    svc = youtube_service(client_secret)
    try:
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(video, chunksize=-1, resumable=True, mimetype="video/mp4")
    except ImportError:
        media = video
    body = {"snippet": {"title": title[:100] or "VisionFlow video", "description": description[:4900],
                        "tags": tags or [], "categoryId": category},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    req = svc.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    return f"https://youtu.be/{resp['id']}"
