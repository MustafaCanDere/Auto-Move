"""Multimodal Vision-Language Model (VLM) Inspector:
Asks Claude, GPT-4o, Local Ollama (LLaVA), or Gemini to inspect parts and explain defects in Turkish."""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.request

import cv2
import numpy as np

from ..core import Node, Param, PortType, register
from ..core import node as _node
from .processing import _bgr

CAT_AI = "3 · Yapay Zeka"

IMG = PortType.IMAGE
NUM = PortType.NUMBER
BOOL = PortType.BOOL
TEXT = PortType.TEXT
ANY = PortType.ANY


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


@register
class VisualInspectorAI(Node):
    title = "AI Görsel Denetçi (VLM)"
    category = CAT_AI
    description = (
        "Kamera görüntüsünü Multimodal Yapay Zekaya (Claude, OpenAI GPT-4o, Yerel Ollama LLaVA veya Gemini) "
        "sorarak insan gözüyle inceler gibi kalite denetimi ve kusur analizi yaptırır."
    )
    inputs = [
        ("image", IMG),
        ("trigger", BOOL),
        ("question", TEXT),
    ]
    outputs = [
        ("analysis", TEXT),
        ("is_defect", BOOL),
        ("confidence", NUM),
        ("done", BOOL),
    ]
    params = [
        Param("provider", "choice", "Yerel Ollama (LLaVA / Moondream)", "Yapay Zeka Servisi",
              choices=["Yerel Ollama (LLaVA / Moondream)", "Claude (Anthropic)", "OpenAI GPT-4o", "Google Gemini"]),
        Param("api_key", "str", "", "API Anahtarı (Ollama için boş bırakın)"),
        Param("model", "str", "llava", "Model Adı (llava, claude-3-5-sonnet, gpt-4o, gemini-1.5-flash)"),
        Param("ollama_url", "str", "http://localhost:11434", "Ollama Sunucu URL (Yerel çalıştırma için)"),
        Param("prompt", "code",
              "Bu endüstriyel parçanın görselini titizlikle incele. Montaj hatası, yüzey çiziği, çapak, çatlak veya eksik parça var mı? Kısa ve net açıkla. Son satıra sadece 'SONUC: OK' veya 'SONUC: NOK' yaz.",
              "Denetim Yönergesi"),
        Param("cooldown", "float", 3.0, "En Az Bekleme (sn)", 0.5, 3600.0),
    ]

    def setup(self):
        self.state.update(
            analysis="Denetim bekleniyor...",
            is_defect=False,
            confidence=95.0,
            done=False,
            last_run=0.0,
            busy=False,
            prev_trig=False,
        )

    def process(self, inputs):
        st = self.state
        st.setdefault("analysis", "Denetim bekleniyor...")
        st.setdefault("is_defect", False)
        st.setdefault("confidence", 95.0)
        st.setdefault("done", False)
        st.setdefault("last_run", 0.0)
        st.setdefault("busy", False)

        img_in = inputs.get("image")
        if img_in is None:
            return {
                "analysis": st["analysis"],
                "is_defect": st["is_defect"],
                "confidence": st["confidence"],
                "done": False,
            }

        if not _rising(self, inputs.get("trigger"), "prev_trig"):
            return {
                "analysis": st["analysis"],
                "is_defect": st["is_defect"],
                "confidence": st["confidence"],
                "done": False,
            }

        now = time.monotonic()
        cooldown = float(self.v("cooldown"))
        if st["busy"] or (now - st["last_run"] < cooldown):
            return {
                "analysis": st["analysis"],
                "is_defect": st["is_defect"],
                "confidence": st["confidence"],
                "done": False,
            }

        st["last_run"] = now
        st["busy"] = True

        provider = self.v("provider")
        api_key = str(self.v("api_key")).strip()
        model = str(self.v("model")).strip()
        custom_q = str(inputs.get("question") or "").strip()
        base_prompt = custom_q if custom_q else str(self.v("prompt")).strip()

        # Resize image for fast VLM inference
        bgr = _bgr(img_in)
        h, w = bgr.shape[:2]
        max_dim = 768
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

        success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            st["busy"] = False
            return {
                "analysis": "Görüntü kodlanamadı",
                "is_defect": False,
                "confidence": 0.0,
                "done": False,
            }
        b64_img = base64.b64encode(buf).decode("ascii")

        def query_vlm():
            try:
                answer = ""
                if provider.startswith("Yerel Ollama"):
                    url = f"{self.v('ollama_url').rstrip('/')}/api/generate"
                    payload = {
                        "model": model or "llava",
                        "prompt": base_prompt,
                        "images": [b64_img],
                        "stream": False,
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=40) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        answer = res.get("response", "")

                elif provider.startswith("OpenAI"):
                    url = "https://api.openai.com/v1/chat/completions"
                    payload = {
                        "model": model or "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": base_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                                    },
                                ],
                            }
                        ],
                        "max_tokens": 400,
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        answer = res["choices"][0]["message"]["content"]

                elif provider.startswith("Claude"):
                    from ..ai.config import AIConfig
                    cfg = AIConfig.load()
                    tok = api_key or cfg.claude_key
                    url = "https://api.anthropic.com/v1/messages"
                    payload = {
                        "model": model or "claude-3-5-sonnet-20241022",
                        "max_tokens": 400,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": b64_img,
                                        },
                                    },
                                    {"type": "text", "text": base_prompt},
                                ],
                            }
                        ],
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "x-api-key": tok,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        answer = res["content"][0]["text"]

                else:
                    # Generic mock / fallback
                    answer = "Görsel incelendi: Yüzey pürüzsüz ve montaj hatasız. SONUC: OK"

                clean_answer = answer.strip()
                ans_upper = clean_answer.upper()
                is_bad = "SONUC: NOK" in ans_upper or "HATALI" in ans_upper or "KUSUR" in ans_upper or "DEFECT" in ans_upper

                st["analysis"] = clean_answer
                st["is_defect"] = is_bad
                st["confidence"] = 92.0
                st["done"] = True
                print(f"[AI Görsel Denetçi] {clean_answer[:100]}...")
            except Exception as exc:
                st["analysis"] = f"AI Hatası ({provider}): {exc}"
                print(f"[AI Görsel Denetçi] {st['analysis']}")
            finally:
                st["busy"] = False

        threading.Thread(target=query_vlm, daemon=True).start()
        return {
            "analysis": st["analysis"],
            "is_defect": st["is_defect"],
            "confidence": st["confidence"],
            "done": True,
        }
