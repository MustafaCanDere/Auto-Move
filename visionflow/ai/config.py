"""AI centre settings, shared by the editor and headless runs (a JSON file in the user's VisionFlow folder)."""
from __future__ import annotations

import json
import os
import pathlib
import sys
from dataclasses import asdict, dataclass, field

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_PERSONA = ("Sen kullanıcının kişisel yapay zeka asistanısın. Türkçe, kısa, net ve samimi cevap verirsin. "
                   "Bilmediğin bir şeyi uydurmazsın; hafızadan gelen bilgileri önceliklendirirsin.")


def home() -> pathlib.Path:
    """Per-user VisionFlow folder (settings, memory). VISIONFLOW_HOME overrides it (tests, portable installs)."""
    if os.environ.get("VISIONFLOW_HOME"):
        return pathlib.Path(os.environ["VISIONFLOW_HOME"])
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return pathlib.Path(os.environ["APPDATA"]) / "VisionFlow"
    return pathlib.Path.home() / ".visionflow"


@dataclass
class AIConfig:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    persona: str = DEFAULT_PERSONA
    memory_path: str = ""          # empty = <home>/hafiza.db
    # e-mail account used by the e-mail blocks when their own fields are empty
    email_user: str = ""
    email_password: str = ""
    imap_host: str = "imap.gmail.com"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    # voice / video / upload
    elevenlabs_key: str = ""
    youtube_client_secret: str = ""
    # Telegram bot (Telegram Mesaj Al / Gönder blokları)
    telegram_token: str = ""
    telegram_allowed: str = ""      # chat IDs allowed to talk to the bot, comma separated; empty = nobody
    # spending guard: Claude calls stop for the day once the estimate passes this (0 = no limit)
    daily_limit_usd: float = 5.0
    extra: dict = field(default_factory=dict)

    @staticmethod
    def path() -> pathlib.Path:
        return home() / "ai_ayarlar.json"

    @classmethod
    def load(cls) -> "AIConfig":
        try:
            data = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self) -> None:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        try:
            os.chmod(p, 0o600)        # holds keys and passwords
        except OSError:
            pass

    def memory_file(self) -> pathlib.Path:
        return pathlib.Path(self.memory_path) if self.memory_path else home() / "hafiza.db"

    def key(self) -> str | None:
        return self.api_key or os.environ.get("ANTHROPIC_API_KEY") or None
