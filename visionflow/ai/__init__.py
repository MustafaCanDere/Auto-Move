"""AI centre: Claude access, settings, and the local memory (knowledge base)."""
from .claude import AIError, ask, research  # noqa: F401
from .config import AIConfig, home  # noqa: F401
from .memory import Memory  # noqa: F401
