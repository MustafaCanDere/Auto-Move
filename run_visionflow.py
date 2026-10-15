"""Entry point used by the PyInstaller build (absolute import, unlike python -m)."""
import sys

from visionflow.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
