"""Text on images with Turkish letters. OpenCV's built-in fonts only know ASCII and draw ş, ğ, ı, İ, ü, ö, ç as "?",
so text with such letters is drawn with a real font through Pillow; plain ASCII still uses fast cv2.putText."""
from __future__ import annotations

import functools
import pathlib

import cv2
import numpy as np

FONTS = ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/TTF/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc")


@functools.lru_cache(maxsize=32)
def _font(px: int):
    from PIL import ImageFont
    for name in FONTS:
        if pathlib.Path(name).exists():
            try:
                return ImageFont.truetype(name, px)
            except OSError:
                continue
    try:
        return ImageFont.load_default(px)
    except TypeError:                        # very old Pillow: fixed-size bitmap font
        return ImageFont.load_default()


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color, thickness: int = 1) -> None:
    """Like cv2.putText(img, text, org, FONT_HERSHEY_SIMPLEX, scale, color, thickness, LINE_AA): ``org`` is the
    left end of the baseline. Draws in place."""
    text = str(text)
    if text.isascii():
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return
    try:
        from PIL import Image, ImageDraw
    except ImportError:                      # no Pillow: readable ASCII instead of question marks
        import unicodedata
        plain = unicodedata.normalize("NFKD", text.replace("ı", "i").replace("İ", "I")).encode("ascii", "ignore")
        cv2.putText(img, plain.decode(), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return
    font = _font(max(8, int(round(30 * scale))))
    x0, y0, x1, y1 = font.getbbox(text, anchor="ls", stroke_width=max(0, thickness - 1))
    h, w = img.shape[:2]
    left, top = max(0, org[0] + x0 - 1), max(0, org[1] + y0 - 1)
    right, bottom = min(w, org[0] + x1 + 1), min(h, org[1] + y1 + 1)
    if right <= left or bottom <= top:
        return
    patch = img[top:bottom, left:right]
    gray = patch.ndim == 2
    pil = Image.fromarray(np.ascontiguousarray(patch) if gray else cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if gray:
        fill = int(color[0] if isinstance(color, (tuple, list)) else color)
    else:
        b, g, r = (int(c) for c in color[:3])
        fill = (r, g, b)
    draw.text((org[0] - left, org[1] - top), text, font=font, fill=fill, anchor="ls",
              stroke_width=max(0, thickness - 1), stroke_fill=fill)
    out = np.asarray(pil)
    img[top:bottom, left:right] = out if gray else cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    """(width, height) the text takes with put_text, for drawing a label background."""
    text = str(text)
    if text.isascii():
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    try:
        x0, y0, x1, y1 = _font(max(8, int(round(30 * scale)))).getbbox(text, anchor="ls")
        return x1 - x0, -y0
    except Exception:  # noqa: BLE001
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
