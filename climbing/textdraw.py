"""CJK-capable text drawing on OpenCV frames.

``cv2.putText`` cannot render Chinese, so labels go through Pillow when a
CJK font is available and fall back to ASCII-only OpenCV text otherwise.
"""

import os

import numpy as np

FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
]

_cache = {}


def find_font():
    override = os.environ.get('CLIMB_FONT')
    candidates = ([override] if override else []) + FONT_CANDIDATES
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _font(size):
    if size in _cache:
        return _cache[size]
    path = find_font()
    font = None
    if path:
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(path, size)
        except Exception:
            font = None
    _cache[size] = font
    return font


def _ascii_safe(text):
    return ''.join(ch if ord(ch) < 128 else '?' for ch in text)


def draw_text(frame, text, xy, size=22, color=(255, 255, 255), anchor='lt'):
    """Draw ``text`` on a BGR frame. ``anchor`` is Pillow-style (lt/mt/rt)."""
    font = _font(size)
    if font is None:
        import cv2
        scale = size / 30.0
        x, y = int(xy[0]), int(xy[1] + size)
        cv2.putText(frame, _ascii_safe(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, max(1, int(size / 14)), cv2.LINE_AA)
        return frame

    from PIL import Image, ImageDraw
    image = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    draw.text(tuple(int(v) for v in xy), text, font=font,
              fill=(int(color[2]), int(color[1]), int(color[0])), anchor=anchor)
    frame[:] = np.asarray(image)[:, :, ::-1]
    return frame


def text_width(text, size=22):
    font = _font(size)
    if font is None:
        return int(len(_ascii_safe(text)) * size * 0.55)
    return int(font.getlength(text))
