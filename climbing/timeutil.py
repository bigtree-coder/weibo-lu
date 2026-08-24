"""Time string parsing shared by the CLI and the project manifest."""

import re

_CLOCK = re.compile(r'^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$')
_MMSS = re.compile(r'^(\d{1,3}):(\d{1,2}(?:\.\d+)?)$')


def parse_time(value, fps=None):
    """Parse a timestamp into seconds.

    Accepts ``12.5`` (seconds), ``1:02.5`` (mm:ss), ``1:02:03.5`` (hh:mm:ss)
    and ``f:1520`` (frame index, requires fps).
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError('时间不能为空')

    if text.lower().startswith('f:'):
        if not fps:
            raise ValueError('使用帧号 (f:N) 需要知道视频帧率')
        frame = int(text[2:])
        if frame < 0:
            raise ValueError('帧号不能为负数')
        return frame / float(fps)

    m = _CLOCK.match(text)
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2))
        seconds = float(m.group(3))
        return hours * 3600 + minutes * 60 + seconds

    m = _MMSS.match(text)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))

    try:
        return float(text)
    except ValueError:
        raise ValueError(f'无法解析时间: {value!r}（支持 12.5 / 1:02.5 / 1:02:03 / f:1520）')


def format_time(seconds):
    """Render seconds as ``m:ss.s`` for HUDs and reports."""
    if seconds is None:
        return '--:--'
    sign = '-' if seconds < 0 else ''
    seconds = abs(float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f'{sign}{minutes}:{rest:04.1f}'


def format_delta(seconds):
    if seconds is None:
        return '--'
    return f'{seconds:+.1f}s'
