"""Compose the aligned comparison video."""

import contextlib
import os
import shutil
import subprocess

import cv2
import numpy as np

from . import align, metrics, pose as pose_mod, video
from .textdraw import draw_text, text_width
from .timeutil import format_time

# BGR, ordered so the reference clip is always the first colour.
PALETTE = [
    (90, 220, 120),    # green  - reference
    (80, 140, 255),    # orange - first comparison
    (230, 160, 90),    # blue
    (200, 120, 230),   # violet
]
BACKGROUND = (24, 24, 28)
PANEL_BG = (16, 16, 18)
MUTED = (170, 170, 178)

HEADER = 46
FOOTER = 88


class Panel:
    """One climber's view: decoder, pose track and time mapping."""

    def __init__(self, clip, track, series, warp, color, project):
        self.clip = clip
        self.track = track
        self.series = series
        self.warp = warp
        self.color = color
        self.reader = video.SequentialReader(project.abspath(clip.path))
        self.aspect = (clip.size[0] / clip.size[1]) if clip.size[1] else 1.0

    def close(self):
        self.reader.close()

    def source_time(self, t_ref):
        return self.warp.to_target(t_ref)

    def frame_at(self, t_ref):
        t = self.source_time(t_ref)
        index = int(round(t * self.clip.fps))
        frame = self.reader.frame_at(index)
        if frame is None:
            return None, index
        if self.clip.flip:
            frame = cv2.flip(frame, 1)
        return frame, index


def _letterbox(frame, width, height):
    """Fit a frame into a fixed box, returning the box plus its transform."""
    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    if frame is None:
        return canvas, (0.0, 0.0, 0.0)
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ox, oy = (width - new_w) // 2, (height - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas, (ox, oy, scale * w, scale * h)


def _to_pixels(landmarks, box):
    ox, oy, w, h = box[0], box[1], box[2], box[3]
    pts = np.empty((landmarks.shape[0], 2), dtype=np.int32)
    pts[:, 0] = np.round(ox + landmarks[:, 0] * w)
    pts[:, 1] = np.round(oy + landmarks[:, 1] * h)
    return pts


def _draw_skeleton(canvas, landmarks, box, color, thickness=3, alpha=1.0):
    if landmarks is None:
        return
    pts = _to_pixels(landmarks, box)
    layer = canvas if alpha >= 1.0 else canvas.copy()
    for a, b in pose_mod.SKELETON:
        if landmarks[a, 3] < 0.2 or landmarks[b, 3] < 0.2:
            continue
        cv2.line(layer, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    for i in (pose_mod.L_WRIST, pose_mod.R_WRIST, pose_mod.L_FOOT, pose_mod.R_FOOT):
        if landmarks[i, 3] >= 0.2:
            cv2.circle(layer, tuple(pts[i]), thickness + 3, color, -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0, canvas)


def _draw_trail(canvas, panel, t_source, box, seconds=2.0):
    series = panel.series
    end = int(round(t_source * panel.clip.fps))
    start = max(0, end - int(round(seconds * panel.clip.fps)))
    if end - start < 2:
        return
    com = series.com[start:end + 1]
    xs = com[:, 0] / max(series.aspect, 1e-6)
    ys = com[:, 1]
    points = np.stack([xs, ys], axis=1)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return
    pts = _to_pixels(np.pad(points, ((0, 0), (0, 2))), box)
    for i in range(1, len(pts)):
        weight = i / len(pts)
        color = tuple(int(c * (0.35 + 0.65 * weight)) for c in panel.color)
        cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), color, 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[-1]), 5, panel.color, -1, cv2.LINE_AA)


def _overlay_panel(canvas, box, ref_panel, ref_lm, other_panels, other_lms):
    """Superimpose every climber's skeleton on the reference frame.

    Each skeleton is translated so the hips coincide and scaled so the torso
    lengths match, which removes camera distance and framing differences.
    """
    if ref_lm is None:
        return
    _draw_skeleton(canvas, ref_lm, box, ref_panel.color, thickness=3)
    ref_hip = np.nanmean(ref_lm[[pose_mod.L_HIP, pose_mod.R_HIP], :2], axis=0)
    ref_scale = ref_panel.series.scale
    for panel, lm in zip(other_panels, other_lms):
        if lm is None:
            continue
        hip = np.nanmean(lm[[pose_mod.L_HIP, pose_mod.R_HIP], :2], axis=0)
        ratio = ref_scale / max(panel.series.scale, 1e-6)
        moved = lm.copy()
        moved[:, 0] = (lm[:, 0] - hip[0]) * ratio * (panel.aspect / max(ref_panel.aspect, 1e-6)) + ref_hip[0]
        moved[:, 1] = (lm[:, 1] - hip[1]) * ratio + ref_hip[1]
        _draw_skeleton(canvas, moved, box, panel.color, thickness=3, alpha=0.75)


def _draw_header(canvas, x, width, panel, t_source, is_ref):
    cv2.rectangle(canvas, (x, 0), (x + width, HEADER), (34, 34, 40), -1)
    cv2.line(canvas, (x, HEADER - 1), (x + width, HEADER - 1), panel.color, 3)
    tag = '参考' if is_ref else '对比'
    draw_text(canvas, f'{panel.clip.display_name}', (x + 14, HEADER // 2),
              size=24, color=panel.color, anchor='lm')
    draw_text(canvas, f'{tag}  {format_time(t_source)}', (x + width - 14, HEADER // 2),
              size=20, color=MUTED, anchor='rm')


def _draw_footer(canvas, y, width, height, segment, panels, times, ref_panel):
    cv2.rectangle(canvas, (0, y), (width, y + height), (34, 34, 40), -1)
    draw_text(canvas, f'段落 {segment}', (14, y + 10), size=20, color=(235, 235, 240))

    x = 240
    for panel, t in zip(panels, times):
        if x > width - 150:
            break
        speed = _speed_at(panel, t)
        state = '静止' if speed is not None and speed < metrics.STATIC_SPEED else '移动'
        draw_text(canvas, f'{panel.clip.display_name}: {state}',
                  (x, y + 10), size=18, color=panel.color)
        cv2.rectangle(canvas, (x, y + 36), (x + 120, y + 43), (60, 60, 68), -1)
        if speed is not None:
            bar = int(min(1.0, speed / 1.5) * 120)
            cv2.rectangle(canvas, (x, y + 36), (x + bar, y + 43), panel.color, -1)
        x += 230


def _speed_at(panel, t_source):
    index = int(round(t_source * panel.clip.fps))
    if 0 <= index < panel.series.speed.shape[0]:
        value = panel.series.speed[index]
        if np.isfinite(value):
            return float(value)
    return None


def _draw_progress(canvas, y, width, warp, t_ref):
    span = warp.ref[-1] - warp.ref[0]
    if span <= 0:
        return
    ratio = float(np.clip((t_ref - warp.ref[0]) / span, 0.0, 1.0))
    left, right = 14, width - 14
    cv2.line(canvas, (left, y), (right, y), (70, 70, 78), 4)
    cv2.line(canvas, (left, y), (int(left + (right - left) * ratio), y), (235, 235, 240), 4)
    for name, t in zip(warp.names, warp.ref):
        mark = (t - warp.ref[0]) / span
        mx = int(left + (right - left) * mark)
        cv2.circle(canvas, (mx, y), 6, (235, 235, 240), -1, cv2.LINE_AA)
        half = text_width(name, 16) // 2 + 4
        label_x = int(np.clip(mx, left + half, right - half))
        draw_text(canvas, name, (label_x, y - 16), size=16, color=MUTED, anchor='mm')


def render(project, ref_label, target_labels, tracks, series_map, warps, out_path,
           layout='side', height=720, fps=None, pad=0.5, trail=2.0, speed=1.0,
           codec='auto', progress=None):
    """Write the aligned comparison video and return a summary dict."""
    ref_clip = project.clip(ref_label)
    panels = []
    ref_panel = Panel(ref_clip, tracks[ref_label], series_map[ref_label],
                      align.identity_warp([t for _, t in ref_clip.keyframes] or [0.0, 1.0]),
                      PALETTE[0], project)
    panels.append(ref_panel)
    for i, label in enumerate(target_labels):
        panels.append(Panel(project.clip(label), tracks[label], series_map[label],
                            warps[label], PALETTE[(i + 1) % len(PALETTE)], project))

    master_warp = warps[target_labels[0]]
    out_fps = float(fps or ref_clip.fps)
    step = float(speed) / out_fps
    start = master_warp.ref[0] - pad
    end = master_warp.ref[-1] + pad
    frame_total = max(1, int(round((end - start) / step)))

    panel_h = height - HEADER - FOOTER
    aspect = max(p.aspect for p in panels)
    panel_w = max(160, int(round(panel_h * aspect)))
    columns = len(panels) + (1 if layout in ('overlay', 'both') else 0)
    if layout == 'overlay':
        columns = 1
    width = panel_w * columns

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        writer, fourcc = _open_writer(out_path, out_fps, width, height, codec)
    except RuntimeError:
        for panel in panels:
            panel.close()
        raise

    try:
        for i in range(frame_total + 1):
            t_ref = start + i * step
            canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

            frames, landmarks, times = [], [], []
            for panel in panels:
                frame, index = panel.frame_at(t_ref)
                frames.append(frame)
                landmarks.append(panel.track.frame(index))
                times.append(panel.source_time(t_ref))

            column = 0
            if layout in ('side', 'both'):
                for panel, frame, lm, t in zip(panels, frames, landmarks, times):
                    x = column * panel_w
                    box_frame, box = _letterbox(frame, panel_w, panel_h)
                    canvas[HEADER:HEADER + panel_h, x:x + panel_w] = box_frame
                    view = canvas[HEADER:HEADER + panel_h, x:x + panel_w]
                    _draw_trail(view, panel, t, box, seconds=trail)
                    _draw_skeleton(view, lm, box, panel.color)
                    _draw_header(canvas, x, panel_w, panel, t, panel is ref_panel)
                    column += 1

            if layout in ('overlay', 'both'):
                x = column * panel_w
                box_frame, box = _letterbox(frames[0], panel_w, panel_h)
                dim = (box_frame * 0.55).astype(np.uint8)
                canvas[HEADER:HEADER + panel_h, x:x + panel_w] = dim
                view = canvas[HEADER:HEADER + panel_h, x:x + panel_w]
                _overlay_panel(view, box, ref_panel, landmarks[0], panels[1:], landmarks[1:])
                cv2.rectangle(canvas, (x, 0), (x + panel_w, HEADER), (34, 34, 40), -1)
                draw_text(canvas, '骨骼叠加', (x + 14, HEADER // 2), size=24,
                          color=(235, 235, 240), anchor='lm')

            segment, _ = master_warp.segment_at(t_ref)
            _draw_footer(canvas, height - FOOTER, width, FOOTER, segment, panels, times, ref_panel)
            _draw_progress(canvas, height - 14, width, master_warp, t_ref)
            writer.write(canvas)

            if progress and i and i % 60 == 0:
                progress(f'  合成 {i}/{frame_total} 帧')
    finally:
        writer.release()
        for panel in panels:
            panel.close()

    browser_ready = fourcc == 'avc1' or _reencode_h264(out_path, progress)
    return {
        'path': out_path,
        'frames': frame_total + 1,
        'fps': out_fps,
        'size': (width, height),
        'duration': (frame_total + 1) / out_fps,
        'codec': fourcc,
        'browser_ready': browser_ready,
    }


@contextlib.contextmanager
def _quiet_stderr():
    """Hide libav's codec-probing chatter."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def _open_writer(path, fps, width, height, codec='auto'):
    """Prefer H.264 (plays everywhere) and fall back to MPEG-4 Part 2."""
    candidates = ['avc1', 'mp4v'] if codec == 'auto' else [codec]
    for tag in candidates:
        with _quiet_stderr():
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag),
                                     fps, (width, height))
            opened = writer.isOpened()
        if opened:
            return writer, tag
        writer.release()
    raise RuntimeError(
        f'无法创建输出视频 {path}（尝试过 {", ".join(candidates)}）'
    )


def _reencode_h264(path, progress=None):
    """Re-wrap an mp4v file as H.264 when ffmpeg is available."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return False
    tmp = path + '.h264.mp4'
    command = [ffmpeg, '-y', '-loglevel', 'error', '-i', path,
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, path)
    if progress:
        progress('  已用 ffmpeg 转成 H.264，浏览器可直接播放')
    return True
