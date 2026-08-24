"""Thin wrappers over cv2.VideoCapture used by the pose and render stages."""

import os

import cv2


class VideoError(RuntimeError):
    pass


def probe(path):
    """Return basic metadata for a video file."""
    if not os.path.exists(path):
        raise VideoError(f'视频文件不存在: {path}')
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f'无法打开视频: {path}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if fps <= 0 or fps > 480:
        fps = 30.0
    if frame_count <= 0:
        frame_count = _count_frames(path)
    return {
        'fps': round(float(fps), 6),
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': round(frame_count / fps, 3) if fps else 0.0,
    }


def _count_frames(path):
    cap = cv2.VideoCapture(path)
    count = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def iter_frames(path, stride=1):
    """Yield ``(frame_index, bgr_frame)`` sequentially."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f'无法打开视频: {path}')
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stride <= 1 or index % stride == 0:
                yield index, frame
            index += 1
    finally:
        cap.release()


class SequentialReader:
    """Decode forward-only to an arbitrary frame index.

    Random seeking is unreliable for long-GOP phone footage, and every
    consumer here walks a monotonically increasing timeline, so a forward
    scan with a one-frame cache is both faster and more accurate.
    """

    def __init__(self, path):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise VideoError(f'无法打开视频: {path}')
        self._index = -1
        self._frame = None
        self._exhausted = False

    def frame_at(self, index):
        """Return the frame at ``index`` (clamped, never rewinds)."""
        index = max(0, int(index))
        if index < self._index:
            return self._frame
        while self._index < index and not self._exhausted:
            ok, frame = self._cap.read()
            if not ok:
                self._exhausted = True
                break
            self._index += 1
            self._frame = frame
        return self._frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
