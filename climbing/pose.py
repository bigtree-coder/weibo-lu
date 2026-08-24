"""Pose extraction (MediaPipe) and the on-disk pose track cache."""

import gzip
import json
import os
import urllib.request

import numpy as np

from . import video

NUM_LANDMARKS = 33

# MediaPipe BlazePose landmark indices we actually reason about.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT),
]

# Landmarks that must be visible for a frame to count as a usable pose.
CORE = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

MODEL_URLS = {
    'lite': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
            'pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'full': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
            'pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'heavy': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
             'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
}


class PoseError(RuntimeError):
    pass


def model_dir():
    return os.environ.get(
        'CLIMB_MODEL_DIR',
        os.path.join(os.path.expanduser('~'), '.cache', 'climbing-pose'),
    )


def ensure_model(variant='lite', progress=None):
    """Download the PoseLandmarker task bundle on first use."""
    if variant not in MODEL_URLS:
        raise PoseError(f'未知模型: {variant}（可选 {", ".join(MODEL_URLS)}）')
    directory = model_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'pose_landmarker_{variant}.task')
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if progress:
        progress(f'首次运行，正在下载姿态模型 {variant} ...')
    tmp = path + '.part'
    try:
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
    except Exception as exc:
        raise PoseError(
            f'下载姿态模型失败: {exc}\n'
            f'可手动下载 {MODEL_URLS[variant]} 后放到 {path}'
        )
    os.replace(tmp, path)
    return path


class PoseTrack:
    """Per-frame landmarks for one clip, stored as ``(frames, 33, 4)``."""

    def __init__(self, label, fps, frame_count, width, height, data=None, model=''):
        self.label = label
        self.fps = float(fps)
        self.frame_count = int(frame_count)
        self.width = int(width)
        self.height = int(height)
        self.model = model
        if data is None:
            data = np.full((self.frame_count, NUM_LANDMARKS, 4), np.nan, dtype=np.float32)
        self.data = data

    @property
    def valid(self):
        """Boolean mask of frames with a usable pose."""
        return ~np.isnan(self.data[:, CORE, 0]).any(axis=1)

    @property
    def coverage(self):
        return float(self.valid.mean()) if self.frame_count else 0.0

    def frame(self, index):
        index = max(0, min(self.frame_count - 1, int(index)))
        row = self.data[index]
        return None if np.isnan(row[CORE, 0]).any() else row

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            'version': 1,
            'label': self.label,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'model': self.model,
            'landmarks': [
                None if np.isnan(row[CORE, 0]).any()
                else [[round(float(v), 4) for v in point] for point in row]
                for row in self.data
            ],
        }
        with gzip.open(path, 'wt', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            payload = json.load(fh)
        rows = payload['landmarks']
        data = np.full((len(rows), NUM_LANDMARKS, 4), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            if row is not None:
                data[i] = np.asarray(row, dtype=np.float32)
        return cls(
            payload['label'], payload['fps'], payload['frame_count'],
            payload['width'], payload['height'], data, payload.get('model', ''),
        )


def _select_pose(candidates, previous):
    """Pick the climber when several people are detected."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if previous is not None:
        prev_hip = np.nanmean(previous[[L_HIP, R_HIP], :2], axis=0)
        return min(
            candidates,
            key=lambda p: float(np.linalg.norm(np.nanmean(p[[L_HIP, R_HIP], :2], axis=0) - prev_hip)),
        )
    def area(p):
        xs, ys = p[:, 0], p[:, 1]
        return float((np.nanmax(xs) - np.nanmin(xs)) * (np.nanmax(ys) - np.nanmin(ys)))
    return max(candidates, key=area)


def _landmarks_to_array(landmarks, flip):
    arr = np.full((NUM_LANDMARKS, 4), np.nan, dtype=np.float32)
    for i, lm in enumerate(landmarks[:NUM_LANDMARKS]):
        visibility = getattr(lm, 'visibility', None)
        presence = getattr(lm, 'presence', None)
        score = 1.0
        if visibility is not None:
            score = float(visibility)
        elif presence is not None:
            score = float(presence)
        x = float(lm.x)
        if flip:
            x = 1.0 - x
        arr[i] = (x, float(lm.y), float(getattr(lm, 'z', 0.0) or 0.0), score)
    if flip:
        arr = _mirror_sides(arr)
    return arr


_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def _mirror_sides(arr):
    out = arr.copy()
    for a, b in _MIRROR_PAIRS:
        out[a], out[b] = arr[b].copy(), arr[a].copy()
    return out


def extract(clip_path, label, meta, flip=False, model='lite', min_confidence=0.5,
            num_poses=1, progress=None):
    """Run PoseLandmarker over the whole clip and return a :class:`PoseTrack`."""
    os.environ.setdefault('GLOG_minloglevel', '2')
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base
    except ImportError as exc:
        raise PoseError(
            f'需要 mediapipe 才能做姿态分析: {exc}\n'
            '安装: pip install -r requirements-climbing.txt'
        )

    model_path = ensure_model(model, progress=progress)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=max(1, int(num_poses)),
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    fps = float(meta['fps'])
    track = PoseTrack(label, fps, int(meta['frame_count']),
                      int(meta['width']), int(meta['height']), model=model)
    previous = None
    seen = 0

    try:
        landmarker_cm = mp_vision.PoseLandmarker.create_from_options(options)
    except OSError as exc:
        raise PoseError(
            f'加载 mediapipe 原生库失败: {exc}\n'
            '在没有图形界面的 Linux 上通常缺系统库，安装后重试:\n'
            '  sudo apt-get install -y libegl1 libgles2 libgl1'
        )

    with landmarker_cm as landmarker:
        for index, frame in video.iter_frames(clip_path):
            if index >= track.frame_count:
                break
            rgb = frame[:, :, ::-1]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(image, int(round(index * 1000.0 / fps)))
            poses = [
                _landmarks_to_array(p, flip)
                for p in (result.pose_landmarks or [])
                if len(p) >= NUM_LANDMARKS
            ]
            chosen = _select_pose(poses, previous)
            if chosen is not None:
                track.data[index] = chosen
                previous = chosen
                seen += 1
            if progress and index and index % 60 == 0:
                progress(f'  姿态提取 {index}/{track.frame_count} 帧，已识别 {seen}')

    if seen == 0:
        raise PoseError(
            f'{label}: 整段视频都没有识别到人体。'
            '常见原因是画面里人太小、逆光或被遮挡，可尝试裁剪画面或换 --model full'
        )
    return track
