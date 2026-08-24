"""Movement metrics derived from a pose track.

Everything is normalised by the climber's torso length, so a phone held
closer to the wall does not change the numbers.
"""

import warnings

import numpy as np

from . import pose as pose_mod

# Rough segment-mass weighting used to approximate the centre of mass.
COM_WEIGHTS = [
    ((pose_mod.NOSE,), 0.08),
    ((pose_mod.L_SHOULDER, pose_mod.R_SHOULDER), 0.26),
    ((pose_mod.L_ELBOW, pose_mod.R_ELBOW), 0.05),
    ((pose_mod.L_WRIST, pose_mod.R_WRIST), 0.03),
    ((pose_mod.L_HIP, pose_mod.R_HIP), 0.34),
    ((pose_mod.L_KNEE, pose_mod.R_KNEE), 0.16),
    ((pose_mod.L_ANKLE, pose_mod.R_ANKLE), 0.08),
]

STATIC_SPEED = 0.15      # torso lengths / second
PAUSE_SECONDS = 0.6      # a static stretch this long counts as a pause
STRAIGHT_ARM = 0.85      # fraction of full reach that counts as "直臂"
MAX_GAP_SECONDS = 0.5    # longest pose dropout we interpolate across


class Series:
    """Per-frame metric arrays for one clip."""

    def __init__(self, track, aspect=1.0):
        self.track = track
        self.aspect = float(aspect)
        self.fps = track.fps
        self.frame_count = track.frame_count
        self.time = np.arange(track.frame_count, dtype=np.float64) / track.fps
        self._build(aspect)

    def _build(self, aspect):
        # Frames without a pose are all-NaN; nanmean over them is intentional.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self._build_unchecked(aspect)

    def _build_unchecked(self, aspect):
        data = self.track.data.astype(np.float64).copy()
        valid = self.track.valid
        # Normalised coordinates squash the image; undo it so distances and
        # angles are measured in real proportions.
        data[:, :, 0] *= aspect

        points = {}
        for name, idx in (
            ('shoulder', (pose_mod.L_SHOULDER, pose_mod.R_SHOULDER)),
            ('hip', (pose_mod.L_HIP, pose_mod.R_HIP)),
            ('wrist', (pose_mod.L_WRIST, pose_mod.R_WRIST)),
            ('ankle', (pose_mod.L_ANKLE, pose_mod.R_ANKLE)),
            ('foot', (pose_mod.L_FOOT, pose_mod.R_FOOT)),
            ('knee', (pose_mod.L_KNEE, pose_mod.R_KNEE)),
        ):
            points[name] = np.nanmean(data[:, idx, :2], axis=1)

        com = np.zeros((self.frame_count, 2))
        for idx, weight in COM_WEIGHTS:
            com += weight * np.nanmean(data[:, idx, :2], axis=1)
        com[~valid] = np.nan

        torso = np.linalg.norm(points['shoulder'] - points['hip'], axis=1)
        torso[~valid] = np.nan
        scale = float(np.nanmedian(torso)) if valid.any() else np.nan
        if not scale or not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        self.scale = scale

        self.com = _fill_gaps(com, self.fps)
        self.valid = valid
        self.torso = torso

        # Height above the first tracked frame, in torso lengths (up positive).
        base = _first_finite(self.com[:, 1])
        self.height = (base - self.com[:, 1]) / scale

        smooth = _smooth(self.com, max(3, int(round(0.2 * self.fps))))
        velocity = np.gradient(smooth, self.time, axis=0) / scale
        self.speed = np.linalg.norm(velocity, axis=1)
        self.velocity = velocity
        self.static = self.speed < STATIC_SPEED

        hip = _fill_gaps(points['hip'], self.fps)
        foot = _fill_gaps(points['foot'], self.fps)
        shoulder = _fill_gaps(points['shoulder'], self.fps)
        self.hip = hip
        self.hip_over_feet = (hip[:, 0] - foot[:, 0]) / scale

        torso_vec = shoulder - hip
        self.lean = np.degrees(np.arctan2(torso_vec[:, 0], -torso_vec[:, 1]))

        reach = np.linalg.norm(
            _fill_gaps(points['wrist'], self.fps) - shoulder, axis=1)
        self.arm_extension = _normalise_reach(reach)
        leg = np.linalg.norm(_fill_gaps(points['ankle'], self.fps) - hip, axis=1)
        self.leg_extension = _normalise_reach(leg)

    # ---- windowed aggregation -----------------------------------------
    def window(self, t_start, t_end):
        lo = max(0, int(np.floor(t_start * self.fps)))
        hi = min(self.frame_count, int(np.ceil(t_end * self.fps)) + 1)
        if hi <= lo:
            hi = min(self.frame_count, lo + 1)
        return slice(lo, hi)

    def summarize(self, t_start, t_end):
        sl = self.window(t_start, t_end)
        com = self.com[sl]
        speed = self.speed[sl]
        static = self.static[sl]
        duration = max(0.0, float(t_end - t_start))

        steps = np.linalg.norm(np.diff(com, axis=0), axis=1) / self.scale
        path = float(np.nansum(steps))
        rise = float(_last_finite(self.height[sl]) - _first_finite(self.height[sl]))
        finite_speed = speed[np.isfinite(speed)]
        tracked = self.valid[sl]

        return {
            'duration': duration,
            'tracked_ratio': float(tracked.mean()) if tracked.size else 0.0,
            'static_ratio': float(np.mean(static[np.isfinite(speed)])) if finite_speed.size else float('nan'),
            'pauses': _count_pauses(static, self.fps),
            'longest_pause': _longest_pause(static, self.fps),
            'mean_speed': float(np.nanmean(finite_speed)) if finite_speed.size else float('nan'),
            'peak_speed': float(np.nanmax(finite_speed)) if finite_speed.size else float('nan'),
            'rise': rise,
            'path_length': path,
            'efficiency': float(rise / path) if path > 1e-6 else float('nan'),
            'straight_arm_ratio': _ratio_above(self.arm_extension[sl], STRAIGHT_ARM),
            'mean_hip_over_feet': float(np.nanmean(np.abs(self.hip_over_feet[sl]))),
            'mean_lean': float(np.nanmean(np.abs(self.lean[sl]))),
        }

    def segments(self, keyframes):
        """Summaries for each consecutive keyframe pair."""
        out = []
        for (name_a, t_a), (name_b, t_b) in zip(keyframes, keyframes[1:]):
            summary = self.summarize(t_a, t_b)
            summary['name'] = f'{name_a}→{name_b}'
            summary['start'] = t_a
            summary['end'] = t_b
            out.append(summary)
        return out


def build(track, aspect=1.0):
    return Series(track, aspect=aspect)


# ---- helpers -----------------------------------------------------------
def _first_finite(arr):
    finite = arr[np.isfinite(arr)]
    return float(finite[0]) if finite.size else float('nan')


def _last_finite(arr):
    finite = arr[np.isfinite(arr)]
    return float(finite[-1]) if finite.size else float('nan')


def _fill_gaps(arr, fps):
    """Linearly interpolate short dropouts; leave long ones as NaN."""
    out = np.array(arr, dtype=np.float64, copy=True)
    limit = max(1, int(round(MAX_GAP_SECONDS * fps)))
    for col in range(out.shape[1]) if out.ndim == 2 else range(1):
        values = out[:, col] if out.ndim == 2 else out
        good = np.flatnonzero(np.isfinite(values))
        if good.size < 2:
            continue
        filled = values.copy()
        interpolated = np.interp(np.arange(values.size), good, values[good])
        for a, b in zip(good, good[1:]):
            if b - a > 1 and (b - a) <= limit + 1:
                filled[a + 1:b] = interpolated[a + 1:b]
        if out.ndim == 2:
            out[:, col] = filled
        else:
            out = filled
    return out


def _smooth(arr, window):
    if window < 2 or arr.shape[0] < window:
        return np.array(arr, dtype=np.float64, copy=True)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    out = np.array(arr, dtype=np.float64, copy=True)
    pad = window // 2
    for col in range(out.shape[1]):
        values = out[:, col]
        mask = np.isfinite(values)
        if mask.sum() < 2:
            continue
        filled = np.interp(np.arange(values.size), np.flatnonzero(mask), values[mask])
        padded = np.pad(filled, pad, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        smoothed[~mask] = np.nan
        out[:, col] = smoothed
    return out


def _normalise_reach(distances):
    """Express a limb distance as a fraction of that climber's full reach."""
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return np.full_like(distances, np.nan)
    full = float(np.percentile(finite, 95))
    if full <= 1e-6:
        return np.full_like(distances, np.nan)
    return np.clip(distances / full, 0.0, 1.2)


def _ratio_above(values, threshold):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    return float((finite >= threshold).mean())


def _runs(mask):
    """Yield ``(start, length)`` for each run of True."""
    if mask.size == 0:
        return
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[::2], edges[1::2]):
        yield int(start), int(end - start)


def _count_pauses(static, fps):
    minimum = max(1, int(round(PAUSE_SECONDS * fps)))
    return sum(1 for _, length in _runs(static) if length >= minimum)


def _longest_pause(static, fps):
    lengths = [length for _, length in _runs(static)]
    return float(max(lengths) / fps) if lengths else 0.0
