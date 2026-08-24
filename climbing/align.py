"""Keyframe-driven time alignment between clips of the same route."""

import bisect


class AlignError(RuntimeError):
    pass


class TimeWarp:
    """Piecewise-linear map from reference seconds to target seconds.

    Anchors are the shared keyframes. Between two anchors time is stretched
    linearly, so the same move lands on the same output frame even when one
    climber is twice as fast. Outside the anchor range the slope of the first
    or last segment is extended, which keeps the run-up and the top-out from
    jumping.
    """

    def __init__(self, ref_times, target_times, names=None):
        if len(ref_times) != len(target_times):
            raise AlignError('参考与对比片段的关键帧数量不一致')
        if len(ref_times) < 2:
            raise AlignError(
                '至少需要 2 个共同关键帧才能对齐（建议至少标记 start 和 top）'
            )
        _check_monotonic(ref_times, names, '参考片段')
        _check_monotonic(target_times, names, '对比片段')
        self.ref = [float(t) for t in ref_times]
        self.target = [float(t) for t in target_times]
        self.names = list(names) if names else [f'k{i}' for i in range(len(self.ref))]

    def __call__(self, t):
        return self.to_target(t)

    def to_target(self, t):
        t = float(t)
        ref, tgt = self.ref, self.target
        if t <= ref[0]:
            return tgt[0] + (t - ref[0]) * _slope(ref, tgt, 0)
        if t >= ref[-1]:
            last = len(ref) - 2
            return tgt[-1] + (t - ref[-1]) * _slope(ref, tgt, last)
        i = bisect.bisect_right(ref, t) - 1
        i = min(max(i, 0), len(ref) - 2)
        span = ref[i + 1] - ref[i]
        ratio = (t - ref[i]) / span if span else 0.0
        return tgt[i] + ratio * (tgt[i + 1] - tgt[i])

    def segment_at(self, t):
        """``(name, index)`` of the segment containing reference time ``t``."""
        if t < self.ref[0]:
            return f'→{self.names[0]}', -1
        if t >= self.ref[-1]:
            return f'{self.names[-1]}→', len(self.ref) - 1
        i = bisect.bisect_right(self.ref, t) - 1
        i = min(max(i, 0), len(self.ref) - 2)
        return f'{self.names[i]}→{self.names[i + 1]}', i

    def pace_at(self, t):
        """Local speed ratio: >1 means the target clip is slower here."""
        _, i = self.segment_at(t)
        i = min(max(i, 0), len(self.ref) - 2)
        return _slope(self.ref, self.target, i)

    @property
    def segments(self):
        """``[(name, ref_start, ref_end, target_start, target_end)]``."""
        out = []
        for i in range(len(self.ref) - 1):
            out.append((
                f'{self.names[i]}→{self.names[i + 1]}',
                self.ref[i], self.ref[i + 1],
                self.target[i], self.target[i + 1],
            ))
        return out


def _slope(ref, tgt, i):
    span = ref[i + 1] - ref[i]
    return (tgt[i + 1] - tgt[i]) / span if span else 1.0


def _check_monotonic(times, names, who):
    for i in range(len(times) - 1):
        if times[i + 1] <= times[i]:
            a = names[i] if names else i
            b = names[i + 1] if names else i + 1
            raise AlignError(
                f'{who}的关键帧顺序有问题: {a} ({times[i]:.2f}s) '
                f'不早于 {b} ({times[i + 1]:.2f}s)'
            )


def identity_warp(ref_times, names=None):
    return TimeWarp(ref_times, ref_times, names)


def build(project, ref_label, target_label, names=None):
    """Build the warp from a project's shared keyframes."""
    ref_clip = project.clip(ref_label)
    target_clip = project.clip(target_label)
    names = names or project.shared_keyframes([ref_label, target_label])
    if len(names) < 2:
        ref_marks = ', '.join(n for n, _ in ref_clip.keyframes) or '(无)'
        tgt_marks = ', '.join(n for n, _ in target_clip.keyframes) or '(无)'
        raise AlignError(
            f'{ref_label} 与 {target_label} 的共同关键帧不足 2 个，无法对齐。\n'
            f'  {ref_label} 已标记: {ref_marks}\n'
            f'  {target_label} 已标记: {tgt_marks}\n'
            f'  用 `mark` 给两条片段标上同名节点，例如 start / crux / top'
        )
    ref_map = ref_clip.keyframe_map
    tgt_map = target_clip.keyframe_map
    return TimeWarp([ref_map[n] for n in names], [tgt_map[n] for n in names], names)


def master_timeline(warp, fps, pad=0.5):
    """Output timestamps (in reference time) covering the aligned climb."""
    start = warp.ref[0] - pad
    end = warp.ref[-1] + pad
    step = 1.0 / float(fps)
    count = max(1, int(round((end - start) / step)))
    return [start + i * step for i in range(count + 1)]
