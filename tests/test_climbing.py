"""Unit tests for the climbing comparison pipeline (no mediapipe required)."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from climbing import align, metrics, pose, report
from climbing.pose import PoseTrack
from climbing.project import Project, ProjectError
from climbing.timeutil import format_time, parse_time


class TimeUtilTest(unittest.TestCase):
    def test_plain_seconds(self):
        self.assertAlmostEqual(parse_time('12.5'), 12.5)
        self.assertAlmostEqual(parse_time(7), 7.0)

    def test_clock_forms(self):
        self.assertAlmostEqual(parse_time('1:02.5'), 62.5)
        self.assertAlmostEqual(parse_time('1:02:03'), 3723.0)

    def test_frame_index(self):
        self.assertAlmostEqual(parse_time('f:60', fps=30), 2.0)
        with self.assertRaises(ValueError):
            parse_time('f:60')

    def test_rejects_garbage(self):
        for bad in ('', 'abc', '1:2:3:4'):
            with self.assertRaises(ValueError):
                parse_time(bad)

    def test_format(self):
        self.assertEqual(format_time(62.5), '1:02.5')
        self.assertEqual(format_time(None), '--:--')


class TimeWarpTest(unittest.TestCase):
    def setUp(self):
        # Reference climbs 1→8s; the target takes twice as long after mid.
        self.warp = align.TimeWarp([1.0, 3.0, 8.0], [2.0, 4.0, 14.0],
                                   ['start', 'mid', 'top'])

    def test_anchors_map_exactly(self):
        for ref, target in zip(self.warp.ref, self.warp.target):
            self.assertAlmostEqual(self.warp.to_target(ref), target)

    def test_linear_between_anchors(self):
        self.assertAlmostEqual(self.warp.to_target(2.0), 3.0)
        self.assertAlmostEqual(self.warp.to_target(5.5), 9.0)

    def test_extrapolates_with_edge_slope(self):
        self.assertAlmostEqual(self.warp.to_target(0.0), 1.0)
        self.assertAlmostEqual(self.warp.to_target(9.0), 16.0)

    def test_pace_reports_relative_speed(self):
        self.assertAlmostEqual(self.warp.pace_at(2.0), 1.0)
        self.assertAlmostEqual(self.warp.pace_at(5.0), 2.0)

    def test_segment_names(self):
        self.assertEqual(self.warp.segment_at(2.0)[0], 'start→mid')
        self.assertEqual(self.warp.segment_at(5.0)[0], 'mid→top')
        self.assertEqual(self.warp.segment_at(0.0)[0], '→start')
        self.assertEqual(self.warp.segment_at(9.0)[0], 'top→')

    def test_needs_two_anchors(self):
        with self.assertRaises(align.AlignError):
            align.TimeWarp([1.0], [2.0], ['start'])

    def test_rejects_out_of_order_anchors(self):
        with self.assertRaises(align.AlignError):
            align.TimeWarp([1.0, 3.0], [4.0, 2.0], ['start', 'top'])

    def test_identity(self):
        warp = align.identity_warp([1.0, 5.0])
        self.assertAlmostEqual(warp.to_target(3.3), 3.3)

    def test_master_timeline_covers_padding(self):
        times = align.master_timeline(self.warp, fps=10, pad=0.5)
        self.assertAlmostEqual(times[0], 0.5)
        self.assertAlmostEqual(times[-1], 8.5, places=6)


def synthetic_track(frames=120, fps=30.0, pause=None):
    """A stick figure rising at a constant rate, optionally pausing."""
    data = np.full((frames, pose.NUM_LANDMARKS, 4), np.nan, dtype=np.float32)
    height = 0.0
    for i in range(frames):
        moving = not (pause and pause[0] <= i < pause[1])
        if moving:
            height += 0.004
        y_hip = 0.8 - height
        body = {
            pose.NOSE: (0.5, y_hip - 0.30),
            pose.L_SHOULDER: (0.45, y_hip - 0.20),
            pose.R_SHOULDER: (0.55, y_hip - 0.20),
            pose.L_ELBOW: (0.40, y_hip - 0.10),
            pose.R_ELBOW: (0.60, y_hip - 0.10),
            pose.L_WRIST: (0.42, y_hip - 0.35),
            pose.R_WRIST: (0.58, y_hip - 0.35),
            pose.L_HIP: (0.47, y_hip),
            pose.R_HIP: (0.53, y_hip),
            pose.L_KNEE: (0.45, y_hip + 0.12),
            pose.R_KNEE: (0.55, y_hip + 0.12),
            pose.L_ANKLE: (0.46, y_hip + 0.24),
            pose.R_ANKLE: (0.54, y_hip + 0.24),
            pose.L_HEEL: (0.45, y_hip + 0.25),
            pose.R_HEEL: (0.55, y_hip + 0.25),
            pose.L_FOOT: (0.47, y_hip + 0.26),
            pose.R_FOOT: (0.53, y_hip + 0.26),
        }
        for index in range(pose.NUM_LANDMARKS):
            x, y = body.get(index, (0.5, y_hip))
            data[i, index] = (x, y, 0.0, 1.0)
    return PoseTrack('test', fps, frames, 1080, 1080, data)


class MetricsTest(unittest.TestCase):
    def test_rising_climber_gains_height(self):
        series = metrics.build(synthetic_track())
        summary = series.summarize(0.0, 4.0)
        self.assertGreater(summary['rise'], 0)
        self.assertGreater(summary['efficiency'], 0.9)
        self.assertLess(summary['static_ratio'], 0.2)

    def test_pause_is_detected(self):
        series = metrics.build(synthetic_track(frames=180, pause=(60, 150)))
        summary = series.summarize(0.0, 6.0)
        self.assertGreater(summary['static_ratio'], 0.3)
        self.assertGreaterEqual(summary['pauses'], 1)
        self.assertGreater(summary['longest_pause'], 2.0)

    def test_scale_is_camera_independent(self):
        near = synthetic_track()
        far = PoseTrack('far', near.fps, near.frame_count, 1080, 1080,
                        near.data.copy())
        far.data[:, :, :2] = 0.5 + (far.data[:, :, :2] - 0.5) * 0.5
        a = metrics.build(near).summarize(0.0, 4.0)
        b = metrics.build(far).summarize(0.0, 4.0)
        self.assertAlmostEqual(a['rise'], b['rise'], places=2)
        self.assertAlmostEqual(a['mean_speed'], b['mean_speed'], places=2)

    def test_dropouts_do_not_break_summary(self):
        track = synthetic_track()
        track.data[40:45] = np.nan
        summary = metrics.build(track).summarize(0.0, 4.0)
        self.assertLess(summary['tracked_ratio'], 1.0)
        self.assertTrue(np.isfinite(summary['mean_speed']))

    def test_segments_follow_keyframes(self):
        series = metrics.build(synthetic_track(frames=180))
        segments = series.segments([('start', 0.0), ('mid', 2.0), ('top', 5.0)])
        self.assertEqual([s['name'] for s in segments], ['start→mid', 'mid→top'])
        self.assertAlmostEqual(segments[0]['duration'], 2.0)
        self.assertAlmostEqual(segments[1]['duration'], 3.0)

    def test_run_helpers(self):
        mask = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
        self.assertEqual(list(metrics._runs(mask)), [(1, 2), (4, 4)])
        # At 4fps a 0.6s pause is >= 2 frames, so both runs count.
        self.assertEqual(metrics._count_pauses(mask, fps=4.0), 2)
        # At 30fps neither run is long enough to be a pause.
        self.assertEqual(metrics._count_pauses(mask, fps=30.0), 0)
        self.assertAlmostEqual(metrics._longest_pause(mask, fps=4.0), 1.0)

    def test_short_gaps_are_filled_long_ones_are_not(self):
        values = np.array([[0.0], [np.nan], [2.0]], dtype=float)
        filled = metrics._fill_gaps(values, fps=10.0)
        self.assertAlmostEqual(filled[1, 0], 1.0)

        long_gap = np.full((40, 1), np.nan)
        long_gap[0, 0], long_gap[-1, 0] = 0.0, 1.0
        filled = metrics._fill_gaps(long_gap, fps=10.0)
        self.assertTrue(np.isnan(filled[20, 0]))


class PoseTrackTest(unittest.TestCase):
    def test_round_trip(self):
        track = synthetic_track(frames=20)
        track.data[5] = np.nan
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, 'cache', 't.pose.json.gz')
            track.save(path)
            loaded = PoseTrack.load(path)
            self.assertEqual(loaded.frame_count, track.frame_count)
            self.assertIsNone(loaded.frame(5))
            self.assertIsNotNone(loaded.frame(6))
            self.assertAlmostEqual(loaded.coverage, 19 / 20)
        finally:
            shutil.rmtree(directory)


def make_video(path, frames=30, size=(160, 120), fps=30.0):
    import cv2
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class ProjectTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.video = os.path.join(self.dir, 'clip.mp4')
        make_video(self.video)
        self.project = Project.create(os.path.join(self.dir, 'proj'), '测试')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_sets_first_clip_as_reference(self):
        clip = self.project.add_clip('coach', self.video, display_name='教练')
        self.assertEqual(self.project.reference, 'coach')
        self.assertEqual(clip.display_name, '教练')
        self.assertEqual(clip.frame_count, 30)
        self.assertTrue(os.path.exists(self.project.abspath(clip.path)))

    def test_rejects_bad_labels_and_duplicates(self):
        self.project.add_clip('coach', self.video)
        with self.assertRaises(ProjectError):
            self.project.add_clip('coach', self.video)
        with self.assertRaises(ProjectError):
            self.project.add_clip('我的', self.video)

    def test_keyframes_stay_sorted(self):
        clip = self.project.add_clip('coach', self.video)
        clip.set_keyframe('top', 0.9)
        clip.set_keyframe('start', 0.1)
        clip.set_keyframe('start', 0.2)
        self.assertEqual([n for n, _ in clip.keyframes], ['start', 'top'])
        self.assertAlmostEqual(clip.keyframe_map['start'], 0.2)

    def test_shared_keyframes_and_warp(self):
        a = self.project.add_clip('coach', self.video)
        b = self.project.add_clip('me', self.video)
        a.set_keyframe('start', 0.1); a.set_keyframe('top', 0.9)
        b.set_keyframe('start', 0.2); b.set_keyframe('top', 0.8)
        b.set_keyframe('extra', 0.5)
        self.assertEqual(self.project.shared_keyframes(), ['start', 'top'])
        warp = align.build(self.project, 'coach', 'me')
        self.assertAlmostEqual(warp.to_target(0.1), 0.2)

    def test_warp_needs_shared_marks(self):
        a = self.project.add_clip('coach', self.video)
        self.project.add_clip('me', self.video)
        a.set_keyframe('start', 0.1)
        with self.assertRaises(align.AlignError):
            align.build(self.project, 'coach', 'me')

    def test_save_and_reload(self):
        clip = self.project.add_clip('coach', self.video, display_name='教练')
        clip.set_keyframe('start', 0.3)
        self.project.save()
        reloaded = Project.load(self.project.root)
        self.assertEqual(reloaded.clip('coach').display_name, '教练')
        self.assertAlmostEqual(reloaded.clip('coach').keyframe_map['start'], 0.3)

    def test_discover_walks_up(self):
        nested = os.path.join(self.project.root, 'out')
        found = Project.discover(nested)
        self.assertEqual(found.root, self.project.root)


class ReportTest(unittest.TestCase):
    def _totals(self, duration, static, pauses, efficiency, arm=0.5, hip=0.2):
        return {
            'duration': duration, 'static_ratio': static, 'pauses': pauses,
            'longest_pause': 1.0, 'mean_speed': 1.0, 'peak_speed': 2.0,
            'rise': 5.0, 'path_length': 6.0, 'efficiency': efficiency,
            'straight_arm_ratio': arm, 'mean_hip_over_feet': hip,
            'mean_lean': 5.0, 'tracked_ratio': 0.9,
        }

    def test_findings_call_out_the_slow_segment(self):
        ref_segments = [{'name': 'start→mid', 'duration': 3.0, 'static_ratio': 0.1,
                         'efficiency': 0.9}]
        target_segments = [{'name': 'start→mid', 'duration': 8.0, 'static_ratio': 0.6,
                            'efficiency': 0.6}]
        notes = report.build_findings(
            '教练', '我',
            self._totals(10, 0.1, 1, 0.9), self._totals(18, 0.5, 5, 0.6, arm=0.2, hip=0.5),
            ref_segments, target_segments)
        joined = ' '.join(notes)
        self.assertIn('start→mid', joined)
        self.assertIn('慢', joined)
        self.assertTrue(any('静止' in n for n in notes))

    def test_similar_climbs_get_a_neutral_note(self):
        totals = self._totals(10, 0.2, 2, 0.85)
        segments = [{'name': 'start→top', 'duration': 10.0, 'static_ratio': 0.2,
                     'efficiency': 0.85}]
        notes = report.build_findings('教练', '我', totals, dict(totals),
                                      segments, list(segments))
        self.assertEqual(len(notes), 1)

    def test_writes_html(self):
        directory = tempfile.mkdtemp()
        try:
            video_path = os.path.join(directory, 'clip.mp4')
            make_video(video_path)
            project = Project.create(os.path.join(directory, 'proj'), '测试')
            project.add_clip('coach', video_path, display_name='教练')
            project.add_clip('me', video_path, display_name='我')
            totals = self._totals(10, 0.2, 2, 0.85)
            segments = [{'name': 'start→top', 'duration': 10.0, 'static_ratio': 0.2,
                         'efficiency': 0.85}]
            out = os.path.join(directory, 'report.html')
            report.write(out, project, 'coach', [{
                'target': 'me', 'ref_total': totals, 'target_total': dict(totals),
                'ref_segments': segments, 'target_segments': list(segments),
                'findings': ['测试结论'],
                'height_curves': [[0.0, 1.0, 2.0], [0.0, 0.5, 2.0]],
                'speed_curves': [[1.0, 1.0, None], [0.5, 0.5, 0.5]],
            }])
            with open(out, encoding='utf-8') as fh:
                html = fh.read()
            self.assertIn('教练', html)
            self.assertIn('测试结论', html)
            self.assertIn('<svg', html)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
