"""Command line pipeline: init → add → mark → analyze → compare."""

import argparse
import os
import sys

import numpy as np

from . import align, metrics, pose as pose_mod, report as report_mod, video
from .project import Project, ProjectError
from .timeutil import format_time, parse_time


def log(message):
    print(message, flush=True)


def _open(args):
    if getattr(args, 'project', None):
        return Project.load(args.project)
    return Project.discover()


def _aspect(clip):
    w, h = clip.size
    return (w / h) if h else 1.0


# ---- commands ----------------------------------------------------------
def cmd_init(args):
    root = os.path.abspath(args.dir or os.path.join('climb-projects', args.name))
    project = Project.create(root, args.name, route=args.route,
                             grade=args.grade, notes=args.notes)
    log(f'已创建比对项目「{project.name}」')
    log(f'  目录: {project.root}')
    log('')
    log('下一步:')
    log(f'  python -m climbing -p {project.root} add coach 教练示范.mp4 --name 教练')
    log(f'  python -m climbing -p {project.root} add me 我的尝试.mp4 --name 我')
    return 0


def cmd_add(args):
    project = _open(args)
    clip = project.add_clip(args.label, args.video, display_name=args.name,
                            flip=args.flip, copy=not args.link)
    project.save()
    log(f'已加入片段 {clip.label}（{clip.display_name}）')
    log(f'  {clip.size[0]}×{clip.size[1]} · {clip.fps:.2f}fps · '
        f'{clip.frame_count} 帧 · {format_time(clip.duration)}')
    if project.reference == clip.label:
        log(f'  已设为参考片段（可用 `reference` 命令更改）')
    return 0


def cmd_remove(args):
    project = _open(args)
    project.remove_clip(args.label)
    project.save()
    log(f'已删除片段 {args.label}')
    return 0


def cmd_reference(args):
    project = _open(args)
    project.reference = args.label
    project.save()
    log(f'参考片段已设为 {args.label}')
    return 0


def cmd_mark(args):
    project = _open(args)
    clip = project.clip(args.label)
    if args.clear:
        clip.clear_keyframes()
    for item in args.at or []:
        if '=' not in item:
            raise ProjectError(f'关键帧格式应为 名称=时间，例如 start=0:03.2，收到: {item}')
        name, raw = item.split('=', 1)
        name = name.strip()
        if not name:
            raise ProjectError(f'关键帧名称不能为空: {item}')
        seconds = parse_time(raw.strip(), fps=clip.fps)
        if seconds < 0 or seconds > clip.duration + 1e-6:
            raise ProjectError(
                f'{name}={raw} 超出该片段长度 {format_time(clip.duration)}')
        clip.set_keyframe(name, seconds)
    project.save()

    log(f'{clip.display_name} 的关键帧:')
    for name, t in clip.keyframes:
        log(f'  {name:<12} {format_time(t)}  (帧 {clip.frame_index(t)})')
    if not clip.keyframes:
        log('  (空)')
    return 0


def cmd_thumbs(args):
    """Contact sheet with burnt-in timestamps, for picking keyframes by eye."""
    import cv2
    from .textdraw import draw_text

    project = _open(args)
    clip = project.clip(args.label)
    step = max(1, int(round(args.every * clip.fps)))
    columns = max(1, args.columns)
    aspect = _aspect(clip)
    cell_h = 180
    cell_w = int(round(cell_h * aspect))
    if cell_w < 130:  # portrait phone footage: widen so the timestamp fits
        cell_w = 130
        cell_h = int(round(cell_w / max(aspect, 1e-6)))

    thumbs = []
    for index, frame in video.iter_frames(project.abspath(clip.path), stride=step):
        if clip.flip:
            frame = cv2.flip(frame, 1)
        cell = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(cell, (0, cell_h - 22), (cell_w, cell_h), (20, 20, 24), -1)
        draw_text(cell, f'{format_time(index / clip.fps)} f:{index}',
                  (5, cell_h - 20), size=15, color=(240, 240, 240))
        thumbs.append(cell)

    if not thumbs:
        raise ProjectError('没有读到任何帧')

    rows = (len(thumbs) + columns - 1) // columns
    sheet = np.full((rows * cell_h, columns * cell_w, 3), 24, dtype=np.uint8)
    for i, cell in enumerate(thumbs):
        r, c = divmod(i, columns)
        sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell

    out = args.out or project.path('out', f'{clip.label}-thumbs.jpg')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    log(f'已生成缩略图表 {out}（{len(thumbs)} 帧，每 {args.every}s 一张）')
    log('看图挑出起攀 / 关键点 / 完攀的时间，再用 mark 标记，例如:')
    log(f'  python -m climbing -p {project.root} mark {clip.label} '
        f'--at start=0:02.0 --at crux=0:11.5 --at top=0:26.0')
    return 0


def cmd_analyze(args):
    project = _open(args)
    labels = args.clips or list(project.clips)
    if not labels:
        raise ProjectError('项目里还没有片段，先用 `add` 加入视频')
    for label in labels:
        clip = project.clip(label)
        cache = project.path('cache', f'{label}.pose.json.gz')
        if os.path.exists(cache) and not args.force:
            log(f'{label}: 已有姿态缓存，跳过（--force 可重算）')
            project.set_pose_cache(label, os.path.relpath(cache, project.root))
            continue
        log(f'{label}: 开始姿态提取（{clip.frame_count} 帧）...')
        track = pose_mod.extract(
            project.abspath(clip.path), label,
            {'fps': clip.fps, 'frame_count': clip.frame_count,
             'width': clip.size[0], 'height': clip.size[1]},
            flip=clip.flip, model=args.model, min_confidence=args.min_confidence,
            num_poses=args.num_poses, progress=log,
        )
        track.save(cache)
        project.set_pose_cache(label, os.path.relpath(cache, project.root))
        project.save()
        log(f'{label}: 完成，识别覆盖率 {track.coverage * 100:.0f}%'
            f'  → {os.path.relpath(cache, project.root)}')
        if track.coverage < 0.6:
            log(f'  ⚠ 覆盖率偏低，比对结果可能不稳。可试 --model full 或裁掉多余画面')
    return 0


def _load_track(project, label, args):
    cache = project.pose_cache_path(label) or project.path('cache', f'{label}.pose.json.gz')
    if not os.path.exists(cache):
        if args.no_auto:
            raise ProjectError(f'{label} 还没有姿态数据，先运行 `analyze {label}`')
        log(f'{label}: 没有姿态缓存，自动补跑 analyze')
        sub = argparse.Namespace(project=project.root, clips=[label], force=False,
                                 model=getattr(args, 'model', 'lite'),
                                 min_confidence=getattr(args, 'min_confidence', 0.5),
                                 num_poses=getattr(args, 'num_poses', 1))
        cmd_analyze(sub)
        project = Project.load(project.root)
        cache = project.pose_cache_path(label)
    return pose_mod.PoseTrack.load(cache)


def cmd_compare(args):
    from . import render as render_mod

    project = _open(args)
    ref_label = args.ref or project.reference
    if not ref_label:
        raise ProjectError('没有参考片段，用 `reference <label>` 指定一条')
    targets = args.targets or [l for l in project.clips if l != ref_label]
    if not targets:
        raise ProjectError('没有可比对的片段，至少再 `add` 一条')
    if ref_label in targets:
        raise ProjectError('参考片段不能同时作为对比片段')

    # Check the keyframes first: pose extraction is the slow part, and there is
    # no point paying for it only to find out the marks do not line up.
    warps = {}
    for label in targets:
        warps[label] = align.build(project, ref_label, label)

    labels = [ref_label] + targets
    tracks, series_map = {}, {}
    for label in labels:
        tracks[label] = _load_track(project, label, args)
        series_map[label] = metrics.build(tracks[label], aspect=_aspect(project.clip(label)))

    ref_clip = project.clip(ref_label)
    ref_series = series_map[ref_label]
    comparisons = []
    sample_count = 240

    for label in targets:
        warp = warps[label]
        names = warp.names
        ref_marks = [(n, t) for n, t in zip(names, warp.ref)]
        tgt_marks = [(n, t) for n, t in zip(names, warp.target)]
        ref_total = ref_series.summarize(warp.ref[0], warp.ref[-1])
        tgt_total = series_map[label].summarize(warp.target[0], warp.target[-1])
        ref_segments = ref_series.segments(ref_marks)
        tgt_segments = series_map[label].segments(tgt_marks)

        times = [warp.ref[0] + (warp.ref[-1] - warp.ref[0]) * i / (sample_count - 1)
                 for i in range(sample_count)]
        identity = align.identity_warp(warp.ref, names)
        comparisons.append({
            'target': label,
            'ref_total': ref_total,
            'target_total': tgt_total,
            'ref_segments': ref_segments,
            'target_segments': tgt_segments,
            'findings': report_mod.build_findings(
                ref_clip.display_name, project.clip(label).display_name,
                ref_total, tgt_total, ref_segments, tgt_segments),
            'height_curves': [
                report_mod.sample_curves(ref_series, identity, times, 'height', rebase=True),
                report_mod.sample_curves(series_map[label], warp, times, 'height', rebase=True),
            ],
            'speed_curves': [
                report_mod.sample_curves(ref_series, identity, times, 'speed'),
                report_mod.sample_curves(series_map[label], warp, times, 'speed'),
            ],
        })

    stem = args.name or f'{ref_label}-vs-{"-".join(targets)}'
    video_info = None
    if not args.no_video:
        out_video = args.out or project.path('out', f'{stem}.mp4')
        log(f'合成比对视频 → {os.path.relpath(out_video, project.root)}')
        video_info = render_mod.render(
            project, ref_label, targets, tracks, series_map, warps, out_video,
            layout=args.layout, height=args.height, fps=args.fps, pad=args.pad,
            trail=args.trail, speed=args.speed, codec=args.codec, progress=log,
        )

    report_path = args.report or project.path('out', f'{stem}.html')
    report_mod.write(report_path, project, ref_label, comparisons, video_info)

    log('')
    log('比对完成:')
    if video_info:
        log(f'  视频  {video_info["path"]}  '
            f'（{video_info["duration"]:.1f}s · {video_info["size"][0]}×{video_info["size"][1]}）')
        if not video_info['browser_ready']:
            log('        编码为 MPEG-4 Part 2，浏览器可能不解码，'
                '用 QuickTime/VLC 打开，或装上 ffmpeg 后重跑自动转 H.264')
    log(f'  报告  {report_path}')
    log('')
    for comp in comparisons:
        target_name = project.clip(comp['target']).display_name
        log(f'【{target_name}】')
        for note in comp['findings']:
            log('  - ' + _strip_tags(note))
    return 0


def cmd_status(args):
    project = _open(args)
    route = project.route
    log(f'项目「{project.name}」  {project.root}')
    log(f'  线路: {route.get("name") or "-"}  难度: {route.get("grade") or "-"}')
    log(f'  参考片段: {project.reference or "(未设置)"}')
    log('')
    clips = project.clips
    if not clips:
        log('  还没有片段，用 `add <label> <video>` 加入视频')
        return 0
    for label, clip in clips.items():
        cache = project.pose_cache_path(label)
        pose_state = '已分析' if cache and os.path.exists(cache) else '未分析'
        marks = ', '.join(f'{n}@{format_time(t)}' for n, t in clip.keyframes) or '(未标记)'
        star = '★' if label == project.reference else ' '
        log(f'{star} {label:<10} {clip.display_name:<10} {format_time(clip.duration):>7}  '
            f'{pose_state}')
        log(f'    关键帧: {marks}')
    shared = project.shared_keyframes()
    log('')
    if len(shared) >= 2:
        log(f'  共同关键帧: {" → ".join(shared)}（可以 compare 了）')
    else:
        log('  ⚠ 共同关键帧不足 2 个，先给每条片段标上同名节点，例如 start / top')
    return 0


def cmd_pipeline(args):
    project = _open(args)
    labels = list(project.clips)
    if not labels:
        raise ProjectError('项目里还没有片段，先用 `add` 加入视频')
    analyze_args = argparse.Namespace(
        project=project.root, clips=labels, force=args.force,
        model=args.model, min_confidence=args.min_confidence, num_poses=args.num_poses)
    cmd_analyze(analyze_args)
    log('')
    return cmd_compare(args)


def _strip_tags(text):
    import re
    return re.sub(r'<[^>]+>', '', text)


# ---- parser ------------------------------------------------------------
def _add_pose_flags(parser):
    parser.add_argument('--model', default='lite', choices=['lite', 'full', 'heavy'],
                        help='姿态模型，画面小或逆光可换 full/heavy（默认 lite）')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                        help='姿态检测置信度阈值（默认 0.5）')
    parser.add_argument('--num-poses', type=int, default=1,
                        help='画面里最多识别几个人，>1 时自动跟踪最连续的那个（默认 1）')


def _add_compare_flags(parser):
    parser.add_argument('--ref', help='参考片段标识（默认用项目的参考片段）')
    parser.add_argument('--targets', nargs='*', help='对比片段标识（默认其余全部）')
    parser.add_argument('--layout', default='both', choices=['side', 'overlay', 'both'],
                        help='side=并排, overlay=骨骼叠加, both=两者都要（默认 both）')
    parser.add_argument('--height', type=int, default=720, help='输出视频高度（默认 720）')
    parser.add_argument('--fps', type=float, default=None, help='输出帧率（默认跟参考片段）')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='播放倍速，0.5 表示放慢一倍便于看细节（默认 1.0）')
    parser.add_argument('--pad', type=float, default=0.5,
                        help='首尾各多留几秒（默认 0.5）')
    parser.add_argument('--trail', type=float, default=2.0,
                        help='重心轨迹拖尾秒数，0 关闭（默认 2.0）')
    parser.add_argument('--codec', default='auto', choices=['auto', 'avc1', 'mp4v'],
                        help='输出编码，auto 优先 H.264（默认 auto）')
    parser.add_argument('--out', help='输出视频路径')
    parser.add_argument('--report', help='输出报告路径')
    parser.add_argument('--name', help='输出文件名前缀')
    parser.add_argument('--no-video', action='store_true', help='只出报告，不合成视频')
    parser.add_argument('--no-auto', action='store_true', help='缺少姿态数据时不自动补跑 analyze')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m climbing',
        description='攀岩训练视频比对：把同一条线路的两次攀爬对齐后逐动作比较',
    )
    parser.add_argument('-p', '--project', help='项目目录（默认从当前目录向上查找）')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('init', help='新建比对项目')
    p.add_argument('name', help='项目名')
    p.add_argument('--dir', help='项目目录（默认 climb-projects/<name>）')
    p.add_argument('--route', help='线路名')
    p.add_argument('--grade', help='难度，例如 V5 / 5.12a')
    p.add_argument('--notes', help='备注')
    p.set_defaults(func=cmd_init)

    p = sub.add_parser('add', help='加入一段攀爬视频')
    p.add_argument('label', help='片段标识（英文/数字，例如 coach、me、try2）')
    p.add_argument('video', help='视频文件路径')
    p.add_argument('--name', help='显示名，例如 教练')
    p.add_argument('--flip', action='store_true',
                   help='左右镜像，用于比对镜像 beta 或反向机位')
    p.add_argument('--link', action='store_true', help='不复制文件，直接引用原路径')
    p.set_defaults(func=cmd_add)

    p = sub.add_parser('remove', help='删除片段')
    p.add_argument('label')
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser('reference', help='设置参考片段')
    p.add_argument('label')
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser('mark', help='标记关键帧（两条片段用同样的名字才能对齐）')
    p.add_argument('label')
    p.add_argument('--at', action='append',
                   help='名称=时间，例如 --at start=0:02.4 --at crux=f:340')
    p.add_argument('--clear', action='store_true', help='先清空已有关键帧')
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser('thumbs', help='导出带时间戳的缩略图表，方便挑关键帧')
    p.add_argument('label')
    p.add_argument('--every', type=float, default=1.0, help='每几秒取一帧（默认 1.0）')
    p.add_argument('--columns', type=int, default=8, help='每行几张（默认 8）')
    p.add_argument('--out', help='输出图片路径')
    p.set_defaults(func=cmd_thumbs)

    p = sub.add_parser('analyze', help='跑姿态识别并缓存结果')
    p.add_argument('clips', nargs='*', help='片段标识（默认全部）')
    p.add_argument('--force', action='store_true', help='已有缓存也重算')
    _add_pose_flags(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser('compare', help='对齐并生成比对视频与报告')
    _add_compare_flags(p)
    _add_pose_flags(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser('pipeline', help='一条命令跑完 analyze + compare')
    p.add_argument('--force', action='store_true', help='姿态数据强制重算')
    _add_compare_flags(p)
    _add_pose_flags(p)
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser('status', help='查看项目状态与对齐准备情况')
    p.set_defaults(func=cmd_status)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (ProjectError, align.AlignError, pose_mod.PoseError, video.VideoError) as exc:
        print(f'错误: {exc}', file=sys.stderr)
        return 1
