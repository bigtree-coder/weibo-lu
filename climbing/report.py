"""HTML comparison report: metric tables, curves and plain-language findings."""

import html
import os
from datetime import datetime

import numpy as np

from .timeutil import format_time

METRIC_ROWS = [
    ('duration', '总耗时', 's', 'lower'),
    ('static_ratio', '静止占比', '%', 'lower'),
    ('pauses', '停顿次数', '次', 'lower'),
    ('longest_pause', '最长停顿', 's', 'lower'),
    ('mean_speed', '平均重心速度', '躯干/s', None),
    ('peak_speed', '峰值重心速度', '躯干/s', None),
    ('efficiency', '重心路径效率', '', 'higher'),
    ('straight_arm_ratio', '直臂时间占比', '%', 'higher'),
    ('mean_hip_over_feet', '髋部偏离脚点', '躯干', 'lower'),
    ('mean_lean', '躯干平均倾角', '°', None),
    ('tracked_ratio', '姿态识别覆盖率', '%', 'higher'),
]

PERCENT_KEYS = {'static_ratio', 'straight_arm_ratio', 'tracked_ratio'}


def _fmt(key, value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '—'
    if key in PERCENT_KEYS:
        return f'{value * 100:.0f}%'
    if key == 'pauses':
        return f'{int(value)}'
    if key in ('duration', 'longest_pause'):
        return f'{value:.1f}'
    return f'{value:.2f}'


def _delta(key, ref, target):
    if ref is None or target is None:
        return '', ''
    if not (np.isfinite(ref) and np.isfinite(target)):
        return '', ''
    diff = target - ref
    if key in PERCENT_KEYS:
        text = f'{diff * 100:+.0f}pp'
    elif key == 'pauses':
        text = f'{diff:+.0f}'
    else:
        text = f'{diff:+.2f}'
    return text, diff


def build_findings(ref_name, target_name, ref_total, target_total, ref_segments, target_segments):
    """Turn the numbers into a short list of things worth practising."""
    notes = []
    gap = target_total['duration'] - ref_total['duration']
    if abs(gap) >= 0.5:
        faster = '慢' if gap > 0 else '快'
        notes.append(
            f'整条线 {target_name} 比 {ref_name} {faster} {abs(gap):.1f}s '
            f'（{format_time(target_total["duration"])} vs {format_time(ref_total["duration"])}）'
        )

    worst = None
    for ref_seg, tgt_seg in zip(ref_segments, target_segments):
        diff = tgt_seg['duration'] - ref_seg['duration']
        if worst is None or diff > worst[1]:
            worst = (tgt_seg['name'], diff, ref_seg, tgt_seg)
    if worst and worst[1] > 0.4:
        name, diff, ref_seg, tgt_seg = worst
        notes.append(
            f'最拖时间的是 <b>{html.escape(name)}</b> 段，多花 {diff:.1f}s；'
            f'该段静止占比 {_fmt("static_ratio", tgt_seg["static_ratio"])} '
            f'(参考 {_fmt("static_ratio", ref_seg["static_ratio"])})'
        )

    static_gap = target_total['static_ratio'] - ref_total['static_ratio']
    if np.isfinite(static_gap) and static_gap > 0.1:
        notes.append(
            f'静止时间多出 {static_gap * 100:.0f} 个百分点，'
            f'停顿 {int(target_total["pauses"])} 次、最长 {target_total["longest_pause"]:.1f}s，'
            '通常是在墙上找点或犹豫，建议先在地面把 beta 背下来'
        )

    eff_gap = target_total['efficiency'] - ref_total['efficiency']
    if np.isfinite(eff_gap) and eff_gap < -0.05:
        notes.append(
            f'重心路径效率 {_fmt("efficiency", target_total["efficiency"])} '
            f'低于参考 {_fmt("efficiency", ref_total["efficiency"])}，'
            '说明重心绕了远路，多余的横向晃动可以靠脚下先调整来减少'
        )

    arm_gap = target_total['straight_arm_ratio'] - ref_total['straight_arm_ratio']
    if np.isfinite(arm_gap) and arm_gap < -0.1:
        notes.append(
            f'直臂时间占比 {_fmt("straight_arm_ratio", target_total["straight_arm_ratio"])}，'
            f'比参考少 {abs(arm_gap) * 100:.0f} 个百分点，屈臂挂点更耗前臂'
        )

    hip_gap = target_total['mean_hip_over_feet'] - ref_total['mean_hip_over_feet']
    if np.isfinite(hip_gap) and hip_gap > 0.15:
        notes.append(
            '髋部相对脚点的水平偏移偏大，重心没压在脚上，'
            '容易变成手臂拉、脚打滑'
        )

    if not notes:
        notes.append('两条片段的节奏、停顿和重心路径都接近，差别主要在细节动作上。')
    return notes


def _sample_curve(series, warp, times, attr, rebase=False):
    values = []
    array = getattr(series, attr)
    for t_ref in times:
        index = int(round(warp.to_target(t_ref) * series.fps))
        if 0 <= index < array.shape[0] and np.isfinite(array[index]):
            values.append(float(array[index]))
        else:
            values.append(None)
    if rebase:
        # Height is measured from each clip's first tracked frame; re-zero it at
        # the first shared keyframe so the two curves start from the same line.
        base = next((v for v in values if v is not None), None)
        if base is not None:
            values = [None if v is None else v - base for v in values]
    return values


def _svg_chart(title, curves, colors, names, height=200, y_label=''):
    width = 720
    pad_l, pad_r, pad_t, pad_b = 46, 14, 26, 24
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    flat = [v for curve in curves for v in curve if v is not None]
    if not flat:
        return f'<p class="muted">{html.escape(title)}：数据不足</p>'
    # Clip the range to the bulk of the data so one tracking spike does not
    # flatten the whole curve; outliers are clamped to the box edge.
    lo = float(np.percentile(flat, 1))
    hi = float(np.percentile(flat, 99))
    if hi - lo < 1e-6:
        lo, hi = min(flat), max(flat)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    margin = (hi - lo) * 0.08
    lo, hi = lo - margin, hi + margin

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
             f'aria-label="{html.escape(title)}">']
    for i in range(5):
        y = pad_t + plot_h * i / 4
        value = hi - (hi - lo) * i / 4
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="tick" '
                     f'text-anchor="end">{value:.2f}</text>')

    for curve, color in zip(curves, colors):
        points, run = [], []
        for i, value in enumerate(curve):
            if value is None:
                if len(run) > 1:
                    points.append(run)
                run = []
                continue
            x = pad_l + plot_w * i / max(1, len(curve) - 1)
            clamped = min(max(value, lo), hi)
            y = pad_t + plot_h * (1 - (clamped - lo) / (hi - lo))
            run.append(f'{x:.1f},{y:.1f}')
        if len(run) > 1:
            points.append(run)
        for run in points:
            parts.append(f'<polyline points="{" ".join(run)}" fill="none" '
                         f'stroke="{color}" stroke-width="2.5" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')

    parts.append(f'<text x="{pad_l}" y="16" class="chart-title">'
                 f'{html.escape(title)}{" · " + html.escape(y_label) if y_label else ""}</text>')
    parts.append(f'<text x="{pad_l}" y="{height - 6}" class="tick">起攀</text>')
    parts.append(f'<text x="{width - pad_r}" y="{height - 6}" class="tick" '
                 f'text-anchor="end">完攀</text>')
    parts.append('</svg>')

    legend = ' '.join(
        f'<span class="key"><i style="background:{color}"></i>{html.escape(name)}</span>'
        for color, name in zip(colors, names)
    )
    return f'<figure>{"".join(parts)}<figcaption>{legend}</figcaption></figure>'


CSS = """
:root { color-scheme: light dark; --bg:#ffffff; --fg:#16181d; --muted:#6b7280;
        --line:#e5e7eb; --card:#f8f9fb; --good:#15803d; --bad:#b91c1c; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --fg:#e8eaed; --muted:#9aa1ac; --line:#262a31;
          --card:#171a1f; --good:#4ade80; --bad:#f87171; }
}
* { box-sizing: border-box; }
body { margin:0; padding:32px 20px 64px; background:var(--bg); color:var(--fg);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
       "Hiragino Sans GB","Microsoft YaHei",sans-serif; }
.wrap { max-width: 860px; margin: 0 auto; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.01em; }
h2 { font-size:18px; margin:36px 0 12px; }
.sub { color:var(--muted); margin:0 0 24px; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 8px; }
.card { flex:1 1 150px; background:var(--card); border:1px solid var(--line);
        border-radius:10px; padding:12px 14px; }
.card b { display:block; font-size:22px; font-weight:600; }
.card span { color:var(--muted); font-size:13px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:right; padding:8px 10px; border-bottom:1px solid var(--line);
         white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:600; }
.good { color:var(--good); } .bad { color:var(--bad); }
.muted { color:var(--muted); }
ul.findings { padding-left:20px; } ul.findings li { margin:8px 0; }
figure { margin:16px 0; }
.chart { width:100%; height:auto; background:var(--card);
         border:1px solid var(--line); border-radius:10px; }
.grid { stroke:var(--line); stroke-width:1; }
.tick { fill:var(--muted); font-size:11px; }
.chart-title { fill:var(--fg); font-size:13px; font-weight:600; }
figcaption { margin-top:6px; color:var(--muted); font-size:13px; }
.key { margin-right:14px; } .key i { display:inline-block; width:10px; height:10px;
       border-radius:2px; margin-right:5px; vertical-align:middle; }
video { width:100%; border-radius:10px; border:1px solid var(--line); margin-top:8px; }
code { background:var(--card); padding:2px 5px; border-radius:4px; font-size:13px; }
"""

CHART_COLORS = ['#22c55e', '#f97316', '#3b82f6', '#a855f7']


def write(path, project, ref_label, comparisons, video_info=None):
    """Render the HTML report for one reference and one or more targets."""
    ref_clip = project.clip(ref_label)
    route = project.route
    title = route.get('name') or project.name

    body = ['<div class="wrap">']
    body.append(f'<h1>{html.escape(title)} 动作比对</h1>')
    grade = f' · {html.escape(route.get("grade"))}' if route.get('grade') else ''
    body.append(
        f'<p class="sub">参考：{html.escape(ref_clip.display_name)}{grade} · '
        f'生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>'
    )
    if route.get('notes'):
        body.append(f'<p class="muted">{html.escape(route["notes"])}</p>')

    if video_info:
        rel = os.path.relpath(video_info['path'], os.path.dirname(os.path.abspath(path)))
        body.append('<h2>比对视频</h2>')
        body.append(f'<video controls preload="metadata" src="{html.escape(rel)}"></video>')
        body.append(
            f'<p class="muted">{video_info["size"][0]}×{video_info["size"][1]} · '
            f'{video_info["duration"]:.1f}s · {video_info["fps"]:.0f}fps · '
            f'<code>{html.escape(rel)}</code></p>'
        )
        if not video_info.get('browser_ready', True):
            body.append(
                '<p class="muted">⚠ 该文件是 MPEG-4 Part 2 编码，浏览器可能不解码，'
                '请用 QuickTime / VLC 打开；装上 ffmpeg 后重跑会自动转成 H.264。</p>'
            )

    for comp in comparisons:
        target_clip = project.clip(comp['target'])
        body.append(f'<h2>{html.escape(target_clip.display_name)} vs '
                    f'{html.escape(ref_clip.display_name)}</h2>')

        ref_total, tgt_total = comp['ref_total'], comp['target_total']
        body.append('<div class="cards">')
        for key, label in (('duration', '总耗时 (s)'), ('static_ratio', '静止占比'),
                           ('pauses', '停顿次数'), ('efficiency', '路径效率')):
            delta_text, diff = _delta(key, ref_total.get(key), tgt_total.get(key))
            cls = ''
            if diff not in ('', None) and np.isfinite(diff):
                better = key in ('efficiency',)
                cls = 'good' if (diff > 0) == better else 'bad'
                if abs(diff) < 1e-9:
                    cls = 'muted'
            body.append(
                f'<div class="card"><b>{_fmt(key, tgt_total.get(key))}</b>'
                f'<span>{html.escape(label)} '
                f'<span class="{cls}">{html.escape(delta_text)}</span></span></div>'
            )
        body.append('</div>')

        body.append('<ul class="findings">')
        for note in comp['findings']:
            body.append(f'<li>{note}</li>')
        body.append('</ul>')

        body.append('<h3>整体指标</h3><div class="scroll"><table><thead><tr>'
                    f'<th>指标</th><th>{html.escape(ref_clip.display_name)}</th>'
                    f'<th>{html.escape(target_clip.display_name)}</th><th>差值</th>'
                    '</tr></thead><tbody>')
        for key, label, unit, better in METRIC_ROWS:
            delta_text, diff = _delta(key, ref_total.get(key), tgt_total.get(key))
            cls = ''
            if better and diff not in ('', None) and np.isfinite(diff) and abs(diff) > 1e-9:
                improved = diff < 0 if better == 'lower' else diff > 0
                cls = 'good' if improved else 'bad'
            unit_text = f' <span class="muted">{html.escape(unit)}</span>' if unit else ''
            body.append(
                f'<tr><td>{html.escape(label)}{unit_text}</td>'
                f'<td>{_fmt(key, ref_total.get(key))}</td>'
                f'<td>{_fmt(key, tgt_total.get(key))}</td>'
                f'<td class="{cls}">{html.escape(delta_text) or "—"}</td></tr>'
            )
        body.append('</tbody></table></div>')

        body.append('<h3>分段对比</h3><div class="scroll"><table><thead><tr>'
                    '<th>段落</th><th>参考耗时</th><th>对比耗时</th><th>差值</th>'
                    '<th>参考静止</th><th>对比静止</th><th>参考效率</th><th>对比效率</th>'
                    '</tr></thead><tbody>')
        for ref_seg, tgt_seg in zip(comp['ref_segments'], comp['target_segments']):
            diff = tgt_seg['duration'] - ref_seg['duration']
            cls = 'bad' if diff > 0.2 else ('good' if diff < -0.2 else '')
            body.append(
                f'<tr><td>{html.escape(tgt_seg["name"])}</td>'
                f'<td>{ref_seg["duration"]:.1f}</td><td>{tgt_seg["duration"]:.1f}</td>'
                f'<td class="{cls}">{diff:+.1f}</td>'
                f'<td>{_fmt("static_ratio", ref_seg["static_ratio"])}</td>'
                f'<td>{_fmt("static_ratio", tgt_seg["static_ratio"])}</td>'
                f'<td>{_fmt("efficiency", ref_seg["efficiency"])}</td>'
                f'<td>{_fmt("efficiency", tgt_seg["efficiency"])}</td></tr>'
            )
        body.append('</tbody></table></div>')

        names = [ref_clip.display_name, target_clip.display_name]
        colors = CHART_COLORS[:2]
        body.append(_svg_chart('重心高度（对齐后的时间轴）', comp['height_curves'],
                               colors, names, y_label='躯干长度'))
        body.append(_svg_chart('重心速度（对齐后的时间轴）', comp['speed_curves'],
                               colors, names, y_label='躯干/s'))

    body.append('<h2>指标怎么读</h2><ul class="findings">')
    body.append('<li><b>躯干长度</b>：所有距离都按肩-髋距离归一化，'
                '所以手机远近、身高差异都不影响对比。</li>')
    body.append('<li><b>静止占比</b>：重心速度低于阈值的帧数比例，'
                '反映在墙上找点、调整、犹豫的时间。</li>')
    body.append('<li><b>路径效率</b>：净爬升 ÷ 重心走过的总路程，'
                '越接近 1 说明重心越直上，晃动越少。</li>')
    body.append('<li><b>直臂时间占比</b>：手腕到肩距离达到该人满伸展 85% 以上的帧数比例。</li>')
    body.append('<li><b>髋部偏离脚点</b>：髋中点与双脚中点的水平距离，越小说明重心越压在脚上。</li>')
    body.append('</ul></div>')

    document = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)} 动作比对</title><style>{CSS}</style></head>'
        f'<body>{"".join(body)}</body></html>'
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(document)
    return path


def sample_curves(series, warp, times, attr, rebase=False):
    return _sample_curve(series, warp, times, attr, rebase=rebase)
