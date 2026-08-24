# 攀岩训练视频比对

把同一条线路的两次（或多次）攀爬，按关键动作对齐到同一条时间轴上，
输出**并排 + 骨骼叠加**的比对视频，以及一份带指标和结论的 HTML 报告。

解决的核心问题：两个人爬同一条线，节奏不一样，直接并排播放几秒后就完全对不上了。
这里用**手动标记的关键帧 + 分段时间伸缩**来对齐，保证同一个动作永远出现在同一帧。

## 安装

```bash
pip install -r requirements-climbing.txt
```

Linux 无图形界面的机器还需要 mediapipe 的原生库依赖：

```bash
sudo apt-get install -y libegl1 libgles2 libgl1
```

首次运行会自动下载姿态模型（约 6MB）到 `~/.cache/climbing-pose/`，
可用环境变量 `CLIMB_MODEL_DIR` 换位置。

## 五步流程

```bash
# 1) 建项目
python -m climbing init 左墙红线 --grade V5

cd climb-projects/左墙红线      # 之后的命令都可以省略 -p

# 2) 加视频（第一条自动成为参考片段）
python -m climbing add coach ~/Movies/教练示范.mp4 --name 教练
python -m climbing add me    ~/Movies/我的第三次.mp4 --name 我

# 3) 标关键帧：两条片段必须用同样的名字
#    先出一张带时间戳的缩略图表来挑时间点
python -m climbing thumbs coach --every 0.5
python -m climbing mark coach --at start=0:02.4 --at crux=0:11.2 --at top=0:26.0
python -m climbing mark me    --at start=0:01.9 --at crux=0:14.8 --at top=0:35.5

# 4) 姿态识别（结果会缓存，改参数重跑加 --force）
python -m climbing analyze

# 5) 出片 + 出报告
python -m climbing compare
```

第 4、5 步可以合成一条：

```bash
python -m climbing pipeline
```

产物在 `out/` 下：`coach-vs-me.mp4` 和 `coach-vs-me.html`。

随时用 `python -m climbing status` 看每条片段的标记情况和是否已经能对齐。

## 关键帧怎么标

关键帧就是两条片段里**同一个动作瞬间**，名字必须一致，至少两个（通常是 `start` 和 `top`）。
标得越多，对齐越准——建议在每个难点前后、换手换脚的节点各标一个。

时间写法都支持：`26`、`0:26`、`1:02.5`、`f:640`（帧号）。

标错了重标同名的即可覆盖，`--clear` 清空重来：

```bash
python -m climbing mark me --clear --at start=0:01.9 --at top=0:35.5
```

## 输出视频

- `--layout side` 只并排，`overlay` 只骨骼叠加，`both`（默认）两者都要
- `--speed 0.5` 放慢一倍看细节
- `--trail 3` 重心轨迹拖尾秒数，`0` 关闭
- `--height 1080` 输出分辨率
- `--out / --report` 指定输出路径

骨骼叠加面板会把对比者的骨架**平移到髋部重合、按躯干长度缩放**后画在参考画面上，
这样机位远近、身高差异都不影响形状对比。

编码优先用 H.264；如果本机 OpenCV 不带 H.264 编码器就退回 MPEG-4 Part 2
（QuickTime/VLC 能播，浏览器可能不行），装了 `ffmpeg` 会自动转成 H.264。

## 报告里的指标

所有距离都按**躯干长度**（肩中点到髋中点）归一化，所以手机远近、身高差异不影响对比。

| 指标 | 含义 |
| --- | --- |
| 静止占比 | 重心速度低于阈值的帧数比例，反映在墙上找点、调整、犹豫的时间 |
| 停顿次数 / 最长停顿 | 连续静止超过 0.6s 记一次 |
| 重心路径效率 | 净爬升 ÷ 重心走过的总路程，越接近 1 说明重心越直上 |
| 直臂时间占比 | 手腕到肩的距离达到该人满伸展 85% 以上的帧数比例 |
| 髋部偏离脚点 | 髋中点与双脚中点的水平距离，越小说明重心越压在脚上 |
| 姿态识别覆盖率 | 有多少帧成功识别到人；低于 60% 时结论仅供参考 |

## 拍摄建议

识别质量直接决定指标质量：

- 机位固定，别跟拍；人在画面里尽量占到 1/3 以上高度
- 避免逆光和强背光，人和墙有明显色差最好
- 画面里只有攀爬者；确实有旁人时加 `--num-poses 2`，程序会跟踪连续性最好的那个
- 识别率低时换更大的模型：`--model full` 或 `--model heavy`

## 常见问题

**「共同关键帧不足 2 个」** — 两条片段的关键帧名字没对上，`status` 看一眼再补 `mark`。

**「整段视频都没有识别到人体」** — 人太小或逆光，先裁剪画面，再试 `--model full`。

**镜像 beta** — 加视频时用 `--flip`，程序会水平翻转画面并同步交换左右骨骼点。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `project.py` | `project.json` 清单：线路、片段、关键帧、缓存路径 |
| `video.py` | 视频元信息、顺序解码（长 GOP 手机视频随机 seek 不可靠） |
| `pose.py` | MediaPipe 姿态提取与 `PoseTrack` 缓存 |
| `align.py` | 关键帧 → 分段线性时间映射 `TimeWarp` |
| `metrics.py` | 重心、速度、静止、路径效率等逐帧与分段指标 |
| `render.py` | 并排 / 叠加合成，HUD、重心拖尾、进度尺 |
| `report.py` | HTML 报告、SVG 曲线、自动结论 |
| `cli.py` | 命令行流水线 |

测试：`python -m unittest discover -s tests`（纯逻辑，不需要 mediapipe）。
