"""
plot_diagnostics.py
-------------------
两张用于口径答辩的诊断图，不改动任何既有指标：

  fig_steady_window.png   稳态窗口口径敏感性（250 步标准实验 vs 1000 步长时实验）
  benchmark_full.png      P3P 对比基准的全维度可视化（失败率 / 中位 / p95 / 旋转 / 耗时 / 近奇异）

数据来源全部是仓库内的一手文件：
  out_arm/log.csv, out_long_1000/log.csv, benchmark_out.txt

用法（在 src/ 目录下）：
  python plot_diagnostics.py
"""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC",
                                   "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL_HZ = 30.0


def read_errors(path):
    rows = list(csv.DictReader(open(path)))
    return np.array([float(r["e_t_mm"]) for r in rows])


def plot_steady_window(out_path):
    short_path = os.path.join(HERE, "out_arm", "log.csv")
    long_path = os.path.join(HERE, "out_long_1000", "log.csv")
    e_short = read_errors(short_path)
    e_long = read_errors(long_path) if os.path.exists(long_path) else None
    if e_long is None:
        raise SystemExit("缺少 out_long_1000/log.csv；请先运行 main_arm.py --steps 1000")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) 全程曲线：250 步窗口 vs 1000 步
    ax = axes[0, 0]
    t = np.arange(len(e_long)) / CTRL_HZ
    ax.semilogy(t, e_long, lw=1.2, color="#c0392b", label="1000 步长时实验")
    ax.semilogy(np.arange(len(e_short)) / CTRL_HZ, e_short, lw=1.2,
                color="#2471a3", alpha=0.75, ls="--", label="250 步标准实验")
    ax.axhline(5.0, ls="--", lw=1, color="gray")
    ax.axvspan(235 / CTRL_HZ, 250 / CTRL_HZ, color="#f1c40f", alpha=0.35)
    ax.text(0.98, 0.92, "5 mm 判据", transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_ylim(0.15, 400)
    ax.set_xlabel("闭环仿真时间 (s)")
    ax.set_ylabel("平移误差 (mm，对数)")
    ax.set_title("(a) 误差不会收敛到 0，而是进入约 1.2 mm 的振荡带")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

    # (b) 每 100 步分段均值：是否"越跑越小"
    ax = axes[0, 1]
    seg_means = [e_long[a:a + 100].mean() for a in range(100, 1000, 100)]
    seg_p2p = [e_long[a:a + 100].max() - e_long[a:a + 100].min()
               for a in range(100, 1000, 100)]
    x = np.arange(100, 1000, 100)
    ax.plot(x, seg_means, marker="o", lw=1.8, color="#8e44ad", label="每 100 步均值")
    ax.fill_between(x, 0, seg_p2p, color="#e8daef", alpha=0.8,
                    label="每 100 步峰峰值 (参考)")
    ax.set_xlabel("步数 (每 100 步一个区间)")
    ax.set_ylabel("平移误差 (mm)")
    ax.set_title("(b) 100→600 步缓慢下降，600 步后斜率≈0，进入平台")
    ax.set_ylim(0, 4.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    for xx, yy in zip(x[::2], seg_means[::2]):
        ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, 7),
                    textcoords="offset points", fontsize=7.5, ha="center")

    # (c) 窗口长度敏感性（两条日志）
    ax = axes[1, 0]
    ws = np.arange(1, 101)
    ax.plot(ws, [e_short[-w:].mean() for w in ws], lw=1.8,
            color="#2471a3", label="250 步日志")
    ax.plot(ws, [e_long[-w:].mean() for w in ws], lw=1.8,
            color="#c0392b", label="1000 步日志")
    ax.axvline(15, ls="--", color="#d35400", alpha=0.7)
    ax.annotate("summary.json 固定口径\n末 15 步", xy=(15, 2.0), xytext=(28, 3.0),
                fontsize=8.5, color="#d35400",
                arrowprops=dict(arrowstyle="->", color="#d35400"))
    ax.scatter([15], [e_short[-15:].mean()], s=45, color="#2471a3", zorder=5)
    ax.scatter([15], [e_long[-15:].mean()], s=45, color="#c0392b", zorder=5)
    ax.set_xlabel("稳态窗口长度 W (末 W 步)")
    ax.set_ylabel("窗口内均值 (mm)")
    ax.set_title("(c) 同一口径，窗口一变数字就变")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (d) 截断步数敏感性：最容易踩的坑
    ax = axes[1, 1]
    ns = np.arange(150, 1001, 10)
    reported = [e_long[:n][-15:].mean() for n in ns]
    ax.plot(ns, reported, lw=1.3, color="#16a085")
    ax.axhline(e_long[-200:].mean(), ls=":", color="#7f8c8d",
               label=f"长时平台均值 {e_long[-200:].mean():.2f} mm")
    ax.scatter([250], [e_long[:250][-15:].mean()], s=60, color="#d35400", zorder=5)
    ax.annotate("标准实验停在 250 步\n报出 2.01 mm", xy=(250, 2.01), xytext=(330, 3.1),
                fontsize=8.5, color="#d35400",
                arrowprops=dict(arrowstyle="->", color="#d35400"))
    ax.set_xlabel("实验跑到第 N 步停止")
    ax.set_ylabel("按「末 15 步均值」报出的稳态 (mm)")
    ax.set_title("(d) 换个停表时刻，同一个口径能报出 0.9–4.0 mm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("稳态误差口径诊断｜数据: out_arm/log.csv + out_long_1000/log.csv",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved", out_path)


def parse_benchmark(path):
    """解析 benchmark_out.txt 的 5 张主表 + 近奇异专项。"""
    text = open(path).read()
    tables = {}
    pattern = re.compile(r"--\s*(.+?)\s*--\n(.*?)(?=\n--|\n=====|\Z)", re.S)
    for m in pattern.finditer(text):
        title, body = m.group(1).strip(), m.group(2)
        rec = {}
        for line in body.strip().splitlines():
            if "|" not in line or line.strip().startswith("method"):
                continue
            parts = [p.strip() for p in line.split("|")]
            name = parts[0]
            vals = []
            for p in parts[1:]:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(np.nan)
            if vals:
                rec[name] = vals
        if rec:
            tables[title] = rec
    singular = {}
    sm = re.search(r"=====\s*近奇异专项.*?=====\n(.*?)(?=\n图已保存|\Z)", text, re.S)
    if sm:
        for line in sm.group(1).strip().splitlines():
            mm = re.match(r"(\S+)\s+失败率\s+([\d.]+)%\s+平移中位\s+([\d.]+)\s*mm", line)
            if mm:
                singular[mm.group(1)] = (float(mm.group(2)), float(mm.group(3)))
    return tables, singular


def plot_benchmark_full(out_path):
    path = os.path.join(HERE, "benchmark_out.txt")
    tables, singular = parse_benchmark(path)
    sigmas = [0.3, 0.5, 1.0, 2.0]
    colors = ["#2471a3", "#d35400", "#c0392b", "#8e44ad", "#16a085", "#7f8c8d"]
    focus = ["ours", "ours+LM", "epnp+LM", "cvp3p+LM", "ippe+LM"]
    labels = {"ours": "自研 P3P (闭式)", "ours+LM": "自研 P3P + LM",
              "epnp+LM": "cv2 EPnP + LM", "cvp3p+LM": "cv2 P3P + LM",
              "ippe+LM": "cv2 IPPE + LM"}

    specs = [
        ("失败率    %", "失败率 (%)", "(a) 失败率 vs 像素噪声"),
        ("平移中位 mm", "平移中位误差 (mm)", "(b) 平移中位误差"),
        ("平移p95  mm", "平移 p95 误差 (mm)", "(c) 平移 p95 (长尾)"),
        ("旋转中位deg", "旋转中位误差 (°)", "(d) 旋转中位误差"),
        ("耗时   us", "单次耗时 (ms)", "(e) 单次耗时（同批次，仅比量级）"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    flat = axes.ravel()
    for ax, (key, ylab, title) in zip(flat[:5], specs):
        rec = tables.get(key, {})
        for c, name in zip(colors, focus):
            if name in rec:
                y = np.array(rec[name], dtype=float)
                if key.startswith("耗时"):
                    y = y / 1000.0
                ax.plot(sigmas, y, marker="o", lw=1.6, color=c, label=labels[name])
        ax.set_xlabel("像素噪声 σ (px)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)

    # (f) 近奇异专项
    ax = flat[5]
    names = ["ours", "ours+LM", "epnp", "epnp+LM", "cvp3p+LM", "ippe+LM", "ippesq+LM"]
    vals = [singular.get(n, (np.nan, np.nan))[0] for n in names]
    xs = np.arange(len(names))
    bars = ax.bar(xs, vals, color=["#85c1e9" if "ours" not in n else "#d35400" for n in names])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("失败率 (%)")
    ax.set_title("(f) 近奇异专项（σ=1.0px，最病态 20%）", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                    ha="center", fontsize=7.5)

    fig.suptitle("P3P 对比基准全维度｜数据: benchmark_out.txt (1500 组位姿, seed=0)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved", out_path, "| tables:", list(tables.keys()), "| singular:", len(singular))


if __name__ == "__main__":
    plot_steady_window(os.path.join(HERE, "fig_steady_window.png"))
    plot_benchmark_full(os.path.join(HERE, "benchmark_full.png"))
