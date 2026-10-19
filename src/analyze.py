"""
analyze.py
----------
把仿真日志画成对比曲线 (论文/面试用的成品图)。

产出:
    fig_pbvs.png    自由相机 PBVS 的误差收敛曲线
    fig_arm.png     六轴臂 + IK 的误差、IK 残差、奇异度量
"""

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC",
                                   "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for k in rows[0].keys():
        vals = []
        for r in rows:
            v = r[k]
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def plot_pbvs(d, path):
    t = d["step"] / 30.0
    m = np.isfinite(d["e_t_mm"]) & (d["detect"] > 0.5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    ax[0].semilogy(t[m], d["e_t_mm"][m], lw=1.4, color="#c0392b", label="控制误差 (由估计位姿计算)")
    if "gt_t_mm" in d:
        g = np.isfinite(d["gt_t_mm"]) & (d["detect"] > 0.5)
        ax[0].semilogy(t[g], d["gt_t_mm"][g], lw=1.4, color="#2c3e50",
                       alpha=0.75, label="P3P 估计偏差 (估计 vs 真值)")
    ax[0].axhline(5.0, ls="--", lw=1, color="gray")
    ax[0].set_xlabel("时间 (s)")
    ax[0].set_ylabel("平移误差 (mm)")
    ax[0].set_title("PBVS 平移误差收敛")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)

    ax[1].semilogy(t[m], d["e_R_deg"][m], lw=1.4, color="#c0392b", label="控制误差 (由估计位姿计算)")
    if "gt_R_deg" in d:
        g = np.isfinite(d["gt_R_deg"]) & (d["detect"] > 0.5)
        ax[1].semilogy(t[g], d["gt_R_deg"][g], lw=1.4, color="#2c3e50",
                       alpha=0.75, label="P3P 估计偏差 (估计 vs 真值)")
    ax[1].axhline(0.8, ls="--", lw=1, color="gray")
    ax[1].set_xlabel("时间 (s)")
    ax[1].set_ylabel("旋转误差 (deg)")
    ax[1].set_title("PBVS 旋转误差收敛")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_arm(d, path, summary=None):
    t = d["step"] / 30.0
    m = np.isfinite(d["e_t_mm"])
    fig, ax = plt.subplots(3, 1, figsize=(7.6, 8.2), sharex=True)

    ax[0].semilogy(t[m], d["e_t_mm"][m], lw=1.4, color="#c0392b", label="真实平移误差")
    ax[0].axhline(5.0, ls="--", lw=1, color="gray")
    ax[0].set_ylabel("平移误差 (mm)")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title("xArm6 + DLS-IK 视觉伺服闭环 (eye-in-hand)")

    ax[1].semilogy(t[m], d["e_R_deg"][m], lw=1.4, color="#8e44ad", label="真实旋转误差")
    ax[1].axhline(0.8, ls="--", lw=1, color="gray")
    ax[1].set_ylabel("旋转误差 (deg)")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8, loc="upper right")

    ax[2].semilogy(t[m], np.maximum(d["ik_err_p_mm"][m], 1e-6), lw=1.2,
                   color="#16a085", label="IK 位置残差 (mm)")
    ax2b = ax[2].twinx()
    ax2b.plot(t[m], d["sigma_min"][m], lw=1.1, color="#e67e22",
              alpha=0.85, label="σ_min (奇异度量)")
    ax2b.axhline(0.06, ls=":", lw=1, color="#e67e22")
    ax[2].set_xlabel("时间 (s)")
    ax[2].set_ylabel("IK 残差 (mm)")
    ax2b.set_ylabel("$\\sigma_{min}$")
    ax[2].grid(alpha=0.3, which="both")
    h1, l1 = ax[2].get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    if summary:
        txt = (f"稳态 {summary['steady_t_mm']:.2f} mm / "
               f"{summary['steady_R_deg']:.2f}°\n"
               f"收敛 {summary['conv_time_s']:.2f} s\n"
               f"控制环 {summary['ctrl_hz']:.1f} Hz")
        ax[0].text(0.62, 0.72, txt, transform=ax[0].transAxes, fontsize=9,
                   bbox=dict(fc="white", ec="gray", alpha=0.85))

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    if os.path.exists("out_pbvs/log.csv"):
        d = load("out_pbvs/log.csv")
        plot_pbvs(d, "fig_pbvs.png")
        print("saved fig_pbvs.png")

    if os.path.exists("out_arm/log.csv"):
        d = load("out_arm/log.csv")
        s = None
        if os.path.exists("out_arm/summary.json"):
            s = json.load(open("out_arm/summary.json"))
        plot_arm(d, "fig_arm.png", s)
        print("saved fig_arm.png")


if __name__ == "__main__":
    main()
