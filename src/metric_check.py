"""
metric_check.py
---------------
两件与"指标口径"有关的量化检查 (不参与控制, 只回答评审提出的定义问题):

1. 平移参数误差 ||t - t_gt|| 与相机中心误差 ||C - C_gt|| 的差别。
   约定 X_c = R X_w + t 时, 相机中心 C = -R^T t, 两者一般不相等:
   旋转有误差时, t 的差会混入"绕相机自身旋转"的成分。

2. 采样语义: sample_pose 把 [pitch, yaw, roll] 直接送进 rodrigues,
   因此它是"旋转向量"而不是欧拉角; 报告实际的转角分布, 以及
   d = t_z 与欧氏距离的差别。

运行: python metric_check.py
"""

import numpy as np

from p3p import solve_p3p_robust
from pose_utils import (intrinsic_from_fov, project_points, inv_rodrigues,
                        rodrigues)

K = intrinsic_from_fov(800, 500, 90.0)
S = 0.1                                     # 0.2 m 方形标记
PTS = np.array([[-S, -S, 0], [S, -S, 0], [S, S, 0], [-S, S, 0]], float)


def sample_pose(rng, oblique_only=True):
    lim = 45.0 if oblique_only else 60.0
    rv = np.deg2rad([rng.uniform(-lim, lim) for _ in range(3)])
    R = rodrigues(rv)
    t = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
                  rng.uniform(0.4, 1.2)])
    return R, t, np.linalg.norm(rv)


def cam_center(R, t):
    return -np.asarray(R).T @ np.asarray(t).reshape(3)


def main():
    rng = np.random.default_rng(0)
    n = 2000
    et, ec, er, ang = [], [], [], []
    for _ in range(n):
        R, t, a = sample_pose(rng)
        ang.append(a)
        uv = project_points(PTS, R, t, K) + rng.normal(scale=0.3, size=(4, 2))
        out = solve_p3p_robust(PTS, uv, K, use_lm=True)
        if not (out["success"] and out["R"] is not None):
            continue
        et.append(np.linalg.norm(out["t"] - t) * 1e3)
        ec.append(np.linalg.norm(cam_center(out["R"], out["t"]) - cam_center(R, t)) * 1e3)
        er.append(np.degrees(np.linalg.norm(inv_rodrigues(out["R"].T @ R))))

    et, ec, er = map(np.asarray, (et, ec, er))
    ratio = ec / np.maximum(et, 1e-9)
    print(f"有效样本 {len(et)}/{n}, 噪声 σ=0.3 px, 方形 0.2 m")
    print(f"平移参数误差 ||dt||      中位 {np.median(et):6.2f} mm   P95 {np.percentile(et,95):6.2f} mm")
    print(f"相机中心误差 ||dC||      中位 {np.median(ec):6.2f} mm   P95 {np.percentile(ec,95):6.2f} mm")
    print(f"比值 ||dC||/||dt||       中位 {np.median(ratio):6.3f}     P5 {np.percentile(ratio,5):6.3f}  P95 {np.percentile(ratio,95):6.3f}")

    print("\n按旋转误差分组 (说明旋转误差越大, 两种平移口径差得越多):")
    print(f"{'旋转误差区间':<16}{'样本':>6}{'||dt||中位':>12}{'||dC||中位':>12}{'比值中位':>10}")
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9)]
    for lo, hi in bins:
        m = (er >= lo) & (er < hi)
        if m.sum() < 10:
            continue
        print(f"{f'{lo}–{hi}°':<16}{m.sum():>6}{np.median(et[m]):>12.2f}"
              f"{np.median(ec[m]):>12.2f}{np.median(ratio[m]):>10.3f}")

    print("\n采样语义 (oblique_only=True, 即各分量 U(-45°,45°)):")
    ang = np.asarray(ang)
    print(f"  实际转角 ‖rv‖: 中位 {np.degrees(np.median(ang)):.1f}°  "
          f"P95 {np.degrees(np.percentile(ang,95)):.1f}°  最大 {np.degrees(ang.max()):.1f}°")
    print("  说明: 代码几何意义是'绕随机轴转 ‖rv‖', 不是三个欧拉角分别 ±45°")

    dz, dn = [], []
    rng2 = np.random.default_rng(1)
    for _ in range(n):
        t = np.array([rng2.uniform(-0.3, 0.3), rng2.uniform(-0.3, 0.3),
                      rng2.uniform(0.4, 1.2)])
        dz.append(t[2])
        dn.append(np.linalg.norm(t))
    dz, dn = np.asarray(dz), np.asarray(dn)
    print(f"  t_z 范围 {dz.min():.2f}–{dz.max():.2f} m, "
          f"欧氏距离范围 {dn.min():.2f}–{dn.max():.2f} m, "
          f"比值中位 {np.median(dn/dz):.3f}")


if __name__ == "__main__":
    main()
