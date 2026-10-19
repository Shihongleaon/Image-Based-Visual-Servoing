"""
benchmark_p3p.py
----------------
P3P 求解器蒙特卡洛基准: 自研闭式解(+LM) vs OpenCV 内置方法。

实验设计:
  - 目标: 0.2m 平面方阵标记 (4 角点, 已知 3D 坐标)
  - 相机位姿: 距离 0.5~2.0m, 视角偏航/俯仰 ±45°, 横滚 ±30° 随机采样
  - 像素噪声: 高斯 sigma ∈ {0.3, 0.5, 1.0, 2.0} px
  - 统计: 位姿误差 (平移 mm / 旋转 deg)、失败率 (阈值判定)、单次耗时

方法:
  ours       自研 P3P 闭式解 + 多候选 4 点重投影消歧
  ours+LM    自研 P3P + 每候选 LM 精化 (完整管线)
  epnp       cv2.solvePnP (SOLVEPNP_EPNP)
  epnp+LM    epnp 初值 + LM 精化
  cvp3p      cv2.solvePnP (SOLVEPNP_P3P)
  cvp3p+LM   cvp3p 初值 + LM 精化
"""

import time

import cv2
import numpy as np

from pose_utils import intrinsic_from_fov, project_points, rodrigues, inv_rodrigues
from p3p import solve_p3p, solve_p3p_robust, refine_gn, singularity_measure

IMG_W, IMG_H, FOV = 800, 500, 90.0
K = intrinsic_from_fov(IMG_W, IMG_H, FOV)

MARKER = 0.2
S = MARKER / 2.0
PTS_W = np.array([[-S, -S, 0], [S, -S, 0], [S, S, 0], [-S, S, 0]], float)

FAIL_T_MM = 30.0     # 平移误差失败阈值
FAIL_R_DEG = 5.0     # 旋转误差失败阈值


def sample_pose(rng, oblique_only=False):
    """随机相机位姿 (world->camera)。"""
    if oblique_only:
        yaw = rng.uniform(-45, 45)
        pitch = rng.uniform(-45, 45)
    else:
        yaw = rng.uniform(-60, 60)
        pitch = rng.uniform(-60, 60)
    roll = rng.uniform(-30, 30)
    R_wc = rodrigues(np.deg2rad([pitch, yaw, roll]))
    d = rng.uniform(0.4, 1.2)
    # 相机在标记法向 (+z) 前方 d 处, 带横向偏移
    t_wc = np.array([rng.uniform(-0.3, 0.3),
                     rng.uniform(-0.3, 0.3), d])
    return R_wc, t_wc


def add_noise(uv, sigma, rng):
    return uv + rng.normal(scale=sigma, size=uv.shape)


def pose_errors(R, t, R_gt, t_gt):
    e_t = float(np.linalg.norm(t - t_gt) * 1000.0)
    e_R = float(np.degrees(np.linalg.norm(inv_rodrigues(R.T @ R_gt))))
    return e_t, e_R


def run_ours(uv, lm):
    out = solve_p3p_robust(PTS_W, uv, K, use_lm=lm)
    if not out["success"] or out["R"] is None:
        return None
    return out["R"], out["t"]


def run_cv2(uv, flag, lm):
    ok, rvec, tvec = cv2.solvePnP(PTS_W, uv, K, None, flags=flag)
    if not ok:
        return None
    R, t = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
    if lm:
        out = refine_gn(PTS_W, uv, K, R, t)
        if out is not None and out[0] is not None:
            R, t = out[0], out[1]
    return R, t


METHODS = {
    "ours":     lambda uv: run_ours(uv, lm=False),
    "ours+LM":  lambda uv: run_ours(uv, lm=True),
    "epnp":     lambda uv: run_cv2(uv, cv2.SOLVEPNP_EPNP, lm=False),
    "epnp+LM":  lambda uv: run_cv2(uv, cv2.SOLVEPNP_EPNP, lm=True),
    "cvp3p":    lambda uv: run_cv2(uv, cv2.SOLVEPNP_P3P, lm=False),
    "cvp3p+LM": lambda uv: run_cv2(uv, cv2.SOLVEPNP_P3P, lm=True),
}


def benchmark(n_trials=2000, sigma_list=(0.3, 0.5, 1.0, 2.0), seed=0,
              oblique_only=True, tag=""):
    rng = np.random.default_rng(seed)
    results = {m: {s: dict(errs_t=[], errs_R=[], n_fail=0, n_total=0,
                           time=0.0)
                   for s in sigma_list} for m in METHODS}

    for i in range(n_trials):
        R_gt, t_gt = sample_pose(rng, oblique_only)
        uv0 = project_points(PTS_W, R_gt, t_gt, K)
        # 采样时要求 4 点全部落在图像内
        if (uv0[:, 0] < 8).any() or (uv0[:, 0] > IMG_W - 8).any():
            continue
        if (uv0[:, 1] < 8).any() or (uv0[:, 1] > IMG_H - 8).any():
            continue

        for sigma in sigma_list:
            uv = add_noise(uv0, sigma, rng)
            for name, fn in METHODS.items():
                rec = results[name][sigma]
                rec["n_total"] += 1
                t0 = time.perf_counter()
                try:
                    sol = fn(uv)
                except Exception:
                    sol = None
                rec["time"] += time.perf_counter() - t0
                if sol is None:
                    rec["n_fail"] += 1
                    continue
                e_t, e_R = pose_errors(sol[0], sol[1], R_gt, t_gt)
                if e_t > FAIL_T_MM or e_R > FAIL_R_DEG:
                    rec["n_fail"] += 1
                else:
                    rec["errs_t"].append(e_t)
                    rec["errs_R"].append(e_R)

    print(f"\n===== P3P Benchmark {tag}({n_trials} 组位姿, "
          f"视角{'±45°' if oblique_only else '±60°'}, "
          f"失败阈值 {FAIL_T_MM}mm/{FAIL_R_DEG}°) =====")
    header = f"{'method':<10} | " + " | ".join(
        f"sigma={s:>4}px" for s in sigma_list)
    for metric, fmt in [("median_t", "{:7.2f}"), ("p95_t", "{:7.2f}"),
                        ("fail%", "{:7.1f}"), ("median_R", "{:7.3f}"),
                        ("us/call", "{:7.1f}")]:
        rows = []
        for name in METHODS:
            vals = []
            for s in sigma_list:
                rec = results[name][s]
                n = max(len(rec["errs_t"]), 1)
                if metric == "median_t":
                    v = np.median(rec["errs_t"]) if rec["errs_t"] else np.nan
                elif metric == "p95_t":
                    v = np.percentile(rec["errs_t"], 95) if rec["errs_t"] else np.nan
                elif metric == "fail%":
                    v = 100.0 * rec["n_fail"] / max(rec["n_total"], 1)
                elif metric == "median_R":
                    v = np.median(rec["errs_R"]) if rec["errs_R"] else np.nan
                else:
                    v = 1e6 * rec["time"] / max(rec["n_total"], 1)
                vals.append(fmt.format(v) if np.isfinite(v) else "      -")
            rows.append(f"{name:<10} | " + " | ".join(vals))
        label = {"median_t": "平移中位 mm", "p95_t": "平移p95  mm",
                 "fail%": "失败率    %", "median_R": "旋转中位deg",
                 "us/call": "耗时   us"}[metric]
        print(f"\n-- {label} --\n" + header + "\n" + "\n".join(rows))

    return results


def benchmark_singularity(n_trials=800, seed=1):
    """
    近奇异构型专项: 按 P3P 四次方程最小正实根间距 (danger cylinder 度量)
    挑选最病态的 20% 构型, 统计各方法失败率。
    """
    rng = np.random.default_rng(seed)
    items = []
    while len(items) < n_trials:
        R_gt, t_gt = sample_pose(rng, oblique_only=False)
        uv0 = project_points(PTS_W, R_gt, t_gt, K)
        if (uv0[:, 0] < 8).any() or (uv0[:, 0] > IMG_W - 8).any():
            continue
        if (uv0[:, 1] < 8).any() or (uv0[:, 1] > IMG_H - 8).any():
            continue
        gap, _ = singularity_measure(PTS_W[:3], uv0[:3], K)
        items.append((gap, R_gt, t_gt, uv0))

    items.sort(key=lambda x: x[0])
    hard = items[: n_trials // 5]        # 最病态 20%
    print(f"\n===== 近奇异专项 (最病态 20%, 根间距中位 "
          f"{np.median([h[0] for h in hard]):.4f}) =====")

    sigma = 1.0
    for name, fn in METHODS.items():
        n_fail = n_tot = 0
        errs_t = []
        for gap, R_gt, t_gt, uv0 in hard:
            uv = add_noise(uv0, sigma, rng)
            n_tot += 1
            try:
                sol = fn(uv)
            except Exception:
                sol = None
            if sol is None:
                n_fail += 1
                continue
            e_t, e_R = pose_errors(sol[0], sol[1], R_gt, t_gt)
            if e_t > FAIL_T_MM or e_R > FAIL_R_DEG:
                n_fail += 1
            else:
                errs_t.append(e_t)
        mt = np.median(errs_t) if errs_t else float("nan")
        print(f"{name:<10} 失败率 {100.0*n_fail/n_tot:6.1f}%   "
              f"平移中位 {mt:7.2f} mm   (sigma={sigma}px)")


def plot_results(results, sigma_list, out_prefix="benchmark"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in METHODS:
        xs, ys = [], []
        for s in sigma_list:
            rec = results[name][s]
            ys.append(100.0 * rec["n_fail"] / max(rec["n_total"], 1))
            xs.append(s)
        axes[0].plot(xs, ys, marker="o", label=name)
        ys2 = [np.median(results[name][s]["errs_t"]) if results[name][s]["errs_t"]
               else np.nan for s in sigma_list]
        axes[1].plot(xs, ys2, marker="o", label=name)
    axes[0].set_xlabel("pixel noise sigma (px)")
    axes[0].set_ylabel("failure rate (%)")
    axes[0].set_title(f"Failure rate (>{FAIL_T_MM}mm or >{FAIL_R_DEG} deg)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("pixel noise sigma (px)")
    axes[1].set_ylabel("median translation error (mm)")
    axes[1].set_title("Accuracy vs noise")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_curves.png", dpi=150)
    print(f"\n图已保存: {out_prefix}_curves.png")


if __name__ == "__main__":
    sigma_list = (0.3, 0.5, 1.0, 2.0)
    res = benchmark(n_trials=1500, sigma_list=sigma_list, seed=0,
                    oblique_only=True, tag="")
    benchmark_singularity(n_trials=600, seed=1)
    plot_results(res, sigma_list)
