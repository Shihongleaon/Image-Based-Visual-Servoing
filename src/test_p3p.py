"""
test_p3p.py
-----------
P3P 求解器的正确性冒烟测试:
用已知真值位姿投影 4 个点, 检验闭式解能否恢复位姿。
"""

import numpy as np
import cv2

from pose_utils import (intrinsic_from_fov, project_points, rodrigues,
                        inv_rodrigues, rigid_transform, reproj_error)
from p3p import solve_p3p, solve_p3p_robust, singularity_measure


def make_gt(rng):
    """随机真值位姿: 相机在标记前方某处。"""
    rvec = rng.normal(size=3) * 0.3
    R_wc = rodrigues(rvec)
    t_wc = np.array([rng.uniform(-0.2, 0.2),
                     rng.uniform(-0.2, 0.2),
                     rng.uniform(1.0, 3.0)])
    return R_wc, t_wc


def main():
    rng = np.random.default_rng(0)
    K = intrinsic_from_fov(800, 500, 90.0)
    s = 0.1  # 0.2m 标记的一半边长
    # 标记位于 z=0 平面 (法向 +z, 面向相机), 单元测试用正对构型
    pts_w = np.array([
        [-s, -s, 0.0],
        [ s, -s, 0.0],
        [ s,  s, 0.0],
        [-s,  s, 0.0],
    ])

    n_ok = 0
    n_trials = 200
    errs = []
    for i in range(n_trials):
        R_gt, t_gt = make_gt(rng)
        uv = project_points(pts_w, R_gt, t_gt, K)
        if np.any(uv[:, 0] < 0) or np.any(uv[:, 0] > 800):
            continue
        if np.any(uv[:, 1] < 0) or np.any(uv[:, 1] > 500):
            continue
        uv_n = uv + rng.normal(scale=0.3, size=uv.shape)  # 0.3px 噪声

        out = solve_p3p_robust(pts_w, uv_n, K)
        if not out["success"]:
            continue
        R, t = out["R"], out["t"]

        # 位姿误差: 平移 mm, 旋转 deg
        e_t = np.linalg.norm(t - t_gt) * 1000.0
        e_R = np.degrees(np.linalg.norm(inv_rodrigues(R.T @ R_gt)))
        errs.append((e_t, e_R))
        if e_t < 10.0 and e_R < 2.0:
            n_ok += 1

    errs = np.array(errs)
    print(f"trials with solution: {len(errs)}/{n_trials}")
    print(f"pose recovered within (10mm, 2deg): {n_ok}/{n_trials}")
    if len(errs):
        print(f"trans err  mm: median {np.median(errs[:,0]):.2f}, "
              f"p95 {np.percentile(errs[:,0],95):.2f}")
        print(f"rot err  deg: median {np.median(errs[:,1]):.4f}, "
              f"p95 {np.percentile(errs[:,1],95):.4f}")

    # 无噪声应当精确恢复
    R_gt, t_gt = make_gt(rng)
    uv = project_points(pts_w, R_gt, t_gt, K)
    out = solve_p3p_robust(pts_w, uv, K)
    print("\nzero-noise robust solve:",
          "OK" if out["success"] and
          np.linalg.norm(out["t"] - t_gt) < 1e-6 and
          np.linalg.norm(inv_rodrigues(out["R"].T @ R_gt)) < 1e-6
          else "FAIL",
          f"(reproj {out['reproj_err']:.2e}px, cands {out['n_candidates']}, "
          f"lm {out['used_lm']}, singular {out['singular']})")

    # 与 cv2 内置 P3P 对照
    ok, rvec, tvec = cv2.solvePnP(pts_w.astype(np.float64),
                                  uv.astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_P3P)
    t_cv = tvec.reshape(3)
    print("cv2 P3P same pose:",
          np.linalg.norm(t_cv - t_gt) < 1e-6,
          f"|t_cv - t_gt| = {np.linalg.norm(t_cv - t_gt):.2e}")

    gap, n = singularity_measure(pts_w[:3], uv[:3], K)
    print(f"singularity gap (frontal view): {gap:.3f} (n_valid={n})")


if __name__ == "__main__":
    main()
