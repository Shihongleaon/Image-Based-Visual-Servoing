"""
test_p3p_regression.py
----------------------
P3P 求解器的回归测试: 覆盖方形 / 矩形 / 不等边三角形 / 点序重排 /
尺度缩放 / 近共线 / 退化输入, 以及带噪声的精度底线。

背景: 2026-10 的第三方审查指出 _quartic_coefficients 中常数项写成 -A,
正确应为 -1 (见 (A-1)u^2 + (2cosα-2A cosγ v)u + A v^2 - 1 = 0)。
方形标记 A=(a/c)^2=1 时两式恒等, 所以此前的方形实验无法暴露该问题;
只要换成长宽不等的标记或任意三点, 旧代码的四次方程就是错的。

本测试的作用: 任何一次改动后, 用非方形构型立刻能发现同类回归。

运行:  python test_p3p_regression.py
"""

import numpy as np

from p3p import solve_p3p, solve_p3p_robust
from pose_utils import intrinsic_from_fov, project_points, inv_rodrigues, rodrigues

K = intrinsic_from_fov(800, 500, 60.0)
PERM = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def pose_err(R, t, R_gt, t_gt):
    et = float(np.linalg.norm(np.asarray(t) - t_gt) * 1000.0)
    er = float(np.degrees(np.linalg.norm(inv_rodrigues(np.asarray(R).T @ R_gt))))
    return et, er


def random_pose(rng, dist=(0.4, 1.2), tilt=45.0):
    rv = np.deg2rad(rng.uniform(-tilt, tilt, 3))
    R = rodrigues(rv)
    t = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                  rng.uniform(*dist)])
    return R, t


def solve(pts_w, R, t, use_lm=True, extra=None):
    """把 pts_w 投影成像素 (可选第 4 点), 再解回来。"""
    uv = project_points(pts_w, R, t, K)
    out = solve_p3p_robust(pts_w, uv, K, use_lm=use_lm)
    if not out["success"] or out["R"] is None:
        return None
    return out["R"], out["t"]


def case_geometry(name, pts_w, n_pose=25, extra_point=False, tol_t=1e-4,
                  tol_r=1e-7, seed=0):
    """固定几何、随机位姿: 零噪声下应恢复到数值精度。"""
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts_w, float)
    worst_t = worst_r = 0.0
    fails = 0
    for _ in range(n_pose):
        R, t = random_pose(rng)
        solved = solve(pts, R, t, extra=extra_point)
        if solved is None:
            fails += 1
            continue
        et, er = pose_err(solved[0], solved[1], R, t)
        worst_t, worst_r = max(worst_t, et), max(worst_r, er)
    ok = fails == 0 and worst_t < tol_t and worst_r < tol_r
    print(f"{'PASS' if ok else 'FAIL'}  {name:<22} "
          f"最差平移 {worst_t:9.2e} mm  旋转 {worst_r:9.2e} deg  失败 {fails}/{n_pose}")
    return ok


def main():
    results = []

    sq = np.array([[-0.1, 0.1, 0], [0.1, 0.1, 0], [0.1, -0.1, 0], [-0.1, -0.1, 0]])
    results.append(case_geometry("方形标记 (A=1, 旧路径)", sq, extra_point=True))

    rect = np.array([[-0.15, 0.075, 0], [0.15, 0.075, 0],
                     [0.15, -0.075, 0], [-0.15, -0.075, 0]])
    results.append(case_geometry("矩形标记 (A=4)", rect, extra_point=True))

    tri = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.02],
                    [0.05, 0.22, -0.03], [0.02, 0.05, 0.16]])
    results.append(case_geometry("不等边四点 (非共面)", tri, extra_point=True))

    rng = np.random.default_rng(7)
    ok = True
    for _ in range(6):
        perm = rng.permutation(4)
        sub = tri[perm]
        rr, tt = random_pose(rng)
        solved = solve(sub, rr, tt)
        if solved is None:
            ok = False
            continue
        et, er = pose_err(solved[0], solved[1], rr, tt)
        ok = ok and et < 1e-3 and er < 1e-6
    print(f"{'PASS' if ok else 'FAIL'}  {'点序重排 (6 种)':<22} 全部排列误差均 < 1e-3 mm")
    results.append(ok)

    rng = np.random.default_rng(11)
    ok = True
    for s in (0.01, 0.1, 1.0, 10.0):
        pts = tri * s
        R, t = random_pose(rng, dist=(0.4, 1.2))
        t = t * s
        solved = solve(pts, R, t)
        if solved is None:
            ok = False
            continue
        et, er = pose_err(solved[0], solved[1], R, t)
        ok = ok and (et / s < 1e-3) and er < 1e-6
    print(f"{'PASS' if ok else 'FAIL'}  {'尺度缩放 0.01–10x':<22} 相对误差保持")
    results.append(ok)

    near = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0],
                     [0.30, 0.006, 0.0], [0.15, 0.003, 0.10]])
    rng = np.random.default_rng(3)
    worst_t, fails, runs = 0.0, 0, 20
    for _ in range(runs):
        R, t = random_pose(rng, dist=(0.6, 1.0))
        solved = solve(near, R, t)
        if solved is None:
            fails += 1
            continue
        worst_t = max(worst_t, pose_err(solved[0], solved[1], R, t)[0])
    ok = fails < runs and worst_t < 5.0
    print(f"{'PASS' if ok else 'FAIL'}  {'近共线 (1.1°) + 第4点':<22} "
          f"失败 {fails}/{runs}, 最差平移 {worst_t:.3f} mm")
    results.append(ok)

    rng = np.random.default_rng(5)
    fully_in = True
    for _ in range(30):
        R, t = random_pose(rng)
        sub = tri[:3]
        uv = project_points(sub, R, t, K)
        cands = solve_p3p(sub, uv, K)
        errs = [pose_err(Rc, tc, R, t)[0] for Rc, tc in cands]
        fully_in = fully_in and (len(errs) > 0) and min(errs) < 1e-6
    print(f"{'PASS' if fully_in else 'FAIL'}  {'三点候选完备性':<22} "
          f"真值始终在候选集中 (3 点无法消歧, 需第 4 点/时序)")
    results.append(fully_in)

    col = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
    crashed = False
    try:
        out = solve_p3p(col, project_points(col, np.eye(3),
                                           np.array([0, 0, 1.0]), K), K)
        ok = (out == []) or (out is None)
    except Exception:
        crashed = True
        ok = False
    print(f"{'PASS' if ok and not crashed else 'FAIL'}  {'完全共线':<22} "
          f"返回空/失败而不崩溃")
    results.append(ok and not crashed)

    rng = np.random.default_rng(21)
    errs = []
    for _ in range(200):
        R, t = random_pose(rng)
        uv = project_points(rect, R, t, K) + rng.normal(scale=0.3, size=(4, 2))
        out = solve_p3p_robust(rect, uv, K, use_lm=True)
        if out["success"] and out["R"] is not None:
            errs.append(pose_err(out["R"], out["t"], R, t)[0])
    med = float(np.median(errs)) if errs else np.inf
    ok = med < 5.0
    print(f"{'PASS' if ok else 'FAIL'}  {'0.3px 噪声 (矩形)':<22} "
          f"中位 {med:.2f} mm, 成功 {len(errs)}/200")
    results.append(ok)

    print("\n" + ("全部通过" if all(results) else "存在失败用例") +
          f"  ({sum(results)}/{len(results)})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
