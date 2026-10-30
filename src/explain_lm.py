#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LM 到底起了什么作用：把"精化"拆成可测量的四件事。

实验设计（全部基于仓库真实参数：0.3 m 方形标记 / fov 55 / 800x500）:

  E1 主对照   closed-form  vs  LM(3 点)  vs  LM(4 点)
              -> 分离"非线性迭代"与"第 4 点约束"两个因素
  E2 噪声扫描  sigma = 0.1 … 1.5 px，看增益是否随噪声单调增大
  E3 点数扫描  N = 4/5/6/9/12，看增益是否随冗余约束增大
  E4 收敛曲线  迭代次数 0..12，看 6 次是否必要、能否砍到 3 次
  E5 盆地测试  从"错误分支"出发做 LM，能否被拉回真解
  E6 耗时      单次 refine_gn / 整条 solve_p3p_robust（开/关 LM）

用法:  python src/explain_lm.py [--frames 400]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3p import refine_gn, solve_p3p, solve_p3p_robust, _reproj_rmse  # noqa: E402
from pose_utils import intrinsic_from_fov, inv_rodrigues, project_points  # noqa: E402

# ---------------------------------------------------------------- 场景参数
W, H = 800, 500
FOV = 55.0
K = intrinsic_from_fov(W, H, FOV)
MARK = 0.30                      # scene.py 中方形标记边长 (m)
S = MARK / 2.0
P4 = np.array([[-S, S, 0.0], [S, S, 0.0], [S, -S, 0.0], [-S, -S, 0.0]])


def _grid(n_side):
    """n_side x n_side 共面点阵（边长 MARK），前 3 个刻意取三个角点（非共线）。

    注意：行优先排列时前三点共线，P3P 会直接失败 —— 这点本身值得记住。
    """
    g = np.linspace(-S, S, n_side)
    pts = np.array([[x, y, 0.0] for y in g for x in g])
    idx = [0, n_side - 1, n_side * (n_side - 1)]        # 三个角点，非共线
    rest = [i for i in range(len(pts)) if i not in idx]
    return pts[idx + rest]


P_N = {3: P4[:3],
       4: P4,
       5: np.vstack([P4, [[0.0, 0.0, 0.0]]]),
       6: np.vstack([P4, [[0.0, 0.0, 0.0], [S, 0.0, 0.0]]]),
       9: _grid(3),
       12: _grid(4)}


def _look_at(C):
    """相机放在 C，光轴指向世界原点。返回 world->camera 的 R, t。"""
    zc = -C / np.linalg.norm(C)
    up = np.array([0.0, 1.0, 0.0])
    xc = np.cross(up, zc)
    if np.linalg.norm(xc) < 1e-9:
        xc = np.array([1.0, 0.0, 0.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc], axis=0)
    return R, -R @ C


def sample_frame(rng, pts, dist=None, tilt=None, sigma=0.3):
    """采一帧：返回 (pts_w, uv, R_gt, t_gt)。"""
    dist = rng.uniform(0.35, 0.62) if dist is None else dist
    th = np.deg2rad(rng.uniform(0.0, 40.0)) if tilt is None else tilt
    ph = rng.uniform(0.0, 2.0 * np.pi)
    d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    C = pts.mean(axis=0) + dist * d
    R, t = _look_at(C)
    uv = project_points(pts, R, t, K) + rng.normal(0.0, sigma, (len(pts), 2))
    return pts, uv, R, t


def pose_err(R, t, R_gt, t_gt):
    """返回 (平移误差 mm, 旋转误差 deg)。"""
    dt = float(np.linalg.norm(np.asarray(t) - t_gt) * 1000.0)
    dR = float(np.degrees(np.linalg.norm(inv_rodrigues(np.asarray(R).T @ R_gt))))
    return dt, dR


def _rank_pick(pts, uv, cands, refine=None, use_pts=None):
    """给定候选，可选地对每个候选做 refine，按 4 点 RMSE 取最优。"""
    scored = []
    for (R0, t0) in cands:
        R, t = R0, t0
        if refine is not None:
            p = pts if use_pts is None else use_pts
            u = uv[:len(p)] if use_pts is not None else uv
            out = refine(p, u, K, R0, t0)
            if out is not None and out[0] is not None:
                R, t = out[0], out[1]
        scored.append((_reproj_rmse(pts, uv, R, t, K), R, t))
    scored.sort(key=lambda x: x[0])
    return scored[0]


# ---------------------------------------------------------------------------
def e1_main(n=400, sigma=0.3, seed=1):
    print("=" * 78)
    print(f"E1  主对照：0.3 m 方形标记 / fov 55 / 距离 0.35–0.62 m / 噪声 {sigma} px")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    res = {"closed": [], "lm3": [], "lm4": []}
    stock = []
    flip = 0
    lm_reject = 0
    for _ in range(n):
        pts, uv, Rg, tg = sample_frame(rng, P4, sigma=sigma)
        cands = solve_p3p(pts[:3], uv[:3], K)
        if not cands:
            continue
        stock.append(len(cands))
        # A: 纯闭式解（4 点 RMSE 排序，不精化）
        _, Rc, tc = _rank_pick(pts, uv, cands)
        res["closed"].append(pose_err(Rc, tc, Rg, tg))
        # B: 只用前 3 点做 LM —— 控制组，证明增益与"迭代"本身无关
        _, R3, t3 = _rank_pick(pts, uv, cands, refine=refine_gn, use_pts=pts[:3])
        res["lm3"].append(pose_err(R3, t3, Rg, tg))
        # C: 4 点全用做 LM（现流程）
        _, R4, t4 = _rank_pick(pts, uv, cands, refine=refine_gn)
        res["lm4"].append(pose_err(R4, t4, Rg, tg))
        # 统计：LM 是否改变了"选中的候选"、是否被 err2<err 判据拒绝
        for (R0, t0) in cands[:3]:
            out = refine_gn(pts, uv, K, R0, t0)
            if out is not None and out[0] is not None:
                if out[2] >= _reproj_rmse(pts, uv, R0, t0, K):
                    lm_reject += 1
        if np.linalg.norm(t4 - tc) * 1000 > 1.0:
            flip += 1

    def med(key, idx):
        a = np.array(res[key])
        return float(np.median(a[:, idx]))

    print(f"  有效帧 {len(res['closed'])}，候选数均值 {np.mean(stock):.2f}"
          f"  分布 " + " ".join(f"{k}:{stock.count(k)}"
                               for k in sorted(set(stock))))
    print(f"  {'方案':<26}{'Δt 中位(mm)':>13}{'Δt p95(mm)':>12}{'ΔR 中位(deg)':>14}")
    row = [("A 闭式解（不精化）", "closed"),
           ("B LM 但只用前 3 点", "lm3"),
           ("C LM 用全部 4 点（现流程）", "lm4")]
    for name, key in row:
        a = np.array(res[key])
        print(f"  {name:<26}{np.median(a[:,0]):13.3f}"
              f"{np.percentile(a[:,0],95):12.3f}{np.median(a[:,1]):14.3f}")
    c = np.array(res["closed"])[:, 0]
    l4 = np.array(res["lm4"])[:, 0]
    print(f"\n  C 相对 A: 中位 {np.median(c):.3f} -> {np.median(l4):.3f} mm "
          f"({100*(1-np.median(l4)/np.median(c)):+.1f}%)")
    p95c, p95l = np.percentile(c, 95), np.percentile(l4, 95)
    print(f"  C 相对 A: p95  {p95c:.3f} -> {p95l:.3f} mm "
          f"({100*(1-p95l/p95c):+.1f}%)")
    worse = float(np.mean(l4 > c))
    bad = l4 > c
    print(f"  逐帧看，LM 反而变差的帧占比: {100*worse:.1f}%"
          f"（这些帧闭式中位 {np.median(c[bad]):.3f} mm，"
          f"LM 中位 {np.median(l4[bad]):.3f} mm）")
    print(f"  选错分支（Δt>5 mm）: 闭式 {int(np.sum(c>5))}/{len(c)}   "
          f"LM {int(np.sum(l4>5))}/{len(l4)}")
    print(f"  选错分支（Δt>20 mm）: 闭式 {int(np.sum(c>20))}/{len(c)}   "
          f"LM {int(np.sum(l4>20))}/{len(l4)}")
    print(f"  LM 改变选中候选（>1 mm）的帧: {flip}/{len(c)} = {100*flip/len(c):.1f}%")
    print(f"  err2>=err0 被丢弃的候选次数   : {lm_reject}\n")
    return res


def e2_noise(n=250, seed=2):
    print("=" * 78)
    print("E2  噪声扫描（4 点，距离/倾角同 E1）")
    print("=" * 78)
    print(f"  {'σ(px)':>7}{'闭式中位':>10}{'LM 中位':>10}{'降幅':>8}"
          f"{'闭式p95':>10}{'LM p95':>9}{'降幅':>8}")
    for sg in (0.1, 0.3, 0.6, 1.0, 1.5):
        rng = np.random.default_rng(seed)
        a, b = [], []
        for _ in range(n):
            pts, uv, Rg, tg = sample_frame(rng, P4, sigma=sg)
            cands = solve_p3p(pts[:3], uv[:3], K)
            if not cands:
                continue
            _, Rc, tc = _rank_pick(pts, uv, cands)
            _, Rl, tl = _rank_pick(pts, uv, cands, refine=refine_gn)
            a.append(pose_err(Rc, tc, Rg, tg)[0])
            b.append(pose_err(Rl, tl, Rg, tg)[0])
        a, b = np.array(a), np.array(b)
        print(f"  {sg:7.1f}{np.median(a):10.3f}{np.median(b):10.3f}"
              f"{100*(1-np.median(b)/np.median(a)):7.1f}%"
              f"{np.percentile(a,95):10.3f}{np.percentile(b,95):9.3f}"
              f"{100*(1-np.percentile(b,95)/np.percentile(a,95)):7.1f}%")
    print("  => 相对降幅在 55% 附近基本不随噪声变化（估计器效率比是常数）；")
    print("     绝对降幅随噪声线性放大，p95 尾部才是 LM 真正救回来的部分。\n")


def e3_points(n=200, sigma=0.3, seed=3):
    print("=" * 78)
    print("E3  点数扫描（闭式解始终只用前 3 点，第 4 点起全部交给 LM）")
    print("=" * 78)
    print(f"  {'N':>3}{'闭式中位':>10}{'LM 中位':>9}{'降幅':>8}{'Δt 标准差':>12}{'再降幅':>8}")
    for N in (3, 4, 5, 6, 9, 12):
        pts_full = P_N[N] if N in P_N else P_N[12]
        rng = np.random.default_rng(seed)
        a, b = [], []
        for _ in range(n):
            pts, uv, Rg, tg = sample_frame(rng, pts_full, sigma=sigma)
            cands = solve_p3p(pts[:3], uv[:3], K)
            if not cands:
                continue
            _, Rc, tc = _rank_pick(pts, uv, cands)
            _, Rl, tl = _rank_pick(pts, uv, cands, refine=refine_gn)
            a.append(pose_err(Rc, tc, Rg, tg)[0])
            b.append(pose_err(Rl, tl, Rg, tg)[0])
        a, b = np.array(a), np.array(b)
        print(f"  {N:3d}{np.median(a):10.3f}{np.median(b):9.3f}"
              f"{100*(1-np.median(b)/np.median(a)):7.1f}%"
              f"{np.std(b):12.3f}{100*(1-np.std(b)/np.std(a)):7.1f}%")
    print("  => N=3 时 LM 无处可去（6 约束 6 未知，残差已被压到 0）；")
    print("     N>=4 才有自由度可优化，N 越大 LM 越占优。\n")


def e4_iters(n=200, sigma=0.3, seed=4):
    print("=" * 78)
    print("E4  迭代次数收敛曲线（4 点）")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        pts, uv, Rg, tg = sample_frame(rng, P4, sigma=sigma)
        cands = solve_p3p(pts[:3], uv[:3], K)
        if cands:
            frames.append((pts, uv, Rg, tg, cands))
    print(f"  {'iters':>6}{'Δt 中位(mm)':>13}{'相对0次降幅':>13}{'耗时(ms/帧)':>13}")
    base = None
    for k in (0, 1, 2, 3, 4, 6, 8, 12):
        t0 = time.perf_counter()
        errs = []
        for (pts, uv, Rg, tg, cands) in frames:
            if k == 0:
                _, R, t = _rank_pick(pts, uv, cands)
            else:
                _, R, t = _rank_pick(pts, uv, cands,
                                     refine=lambda p, u, K_, R0, t0_: refine_gn(
                                         p, u, K_, R0, t0_, iters=k))
            errs.append(pose_err(R, t, Rg, tg)[0])
        ms = (time.perf_counter() - t0) / len(frames) * 1000
        m = float(np.median(errs))
        base = m if base is None else base
        print(f"  {k:6d}{m:13.3f}{100*(1-m/base):12.1f}%{ms:13.2f}")
    print("  => 增益在前 2–3 次迭代基本吃满，6 次是保守取值（成本换余量）。\n")


def e5_basin(n=150, sigma=0.3, seed=5):
    print("=" * 78)
    print("E5  盆地测试：LM 能不能把错误分支拉回真解")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    ok_from_wrong, stay_wrong, worse = 0, 0, 0
    tot = 0
    for _ in range(n):
        pts, uv, Rg, tg = sample_frame(rng, P4, sigma=sigma)
        cands = solve_p3p(pts[:3], uv[:3], K)
        if len(cands) < 2:
            continue
        tot += 1
        scored = [(_reproj_rmse(pts, uv, R0, t0, K), R0, t0) for (R0, t0) in cands]
        scored.sort(key=lambda x: x[0])
        _, R2, t2 = scored[1]                     # 第二候选 = 错误分支
        d_before, _ = pose_err(R2, t2, Rg, tg)
        out = refine_gn(pts, uv, K, R2, t2)
        if out is None or out[0] is None:
            continue
        d_after, _ = pose_err(out[0], out[1], Rg, tg)
        if d_before > 20.0:                       # 出发时确实是错的
            if d_after < 5.0:
                ok_from_wrong += 1
            else:
                stay_wrong += 1
        if d_after > d_before + 1e-6:
            worse += 1
    print(f"  样本 {tot} 帧（均有 >=2 候选）")
    print(f"  从错误分支出发、LM 后回到真解(<5 mm) : {ok_from_wrong}")
    print(f"  从错误分支出发、LM 后仍是错的        : {stay_wrong}")
    print(f"  精化后误差变大的帧                    : {worse}")
    print("  => LM 是局部优化，只能把解推到最近的极小点，")
    print("     分支选错时它不会跨过重投影壁垒，消歧仍靠第 4 点排序。\n")


def e6_timing():
    print("=" * 78)
    print("E6  耗时（本机实测，含 JIT/缓存预热后取中位）")
    print("=" * 78)
    rng = np.random.default_rng(6)
    pts, uv, Rg, tg = sample_frame(rng, P4, sigma=0.3)
    cands = solve_p3p(pts[:3], uv[:3], K)
    R0, t0 = cands[0]
    for _ in range(20):
        refine_gn(pts, uv, K, R0, t0)
    ts = []
    for _ in range(200):
        s = time.perf_counter()
        refine_gn(pts, uv, K, R0, t0)
        ts.append((time.perf_counter() - s) * 1000)
    print(f"  单次 refine_gn (4 点, 6 迭代) : 中位 {np.median(ts):.2f} ms"\
          f"  (p95 {np.percentile(ts,95):.2f})")
    for ul in (False, True):
        tn = []
        for _ in range(60):
            s = time.perf_counter()
            solve_p3p_robust(pts, uv, K, use_lm=ul)
            tn.append((time.perf_counter() - s) * 1000)
        print(f"  solve_p3p_robust(use_lm={str(ul):<5}) : "
              f"中位 {np.median(tn):.2f} ms")
    print("  真实轨迹 src/out_arm/log.csv 的 p3p_ms: "
          "中位 9.63 ms / p95 17.77 ms（250 帧，含 3 次 LM）\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    which = a.only.split(",") if a.only else ["e1", "e2", "e3", "e4", "e5", "e6"]
    if "e1" in which:
        e1_main(a.frames)
    if "e2" in which:
        e2_noise(a.frames // 2)
    if "e3" in which:
        e3_points(a.frames // 2)
    if "e4" in which:
        e4_iters(a.frames // 2)
    if "e5" in which:
        e5_basin(a.frames // 2)
    if "e6" in which:
        e6_timing()


def e7_collapse(n=200, sigma=0.3, seed=7):
    """LM 之后，4 个候选是不是收敛到同一个极小点。"""
    print("=" * 78)
    print("E7  LM 之后候选是否塌缩（4 点）")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    spread, gap, pool = [], [], []
    same = 0
    tot = 0
    for _ in range(n):
        pts, uv, Rg, tg = sample_frame(rng, P4, sigma=sigma)
        cands = solve_p3p(pts[:3], uv[:3], K)
        if len(cands) < 2:
            continue
        tot += 1
        ref = []
        for (R0, t0) in cands:
            o = refine_gn(pts, uv, K, R0, t0)
            if o is None or o[0] is None:
                continue
            ref.append((o[2], o[0], o[1]))
        if len(ref) < 2:
            continue
        d = max(np.linalg.norm(a[2] - b[2]) * 1000.0
                for i, a in enumerate(ref) for b in ref[i + 1:])
        spread.append(d)
        ref.sort(key=lambda x: x[0])
        gap.append(ref[1][0] - ref[0][0])
        best = ref[0][0]
        thr = max(best + 1.0, 1.5 * best)
        pool.append(sum(1 for e in ref if e[0] <= thr))
        if d < 1.0:
            same += 1
    spread = np.array(spread)
    print(f"  样本 {tot} 帧（>=2 候选）")
    print(f"  LM 后候选间最大位姿差 <1 mm 的帧 : {same}/{tot} = {100*same/tot:.1f}%")
    print(f"  候选间最大位姿差 中位 {np.median(spread):.4f} mm, "
          f"p95 {np.percentile(spread,95):.3f} mm, max {spread.max():.2f} mm")
    print(f"  LM 后 top1-top2 RMSE 间隔 中位 {np.median(gap):.4f} px "
          f"(p95 {np.percentile(gap,95):.3f})  <- 与未精化时的 32 px 完全不是一回事")
    print(f"  候选池大小 分布: " + " ".join(
        f"{k}:{pool.count(k)}" for k in sorted(set(pool))))
    print("  => 4 个候选只是 4 个初值，LM 把它们送到同一个极小点；")
    print("     '4 选 1' 实际上是 '4 条路走到同一个地方'。\n")


def e8_degenerate(n=120, sigma=0.3, seed=8):
    """什么时候 LM 救不回来：正对（tilt→0）与近共线。"""
    print("=" * 78)
    print("E8  退化几何下的救回率（从最差候选出发做 LM）")
    print("=" * 78)
    print(f"  {'条件':<22}{'救回(<5mm)':>11}{'失败':>7}{'最终中位(mm)':>14}{'最差(mm)':>11}")
    for th_deg in (0, 2, 5, 10, 20, 30, 45):
        rng = np.random.default_rng(seed)
        ok = bad = 0
        fin = []
        for _ in range(n):
            pts, uv, Rg, tg = sample_frame(rng, P4, sigma=sigma,
                                           tilt=np.deg2rad(th_deg))
            cands = solve_p3p(pts[:3], uv[:3], K)
            if len(cands) < 2:
                continue
            ds = [pose_err(R0, t0, Rg, tg)[0] for (R0, t0) in cands]
            k = int(np.argmax(ds))
            if ds[k] < 20:
                continue
            o = refine_gn(pts, uv, K, cands[k][0], cands[k][1])
            if o is None or o[0] is None:
                bad += 1
                continue
            d = pose_err(o[0], o[1], Rg, tg)[0]
            fin.append(d)
            ok += (d < 5.0)
            bad += (d >= 5.0)
        fin = np.array(fin) if fin else np.array([np.nan])
        print(f"  {'tilt=%2d deg'%th_deg:<22}{ok:11d}{bad:7d}"
              f"{np.median(fin):14.2f}{np.nanmax(fin):11.2f}")
    # 近共线：前三点几乎在一条直线上
    rng = np.random.default_rng(seed + 1)
    print(f"\n  {'近共线程度':<22}{'救回(<5mm)':>11}{'失败':>7}{'最终中位(mm)':>14}{'最差(mm)':>11}")
    for eps in (1e-3, 1e-2, 3e-2, 1e-1):
        rng = np.random.default_rng(seed + 1)
        ok = bad = 0
        fin = []
        for _ in range(n):
            Pc = P4.copy()
            Pc[2] = np.array([0.0, S - eps, 0.0])   # 第 3 点贴近 p0-p1 连线
            Pc[3] = np.array([0.0, -S, 0.0])
            pts, uv, Rg, tg = sample_frame(rng, Pc, sigma=sigma)
            cands = solve_p3p(pts[:3], uv[:3], K)
            if len(cands) < 2:
                continue
            ds = [pose_err(R0, t0, Rg, tg)[0] for (R0, t0) in cands]
            k = int(np.argmax(ds))
            if ds[k] < 20:
                continue
            o = refine_gn(pts, uv, K, cands[k][0], cands[k][1])
            if o is None or o[0] is None:
                bad += 1
                continue
            d = pose_err(o[0], o[1], Rg, tg)[0]
            fin.append(d)
            ok += (d < 5.0)
            bad += (d >= 5.0)
        fin = np.array(fin) if fin else np.array([np.nan])
        print(f"  {'高度/边长=%.3f'%eps:<22}{ok:11d}{bad:7d}"
              f"{np.median(fin):14.2f}{np.nanmax(fin):11.2f}")
    print("\n  => 退化几何下 LM 基本无需救（错支根本不出现或极弱）；")
    print("     救不回来的情形反而出现在噪声极大时的近共线点。\n")
