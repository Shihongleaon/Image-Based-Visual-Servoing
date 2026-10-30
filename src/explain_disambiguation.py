"""
explain_disambiguation.py
-------------------------
把 `solve_p3p_robust` 的消歧链路拆成可审计的四步，并给出每帧的中间量。

  第 1 步 排序   对每个候选算 4 点重投影 RMSE，升序
  第 2 步 候选池 ok_pool = { err <= max(best + 1.0, 1.5 * best) }
  第 3 步 距度    _pose_dist = ||Δt|| + 0.1 * ||ΔR||     （m + 0.1*rad）
  第 4 步 判决    池内取 _pose_dist 最小；无先验时直接取 RMSE 第一

本脚本用"真实尺度 + 接近正对相机"的合成轨迹复现这个过程，
统计三件事：
  (a) 真解在 RMSE 排名里的名次分布  -> 决定"前 3 个做 LM"是否够
  (b) 单帧判据（只取 RMSE 第一）的选对率
  (c) 加时序先验后的选对率，以及 ok_pool 的大小分布

用法: python src/explain_disambiguation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3p import solve_p3p, refine_gn, _reproj_rmse
from pose_utils import intrinsic_from_fov, rodrigues, project_points, inv_rodrigues

K = intrinsic_from_fov(480, 300, 55.0)     # 与 main_arm 默认 --res 480 一致
MARK = 0.30
HALF = MARK / 2.0
P4 = np.array([[-HALF, HALF, 0.0], [HALF, HALF, 0.0],
               [HALF, -HALF, 0.0], [-HALF, -HALF, 0.0]])


def make_track(n=200, seed=0, noise_px=0.3):
    """生成一条"相机逐步逼近 + 基本正对标记"的轨迹，模拟伺服收敛过程。

    正对是关键: 平面目标的二义分支在正对时重投影几乎相同，
    这才是消歧真正困难的工况（侧看时第 4 点就足以分开）。
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        u = k / max(1, n - 1)
        dist = 0.62 - 0.27 * u                       # 0.62 -> 0.35 m
        az = np.deg2rad(6.0) * np.sin(2.0 * np.pi * u)   # 小幅绕行
        el = np.deg2rad(3.0) * np.cos(2.0 * np.pi * u)
        # 相机中心（标记中心为原点）
        C = np.array([dist * np.sin(az) * np.cos(el),
                      dist * np.sin(el),
                      -dist * np.cos(az) * np.cos(el)])
        # look_at: 相机 z 轴指向原点（相机系 z 朝前）
        zc = -C / np.linalg.norm(C)
        xc = np.cross(np.array([0.0, 1.0, 0.0]), zc)
        if np.linalg.norm(xc) < 1e-9:
            xc = np.array([1.0, 0.0, 0.0])
        xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        R = np.stack([xc, yc, zc], axis=0)           # world -> camera
        t = -R @ C
        uv = project_points(P4, R, t, K) + rng.normal(0.0, noise_px, (4, 2))
        out.append(dict(R=R, t=t, uv=uv))
    return out


def _dR(R1, R2):
    """两个旋转之间的夹角（弧度）。"""
    return float(np.linalg.norm(inv_rodrigues(np.asarray(R1).T @ np.asarray(R2))))


def run(n=200, seed=0, verbose=True):
    track = make_track(n=n, seed=seed)

    rank_hist = {}
    pick_single_ok = 0          # 只取 RMSE 第一的正确率
    pick_temporal_ok = 0        # 加时序先验后的正确率
    pool_hist = {}
    n_cand_hist = {}
    gap_top2 = []               # top1 与 top2 的 RMSE 差
    dpose_true, dpose_false = [], []
    missed = []                 # 时序选错的帧
    detail_frame = None

    R_prev = t_prev = None

    for k, fr in enumerate(track):
        R_gt, t_gt, uv = fr["R"], fr["t"], fr["uv"]
        cands = solve_p3p(P4[:3], uv[:3], K)
        n_cand_hist[len(cands)] = n_cand_hist.get(len(cands), 0) + 1
        if not cands:
            continue

        # ---- 第 1 步: 4 点 RMSE 排序 ----
        scored = [(_reproj_rmse(P4, uv, R0, t0, K), R0, t0) for (R0, t0) in cands]
        scored.sort(key=lambda x: x[0])
        rms = [s[0] for s in scored]

        # 真解 = 位姿最接近真值的那个候选
        truth_idx = int(np.argmin([np.linalg.norm(t0 - t_gt) * 1000.0
                                   + 10.0 * np.degrees(_dR(R0, R_gt))
                                   for (_, R0, t0) in scored]))
        rank_hist[truth_idx + 1] = rank_hist.get(truth_idx + 1, 0) + 1

        if len(rms) >= 2:
            gap_top2.append(rms[1] - rms[0])

        # ---- 第 2 步: 候选池 ----
        best = scored[0][0]
        thr = max(best + 1.0, 1.5 * best)
        pool = [i for i, s in enumerate(scored) if s[0] <= thr]
        pool_hist[len(pool)] = pool_hist.get(len(pool), 0) + 1

        # ---- 第 3 步: 位姿距离 ----
        def pose_dist(item, Rp=None, tp=None):
            _, R_i, t_i = item
            d_t = np.linalg.norm(t_i - tp)
            d_R = _dR(R_i, Rp)
            return d_t + 0.1 * d_R

        # 单帧判据
        pick0 = 0
        pick_single_ok += (pick0 == truth_idx)

        # 时序判据
        if R_prev is None:
            pick = 0
        else:
            pick = min(pool, key=lambda i: pose_dist(scored[i], R_prev, t_prev))
        ok = (pick == truth_idx)
        pick_temporal_ok += ok
        if not ok:
            missed.append(k)

        if detail_frame is None and k == 120 and len(scored) == 4:
            rows = []
            for i, (e, R_i, t_i) in enumerate(scored):
                Xi = (R_i @ P4[:3].T).T + t_i
                dd = np.linalg.norm(Xi, axis=1)
                rows.append(dict(
                    idx=i + 1, rmse=e,
                    in_pool=(i in pool),
                    pd=(pose_dist(scored[i], R_prev, t_prev)
                        if R_prev is not None else float("nan")),
                    dt=np.linalg.norm(t_i - t_gt) * 1000.0,
                    dR=np.degrees(_dR(R_i, R_gt)),
                    p=dd[0] / dd[2], q=dd[1] / dd[2],
                    is_truth=(i == truth_idx)))
            detail_frame = dict(rows=rows, pick=pick, truth=truth_idx,
                                thr=thr, best=best)

        # ---- 先验推进: 用**估计值**（复现 main_arm 的无泄漏写法）----
        _, R_prev, t_prev = scored[pick]

    if verbose:
        _report(n, rank_hist, pick_single_ok, pick_temporal_ok,
                pool_hist, n_cand_hist, gap_top2, missed, detail_frame)


def _report(n, rank_hist, ok1, ok2, pool_hist, ncand, gap, missed, detail):
    print("=" * 78)
    print("消歧链路定量诊断（合成轨迹: 0.3 m 标记 / fov55 / 0.62->0.35 m / 0.3 px 噪声）")
    print("=" * 78)
    print(f"  帧数 {n}，每帧候选数分布: "
          + "  ".join(f"{k}个:{v}" for k, v in sorted(ncand.items())))
    print()
    print("  (a) 真解在 4 点 RMSE 排名中的位置")
    for r in sorted(rank_hist):
        print(f"      第 {r} 名 : {rank_hist[r]:4d} 帧 "
              f"({100*rank_hist[r]/n:5.1f}%)")
    print("      -> '只对前 3 个子做 LM' 是否够，取决于真解是否会掉到第 4 名")
    print()
    print("  (b) 判据正确率")
    print(f"      单帧（只取 RMSE 第一）: {ok1}/{n} = {100*ok1/n:5.1f}%")
    print(f"      加时序先验            : {ok2}/{n} = {100*ok2/n:5.1f}%")
    if missed:
        print(f"      时序判错的帧: {missed[:10]}{' ...' if len(missed) > 10 else ''}")
    print()
    print("  (c) ok_pool 大小分布（池里有几个候选满足 err <= max(best+1.0, 1.5*best)）")
    for k in sorted(pool_hist):
        print(f"      {k} 个 : {pool_hist[k]:4d} 帧 ({100*pool_hist[k]/n:5.1f}%)")
    gap = np.array(gap)
    print(f"      top1 与 top2 的 RMSE 差: 中位 {np.median(gap):.3f} px / "
          f"p10 {np.percentile(gap,10):.3f} / p90 {np.percentile(gap,90):.3f}")
    print(f"      差值 < 1.0 px 的帧占比: {100*np.mean(gap < 1.0):.1f}%"
          "   <- 这些帧若只看 RMSE 就会有歧义")
    print()
    if detail:
        print("  典型帧（第 120 帧）4 个候选的完整评分表")
        print(f"      阈值 thr = max(best+1.0, 1.5*best) = {detail['thr']:.3f} px")
        print(f"      {'#':>2} {'RMSE4':>9} {'入池':>4} {'pose_dist':>10} "
              f"{'dt(mm)':>8} {'dR(deg)':>8} {'p':>6} {'q':>6} {'真解':>4}")
        for r in detail["rows"]:
            print(f"      {r['idx']:2d} {r['rmse']:9.3f} {'Y' if r['in_pool'] else '.':>4} "
                  f"{r['pd']:10.4f} {r['dt']:8.2f} {r['dR']:8.2f} "
                  f"{r['p']:6.3f} {r['q']:6.3f} {'<<' if r['is_truth'] else '':>4}")
        print(f"      判决: 选中 {detail['pick']+1} 号；真解是 {detail['truth']+1} 号")
    print()


if __name__ == "__main__":
    run()


# ---------------------------------------------------------------------------
# 倾角扫描：什么时候"第 4 点"失去区分力，必须靠时序
# ---------------------------------------------------------------------------

def tilt_sweep(n_per=40, seed=11):
    """光轴与标记法向夹角 theta 从 0 扫到 70 度，看 top1-top2 的 RMSE 间隔。"""
    print("=" * 78)
    print("倾角扫描：第 4 点区分度 vs 相机相对标记的倾斜角")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    print(f"  {'theta':>6} {'池=1 比例':>10} {'gap 中位':>10} {'gap p10':>9} "
          f"{'单帧选对':>9} {'候选数':>7}")
    for th_deg in (0, 10, 20, 30, 40, 50, 60, 70):
        th = np.deg2rad(th_deg)
        pools, gaps, ok, nc = [], [], 0, []
        for _ in range(n_per):
            dist = rng.uniform(0.40, 0.55)
            # 相机中心：绕 y 轴倾斜 theta
            C = dist * np.array([np.sin(th), 0.0, np.cos(th)])
            C = C + rng.normal(0.0, 0.008, 3)          # 轻微位置抖动
            zc = -C / np.linalg.norm(C)
            xc = np.cross(np.array([0.0, 1.0, 0.0]), zc)
            if np.linalg.norm(xc) < 1e-9:
                xc = np.array([1.0, 0.0, 0.0])
            xc /= np.linalg.norm(xc)
            yc = np.cross(zc, xc)
            R = np.stack([xc, yc, zc], axis=0)
            t = -R @ C
            uv = project_points(P4, R, t, K) + rng.normal(0.0, 0.3, (4, 2))
            cands = solve_p3p(P4[:3], uv[:3], K)
            if not cands:
                continue
            scored = sorted([(_reproj_rmse(P4, uv, R0, t0, K), R0, t0)
                             for (R0, t0) in cands], key=lambda x: x[0])
            rms = [s[0] for s in scored]
            nc.append(len(rms))
            best = rms[0]
            thr = max(best + 1.0, 1.5 * best)
            pools.append(sum(1 for e in rms if e <= thr))
            if len(rms) > 1:
                gaps.append(rms[1] - rms[0])
            truth = int(np.argmin([np.linalg.norm(s[2] - t) * 1000.0
                                   + 10.0 * np.degrees(_dR(s[1], R))
                                   for s in scored]))
            ok += (truth == 0)
        gaps = np.array(gaps) if gaps else np.array([np.nan])
        print(f"  {th_deg:5d}d {100*np.mean(np.array(pools)==1):9.1f}% "
              f"{np.median(gaps):10.3f} {np.percentile(gaps,10):9.3f} "
              f"{100*ok/max(1,len(nc)):8.1f}% {np.mean(nc):7.2f}")
    print("  => 正对时 gap 很小（对称二义），倾斜后第 4 点迅速变得有区分力。")
    print("     倾角是消歧难度的主导变量，不是视角抖动。\n")


# ---------------------------------------------------------------------------
# 演示：min(iterable, key=fn) 的语义
# ---------------------------------------------------------------------------

def demo_min_key():
    """证明 min(ok_pool, key=_pose_dist) 会逐元素调用 _pose_dist 并按返回值比较。"""
    print("=" * 78)
    print("min(ok_pool, key=_pose_dist) 的语义")
    print("=" * 78)
    t_prev = np.array([0.0, 0.0, 0.500])
    R_prev = np.eye(3)

    # (rmse, R, t, used_lm) —— 与 solve_p3p_robust 里的 evaluated 同构
    evaluated = [
        (0.30, rodrigues(np.array([0.05, 0.10, 0.30])),
         np.array([0.048, -0.012, 0.542]), False),   # RMSE 最优，但位姿跳变
        (0.45, rodrigues(np.array([0.004, 0.008, 0.014])),
         np.array([0.002, -0.001, 0.503]), False),   # RMSE 次优，位姿连续
    ]
    best_err = min(e[0] for e in evaluated)
    thr = max(best_err + 1.0, 1.5 * best_err)
    ok_pool = [e for e in evaluated if e[0] <= thr]

    def _pose_dist(item):
        R_i, t_i = item[1], item[2]
        d_t = float(np.linalg.norm(t_i - t_prev))
        d_R = float(np.linalg.norm(inv_rodrigues(R_i.T @ R_prev)))
        return d_t + 0.1 * d_R

    print(f"  best_err = {best_err:.3f} px, thr = {thr:.3f} px, 池大小 = {len(ok_pool)}")
    print(f"  {'#':>2} {'RMSE(px)':>9} {'d_t(mm)':>9} {'d_R(deg)':>9} {'pose_dist':>11}")
    for i, e in enumerate(ok_pool):
        d_t = np.linalg.norm(e[2] - t_prev) * 1e3
        d_R = np.degrees(np.linalg.norm(inv_rodrigues(e[1].T @ R_prev)))
        star = "  <- RMSE 第一" if e[0] == best_err else ""
        print(f"  {i:2d} {e[0]:9.2f} {d_t:9.2f} {d_R:9.2f} {_pose_dist(e):11.4f}{star}")

    pick = min(ok_pool, key=_pose_dist)
    print(f"\n  min() 返回类型 = {type(pick).__name__}, 长度 = {len(pick)}, "
          f"首元素 RMSE = {pick[0]:.2f}")
    print("  注意: 返回的是**元素本身**（四元组），不是那个距离值 ——")
    print("        所以后面 `err, R, t, lm_any = pick` 才能直接解包。")

    print("\n  等价的手写循环（逐字对应 min 的内部行为）:")
    print("      best, best_val = None, float('inf')")
    print("      for item in ok_pool:")
    print("          v = _pose_dist(item)        # 每个元素调一次")
    print("          if v < best_val:")
    print("              best_val, best = v, item")
    print("      pick = best")

    # 数值对拍
    best, best_val = None, float("inf")
    for item in ok_pool:
        v = _pose_dist(item)
        if v < best_val:
            best_val, best = v, item
    assert best is pick, "min 与手写循环结果不一致"
    print(f"\n  手写循环结果与 min() 完全一致: pose_dist = {best_val:.4f}")
    print(f"  -> 时序判据推翻了 RMSE 第一：选中 RMSE {pick[0]:.2f} 而不是 {best_err:.2f}")
    print("     这就是 ok_pool 长度 > 1 时该函数真正起作用的场景。\n")
