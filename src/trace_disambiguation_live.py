"""
trace_disambiguation_live.py
----------------------------
在不改动源码的前提下，挂到 main_arm 的主循环上，逐帧记录消歧的中间量：

  - solve_p3p 返回几个候选
  - 每个候选的 4 点 RMSE
  - ok_pool 有几个成员（err <= max(best+1.0, 1.5*best)）
  - 时序判据最终选的是第几号，与 RMSE 第一是否一致

跑完打印统计。这是"真实工况下时序选支到底有没有起作用"的直接证据。

用法: python src/trace_disambiguation_live.py [--steps 80]
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import main_arm as MA
from p3p import solve_p3p, _reproj_rmse
from pose_utils import inv_rodrigues

STATS = []


def _dR(R1, R2):
    return float(np.linalg.norm(inv_rodrigues(np.asarray(R1).T @ np.asarray(R2))))


def _pose_dist(item, Rp, tp):
    _, R_i, t_i = item
    return float(np.linalg.norm(t_i - tp) + 0.1 * _dR(R_i, Rp))


_orig = MA.solve_p3p_robust


def _wrapped(pts_w, uv, K, use_lm=True, reproj_thresh=2.0,
             R_prev=None, t_prev=None):
    pts_w = np.asarray(pts_w, float)
    uv = np.asarray(uv, float)
    cands = solve_p3p(pts_w[:3], uv[:3], K)
    if cands:
        scored = sorted([(_reproj_rmse(pts_w, uv, R0, t0, K), R0, t0)
                         for (R0, t0) in cands], key=lambda x: x[0])
        best = scored[0][0]
        thr = max(best + 1.0, 1.5 * best)
        pool = [i for i, s in enumerate(scored) if s[0] <= thr]
        if R_prev is not None:
            pick = int(min(pool, key=lambda i: _pose_dist(scored[i], R_prev, t_prev)))
        else:
            pick = 0
        STATS.append(dict(n=len(cands), rms=[s[0] for s in scored],
                          best=best, thr=thr, pool=len(pool),
                          pick=pick, prior=R_prev is not None,
                          gap=(scored[1][0] - scored[0][0]) if len(scored) > 1 else np.inf))
    return _orig(pts_w, uv, K, use_lm=use_lm, reproj_thresh=reproj_thresh,
                 R_prev=R_prev, t_prev=t_prev)


MA.solve_p3p_robust = _wrapped

if __name__ == "__main__":
    steps = 80
    for a in sys.argv:
        if a.startswith("--steps="):
            steps = int(a.split("=")[1])
    sys.argv = ["main_arm.py", "--steps", str(steps), "--out", "out_diag"]
    MA.main()

    print()
    print("=" * 78)
    print(f"真实工况消歧统计（{len(STATS)} 帧）")
    print("=" * 78)
    n = len(STATS)
    from collections import Counter
    print("  候选数分布 :", dict(sorted(Counter(s["n"] for s in STATS).items())))
    print("  池大小分布 :", dict(sorted(Counter(s["pool"] for s in STATS).items())))
    same = sum(1 for s in STATS if s["pick"] == 0)
    print(f"  时序选支 == RMSE 第一的帧数 : {same}/{n} = {100*same/n:.1f}%")
    gaps = np.array([s["gap"] for s in STATS if np.isfinite(s["gap"])])
    print(f"  top1-top2 RMSE 差 : 中位 {np.median(gaps):.3f} px / "
          f"min {gaps.min():.3f} / max {gaps.max():.1f}")
    print(f"  差值 < 1.0 px 的帧 : {int((gaps<1.0).sum())} 帧 "
          f"({100*np.mean(gaps<1.0):.1f}%)  <- 只有这些帧时序才可能改变判决")
    print()
    print("  前 6 帧明细（best / thr / 池 / 选中 / 各候选 RMSE）")
    for i, s in enumerate(STATS[:6]):
        rr = "  ".join(f"{x:8.2f}" for x in s["rms"])
        print(f"    f{i:02d} n={s['n']} best={s['best']:6.3f} thr={s['thr']:6.3f} "
              f"pool={s['pool']} pick={s['pick']+1} prior={int(s['prior'])} | {rr}")
    print()
