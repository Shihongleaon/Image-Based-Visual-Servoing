"""
tune_lm_iters.py
----------------
`refine_gn` 迭代次数扫描: 收益曲线 vs 耗时, 用于决定默认值该取几次。

设计要点
--------
1. **配对比较**: 同一批 (位姿, 噪声) 样本跑遍所有 iters。若每个 iters 各自重新
   采样, 采样噪声会淹没 iters 带来的差异 —— 早期 E4 就是这么做的, 结论偏保守。
2. **测的是真实管线**: 通过 monkeypatch 替换模块级 `p3p.refine_gn`, 直接调用
   `solve_p3p_robust`, 而不是手写一份"简化版管线"。手写版一旦与原实现有偏差,
   测到的就不是线上行为。
3. 失败判据与 benchmark_p3p 一致 (30 mm / 5 deg), 阈值相同才可跨表比较。

用法: python src/tune_lm_iters.py [--n 1200] [--sigma 0.3]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import p3p
from p3p import solve_p3p_robust
from pose_utils import intrinsic_from_fov, project_points
from benchmark_p3p import PTS_W, FAIL_R_DEG, FAIL_T_MM, pose_errors, sample_pose

K_BENCH = intrinsic_from_fov(800, 500, 90.0)

# 当前工况 (--real 时切换到真实伺服工况: 0.3 m 标记 / fov55 / 0.35~0.62 m)
_PTS = PTS_W
_K = K_BENCH

_ORIG = p3p.refine_gn


def _refiner(iters):
    """返回一个强制 iters 的 refine_gn 包装。"""
    def f(*a, **kw):
        kw["iters"] = iters
        return _ORIG(*a, **kw)
    return f


def build_samples(n, sigma, seed=0, real=False):
    rng = np.random.default_rng(seed)
    data = []
    if real:
        import explain_lm as E
        while len(data) < n:
            pts, uv, R, t = E.sample_frame(rng, _PTS, sigma=sigma)
            data.append((R, t, uv))
        return data
    while len(data) < n:
        R, t = sample_pose(rng)
        uv = project_points(_PTS, R, t, _K)
        uv = uv + rng.normal(scale=sigma, size=uv.shape)
        data.append((R, t, uv))
    return data


def evaluate(data, iters):
    """跑一遍给定 iters 的完整管线, 返回统计量。"""
    p3p.refine_gn = _ORIG if iters == 0 else _refiner(iters)
    try:
        et, er, fails = [], [], 0
        for R_gt, t_gt, uv in data:
            out = solve_p3p_robust(_PTS, uv, _K, use_lm=(iters != 0))
            if not out["success"] or out["R"] is None:
                fails += 1
                continue
            e_t, e_R = pose_errors(out["R"], out["t"], R_gt, t_gt)
            et.append(e_t)
            er.append(e_R)
            if e_t > FAIL_T_MM or e_R > FAIL_R_DEG:
                fails += 1
        et = np.array(et)
        er = np.array(er)
        return dict(fail=100.0 * fails / len(data),
                    et_med=np.median(et), et_p95=np.percentile(et, 95),
                    er_med=np.median(er), er_p95=np.percentile(er, 95))
    finally:
        p3p.refine_gn = _ORIG


def timeit_interleaved(data, iters_list, reps=150, rounds=3):
    """交错多轮耗时测量。

    顺序"测完一个 iters 再测下一个"会被 CPU 调频/负载漂移污染: 先测的那组
    可能恰好处在高频段。交错 + 多轮可把这种漂移摊平到所有组, 取中位。
    """
    buckets = {k: [] for k in iters_list}
    sub = data[:reps]
    for _ in range(rounds):
        for k in iters_list:
            p3p.refine_gn = _ORIG if k == 0 else _refiner(k)
            for R_gt, t_gt, uv in sub:
                t0 = time.perf_counter()
                solve_p3p_robust(_PTS, uv, _K, use_lm=(k != 0))
                buckets[k].append((time.perf_counter() - t0) * 1e3)
    p3p.refine_gn = _ORIG
    return {k: float(np.median(v)) for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--iters", type=str, default="0,1,2,3,4,6")
    ap.add_argument("--real", action="store_true",
                    help="真实伺服工况: 0.3 m 标记 / fov55 / 距离 0.35~0.62 m / 倾角<=40deg")
    args = ap.parse_args()

    global _PTS, _K
    if args.real:
        import explain_lm as E
        _PTS, _K = E.P4, E.K

    iters_list = [int(x) for x in args.iters.split(",")]
    scene = "真实伺服工况 (0.3 m / fov55 / 0.35~0.62 m)" if args.real else \
            "基准工况 (0.2 m / fov90 / 0.4~1.2 m)"
    print("=" * 78)
    print(f"refine_gn 迭代次数扫描   {scene}")
    print(f"样本 {args.n} 组   噪声 sigma={args.sigma} px")
    print("=" * 78)

    data = build_samples(args.n, args.sigma, real=args.real)
    base = None
    rows = []
    for k in iters_list:                      # 精度: 各 iters 独立跑全量
        r = evaluate(data, k)
        rows.append((k, r))
        if base is None:
            base = r
    ms = timeit_interleaved(data, iters_list)  # 耗时: 交错多轮
    print(f"  {'iters':>5} {'失败率':>7} {'Δt中位':>8} {'Δt p95':>8} "
          f"{'ΔR中位':>8} {'ΔR p95':>8} {'管线耗时':>9}")
    for k, r in rows:
        rel = 100.0 * (1.0 - r["et_med"] / base["et_med"]) if base["et_med"] else 0.0
        tag = "" if k == 0 else f"  (Δt 降 {rel:4.1f}%)"
        print(f"  {k:5d} {r['fail']:6.2f}% {r['et_med']:8.3f} {r['et_p95']:8.2f} "
              f"{r['er_med']:8.3f} {r['er_p95']:8.2f} {ms[k]:8.2f}ms{tag}")

    print()
    d = dict(rows)
    k6, k3 = d.get(6), d.get(3)
    if k3 and k6:
        print(f"  3 次 vs 6 次: Δt 中位差 {k6['et_med'] - k3['et_med']:+.4f} mm "
              f"({100*(k6['et_med']-k3['et_med'])/k6['et_med']:+.2f}%), "
              f"p95 差 {k6['et_p95'] - k3['et_p95']:+.3f} mm, "
              f"失败率 {k3['fail']:.2f}% vs {k6['fail']:.2f}%")
        print(f"  耗时: {ms[3]:.2f} vs {ms[6]:.2f} ms "
              f"({100*(ms[6]-ms[3])/ms[6]:+.1f}%)")
    print("  => 以 Δt 中位/p95 与失败率共同找拐点; 耗时收益 <10% 则不动默认值。\n")


if __name__ == "__main__":
    main()
