"""
stability.py
------------
控制稳定性量化 (对应优化路线 B1)。

解决的问题: README 里现在只有"收敛 6.03 s、稳态 3.33 mm",
回答不了面试官必问的一句 —— "收敛之后抖不抖?"

本脚本把"稳定"拆成 6 个可辩护的标量:

  1. 收敛步 / 收敛时间    最后一次进入 5mm & 0.8deg 并保持 10 步的位置
  2. 调节时间 (settling)  首次进入 ±0.5mm / ±1mm 稳态带且此后不再离开
  3. 稳态均值 / 标准差    末段误差的集中趋势与散布
  4. 振荡峰峰值 p2p       末段 max-min, 直接回答"抖多少"
  5. 抖动 RMS             e 的一阶差分 RMS, 反映逐帧高频抖动
  6. 反转率               差分符号翻转次数 / 步数, 反映极限环频率

对带 cam_x/y/z 日志 (out_pbvs) 额外算:
  7. 末端速度 RMS / p95   世界系相机速度, 稳态应趋零
  8. 加速度 RMS           速度的差分, 反映指令是否平滑

用法:
    python stability.py                     # 分析 out_pbvs + out_arm
    python stability.py --sweep             # 附加增益/滤波参数扫描 (会跑仿真)
    python stability.py --sweep-only
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL_HZ = 30.0
DT = 1.0 / CTRL_HZ

CONV_T_MM = 5.0
CONV_R_DEG = 0.8
CONV_HOLD = 10


# ---------------------------------------------------------------- 数据读取

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "")
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(np.nan)
    return np.asarray(vals, float)


# ---------------------------------------------------------------- 核心指标

def first_index(arr):
    """arr 中第一个 True 的下标, 无则 None。"""
    hit = np.where(arr)[0]
    return int(hit[0]) if len(hit) else None


def conv_step(e_t, e_R, hold=CONV_HOLD):
    """首次满足 [i, i+hold) 内误差持续低于阈值 的 i。"""
    ok = np.isfinite(e_t) & np.isfinite(e_R) & \
         (e_t < CONV_T_MM) & (e_R < CONV_R_DEG)
    for i in range(len(ok) - hold + 1):
        if ok[i:i + hold].all():
            return i
    return None


def settling_time(x, band):
    """首次进入最终稳态带、且此后不再离开的步号。

    做法: 先取末段均值为稳态参考, 再从末尾向前扫出连续满足带的区间,
    其起点即调节时间。比"第一次碰到阈值"稳健, 因为不会把中途偶然
    穿越误当成收敛。
    """
    n = len(x)
    finite = np.isfinite(x)
    if finite.sum() < 10:
        return None, np.nan
    tail = x[int(n * 0.75):]
    mu = float(np.nanmean(tail))
    ok = finite & (np.abs(x - mu) <= band)
    i = n - 1
    while i >= 0 and (not finite[i] or ok[i]):
        i -= 1
    # i 现在指向最后一个"不满足或非有限"的位置; 其后即稳态段
    start = i + 1
    if start >= n - 5:          # 稳态段太短, 视为未进入
        return None, mu
    return start, mu


def settling_2pct(x):
    """教科书定义的调节时间: 首次进入 ±2%·e(0) 带且此后不再离开。

    对"误差衰减到零"的调节问题, 2% 的用户标准是相对初始误差, 而不是
    相对稳态值 —— 否则极限环永远不算"调节完成", 与工程直觉不符。
    """
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    if not finite.any():
        return None, np.nan
    e0 = float(np.nanmax(np.abs(x)))     # 用峰值而非首帧, 避免首帧恰好很小
    band = 0.02 * e0
    ok = finite & (np.abs(x) <= band)
    i = len(x) - 1
    while i >= 0 and (not finite[i] or ok[i]):
        i -= 1
    start = i + 1
    if start >= len(x) - 5:
        return None, band
    return start, band


def steady_stats(x, frac=0.2, min_n=20):
    n = len(x)
    k = max(int(n * frac), min_n)
    seg = x[-k:]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 3:
        return dict(mu=np.nan, std=np.nan, p2p=np.nan, lo=np.nan, hi=np.nan, n=0)
    return dict(mu=float(seg.mean()), std=float(seg.std(ddof=1)),
                p2p=float(seg.max() - seg.min()),
                lo=float(seg.min()), hi=float(seg.max()), n=int(len(seg)))


def jitter(x):
    """一阶差分 RMS + 符号翻转率 (只在有限值上算)。"""
    d = np.diff(x)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return np.nan, np.nan
    rms = float(np.sqrt(np.mean(d ** 2)))
    s = np.sign(d)
    s = s[s != 0]
    flips = float(np.mean(s[1:] != s[:-1])) if len(s) > 1 else np.nan
    return rms, flips


def velocity_stats(rows, frac=0.2):
    """由 cam_x/y/z 差分得到世界系速度与加速度 (仅 out_pbvs 有)。"""
    if "cam_x" not in rows[0]:
        return None
    P = np.column_stack([col(rows, k) for k in ("cam_x", "cam_y", "cam_z")])
    good = np.isfinite(P).all(axis=1)
    P = P[good]
    if len(P) < 20:
        return None
    V = np.diff(P, axis=0) / DT
    A = np.diff(V, axis=0) / DT
    k = max(int(len(V) * frac), 10)
    sp = np.linalg.norm(V[-k:], axis=1)
    ac = np.linalg.norm(A[-k:], axis=1)
    return dict(v_rms=float(np.sqrt(np.mean(sp ** 2))),
                v_p95=float(np.percentile(sp, 95)),
                a_rms=float(np.sqrt(np.mean(ac ** 2))))


def analyze(path, label):
    rows = load_csv(path)
    e_t = col(rows, "e_t_mm")
    e_R = col(rows, "e_R_deg")
    dt_ok = np.isfinite(e_t)

    k_conv = conv_step(e_t, e_R)
    st05, mu = settling_time(e_t, 0.5)
    st10, _ = settling_time(e_t, 1.0)
    st2, band2 = settling_2pct(e_t)
    ss = steady_stats(e_t)
    rms, flips = jitter(e_t[-max(int(len(e_t) * 0.4), 30):])

    out = dict(
        label=label,
        steps=int(dt_ok.sum()),
        detect_fail=int(np.sum(col(rows, "detect") == 0)) if "detect" in rows[0] else 0,
        conv_step=k_conv,
        conv_time_s=(k_conv / CTRL_HZ) if k_conv is not None else None,
        settle05_step=st05,
        settle10_step=st10,
        settle2pct_step=st2,
        settle2pct_band_mm=band2,
        steady_t_mm=ss["mu"],
        steady_t_std=ss["std"],
        steady_t_p2p=ss["p2p"],
        steady_t_min=ss["lo"],
        steady_t_max=ss["hi"],
        steady_R_deg=steady_stats(e_R)["mu"],
        steady_R_p2p=steady_stats(e_R)["p2p"],
        jitter_rms_mm=rms,
        flip_rate=flips,
        e0_mm=float(e_t[0]) if len(e_t) else None,
    )
    v = velocity_stats(rows)
    if v:
        out.update(v)

    if "sigma_min" in rows[0]:
        sm = col(rows, "sigma_min")
        lam = col(rows, "lam")
        it = col(rows, "ik_iters")
        out.update(sigma_min_min=float(np.nanmin(sm)),
                   sigma_min_med=float(np.nanmedian(sm)),
                   lam_max=float(np.nanmax(lam)),
                   ik_iters_med=float(np.nanmedian(it)),
                   ik_iters_max=float(np.nanmax(it)))
    return out


# ---------------------------------------------------------------- 参数扫描

def run_case(alpha, lam, seed, steps, outdir, dist=None, fov=None):
    cmd = [sys.executable, "main_pbvs.py", "--alpha", f"{alpha}",
           "--lambda-p", f"{lam}", "--lambda-o", f"{lam}",
           "--seed", str(seed), "--steps", str(steps), "--out", outdir]
    if dist is not None:
        cmd += ["--dist", str(dist)]
    if fov is not None:
        cmd += ["--fov", str(fov)]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [fail] alpha={alpha} lam={lam}: {r.stderr.strip()[-200:]}")
        return None
    return analyze(os.path.join(HERE, outdir, "log.csv"),
                   f"a={alpha} l={lam}")


def sweep(alphas, lams, seed=1, steps=200, dist=None, fov=None):
    print(f"\n===== 参数扫描 (seed={seed}, {steps} 步, "
          f"dist={dist or '默认'} fov={fov or '默认'}) =====")
    print(f"{'alpha':>6} {'lambda':>7} {'收敛s':>7} {'稳态mm':>7} "
          f"{'p2p mm':>7} {'抖动mm':>7} {'反转率':>7}")
    results = []
    for a in alphas:
        for l in lams:
            d = f"out_sweep/a{a}_l{l}".replace(".", "")
            os.makedirs(os.path.join(HERE, d), exist_ok=True)
            r = run_case(a, l, seed, steps, d, dist=dist, fov=fov)
            if r is None:
                continue
            results.append(r)
            ct = r["conv_time_s"]
            print(f"{a:>6.2f} {l:>7.2f} "
                  f"{(ct if ct is not None else float('nan')):>7.2f} "
                  f"{r['steady_t_mm']:>7.2f} {r['steady_t_p2p']:>7.2f} "
                  f"{r['jitter_rms_mm']:>7.2f} {r['flip_rate']:>7.2f}")
    os.makedirs(os.path.join(HERE, "out_sweep"), exist_ok=True)
    with open(os.path.join(HERE, "out_sweep", "sweep.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# ---------------------------------------------------------------- 输出

def fmt(v, nd=2):
    if v is None:
        return "   -  "
    if isinstance(v, float) and not np.isfinite(v):
        return "   -  "
    return f"{v:.{nd}f}"


def report(rec, prefix="out"):
    print(f"\n===== {rec['label']} =====")
    print(f"  有效步数              {rec['steps']}   (检测失败 {rec['detect_fail']} 步)")
    print(f"  初始误差              {fmt(rec['e0_mm'],1)} mm")
    ct = rec["conv_time_s"]
    suffix = "" if ct is None else f" = {ct:.2f} s"
    print(f"  收敛 (5mm/0.8deg)     第 {rec['conv_step']} 步{suffix}")
    print(f"  调节时间 ±0.5mm       {rec['settle05_step']} 步")
    print(f"  调节时间 ±1.0mm       {rec['settle10_step']} 步")
    print(f"  调节时间 2%·e0        {rec['settle2pct_step']} 步 "
          f"(带 ±{fmt(rec['settle2pct_band_mm'])} mm)")
    print(f"  稳态 |e_t|            均值 {fmt(rec['steady_t_mm'])} mm, "
          f"标准差 {fmt(rec['steady_t_std'])} mm")
    print(f"  稳态振荡峰峰值 p2p    {fmt(rec['steady_t_p2p'])} mm  "
          f"[{fmt(rec['steady_t_min'])}, {fmt(rec['steady_t_max'])}]")
    print(f"  稳态 |e_R|            均值 {fmt(rec['steady_R_deg'],3)} deg, "
          f"p2p {fmt(rec['steady_R_p2p'],3)} deg")
    print(f"  逐帧抖动 RMS          {fmt(rec['jitter_rms_mm'],3)} mm/step")
    print(f"  差分符号反转率        {fmt(rec['flip_rate'],2)}  (1.0 = 每步都翻转)")
    if "v_rms" in rec:
        print(f"  末端速度 RMS / p95    {fmt(rec['v_rms']*1e3,2)} / "
              f"{fmt(rec['v_p95']*1e3,2)} mm/s")
        print(f"  末端加速度 RMS        {fmt(rec['a_rms']*1e3,1)} mm/s²")
    if "sigma_min_min" in rec:
        print(f"  sigma_min            最小 {fmt(rec['sigma_min_min'],4)}, "
              f"中位 {fmt(rec['sigma_min_med'],4)}")
        print(f"  IK 迭代数            中位 {fmt(rec['ik_iters_med'],0)}, "
              f"最大 {fmt(rec['ik_iters_max'],0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="附加参数扫描")
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--alphas", default="0.2,0.4,0.6")
    ap.add_argument("--lams", default="1.0,2.0")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dist", type=float, default=None,
                    help="期望工作距离 (与基线日志保持一致, 默认 0.6)")
    ap.add_argument("--fov", type=float, default=None,
                    help="相机 FOV (与基线日志保持一致, 默认 55)")
    args = ap.parse_args()

    if not args.sweep_only:
        recs = []
        for sub, label in (("out_pbvs", "自由相机 PBVS (seed 1)"),
                           ("out_arm", "xArm6 手眼 PBVS")):
            f = os.path.join(HERE, sub, "log.csv")
            if os.path.exists(f):
                recs.append(analyze(f, label))
        for r in recs:
            report(r)
        with open(os.path.join(HERE, "out_stability.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(recs, fh, ensure_ascii=False, indent=2)
        print("\n结果已写入 out_stability.json")

    if args.sweep or args.sweep_only:
        alphas = [float(x) for x in args.alphas.split(",")]
        lams = [float(x) for x in args.lams.split(",")]
        sweep(alphas, lams, seed=args.seed, steps=args.steps,
              dist=args.dist, fov=args.fov)


if __name__ == "__main__":
    main()
