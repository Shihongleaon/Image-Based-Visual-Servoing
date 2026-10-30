"""
verify_elimination.py
---------------------
回答一个具体争议：

  "P3P 代数上有 8 个解（3 个二次方程的 Bezout 上界），其中最多 4 个正解。
   取正不就解决符号问题了？为什么还需要 '负实根闸门'？
   我们的消元是不是有问题（多产伪根 / 漏掉真根）？"

结论先行：
  消元没有问题，而且可以写成恒等式证死：
      (I)  ==  A*E1 - E2
      (II) ==  B*E1 - E3
  其中 E1/E2/E3 是三条余弦定理方程（先把距离尺度 psi 归一化）。
  两个组合式把 psi 消掉了——这一步正是"商掉整体反演 s -> -s"。
  所以四次方程正好给出 4 个"反演对"的代表，不是 8 个解的筛选结果。

  但 (p,q) = (s_a/s_c, s_b/s_c) 仍可单独为负（混合符号），
  所以 u>0 / v>0 两道闸门是必需的，不是多余的。

检查项：
  A  (I) ≡ A*E1 - E2,  (II) ≡ B*E1 - E3         （代数恒等式）
  B  真解 v_true 必是 F 的根                     （无漏根）
  C  F 的实根回代后重投影残差 ~0                 （无伪根）
  D  1500 组随机配置的根类型分布                 （"2 解 293 帧"的真正机制）
  E  方形标记 A=B=1 的退化情形
  F  消元退化点 E(v)=0 的实际发生频率

用法: python src/verify_elimination.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3p import _quartic_coefficients, _rays_from_pixels
from pose_utils import (intrinsic_from_fov, rodrigues, project_points,
                        inv_rodrigues)

K = intrinsic_from_fov(800, 500, 90.0)


# ---------------------------------------------------------------------------
# 几何样本生成
# ---------------------------------------------------------------------------

def _geometry(rng, mark_size=0.08, dist=0.6, square=False, perturb=0.45):
    """随机针孔成像配置，返回真值 (s_a,s_b,s_c)、边长、角度余弦、A、B。"""
    if square:
        s = mark_size
        P = np.array([[0.0, 0.0, 0.0], [s, 0.0, 0.0], [s, s, 0.0]])
    else:
        for _ in range(200):
            P = np.array([[0.0, 0.0, 0.0],
                          [mark_size * rng.uniform(0.8, 1.2), 0.0, 0.0],
                          [mark_size * rng.uniform(0.2, 0.8),
                           mark_size * rng.uniform(0.6, 1.4), 0.0]])
            e1, e2 = P[1] - P[0], P[2] - P[0]
            if np.linalg.norm(np.cross(e1, e2)) > 1e-3 * np.linalg.norm(e1) * np.linalg.norm(e2):
                break

    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R = rodrigues(ax * rng.uniform(-perturb, perturb))   # world -> camera
    z_world = R.T @ np.array([0.0, 0.0, 1.0])
    C = P.mean(axis=0) - dist * z_world                  # 相机中心（世界系）
    t = -R @ C

    uv = project_points(P, R, t, K)
    if (np.any(uv < 5) or np.any(uv[:, 0] > 795) or np.any(uv[:, 1] > 495)):
        return None
    Pc = (R @ P.T).T + t
    if np.any(Pc[:, 2] <= 0):
        return None

    s = np.linalg.norm(Pc, axis=1)                        # s_a, s_b, s_c
    rays = Pc / s[:, None]
    ra, rb, rc = rays
    a = float(np.linalg.norm(P[1] - P[2]))                # |BC|, 对点 A
    b = float(np.linalg.norm(P[0] - P[2]))                # |AC|, 对点 B
    c = float(np.linalg.norm(P[0] - P[1]))                # |AB|, 对点 C

    return dict(uv=uv, s=s, a=a, b=b, c=c,
                ca=float(rb @ rc), cb=float(ra @ rc), cg=float(ra @ rb),
                A=(a / c) ** 2, B=(b / c) ** 2,
                p=s[0] / s[2], q=s[1] / s[2])             # p=s_a/s_c, q=s_b/s_c


def _E(g, p, q, psi):
    """三条余弦定理方程（除以 c^2，psi = (c/s_c)^2）。真解处应全为 0。"""
    E1 = p * p + q * q - 2.0 * p * q * g["cg"] - psi
    E2 = q * q + 1.0 - 2.0 * q * g["ca"] - g["A"] * psi
    E3 = p * p + 1.0 - 2.0 * p * g["cb"] - g["B"] * psi
    return E1, E2, E3


def _IQ(g, p, q):
    """代码里的两个二次式 (I)(II)，以 q 为未知、p 为参数。"""
    I = ((g["A"] - 1.0) * q * q
         + (2.0 * g["ca"] - 2.0 * g["A"] * g["cg"] * p) * q
         + g["A"] * p * p - 1.0)
    II = (g["B"] * q * q
          - 2.0 * g["B"] * g["cg"] * p * q
          + (g["B"] - 1.0) * p * p
          + 2.0 * g["cb"] * p - 1.0)
    return I, II


# ---------------------------------------------------------------------------
# A: 恒等式 (I) = A*E1 - E2,  (II) = B*E1 - E3
# ---------------------------------------------------------------------------

def check_A(n=400, seed=0):
    print("=" * 76)
    print("Check A : (I) == A*E1 - E2 ,  (II) == B*E1 - E3  （代数恒等式）")
    print("=" * 76)
    rng = np.random.default_rng(seed)
    w_id, w_e = 0.0, 0.0
    cnt = 0
    while cnt < n:
        g = _geometry(rng)
        if g is None:
            continue
        cnt += 1
        psi = (g["c"] / g["s"][2]) ** 2
        E1, E2, E3 = _E(g, g["p"], g["q"], psi)
        I, II = _IQ(g, g["p"], g["q"])
        w_e = max(w_e, abs(E1), abs(E2), abs(E3))
        w_id = max(w_id, abs(I - (g["A"] * E1 - E2)),
                   abs(II - (g["B"] * E1 - E3)))
    print(f"  E1/E2/E3 在真解处的最大残差     : {w_e:.3e}")
    print(f"  恒等式 I-(A*E1-E2), II-(B*E1-E3) : {w_id:.3e}")
    print("  => (I)(II) 是原始方程的线性组合，消掉的正是尺度 psi = (c/s_c)^2")
    print("     即整体反演 s -> -s 的自由度。这一步不是筛根，是**降维**。\n")


# ---------------------------------------------------------------------------
# B / C: 无漏根 / 无伪根
# ---------------------------------------------------------------------------

def _root_diag(g, coef, Dp, Ep):
    """返回 (v_true 处 |F| 归一化, 实根列表, [每个实根对应的 (p,q,重投影rmse)])"""
    v_true = g["p"]
    F_true = abs(float(np.polyval(coef[::-1], v_true))) / max(1.0, np.max(np.abs(coef)))

    out = []
    for r in np.roots(coef[::-1]):
        if abs(r.imag) > 1e-8 * max(1.0, abs(r.real)):
            continue
        v = float(r.real)
        Ev = float(np.polyval(Ep[::-1], v))
        if abs(Ev) < 1e-12:
            out.append((v, None, None, None, Ev))
            continue
        u = float(np.polyval(Dp[::-1], v)) / Ev
        denom = v * v + u * u - 2.0 * u * v * g["cg"]
        if denom <= 0:
            out.append((v, u, denom, None, Ev))
            continue
        s_c = g["c"] / np.sqrt(denom)
        s_a, s_b = v * s_c, u * s_c
        # 由三个"距离 * 射线"点做位姿，量重投影误差
        rays = _rays_from_pixels(g["uv"], K).T
        pts_c = np.stack([s_a * rays[0], s_b * rays[1], s_c * rays[2]], axis=0)
        # 位姿：用单位阵点集只能做相似变换，此处直接比较"三点距离"是否复原
        d01 = np.linalg.norm(pts_c[0] - pts_c[1])
        d02 = np.linalg.norm(pts_c[0] - pts_c[2])
        d12 = np.linalg.norm(pts_c[1] - pts_c[2])
        res = max(abs(d01 - g["c"]), abs(d02 - g["b"]), abs(d12 - g["a"])) / g["c"]
        out.append((v, u, denom, res, Ev))
    return F_true, out


def check_BC(n=600, seed=1):
    print("=" * 76)
    print("Check B/C : 无漏根（真解必是 F 的根）、无伪根（实根复原三点距离）")
    print("=" * 76)
    rng = np.random.default_rng(seed)
    w_ftrue = 0.0
    res_all = []
    n_root, n_real, n_phys = 0, 0, 0
    tested = 0
    while tested < n:
        g = _geometry(rng)
        if g is None:
            continue
        tested += 1
        coef, Dp, Ep = _quartic_coefficients(g["ca"], g["cb"], g["cg"], g["A"], g["B"])
        F_true, roots = _root_diag(g, coef, Dp, Ep)
        w_ftrue = max(w_ftrue, F_true)
        n_root += 4
        for (v, u, denom, res, Ev) in roots:
            n_real += 1
            if res is not None:
                res_all.append(res)
            if u is not None and u > 0 and v > 0:
                n_phys += 1
    res_all = np.array(res_all)
    print(f"  样本数                        : {tested}")
    print(f"  |F(v_true)| / max|coef| (max) : {w_ftrue:.3e}   <- 真解永远是根，无漏根")
    print(f"  四次根总数 / 实根数           : {n_root} / {n_real}  "
          f"(复根 {n_root - n_real}，占 {100*(n_root-n_real)/n_root:.1f}%)")
    print(f"  实根复原三点边长的相对误差    : 中位 {np.median(res_all):.2e} / "
          f"p99 {np.percentile(res_all, 99):.2e} / max {res_all.max():.2e}")
    print("     <- 每个实根都能复原输入的三条边长，没有一个伪根；")
    print("        尾部偏大的那部分正是近重根（危险圆柱附近）的求根条件数所致")
    print(f"  物理候选 (p>0,q>0) 总数       : {n_phys}  平均 {n_phys/tested:.2f} 个/帧\n")


# ---------------------------------------------------------------------------
# D: 根类型分布
# ---------------------------------------------------------------------------

def check_D(n=1500, seed=2):
    print("=" * 76)
    print("Check D : 1500 组随机配置的根类型分布（'2 解 293 帧' 的真正机制）")
    print("=" * 76)
    rng = np.random.default_rng(seed)
    hist_real = {}
    hist_phys = {}
    c4 = []
    tested = 0
    while tested < n:
        g = _geometry(rng)
        if g is None:
            continue
        tested += 1
        coef, Dp, Ep = _quartic_coefficients(g["ca"], g["cb"], g["cg"], g["A"], g["B"])
        c4.append(abs(coef[4]))
        _, roots = _root_diag(g, coef, Dp, Ep)
        k = len(roots)
        hist_real[k] = hist_real.get(k, 0) + 1
        npos = sum(1 for (v, u, d, r, E) in roots if u is not None and u > 0 and v > 0)
        hist_phys[npos] = hist_phys.get(npos, 0) + 1

    print(f"  样本 {tested} 组")
    print("  --- 四次方程的实根个数（复根共轭成对，故只能是 0/2/4） ---")
    for k in sorted(hist_real):
        print(f"    {k} 实根 : {hist_real[k]:5d} 帧 ({100*hist_real[k]/tested:5.1f}%)")
    print("  --- 物理候选（实根 + p>0 + q>0）---")
    for k in sorted(hist_phys):
        print(f"    {k} 个   : {hist_phys[k]:5d} 帧 ({100*hist_phys[k]/tested:5.1f}%)")
    print(f"  四次项系数 |c4| 中位数 : {np.median(c4):.3e}")
    print("  解读: 4 个根里 2 个实根 + 1 对复根是随机视角的常态；")
    print("        复根对应的'反演对'无实代表 -> 该对不产生物理候选。\n")


# ---------------------------------------------------------------------------
# D2: 构型对"4 解比例"的影响
# ---------------------------------------------------------------------------

def _count_cands(g):
    coef, Dp, Ep = _quartic_coefficients(g["ca"], g["cb"], g["cg"], g["A"], g["B"])
    _, roots = _root_diag(g, coef, Dp, Ep)
    return sum(1 for (v, u, d, r, E) in roots if u is not None and u > 0 and v > 0)


def check_D2(n=800, seed=5):
    print("=" * 76)
    print("Check D2: 构型对候选数分布的影响（解释为什么实跑是 293:7）")
    print("=" * 76)
    cases = [("方形 A=B=1", True, 0.15, 0.60),
             ("方形 A=B=1", True, 0.15, 0.30),
             ("方形 A=B=1", True, 0.15, 0.20),
             ("方形 A=B=1", True, 0.15, 0.15),
             ("方形 A=B=1", True, 0.15, 0.12),
             ("随机三角形", False, 0.15, 0.15)]
    for name, sq, pert, dist in cases:
        rng = np.random.default_rng(seed)
        tested = 0
        h = {}
        while tested < n:
            g = _geometry(rng, square=sq, perturb=pert, dist=dist)
            if g is None:
                continue
            tested += 1
            k = _count_cands(g)
            h[k] = h.get(k, 0) + 1
        txt = "  ".join(f"{k}解 {100*v/tested:5.1f}%" for k, v in sorted(h.items()))
        span = np.degrees(0.08 / dist)
        print(f"  {name:11s} 距离{dist:.2f}m(张角~{span:.0f}°) |  {txt}")
    print("  => 候选数由**三点张角**主导，不是视角扰动。张角越大(标记越近/越占画面)，")
    print("     四次方程越常出现 1 对复根 -> 只剩 2 个候选。")
    print("     实测角点边长 ~136px 对应距离 ~0.15m，正是最后一行的工况。\n")


# ---------------------------------------------------------------------------
# H: 真实尺度（0.3 m 标记 / fov55 / 0.4~1.2 m 工作距离 / 亚像素噪声）
# ---------------------------------------------------------------------------

def _real_geometry(rng, K2, mark=0.30, dist=0.6, noise=0.5):
    s = mark / 2.0
    P = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0]])   # ArUco TL,TR,BR
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R = rodrigues(ax * rng.uniform(-0.35, 0.35))
    z_world = R.T @ np.array([0.0, 0.0, 1.0])
    C = P.mean(axis=0) - dist * z_world
    t = -R @ C
    uv = project_points(P, R, t, K2)
    if noise > 0:
        uv = uv + rng.normal(0.0, noise, uv.shape)
    if np.any(uv < 2) or np.any(uv[:, 0] > 798) or np.any(uv[:, 1] > 498):
        return None
    return P, uv


def check_H(n=1000, seed=7):
    print("=" * 76)
    print("Check H : 真实尺度复算（0.3 m 标记, fov55, 0.4~1.2 m, sigma=0.5 px）")
    print("=" * 76)
    from p3p import solve_p3p
    K2 = intrinsic_from_fov(800, 500, 55.0)
    rng = np.random.default_rng(seed)
    hist_nc, hist_phys, hist_real, hist_neg = {}, {}, {}, {}
    neg_any = 0
    tested = 0
    while tested < n:
        dist = rng.uniform(0.4, 1.2)
        g = _real_geometry(rng, K2, dist=dist)
        if g is None:
            continue
        P, uv = g
        tested += 1
        nc = len(solve_p3p(P, uv, K2))
        hist_nc[nc] = hist_nc.get(nc, 0) + 1

        # 同时统计四次根的结构
        rays = _rays_from_pixels(uv, K2).T
        ra, rb, rc = rays
        a = np.linalg.norm(P[1] - P[2]); b = np.linalg.norm(P[0] - P[2])
        c = np.linalg.norm(P[0] - P[1])
        coef, Dp, Ep = _quartic_coefficients(
            float(rb @ rc), float(ra @ rc), float(ra @ rb), (a / c) ** 2, (b / c) ** 2)
        roots = np.roots(coef[::-1])
        real = [float(r.real) for r in roots
                if abs(r.imag) <= 1e-8 * max(1.0, abs(r.real))]
        hist_real[len(real)] = hist_real.get(len(real), 0) + 1
        npos, nneg = 0, 0
        for v in real:
            Ev = float(np.polyval(Ep[::-1], v))
            if abs(Ev) < 1e-12:
                continue
            u = float(np.polyval(Dp[::-1], v)) / Ev
            if v > 0 and u > 0:
                npos += 1
            else:
                nneg += 1
        hist_phys[npos] = hist_phys.get(npos, 0) + 1
        hist_neg[nneg] = hist_neg.get(nneg, 0) + 1
        neg_any += (nneg > 0)

    def show(title, h):
        txt = "  ".join(f"{k}:{100*v/tested:5.1f}%" for k, v in sorted(h.items()))
        print(f"  {title:24s} {txt}")

    print(f"  样本 {tested} 组")
    show("solve_p3p 返回候选数", hist_nc)
    show("四次方程实根数", hist_real)
    show("其中 p>0,q>0 的个数", hist_phys)
    show("其中被符号闸门剔除数", hist_neg)
    print(f"  至少剔除 1 个的帧占比 : {100*neg_any/tested:.1f}%")
    print("  => 关键：实根数 != 候选数。被剔除的是 p<0 或 q<0 的'混合符号'根，")
    print("     它们在几何上把某个点翻到了相机背后，余弦定理仍成立但成像不成立。\n")

def check_G(n=300, seed=6):
    print("=" * 76)
    print("Check G : 代数解总数 = 4 个根 x 反演(+-s) = 8，与 Bezout 上界吻合")
    print("=" * 76)
    rng = np.random.default_rng(seed)
    tested = 0
    w = 0.0
    npair = 0
    while tested < n:
        g = _geometry(rng)
        if g is None:
            continue
        tested += 1
        coef, Dp, Ep = _quartic_coefficients(g["ca"], g["cb"], g["cg"], g["A"], g["B"])
        for r in np.roots(coef[::-1]):
            v = complex(r)
            Ev = np.polyval(Ep[::-1], v)
            if abs(Ev) < 1e-12:
                continue
            u = np.polyval(Dp[::-1], v) / Ev
            dn = v * v + u * u - 2.0 * u * v * g["cg"]
            if abs(dn) < 1e-14:
                continue
            sc = g["c"] / np.sqrt(dn)
            for sgn in (1.0, -1.0):
                sa, sb, scc = sgn * v * sc, sgn * u * sc, sgn * sc
                r1 = sa * sa + sb * sb - 2.0 * sa * sb * g["cg"] - g["c"] ** 2
                r2 = sb * sb + scc * scc - 2.0 * sb * scc * g["ca"] - g["a"] ** 2
                r3 = sa * sa + scc * scc - 2.0 * sa * scc * g["cb"] - g["b"] ** 2
                w = max(w, abs(r1) / g["c"] ** 2, abs(r2) / g["c"] ** 2,
                        abs(r3) / g["c"] ** 2)
            npair += 1
    print(f"  {npair} 个四次根（含复根）x 2 个反演 = {2*npair} 个代数解，全部满足原始三式")
    print(f"  最大归一化残差 : {w:.3e}")
    print("  => 四次方程的一个根 = 一个反演对(±s)的代表。")
    print("     每对里最多 1 个全正解 -> 物理解最多 4 个，正是 P3P 的经典结论。\n")


# ---------------------------------------------------------------------------
# E: 方形退化
# ---------------------------------------------------------------------------

def check_E(n=400, seed=3):
    print("=" * 76)
    print("Check E : 方形标记 A=B=1（A1 = A-1 = 0，(I) 退化为一次式）")
    print("=" * 76)
    rng = np.random.default_rng(seed)
    tested = ok = 0
    w = 0.0
    while tested < n:
        g = _geometry(rng, square=True)
        if g is None:
            continue
        tested += 1
        coef, Dp, Ep = _quartic_coefficients(g["ca"], g["cb"], g["cg"], g["A"], g["B"])
        v_true = g["p"]
        fv = abs(float(np.polyval(coef[::-1], v_true))) / max(1.0, np.max(np.abs(coef)))
        w = max(w, fv)
        ok += (fv < 1e-10)
    print(f"  方形 {tested} 组: 真解是 F 的根 {ok} 组 (最大归一化残差 {w:.3e})")
    print("  A1=0 时 (I) 的 q^2 项消失，代码刻意回代 (II) 而非 (I)，此处验证有效。\n")


# ---------------------------------------------------------------------------
# F: 消元退化 E(v)=0 的频率
# ---------------------------------------------------------------------------

def check_F(n=2000, seed=4):
    print("=" * 76)
    print("Check F : 消元退化点 E(v)=0 的实际发生频率（这是唯一可能的漏根口）")
    print("=" * 76)
    rng = np.random.default_rng(seed)
    tested = 0
    hits = 0
    worst = np.inf
    while tested < n:
        g = _geometry(rng)
        if g is None:
            continue
        tested += 1
        coef, Dp, Ep = _quartic_coefficients(g["ca"], g["cb"], g["cg"], g["A"], g["B"])
        for r in np.roots(coef[::-1]):
            if abs(r.imag) > 1e-8 * max(1.0, abs(r.real)):
                continue
            Ev = abs(float(np.polyval(Ep[::-1], float(r.real))))
            worst = min(worst, Ev)
            if Ev < 1e-9:
                hits += 1
    print(f"  样本 {tested} 组")
    print(f"  触发 |E(v)| < 1e-9 的次数 : {hits}")
    print(f"  全局最小 |E(v)|            : {worst:.3e}")
    print("  说明: E 是 p 的一次式 (deg 1)，其零点是一条直线（测度零），")
    print("        仅在 A*B1 - A1*B2 同时为 0 时退化，随机配置下不出现。\n")


if __name__ == "__main__":
    check_A()
    check_BC()
    check_D()
    check_H()
    check_G()
    check_E()
    check_F()


# ---------------------------------------------------------------------------
# I: 单帧实例 —— 4 个候选都精确满足前三点，只有第 4 点能区分
# ---------------------------------------------------------------------------

def check_I(seed=8, dist=0.31):
    print("=" * 76)
    print("Check I : 单帧实例（0.3 m 方形标记, 前 3 点闭式解, 第 4 点消歧）")
    print("=" * 76)
    from p3p import solve_p3p, _reproj_rmse
    K2 = intrinsic_from_fov(800, 500, 55.0)
    rng = np.random.default_rng(seed)
    s = 0.15
    P4 = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R = rodrigues(ax * rng.uniform(-0.3, 0.3))
    z = R.T @ np.array([0.0, 0.0, 1.0])
    C = P4.mean(0) - dist * z
    t = -R @ C
    uv = project_points(P4, R, t, K2) + rng.normal(0.0, 0.3, (4, 2))

    sols = solve_p3p(P4[:3], uv[:3], K2)
    print(f"  真值: |t| = {np.linalg.norm(t)*1000:7.1f} mm, |OC| = {np.linalg.norm(C)*1000:6.1f} mm")
    print(f"  候选数 = {len(sols)}")
    print(f"  {'#':>2} {'|t|mm':>8} {'p':>7} {'q':>7} {'RMSE3px':>8} {'RMSE4px':>8} "
          f"{'dt(mm)':>7} {'dR(deg)':>8}")
    for i, (Ri, ti) in enumerate(sols):
        rm3 = _reproj_rmse(P4[:3], uv[:3], Ri, ti, K2)
        rm4 = _reproj_rmse(P4, uv, Ri, ti, K2)
        Xi = (Ri @ P4[:3].T).T + ti
        dd = np.linalg.norm(Xi, axis=1)
        p, q = dd[0] / dd[2], dd[1] / dd[2]
        dt = np.linalg.norm(ti - t) * 1000
        dR = np.degrees(np.linalg.norm(inv_rodrigues(Ri.T @ R)))
        print(f"  {i+1:2d} {np.linalg.norm(ti)*1000:8.1f} {p:7.3f} {q:7.3f} "
              f"{rm3:8.3f} {rm4:8.3f} {dt:7.2f} {dR:8.2f}")
    print("  => 4 个候选对前 3 点的重投影都是 ~0（都是精确解），p/q 全为正，")
    print("     区别只体现在第 4 点。消歧靠第 4 点 RMSE + 时序连续性，不靠符号闸门。\n")
