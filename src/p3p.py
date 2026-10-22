"""
p3p.py
------
自主实现的 P3P 闭式求解器 (纯 NumPy, 无第三方几何库)。

方法: 经典 Grunert 式解法 —— 将三个余弦定理方程通过 结式消元 (resultant
elimination) 化为一个关于距离比 v = s_a/s_c 的一元四次方程, 解出所有距离
候选后用三点正交基直接恢复旋转 (无 SVD), 最后用第 4 点重投影误差消歧。

这正是计算代数几何中的 "符号消元 + 数值求根" 混合范式:
    F(v) = A1(v) D(v)^2 - B1(v) D(v) E(v) + C1(v) E(v)^2 = 0 (四次)

接口:
    solve_p3p(pts_w, uv, K, ...) -> list[(R, t, err)]
    solve_p3p_robust(...)        -> 带 LM 精化的最终解 / 奇异性检测
"""

import numpy as np

from pose_utils import (intrinsic_from_fov, project_points, rodrigues,
                        inv_rodrigues, rigid_transform, reproj_error)


# ---------------------------------------------------------------------------
# 核心: Grunert 式闭式解
# ---------------------------------------------------------------------------

def _quartic_coefficients(cos_alpha, cos_beta, cos_gamma, A, B):
    """
    构造 Grunert 四次方程的系数 (v 的升幂)。

    A = (a/c)^2, B = (b/c)^2, a=|BC|, b=|AC|, c=|AB|
    返回 (coef, D, E):
      coef: 长度 5, 升幂 [c0, c1, c2, c3, c4], 即 F(v)=sum coef[k] v^k
      D, E: 消元得到的分母/线性多项式 (同为升幂系数), u = D(v)/E(v)

    说明: 用 np.convolve 做多项式乘法 (本机实测约 3.9 us/次, 为
    np.polynomial.polymul 的 1/8)。这一步省的是 Python 层对象开销,
    量级上不改变算法复杂度。
    """
    # 两条二次式 (以 u 为未知量, 系数是 v 的多项式, 升幂):
    # (I):  A1 u^2 + B1 u + C1 = 0
    #   A1 = A - 1 (常数), B1 = 2cos(a) - 2A cos(g) v, C1 = A v^2 - 1
    # (II): A2 u^2 + B2 u + C2 = 0
    #   A2 = B (常数), B2 = -2B cos(g) v, C2 = (B-1) v^2 + 2cos(b) v - 1
    #
    # 消元 (线性组合消去 u^2, 再回代到方程 (II)):
    #   D(v) = A1 C2 - A2 C1      (deg 2)
    #   E(v) = A2 B1 - A1 B2      (deg 1)
    #   u = D / E
    #   F(v) = A2 D^2 + B2 D E + C2 E^2 = 0   (deg 4)
    #
    # 必须回代到 (II) 而非 (I): 方阵标记 A=(a/c)^2=1 使 A1=0, (I) 退化并与
    # 消元组合线性相关, 回代 (I) 得恒零多项式; 而 A2=B>0 恒成立, 回代 (II)
    # 永不退化。
    A1 = np.array([A - 1.0])
    B1 = np.array([2.0 * cos_alpha, -2.0 * A * cos_gamma])
    C1 = np.array([-1.0, 0.0, A])
    A2 = np.array([B])
    B2 = np.array([0.0, -2.0 * B * cos_gamma])
    C2 = np.array([-1.0, 2.0 * cos_beta, B - 1.0])

    D = np.convolve(A1, C2) - np.convolve(A2, C1)
    E = np.convolve(A2, B1) - np.convolve(A1, B2)
    F = (B * np.convolve(D, D)
         + np.convolve(B2, np.convolve(D, E))
         + np.convolve(C2, np.convolve(E, E)))

    coef = np.zeros(5)
    n = min(5, len(F))
    coef[:n] = F[:n]
    return coef, D, E


def _frame_rotation(src, dst):
    """
    三点构型下的刚体旋转闭式解 (正交基对齐, 无 SVD)。

    用 p1->p2 方向与 p1p2 x p1p3 法向构造右手正交基, 两个坐标系间的
    旋转即 R = B_dst @ B_src^T。三点共线时返回 None。

    注意: 不要声称它"比 Kabsch 快 15 倍"。实测三点构型下两者同量级
    (153 us vs 142 us, 3x3 的 SVD 本来就便宜)。真正的理由是**确定性**:
    构造式方法给出固定的符号约定, 不会因 SVD 的反射/符号不确定而返回
    镜像解; 同时不依赖 LAPACK 的 SVD 实现。
    """
    e1s = src[1] - src[0]
    e1d = dst[1] - dst[0]
    ns2 = float(e1s @ e1s)
    nd2 = float(e1d @ e1d)
    if ns2 < 1e-24 or nd2 < 1e-24:
        return None
    ns = np.sqrt(ns2)
    nd = np.sqrt(nd2)
    e1s = e1s / ns
    e1d = e1d / nd
    a = src[2] - src[0]
    b = dst[2] - dst[0]
    # 手写叉积 (np.cross 的 Python 层开销约为其 10 倍)
    e2s = np.array([a[1] * e1s[2] - a[2] * e1s[1],
                    a[2] * e1s[0] - a[0] * e1s[2],
                    a[0] * e1s[1] - a[1] * e1s[0]])
    e2d = np.array([b[1] * e1d[2] - b[2] * e1d[1],
                    b[2] * e1d[0] - b[0] * e1d[2],
                    b[0] * e1d[1] - b[1] * e1d[0]])
    n2s = np.sqrt(float(e2s @ e2s))
    n2d = np.sqrt(float(e2d @ e2d))
    if n2s < 1e-12 or n2d < 1e-12:
        return None                     # 共线退化
    e2s = e2s / n2s
    e2d = e2d / n2d
    Bs = np.empty((3, 3))
    Bd = np.empty((3, 3))
    Bs[:, 0], Bd[:, 0] = e1s, e1d
    Bs[:, 1], Bd[:, 1] = e2s, e2d
    Bs[:, 2] = np.array([e1s[1] * e2s[2] - e1s[2] * e2s[1],
                         e1s[2] * e2s[0] - e1s[0] * e2s[2],
                         e1s[0] * e2s[1] - e1s[1] * e2s[0]])
    Bd[:, 2] = np.array([e1d[1] * e2d[2] - e1d[2] * e2d[1],
                         e1d[2] * e2d[0] - e1d[0] * e2d[2],
                         e1d[0] * e2d[1] - e1d[1] * e2d[0]])
    return Bd @ Bs.T


def _rays_from_pixels(uv, K):
    """像素 -> 相机系单位射线 (3,N)。对常见 K 走解析快路径。"""
    if (K[0, 1] == 0.0 and K[1, 0] == 0.0 and K[2, 0] == 0.0
            and K[2, 1] == 0.0 and K[2, 2] == 1.0):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (uv[:, 0] - cx) / fx
        y = (uv[:, 1] - cy) / fy
        r = np.stack([x, y, np.ones_like(x)], axis=1)
    else:
        uv_h = np.hstack([uv, np.ones((uv.shape[0], 1))])
        r = (np.linalg.inv(K) @ uv_h.T).T
    return (r / np.linalg.norm(r, axis=1, keepdims=True)).T


def solve_p3p(pts_w, uv, K, max_reproj_err=None):
    """
    P3P 闭式求解。

    参数:
        pts_w: (3,3) 三个已知 3D 点 (标记系/世界系)
        uv:    (3,2) 三个点在图像中的像素坐标
        K:     (3,3) 内参矩阵
        max_reproj_err: 若给定, 过滤重投影误差过大的解

    返回:
        solutions: list[(R, t)]  按重投影误差升序, R/t 为 world->camera
    """
    pts_w = np.asarray(pts_w, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)

    # 1) 归一化射线方向 (相机系, z 朝前)
    rays = _rays_from_pixels(uv, K).T            # (3,3), 每行一条射线
    ra, rb, rc = rays

    # 2) 射线两两夹角余弦 (相机中心处的三个角)
    #    alpha: 射线 B-OC, beta: 射线 A-OC, gamma: 射线 A-OB
    cos_alpha = float(np.dot(rb, rc))
    cos_beta = float(np.dot(ra, rc))
    cos_gamma = float(np.dot(ra, rb))

    # 3) 边长 (标记系): a=|BC| 对面 A, b=|AC| 对面 B, c=|AB| 对面 C
    a = np.linalg.norm(pts_w[1] - pts_w[2])
    b = np.linalg.norm(pts_w[0] - pts_w[2])
    c = np.linalg.norm(pts_w[0] - pts_w[1])
    if c < 1e-12:
        return []
    A = (a / c) ** 2
    B = (b / c) ** 2

    # 4) 结式消元 -> 一元四次方程 F(v)=0, v = s_a/s_c
    #    coef 为升幂, np.roots 需要降幂输入
    coef, Dp, Ep = _quartic_coefficients(cos_alpha, cos_beta, cos_gamma, A, B)
    roots = np.roots(coef[::-1])

    solutions = []
    for v in roots:
        if abs(v.imag) > 1e-8 * max(1.0, abs(v.real)):
            continue
        v = float(v.real)
        if v <= 0:
            continue                      # 距离比必须为正
        # D/E 为升幂系数, np.polyval 需降幂
        E_v = float(np.polyval(Ep[::-1], v))
        D_v = float(np.polyval(Dp[::-1], v))
        if abs(E_v) < 1e-9:
            continue                      # 近奇异消元方向, 交给 LM 分支
        u = D_v / E_v                     # u = s_b/s_c (u = D/E, 见 _quartic_coefficients)
        if u <= 0:
            continue
        denom = v * v + u * u - 2.0 * u * v * cos_gamma
        if denom <= 1e-12:
            continue
        s_c2 = (c * c) / denom
        if s_c2 <= 0:
            continue
        s_c = np.sqrt(s_c2)
        s_a = v * s_c
        s_b = u * s_c

        # 5) 相机系下的三个 3D 点 = 距离 * 单位射线
        pts_c = np.stack([s_a * ra, s_b * rb, s_c * rc], axis=0)

        # 6) 三点正交基直接恢复 world->camera 刚体变换 (免 SVD)
        R = _frame_rotation(pts_w, pts_c)
        if R is None:
            continue
        t = pts_c.mean(axis=0) - R @ pts_w.mean(axis=0)

        # 数值验证: 重投影应一致 (内联, 避免重复投影开销)
        Xc = (R @ pts_w.T).T + t
        if np.any(Xc[:, 2] <= 1e-9):
            continue
        uv_h = (K @ (Xc / Xc[:, 2:3]).T).T[:, :2]
        err = float(np.mean(np.linalg.norm(uv_h - uv, axis=1)))
        if max_reproj_err is not None and err > max_reproj_err:
            continue
        solutions.append((R, t, err))

    solutions.sort(key=lambda s: s[2])
    return [(R, t) for (R, t, _) in solutions]


# ---------------------------------------------------------------------------
# 奇异性检测
# ---------------------------------------------------------------------------

def singularity_measure(pts_w, uv, K):
    """
    估计当前构型距离最近 P3P 奇异流形的程度。

    原理: P3P 的经典奇异构型 (danger cylinder / 解对并) 表现为四次方程出现
    重根。用最小正实根间距作为度量, 间距越小越接近奇异。

    返回:
        min_gap: 最小正实根间距 (无量纲), ~0 表示近奇异
        n_valid: 有效的实根数
    """
    pts_w = np.asarray(pts_w, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)

    rays = _rays_from_pixels(uv, K).T
    ra, rb, rc = rays

    cos_alpha = float(np.dot(rb, rc))
    cos_beta = float(np.dot(ra, rc))
    cos_gamma = float(np.dot(ra, rb))

    a = np.linalg.norm(pts_w[1] - pts_w[2])
    b = np.linalg.norm(pts_w[0] - pts_w[2])
    c = np.linalg.norm(pts_w[0] - pts_w[1])
    A = (a / c) ** 2
    B = (b / c) ** 2

    coef, _, _ = _quartic_coefficients(cos_alpha, cos_beta, cos_gamma, A, B)
    roots = np.roots(coef[::-1])

    real_pos = sorted(
        r.real for r in roots
        if abs(r.imag) <= 1e-6 * max(1.0, abs(r.real)) and r.real > 1e-8
    )
    if len(real_pos) < 2:
        return float("inf"), len(real_pos)

    gaps = [real_pos[i + 1] - real_pos[i] for i in range(len(real_pos) - 1)]
    return float(min(gaps)), len(real_pos)


# ---------------------------------------------------------------------------
# LM 精化 (近奇异 / 精度兜底)
# ---------------------------------------------------------------------------

def _reproj_residual(x, pts_w, uv, K):
    R = rodrigues(x[:3])
    t = x[3:]
    uv_hat = project_points(pts_w, R, t, K)
    return (uv_hat - uv).reshape(-1)


def _exp_small(w):
    """小角度轴角 -> 旋转矩阵 (二阶近似, 比通用 Rodrigues 快 ~5x)。"""
    th2 = float(w @ w)
    Kx = np.array([[0.0, -w[2], w[1]],
                   [w[2], 0.0, -w[0]],
                   [-w[1], w[0], 0.0]])
    if th2 < 1e-12:
        return np.eye(3) + Kx + 0.5 * (Kx @ Kx)
    th = np.sqrt(th2)
    return (np.eye(3) + (np.sin(th) / th) * Kx
            + ((1.0 - np.cos(th)) / th2) * (Kx @ Kx))


def refine_gn(pts_w, uv, K, R0, t0, iters=6, lam0=1e-3):
    """
    快速 Levenberg-Marquardt 重投影最小化 (纯 NumPy 解析雅可比)。

    与 scipy.optimize.least_squares 等价但去除了 Python 层开销:
      - 旋转用矩阵参数化 + 小角度增量更新, 免去每步 Rodrigues 展开
      - 残差/雅可比按 (N,2,6) 批量装配, 一次 einsum 完成

    耗时: 4 点 6 次迭代本机实测 **~1.8 ms** (随机器状态波动 ±30%)。
    一次 solve_p3p_robust 最多调用 3 次, 故完整管线 ~5~8 ms。
    (早期文档写的 "~150 us" 是错的, 差一个数量级, 已更正。)

    参数化: 增量 x = [dθ(3), dt(3)], 左乘更新 R <- exp([dθ]x) R。
    返回 (R, t, rmse); 失败时返回 None。
    """
    pts_w = np.asarray(pts_w, float)
    pt = pts_w.T                            # (3,N) 只转置一次
    uv = np.asarray(uv, float)
    N = pts_w.shape[0]
    R = np.asarray(R0, float)
    t = np.asarray(t0, float).reshape(3).copy()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    lam = lam0
    cost = None

    def _project(Rm, tm):
        Xc = Rm @ pt + tm[:, None]          # (3,N)
        z = Xc[2]
        if z.min() <= 1e-9:
            return None
        iz = 1.0 / z
        uvh = np.empty((N, 2))
        uvh[:, 0] = fx * Xc[0] * iz + cx
        uvh[:, 1] = fy * Xc[1] * iz + cy
        return uvh, Xc, iz

    for _ in range(iters):
        out = _project(R, t)
        if out is None:
            return None
        uvh, Xc, iz = out
        r = (uvh - uv).ravel()
        c = float(r @ r)
        if c < 1e-16:
            break
        if cost is not None and abs(cost - c) <= 1e-9 * cost:
            break
        cost = c

        # 解析雅可比 (像素): d(u,v)/dX_c = [[fx/z, 0, -fx*x/z^2],
        #                                    [0, fy/z, -fy*y/z^2]]
        Jp = np.zeros((N, 2, 3))
        Jp[:, 0, 0] = fx * iz
        Jp[:, 0, 2] = -fx * Xc[0] * iz * iz
        Jp[:, 1, 1] = fy * iz
        Jp[:, 1, 2] = -fy * Xc[1] * iz * iz
        # 左乘旋转增量 => dX_c/dθ = -[R x_w]x
        RX = (Xc - t[:, None]).T                       # (N,3)
        J = np.zeros((N, 2, 6))
        J[:, :, :3] = -np.einsum("nij,njk->nik", Jp, _skew_batch(RX))
        J[:, :, 3:] = Jp
        J = J.reshape(-1, 6)

        H = J.T @ J
        g = J.T @ r
        ok_step = False
        small = False
        for _ in range(6):
            try:
                dx = np.linalg.solve(H + lam * np.diag(np.diag(H))
                                     + 1e-12 * np.eye(6), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            R_new = _exp_small(dx[:3]) @ R
            t_new = t + dx[3:]
            out_new = _project(R_new, t_new)
            if out_new is None:
                lam *= 10.0
                continue
            r_new = (out_new[0] - uv).ravel()
            c_new = float(r_new @ r_new)
            if c_new < c:
                R, t = R_new, t_new
                lam = max(lam * 0.3, 1e-9)
                ok_step = True
                # 步长收敛: 位姿增量足够小即可停止
                small = (dx[:3] @ dx[:3] < 1e-14
                         and dx[3:] @ dx[3:] < 1e-14)
                break
            lam *= 10.0
        if not ok_step or small:
            break

    out = _project(R, t)
    if out is None:
        return None
    err = float(np.sqrt(np.mean(np.sum((out[0] - uv) ** 2, axis=1))))
    return R, t, err


def _skew_batch(v):
    """(N,3) -> (N,3,3) 反对称矩阵批处理。"""
    out = np.zeros((v.shape[0], 3, 3))
    out[:, 0, 1] = -v[:, 2]
    out[:, 0, 2] = v[:, 1]
    out[:, 1, 0] = v[:, 2]
    out[:, 1, 2] = -v[:, 0]
    out[:, 2, 0] = -v[:, 1]
    out[:, 2, 1] = v[:, 0]
    return out


def refine_lm(pts_w, uv, K, R0, t0, max_nfev=200):
    """
    重投影最小化精化 (快速 LM/Gauss-Newton, 解析雅可比)。

    保持与旧 scipy 版本相同的接口; 内部使用 refine_gn 纯 NumPy 实现。
    名义上是"LM", 实际是"λ 自适应衰减的 GN": 步长被接受时 λ 下降,
    λ 很小时退化为高斯-牛顿, 只在失败分支才靠 λ 增大来挽救。
    """
    out = refine_gn(pts_w, uv, K, R0, t0)
    if out is None:
        return None
    return out


# ---------------------------------------------------------------------------
# 对外主接口: P3P + 多候选消歧 + LM
# ---------------------------------------------------------------------------

def _reproj_rmse(pts_w, uv, R, t, K):
    """重投影均方根误差 (px)。

    必须与 refine_gn 的返回值同口径 —— pose_utils.reproj_error 返回的是
    "逐点 L2 范数的均值", 恒 <= RMSE (Jensen)。用它和 refine_gn 的 RMSE
    比较, 会系统性低估未精化解的误差, 导致 "err2 < err" 判据把真正更好的
    LM 解丢掉 (实测: 方形标记 0.3px 下平移中位 2.40mm vs 2.28mm)。
    """
    Xc = (R @ np.asarray(pts_w, float).T).T + np.asarray(t, float).reshape(3)
    uv_hat = (K @ (Xc / Xc[:, 2:3]).T).T[:, :2]
    return float(np.sqrt(np.mean(np.sum((uv_hat - uv) ** 2, axis=1))))


def solve_p3p_robust(pts_w, uv, K, use_lm=True, reproj_thresh=2.0,
                     R_prev=None, t_prev=None):
    """
    工程可用的 P3P 求解入口。

    流程:
      1) 闭式解给出至多 4 组候选位姿
      2) 按 4 点重投影 RMSE 排序, **前 3 个候选**用全部 N 点做 LM 精化
         (不是全部候选; 3 个已足够覆盖正确分支, 且把耗时从 4 次 LM 降到 3 次)
      3) 消歧: 无先验取重投影最小者; 给定 (R_prev, t_prev) 时用时序
         连续性选支 (平面目标二义分支的重投影几乎相同)。注意 R_prev/t_prev
         只用于**选支**, 不会作为 LM 的初值 —— LM 初值恒为闭式解候选。

    参数:
        pts_w: (N,3) N>=3, 前 3 个用于闭式解, 其余用于消歧
        uv:    (N,2)
        K:     (3,3)
        reproj_thresh: 像素。**当前仅占位, 返回前未用它做拒绝门限** ——
                  success=True 只表示"产生了可用解", 不表示"通过了该阈值"。
                  (若要启用, 需同时改基准的失败判据, 见 README 的措辞边界)
        R_prev, t_prev: 上一帧位姿 (world->camera), 仅用于时序选支

    返回:
        dict(success, R, t, reproj_err, n_candidates, used_lm, singular, singular_gap)
    """
    pts_w = np.asarray(pts_w, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    pts3, uv3 = pts_w[:3], uv[:3]

    cands = solve_p3p(pts3, uv3, K)
    singular = False
    gap = np.inf

    # 奇异性度量: 四次方程最小正实根间距 (danger cylinder 判据)
    if not cands:
        singular = True
    else:
        gap, _ = singularity_measure(pts3, uv3, K)
        if gap < 1e-3:                    # 根间距过小 => 解对并 => 近奇异
            singular = True

    # 只对原始重投影误差最小的 3 个候选做精化 (足够覆盖正确分支,
    # 且把耗时从 4 次 LM 降到 3 次)
    ranked = sorted(
        ((_reproj_rmse(pts_w, uv, R0, t0, K), R0, t0) for (R0, t0) in cands),
        key=lambda x: x[0])
    evaluated = []                        # (rmse, R, t, used_lm)
    for idx, (err0, R0, t0) in enumerate(ranked):
        R, t, err = R0, t0, err0
        lm_used = False
        if use_lm and idx < 3:
            out = refine_gn(pts_w, uv, K, R0, t0)
            if out is not None and out[0] is not None:
                R2, t2, err2 = out
                # 同口径比较 (两者都是 RMSE), 否则 LM 的收益会被吞掉
                if err2 < err:
                    R, t, err, lm_used = R2, t2, err2, True
        evaluated.append((err, R, t, lm_used))

    if use_lm and not evaluated:
        # 无候选: EPnP 初值兜底
        init = None
        try:
            import cv2
            ok, rvec, tvec = cv2.solvePnP(
                pts_w.astype(np.float64), uv.astype(np.float64),
                K.astype(np.float64), None, flags=cv2.SOLVEPNP_EPNP)
            if ok:
                init = (cv2.Rodrigues(rvec)[0], tvec.reshape(3))
        except Exception:
            init = None
        if init is not None:
            out = refine_gn(pts_w, uv, K, init[0], init[1])
            if out is not None and out[0] is not None:
                R, t, err = out
                evaluated.append((err, R, t, True))

    if not evaluated:
        return dict(success=False, R=None, t=None, reproj_err=np.inf,
                    n_candidates=0, used_lm=False, singular=singular,
                    singular_gap=gap)

    # 消歧策略:
    #   无先验   -> 取重投影误差最小者 (单帧模式)
    #   有先验   -> 平面目标的二义分支重投影几乎相同 (方形对称目标第 4 点
    #               无法消歧), 此处以时序连续性为准: 在重投影可接受的候选
    #               中选与上一帧位姿最近者。这是真实伺服系统的标准做法。
    evaluated.sort(key=lambda x: x[0])
    if R_prev is not None:
        R_prev = np.asarray(R_prev, float)
        t_prev = np.asarray(t_prev, float).reshape(3)
        best_err = evaluated[0][0]
        ok_pool = [e for e in evaluated
                   if e[0] <= max(best_err + 1.0, 1.5 * best_err)]
        def _pose_dist(item):
            R_i, t_i = item[1], item[2]
            d_t = np.linalg.norm(t_i - t_prev)
            d_R = np.linalg.norm(inv_rodrigues(R_i.T @ R_prev))
            return d_t + 0.1 * d_R
        pick = min(ok_pool, key=_pose_dist)
    else:
        pick = evaluated[0]
    err, R, t, lm_any = pick
    return dict(success=True, R=R, t=t, reproj_err=err,
                n_candidates=len(cands), used_lm=lm_any, singular=singular,
                singular_gap=gap)
