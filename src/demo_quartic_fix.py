"""
demo_quartic_fix.py
-------------------
现场演示四次方程常数项 -A -> -1 的修复:
同一组非方形三点, 用旧/新系数构造多项式, 把"真值距离比 v=s_a/s_c"
代入, 看残差差几个数量级。

运行: python demo_quartic_fix.py
"""

import numpy as np

from pose_utils import intrinsic_from_fov, project_points

K = intrinsic_from_fov(800, 500, 60.0)


def coeffs(cos_alpha, cos_beta, cos_gamma, A, B, old_bug):
    """旧代码把 C1 写成 [-A, 0, A]; 正确是 [-1, 0, A]。"""
    A1 = np.array([A - 1.0])
    B1 = np.array([2.0 * cos_alpha, -2.0 * A * cos_gamma])
    C1 = np.array([-A, 0.0, A]) if old_bug else np.array([-1.0, 0.0, A])
    A2 = np.array([B])
    B2 = np.array([0.0, -2.0 * B * cos_gamma])
    C2 = np.array([-1.0, 2.0 * cos_beta, B - 1.0])
    D = np.convolve(A1, C2) - np.convolve(A2, C1)
    E = np.convolve(A2, B1) - np.convolve(A1, B2)
    F = (B * np.convolve(D, D)
         + np.convolve(B2, np.convolve(D, E))
         + np.convolve(C2, np.convolve(E, E)))
    return F[:5]


def normalize_residual(c, v):
    """|F(v)| / 各项量级最大值, 无量纲。"""
    terms = np.abs(c * v ** np.arange(len(c)))
    return float(abs(np.polyval(c[::-1], v)) / max(terms.max(), 1e-300))


def run(name, P, R, t):
    uv = project_points(P, R, t, K)
    rays = (np.linalg.inv(K) @ np.hstack([uv, np.ones((3, 1))]).T).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    ra, rb, rc = rays
    ca, cb, cg = rb @ rc, ra @ rc, ra @ rb
    a = np.linalg.norm(P[1] - P[2])       # |BC|
    b = np.linalg.norm(P[0] - P[2])       # |AC|
    c = np.linalg.norm(P[0] - P[1])       # |AB|
    A, B = (a / c) ** 2, (b / c) ** 2

    C = -R.T @ t                          # 相机中心 (世界系)
    s = np.linalg.norm(P - C, axis=1)      # 三点到相机中心的真实距离
    v_true = s[0] / s[2]                  # s_a / s_c

    c_new = coeffs(ca, cb, cg, A, B, old_bug=False)
    c_old = coeffs(ca, cb, cg, A, B, old_bug=True)
    r_new = normalize_residual(c_new, v_true)
    r_old = normalize_residual(c_old, v_true)

    roots_new = np.roots(c_new[::-1])
    roots_old = np.roots(c_old[::-1])
    ok_new = min(abs(roots_new - v_true))
    ok_old = min(abs(roots_old - v_true))

    print(f"\n[{name}]  A=(a/c)^2={A:.4f}  B=(b/c)^2={B:.4f}  v_true={v_true:.6f}")
    print(f"  新系数 [-1,0,A]  归一化残差 {r_new:.2e}   最近根距 {ok_new:.2e}")
    print(f"  旧系数 [-A,0,A]  归一化残差 {r_old:.2e}   最近根距 {ok_old:.2e}")
    print(f"  两套系数是否逐项相同: {np.allclose(c_new, c_old)}")
    return r_new, r_old


def main():
    R = np.eye(3)
    t = np.array([0.02, -0.01, 0.80])

    # 1) 方形标记: A=1 => -A ≡ -1, bug 隐身
    sq = np.array([[-0.10, 0.10, 0], [0.10, 0.10, 0],
                   [0.10, -0.10, 0]], float)
    run("方形三点 A=1", sq, R, t)

    # 2) 矩形标记: 长 0.30 x 宽 0.15, 点序取 |BC|=0.30 (长)、|AB|=0.15 (宽)
    #    => A=(a/c)^2=(0.30/0.15)^2=4
    rect = np.array([[0.15, 0.075, 0], [0.15, -0.075, 0],
                     [-0.15, -0.075, 0]], float)
    run("矩形三点 A=4", rect, R, t)

    # 3) 任意不等边三点 (非共面)
    tri = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.02],
                    [0.05, 0.22, -0.03]], float)
    run("不等边三点", tri, R, t)

    print("\n结论: A=1 时两套系数逐项相同 -> 方形实验无法暴露;")
    print("      A!=1 时旧系数使真值不再是根, 位姿必然错。")


if __name__ == "__main__":
    main()
