"""
trace_pose.py
-------------
单帧位姿估计全链路追踪（面试核查用）。

把一帧里"点从哪来、用几个、候选怎么选、误差多大"逐项打印出来, 便于
对着 p3p.py / scene.py 逐行核对。不参与任何实验结论。

运行: python trace_pose.py [--dist 0.6] [--seed 1]
"""

import argparse

import cv2
import numpy as np

import scene as S
from pose_utils import (intrinsic_from_fov, inv_rodrigues, project_points)
from p3p import (refine_gn, singularity_measure, solve_p3p, solve_p3p_robust,
                 _reproj_rmse)


def detect(rgb):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    det = cv2.aruco.ArucoDetector(d, params)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if not corners:
        return None, None
    return corners[0].reshape(4, 2).astype(np.float64), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=float, default=0.6)
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    W, H = 480, 300
    K = intrinsic_from_fov(W, H, args.fov)

    S.init_pybullet(gui=False)
    S.build_scene()

    # 与 main_pbvs.initial_pose 同构, 但不加随机横向/角度偏差, 便于人工核对
    rng = np.random.default_rng(args.seed)
    c_m = S.MARKER_CENTER
    cam_pos = np.array([rng.uniform(-0.35, 0.35),
                        c_m[1] - args.dist,
                        c_m[2] + rng.uniform(-0.25, 0.25)])
    R_gt, t_gt = S.look_at_pose(cam_pos, c_m)

    rgb, _, _ = S.render(R_gt, t_gt, width=W, height=H, fov=args.fov)
    uv, ids = detect(rgb)
    if uv is None:
        print("检测失败")
        return

    pts_w = S.to_world(S.PTS_M)          # (4,3) 世界系, 由标记模型直接给出

    print("=" * 68)
    print(f"1) 世界点 (由 scene.PTS_M 经 R_WM/MARKER_CENTER 变换, 每帧恒定)")
    print(f"   标记印刷边长 {S.PRINTED_SIZE} m, 半边长 {S.PHALF} m")
    for k in range(4):
        print(f"   P{k} = [{pts_w[k,0]: .4f}, {pts_w[k,1]: .4f}, {pts_w[k,2]: .4f}]")

    print(f"\n2) 图像点 (ArUco id={None if ids is None else ids.ravel()[0]}, "
          f"{W}x{H}, fov={args.fov})")
    for k in range(4):
        print(f"   p{k} = [{uv[k,0]:8.2f}, {uv[k,1]:8.2f}] px")

    # 对应关系核查: 4 种循环顺序
    uv_pred = project_points(pts_w, R_gt, t_gt, K)
    print(f"\n3) 对应关系核查 (检测点 vs 解析投影)")
    for k in range(4):
        idx = [(i + k) % 4 for i in range(4)]
        e = float(np.abs(uv - uv_pred[idx]).mean())
        tag = "  <-- 采用" if k == 0 else ""
        print(f"   循环偏移 {k}: mean|duv| = {e:7.2f} px{tag}")

    # 闭式解: 只用前 3 点
    print(f"\n4) 闭式解 (仅前 3 点 P0/P1/P2)")
    cands = solve_p3p(pts_w[:3], uv[:3], K)
    gap, n_real = singularity_measure(pts_w[:3], uv[:3], K)
    print(f"   根间距 min_gap = {gap:.4f} (实根数 {n_real}), 候选数 = {len(cands)}")

    ranked = sorted(((_reproj_rmse(pts_w, uv, R0, t0, K), R0, t0)
                     for (R0, t0) in cands), key=lambda x: x[0])
    print(f"\n5) 候选按 4 点重投影 RMSE 排序 (+ LM, 仅前 3 个)")
    print(f"   {'#':>2} {'RMSE前':>9} {'RMSE后':>9} {'Δt mm':>8} {'ΔC mm':>8} "
          f"{'Δθ °':>7}  用LM")
    best = None
    for i, (err0, R0, t0) in enumerate(ranked[:4]):
        R1, t1, err1 = R0, t0, err0
        used = False
        if i < 3:
            out = refine_gn(pts_w, uv, K, R0, t0)
            if out is not None and out[0] is not None and out[2] < err0:
                R1, t1, err1 = out[0], out[1], out[2]
                used = True
        dt = float(np.linalg.norm(t1 - t_gt)) * 1e3
        c_est, c_gt = -R1.T @ t1, -R_gt.T @ t_gt
        dc = float(np.linalg.norm(c_est - c_gt)) * 1e3
        dR = float(np.degrees(np.linalg.norm(inv_rodrigues(R1.T @ R_gt))))
        print(f"   {i:>2} {err0:9.4f} {err1:9.4f} {dt:8.2f} {dc:8.2f} "
              f"{dR:7.3f}  {int(used)}")
        best = (err1, R1, t1)

    print(f"\n6) 工程入口 solve_p3p_robust (无时序先验)")
    out = solve_p3p_robust(pts_w, uv, K, use_lm=True)
    R, t = out["R"], out["t"]
    print(f"   success={out['success']} n_candidates={out['n_candidates']} "
          f"used_lm={out['used_lm']} singular={out['singular']}")
    print(f"   reproj = {out['reproj_err']:.4f} px")

    c_est, c_gt = -R.T @ t, -R_gt.T @ t_gt
    dt = float(np.linalg.norm(t - t_gt)) * 1e3
    dc = float(np.linalg.norm(c_est - c_gt)) * 1e3
    dR = float(np.degrees(np.linalg.norm(inv_rodrigues(R.T @ R_gt))))
    print(f"\n7) 误差 (同一解, 两种口径)")
    print(f"   平移参数误差  ||Δt|| = {dt:7.2f} mm   (基准表用的口径)")
    print(f"   相机中心误差  ||ΔC|| = {dc:7.2f} mm   (闭环/真机关心的口径)")
    print(f"   比值 = {dc / max(dt, 1e-9):.2f}×    旋转误差 = {dR:.3f}°")
    print(f"   相机中心: 估计 {np.round(c_est,4)} / 真值 {np.round(c_gt,4)}")

    print(f"\n8) 注意: 上面 reproj 只有 {out['reproj_err']:.3f} px, 但位姿误差"
          f"仍有 {dt:.1f} mm")
    print("   4 点 = 8 个标量约束, 未知量 6 个 => 只剩 2 个残差自由度;")
    print("   角点误差几乎全被位姿吸收, 所以『低重投影』不等于『低位姿误差』。")
    print("=" * 68)


if __name__ == "__main__":
    main()
