"""
main_pbvs.py
------------
PBVS 闭环主程序: PyBullet 真实尺度场景 + 自研 P3P + SE(3) 控制律。

运行:
    python main_pbvs.py [--dist 0.8] [--steps 200] [--gui] [--seed 0]

输出 (out_pbvs/):
    log.csv           逐步日志: 时间/误差/耗时/控制周期
    curve.png         误差收敛曲线 (平移 mm, 旋转 deg)
    traj.png          相机 3D 轨迹
    frame_XXX.png     关键帧
    summary.json      汇总指标 (收敛步数, 稳态误差, 控制频率)
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

import scene as S
import pybullet as p
from image import get_image_config
from pose_utils import (intrinsic_from_fov, inv_rodrigues, rodrigues,
                        project_points)
from p3p import solve_p3p_robust
from pbvs import PBVSController, integrate_pose

W, H = 800, 500
K = intrinsic_from_fov(W, H, 90.0)

CTRL_HZ = 30.0
DT = 1.0 / CTRL_HZ
CONV_T_MM = 2.0
CONV_R_DEG = 0.5


def detect_corners(rgb, subpix=True):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    params = cv2.aruco.DetectorParameters()
    if subpix:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
    det = cv2.aruco.ArucoDetector(d, params)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if not corners:
        return None
    return corners[0].reshape(4, 2).astype(np.float64)


def initial_pose(distance, seed=0):
    """带初始横向/角度偏差的起始位姿。"""
    rng = np.random.default_rng(seed)
    c_m = S.MARKER_CENTER
    side = rng.uniform(-0.35, 0.35)
    zoff = rng.uniform(-0.25, 0.25)
    cam_pos = np.array([side, c_m[1] - distance, c_m[2] + zoff])
    R, t = S.look_at_pose(cam_pos, c_m + rng.normal(scale=0.05, size=3))
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--res", type=int, default=480,
                    help="控制回路渲染宽度 (高按 5:8 比例)")
    ap.add_argument("--fov", type=float, default=60.0,
                    help="垂直 FOV (伺服相机典型 45~70 度, 越窄标记占比越大)")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="误差低通系数 (越小越平滑)")
    ap.add_argument("--lambda-p", type=float, default=1.5,
                    help="平移增益 lambda_p")
    ap.add_argument("--lambda-o", type=float, default=1.5,
                    help="旋转增益 lambda_o")
    ap.add_argument("--out", type=str, default="out_pbvs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    S.init_pybullet(gui=args.gui)
    S.build_scene()

    W, H = args.res, int(args.res * 5 / 8)
    K = intrinsic_from_fov(W, H, args.fov)

    R_wc, t_wc = initial_pose(args.dist, args.seed)
    ctrl = PBVSController(args.dist, lambda_p=args.lambda_p,
                          lambda_o=args.lambda_o, alpha=args.alpha)

    rows = []
    frames = {}
    conv_step = None
    n_steps = args.steps
    R_prev = t_prev = None

    for i in range(n_steps):
        t0 = time.perf_counter()

        rgb, dep, seg = S.render(R_wc, t_wc, width=W, height=H,
                                 fov=args.fov)
        uv = detect_corners(rgb)

        if uv is None:
            rows.append(dict(step=i, t=i * DT, e_t_mm=np.nan, e_R_deg=np.nan,
                             reproj=np.nan, ms=np.nan, detect=0))
            print(f"[{i:3d}] 检测失败, 保持不动")
            continue

        pts_w = S.to_world(S.PTS_M)
        t1 = time.perf_counter()
        out = solve_p3p_robust(pts_w, uv, K, use_lm=True,
                               R_prev=R_prev, t_prev=t_prev)
        t2 = time.perf_counter()

        if not out["success"]:
            rows.append(dict(step=i, t=i * DT, e_t_mm=np.nan, e_R_deg=np.nan,
                             reproj=np.inf, ms=(t2 - t1) * 1e3, detect=1))
            continue

        # P3P 输出 world->camera 位姿 (仅作为传感器读数, 不覆盖真实状态)
        R_mc, t_mc = S.world_pose_to_marker(out["R"], out["t"])
        v_cam, w_cam, e_p, theta_u = ctrl.step(R_mc, t_mc)

        e_t_mm = float(np.linalg.norm(e_p) * 1000.0)
        e_R_deg = float(np.degrees(np.linalg.norm(theta_u)))

        t3 = time.perf_counter()
        # 真实位姿只由运动学积分更新 (估计误差不进入状态)
        R_wc, t_wc = integrate_pose(R_wc, t_wc, v_cam, w_cam, DT)
        t4 = time.perf_counter()
        R_prev, t_prev = out["R"].copy(), out["t"].copy()

        # 位姿估计精度真值: 当前积分出的真实位姿 (不经图像)
        R_mc_gt, t_mc_gt = S.world_pose_to_marker(R_wc, t_wc)
        gt_t_mm = float(np.linalg.norm(t_mc - t_mc_gt) * 1000.0)
        gt_R_deg = float(np.degrees(np.linalg.norm(
            inv_rodrigues(R_mc.T @ R_mc_gt))))
        cam_pos = -R_wc.T @ t_wc

        rows.append(dict(
            step=i, t=i * DT, e_t_mm=e_t_mm, e_R_deg=e_R_deg,
            reproj=out["reproj_err"], ms=(t2 - t1) * 1e3,
            control_ms=(t4 - t0) * 1e3, detect=1,
            used_lm=out["used_lm"], singular=out["singular"],
            cam_x=cam_pos[0], cam_y=cam_pos[1], cam_z=cam_pos[2],
            gt_t_mm=gt_t_mm, gt_R_deg=gt_R_deg,
        ))

        if i in (0, 1, 4, 10, 25, 50, 100):
            frames[i] = rgb.copy()

        if conv_step is None and i >= 10:
            tail = [r for r in rows[-10:] if r.get("e_t_mm") == r.get("e_t_mm")]
            if len(tail) >= 10 and all(r["e_t_mm"] < 5.0
                                       and r["e_R_deg"] < 0.8
                                       for r in tail):
                conv_step = i - 9
                print(f"*** 收敛于第 {conv_step} 个控制步 "
                      f"({conv_step / CTRL_HZ:.2f} s, 阈值 5mm/0.8deg) ***")

        if i % 10 == 0 or i == n_steps - 1:
            print(f"[{i:3d}] e_t={e_t_mm:7.2f}mm e_R={e_R_deg:6.2f}deg "
                  f"reproj={out['reproj_err']:5.2f}px "
                  f"p3p={rows[-1]['ms']:5.1f}ms lm={int(out['used_lm'])}")

        if e_t_mm < 0.3 and e_R_deg < 0.05 and i > 20:
            print("达到准零误差, 提前结束")
            break

    # ---- 输出 ----
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out, "log.csv"), "w", newline="",
              encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        wcsv.writerows(rows)

    for i, img in frames.items():
        cv2.imwrite(os.path.join(args.out, f"frame_{i:03d}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    valid = [r for r in rows if r.get("e_t_mm") == r.get("e_t_mm")]
    err_t = np.array([r["e_t_mm"] for r in valid])
    err_R = np.array([r["e_R_deg"] for r in valid])
    ts = np.array([r["t"] for r in valid])
    p3p_ms = np.array([r["ms"] for r in valid])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ts, err_t)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("|dp| (mm)")
    ax[0].set_title("PBVS translation error")
    ax[0].grid(alpha=.3)
    ax[1].plot(ts, err_R, color="tab:orange")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("rotation error (deg)")
    ax[1].set_title("PBVS rotation error")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "curve.png"), dpi=150)

    if len(valid) > 2 and "cam_x" in rows[-1]:
        xs = np.array([r["cam_x"] for r in valid])
        ys = np.array([r["cam_y"] for r in valid])
        zs = np.array([r["cam_z"] for r in valid])
        fig3 = plt.figure(figsize=(6, 5))
        a3 = fig3.add_subplot(111, projection="3d")
        a3.plot(xs, ys, zs, "-o", ms=2)
        a3.scatter(*S.MARKER_CENTER, color="r", s=60, label="marker")
        a3.scatter(xs[0], ys[0], zs[0], color="g", s=60, label="start")
        a3.scatter(xs[-1], ys[-1], zs[-1], color="b", s=60, label="end")
        a3.set_xlabel("x (m)"); a3.set_ylabel("y (m)"); a3.set_zlabel("z (m)")
        a3.legend()
        fig3.tight_layout()
        fig3.savefig(os.path.join(args.out, "traj.png"), dpi=150)

    steady = err_t[-20:]
    summary = dict(
        config=dict(dist_m=args.dist, fov_deg=args.fov,
                    resolution=f"{W}x{H}", ctrl_hz_nominal=CTRL_HZ),
        converged=conv_step is not None,
        conv_step=conv_step,
        conv_time_s=(conv_step / CTRL_HZ) if conv_step is not None else None,
        steady_t_mm=float(np.mean(steady)),
        steady_R_deg=float(np.mean(err_R[-20:])),
        p95_t_mm=float(np.percentile(err_t[-40:], 95)),
        ctrl_hz_measured=float(1.0 / np.mean([r.get("control_ms", np.nan)
                                              for r in valid]) * 1e3)
        if all("control_ms" in r for r in valid) else None,
        p3p_ms_median=float(np.median(p3p_ms)),
        steps=len(valid),
    )
    with open(os.path.join(args.out, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nsummary:", json.dumps(summary, ensure_ascii=False, indent=2))

    p.disconnect()


if __name__ == "__main__":
    main()
