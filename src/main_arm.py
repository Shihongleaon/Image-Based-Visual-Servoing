"""
main_arm.py
-----------
xArm6 六轴机械臂 + 眼在手上 (eye-in-hand) 的 PBVS 视觉伺服闭环。

链路:
    PyBullet 渲染 -> ArUco 角点 -> 自研 P3P(+LM) -> marker->camera 位姿
      -> PBVS 速度律 (相机系 6D 速度)
      -> 目标末端位姿 -> DLS/LM 逆运动学 (自适应阻尼)
      -> 关节位置控制 -> 仿真步进 -> 真实相机位姿 -> 回到第一步

与 main_pbvs.py 的区别: 那里相机是"自由漂浮"的, 位姿可以任意给定;
这里相机固连在六轴臂末端, 速度指令必须经 IK 变成关节指令, 再由关节伺服
执行 —— 这才是真实机械臂视觉伺服的完整闭环, 也才能测出 IK 残差、关节
速度、奇异度量等真实工程指标。

用法:
    python main_arm.py --steps 220
    python main_arm.py --steps 220 --gui       # 可视化
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import pybullet as p

import scene as S
from ik import ArmModel, DLSKinematics, quat_to_matrix, matrix_to_quat
from p3p import solve_p3p_robust
from pbvs import PBVSController, desired_pose, pose_error
from pose_utils import inv_rodrigues, intrinsic_from_fov

# ---- 相机安装 (eye-in-hand) ----------------------------------------------
# 相机坐标系相对 link6 参考点: 光轴沿 link6 的 +x, 像面右向为 link6 的 +y
R_L6_CAM = np.array([[0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]])
T_L6_CAM = np.array([0.035, 0.0, 0.0])      # 支架偏移 (m)

ARM_BASE = np.array([-0.18, -0.26, 0.0])
JOINT_IDS = list(range(1, 7))
EE_LINK = 6
CTRL_HZ = 30.0
DT = 1.0 / CTRL_HZ


# ---- 位姿变换 -------------------------------------------------------------

def cam_pose_from_link6(p_l6, R_l6):
    """link6 位姿 (世界系) -> 相机 OpenCV 位姿 (world->camera)。"""
    R_w_cam = np.asarray(R_l6) @ R_L6_CAM
    p_w_cam = np.asarray(p_l6) + np.asarray(R_l6) @ T_L6_CAM
    R_wc = R_w_cam.T
    return R_wc, -R_wc @ p_w_cam


def link6_pose_from_cam(R_wc, t_wc):
    """相机 OpenCV 位姿 -> link6 目标位姿 (世界系)。"""
    R_w_cam = np.asarray(R_wc).T
    p_w_cam = -np.asarray(R_wc).T @ np.asarray(t_wc).reshape(3)
    R_l6 = R_w_cam @ R_L6_CAM.T
    p_l6 = p_w_cam - R_l6 @ T_L6_CAM
    return p_l6, R_l6


def desired_world_pose(distance):
    """期望相机在 world 系下的 (R_wc, t_wc)。"""
    R_des, t_des = desired_pose(distance)
    R_wc = R_des @ S.R_WM.T
    t_wc = t_des - R_wc @ S.MARKER_CENTER
    return R_wc, t_wc


# ---- 检测 -----------------------------------------------------------------

def detect_corners(rgb):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if not corners:
        return None
    pts = corners[0].reshape(4, 2).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    try:
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), crit)
    except cv2.error:
        pass
    return pts.astype(np.float64)


# ---- 初始化 ---------------------------------------------------------------

def setup(gui=False):
    S.init_pybullet(gui=gui)
    sc = S.build_scene()
    arm = p.loadURDF("xarm/xarm6_robot.urdf", ARM_BASE.tolist(), useFixedBase=True)
    for j in JOINT_IDS:
        p.changeDynamics(arm, j, jointDamping=0.1, linearDamping=0.0,
                         angularDamping=0.0)
        p.setJointMotorControl2(arm, j, p.VELOCITY_CONTROL, force=0)  # 关闭默认速度电机
    model = ArmModel(p, arm, JOINT_IDS, EE_LINK)
    dls = DLSKinematics(model, lam0=2e-3, sigma_safe=0.06)
    return sc, arm, model, dls


def initial_joints(model, dls, distance, seed=0):
    """由"偏离期望位姿"的初始相机位姿反解初始关节角。

    初始位姿构造: 沿标记法向后退 0.28 m、侧移 0.10 m、抬高 0.06 m,
    朝向仍用 look_at 指向标记中心 (保证标记在视野内), 再叠加 ~7 deg 姿态偏差。
    这样初始误差足够大 (约 0.3 m / 7 deg), 又能全程看到标记。
    """
    from pose_utils import rodrigues
    R_wc_des, t_wc_des = desired_world_pose(distance)
    p_w_cam_des = -R_wc_des.T @ t_wc_des
    p_start = p_w_cam_des + np.array([0.10, -0.26, 0.06])
    R_wc_start, t_wc_start = S.look_at_pose(p_start, S.MARKER_CENTER)
    rng = np.random.default_rng(seed)
    R_start = rodrigues(rng.normal(scale=0.07, size=3)) @ R_wc_start
    t_start = -R_start @ p_start
    p_l6, R_l6 = link6_pose_from_cam(R_start, t_start)

    q_home = np.array([0.0, -0.6, -0.4, 0.0, 0.9, 0.0])
    q, info = dls.solve(q_home, p_l6, R_l6, iters=80)
    return q, info


# ---- 第三视角渲染 (演示用) -------------------------------------------------

def render_world(width, height):
    """固定外部相机俯视工作区, 用于展示机械臂与相机的空间运动。

    半分辨率渲染再放大: 第三视角只用于看运动趋势, 不参与任何算法,
    没必要按相机分辨率出图 (软光栅代价与像素数成正比)。
    """
    w, h = max(240, width // 2), max(150, height // 2)
    target = [-0.04, 0.22, 0.30]
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target, distance=1.12,
        yaw=58.0, pitch=-32.0, roll=0.0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=52.0, aspect=w / h,
                                        nearVal=0.05, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(
        w, h, view, proj, renderer=p.ER_TINY_RENDERER, shadow=0,
        lightDirection=[-0.5, -0.6, -0.6],
        lightAmbientCoeff=0.6, lightDiffuseCoeff=0.5, lightSpecularCoeff=0.0)
    img = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
    return img


# ---- 主循环 ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--dist", type=float, default=0.55, help="期望工作距离 (m)")
    ap.add_argument("--res", type=int, default=480, help="渲染宽度 (高 = 0.625W)")
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--lam", type=float, default=2.8)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--video", type=str, default="",
                    help="第三视角分屏演示视频输出路径 (mp4), 空则不录")
    ap.add_argument("--out", type=str, default="out_arm")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    W, H = args.res, int(args.res * 0.625)
    K = intrinsic_from_fov(W, H, args.fov)

    import cv2
    writer = None
    if args.video:
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 30.0, (2 * W, H))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频: {args.video}")

    def emit(world_img, cam_img, idx, et=None, er=None):
        """左: 第三视角; 右: 手上相机; 顶部叠加实时真值误差。"""
        if writer is None:
            return
        left = cv2.cvtColor(world_img, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
        left = cv2.resize(left, (right.shape[1], right.shape[0]))
        canvas = np.hstack([left, right])
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        txt = f"step {idx:03d}"
        if et is not None:
            txt += f"   e_t {et:6.2f} mm   e_R {er:5.2f} deg"
        cv2.putText(canvas, txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (right.shape[1], 0), (right.shape[1], canvas.shape[0]),
                 (255, 255, 255), 1)
        writer.write(canvas)

    sc, arm, model, dls = setup(gui=args.gui)
    ctrl = PBVSController(args.dist, lambda_p=args.lam, lambda_o=args.lam,
                          max_lin_vel=0.35, max_ang_vel=0.8, alpha=args.alpha)

    q0, init_info = initial_joints(model, dls, args.dist, args.seed)
    for j, v in zip(JOINT_IDS, q0):
        p.resetJointState(arm, j, float(v))
    print(f"初始 IK 残差 {init_info['err_p']*1e3:.2f} mm / "
          f"{np.degrees(init_info['err_r']):.2f} deg")

    R_wc_des, t_wc_des = desired_world_pose(args.dist)
    rows, frames = [], {}
    conv_step = None
    R_prev = t_prev = None
    save_steps = {0, 2, 5, 10, 20, 40, 80, 120, 200}

    for i in range(args.steps):
        world_img = render_world(W, H) if writer is not None else None
        t_start = time.perf_counter()

        # 1) 真实相机位姿 (由仿真状态读出)
        st = p.getLinkState(arm, EE_LINK, computeForwardKinematics=True)
        R_wc, t_wc = cam_pose_from_link6(np.array(st[0]), quat_to_matrix(st[1]))

        # 2) 渲染 + 检测
        rgb, dep, seg = S.render(R_wc, t_wc, width=W, height=H, fov=args.fov)
        uv = detect_corners(rgb)
        if uv is None:
            rows.append(dict(step=i, detect=0, e_t_mm=np.nan, e_R_deg=np.nan))
            emit(world_img, rgb, i)
            continue

        # 3) P3P -> marker->camera
        t_p3p0 = time.perf_counter()
        out = solve_p3p_robust(S.to_world(S.PTS_M), uv, K, use_lm=True,
                               R_prev=R_prev, t_prev=t_prev)
        p3p_ms = (time.perf_counter() - t_p3p0) * 1e3
        if not out["success"]:
            rows.append(dict(step=i, detect=1, e_t_mm=np.nan, e_R_deg=np.nan))
            emit(world_img, rgb, i)
            continue
        # P3P 解的是 world->camera, 控制律需要 marker->camera
        R_mc, t_mc = S.world_pose_to_marker(out["R"], out["t"])

        # 4) PBVS 速度
        v_cam, w_cam, e_p, theta_u = ctrl.step(R_mc, t_mc)

        # 5) 目标相机位姿 -> 目标末端位姿 -> IK
        from pbvs import integrate_pose
        R_wc_tgt, t_wc_tgt = integrate_pose(R_wc, t_wc, v_cam, w_cam, DT)
        p_l6_tgt, R_l6_tgt = link6_pose_from_cam(R_wc_tgt, t_wc_tgt)
        q_cur = np.array([p.getJointState(arm, j)[0] for j in JOINT_IDS])
        t_ik0 = time.perf_counter()
        q_tgt, ik_info = dls.solve(q_cur, p_l6_tgt, R_l6_tgt, iters=8)
        ik_ms = (time.perf_counter() - t_ik0) * 1e3

        # 6) 关节位置伺服 + 步进
        n_sub = max(1, int(round(DT / (1.0 / 240.0))))
        for j, v in zip(JOINT_IDS, q_tgt):
            p.setJointMotorControl2(arm, j, p.POSITION_CONTROL,
                                    targetPosition=float(v), force=180.0,
                                    maxVelocity=2.5, positionGain=0.12,
                                    velocityGain=0.8)
        for _ in range(n_sub):
            p.stepSimulation()

        # 7) 真值误差 (由实际相机位姿与期望位姿算, 不经图像)
        R_mc_gt, t_mc_gt = S.world_pose_to_marker(R_wc, t_wc)
        e_p_gt, th_gt = pose_error(R_mc_gt, t_mc_gt, *desired_pose(args.dist))
        e_t_gt = float(np.linalg.norm(e_p_gt) * 1e3)
        e_R_gt = float(np.degrees(np.linalg.norm(th_gt)))
        # 估计误差 (P3P 认为的误差) —— 与真值之差即"感知误差"
        e_t_est = float(np.linalg.norm(pose_error(R_mc, t_mc, *desired_pose(args.dist))[0]) * 1e3)
        # 下一帧 P3P 的"选支先验": 必须用**估计值**, 不能用真值 R_wc/t_wc
        # (否则等于把真值喂进候选消歧, 属于 oracle 泄漏; 与 main_pbvs 保持一致)。
        # 注意它只参与候选间选支, 不作为 LM 初值。
        R_prev, t_prev = out["R"].copy(), out["t"].copy()

        loop_ms = (time.perf_counter() - t_start) * 1e3
        emit(world_img, rgb, i, e_t_gt, e_R_gt)
        rows.append(dict(
            step=i, detect=1,
            e_t_mm=e_t_gt, e_R_deg=e_R_gt,
            e_t_est_mm=e_t_est,
            p3p_reproj=out["reproj_err"],
            p3p_ms=p3p_ms, ik_ms=ik_ms, loop_ms=loop_ms,
            ik_err_p_mm=ik_info["err_p"] * 1e3,
            ik_err_r_deg=float(np.degrees(ik_info["err_r"])),
            ik_iters=ik_info["iterations"],
            sigma_min=ik_info["sigma_min"],
            lam=ik_info["lam"],
            singular=int(ik_info["singular"]),
            manipulability=ik_info["manipulability"],
            n_cand=out["n_candidates"],
        ))

        if i % 20 == 0 or i < 5:
            print(f"[{i:3d}] e_t={e_t_gt:7.2f}mm e_R={e_R_gt:6.3f}deg | "
                  f"P3P {p3p_ms:5.2f}ms IK {ik_ms:5.2f}ms "
                  f"(σmin={ik_info['sigma_min']:.3f}) | loop {loop_ms:5.1f}ms")

        if i in save_steps:
            frames[i] = rgb.copy()

        if conv_step is None and i >= 10:
            tail = [r for r in rows[-12:] if r.get("detect")]
            if len(tail) >= 12 and all(r["e_t_mm"] < 5.0 and r["e_R_deg"] < 0.8
                                       for r in tail):
                conv_step = i - 11
                print(f"*** 收敛于第 {conv_step} 步 ({conv_step / CTRL_HZ:.2f} s, "
                      f"判据 <5 mm / <0.8 deg 连续 12 步) ***")

    # ---- 汇总 ----
    if writer is not None:
        writer.release()
        print(f"演示视频已保存: {args.video}")
    valid = [r for r in rows if r.get("detect") and np.isfinite(r.get("e_t_mm", np.nan))]
    err_t = np.array([r["e_t_mm"] for r in valid])
    err_r = np.array([r["e_R_deg"] for r in valid])
    ik_p = np.array([r["ik_err_p_mm"] for r in valid])
    loop_ms = np.array([r["loop_ms"] for r in valid])
    summary = dict(
        converged=conv_step is not None,
        conv_step=conv_step,
        conv_time_s=(conv_step / CTRL_HZ) if conv_step is not None else None,
        steady_t_mm=float(np.mean(err_t[-15:])),
        steady_R_deg=float(np.mean(err_r[-15:])),
        final_t_mm=float(err_t[-1]),
        final_R_deg=float(err_r[-1]),
        detect_fail=int(sum(1 for r in rows if r.get("detect") == 0)),
        p3p_ms_median=float(np.median([r["p3p_ms"] for r in valid])),
        ik_ms_median=float(np.median([r["ik_ms"] for r in valid])),
        loop_ms_median=float(np.median(loop_ms)),
        ctrl_hz=float(1e3 / np.median(loop_ms)),
        ik_err_p_mm_median=float(np.median(ik_p)),
        ik_err_p_mm_max=float(np.max(ik_p)),
        sigma_min_min=float(min(r["sigma_min"] for r in valid)),
        singular_steps=int(sum(r["singular"] for r in valid)),
        steps=len(valid),
    )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in r.items()})

    import cv2
    for i, img in frames.items():
        cv2.imwrite(os.path.join(args.out, f"frame_{i:03d}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print("\n=== 汇总 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    main()
