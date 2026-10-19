"""
检查 scene.py 的渲染坐标系与 P3P 位姿是否自洽。

做法: 在已知相机位姿下渲染, 用 ArUco 检测角点, 与解析投影对比。
若两者一致 (误差 < 1px), 说明 view 矩阵 / 内参 / 标记系定义全部对齐。
"""

import numpy as np
import cv2

import scene as S
from pose_utils import intrinsic_from_fov, project_points, inv_rodrigues
from p3p import solve_p3p_robust

W, H = 800, 500
K = intrinsic_from_fov(W, H, 90.0)


def detect(img):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    if not corners:
        return None
    return corners[0].reshape(4, 2)


def main():
    S.init_pybullet(gui=False)
    S.build_scene()

    cam_pos = np.array([0.05, 0.10, 0.32])
    R, t = S.look_at_pose(cam_pos, S.MARKER_CENTER)
    rgb, dep, seg = S.render(R, t)
    cv2.imwrite("scene_check.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    uv = detect(rgb)
    if uv is None:
        print("FAIL: 未检测到标记")
        return

    pts_w = S.to_world(S.PTS_M)
    uv_pred = project_points(pts_w, R, t, K)
    print("detected:\n", np.round(uv, 2))
    print("predicted:\n", np.round(uv_pred, 2))

    # 尝试 4 种循环顺序, 找出纹理与标记系的对应关系
    best = None
    for k in range(4):
        idx = [(i + k) % 4 for i in range(4)]
        e = np.abs(uv - uv_pred[idx]).mean()
        print(f"  offset {k}: mean |duv| = {e:6.2f} px")
        if best is None or e < best[0]:
            best = (e, k)
    print("best corner offset:", best)

    # 用最佳对应 (恒等) 做 P3P, 检查位姿
    out = solve_p3p_robust(pts_w, uv, K, use_lm=True)
    print("P3P success:", out["success"], "reproj", round(out["reproj_err"], 3))
    if out["success"]:
        dt = np.linalg.norm(out["t"] - t) * 1000
        dR = np.degrees(np.linalg.norm(inv_rodrigues(out["R"].T @ R)))
        print(f"pose error: {dt:.2f} mm, {dR:.3f} deg")
        # 相机位置反解
        c_est = -out["R"].T @ out["t"]
        print("cam pos est", np.round(c_est, 4), "| gt", np.round(cam_pos, 4))


if __name__ == "__main__":
    main()
