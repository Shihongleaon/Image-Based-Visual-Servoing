"""
pose_utils.py
-------------
相机内参、SE(3) 工具、刚体变换求解（Kabsch/Umeyama）。

约定:
- 相机坐标系采用 OpenCV 约定: x 右, y 下, z 向前(指向场景)。
- R_wc: world -> camera 的旋转, 点在相机系下坐标 X_c = R_wc @ X_w + t_wc。
- 轴角向量 (axis-angle) 用 Rodrigues 参数化, 与 cv2.Rodrigues 兼容。
"""

import numpy as np


# ---------------------------------------------------------------------------
# 相机内参
# ---------------------------------------------------------------------------

def intrinsic_from_fov(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    """
    由 PyBullet computeProjectionMatrixFOV 的垂直 FOV 反推 3x3 内参矩阵 K。

    PyBullet 使用垂直 FOV + 方形像素, 因此 fx = fy。
    """
    fy = height / (2.0 * np.tan(np.deg2rad(fov_y_deg) / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])


def project_points(points_w: np.ndarray, R: np.ndarray, t: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """
    世界系 3D 点 -> 像素坐标。

    points_w: (N,3), R/t: world->camera, K: (3,3)
    返回 (N,2) 像素坐标。
    """
    pts_c = (R @ points_w.T).T + t.reshape(1, 3)
    z = pts_c[:, 2]
    if np.any(z <= 0):
        # 点在相机背后, 投影无意义; 调用方应保证所有点在前方
        pass
    uv = (K @ (pts_c / z[:, None]).T).T
    return uv[:, :2]


# ---------------------------------------------------------------------------
# 旋转工具
# ---------------------------------------------------------------------------

def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """轴角向量 (3,) -> 旋转矩阵 (3,3)。"""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    Kx = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    # Rodrigues 公式: R = I + sin(t) Kx + (1-cos(t)) Kx^2
    return np.eye(3) + np.sin(theta) * Kx + (1.0 - np.cos(theta)) * (Kx @ Kx)


def inv_rodrigues(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 (3,3) -> 轴角向量 (3,)。"""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # theta ~ pi 的特殊情形: 用对角元恢复轴
        k = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        # 修正符号
        if R[0, 1] < 0:
            k[1] = -k[1]
        if R[0, 2] < 0:
            k[2] = -k[2]
        return k * theta
    k = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return k * theta


def se3_log_rotation(R: np.ndarray) -> np.ndarray:
    """so(3) 对数映射: 相对旋转矩阵 -> 轴角向量。"""
    return inv_rodrigues(R)


def pose_error(R_cur, t_cur, R_des, t_des):
    """
    计算位姿误差。

    返回 (e_t, e_w): e_t (3,) 平移误差向量, e_w (3,) 旋转误差轴角
    (在世界系下的旋转载体, 使 R_des = exp([e_w]x) @ R_cur)。
    """
    e_t = np.asarray(t_cur, float).reshape(3) - np.asarray(t_des, float).reshape(3)
    R_rel = R_des @ R_cur.T          # 从当前到期望的旋转
    e_w = se3_log_rotation(R_rel)
    return e_t, e_w


# ---------------------------------------------------------------------------
# 刚体变换求解 (Kabsch / Umeyama, 无缩放)
# ---------------------------------------------------------------------------

def rigid_transform(src: np.ndarray, dst: np.ndarray):
    """
    求刚体变换 (R, t) 使 dst ~= R @ src + t。

    src, dst: (N,3), N >= 3, 对应点顺序一致。
    返回 (R, t), R: (3,3) 正交且 det=+1, t: (3,)。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    P = src - c_src
    Q = dst - c_dst
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    t = c_dst - R @ c_src
    return R, t


def reproj_error(points_w: np.ndarray, uv: np.ndarray, R, t, K) -> float:
    """均方重投影误差 (像素)。"""
    uv_hat = project_points(points_w, R, t, K)
    return float(np.mean(np.linalg.norm(uv_hat - uv, axis=1)))


# ---------------------------------------------------------------------------
# 角点几何模型
# ---------------------------------------------------------------------------

def marker_corner_model(face_size: float, front_normal=-1):
    """
    ARUCO 标记 3D 角点模型 (位于标记/立方体自身坐标系)。

    标记贴在立方体 -y 朝向的面 (面向位于 -y 方向的相机)。
    返回 (4,3): [左上, 右上, 右下, 左下] 顺序 (OpenCV 检测的标准顺序)。

    OpenCV ArUco 检测的角点顺序: top-left, top-right, bottom-right, bottom-left,
    其中 "上" 是图像里 y 较小的一侧。以面向相机的 -y 面、图像 x 对应世界 +x、
    图像 y 对应世界 -z (z 向上) 为约定。
    """
    s = face_size / 2.0
    # 面向 -y 的面: y = -s
    # 图像 "上" (u 轴负方向) -> 世界 +z; 图像 "右" -> 世界 +x
    return np.array([
        [-s, -s,  s],   # top-left
        [ s, -s,  s],   # top-right
        [ s, -s, -s],   # bottom-right
        [-s, -s, -s],   # bottom-left
    ])
