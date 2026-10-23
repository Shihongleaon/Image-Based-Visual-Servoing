"""
scene.py
--------
PBVS 仿真场景与相机模型 (真实尺度: 标记边长 0.3 m, 工作距离 0.4~1.2 m)。

与原 IBVS 场景 (10 m 立方体 + 20x 缩放) 不同, 这里用米制真实尺度, 使
P3P/PBVS 的位姿误差可以直接用 mm / deg 表述, 也便于和真实机械臂迁移。

坐标约定
--------
- 世界系: x 右, y 前 (指向标记), z 上。
- 标记系: X_m = +x, Y_m = +z (世界 +z, 即图像向上), Z_m = -y (法向, 指向相机)。
  该定义与 OpenCV ArUco 的角点顺序 [TL, TR, BR, BL] 一致
  (实测: 检测点与解析投影在循环偏移 0 时 mean|duv| = 0.61 px,
   其它三种偏移为 68/131/69 px, 对应关系唯一)。
- 相机系 (OpenCV): x 右, y 下, z 前。R_wc: world -> camera。
"""

import numpy as np
import pybullet as p
import pybullet_data

from image import get_image_config

# ---- 标记几何 -------------------------------------------------------------
PRINTED_SIZE = 0.30                  # 印刷黑色方块边长, m
BOARD_SIZE = PRINTED_SIZE / (380.0 / 500.0)   # 贴图整面边长 (黑色方块占 76%)
HALF = BOARD_SIZE / 2.0
PHALF = PRINTED_SIZE / 2.0
BOARD_CENTER = np.array([0.0, 0.60, 0.30])
CUBE_SCALE = BOARD_SIZE / 0.05       # cube_small.urdf 原始 0.05 m
MARKER_CENTER = BOARD_CENTER + np.array([0.0, -HALF, 0.0])

# 标记->世界 旋转 (列向量为标记三轴在世界系下的表示)
R_WM = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0],
                 [0.0, 1.0, 0.0]])

# 标记系下 4 角点 (印刷黑框角点), 顺序 = OpenCV ArUco [TL, TR, BR, BL]
PTS_M = np.array([[-PHALF,  PHALF, 0.0],
                  [ PHALF,  PHALF, 0.0],
                  [ PHALF, -PHALF, 0.0],
                  [-PHALF, -PHALF, 0.0]])


def to_world(pts_m):
    """标记系 -> 世界系。"""
    return (R_WM @ np.asarray(pts_m).T).T + MARKER_CENTER


def marker_pose_world():
    """返回标记在世界系下的 (R_wm, c_m)。"""
    return R_WM.copy(), MARKER_CENTER.copy()


# ---- 场景 -----------------------------------------------------------------

def init_pybullet(gui=False):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    return cid


def build_scene(texture_path="static/aruco_marker_face.png", distractors=True):
    """
    构建场景: 地面 + 竖立标记立方体 (0.3 m, 正面贴 ARUCO) + 干扰物。

    返回 dict(plane, board, obstacles)
    """
    plane = p.loadURDF("plane.urdf")
    # 地面调暗: 降低 ArUco 自适应阈值受亮背景干扰的风险
    p.changeVisualShape(plane, -1, rgbaColor=[0.35, 0.35, 0.38, 1.0])

    board = p.loadURDF("cube_small.urdf", BOARD_CENTER.tolist(),
                       globalScaling=CUBE_SCALE, useFixedBase=True)
    tex = p.loadTexture(texture_path)
    p.changeVisualShape(board, -1, textureUniqueId=tex)

    obstacles = []
    if distractors:
        for (x, y, z, s) in [(-0.45, 0.75, 0.05, 0.10),
                             (0.42, 0.70, 0.04, 0.08),
                             (0.10, 1.10, 0.06, 0.12)]:
            obstacles.append(p.loadURDF("cube_small.urdf", [x, y, z],
                                        globalScaling=s / 0.05,
                                        useFixedBase=True))

    return dict(plane=plane, board=board, obstacles=obstacles)


# ---- 相机 -----------------------------------------------------------------

def hidden_texture_flag():
    return 0


def set_texture_visible(on: bool):
    return None


def camera_view_matrix(R_wc, t_wc):
    """
    OpenCV 相机位姿 (world->camera) -> PyBullet viewMatrix。

    PyBullet 的视图矩阵以 '右/上/后' 为相机轴, 故上方向取 -y_cv。
    """
    R_wc = np.asarray(R_wc, float)
    C = -R_wc.T @ np.asarray(t_wc, float).reshape(3)
    fwd = R_wc.T @ np.array([0.0, 0.0, 1.0])
    up = R_wc.T @ np.array([0.0, -1.0, 0.0])
    return p.computeViewMatrix(C.tolist(), (C + fwd).tolist(), up.tolist())


def camera_position(R_wc, t_wc):
    return -np.asarray(R_wc, float).T @ np.asarray(t_wc, float).reshape(3)


def look_at_pose(cam_pos, target_pos, up_world=(0, 0, 1)):
    """由相机位置与注视点构造 OpenCV 位姿 (world->camera)。"""
    cam_pos = np.asarray(cam_pos, float)
    fwd = np.asarray(target_pos, float) - cam_pos
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up_world, float)
    right = np.cross(fwd, up)
    nr = np.linalg.norm(right)
    if nr < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= nr
    down = np.cross(fwd, right)          # y_cv = z x x
    R = np.vstack([right, down, fwd])    # 行 = 相机轴在世界系
    t = -R @ cam_pos
    return R, t


def render(R_wc, t_wc, renderer=None, shadows=0, width=None, height=None,
           fov=None):
    """返回 (rgb HxWx3, depth HxW, seg HxW)。

    光源方向绑定相机视线: 保证标记面 (面向相机) 均匀受光, 消除贴图静区
    因明暗渐变导致的角点定位偏差。
    """
    cfg = get_image_config()
    W = width or cfg["width"]
    H = height or cfg["height"]
    fov = fov or cfg["fov"]
    proj = p.computeProjectionMatrixFOV(fov, W / H,
                                        cfg["near_val"], cfg["far_val"])
    view = camera_view_matrix(R_wc, t_wc)
    fwd = np.asarray(R_wc, float).T @ np.array([0.0, 0.0, 1.0])
    kw = {}
    if renderer is not None:
        kw["renderer"] = renderer
    _, _, rgb, dep, seg = p.getCameraImage(
        W, H, view, proj, lightDirection=(-fwd).tolist(),
        lightAmbientCoeff=0.5, lightDiffuseCoeff=0.5,
        lightSpecularCoeff=0.0, **kw)
    rgb = np.array(rgb, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
    dep = np.array(dep, dtype=np.float32).reshape(H, W)
    seg = np.array(seg, dtype=np.int32).reshape(H, W)
    return rgb, dep, seg


def world_pose_to_marker(R_wc, t_wc):
    """
    world->camera 位姿 -> marker->camera 位姿。

    X_c = R_wc X_w + t_wc,  X_w = R_wm X_m + c_m
      =>  R_mc = R_wc R_wm,  t_mc = R_wc c_m + t_wc
    """
    R_wc = np.asarray(R_wc, float)
    t_wc = np.asarray(t_wc, float).reshape(3)
    return R_wc @ R_WM, R_wc @ MARKER_CENTER + t_wc


def marker_pose_to_world(R_mc, t_mc):
    """
    marker->camera 位姿 -> world->camera 位姿。

    R_wc = R_mc R_wm^T,  t_wc = t_mc - R_wc c_m
    """
    R_mc = np.asarray(R_mc, float)
    t_mc = np.asarray(t_mc, float).reshape(3)
    R_wc = R_mc @ R_WM.T
    return R_wc, t_mc - R_wc @ MARKER_CENTER
