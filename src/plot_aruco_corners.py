"""
plot_aruco_corners.py
---------------------
把一帧仿真图里检测到的 4 个 ArUco 角点圈出来, 并对比
"3D 是正方形 / 2D 投影是任意四边形"。用于面试时指着图讲。

运行: python plot_aruco_corners.py
输出: fig_aruco_corners.png
"""

import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import scene as S
from pose_utils import intrinsic_from_fov

W, H, FOV = 480, 300, 55.0
K = intrinsic_from_fov(W, H, FOV)


def detect(rgb):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    det = cv2.aruco.ArucoDetector(d, p)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if not corners:
        return None, None
    return corners[0].reshape(4, 2).astype(np.float64), ids


def main():
    S.init_pybullet(gui=False)
    S.build_scene()

    c = S.MARKER_CENTER
    cam = np.array([0.18, c[1] - 0.60, c[2] + 0.12])
    R, t = S.look_at_pose(cam, c)
    rgb, _, _ = S.render(R, t, width=W, height=H, fov=FOV)

    uv, ids = detect(rgb)
    if uv is None:
        print("检测失败")
        return

    pts_w = S.to_world(S.PTS_M)
    names = ["p0 = TL", "p1 = TR", "p2 = BR", "p3 = BL"]

    fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.0),
                           gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------------- (a) 渲染图 + 角点 ----------------
    ax[0].imshow(rgb)
    ax[0].set_xlim(0, W)
    ax[0].set_ylim(H, 0)
    colors = ["#e74c3c", "#f39c12", "#27ae60", "#2980b9"]
    for i, (u, v) in enumerate(uv):
        ax[0].plot(u, v, "o", ms=13, mfc="none", mec=colors[i], mew=2.4)
        ax[0].plot(u, v, "+", ms=8, color=colors[i], mew=1.6)
        dx, dy = (28, -22) if i in (0, 3) else (-30, -22)
        ax[0].annotate(f"{names[i]}\n({u:.1f}, {v:.1f})",
                       xy=(u, v), xytext=(u + dx, v + dy),
                       fontsize=8.5, color="white", weight="bold",
                       bbox=dict(boxstyle="round,pad=0.28", fc=colors[i],
                                 ec="none", alpha=0.92),
                       arrowprops=dict(arrowstyle="-", color=colors[i], lw=1.4))
    # 把四边形连起来
    loop = np.vstack([uv, uv[:1]])
    ax[0].plot(loop[:, 0], loop[:, 1], "-", color="#00e5ff", lw=1.4, alpha=0.9)
    ax[0].set_xlabel("u (px)  → 图像向右")
    ax[0].set_ylabel("v (px)  ↓ 图像向下")
    ax[0].set_title(f"(a) 检测结果: {W}x{H} px, fov={FOV:.0f}°, "
                    f"ArUco id={None if ids is None else int(ids.ravel()[0])}",
                    fontsize=10)

    # ---------------- (b) 四边形几何量 ----------------
    q = uv - uv.mean(axis=0)
    loopq = np.vstack([q, q[:1]])
    ax[1].plot(loopq[:, 0], loopq[:, 1], "-o", color="#00bcd4", lw=2,
               ms=9, mfc="white", mew=2)
    for i, (x, y) in enumerate(q):
        ax[1].annotate(f"p{i}", (x, y), xytext=(6, 6),
                       textcoords="offset points", fontsize=10,
                       color=colors[i], weight="bold")

    def ang(a, b, c):                       # b 处的内角
        v1, v2 = a - b, c - b
        cosv = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))

    sides = [float(np.linalg.norm(uv[(i + 1) % 4] - uv[i])) for i in range(4)]
    angs = [ang(uv[i - 1], uv[i], uv[(i + 1) % 4]) for i in range(4)]
    txt = ("3D (标记系, 米):\n"
           f"  4 条边全部 = {S.PRINTED_SIZE:.3f} m, 4 个角全部 = 90°\n\n"
           "2D (像素, 投影后):\n"
           f"  边长 = {sides[0]:.1f} / {sides[1]:.1f} / {sides[2]:.1f} / {sides[3]:.1f} px\n"
           f"  内角 = {angs[0]:.1f}° / {angs[1]:.1f}° / {angs[2]:.1f}° / {angs[3]:.1f}°\n"
           f"  对边比 = {sides[0]/sides[2]:.3f} (1.0 才是平行四边形)")
    ax[1].text(0.5, 0.5, txt, transform=ax[1].transAxes, fontsize=8.6,
               va="center", ha="center",
               bbox=dict(boxstyle="round,pad=0.45", fc="#f7f7f7",
                         ec="#bbbbbb", alpha=0.96))
    ax[1].set_aspect("equal")
    ax[1].invert_yaxis()          # 与图像坐标一致: v 向下为正
    ax[1].margins(0.22)
    ax[1].grid(alpha=0.3)
    ax[1].set_xlabel("u − ū (px)")
    ax[1].set_ylabel("v − v̄ (px)")
    ax[1].set_title("(b) 透视投影: 正方形 → 任意四边形", fontsize=10)

    fig.suptitle("ArUco 4 角点: 3D 是正方形, 2D 是四边形; "
                 "配对靠 ArUco 固定角点序 [TL, TR, BR, BL]",
                 fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("fig_aruco_corners.png", dpi=150)
    print("saved fig_aruco_corners.png")
    print("uv =\n", np.round(uv, 2))


if __name__ == "__main__":
    main()
