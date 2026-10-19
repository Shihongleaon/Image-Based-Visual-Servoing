"""
pbvs.py
-------
基于位置的视觉伺服 (PBVS) 控制器。

控制律 (Chaumette & Hutchinson, 2006, position-based):
    设 P3P 解出标记->相机位姿 (R_mc, t_mc), 期望位姿 (R*, t*)。
    位置误差 (标记系):  e_p = p_cam - p*_cam, 其中 p_cam = -R_mc^T t_mc
    旋转误差 (相机系):  theta_u = log(R* R_mc^T)
    速度指令 (相机系):  v_c = -lambda_p * R_mc @ e_p
                        w_c = -lambda_o * theta_u
随后在仿真端把相机系速度旋到世界系积分。

与 IBVS 的本质区别: 误差定义在 SE(3) 而非像素空间, 深度由 P3P 闭式解
直接给出 (不需要 depth buffer 查表), 控制增益与距离解耦。
"""

import numpy as np

from pose_utils import inv_rodrigues, rodrigues

# 期望位姿 (标记系): 相机位于标记法向正前方距离 D 处, 光轴垂直标记面
# R*_mc = diag(1,-1,-1),  t*_mc = (0,0,D)
R_DES = np.diag([1.0, -1.0, -1.0]).astype(float)


def desired_pose(distance: float):
    """返回期望 (R*_mc, t*_mc), D 为相机光心到标记面的距离 (m)。"""
    return R_DES.copy(), np.array([0.0, 0.0, distance])


def pose_error(R_mc, t_mc, R_des, t_des):
    """
    PBVS 误差。

    返回:
        e_p: (3,) 相机中心在标记系下的位置误差 [m]
        theta_u: (3,) 旋转误差轴角 (当前相机系下, 使 R*=exp([w]x) R_mc)
    """
    p_cur = -R_mc.T @ np.asarray(t_mc, float).reshape(3)
    p_des = -R_des.T @ np.asarray(t_des, float).reshape(3)
    e_p = p_cur - p_des
    R_err = R_des @ R_mc.T
    theta_u = inv_rodrigues(R_err)
    return e_p, theta_u


class PBVSController:
    """PBVS 速度控制器 (带误差低通滤波)。

    alpha: 误差一阶低通系数。P3P 的 LM 精化存在两个相近的局部极小,
    逐帧估计会在其间交替形成极限环; 滤波后系统收敛到极小的均值,
    稳态误差由估计噪声水平决定 (实测 ~mm 级)。
    """

    def __init__(self, distance_desired, lambda_p=2.0, lambda_o=2.0,
                 max_lin_vel=0.5, max_ang_vel=1.0, alpha=0.4):
        self.R_des, self.t_des = desired_pose(distance_desired)
        self.lam_p = lambda_p
        self.lam_o = lambda_o
        self.v_max = max_lin_vel
        self.w_max = max_ang_vel
        self.alpha = alpha
        self._e_f = None
        self._w_f = None

    def step(self, R_mc, t_mc):
        """
        由当前 P3P 位姿计算相机系速度指令。

        返回 (v_cam, w_cam, e_p, theta_u); e_p/theta_u 为滤波后的误差。
        """
        e_p, theta_u = pose_error(R_mc, t_mc, self.R_des, self.t_des)

        # 一阶低通 (首帧直接置为测量值)
        if self._e_f is None:
            self._e_f, self._w_f = e_p, theta_u
        else:
            a = self.alpha
            self._e_f = a * e_p + (1.0 - a) * self._e_f
            self._w_f = a * theta_u + (1.0 - a) * self._w_f
        e_pf, w_f = self._e_f, self._w_f

        v_cam = -self.lam_p * (R_mc @ e_pf)
        w_cam = -self.lam_o * w_f

        # 限幅 (真实伺服的执行器约束)
        nv = np.linalg.norm(v_cam)
        nw = np.linalg.norm(w_cam)
        if nv > self.v_max:
            v_cam *= self.v_max / nv
        if nw > self.w_max:
            w_cam *= self.w_max / nw
        return v_cam, w_cam, e_pf, w_f


def integrate_pose(R_wc, t_wc, v_cam, w_cam, dt):
    """
    把相机系速度指令积分到世界系位姿。

    - 平移: 位置在世界系积分, v_w = R_wc^T v_cam (R_wc: world->camera)
    - 旋转: w_cam 是相机系 (body系) 角速度, 直接左乘
        R_new = exp(-[w_cam] dt) R_wc
      推导: body 姿态 C = R_wc^T, C_new = C exp([w]dt)
            => R_new = exp(-[w]dt) R_wc

    返回新 (R_wc, t_wc)。
    """
    v_w = R_wc.T @ v_cam
    cam_pos = -R_wc.T @ t_wc + v_w * dt
    w = np.asarray(w_cam, float).reshape(3)
    th = np.linalg.norm(w) * dt
    if th > 1e-12:
        R_new = rodrigues(-w / np.linalg.norm(w) * th) @ R_wc
    else:
        R_new = R_wc
    t_new = -R_new @ cam_pos
    return R_new, t_new
