# IBVS → PBVS: 自研 P3P + 六轴机械臂视觉伺服

> **来源与致谢**：本仓库的单目 IBVS 仿真基线来自 Varun Edachali 的 MIT 开源项目
> `Image-Based-Visual-Servoing`（原始 LICENSE 保留，见 `LICENSE`）。
> 在此之上新增：自研 P3P 闭式求解器、PBVS 控制律、解析 FK + DLS 逆运动学、
> xArm6 eye-in-hand 闭环、蒙特卡洛基准与全部实验分析。
>
> 环境：Python 3.10（`conda env ibvs310`）；机械臂模型用 `pybullet_data` 自带的
> `xarm/xarm6_robot.urdf`，无需额外下载资产。

基于 PyBullet 的视觉伺服仿真, 三个递进阶段:

1. **IBVS** (原始基线): 图像雅可比把像素误差映射为相机速度, 深度来自 depth buffer 查表
2. **PBVS** (自由相机): 自研 P3P 闭式解 + LM 精化直接从 4 个角点恢复 6D 位姿,
   误差定义在 SE(3) 空间, 深度"免费"获得, 不再依赖深度图
3. **PBVS + xArm6** (eye-in-hand): 位姿误差经 DLS/LM 逆运动学变成六轴关节指令,
   关节位置伺服执行 —— 完整的真实机器人闭环

## 实测指标 (全部来自本仓库脚本的真实运行结果)

### P3P 蒙特卡洛基准 (1500 组位姿, 视角 ±45°, 0.2 m 方形标记, 失败阈值 30 mm / 5°)

| 方法 | 失败率 σ=0.3px | 失败率 σ=0.5px | 平移中位误差 σ=0.3px | 旋转中位 σ=0.3px | 单次耗时 |
|---|---|---|---|---|---|
| 自研 P3P | 8.8% | 18.5% | 3.51 mm | 1.05° | 1.05 ms |
| **自研 P3P + LM** | **0.8%** | **4.5%** | **2.47 mm** | **0.77°** | 3.1 ms |
| cv2 EPnP | 31.4% | 49.4% | 4.55 mm | 1.79° | 0.11 ms |
| cv2 EPnP + LM | 10.6% | 13.1% | 2.29 mm | 0.73° | 1.1 ms |
| cv2 P3P + LM | 0.7% | 3.8% | 2.26 mm | 0.70° | 0.94 ms |

结论: 自研闭式解 + LM 的精度与 OpenCV C++ 实现持平 (2.47 vs 2.26 mm),
失败率比 EPnP 低一个数量级 (0.8% vs 31.4%); 纯 NumPy 实现慢 ~13×,
C++ 移植可消除 (瓶颈是四次方程求根与 SVD)。

**近奇异专项** (最病态 20% 位形, P3P 根间距中位 0.04):

| 方法 | 失败率 |
|---|---|
| 自研 P3P + LM | **25.8%** |
| EPnP | 78.3% |
| EPnP + LM | 33.3% |

图: `src/benchmark_curves.png`

### PBVS 自由相机闭环 (0.3 m 方形 ArUco, 0.6 m 工作距离, 480×300 @ 55° FOV)

| 指标 | 数值 |
|---|---|
| 收敛时间 (判据 <5 mm / <0.8° 连续 10 步) | **4.47 s** (第 134 步) |
| 稳态平移误差 | **3.36 mm** |
| 稳态旋转误差 | **0.31°** |
| P3P+LM 单次求解 | 4.8 ms (纯 Python) |
| 实测控制频率 | 9.2 Hz (受仿真软件光栅化渲染限制) |

曲线: `src/fig_pbvs.png`

### xArm6 + DLS-IK 闭环 (eye-in-hand, 0.55 m 工作距离)

| 指标 | 数值 |
|---|---|
| 收敛时间 | **2.67 s** (第 80 步) |
| 稳态平移误差 | **1.04 mm** |
| 稳态旋转误差 | **0.44°** |
| IK 位置残差 (中位) | **0.0002 mm** |
| IK 耗时 | 3.0 ms (8 次迭代, 解析雅可比) |
| 全程检测失败 | 0 / 250 步 |
| σ_min 最小值 | 0.088 (未进入奇异区, 阻尼全程自适应) |

曲线: `src/fig_arm.png`  视频: `src/arm_servo_demo.mp4` (相机视角)

**第三视角演示** (`--video`): 左侧为固定外部相机俯视工作区, 右侧为 eye-in-hand 相机画面,
顶部叠加逐步真值误差 —— 用于展示机械臂与相机在空间中的实际运动。

```
python main_arm.py --steps 250 --res 480 --video arm_world_demo.mp4 --out out_demo
```

成片: `src/arm_world_demo.mp4` (960×300, 30 fps, 8.3 s)

## 自研 P3P 求解器 (`src/p3p.py`)

- **闭式核**: Grunert 式消元 —— 三角约束 (余弦定理) + 结式消元 → 一元四次方程,
  每个正实根回代出距离三元组, 再由三点正交基直接恢复旋转 (无 SVD)
- **工程层**: 每候选 LM (Levenberg-Marquardt) 精化 + 时序 warm-start 消歧
  (平面标记的二义分支重投影几乎相同, 用上一帧位姿选支 —— 真实伺服系统标准做法)
- **奇异性度量**: 四次方程最小正实根间距 (danger cylinder 判据), 连续量输出
- 实现中踩过的坑 (面试可展开):
  - 方形标记使 a/c ≡ 1, 回代方程 (I) 时结式恒为零 → 必须回代 (II)
  - `np.polynomial` 升幂系数 vs `np.roots` 降幂约定不一致 → 解出了倒数多项式
  - 归一化坐标与像素坐标残差混用 → GN 不收敛

## DLS / LM 逆运动学 (`src/ik.py`)

- 不用 PyBullet 内置 IK (残差 ~10 mm 且不可复现), 自己实现解析 FK + 几何雅可比 + 阻尼最小二乘 `Δq = Jᵀ(JJᵀ+λ²I)⁻¹e`
- **自适应阻尼**: λ 由雅可比最小奇异值 σ_min 自适应, 近奇异时按
  (σ_safe/σ_min)² 放大 —— 关节速度有界, 这就是"奇异构型检测 + LM 阻尼切换"
- FK 经三配置标定与 PyBullet 逐位姿一致 (1e-6 mm), 雅可比与 calculateJacobian 吻合

## 运行

```bash
cd src
conda activate ibvs310          # 或按 docs/env.yml 创建

python test_p3p.py              # P3P 正确性冒烟测试 (零噪声 5.5e-11 px)
python benchmark_p3p.py         # 蒙特卡洛基准 (~5 min) -> benchmark_out.txt
python main_pbvs.py --steps 200 --res 480 --fov 55 --dist 0.6   # 自由相机 PBVS
python main_arm.py --steps 250 --res 320                        # 六轴臂闭环
python analyze.py               # 生成 fig_pbvs.png / fig_arm.png
```

## 目录

```
src/
├── pose_utils.py     # 内参 K / SE(3) 工具 / Kabsch
├── p3p.py            # 自研 P3P 闭式解 + LM 精化 + 奇异性度量
├── pbvs.py           # PBVS 控制律 (SE(3) 误差 -> 相机系 6D 速度)
├── ik.py             # 解析 FK / 雅可比 / DLS-LM 逆运动学 / 奇异度量
├── scene.py          # PyBullet 场景 (真实尺度 0.3 m 标记) 与相机模型
├── main_pbvs.py      # 自由相机 PBVS 闭环
├── main_arm.py       # xArm6 eye-in-hand 闭环
├── benchmark_p3p.py  # P3P vs EPnP 蒙特卡洛基准
├── test_p3p.py       # 正确性测试
├── analyze.py        # 成果图绘制
├── servo.py/motion.py/main.py   # 原始 IBVS (保留作基线)
└── out_pbvs/ out_arm/           # 运行日志、汇总、关键帧
```

## 已知边界 (诚实声明)

- 控制频率受 PyBullet 软件光栅化渲染限制 (~10 Hz @ 480×300); 算法核
  (P3P+LM+IK) 仅 ~8 ms, 真机 30 Hz 相机下绰绰有余
- 稳态旋转误差 ~0.3-0.4° 来自渲染抗锯齿引起的角点亚像素偏差, 可通过
  更高分辨率或角点模型标定继续压低
- 纯 NumPy P3P 比 OpenCV C++ 慢 ~13×, 工程化需 C++ 移植
