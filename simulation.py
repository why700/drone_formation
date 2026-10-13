# simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

# --- 中文显示配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 正在运行的动画仿真（模块级强引用表）---
# 必需：GUI 里 sim 只是一个局部变量，run_matplotlib() 返回后就没有外部引用了；
# 而 sim._mp_timer <-> 定时器回调(bound method) 之间构成引用环，Python 的 gc
# 会在几秒后把整个环回收，QTimer 随之销毁 —— 表现就是动画跑到几百步突然静止。
# 放在这里当“锚点”，图窗关闭时再由 stop_matplotlib() 移除，避免内存泄漏。
_ACTIVE_SIMULATIONS = {}


class FlockingSimulation:
    def __init__(self, N, D, K, A, B, C1_ALPHA, target_params=None, obstacle_params=None, map_widget=None):
        # === 物理参数 ===
        self.N = int(N)
        self.D = int(D)
        self.K = float(K)
        self.A = int(A)
        self.B = int(B)
        self.C1_ALPHA = int(C1_ALPHA)

        # === 组件引用 ===
        self.map_widget = map_widget

        # === 目标与障碍物 ===
        self.target_params = target_params
        self.obstacle_params = obstacle_params

        # 默认目标初始化
        self.q_mt = None
        self.p_mt = np.zeros(2)
        if self.target_params:
            self.c1_mt = self.target_params.get('c1_mt', 1.1)
            self.c2_mt = 2 * np.sqrt(self.c1_mt)
            self.q_mt = np.array(self.target_params.get('pos', [100, 100]), dtype=float)

        # 默认障碍物初始化
        if self.obstacle_params:
            self.obs_pos = np.array(self.obstacle_params.get('pos', [100, 130]), dtype=float)
            self.obs_r = self.obstacle_params.get('radius', 20)
            self.c1_beta = self.obstacle_params.get('c1_beta', 500)
            self.c2_beta = 2 * np.sqrt(self.c1_beta)
            self.d_prime = self.obstacle_params.get('buffer', 10) + self.obs_r

        # === 仿真常量 ===
        self.M = 2
        self.EPSILON = 0.1
        self.H = 0.2
        self.DELTA_T = 0.009
        self.R = self.K * self.D

        self.C2_ALPHA = 2 * np.sqrt(self.C1_ALPHA)
        self.C = np.abs(self.A - self.B) / np.sqrt(4 * self.A * self.B)

        # === 状态初始化 ===
        self.nodes = np.random.rand(self.N, self.M) * 50
        self.nodes_velocity_p = np.zeros([self.N, self.M])

        self.current_step = 0
        self.MAX_ITERATION = 99999999

        # 轨迹记录
        self.traj_x = []  # 目标轨迹
        self.traj_y = []
        self.center_traj_x = []  # 重心轨迹
        self.center_traj_y = []

    # ------------------ 动态参数更新 ------------------
    def update_obstacle_pos(self, x, y):
        if self.obstacle_params:
            self.obs_pos = np.array([float(x), float(y)])
            self.obstacle_params['pos'] = [float(x), float(y)]
            if self.map_widget:
                self.map_widget.update_scene(self.nodes, [], self.obstacle_params, self.q_mt)

    def update_target_pos(self, x, y):
        if self.target_params and self.target_params.get('mode') == 'static':
            self.q_mt = np.array([float(x), float(y)])
            self.target_params['pos'] = [float(x), float(y)]
            if self.map_widget:
                self.map_widget.update_scene(self.nodes, [], self.obstacle_params, self.q_mt)

    # ------------------ 核心数学计算 ------------------
    def _sigma_norm(self, z):
        return (np.sqrt(1 + self.EPSILON * z ** 2) - 1) / self.EPSILON

    def _bump_function(self, z):
        return np.where((0 <= z) & (z < self.H), 1,
                        np.where((self.H <= z) & (z < 1), 0.5 * (1 + np.cos(np.pi * (z - self.H) / (1 - self.H))), 0))

    def _sigma_1(self, z):
        return z / np.sqrt(1 + z ** 2)

    def _phi(self, z):
        return 0.5 * ((self.A + self.B) * self._sigma_1(z + self.C) + (self.A - self.B))

    def _phi_alpha(self, z):
        sig_r = self._sigma_norm(self.R)
        val = np.divide(z, sig_r, out=np.zeros_like(z), where=sig_r != 0)
        return self._bump_function(val) * self._phi(z - self._sigma_norm(self.D))

    def calculate_u_all(self):
        # 1. Alpha 交互
        diff_matrix = self.nodes[np.newaxis, :, :] - self.nodes[:, np.newaxis, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        mask = (dist_matrix <= self.R) & (dist_matrix > 0)
        mask_exp = mask[:, :, np.newaxis]

        sig_norm_dist = self._sigma_norm(dist_matrix)
        scaling = np.sqrt(1 + self.EPSILON * dist_matrix ** 2)

        term1 = np.sum(
            self._phi_alpha(sig_norm_dist)[:, :, np.newaxis] * (diff_matrix / scaling[:, :, np.newaxis]) * mask_exp,
            axis=1)

        sig_r = self._sigma_norm(self.R)
        vel_diff = self.nodes_velocity_p[np.newaxis, :, :] - self.nodes_velocity_p[:, np.newaxis, :]
        term2 = np.sum(
            self._bump_function(sig_norm_dist / sig_r)[:, :, np.newaxis] * vel_diff * mask_exp,
            axis=1)

        u = self.C1_ALPHA * term1 + self.C2_ALPHA * term2

        # 2. Gamma 目标 (包含速度匹配修复)
        if self.target_params and self.q_mt is not None:
            # 关键修复：阻尼项加入 - (v_agent - v_target)
            u += -self.c1_mt * (self.nodes - self.q_mt) - self.c2_mt * (self.nodes_velocity_p - self.p_mt)

        # 3. Beta 避障
        if self.obstacle_params:
            diff_obs = self.nodes - self.obs_pos
            dist_obs = np.linalg.norm(diff_obs, axis=1)

            mu = self.obs_r / dist_obs
            a_k = diff_obs / dist_obs[:, np.newaxis]
            P = np.eye(2) - a_k[:, :, np.newaxis] * a_k[:, np.newaxis, :]

            q_ik = mu[:, np.newaxis] * self.nodes + (1 - mu[:, np.newaxis]) * self.obs_pos
            p_ik = mu[:, np.newaxis] * np.einsum('nij,nj->ni', P, self.nodes_velocity_p)

            beta_dist = np.linalg.norm(q_ik - self.nodes, axis=1)
            avoid_mask = beta_dist < self.d_prime

            if np.any(avoid_mask):
                idx = avoid_mask
                sig_beta = self._sigma_norm(beta_dist[idx])
                sig_d = self._sigma_norm(self.d_prime)

                b_ik = self._bump_function(sig_beta / sig_d)
                n_ik = (q_ik[idx] - self.nodes[idx]) / np.sqrt(1 + self.EPSILON * beta_dist[idx, np.newaxis] ** 2)

                repulsion = self.c1_beta * (b_ik * (self._sigma_1(sig_beta - sig_d) - 1))[:, np.newaxis] * n_ik
                damping = self.c2_beta * b_ik[:, np.newaxis] * (p_ik[idx] - self.nodes_velocity_p[idx])

                u[idx] += repulsion + damping

        return u

    def _physics_step(self):
        curr_time = self.current_step * self.DELTA_T

        # 更新动态目标
        if self.target_params:
            mode = self.target_params['mode']
            start_pos = np.array(self.target_params['pos'])
            if mode == 'sine':
                vx = self.target_params['vx']
                amp = self.target_params['amp']
                freq = self.target_params['freq']
                self.q_mt = [start_pos[0] + vx * curr_time, start_pos[1] - amp * np.sin(freq * curr_time)]
                # 计算目标速度向量
                self.p_mt = [vx, -amp * freq * np.cos(freq * curr_time)]
                self.q_mt = np.array(self.q_mt)
                self.p_mt = np.array(self.p_mt)
            elif mode == 'circle':
                r = self.target_params['radius']
                w = self.target_params['omega']
                self.q_mt = [start_pos[0] + r * np.cos(w * curr_time), start_pos[1] + r * np.sin(w * curr_time)]
                self.p_mt = [-r * w * np.sin(w * curr_time), r * w * np.cos(w * curr_time)]
                self.q_mt = np.array(self.q_mt)
                self.p_mt = np.array(self.p_mt)

        # 积分位置
        u = self.calculate_u_all()
        self.nodes_velocity_p += u * self.DELTA_T
        self.nodes += self.nodes_velocity_p * self.DELTA_T + 0.5 * u * self.DELTA_T ** 2

        # 记录重心位置
        center = np.mean(self.nodes, axis=0)
        self.center_traj_x.append(center[0])
        self.center_traj_y.append(center[1])

        self.current_step += 1

    # === 辅助绘图函数 ===
    def _render_matplotlib_frame(self, ax, step):
        ax.clear()

        # 1. 画目标 (绿色虚线 + 叉号)
        if self.target_params and self.q_mt is not None:
            ax.plot(self.q_mt[0], self.q_mt[1], 'gx', markersize=10, markeredgewidth=2, label='目标')

            # 记录轨迹用于绘图 (仅Matplotlib模式记录)
            if step > 0 or len(self.traj_x) == 0:
                self.traj_x.append(self.q_mt[0])
                self.traj_y.append(self.q_mt[1])

            if len(self.traj_x) > 1:
                ax.plot(self.traj_x, self.traj_y, 'g--', alpha=0.5, label='目标轨迹')

        # 2. 画重心轨迹 (黑色实线)
        if len(self.center_traj_x) > 1:
            ax.plot(self.center_traj_x, self.center_traj_y, 'k-', linewidth=1.5, alpha=0.8, label='重心轨迹')

        # 3. 画障碍物
        if self.obstacle_params:
            circle = Circle(self.obs_pos, self.obs_r, color='red', alpha=0.3)
            ax.add_patch(circle)
            ax.plot(self.obs_pos[0], self.obs_pos[1], 'r+')

        # 4. 画无人机
        ax.plot(self.nodes[:, 0], self.nodes[:, 1], 'b.', markersize=8, label='无人机')

        # 5. 画连线
        diff = self.nodes[np.newaxis, :, :] - self.nodes[:, np.newaxis, :]
        dist = np.linalg.norm(diff, axis=2)
        rows, cols = np.where(np.triu(dist <= self.R, k=1))
        if len(rows) > 0:
            segs = [[self.nodes[r], self.nodes[c]] for r, c in zip(rows, cols)]
            lc = LineCollection(segs[:800], colors='cyan', linewidths=0.5, alpha=0.6)
            ax.add_collection(lc)

        ax.set_title(f"Step: {step}")
        ax.axis('equal')
        ax.grid(True)

        # 处理图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # ------------------ 模式一：Matplotlib 动画（定时器驱动，非阻塞） ------------------
    # 定时器间隔(毫秒)。注意：这只是“请求”的间隔，实际每帧耗时约 45ms
    # （清空坐标轴 + 重建图例 + Qt 重绘），所以真实帧率约 22fps，调小这个值没有效果。
    # 想真正提高帧率要缩小画布（见 run_matplotlib 里的 figsize）。
    MATPLOTLIB_INTERVAL_MS = 20
    # 每帧推进的物理步数 = 播放倍速。
    # 实测每帧真实耗时约 45ms（Python 建图 ~5ms + Qt 重绘 ~40ms）≈ 22fps，
    # 所以 6 步/帧 ≈ 132 步/秒 ≈ 1.2 秒仿真时间/秒，γ 约 9 秒、β 约 17 秒跑完。
    # 想更快就调大（8~10），但超过 10 后每帧跳得太远，编队收拢过程会像"瞬移"。
    # 注意：这个值只影响播放快慢，不影响物理模型；改 DELTA_T 才会改动物理，但
    # β 的 c1_beta=500 斥力很硬，加大 DELTA_T 会逼近数值发散，不要动。
    MATPLOTLIB_STEPS_PER_FRAME = 6

    # 步数安全上限（只是兜底，正常情况由下面的“收敛判据”提前结束）。
    # 实测（N=50, D=15, K=1.2, A=B=5, C1α=30）：γ 约 1500 步收敛，β 约 2000 步收敛，
    # 旧代码 γ/β 一律用 2000 步作上限，β 勉强够、余量极小，看起来就像“没跑完就被掐了”。
    MAX_STEPS_ALPHA = 600         # α 无目标，论文 Algorithm 1 初值一般时本就碎裂，跑固定步数即可
    MAX_STEPS_GAMMA = 5000        # γ：静态目标约 1500 步收敛；动目标不判收敛，由上限兜底
    MAX_STEPS_BETA = 8000         # β：约 2000 步收敛（中途有避障斥力尖峰），留足余量

    # 收敛判据：全场 max|v_i - v_target| 连续低于阈值，判定编队已稳定成形
    CONVERGE_TOL = 0.05
    CONVERGE_FRAMES = 25          # 需连续满足 25 帧（≈50 步 ≈1 秒），避免瞬时误判
    CONVERGE_MIN_STEP = 200       # 初始速度为 0，前 200 步内不判收敛（否则会立刻“收敛”）

    @property
    def mode_name(self):
        if self.target_params is None:
            return 'α'
        return 'β' if self.obstacle_params else 'γ'

    def run_matplotlib(self):
        """启动动画窗口后立即返回，由 matplotlib 自己的定时器逐帧推进物理仿真。

        旧实现用的是 `for t in range(...) + plt.pause()` 的阻塞循环：整段仿真在
        1 秒内一次性算完，屏幕上看不到过程，最后 plt.show() 把画面永久冻在最后一帧。
        """
        print(f">>> 启动 Matplotlib 仿真模式（{self.mode_name}）")
        plt.ion()
        # 画布从 (10, 8) 缩到 (7, 5.6)：Qt 重绘耗时约按面积下降，帧率从 ~22fps 提到 ~35fps
        fig, ax = plt.subplots(figsize=(7, 5.6))

        if self.target_params is None:
            self._mp_limit = self.MAX_STEPS_ALPHA
        elif self.obstacle_params is not None:
            self._mp_limit = self.MAX_STEPS_BETA
        else:
            self._mp_limit = self.MAX_STEPS_GAMMA

        self._mp_fig, self._mp_ax = fig, ax
        self._mp_stable_frames = 0
        self._mp_stop_reason = None

        # 先把初始分布画出来并立即上屏
        self._render_matplotlib_frame(ax, 0)
        fig.canvas.draw()
        plt.show(block=False)

        # canvas.new_timer 会挂到 Qt 事件循环上，不会卡住主窗口
        timer = fig.canvas.new_timer(interval=self.MATPLOTLIB_INTERVAL_MS)
        timer.add_callback(self._matplotlib_tick)
        timer.start()
        self._mp_timer = timer   # 保持引用，否则定时器会被回收

        # 关键：挂到模块级强引用表，防止整个“仿真对象 + 定时器”引用环被 gc 回收
        _ACTIVE_SIMULATIONS[id(self)] = self

        # 用户手动关掉图窗时，停表并解除强引用
        fig.canvas.mpl_connect('close_event', lambda _e: self.stop_matplotlib("用户关闭了窗口"))

    def _matplotlib_tick(self):
        """动画的每一帧：推进若干物理步后重绘。返回 False 会让定时器停止。"""
        for _ in range(self.MATPLOTLIB_STEPS_PER_FRAME):
            if self.current_step >= self._mp_limit:
                self._finish_matplotlib(f"达到步数上限 {self._mp_limit}")
                return False
            self._physics_step()

        # 收敛判据：只对带目标的 γ/β 有意义（α 无目标且论文本身就碎裂，不判收敛）
        if self.target_params is not None and self.current_step >= self.CONVERGE_MIN_STEP:
            v_target = self.p_mt if self.p_mt is not None else np.zeros(2)
            v_err = float(np.max(np.linalg.norm(self.nodes_velocity_p - v_target, axis=1)))
            if v_err < self.CONVERGE_TOL:
                self._mp_stable_frames += 1
                if self._mp_stable_frames >= self.CONVERGE_FRAMES:
                    self._render_matplotlib_frame(self._mp_ax, self.current_step)
                    self._mp_fig.canvas.draw_idle()
                    self._finish_matplotlib("编队已稳定收敛")
                    return False
            else:
                self._mp_stable_frames = 0

        self._render_matplotlib_frame(self._mp_ax, self.current_step)
        self._mp_fig.canvas.draw_idle()   # 交给 Qt 事件循环重绘，不阻塞

        if self.current_step % 500 == 0:
            print(f"    ... {self.mode_name} 模式已推进到 Step = {self.current_step}")
        return True

    def _mp_stop_timer(self):
        timer = getattr(self, '_mp_timer', None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._mp_timer = None

    def _finish_matplotlib(self, reason):
        """仿真自然结束（收敛 / 到上限）：停表，画面留在最后一帧。"""
        self._mp_stop_timer()
        self._mp_stop_reason = reason
        print(f">>> {self.mode_name} 模式仿真结束（{reason}），画面停在 Step = {self.current_step}")

    def stop_matplotlib(self, reason="手动停止"):
        """图窗关闭或外部主动停止：停表并解除强引用，防止内存泄漏。"""
        self._mp_stop_timer()
        _ACTIVE_SIMULATIONS.pop(id(self), None)
        print(f">>> {self.mode_name} 模式仿真停止（{reason}），停在 Step = {self.current_step}")


    # ------------------ 模式二：地图模式 单步更新 ------------------
    def update_map_step(self):
        if self.current_step >= self.MAX_ITERATION:
            return False

        self._physics_step()

        if self.current_step % 3 == 0 and self.map_widget:
            diff = self.nodes[np.newaxis, :, :] - self.nodes[:, np.newaxis, :]
            dist = np.linalg.norm(diff, axis=2)
            rows, cols = np.where(np.triu(dist <= self.R, k=1))

            adj = []
            if len(rows) > 0:
                limit = 150
                for r, c in zip(rows[:limit], cols[:limit]):
                    adj.append([int(r), int(c)])

            self.map_widget.update_scene(self.nodes, adj, self.obstacle_params, self.q_mt)

        return True
