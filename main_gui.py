# main_gui.py
import os
import re
import sys

# === 环境变量设置 ===
# 1. Linux兼容性修复
os.environ["QT_XCB_GL_INTEGRATION"] = "none"
# 2. 针对 WebEngine 禁用 GPU (防止地图黑屏/闪退)
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-software-rasterizer"
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

# 3. 强制使用可交互的 Qt 后端，并锁死 Qt 绑定为 PyQt5
#    PyCharm 的 Scientific Mode 会把 matplotlib 后端换成 module://backend_interagg，
#    该后端只会把每一帧导出成一张静态 PNG 丢进 "Plots" 面板 —— 动画会退化成一堆静止图。
#    QT_API 必须先设：本进程只用 PyQt5(Qt5)，若让 matplotlib 自己去挑，
#    它可能选中 PySide6(Qt6)，同进程混用 Qt5/Qt6 会警告甚至崩溃。
#    这两句必须在 import matplotlib.pyplot（即 import simulation）之前执行。
os.environ.setdefault("QT_API", "pyqt5")
try:
    import matplotlib
    matplotlib.use("qtagg", force=True)
except Exception as _e:
    print("警告：强制 Qt 后端失败，将使用 matplotlib 默认后端 ->", _e)

from PyQt5.QtWidgets import (QApplication, QWidget, QMessageBox, QDialog,
                             QPushButton, QGridLayout, QFrame, QSlider, QLabel, QTabWidget)
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from simulation import FlockingSimulation
from map_widget import MapWidget

# === 路径基准 ===
# 一律以“本文件所在目录”定位资源，这样无论工作目录是哪儿（PyCharm / 命令行 /
# 双击运行）都能找到 .ui 与 assets 里的图片，不会出现“换台机器就丢背景图”。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
UI_FILE = os.path.join(BASE_DIR, "drone_formation.ui")
LOGO_FILE = os.path.join(ASSETS_DIR, "logo.png")


def _abs_asset_urls(css):
    """把样式表里的 url(图片名) 换成 assets/ 下的绝对路径。

    坑：Qt 解析样式表的 url() 是按“进程当前工作目录”找文件，而不是按 .ui 文件
    所在目录。图片挪进 assets/ 之后，换个目录启动就会丢背景图，所以这里换成绝对路径。
    """
    def repl(m):
        name = m.group(1).strip('\'"')
        if os.path.isabs(name):
            return m.group(0)
        for cand in (os.path.join(ASSETS_DIR, name),
                     os.path.join(BASE_DIR, name),
                     os.path.join(ASSETS_DIR, "images", name)):
            if os.path.exists(cand):
                return "url('" + cand.replace("\\", "/") + "')"
        return m.group(0)

    return re.sub(r"url\(\s*([^)]+?)\s*\)", repl, css)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        try:
            uic.loadUi(UI_FILE, self)
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", f"找不到界面文件：{UI_FILE}")
            sys.exit(1)

        self._apply_assets()

        # 定时器用于驱动地图仿真的物理引擎
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_map_loop)

        self.sim = None
        self.map_dialog = None
        self.mp_sim = None          # 当前正在跑的 matplotlib 动画（强引用，见 _launch_matplotlib）

        self.bind_controls()
        self.update_ui_state()

    def _apply_assets(self):
        """修正样式表里的图片路径（改为 assets/ 的绝对路径），并设置窗口图标。"""
        for w in [self] + self.findChildren(QWidget):
            css = w.styleSheet()
            if css and "url(" in css:
                w.setStyleSheet(_abs_asset_urls(css))

        if os.path.exists(LOGO_FILE):
            icon = QIcon(LOGO_FILE)
            self.setWindowIcon(icon)
            QApplication.instance().setWindowIcon(icon)   # 地图弹窗等顶层窗口也带上

    def bind_controls(self):
        # === 绑定 Matplotlib 仿真按钮 (普通模式) ===
        if hasattr(self, 'btn_start'):
            self.btn_start.clicked.connect(self.on_start_alpha_clicked)
        if hasattr(self, 'btn_start_gamma'):
            self.btn_start_gamma.clicked.connect(self.on_start_gamma_clicked)
        if hasattr(self, 'btn_start_obstacle'):
            self.btn_start_obstacle.clicked.connect(self.on_start_obstacle_clicked)

        # === 绑定 地图仿真按钮 (地图模式) ===
        if hasattr(self, 'btn_map_alpha'):
            self.btn_map_alpha.clicked.connect(self.on_map_alpha_clicked)
        if hasattr(self, 'btn_map_gamma'):
            self.btn_map_gamma.clicked.connect(self.on_map_gamma_clicked)
        if hasattr(self, 'btn_map_obstacle'):
            self.btn_map_obstacle.clicked.connect(self.on_map_obstacle_clicked)

        # 绑定单选框状态切换
        if hasattr(self, 'radioButton_static'):
            self.radioButton_static.toggled.connect(self.update_ui_state)
            self.radioButton_sine.toggled.connect(self.update_ui_state)
            self.radioButton_circle.toggled.connect(self.update_ui_state)

    def update_ui_state(self):
        is_sine = self.radioButton_sine.isChecked() if hasattr(self, 'radioButton_sine') else False
        is_circle = self.radioButton_circle.isChecked() if hasattr(self, 'radioButton_circle') else False
        if hasattr(self, 'frame_sine'): self.frame_sine.setEnabled(is_sine)
        if hasattr(self, 'frame_circle'): self.frame_circle.setEnabled(is_circle)

    def get_common_params(self):
        try:
            return {
                'N': self.N.value(),
                'D': self.D.value(),
                'K': self.K.value(),
                'A': self.A.value(),
                'B': self.B.value(),
                'C1_ALPHA': self.C1ALPHA.value()
            }
        except AttributeError:
            QMessageBox.critical(self, "错误", "无法读取通用参数")
            return None

    # ================= 模式 1：Matplotlib 仿真 =================
    def _launch_matplotlib(self, sim):
        """启动动画窗口，并把 sim 挂在 self 上（保持强引用，防止被 GC 回收）。"""
        # 同一时刻只保留一个在跑的动画：新的开始前先停掉上一个，
        # 否则连点按钮会开出多个图窗一起算，CPU 抢起来全都变卡。
        if self.mp_sim is not None:
            self.mp_sim.stop_matplotlib("被新的仿真替换")
        self.mp_sim = sim
        sim.run_matplotlib()

    def on_start_alpha_clicked(self):
        p = self.get_common_params()
        if p:
            sim = FlockingSimulation(**p, target_params=None, obstacle_params=None)
            self._launch_matplotlib(sim)

    def on_start_gamma_clicked(self):
        p = self.get_common_params()
        if not p: return
        t_conf = self.get_target_conf()
        if t_conf:
            sim = FlockingSimulation(**p, target_params=t_conf, obstacle_params=None)
            self._launch_matplotlib(sim)

    def on_start_obstacle_clicked(self):
        p = self.get_common_params()
        if not p: return
        t_conf, o_conf = self.get_beta_conf()
        sim = FlockingSimulation(**p, target_params=t_conf, obstacle_params=o_conf)
        self._launch_matplotlib(sim)

    # ================= 模式 2：地图仿真 (WebEngine) =================
    def on_map_alpha_clicked(self):
        p = self.get_common_params()
        if p:
            self.launch_map_window(p, None, None)

    def on_map_gamma_clicked(self):
        p = self.get_common_params()
        if not p: return
        t_conf = self.get_target_conf()
        if t_conf:
            self.launch_map_window(p, t_conf, None)

    def on_map_obstacle_clicked(self):
        p = self.get_common_params()
        if not p: return
        t_conf, o_conf = self.get_beta_conf()
        self.launch_map_window(p, t_conf, o_conf)

    # ================= 辅助函数：参数读取 =================
    def get_target_conf(self):
        is_static = self.radioButton_static.isChecked()
        is_sine = self.radioButton_sine.isChecked()
        is_circle = self.radioButton_circle.isChecked()

        if not (is_static or is_sine or is_circle):
            QMessageBox.warning(self, "提示", "请先选择目标运动轨迹！")
            return None

        conf = {'pos': [self.pos_x.value(), self.pos_y.value()], 'c1_mt': self.c1_mt.value()}
        if is_static:
            conf['mode'] = 'static'
        elif is_sine:
            conf.update({'mode': 'sine', 'vx': self.spin_speed.value(),
                         'amp': self.spin_amp.value(), 'freq': self.spin_freq.value()})
        elif is_circle:
            conf.update({'mode': 'circle', 'radius': self.spin_radius.value(),
                         'omega': self.spin_angular.value()})
        return conf

    def get_beta_conf(self):
        t_conf = {'mode': 'static', 'pos': [self.beta_target_x.value(), self.beta_target_y.value()],
                  'c1_mt': self.beta_c1_mt.value()}
        o_conf = {'pos': [self.obs_x.value(), self.obs_y.value()], 'radius': self.obs_radius.value(),
                  'c1_beta': self.c1_beta.value(), 'buffer': self.buffer_dist.value()}
        return t_conf, o_conf

    # ================= 地图窗口与控制逻辑 =================
    def launch_map_window(self, params, t_conf, o_conf):
        """启动地图弹窗"""
        if self.timer.isActive(): self.timer.stop()

        # 1. 创建地图控件
        self.map_view = MapWidget()
        self.map_dialog = QDialog(self)
        self.map_dialog.setWindowTitle("无人机编队 - 百度地图实景仿真")
        self.map_dialog.resize(1000, 700)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.map_view, 0, 0, 1, 1)
        self.map_dialog.setLayout(layout)

        # 2. 创建悬浮控制板 (包含 开始/停止/退出 按钮)
        self.create_control_panel(t_conf, o_conf)

        # 3. 初始化仿真器
        self.sim = FlockingSimulation(**params, target_params=t_conf, obstacle_params=o_conf, map_widget=self.map_view)

        # 4. 初始刷新 (让无人机显示在地图上，但保持静止)
        if self.sim:
            self.sim.map_widget.update_scene(self.sim.nodes, [], self.sim.obstacle_params, self.sim.q_mt)

        # 5. 【关键修改】不要立即启动定时器
        # self.timer.start(30)  <-- 删掉或注释这行

        self.map_dialog.finished.connect(self.stop_map_simulation)
        self.map_dialog.show()

    def update_map_loop(self):
        """定时器回调"""
        if self.sim:
            running = self.sim.update_map_step()
            if not running:
                self.timer.stop()
                self.btn_start.setEnabled(False)  # 结束了就禁用开始

    def stop_map_simulation(self):
        self.timer.stop()
        self.sim = None

    def create_control_panel(self, t_conf, o_conf):
        """创建地图上的悬浮控制板"""
        has_sliders = (o_conf is not None)
        panel_height = 200 if has_sliders else 60  # 稍微增加高度以容纳按钮

        panel = QFrame(self.map_view)
        panel.setGeometry(10, 10, 300, panel_height)  # 稍微加宽
        panel.setStyleSheet("QFrame { background: rgba(0, 0, 0, 0.7); border-radius: 8px; } QLabel { color: white; }")

        btn_css = "QPushButton { color: white; border: 1px solid #555; border-radius: 4px; font-weight: bold; font-family: 'Microsoft YaHei'; }"

        # === 1. 开始按钮 ===
        self.btn_start = QPushButton("▶ 开始", panel)
        self.btn_start.setGeometry(10, 15, 80, 30)
        self.btn_start.setStyleSheet(
            btn_css + "QPushButton { background: #52c41a; } QPushButton:disabled { background: #555; color: #aaa; }")
        self.btn_start.clicked.connect(self.map_start)

        # === 2. 停止按钮 (初始禁用) ===
        self.btn_stop = QPushButton("⏸ 停止", panel)
        self.btn_stop.setGeometry(100, 15, 80, 30)
        self.btn_stop.setStyleSheet(
            btn_css + "QPushButton { background: #faad14; } QPushButton:disabled { background: #555; color: #aaa; }")
        self.btn_stop.setEnabled(False)  # 初始静止，所以停止按钮禁用
        self.btn_stop.clicked.connect(self.map_stop)

        # === 3. 退出按钮 ===
        self.btn_quit = QPushButton("✖ 退出", panel)
        self.btn_quit.setGeometry(190, 15, 80, 30)
        self.btn_quit.setStyleSheet(btn_css + "QPushButton { background: #ff4d4f; }")
        self.btn_quit.clicked.connect(lambda: self.map_dialog.close())

        # === 滑块部分 (Beta模式专用) ===
        if has_sliders:
            self.tabs = QTabWidget(panel)
            self.tabs.setGeometry(10, 55, 280, 135)
            self.tabs.setStyleSheet(
                "QTabBar::tab { background: #555; color: white; padding: 5px; } QTabBar::tab:selected { background: #1890ff; }")

            # 障碍物控制
            tab_obs = QWidget()
            self.slider_ox = QSlider(Qt.Horizontal, tab_obs)
            self.slider_ox.setGeometry(0, 30, 240, 20)
            self.slider_ox.setRange(-1000, 1000)
            self.slider_ox.setValue(int(o_conf['pos'][0]))
            self.slider_ox.valueChanged.connect(self.on_obs_slider_change)

            self.slider_oy = QSlider(Qt.Horizontal, tab_obs)
            self.slider_oy.setGeometry(0, 60, 240, 20)
            self.slider_oy.setRange(-1000, 1000)
            self.slider_oy.setValue(int(o_conf['pos'][1]))
            self.slider_oy.valueChanged.connect(self.on_obs_slider_change)

            QLabel("障碍 X 轴", tab_obs).move(0, 10)
            QLabel("障碍 Y 轴", tab_obs).move(0, 80)
            self.tabs.addTab(tab_obs, "移动障碍")

            # 目标控制
            tab_tgt = QWidget()
            self.slider_tx = QSlider(Qt.Horizontal, tab_tgt)
            self.slider_tx.setGeometry(0, 30, 240, 20)
            self.slider_tx.setRange(-1000, 1000)
            self.slider_tx.setValue(int(t_conf['pos'][0]))
            self.slider_tx.valueChanged.connect(self.on_tgt_slider_change)

            self.slider_ty = QSlider(Qt.Horizontal, tab_tgt)
            self.slider_ty.setGeometry(0, 60, 240, 20)
            self.slider_ty.setRange(-1000, 1000)
            self.slider_ty.setValue(int(t_conf['pos'][1]))
            self.slider_ty.valueChanged.connect(self.on_tgt_slider_change)

            QLabel("目标 X 轴", tab_tgt).move(0, 10)
            QLabel("目标 Y 轴", tab_tgt).move(0, 80)
            self.tabs.addTab(tab_tgt, "移动目标")

        panel.show()

    # === 新增：开始/停止 逻辑控制 ===
    def map_start(self):
        """点击开始：启动定时器，切换按钮状态"""
        if not self.timer.isActive():
            self.timer.start(30)  # 启动！
            self.btn_start.setEnabled(False)  # 变灰
            self.btn_start.setText("运行中...")
            self.btn_stop.setEnabled(True)  # 亮起
            self.btn_stop.setText("⏸ 停止")

    def map_stop(self):
        """点击停止：停止定时器，切换按钮状态"""
        if self.timer.isActive():
            self.timer.stop()  # 暂停！
            self.btn_start.setEnabled(True)  # 亮起
            self.btn_start.setText("▶ 继续")
            self.btn_stop.setEnabled(False)  # 变灰
            self.btn_stop.setText("已停止")

    def on_obs_slider_change(self):
        if self.sim:
            self.sim.update_obstacle_pos(self.slider_ox.value(), self.slider_oy.value())

    def on_tgt_slider_change(self):
        if self.sim:
            self.sim.update_target_pos(self.slider_tx.value(), self.slider_ty.value())


if __name__ == '__main__':
    # 启用 OpenGL 上下文共享 (核心修复)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
