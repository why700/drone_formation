# map_widget.py
import os
import json
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtCore import QUrl


class MapWidget(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        self.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        self.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)

        base_path = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(base_path, "baidu_map.html")

        html_content = ""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            html_content = f"<h3>Error reading file: {e}</h3>"

        self.setHtml(html_content, QUrl.fromLocalFile(base_path + "/"))

        self.base_lng = 116.404
        self.base_lat = 39.915
        self.scale = 0.00001

        # === 修改：增加 target_pos 参数 ===

    def update_scene(self, nodes, adj_list, obstacle_params=None, target_pos=None):
        geo_nodes = []
        for p in nodes:
            lng = self.base_lng + p[0] * self.scale
            lat = self.base_lat + p[1] * self.scale
            geo_nodes.append([lng, lat])

        geo_obs = None
        if obstacle_params:
            o_pos = obstacle_params['pos']
            geo_obs = {
                "pos": [
                    self.base_lng + o_pos[0] * self.scale,
                    self.base_lat + o_pos[1] * self.scale
                ],
                "radius": obstacle_params['radius']
            }

        # === 处理目标点坐标 ===
        geo_target = None
        if target_pos is not None:
            geo_target = [
                self.base_lng + target_pos[0] * self.scale,
                self.base_lat + target_pos[1] * self.scale
            ]
        # ===================

        center_geo = [self.base_lng, self.base_lat]
        if len(nodes) > 0:
            import numpy as np
            cx = np.mean(nodes[:, 0])
            cy = np.mean(nodes[:, 1])
            center_geo = [self.base_lng + cx * self.scale, self.base_lat + cy * self.scale]

        data = {
            "nodes": geo_nodes,
            "adj": adj_list,
            "obstacle": geo_obs,
            "target": geo_target,  # 传给 JS
            "center": center_geo
        }

        js_code = f"updateMap('{json.dumps(data)}')"
        self.page().runJavaScript(js_code)
