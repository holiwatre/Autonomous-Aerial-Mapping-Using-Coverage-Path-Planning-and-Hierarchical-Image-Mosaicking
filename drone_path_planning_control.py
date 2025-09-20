#!/usr/bin/env python3
"""
coverage_photo_mission_fixed_altitude.py

사용법:
    python3 coverage_photo_mission_fixed_altitude.py  # ROS2 환경에서 실행

설명:
    - 다각형 꼭짓점 입력 -> 최적화된 boustrophedon 경로 계산
    - 드론 시작 위치: (0,0)
    - 스캔 스트립 방향 구간에서는 지정 간격마다 호버 및 사진 촬영
    - ±5° 이내 평행하지 않은 구간은 이동만 수행
    - 초기 호버: 시작 위치 5m에서 20초 유지
    - 미션 완료 후 마지막 위치에서 무한 호버링
    - 10Hz 주기로 setpoint 퍼블리시하여 오프보드 제어
    - 호버링 지점마다 libcamera-jpeg로 사진 촬영

변경 사항:
    * 모든 고도를 "5.0"이라는 리터럴 숫자로 직접 사용 (# CHANGED)
    * self.alt 속성 및 관련 파라미터 완전 제거 (# CHANGED)
"""

import math
import time
import subprocess
import threading

from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty


def get_principal_angle(polygon: Polygon) -> float:
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:4]
    edges = []
    for i in range(4):
        p0, p1 = coords[i], coords[(i+1) % 4]
        length = Point(p0).distance(Point(p1))
        angle = math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))
        edges.append((length, angle))
    return max(edges, key=lambda x: x[0])[1]


def generate_boustrophedon_path(coords, spacing, angle_deg):
    poly = Polygon(coords)
    rotated = rotate(poly, -angle_deg, origin='centroid', use_radians=False)
    minx, miny, maxx, maxy = rotated.bounds

    lines = []
    y = miny
    while y <= maxy:
        scan_line = LineString([(minx, y), (maxx, y)])
        inter = rotated.intersection(scan_line)
        if not inter.is_empty:
            if inter.geom_type == 'MultiLineString':
                lines.extend(inter.geoms)
            else:
                lines.append(inter)
        y += spacing

    path = []
    for idx, seg in enumerate(lines):
        pts = list(seg.coords)
        if idx % 2:
            pts.reverse()
        path.extend(pts)

    final = [rotate(Point(p), angle_deg, origin=poly.centroid).coords[0] for p in path]
    return final


class CoveragePhotoMission(Node):
    def __init__(self, full_path, scan_angle, hover_time, hover_interval):
        super().__init__('coverage_photo_mission')
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.setpoint_pub = self.create_publisher(
            PoseStamped, '/dr1/setpoint_position/local', qos)
        self.photo_trigger_pub = self.create_publisher(
            Empty, '/dr1/photo_trigger', qos)

        self.path = full_path
        self.scan_rad = math.radians(scan_angle)
        self.hover_time = hover_time
        self.hover_interval = hover_interval

        self.current_target = None
        self.photo_counter = 1
        self.create_timer(0.1, self._timer_callback)

    def _timer_callback(self):
        if self.current_target is None:
            return
        x, y, z = self.current_target
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.setpoint_pub.publish(msg)
        self.get_logger().info(f"Setpoint -> x:{x:.2f}, y:{y:.2f}, z:{z:.2f}")

    def _capture_photo(self):
        filename = f"image{self.photo_counter}.jpg"
        self.photo_trigger_pub.publish(Empty())
        cmd = ["libcamera-jpeg", "-o", filename, "-n",
               "--width", "2560", "--height", "1440"]
        try:
            subprocess.run(cmd, check=True)
            self.get_logger().info(f"Captured photo: {filename}")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Capture error: {e}")
        time.sleep(2)
        self.photo_counter += 1

    def run_mission(self):
        current_x = 0.0
        current_y = 0.0

        # 초기 호버 고도 5.0m
        self.current_target = (current_x, current_y, 5.0)  # CHANGED
        self.get_logger().info("Phase 1: z=5.0m에서 20초 유지")
        time.sleep(20)

        for idx in range(len(self.path)):
            x_prev, y_prev = (current_x, current_y) if idx == 0 else self.path[idx - 1]
            x_cur, y_cur = self.path[idx]
            dx, dy = x_cur - x_prev, y_cur - y_prev
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                current_x, current_y = x_cur, y_cur
                continue

            v_norm = (dx / dist, dy / dist)
            if abs(v_norm[0] * math.cos(self.scan_rad) +
                   v_norm[1] * math.sin(self.scan_rad)) > math.cos(math.radians(5)):
                steps = int(dist // self.hover_interval)
                for s in range(1, steps + 1):
                    xi = x_prev + v_norm[0] * self.hover_interval * s
                    yi = y_prev + v_norm[1] * self.hover_interval * s
                    self.current_target = (xi, yi, 5.0)  # CHANGED
                    time.sleep(2)
                    self._capture_photo()
                    time.sleep(self.hover_time)
                self.current_target = (x_cur, y_cur, 5.0)  # CHANGED
                time.sleep(2)
            else:
                self.current_target = (x_cur, y_cur, 5.0)  # CHANGED
                time.sleep(2)

            current_x, current_y = x_cur, y_cur

        last_x, last_y = self.path[-1]
        self.get_logger().info("Mission complete. Holding position indefinitely.")
        while rclpy.ok():
            self.current_target = (last_x, last_y, 5.0)  # CHANGED
            time.sleep(1)


def main():
    inp = input("다각형 꼭짓점 좌표 (예: 0,0 10,0 ...):\n")
    coords = [tuple(map(float, p.split(','))) for p in inp.split()]
    spacing = float(input("스캔 간격 (m, 기본 2): ") or 2)
    hover_time = float(input("호버링 시간 (s, 기본 10): ") or 10)
    hover_interval = float(input("호버링 간격 거리 (m, 기본 3): ") or 3)

    angle = get_principal_angle(Polygon(coords))
    print(f"최적 스캔 각도: {angle:.2f}°")
    path_pts = generate_boustrophedon_path(coords, spacing, angle)
    full_path = [(0.0, 0.0)] + path_pts

    rclpy.init()
    node = CoveragePhotoMission(full_path, angle, hover_time, hover_interval)
    mission_thread = threading.Thread(target=node.run_mission)
    mission_thread.start()
    rclpy.spin(node)
    mission_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
