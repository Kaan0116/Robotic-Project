#!/usr/bin/env python3
"""
Detection Simulator Node
=========================
Simulates an AR-tag detector using the robot's ground-truth odometry.

Because the DiffDrive plugin in Gazebo Ignition produces essentially
noise-free odometry (the true pose in sim), this node acts as the
"camera detector" — it computes which tags are inside the robot's
field of view and publishes a noisy observation for each visible tag.

Subscriptions
  /odom   nav_msgs/Odometry  –  robot pose in world frame (GT in sim)

Publications
  /ar_detection  std_msgs/Float64MultiArray – one message per visible tag
                   data = [distance_m, bearing_rad]

Parameters (ROS)
  fov_half_deg    float  default 45.0  –  half-FOV in degrees
  max_range       float  default 3.5   –  detection range [m]
  sigma_dist      float  default 0.05  –  distance noise std [m]
  sigma_bear      float  default 0.03  –  bearing noise std [rad]
  min_publish_gap float  default 0.05  –  min seconds between detections
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

from pf_localization.constants import TAG_POSITIONS


class DetectionSimulatorNode(Node):

    def __init__(self):
        super().__init__('detection_simulator')

        # Declare parameters
        self.declare_parameter('fov_half_deg',    45.0)
        self.declare_parameter('max_range',        3.5)
        self.declare_parameter('sigma_dist',       0.05)
        self.declare_parameter('sigma_bear',       0.03)
        self.declare_parameter('min_publish_gap',  0.05)

        fov_deg  = self.get_parameter('fov_half_deg').value
        self.fov_half       = math.radians(fov_deg)
        self.max_range      = self.get_parameter('max_range').value
        self.sigma_dist     = self.get_parameter('sigma_dist').value
        self.sigma_bear     = self.get_parameter('sigma_bear').value
        self.min_pub_gap    = self.get_parameter('min_publish_gap').value

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self._last_pub = 0.0

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.det_pub = self.create_publisher(
            Float64MultiArray, '/ar_detection', 10)

        # 10 Hz detection loop
        self.create_timer(0.10, self._detect_and_publish)

        self.get_logger().info(
            f'Detection simulator ready — FOV ±{fov_deg}°, '
            f'range {self.max_range} m')

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_yaw(q) -> float:
        return math.atan2(2.0*(q.w*q.z + q.x*q.y),
                          1.0 - 2.0*(q.y**2 + q.z**2))

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2*math.pi) - math.pi

    # ── Callback: odometry ────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.robot_x   = msg.pose.pose.position.x
        self.robot_y   = msg.pose.pose.position.y
        self.robot_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

    # ── Timer: detect and publish ─────────────────────────────────────────

    def _detect_and_publish(self):
        now = time.monotonic()
        if (now - self._last_pub) < self.min_pub_gap:
            return

        visible = []
        for (tx, ty) in TAG_POSITIONS:
            dx = tx - self.robot_x
            dy = ty - self.robot_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > self.max_range:
                continue

            world_bear = math.atan2(dy, dx)
            rel_bear   = self._wrap(world_bear - self.robot_yaw)

            if abs(rel_bear) > self.fov_half:
                continue

            visible.append((dist, rel_bear))

        if not visible:
            return

        # Publish all visible tags (each as a separate message)
        # Shuffling introduces healthy ordering randomness
        np.random.shuffle(visible)
        for dist, bear in visible:
            noisy_dist = dist + np.random.normal(0, self.sigma_dist)
            noisy_bear = bear + np.random.normal(0, self.sigma_bear)
            noisy_dist = max(0.05, noisy_dist)   # physical lower bound
            noisy_bear = self._wrap(noisy_bear)

            msg = Float64MultiArray()
            msg.data = [noisy_dist, noisy_bear]
            self.det_pub.publish(msg)

        self._last_pub = now


def main(args=None):
    rclpy.init(args=args)
    node = DetectionSimulatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
