#!/usr/bin/env python3
"""
ArUco Detector Node
====================
Detects ArUco markers (DICT_4X4_50, ID 0) in the camera image stream
and publishes distance + bearing observations for the particle filter.

Subscriptions
  /camera/image    sensor_msgs/msg/Image

Publications
  /ar_detection    std_msgs/msg/Float64MultiArray
                     data = [distance_m, bearing_rad]
                     One message published per detected marker per frame.
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from cv_bridge import CvBridge

# Camera intrinsics — derived from SDF: hfov=60° (1.0472 rad), 640×480
#   fx = (width/2) / tan(hfov/2) = 320 / tan(30°) ≈ 554.26
_FX = 554.26
_FY = 554.26
_CX = 320.0
_CY = 240.0
_CAM_MTX = np.array([[_FX,  0.0, _CX],
                      [0.0, _FY,  _CY],
                      [0.0,  0.0,  1.0]], dtype=np.float64)
_DIST = np.zeros((4, 1), dtype=np.float64)

# Physical marker size: 0.25 m side length
# Corner order matches what OpenCV returns: TL, TR, BR, BL
_HALF = 0.25 / 2.0
_OBJ_PTS = np.array([
    [-_HALF,  _HALF, 0],
    [ _HALF,  _HALF, 0],
    [ _HALF, -_HALF, 0],
    [-_HALF, -_HALF, 0]], dtype=np.float64)


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        self._bridge = CvBridge()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params     = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        self.det_pub = self.create_publisher(
            Float64MultiArray, '/ar_detection', 10)
        self.create_subscription(
            Image, '/camera/image', self._image_cb, 10)

        self._frame_count = 0

        self.get_logger().info('ArUco detector ready — listening on /camera/image')

    def _image_cb(self, msg: Image):
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(
                f'cv_bridge conversion failed: {e}',
                throttle_duration_sec=5.0)
            return

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        self._frame_count += 1
        log_this_frame = (self._frame_count % 30 == 0)

        if ids is None or len(ids) == 0:
            if log_this_frame:
                self.get_logger().info('ArUco detector — no markers detected')
            return

        detected = 0
        last_dist = last_bear = 0.0

        for i in range(len(ids)):
            img_pts = corners[i].reshape(4, 2).astype(np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                _OBJ_PTS, img_pts, _CAM_MTX, _DIST,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)

            if not ok:
                continue

            distance = float(np.linalg.norm(tvec))
            bearing  = float(math.atan2(tvec[0][0], tvec[2][0]))

            out = Float64MultiArray()
            out.data = [distance, bearing]
            self.det_pub.publish(out)

            detected += 1
            last_dist, last_bear = distance, bearing

        if log_this_frame and detected > 0:
            self.get_logger().info(
                f'Detected {detected} marker(s) — '
                f'dist={last_dist:.2f} m, bearing={last_bear:.2f} rad')


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
