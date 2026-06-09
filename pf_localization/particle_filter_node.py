#!/usr/bin/env python3
"""
Particle Filter Localization Node
===================================
Subscriptions
  /odom           nav_msgs/Odometry        – motion model input
  /ar_detection   std_msgs/Float64MultiArray – [distance, bearing_rad]

Publications
  /particles      geometry_msgs/PoseArray  – particle cloud
                    pose.position.z encodes normalised weight × N
  /pf_estimate    geometry_msgs/Pose       – weighted mean estimate
                    position.z encodes yaw [rad]

Algorithm: standard SIR particle filter (Probabilistic Robotics, Ch. 4)
Motion model  : odometry-based (Thrun et al.)
Sensor model  : multi-hypothesis likelihood  p(z|x) = Σ p(z|x, tag_i)
Resampling    : low-variance resampling, triggered when N_eff < N/2
"""

import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

from pf_localization.constants import (
    TAG_POSITIONS, ROOM_X_MIN, ROOM_X_MAX, ROOM_Y_MIN, ROOM_Y_MAX
)

# ── Tunable parameters ─────────────────────────────────────────────────────
N_PARTICLES = 2000

# Odometry noise (Probabilistic Robotics Table 5.3)
ALPHA = [0.25, 0.15, 0.15, 0.08]  # [a1, a2, a3, a4]

# Sensor noise
SIGMA_DIST  = 0.18   # [m]   distance std
SIGMA_BEAR  = 0.18   # [rad] bearing std

MIN_WEIGHT  = 1e-200  # floor to avoid log-sum underflow
# ───────────────────────────────────────────────────────────────────────────


def quat_to_yaw(q) -> float:
    return math.atan2(2.0*(q.w*q.z + q.x*q.y),
                      1.0 - 2.0*(q.y**2 + q.z**2))


def angle_diff(a: float, b: float) -> float:
    """Signed shortest-arc difference a - b in [-π, π]."""
    d = a - b
    return (d + math.pi) % (2*math.pi) - math.pi


class ParticleFilterNode(Node):

    def __init__(self):
        super().__init__('particle_filter')

        self.lock = threading.Lock()

        # ── Particles: columns = [x, y, θ] ───────────────────────────────
        self.particles = self._uniform_init()
        self.weights   = np.ones(N_PARTICLES) / N_PARTICLES

        # ── Odom state ────────────────────────────────────────────────────
        self.prev_odom  = None   # (x, y, θ)

        # ── Trajectory buffers (for visualiser) ──────────────────────────
        self.pf_traj   = []   # [(x,y), ...]
        self.odom_traj = []   # [(x,y), ...]

        # ── ROS interfaces ────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.det_sub  = self.create_subscription(
            Float64MultiArray, '/ar_detection', self._detection_cb, 10)

        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.estimate_pub  = self.create_publisher(Pose, '/pf_estimate', 10)

        self.get_logger().info(
            f'Particle filter ready — {N_PARTICLES} particles, '
            f'σ_d={SIGMA_DIST} m, σ_α={SIGMA_BEAR} rad')

    # ── Initialisation ────────────────────────────────────────────────────

    def _uniform_init(self) -> np.ndarray:
        p = np.zeros((N_PARTICLES, 3))
        p[:, 0] = np.random.uniform(ROOM_X_MIN, ROOM_X_MAX, N_PARTICLES)
        p[:, 1] = np.random.uniform(ROOM_Y_MIN, ROOM_Y_MAX, N_PARTICLES)
        p[:, 2] = np.random.uniform(-math.pi, math.pi, N_PARTICLES)
        return p

    # ── Motion model (predict step) ───────────────────────────────────────

    def _predict(self, odom_curr, odom_prev):
        """
        Odometry-based motion model (Probabilistic Robotics Alg. 5.4).
        odom_curr / odom_prev: (x, y, θ) tuples
        """
        dx    = odom_curr[0] - odom_prev[0]
        dy    = odom_curr[1] - odom_prev[1]
        dth   = angle_diff(odom_curr[2], odom_prev[2])
        dtran = math.sqrt(dx*dx + dy*dy)

        if dtran > 1e-4:
            drot1 = angle_diff(math.atan2(dy, dx), odom_prev[2])
        else:
            drot1 = 0.0
        drot2 = angle_diff(dth, drot1)

        a1, a2, a3, a4 = ALPHA
        N = N_PARTICLES

        def safe_std(v2):
            return math.sqrt(max(v2, 1e-12))

        hat_rot1  = drot1  + np.random.normal(
            0, safe_std(a1*drot1**2  + a2*dtran**2), N)
        hat_tran  = dtran  + np.random.normal(
            0, safe_std(a3*dtran**2  + a4*(drot1**2 + drot2**2)), N)
        hat_rot2  = drot2  + np.random.normal(
            0, safe_std(a1*drot2**2  + a2*dtran**2), N)

        self.particles[:, 0] += hat_tran * np.cos(self.particles[:, 2] + hat_rot1)
        self.particles[:, 1] += hat_tran * np.sin(self.particles[:, 2] + hat_rot1)
        self.particles[:, 2] += hat_rot1 + hat_rot2
        self.particles[:, 2]  = (self.particles[:, 2] + math.pi) % (2*math.pi) - math.pi

        # Clip to room (cannot pass through walls)
        self.particles[:, 0] = np.clip(
            self.particles[:, 0], ROOM_X_MIN + 0.05, ROOM_X_MAX - 0.05)
        self.particles[:, 1] = np.clip(
            self.particles[:, 1], ROOM_Y_MIN + 0.05, ROOM_Y_MAX - 0.05)

    # ── Sensor model (update step) ────────────────────────────────────────

    def _update(self, z_dist: float, z_bear: float):
        """
        Multi-hypothesis likelihood:
            p(z | x_particle) = Σ_{i=1}^{8} p(z | x_particle, tag_i)
        where each term is a product of two Gaussians (distance, bearing).
        """
        px = self.particles[:, 0]
        py = self.particles[:, 1]
        pθ = self.particles[:, 2]

        likelihood = np.zeros(N_PARTICLES)

        for tx, ty in TAG_POSITIONS:
            dx = tx - px
            dy = ty - py

            exp_dist = np.sqrt(dx**2 + dy**2)
            exp_bear = np.arctan2(dy, dx) - pθ
            # normalise bearing to [-π, π]
            exp_bear = (exp_bear + math.pi) % (2*math.pi) - math.pi

            bear_diff = z_bear - exp_bear
            bear_diff = (bear_diff + math.pi) % (2*math.pi) - math.pi

            p_dist = np.exp(-0.5 * ((z_dist - exp_dist) / SIGMA_DIST)**2)
            p_bear = np.exp(-0.5 * (bear_diff / SIGMA_BEAR)**2)

            likelihood += p_dist * p_bear

        self.weights *= (likelihood + MIN_WEIGHT)
        total = self.weights.sum()
        if total > 1e-15:
            self.weights /= total
        else:
            self.get_logger().warn('Weight collapse — re-initialising particles')
            self.particles = self._uniform_init()
            self.weights   = np.ones(N_PARTICLES) / N_PARTICLES
            return

        # Low-variance resampling when N_eff < N/2
        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-300)
        if n_eff < N_PARTICLES / 2:
            self._resample()

    # ── Low-variance resampling ───────────────────────────────────────────

    def _resample(self):
        cumsum = np.cumsum(self.weights)
        r = np.random.uniform(0, 1.0 / N_PARTICLES)
        step = 1.0 / N_PARTICLES
        idx = []
        j = 0
        for m in range(N_PARTICLES):
            u = r + m * step
            while u > cumsum[j]:
                j += 1
                if j >= N_PARTICLES:
                    j = N_PARTICLES - 1
                    break
            idx.append(j)
        self.particles = self.particles[idx]
        self.weights   = np.ones(N_PARTICLES) / N_PARTICLES

    # ── Pose estimate ─────────────────────────────────────────────────────

    def _weighted_mean(self):
        w = self.weights
        x = np.sum(w * self.particles[:, 0])
        y = np.sum(w * self.particles[:, 1])
        # Circular mean for angle
        θ = math.atan2(
            np.sum(w * np.sin(self.particles[:, 2])),
            np.sum(w * np.cos(self.particles[:, 2])))
        return x, y, θ

    # ── Publishers ────────────────────────────────────────────────────────

    def _publish(self):
        stamp = self.get_clock().now().to_msg()

        # Particle cloud
        pa = PoseArray()
        pa.header.stamp    = stamp
        pa.header.frame_id = 'map'
        for i in range(N_PARTICLES):
            p = Pose()
            p.position.x = self.particles[i, 0]
            p.position.y = self.particles[i, 1]
            p.position.z = float(self.weights[i] * N_PARTICLES)  # encoded weight
            pa.poses.append(p)
        self.particles_pub.publish(pa)

        # Weighted mean
        ex, ey, eθ = self._weighted_mean()
        ep = Pose()
        ep.position.x = ex
        ep.position.y = ey
        ep.position.z = eθ   # yaw encoded in z
        self.estimate_pub.publish(ep)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        curr = (x, y, yaw)

        with self.lock:
            if self.prev_odom is None:
                self.prev_odom = curr
                return

            # Predict
            self._predict(curr, self.prev_odom)
            self.prev_odom = curr

            # Track odom trajectory
            self.odom_traj.append((x, y))
            if len(self.odom_traj) > 1000:
                self.odom_traj.pop(0)

            self._publish()

    def _detection_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            return
        z_dist = msg.data[0]
        z_bear = msg.data[1]

        with self.lock:
            self._update(z_dist, z_bear)

            # Track pf trajectory
            ex, ey, _ = self._weighted_mean()
            self.pf_traj.append((ex, ey))
            if len(self.pf_traj) > 1000:
                self.pf_traj.pop(0)

            self._publish()


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
