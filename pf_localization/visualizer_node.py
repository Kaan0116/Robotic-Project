#!/usr/bin/env python3
"""
Visualizer Node
================
Real-time matplotlib display showing:
  • Room outline and AR tag positions (red squares)
  • Particle cloud coloured by normalised weight  (plasma colormap)
  • Odometry-only trajectory (blue)
  • Particle-filter pose estimate trajectory (green)

Subscriptions
  /particles    geometry_msgs/PoseArray  – particle cloud from PF node
  /pf_estimate  geometry_msgs/Pose       – weighted-mean estimate from PF node
  /odom         nav_msgs/Odometry        – for odometry-only trajectory

ROS spinning happens in a background thread; matplotlib updates in the
main thread at ~10 Hz via plt.pause().

Screenshots are saved to ~/pf_ws/screenshots/ at three ESS milestones:
  initial_spread.png      ESS > 0.80 * N  (particles still spread)
  partial_convergence.png 0.30*N < ESS < 0.80*N
  converged.png           ESS < 0.30 * N  (filter has converged)
"""

import math
import os
import threading

import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' if TkAgg is not available
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry

from pf_localization.constants import (
    TAG_POSITIONS, ROOM_X_MIN, ROOM_X_MAX, ROOM_Y_MIN, ROOM_Y_MAX
)

MAX_TRAJ_LEN   = 800
SCREENSHOT_DIR = os.path.expanduser('~/pf_ws/screenshots')


class VisualizerNode(Node):

    def __init__(self):
        super().__init__('visualizer')

        self.lock = threading.Lock()

        # Data buffers
        self.particles       = np.zeros((2000, 3))
        self.weights         = np.ones(2000) / 2000
        self.pf_trajectory   = []   # [(x, y), ...]
        self.odom_trajectory = []   # [(x, y), ...]

        # Screenshot state — save each milestone at most once,
        # but only after the first real /particles message arrives.
        self._particles_received = False
        self._saved = {
            'initial_spread':      False,
            'partial_convergence': False,
            'converged':           False,
        }
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

        # Subscriptions
        self.create_subscription(
            PoseArray, '/particles', self._particles_cb, 10)
        self.create_subscription(
            Pose, '/pf_estimate', self._estimate_cb, 10)
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)

        self.get_logger().info('Visualizer ready — opening matplotlib window')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _particles_cb(self, msg: PoseArray):
        n = len(msg.poses)
        if n == 0:
            return
        pts = np.zeros((n, 3))
        wts = np.zeros(n)
        for i, pose in enumerate(msg.poses):
            pts[i, 0] = pose.position.x
            pts[i, 1] = pose.position.y
            pts[i, 2] = 0.0
            wts[i]    = pose.position.z / n   # un-encode weight
        with self.lock:
            self.particles           = pts
            self.weights             = wts
            self._particles_received = True

    def _estimate_cb(self, msg: Pose):
        with self.lock:
            self.pf_trajectory.append((msg.position.x, msg.position.y))
            if len(self.pf_trajectory) > MAX_TRAJ_LEN:
                self.pf_trajectory.pop(0)

    def _odom_cb(self, msg: Odometry):
        with self.lock:
            self.odom_trajectory.append((
                msg.pose.pose.position.x,
                msg.pose.pose.position.y))
            if len(self.odom_trajectory) > MAX_TRAJ_LEN:
                self.odom_trajectory.pop(0)

    # ── One-time figure setup ──────────────────────────────────────────────

    def setup_figure(self):
        """Create figure, axes, and all artists once.

        The scatter is initialised with the current particles buffer
        (same N as the PF publishes) so that set_offsets / set_array
        never encounter a size mismatch — which would silently drop
        the color update in some matplotlib versions.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        self._fig = fig          # stored for screenshot saving
        plt.tight_layout()

        # Static background
        ax.set_facecolor('#f9f9e8')
        room = mpatches.FancyBboxPatch(
            (ROOM_X_MIN, ROOM_Y_MIN),
            ROOM_X_MAX - ROOM_X_MIN,
            ROOM_Y_MAX - ROOM_Y_MIN,
            boxstyle='square,pad=0',
            linewidth=3, edgecolor='#333333', facecolor='#f5f5dc', zorder=1)
        ax.add_patch(room)

        for i, (tx, ty) in enumerate(TAG_POSITIONS):
            ax.plot(tx, ty, 's', color='#cc0000', markersize=14, zorder=5,
                    markeredgecolor='black', markeredgewidth=0.8)
            ax.text(tx + 0.08, ty + 0.10, f'T{i}', fontsize=7,
                    color='#800000', fontweight='bold', zorder=6)

        # Particle scatter — initialised with the SAME number of points that
        # draw() will supply, avoiding colour-array size mismatches.
        init_w = self.weights.copy()
        w_max  = init_w.max()
        init_c = (init_w / w_max) if w_max > 1e-15 else np.ones(len(init_w)) * 0.5
        self._sc = ax.scatter(
            self.particles[:, 0], self.particles[:, 1],
            c=init_c, cmap='plasma', vmin=0, vmax=1,
            s=8, alpha=0.70, zorder=3, linewidths=0)
        fig.colorbar(self._sc, ax=ax, fraction=0.03, pad=0.02,
                     label='Normalised weight')

        # Trajectory line artists — created once, data updated each frame
        self._odom_line, = ax.plot([], [], '-', color='#3399ff',
                                   linewidth=1.8, label='Odometry only', zorder=4)
        self._odom_dot,  = ax.plot([], [], 'o', color='#0055aa',
                                   markersize=7, zorder=7)
        self._pf_line,   = ax.plot([], [], '-', color='#22aa44',
                                   linewidth=2.2, label='PF estimate', zorder=4)
        self._pf_star,   = ax.plot([], [], '*', color='#006622',
                                   markersize=13, zorder=7)

        # Fixed decorations
        ax.set_xlim(ROOM_X_MIN - 0.4, ROOM_X_MAX + 0.4)
        ax.set_ylim(ROOM_Y_MIN - 0.4, ROOM_Y_MAX + 0.4)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]', fontsize=10)
        ax.set_ylabel('Y [m]', fontsize=10)
        ax.set_title('Particle Filter Localization — AR Tag Room', fontsize=12)
        ax.grid(True, alpha=0.25)

        tag_patch = mpatches.Patch(color='#cc0000', label='AR tag (ID 0)')
        ax.legend(handles=[self._odom_line, self._pf_line, tag_patch],
                  loc='upper right', fontsize=9)

        return fig, ax

    # ── Per-frame update ───────────────────────────────────────────────────

    def draw(self):
        """Update mutable artists in-place, compute ESS, and save screenshots."""
        with self.lock:
            particles         = self.particles.copy()
            weights           = self.weights.copy()
            pf_traj           = list(self.pf_trajectory)
            odom_traj         = list(self.odom_trajectory)
            particles_arrived = self._particles_received

        n = len(weights)

        # ── Particle cloud ─────────────────────────────────────────────
        if n > 0:
            w_max  = weights.max()
            w_norm = (weights / w_max) if w_max > 1e-15 else np.ones(n) * 0.5
            self._sc.set_offsets(particles[:, :2])
            self._sc.set_array(w_norm)

        # ── Odometry trajectory (blue) ──────────────────────────────────
        if len(odom_traj) > 1:
            oa = np.array(odom_traj)
            self._odom_line.set_data(oa[:, 0], oa[:, 1])
            self._odom_dot.set_data([oa[-1, 0]], [oa[-1, 1]])
        else:
            self._odom_line.set_data([], [])
            self._odom_dot.set_data([], [])

        # ── PF estimate trajectory (green) ──────────────────────────────
        if len(pf_traj) > 1:
            pa = np.array(pf_traj)
            self._pf_line.set_data(pa[:, 0], pa[:, 1])
            self._pf_star.set_data([pa[-1, 0]], [pa[-1, 1]])
        else:
            self._pf_line.set_data([], [])
            self._pf_star.set_data([], [])

        # ── Screenshot milestones ───────────────────────────────────────
        if particles_arrived and n > 0:
            ess = 1.0 / (float(np.sum(weights ** 2)) + 1e-300)
            self._maybe_save_screenshot(ess, n)

    def _maybe_save_screenshot(self, ess: float, n: int):
        """Save figure once per ESS milestone."""
        if not self._saved['initial_spread'] and ess > 0.8 * n:
            self._save_png('initial_spread.png', ess, n)

        elif not self._saved['partial_convergence'] and 0.3 * n < ess <= 0.8 * n:
            self._save_png('partial_convergence.png', ess, n)

        elif not self._saved['converged'] and ess <= 0.3 * n:
            self._save_png('converged.png', ess, n)

    def _save_png(self, filename: str, ess: float, n: int):
        path = os.path.join(SCREENSHOT_DIR, filename)
        self._fig.savefig(path, dpi=150, bbox_inches='tight')
        self._saved[filename.replace('.png', '')] = True
        self.get_logger().info(
            f'Screenshot saved → {path}  (ESS={ess:.0f} / N={n})')


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()

    # Spin ROS in a daemon thread
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Build figure once, then loop updating artists only
    fig, ax = node.setup_figure()
    plt.ion()

    try:
        while rclpy.ok():
            node.draw()
            fig.canvas.draw()
            plt.pause(0.10)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
