"""
pf_sim.launch.py
=================
Launches:
  1. Gazebo Ignition Fortress  (ar_room.sdf world)
  2. ros_gz_bridge             (Ignition ↔ ROS 2 topic bridge)
  3. aruco_detector_node       (real camera-based ArUco detection)
  4. particle_filter_node      (the PF)
  5. visualizer_node           (matplotlib window)

Usage:
  ros2 launch pf_localization pf_sim.launch.py

Then in a separate terminal drive the robot:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/cmd_vel
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                             IncludeLaunchDescription, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('pf_localization')
    world_file = os.path.join(pkg_share, 'worlds', 'ar_room.sdf')

    # ── 1. Gazebo Ignition Fortress ────────────────────────────────────
    # Prepend the installed worlds directory so Gazebo can resolve the
    # file:// texture paths declared in ar_room.sdf.
    worlds_dir = os.path.join(pkg_share, 'worlds')
    _existing = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ign_resource_path = (worlds_dir + ':' + _existing) if _existing else worlds_dir

    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_file, '--force-version', '6'],
        additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_resource_path},
        output='screen',
    )

    # ── 2. ROS ↔ Gazebo bridge ────────────────────────────────────────
    #
    # Bridge syntax:
    #   topic@ros_type[ign_type   →  Gazebo→ROS
    #   topic@ros_type]ign_type   →  ROS→Gazebo
    #
    bridge_args = [
        # Odometry: Gazebo → ROS
        '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        # Cmd vel: ROS → Gazebo
        '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
        # Camera image: Gazebo → ROS  (optional, for camera-based detection)
        '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
        # Simulation clock
        '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
    ]

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
        output='screen',
        # If ros_gz_bridge is not installed, try ros_ign_bridge:
        # package='ros_ign_bridge',
    )

    # ── 3-5. Our nodes ────────────────────────────────────────────────
    detector_node = Node(
        package='pf_localization',
        executable='aruco_detector',
        name='aruco_detector',
        output='screen',
    )

    pf_node = Node(
        package='pf_localization',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
    )

    viz_node = Node(
        package='pf_localization',
        executable='visualizer',
        name='visualizer',
        output='screen',
    )

    # Delay PF nodes slightly so Gazebo/bridge can initialise
    delayed_nodes = TimerAction(
        period=4.0,
        actions=[detector_node, pf_node, viz_node],
    )

    return LaunchDescription([
        gazebo,
        bridge,
        delayed_nodes,
    ])
