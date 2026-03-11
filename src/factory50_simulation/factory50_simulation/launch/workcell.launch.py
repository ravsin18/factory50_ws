"""
factory50_simulation — workcell.launch.py
Launches the complete Factory 5.0 workcell simulation:
  - Gazebo with workcell.sdf world
  - UR5e robot (arm0) with ros2_control
  - Robot state publishers for both arms
  - ROS-Gazebo bridge for camera + joint topics
  - RViz2 for visualization
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Package Paths ─────────────────────────────────────────────────────────
    sim_pkg     = get_package_share_directory('factory50_simulation')
    ur_desc_pkg = get_package_share_directory('ur_description')

    # ── Launch Arguments ──────────────────────────────────────────────────────
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz2 for visualization')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time')

    use_rviz     = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── 1. Launch Gazebo with workcell world ──────────────────────────────────
    world_file = os.path.join(sim_pkg, 'worlds', 'workcell.sdf')

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen'
    )

    # ── 2. UR5e Robot — Arm 0 (LEFT) ─────────────────────────────────────────
    ur_description_file = os.path.join(ur_desc_pkg, 'urdf', 'ur.urdf.xacro')

    # Robot state publisher for UR5e
    arm0_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='arm0_state_publisher',
        namespace='arm0',
        parameters=[{
            'robot_description': _load_ur5e_urdf(ur_description_file),
            'use_sim_time': use_sim_time,
            'frame_prefix': 'arm0/'
        }],
        output='screen'
    )

    # Spawn UR5e into Gazebo at left position
    arm0_spawn = TimerAction(
        period=3.0,  # wait for Gazebo to fully load
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                name='spawn_arm0',
                arguments=[
                    '-name',  'ur5e_arm0',
                    '-topic', '/arm0/robot_description',
                    '-x',     '-0.65',
                    '-y',      '0.0',
                    '-z',      '0.72',
                    '-R',      '0',
                    '-P',      '0',
                    '-Y',      '0',
                ],
                output='screen'
            )
        ]
    )

    # ── 3. ROS-Gazebo Bridge ──────────────────────────────────────────────────
    # Bridges Gazebo topics → ROS2 topics
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        arguments=[
            # Clock
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Camera topics
            '/camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            # Joint states
            '/arm0/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/arm1/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            # Velocity commands
            '/arm0/joint_cmd@trajectory_msgs/msg/JointTrajectory]gz.msgs.JointTrajectory',
        ],
        output='screen'
    )

    # ── 4. Joint State Publisher (for visualization without controller) ────────
    arm0_jsp = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='arm0_joint_state_publisher',
        namespace='arm0',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # ── 5. RViz2 ─────────────────────────────────────────────────────────────
    rviz_config = os.path.join(sim_pkg, 'rviz', 'workcell.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=_if_true(use_rviz)
    )

    # ── 6. Foxglove Bridge (Digital Twin) ─────────────────────────────────────
    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': 8765,
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )

    return LaunchDescription([
        use_rviz_arg,
        use_sim_time_arg,
        gazebo,
        bridge,
        arm0_rsp,
        arm0_jsp,
        arm0_spawn,
        foxglove,
        rviz,
    ])


def _load_ur5e_urdf(xacro_file: str) -> str:
    """Process UR5e xacro file and return URDF string."""
    import xacro
    doc = xacro.process_file(
        xacro_file,
        mappings={
            'ur_type':          'ur5e',
            'name':             'ur5e',
            'prefix':           'arm0_',
            'use_fake_hardware': 'true',
            'fake_sensor_commands': 'true',
            'sim_gazebo':       'true',
        }
    )
    return doc.toxml()


def _if_true(condition):
    """Helper for conditional launch."""
    from launch.conditions import IfCondition
    return IfCondition(condition)
