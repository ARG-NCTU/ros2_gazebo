"""
Launch jong_shyn No.5 rover in Gazebo and Rviz

ros2 launch ardupilot_sitl sitl_dds_udp.launch.py
transport:=udp4
port:=2019
synthetic_clock:=True
wipe:=False
model:=json
speedup:=1
slave:=0
instance:=0
defaults:=$(ros2 pkg prefix ardupilot_sitl)
          /share/ardupilot_sitl/config/default_params/rover-skid.parm,
          $(ros2 pkg prefix ardupilot_sitl)
          /share/ardupilot_sitl/config/default_params/dds_udp.parm
sim_address:=127.0.0.1
master:=tcp:127.0.0.1:5760
sitl:=127.0.0.1:5501
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.actions import ExecuteProcess
from launch.actions import OpaqueFunction

from launch.conditions import IfCondition

from launch.event_handlers import OnProcessStart, OnProcessExit

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration, Command

from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_urdf_and_sdf(context, *args, **kwargs):
    """Generate URDF from XACRO and convert to SDF."""
    pkg_ardupilot_sitl_models = get_package_share_directory("js5_sitl")

    # Get launch configuration values
    namespace = LaunchConfiguration("namespace").perform(context)
    thruster_config = LaunchConfiguration("thruster_config").perform(context)

    # Use /tmp directory for generated files (writable)
    temp_dir = "/tmp/js5_launch"
    os.makedirs(temp_dir, exist_ok=True)

    # Define file paths
    xacro_file = os.path.join(pkg_ardupilot_sitl_models, "models", "js5", "urdf", "js.xacro")
    urdf_file = os.path.join(temp_dir, "model.urdf")
    sdf_file = os.path.join(temp_dir, "model.sdf")

    print(f"Looking for XACRO file at: {xacro_file}")
    print(f"Will generate URDF at: {urdf_file}")
    print(f"Will generate SDF at: {sdf_file}")

    # Step 1: Process XACRO to URDF
    xacro_to_urdf = ExecuteProcess(
        cmd=[
            f'xacro {xacro_file} namespace:={namespace} thruster_config:={thruster_config} > {urdf_file}'
        ],
        shell=True,
        output='screen',
        name='xacro_to_urdf'
    )

    # Step 2: Convert URDF to SDF (runs after XACRO processing completes)
    urdf_to_sdf = ExecuteProcess(
        cmd=[
            f'~/ros2_gazebo/scripts/urdf2sdf.sh {urdf_file} {sdf_file}'
        ],
        shell=True,
        output='screen',
        name='urdf_to_sdf',
    )

    # Step 3: Create robot state publisher (runs after SDF is created)
    def create_robot_state_publisher():
        # Read the generaed SDF file
        if os.path.exists(sdf_file):
            with open(sdf_file, 'r') as infp:
                robot_desc = infp.read()
                # Substitute `models://` with `package://js5_sitl/models/js5/` for sdformat_urdf plugin
                robot_desc = robot_desc.replace(
                    "model://js5", "package://js5_sitl/models/js5"
                )
        else:
            # Fallback: read URDF if SDF doesn't exist
            with open(urdf_file, 'r') as infp:
                robot_desc = infp.read()
                robot_desc = robot_desc.replace(
                    "model://js5", "package://js5_sitl/models/js5"
                )
        return Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="both",
            parameters=[
                {"robot_description": robot_desc},
                {"frame_prefix": ""},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        )
    
    robot_state_publisher = RegisterEventHandler(
        OnProcessExit(
            target_action=urdf_to_sdf,
            on_exit=[OpaqueFunction(function=lambda context: [create_robot_state_publisher()])]
        )
    )

    return [
        xacro_to_urdf,
        RegisterEventHandler(
            OnProcessExit(
                target_action=xacro_to_urdf,
                on_exit=[urdf_to_sdf]
            )
        ),
        robot_state_publisher
    ]

def generate_launch_description():
    """Generate a launch description for a jong_shyn No.5 rover."""
    pkg_ardupilot_sitl = get_package_share_directory("ardupilot_sitl")
    pkg_ardupilot_sitl_models = get_package_share_directory("js5_sitl")
    pkg_project_bringup = get_package_share_directory("js5_sitl")

    # Launch arguments
    namespace_arg = DeclareLaunchArgument(
        "namespace", 
        default_value="js", 
        description="Robot namespace"
    )
    
    thruster_config_arg = DeclareLaunchArgument(
        "thruster_config", 
        default_value="H", 
        description="Thruster configuration (H or M)"
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", 
        default_value="true", 
        description="Use simulation time"
    )

    use_gz_tf_arg = DeclareLaunchArgument(
        "use_gz_tf", 
        default_value="true", 
        description="Use Gazebo TF"
    )

    use_sitl_arg = DeclareLaunchArgument(
        "use_sitl",
        default_value="true",
        description="Launch ArduPilot SITL"
    )

    # Include component launch files.
    sitl_dds = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("ardupilot_sitl"),
                        "launch",
                        "sitl_dds_udp.launch.py",
                    ]
                ),
            ]
        ),
        launch_arguments={
            "transport": "udp4",
            "port": "2019",
            "command": "ardurover",
            "synthetic_clock": "True",
            "wipe": "False",
            "model": "json",
            "speedup": "1",
            "slave": "0",
            "instance": "0",
            "defaults": os.path.join(
                pkg_ardupilot_sitl,
                "config",
                "default_params",
                "rover-skid.parm",
            )
            + ","
            + os.path.join(
                pkg_ardupilot_sitl,
                "config",
                "default_params",
                "dds_udp.parm",
            ),
            "sim_address": "127.0.0.1",
            "master": "tcp:127.0.0.1:5760",
            "sitl": "127.0.0.1:5501",
        }.items(),
        condition=IfCondition(LaunchConfiguration("use_sitl")),
    )

    # Ensure `SDF_PATH` is populated as `sdformat_urdf`` uses this rather
    # than `GZ_SIM_RESOURCE_PATH` to locate resources.
    if "GZ_SIM_RESOURCE_PATH" in os.environ:
        gz_sim_resource_path = os.environ["GZ_SIM_RESOURCE_PATH"]
        os.environ['GZ_SIM_RESOURCE_PATH'] += ':' + pkg_ardupilot_sitl_models
        if "SDF_PATH" in os.environ:
            sdf_path = os.environ["SDF_PATH"]
            os.environ["SDF_PATH"] = sdf_path + ":" + gz_sim_resource_path
        else:
            os.environ["SDF_PATH"] = gz_sim_resource_path
    else:
        os.environ['GZ_SIM_RESOURCE_PATH'] = pkg_ardupilot_sitl_models

    # Bridge
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[
            {
                "config_file": os.path.join(
                    pkg_project_bringup, "gz_bridges", "js5_bridge.yaml"
                ),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "qos_overrides./tf_static.publisher.durability": "transient_local",
            }
        ],
        output="screen",
    )

    # Relay - use instead of transform when Gazebo is only publishing odom -> base_link
    topic_tools_tf = Node(
        package="topic_tools",
        executable="relay",
        arguments=[
            "/gz/tf",
            "/tf",
        ],
        output="screen",
        respawn=False,
        condition=IfCondition(LaunchConfiguration("use_gz_tf")),
    )

    return LaunchDescription(
        [
            namespace_arg,
            thruster_config_arg,
            use_sim_time_arg,
            use_gz_tf_arg,
            spawn_x_arg,
            spawn_y_arg,
            spawn_z_arg,
            spawn_yaw_arg,
            use_sitl_arg,
            sitl_dds,
            OpaqueFunction(function=generate_urdf_and_sdf),
            bridge,
            RegisterEventHandler(
                OnProcessStart(
                    target_action=bridge,
                    on_start=[
                        topic_tools_tf,
                    ]
                )
            )
        ]
    )