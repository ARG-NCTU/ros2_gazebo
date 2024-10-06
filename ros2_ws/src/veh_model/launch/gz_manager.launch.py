from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
import os


def launch_setup(context, *args, **kwargs):
    # Retrieve the values of the arguments at runtime
    veh = LaunchConfiguration('veh').perform(context)
    task = LaunchConfiguration('task').perform(context)
    world = LaunchConfiguration('world').perform(context)

    pkg_project_gazebo = get_package_share_directory("veh_model")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_project_bringup = get_package_share_directory("veh_model")
    pkg_veh_models = get_package_share_directory("veh_model")

    # Load SDF file.
    sdf_file = os.path.join(
        pkg_veh_models, "models", veh, "model.sdf"
    )
    with open(sdf_file, "r") as infp:
        robot_desc = infp.read()
        robot_desc = robot_desc.replace(
            f"models://{veh}",
            f"package://veh_model/models/{veh}"
        )

    # Publish /tf and /tf_static.
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {"robot_description": robot_desc},
            {"frame_prefix": ""},
        ],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        namespace=veh,
    )

    # Define the Gazebo simulation server
    gz_sim_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_ros_gz_sim, "/launch/gz_sim.launch.py"]),
        launch_arguments={
            "gz_args": f"-v4 -s -r {pkg_project_gazebo}/worlds/{world}.sdf"
        }.items(),
    )

    gz_sim_guest = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_ros_gz_sim, "/launch/gz_sim.launch.py"]),
        launch_arguments={"gz_args": "-v4 -g"}.items(),
        condition=IfCondition(LaunchConfiguration('gz_gui')),
    )

    # Define the topic_bridge node to convert Gazebo topics to ROS2 topics
    topic_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[
            {
                "config_file": f"{pkg_project_bringup}/gz_bridges/{veh}/{task}.yaml",
                "qos_overrides./tf_static.publisher.durability": "transient_local",
            }
        ],
        output="screen",
        namespace=veh,
    )

    server_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'/world/{world}/control@ros_gz_interfaces/srv/ControlWorld@ignition.msgs.WorldControl@ignition.msgs.Boolean'
        ],
        output='screen',
        namespace=veh,
    )

    # Relay /gz/tf -> /tf for tf data from Gazebo to ROS
    topic_tools_tf = Node(
        package="topic_tools",
        executable="relay",
        arguments=[
            f"/{veh}/tf",
            "tf",
        ],
        output="screen",
        respawn=False,
        condition=IfCondition(LaunchConfiguration("use_gz_tf")),
        namespace=veh,
    )

    # Define the RViz node
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", f"{pkg_project_gazebo}/rviz/{veh}.rviz"],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output="screen",
        remappings=(('/tf', 'tf'), ('/tf_static', 'tf_static')),
        namespace=veh,
    )

    # Return all the launch actions
    return [
        robot_state_publisher,
        topic_bridge,
        server_bridge,
        RegisterEventHandler(
            OnProcessStart(
                target_action=topic_bridge,
                on_start=[
                    topic_tools_tf
                ]
            )
        ), 
        gz_sim_server, 
        gz_sim_guest, 
        rviz
    ]


def generate_launch_description():
    # Declare the launch arguments
    return LaunchDescription([
        DeclareLaunchArgument(
            'veh', default_value='blueboat', description='Vehicle model to load'
        ),
        DeclareLaunchArgument(
            'world', default_value='waves', description='World to launch'
        ),
        DeclareLaunchArgument(
            'task', default_value='train', description='Purpose of Gazebo simulation'
        ),
        DeclareLaunchArgument(
            'rviz', default_value='true', description='Launch rviz'
        ),
        DeclareLaunchArgument(
            'gz_gui', default_value='true', description='Launch GZ GUI'
        ),
        DeclareLaunchArgument(
            "use_gz_tf", default_value="true", description="Use Gazebo TF."
        ),
        # Opaque function allows dynamic launch setup
        OpaqueFunction(function=launch_setup)
    ])
