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
    world = LaunchConfiguration('world').perform(context)

    pkg_project_gazebo = get_package_share_directory("veh_model")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

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

    server = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'/world/{world}/control@ros_gz_interfaces/srv/ControlWorld@ignition.msgs.WorldControl@ignition.msgs.Boolean', 
            f'/world/{world}/create@ros_gz_interfaces/srv/SpawnEntity@ignition.msgs.EntityFactory@ignition.msgs.Boolean',
            f'/world/{world}/remove@ros_gz_interfaces/srv/DeleteEntity@ignition.msgs.Entity@ignition.msgs.Boolean'
        ],
        output='screen',
    )

    move_entity = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'/world/{world}/move@ros_gz_interfaces/srv/SetEntityPose@ignition.msgs.Entity@ignition.msgs.Boolean'
        ],
        output='screen',
    )



    # Return all the launch actions
    return [
        server,
        # move_entity,
        gz_sim_server, 
        gz_sim_guest, 
    ]


def generate_launch_description():
    # Declare the launch arguments
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='waves', description='World to launch'
        ),
        DeclareLaunchArgument(
            'gz_gui', default_value='true', description='Launch GZ GUI'
        ),
        OpaqueFunction(function=launch_setup)
    ])
