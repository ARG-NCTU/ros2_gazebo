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
    spawn_entity = LaunchConfiguration('spawn_entity').perform(context)
    rviz = LaunchConfiguration('rviz').perform(context)
    gz_gui = LaunchConfiguration('gz_gui').perform(context)

    gz_manager_pkg = get_package_share_directory("veh_model")
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gz_manager_pkg, "/launch/gz_manager.launch.py"]),
        launch_arguments={
            "veh": veh,
            "world": world,
            "task": task,
            "spawn_entity": spawn_entity,
            "rviz": rviz,
            "gz_gui": gz_gui,
        }.items(),
    )
    return [gz_sim]

def generate_launch_description():
    # Declare the launch arguments
    return LaunchDescription([
        DeclareLaunchArgument(
            'veh', default_value='wamv_alpha', description='Vehicle model to load'
        ),
        DeclareLaunchArgument(
            'world', default_value='waves', description='World to launch'
        ),
        DeclareLaunchArgument(
            'task', default_value='test_propeller', description='Purpose of Gazebo simulation'
        ),
        DeclareLaunchArgument(
            'spawn_entity', default_value='true', description='spawn the veh model along with gz'
        ),
        DeclareLaunchArgument(
            'rviz', default_value='false', description='Launch rviz'
        ),
        DeclareLaunchArgument(
            'gz_gui', default_value='true', description='Launch GZ GUI'
        ),
        # Opaque function allows dynamic launch setup
        OpaqueFunction(function=launch_setup)
    ])