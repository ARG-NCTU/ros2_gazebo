from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
import os
import launch


def launch_setup(context, *args, **kwargs):
    veh = LaunchConfiguration('veh').perform(context)

    joy = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{'dev': '/dev/input/js0'}],  # Adjust the device path if necessary
        remappings=[
            ('joy', f'/model/{veh}/joy'),
            ('joy/set_feedback', f'/model/{veh}/joy/set_feedback'),
            ],
        namespace=veh  # Use the 'veh' argument as the namespace
    )
    joy2cmdvel = Node(
        package='veh_model',
        executable='joy2cmdvel',
        name='joy_to_cmd_vel',
        output='screen',
        remappings=[
            (f'joy', f'/model/{veh}/joy'),
            (f'cmd_vel', f'/model/{veh}/thrust_calculator/cmd_vel'),
            (f'auto', f'/model/{veh}/auto'),
            ],
        namespace=veh  # Use the 'veh' argument as the namespace
    )

    return [joy, joy2cmdvel]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'veh', default_value='veh', description='Vehicle to launch'
        ),
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    launch_service = launch.LaunchService()
    ld = generate_launch_description()
    launch_service.include_launch_description(ld)
    launch_service.run()