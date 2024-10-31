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
'''
!!!
USE CPU Docker container to run this launch file
!!!
'''


def launch_setup(context, *args, **kwargs):
    veh = LaunchConfiguration('veh').perform(context)
    world = LaunchConfiguration('world').perform(context)
    max_thrust = float(LaunchConfiguration('max_thrust').perform(context))
    
    pkg_joy = get_package_share_directory('veh_model')


    joy2cmdvel = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_joy, "/launch/blueboat/sim_joy.launch.py"]),
        launch_arguments={"veh": veh}.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f"/model/{veh}/pose@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V",
            f"/world/{world}/model/{veh}/link/imu_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            f"/model/{veh}/joint/motor_port_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
            f"/model/{veh}/joint/motor_stbd_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
        ],
        output='screen',
        namespace=veh
    )

    twist2thrust = Node(
        package='veh_model',
        executable=f'{veh}_twist2thrust',
        parameters=[{'name': veh, 'max_thrust': max_thrust}],
        output='screen',
        namespace=veh,
    )

    sb3_dp = Node(
        package='veh_model',
        executable=f'{veh}_sb3_dp',
        output='screen',
        parameters=[{'veh': veh}],
        namespace=veh,
        condition=IfCondition(LaunchConfiguration('sb3')),
    )

    acme_dp = Node(
        package='veh_model',
        executable=f'{veh}_acme_dp',
        output='screen',
        parameters=[{'veh': veh}],
        namespace=veh,
        condition=IfCondition(launch.substitutions.NotSubstitution(LaunchConfiguration('sb3'))),
    )

    return [joy2cmdvel, bridge, twist2thrust, sb3_dp, acme_dp]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'veh', default_value='blueboat', description='Vehicle to launch'
        ),
        DeclareLaunchArgument(
            'world', default_value='waves', description='World to launch'
        ),
        DeclareLaunchArgument(
            'max_thrust', default_value='50', description='Maximum thrust'
        ),
        DeclareLaunchArgument(
            'sb3', default_value='false', description='Use stable baselines 3'
        ),
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    launch_service = launch.LaunchService()
    ld = generate_launch_description()
    launch_service.include_launch_description(ld)
    launch_service.run()