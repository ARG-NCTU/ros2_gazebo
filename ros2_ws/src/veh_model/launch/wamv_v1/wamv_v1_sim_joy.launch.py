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
    world = LaunchConfiguration('world').perform(context)
    max_thrust = float(LaunchConfiguration('max_thrust').perform(context))
    
    pkg_joy = get_package_share_directory('veh_model')


    joy2cmdvel = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_joy, "/launch/sim_joy.launch.py"]),
        launch_arguments={"veh": veh}.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f"/world/{world}/model/{veh}/link/imu_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            f"/{veh}/joint/left/thruster/cmd_thrust@std_msgs/msg/Float64]ignition.msgs.Double",
            f"/{veh}/joint/left_front/thruster/cmd_thrust@std_msgs/msg/Float64]ignition.msgs.Double",
            f"/{veh}/joint/right/thruster/cmd_thrust@std_msgs/msg/Float64]ignition.msgs.Double",
            f"/{veh}/joint/right_front/thruster/cmd_thrust@std_msgs/msg/Float64]ignition.msgs.Double",
            f"/model/{veh}/pose@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V",
        ],
        output='screen',
        namespace=veh
    )

    twist2thrust = Node(
        package='veh_model',
        executable=f'{veh}_twist2thrust',
        output='screen',
        parameters=[{'name': veh, 'max_thrust': max_thrust}],
        namespace=veh
    )

    sb3_dp = Node(
        package='veh_model',
        executable=f'{veh}_sb3_dp',
        output='screen',
        parameters=[{'veh': veh}],
        namespace=veh
    )

    return [joy2cmdvel, bridge, twist2thrust, sb3_dp]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'veh', default_value='wamv_v1', description='Vehicle to launch'
        ),
        DeclareLaunchArgument(
            'world', default_value='waves', description='World to launch'
        ),
        DeclareLaunchArgument(
            'max_thrust', default_value='114.183', description='Maximum thrust'
        ),
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    launch_service = launch.LaunchService()
    ld = generate_launch_description()
    launch_service.include_launch_description(ld)
    launch_service.run()