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
    # Retrieve the values of the arguments at runtime
    world = LaunchConfiguration('world').perform(context)
    visual = LaunchConfiguration('visual').perform(context)
    file = "./"
    if visual == 'no_visual':
        file = f"no_visual_{world}/model.sdf"
    else:
        file = f"./{world}.sdf"
    pkg_project_gazebo = get_package_share_directory("veh_model")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    # Define the Gazebo simulation server
    gz_sim_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_ros_gz_sim, "/launch/gz_sim.launch.py"]),
        launch_arguments={
            "gz_args": f"-v4 -s -r {pkg_project_gazebo}/worlds/{file}"
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
            f'/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen',
    )

    # Return all the launch actions
    return [
        server,
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
            'visual', default_value='no_visual', description='World without visual'
        ),
        DeclareLaunchArgument(
            'gz_gui', default_value='true', description='Launch GZ GUI'
        ),
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    # Create the LaunchService and add the generated launch description
    launch_service = launch.LaunchService()
    ld = generate_launch_description()  # Correctly call the function to get the LaunchDescription
    launch_service.include_launch_description(ld)
    
    # Run the launch service
    launch_service.run()
