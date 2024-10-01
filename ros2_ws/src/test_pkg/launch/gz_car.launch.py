import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    # Define the path to the world file
    # world_path = os.path.join(arg_gz_package_path, 'worlds', 'waves.sdf')
    # def get_gz_resource_paths(model_name: str = 'blueboat'):
    #     # Get the GZ_RESOURCE_PATH from the environment
    #     gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')

    #     # Split the paths by colon (':')
    #     paths = gz_resource_path.split(':')
    #     for path in paths:
    #         # Expand any environment variables in the path
    #         full_path = os.path.join(path, model_name)
    #         # Check if it exists and is a file (executable) or directory
    #         if os.path.exists(full_path):
    #             return os.path.join(full_path, 'model.sdf')  # Return the path if it exists
    #     return None
    # model_path = get_gz_resource_paths('blueboat')
    # if model_path is None:
        # raise FileNotFoundError('Model path not found. Please set the GZ_SIM_RESOURCE_PATH environment variable.')

    return LaunchDescription([
        # Launch Gazebo Sim (Ignition) with the waves world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        ),

        # Bridge for port motor: ROS2 Float64 <-> Gazebo Sim Double
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'],
            output='screen'
        ),

    ])