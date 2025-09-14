from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import TextSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Generate a launch description fo a jong_shyn No.5 rover."""
    # pkg_project_bringup = get_package_share_directory("ardupilot_gz_bringup")
    pkg_project_gazebo = get_package_share_directory("js5_sitl")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

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
    
    world_arg = DeclareLaunchArgument(
        "world", 
        default_value="js5_waves", 
        description="World name (without .sdf extension)"
    )
    
    rviz_arg = DeclareLaunchArgument(
        "rviz", 
        default_value="false", 
        description="Open RViz"
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", 
        default_value="true", 
        description="Use simulation time"
    )

    use_sitl_arg = DeclareLaunchArgument(
        "use_sitl",
        default_value="false",
        description="Launch ArduPilot SITL"
    )

    # Jong Shyn No.5 rover - FIXED: Added launch arguments
    rover = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("js5_sitl"),
                "launch",
                "robots",
                "js5.launch.py",
            ])
        ]),
        launch_arguments=[
            ("namespace", LaunchConfiguration("namespace")),
            ("thruster_config", LaunchConfiguration("thruster_config")),
            ("use_sim_time", LaunchConfiguration("use_sim_time")),
            ("use_sitl", LaunchConfiguration("use_sitl")),
        ],
    )

    gz_sim_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(pkg_ros_gz_sim) / "launch" / "gz_sim.launch.py")
        ),
        launch_arguments=[
            ("gz_args", [
                TextSubstitution(text="-v4 -s --headless -r "),
                PathJoinSubstitution([
                    FindPackageShare("js5_sitl"),
                    "worlds",
                    LaunchConfiguration("world")
                ]),
                TextSubstitution(text=".sdf")
            ])
        ],
    )

    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f'{Path(pkg_ros_gz_sim) / "launch" / "gz_sim.launch.py"}'
        ),
        launch_arguments={"gz_args": "-v4 -g"}.items(),
    )

    # RViz
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare("js5_sitl"),
        "rviz",
        "js5.rviz"
    ])

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            namespace_arg,
            thruster_config_arg,
            world_arg,
            rviz_arg,
            use_sim_time_arg,
            use_sitl_arg,
            gz_sim_server,
            # gz_sim_gui,
            rover,
            rviz,
        ]
    )