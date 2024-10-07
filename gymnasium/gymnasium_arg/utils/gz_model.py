import os, shutil, signal, asyncio
import rclpy
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
import launch_ros.actions
from launch import LaunchService
import launch

from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Pose, Twist, Point, Quaternion
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import Float32, Float64
import numpy as np
import xml.etree.ElementTree as ET



class GZ_MODEL(Node):
    def __init__(self, world, orig_name, name, path, init_pose=Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))):
        super().__init__(name)
        self.info = {
            'name': name,
            'world': world,
            'model_path': path,
            'init_pose': init_pose,
            'yaml_path': None,
        }
        self.name = name
        self.world = world
        self.model_path = path
        self.init_pose = init_pose
        self.orig_name = orig_name
        self.pub = {}
        self.sub = {}
        self.obs = {}
        self.model_path = os.path.expanduser(self.model_path)
        self.__modify_and_copy(f'/tmp/{self.name}', self.model_path)
        self.info['model_path'] = f'/tmp/{self.name}/model.sdf'

    def move_pose(self, pose: Pose):
        pass

    def setup(self):
        self.get_logger().info(f'GZ model: {self.name} setting up')
        quaternion = [self.init_pose.orientation.x, self.init_pose.orientation.y, self.init_pose.orientation.z, self.init_pose.orientation.w]
        rotation = R.from_quat(quaternion)
        r = rotation.as_euler('xyz')
        spawn_entity_node = launch_ros.actions.Node(
            package='gz_entity_manager',
            executable='spawn_entity',
            output='screen',
            arguments=[
                '--ros-args',
                '-p', f'world:={self.world}',
                '-p', f"file_path:={self.info['model_path']}",
                '-p', f'entity_name:={self.name}',
                '-p', f'x:={self.init_pose.position.x}',
                '-p', f'y:={self.init_pose.position.y}',
                '-p', f'z:={self.init_pose.position.z}',
                '-p', f'roll:={r[0]}',
                '-p', f'pitch:={r[1]}',
                '-p', f'yaw:={r[2]}'
            ],
        )
        ld = launch.LaunchDescription([spawn_entity_node])
        launch_service = launch.LaunchService()
        launch_service.include_launch_description(ld)
        launch_service.run()
        self.get_logger().info(f'GZ model: {self.name} loaded')

    def delete_entity(self):
        delete_entity_node = launch_ros.actions.Node(
            package='gz_entity_manager',
            executable='delete_entity',
            output='screen',
            arguments=[
                '--ros-args',
                '-p', f'world:={self.world}',
                '-p', f'entity_name:={self.name}',
            ],
        )
        ld = launch.LaunchDescription([delete_entity_node])
        launch_service = launch.LaunchService()
        launch_service.include_launch_description(ld)
        launch_service.run()
        self.get_logger().info(f'GZ model: {self.name} deleted')

    def reset(self):
        self.delete_entity()
        self.setup()
        self.get_logger().info(f'GZ model: {self.name} reset')  # Corrected: reseted -> reset

    def close(self):
        self.delete_entity()
        self.get_logger().info(f'GZ model: {self.name} closed')  # Corrected: self.logger -> self.get_logger()
        self.destroy_node()
    ############################# private funcs #############################
    def __modify_and_copy(self, dir_path, model_dir_path):
        if os.path.exists(dir_path):
            print(f"Destination folder '{dir_path}' exists. Removing it...")
            shutil.rmtree(dir_path)

        print(f"Copying '{model_dir_path}' to '{dir_path}'...")
        shutil.copytree(model_dir_path, dir_path)

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file == 'model.sdf':
                    file_path = os.path.join(root, file)
                    self.__modify_model_sdf(dir_path, file_path)
        

    def __modify_model_sdf(self, dir_path, file_path):
        print(f"Modifying '{file_path}'...")
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Replace the text
        content = content.replace(
            f'models://{self.orig_name}',
            f'{dir_path}'
        ).replace(
            f'model/{self.orig_name}',
            f'model/{self.name}'
        ).replace(
            f'<namespace>{self.orig_name}',
            f'<namespace>{self.name}'
        ).replace(
            f'<enable>{self.orig_name}',
            f'<enable>{self.name}'
        )
        # Write back the modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


class BlueBoat_GZ_MODEL(GZ_MODEL):

        
    def __init__(self, world, name, path, pose: Pose, info={'veh':'blueboat', 'maxstep': 4096, 'max_thrust': 10.0, 'hist_frame': 5}):
        super().__init__(orig_name=info['veh'], name=name, path=path, world=world, init_pose=pose)
        self.max_thrust = 10.0
        self.gz_sub = {}  # Added: Initialize gz_sub dictionary
        self.gz_sub['pose'] = None  # Corrected: Initialize gz_sub['pose']
        self.gz_sub['imu'] = None  # Corrected: Initialize gz_sub['imu']
        self.gz_sub['termination'] = None  # Corrected: Initialize gz_sub['termination']

        self.gz_sub['pose'] = self.create_subscription(Pose, f"/model/{name}/pose", self.__pose_cb, 10)  # Corrected: Subscriber to create_subscription
        self.gz_sub['imu'] = self.create_subscription(Float64, f"/model/{name}/link/imu_link/sensor/imu_sensor/imu", self.__imu_cb, 10)
        self.gz_sub['termination'] = self.create_subscription(Float64, f"/world/empty/model/{name}/link/base_link/sensor/sensor_contact/contact", self.__termination_cb, 10)
        
        self.pub['cmd_vel'] = self.create_publisher(TwistStamped, f'/model/{name}/thrust_calculator/cmd_vel', 10)


        self.launch_service = LaunchService()

        # Create the launch script nodes
        launch_script = [
            launch_ros.actions.Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                output='screen',
                parameters=[],
                arguments=[
                    f"/model/{name}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
                    f"/world/{world}/model/{name}/link/imu_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
                    f"/model/{name}/joint/motor_port_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
                    f"/model/{name}/joint/motor_stbd_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
                ],
            ),
            launch_ros.actions.Node(
                package='veh_model',
                executable='bb_twist2thrust',
                output='screen',
                parameters=[],
                arguments=[
                    name
                ],
            ),
        ]

        # Set up launch description and include it in the service
        # Set up launch description and include it in the service
        ld = launch.LaunchDescription(launch_script)
        self.launch_service = LaunchService()
        self.launch_service.include_launch_description(ld)

        # Run the launch service in the main thread
        self.launch_future = asyncio.ensure_future(self.launch_service.run_async())

        self.obs['pose'] = Pose()
        self.obs['twist'] = Twist()
        self.obs['termination'] = False
        self.obs['truncation'] = False

        self.info['maxstep'] = info['maxstep']
        self.info['max_thrust'] = info['max_thrust']
        self.info['hist_frame'] = info['hist_frame']
        self.info['step_cnt'] = 0

        self.hist_obs = []
        self.setup()
        
    def get_observation(self):
        return self.obs
    
    def reset(self):
        super().reset()
        self.obs['termination'] = False
        self.obs['truncation'] = False
        self.info['step_cnt'] = 0

    def step(self, action: Twist):
        self.info['step_cnt'] += 1
        self.pub['cmd_vel'].publish(action)
        if self.info['step_cnt'] >= self.info['maxstep']:  # Corrected: self.step_cnt -> self.info['step_cnt']
            self.obs['truncation'] = True
    
    def close(self):
        self.get_logger().info("Closing the service...")
        super().close()
        self.launch_service.shutdown()
        self.launch_future.cancel()
        self.get_logger().info("Service closed successfully.")
    
    ############################# private funcs #############################
    def __pose_cb(self, msg):
        self.obs['pose'] = msg.pose

    def __imu_cb(self, msg):
        self.obs['twist'] = Twist()
        linear = msg.linear_acceleration
        angular = msg.angular_velocity
        self.obs['twist'].linear = Point(x=linear.x, y=linear.y, z=linear.z)
        self.obs['twist'].angular = Point(x=angular.x, y=angular.y, z=angular.z)

    def __termination_cb(self, msg):
        self.obs['termination'] = True if msg is not None else False