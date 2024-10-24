import os, shutil, signal, asyncio, time
import rclpy
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
import launch_ros.actions
from launch import LaunchService
import launch

from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import TwistStamped, Pose, Twist, Point, Quaternion, PoseStamped
from sensor_msgs.msg import Imu
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import Float32, Float64
import numpy as np
import xml.etree.ElementTree as ET
from gymnasium_arg.utils.gz_model import GZ_MODEL


class WAMVV1_GZ_MODEL(GZ_MODEL):

        
    def __init__(self, world, name, path, pose: Pose, info={'veh':'wamv_v1', 'maxstep': 4096, 'max_thrust': 15*746/9.8, 'hist_frame': 5}):
        super().__init__(orig_name=info['veh'], name=name, path=path, world=world, init_pose=pose)
        self.info['maxstep'] = info['maxstep']
        self.info['max_thrust'] = info['max_thrust']
        self.info['hist_frame'] = info['hist_frame']
        self.info['step_cnt'] = 0
        self.sub['imu'] = self.create_subscription(Imu, f"/world/{world}/model/{name}/link/imu_link/sensor/imu_sensor/imu", self.__imu_cb, 10)
        self.sub['termination'] = self.create_subscription(Float64, f"/world/{world}/model/{name}/link/base_link/sensor/sensor_contact/contact", self.__termination_cb, 1)
        self.pub['cmd_vel'] = self.create_publisher(TwistStamped, f'/model/{name}/thrust_calculator/cmd_vel', 1)

        self.launch_service = LaunchService()

        # Create the launch script nodes
        launch_script = [
            launch_ros.actions.Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                output='screen',
                parameters=[],
                arguments=[
                    f"/world/{world}/model/{name}/link/imu_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
                    f"/model/{self.name}/joint/left_engine_propeller_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
                    f"/model/{self.name}/joint/left_front_engine_propeller_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
                    f"/model/{self.name}/joint/right_engine_propeller_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
                    f"/model/{self.name}/joint/right_front_engine_propeller_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
                ],
                on_exit=launch.actions.Shutdown(),
            ),
            launch_ros.actions.Node(
                package='veh_model',
                executable='wamv_v1_twist2thrust',
                output='screen',
                parameters=[{'name': name, 'max_thrust': info["max_thrust"]},],
                on_exit=launch.actions.Shutdown(),
            ),
        ]

        # Set up launch description and include it in the service
        # Set up launch description and include it in the service
        ld = launch.LaunchDescription(launch_script)
        self.launch_service = LaunchService()
        self.launch_service.include_launch_description(ld)

        # Run the launch service in the main thread
        self.launch_future = asyncio.ensure_future(self.launch_service.run_async())

        # self.obs['action'] = Twist()
        self.obs['action'] = np.zeros((self.info['hist_frame'], 6))
        # self.obs['pose'] = Pose()
        self.obs['imu'] = np.array([])
        
        self.obs['termination'] = False
        self.obs['truncation'] = False

        self.hist_obs = np.array([])
        self.setup()
        
    def get_observation(self):
        while self.obs['imu'].shape != (self.info['hist_frame'], 10):
            # print(f"Waiting for {self.info['name']} imu...")
            pass
        return self.obs
    
    def reset(self):
        super().reset()
        self.obs['action'] = np.zeros((self.info['hist_frame'], 6))
        self.obs['imu'] = np.array([])
        self.obs['termination'] = False
        self.obs['truncation'] = False
        self.info['step_cnt'] = 0

    def step(self, action: TwistStamped):
        self.obs['action'] = np.roll(self.obs['action'], 1, axis=0)
        self.obs['action'][0] = np.array(
            [action.twist.linear.x, action.twist.linear.y, action.twist.linear.z, 
             action.twist.angular.x, action.twist.angular.y, action.twist.angular.z]
        )
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
    def __imu_cb(self, msg):
        imu = np.array([
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        if self.obs['imu'].shape != (self.info['hist_frame'], 10):
            if self.obs['imu'].shape == (0,):
                self.obs['imu'] = imu
            else:
                self.obs['imu'] = np.vstack((imu, self.obs['imu']))
        else:
            self.obs['imu'] = np.roll(self.obs['imu'], 1, axis=0)
            self.obs['imu'][0] = imu

    def __termination_cb(self, msg):
        self.obs['termination'] = True if msg is not None else False