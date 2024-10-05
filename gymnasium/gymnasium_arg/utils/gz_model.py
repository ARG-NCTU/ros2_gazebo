import time, math, random, sys, os, queue, subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose, Twist, Point, Quaternion
from std_msgs.msg import Float32
from ros_gz_interfaces.srv import ControlWorld, DeleteEntity, SpawnEntity
import numpy as np


class GZ_MODEL(Node):
    def __init__(self, world, name, path, init_pose=Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))):
        super().__init__(name)
        self.name = name
        self.world = world
        self.model_path = path
        self.init_pose = init_pose
        self.pub = {}
        self.sub = {}
        self.obs = {}
        self.bridge = []
        self.robot_desc = None
        with open(self.model_path, "r") as infp:
            self.robot_desc = infp.read()
            self.robot_desc = self.robot_desc.replace(
                f"models://{self.name}",
                f"package://veh_model/models/{self.name}"
            )
        

    def setup(self):
        self.logger.info(f'GZ model: {self.name} setting up')
        cli = self.create_client(SpawnEntity, f'/world/{self.world}/create')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for GZ world: {self.world} control service...')
        spawn_req = SpawnEntity.Request()
        spawn_req.xml = self.robot_desc
        spawn_req.name = self.model_name
        spawn_req.initial_pose = self.init_pose
        future = cli.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'GZ model: {self.name} loaded')
        else:
            self.get_logger().error(f'GZ model: {self.name} failed to load')
        cli.destroy()

    def delete_entity(self):
        cli = self.create_client(DeleteEntity, f'/world/{self.world}/remove')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for delete entity service...')
        req = DeleteEntity.Request()
        req.name = self.name
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'GZ model: {self.name} deleted')
        else:
            self.get_logger().error(f'GZ model: {self.name} failed to delete')
        cli.destroy()
        self.get_logger().info(f'GZ model: {self.name} reseted')

    def reset(self):
        self.delete_entity()
        self.setup()
        self.get_logger().info(f'GZ model: {self.name} reseted')

    def close(self):
        self.delete_entity()
        for b in self.bridge:
            b.terminate()
            b.wait()
        self.logger.info(f'GZ model: {self.name} closed')
        self.destroy_node()
    ############################# private funcs #############################
    
class BlueBoat_GZ_MODEL(GZ_MODEL):
    def __init__(self, world, name, path, pose=np.zeros(7)):
        super().__init__(name=name, path=path, world=world, init_pose=pose)
        self.pub['cmd_vel'] = self.create_publisher(TwistStamped, f'/{self.name}/cmd_vel', 10)
        self.sub['pose'] = self.create_subscription(PoseStamped, f'/model/{self.name}/pose', self.__pose_cb, 10)
        self.sub['imu'] = self.create_subscription(TwistStamped, f'/world/{world}/model/{name}/link/imu_link/sensor/imu_sensor/imu', self.__imu_cb, 10)
        self.obs['pose'] = Pose()
        self.obs['twist'] = Twist()
        self.bridge.append(
            subprocess.Popen([
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/model/{name}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose"
            ])
        )
        self.bridge.append(
            subprocess.Popen([
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/world/{world}/model/{name}/link/imu_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU"
            ])
        )
        self.bridge.append(
            subprocess.Popen([
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/model/{name}/joint/motor_port_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double"
            ])
        )
        self.bridge.append(
            subprocess.Popen([
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/model/{name}/joint/motor_stbd_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double"
            ])
        )
        self.bridge.append(
            subprocess.Popen([
                "ros2", "run", "veh_model", "bb_twist2thrust",
                f"{name}"
            ])
        )
        self.setup()
        
    def get_observation(self):
        return {
            'pose': self.obs['pose'],
            'pos_acc': np.array([self.obs['twist'].linear.x, self.obs['twist'].linear.y, self.obs['twist'].linear.z]),
            'ang_vel': np.array([self.obs['twist'].angular.x, self.obs['twist'].angular.y, self.obs['twist'].angular.z]),
        }
    
    ############################# private funcs #############################
    def __pose_cb(self, msg):
        self.obs['pose'] = msg.pose
    def __imu_cb(self, msg):
        self.obs['twist'].linear = msg.linear_acceleration
        self.obs['twist'].angular = msg.angular_velocity
        # normalize the acceleration
        sum = (self.obs['twist'].linear.x**2 + self.obs['twist'].linear.y**2 + self.obs['twist'].linear.z**2)**0.5
        self.obs['twist'].linear.x /= sum
        self.obs['twist'].linear.y /= sum
        self.obs['twist'].linear.z /= sum
        # normalize the angular velocity
        sum = (self.obs['twist'].angular.x**2 + self.obs['twist'].angular.y**2 + self.obs['twist'].angular.z**2)**0.5
        self.obs['twist'].angular.x /= sum
        self.obs['twist'].angular.y /= sum
        self.obs['twist'].angular.z /= sum