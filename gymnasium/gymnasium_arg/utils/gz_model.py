import os, subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Pose, Twist, Point, Quaternion
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import Float32, Float64
import numpy as np


class GZ_MODEL(Node):
    def __init__(self, world, name, path, init_pose=Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))):
        super().__init__(name)
        self.info = {
            'name': name,
            'world': world,
            'model_path': path,
            'init_pose': init_pose,
        }
        self.name = name
        self.world = world
        self.model_path = path
        self.init_pose = init_pose
        self.pub = {}
        self.sub = {}
        self.obs = {}
        self.bridge = []
        self.robot_desc = None
        self.model_path = os.path.expanduser(self.model_path)
        with open(self.model_path, "r") as infp:
            self.robot_desc = infp.read()
            self.robot_desc = self.robot_desc.replace(
                f"models://{self.name}",
                f"package://veh_model/models/{self.name}"
            )
        
    def move_pose(self, pose: Pose):
        pass

    def setup(self):
        cli = self.create_client(SpawnEntity, f'/world/{self.world}/create')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for GZ world: {self.world} control service...')
        spawn_req = SpawnEntity.Request()
        spawn_req.xml = self.robot_desc
        spawn_req.name = self.name  # Corrected: self.model_name -> self.name
        spawn_req.initial_pose = self.init_pose
        future = cli.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'GZ model: {self.name} loaded')
        else:
            self.get_logger().error(f'GZ model: {self.name} failed to load')
        cli.destroy()  # Corrected: cli.destroy() -> cli.destroy()
        self.get_logger().info(f'GZ model: {self.name} setting up')

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
        cli.destroy()  # Corrected: cli.destroy() -> cli.destroy()
        self.get_logger().info(f'GZ model: {self.name} reset')  # Corrected: reseted -> reset

    def reset(self):
        self.delete_entity()
        self.setup()
        self.get_logger().info(f'GZ model: {self.name} reset')  # Corrected: reseted -> reset

    def close(self):
        self.delete_entity()
        for b in self.bridge:
            b.terminate()
            b.wait()
        self.get_logger().info(f'GZ model: {self.name} closed')  # Corrected: self.logger -> self.get_logger()
        self.destroy_node()
    ############################# private funcs #############################
    
class BlueBoat_GZ_MODEL(GZ_MODEL):

        
    def __init__(self, world, name, path, pose: Pose, info={'maxstep': 4096, 'max_thrust': 10.0, 'hist_frame': 5}):
        super().__init__(name=name, path=path, world=world, init_pose=pose)
        self.max_thrust = 10.0
        self.gz_sub = {}  # Added: Initialize gz_sub dictionary
        self.gz_sub['pose'] = None  # Corrected: Initialize gz_sub['pose']
        self.gz_sub['imu'] = None  # Corrected: Initialize gz_sub['imu']
        self.gz_sub['termination'] = None  # Corrected: Initialize gz_sub['termination']

        self.gz_sub['pose'] = self.create_subscription(Pose, f"/model/{name}/pose", self.__pose_cb, 10)  # Corrected: Subscriber to create_subscription
        self.gz_sub['imu'] = self.create_subscription(Float64, f"/model/{name}/link/imu_link/sensor/imu_sensor/imu", self.__imu_cb, 10)
        self.gz_sub['termination'] = self.create_subscription(Float64, f"/world/empty/model/{name}/link/base_link/sensor/sensor_contact/contact", self.__termination_cb, 10)
        
        self.pub['cmd_vel'] = self.create_publisher(TwistStamped, f'/model/{name}/thrust_calculator/cmd_vel', 10)
                            
        # self.sub['stbd_thrust'] = self.create_subscription(Float64, f"/model/{name}/joint/motor_stbd_joint/cmd_thrust", self.__stbd_thrust_cb, 10)  # Corrected: subscriber typo
        # self.sub['port_thrust'] = self.create_subscription(Float64, f"/model/{name}/joint/motor_port_joint/cmd_thrust", self.__port_thrust_cb, 10)  # Corrected: subscriber typo
        # self.pub['stbd_thrust'] = self.create_publisher(Float64, f"/model/{name}/joint/motor_stbd_joint/cmd_thrust", 10)
        # self.pub['port_thrust'] = self.create_publisher(Float64, f"/model/{name}/joint/motor_port_joint/cmd_thrust", 10)
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
                f"{name}" #topic: sub /{name}/{name}_thrust_calculator/cmd_vel@TwistStamped
            ])
        )
        self.bridge.append(
            subprocess.Popen([
                "ros2", "run", "veh_model", "bb_twist2thrust", name
            ])
        )

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
        self.obs['termination'] = False
        self.obs['truncation'] = False
        self.info['step_cnt'] = 0
        super().reset()

    def step(self, action: Twist):
        self.info['step_cnt'] += 1
        self.pub['cmd_vel'].publish(action)
        if self.info['step_cnt'] >= self.info['maxstep']:  # Corrected: self.step_cnt -> self.info['step_cnt']
            self.obs['truncation'] = True
    
    def close(self):
        super().close()
    
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