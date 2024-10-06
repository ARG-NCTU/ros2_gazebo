import os, subprocess
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Pose, Twist, Point, Quaternion
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import Float32, Float64
import numpy as np
import yaml
import xml.etree.ElementTree as ET
import argparse

# def parse_sdf_to_yaml(sdf_file, world, entity_name, pose: Pose):
#     try:
#         x = pose.position.x
#         y = pose.position.y
#         z = pose.position.z
#         quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
#         rotation = R.from_quat(quaternion)
#         r = rotation.as_euler('xyz')
#         roll = r[0]
#         pitch = r[1]
#         yaw = r[2]
#         sdf_string = "\n".join([line.strip() for line in sdf_file.splitlines() if line.strip()])
#         # Create dictionary for YAML structure
#         yaml_dict = {
#             "spawn_entity_node": {
#                 "ros_parameters": {
#                     "world": world,
#                     "entity_name": entity_name,
#                     "sdf_string": sdf_string,
#                     "x": x,
#                     "y": y,
#                     "z": z,
#                     "roll": roll,
#                     "pitch": pitch,
#                     "yaw": yaw
#                 }
#             }
#         }
#         # Generate YAML string
#         yaml_string = yaml.dump(yaml_dict, default_flow_style=False)
#         return yaml_string
#     except Exception as e:
#         return f"Error: {str(e)}"


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
        self.pub = {}
        self.sub = {}
        self.obs = {}
        self.bridge = []
        self.robot_desc = None
        self.model_path = os.path.expanduser(self.model_path)
        with open(self.model_path, "r") as infp:
            self.robot_desc = infp.read()
            self.robot_desc = self.robot_desc.replace(
                f"models://{orig_name}",
                f"models://{self.name}"
            )
            self.robot_desc = self.robot_desc.replace(" ", "").replace("\n", "")
        
    def move_pose(self, pose: Pose):
        pass

    def setup(self):
        quaternion = [self.init_pose.orientation.x, self.init_pose.orientation.y, self.init_pose.orientation.z, self.init_pose.orientation.w]
        rotation = R.from_quat(quaternion)
        r = rotation.as_euler('xyz')
        cli = subprocess.Popen([
            "ros2", "run", "gz_entity_manager", "spawn_entity", "--ros-args",
            f"-p world:={self.world}", f"-p entity_name:={self.name}", 
            f"-p sdf_string:=<sdf version='1.7'><model name='my_robot'><pose>0 0 0 0 0 0</pose><link name='chassis'><visual name='chassis_visual'><geometry><box><size>1 1 1</size></box></geometry><material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse></material></visual><collision name='chassis_collision'><geometry><box><size>1 1 1</size></box></geometry></collision></link></model></sdf>",
            f"-p x:={self.init_pose.position.x}", f"-p y:={self.init_pose.position.y}", f"-p z:={self.init_pose.position.z}",
            f"-p roll:={r[0]}", f"-p pitch:={r[1]}", f"-p yaw:={r[2]}",
        ])
        self.get_logger().info(f'GZ model: {self.name} setting up')
        cli.wait()
        cli.kill()
        self.get_logger().info(f'GZ model: {self.name} loaded')

    def delete_entity(self):
        cli = subprocess.Popen([
            "ros2", "run", "gz_entity_manager", "delete_entity",
            f"-p world:={self.world}", f"-p entity_name:={self.name}"
        ])
        self.get_logger().info(f'GZ model: {self.name} deleting')
        cli.wait()
        cli.kill()
        self.get_logger().info(f'GZ model: {self.name} deleted')

    def reset(self):
        self.delete_entity()
        self.setup()
        self.get_logger().info(f'GZ model: {self.name} reset')  # Corrected: reseted -> reset

    def close(self):
        self.delete_entity()
        for b in self.bridge:
            b.kill()
        self.get_logger().info(f'GZ model: {self.name} closed')  # Corrected: self.logger -> self.get_logger()
        self.destroy_node()
    ############################# private funcs #############################
    
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
                            
        self.bridge.append(
            subprocess.Popen([
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/model/{name}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
                f"/world/{world}/model/{name}/link/imu_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
                f"/model/{name}/joint/motor_port_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
                f"/model/{name}/joint/motor_stbd_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double",
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