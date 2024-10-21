import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose, Twist
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import gymnasium as gym
from stable_baselines3 import PPO, TD3
import numpy as np
from scipy.spatial.transform import Rotation as R


class SB3_DP(Node):
    def __init__(self):
        super().__init__('wamv_dp_model')
        self.declare_parameter('veh', 'wamv')
        self.declare_parameter('max_thrust', 15*746/9.8)
        
        self.veh = self.get_parameter('veh').get_parameter_value().string_value
        self.max_thrust = self.get_parameter('max_thrust').get_parameter_value().double_value

        self.suber = {   
            'goal': self.create_subscription(
            PoseStamped,
            f'/model/{self.veh}/goal_pose',
            self.__goal_callback,
            10),
            'pose': self.create_subscription(
            PoseStamped,
            f'/model/{self.veh}/pose',
            self.__pose_callback,
            10),
            'activation': self.create_subscription(
            Bool,
            f'/model/{self.veh}/dp_activation',
            self.__activation_callback,
            10),
        }

        self.obs = {
            'goal': Pose(),
            'activate': True,
            'lidar' : np.zeros((4, 241)),
            'goal_diff': np.zeros((10, 3)),
            'velocity': np.zeros((10,)),
        }
        
        self.cmd_vel_puber = self.create_publisher(
            TwistStamped,
            f'/model/{self.veh}/thrust_calculator/cmd_vel',
            1)
        
        self.model = TD3.load("logs/forest_td3_2024-05-22_2900000_steps.zip")
        
    def eval(self):
        if self.obs['activate'] is False:
            return

    def __goal_callback(self, msg):
        self.obs['goal'] = msg.pose
        
    def __pose_callback(self, msg):
        q_goal = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        _, _, yaw_goal = R.from_quat(q_goal).as_euler('xyz')
        q = np.array([msg.pose.orintation.x, msg.pose.orintation.y, msg.pose.orintation.z, msg.pose.orintation.w])
        _, _, yaw = R.from_quat(q).as_euler('xyz')
        angle = yaw_goal - yaw
        if angle >= np.pi:
            angle -= 2*np.pi
        elif angle <= -np.pi:
            angle += 2*np.pi

        pos_diff = np.array([self.obs['goal'].position.x - msg.pose.position.x, self.obs['goal'].position.y - msg.pose.position.y, angle])
        self.obs['goal_diff'] = np.roll(self.obs['goal_diff'], 1, axis=0)
        self.obs['goal_diff'][0] = pos_diff
    
    def __activation_callback(self, msg):
        self.obs['activate'] = msg.data

    def map_to_model_frame(self, map_goal_pose, map_robot_pose, map_robot_orientation):
        # get robot pose in map frame
        x, y = map_goal_pose - map_robot_pose
        # get robot orientation in map frame
        robot_orientation = - map_robot_orientation  
        # get rotation matrix
        rotation_matrix = R.from_euler('z', robot_orientation).as_matrix()
        map_point = np.array([x, y, 1.0])
        # get goal point in model frame 
        model_point = np.dot(rotation_matrix, map_point)
        x_prime, y_prime = model_point[:2]
        return x_prime, y_prime
