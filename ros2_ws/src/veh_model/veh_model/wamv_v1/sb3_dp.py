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
import os, sys
import torch

from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'laser':
                extractors[key] = nn.Sequential(
                        nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                total_concat_size += 32*241
            elif key == 'track' or key == 'velocity':
                extractors[key] = nn.Sequential()
                total_concat_size += subspace.shape[0]
        self.extractors = nn.ModuleDict(extractors)

        self.mlp_network = nn.Sequential(
            nn.Linear(total_concat_size, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self._features_dim = 256

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            tensor_append = extractor(observations[key])
            encoded_tensor_list.append(tensor_append)
        features = th.cat(encoded_tensor_list, dim=1)
        return self.mlp_network(features)
    
class SB3_DP(Node):
    def __init__(self):
        super().__init__('sb3_dp')
        self.declare_parameter('veh', 'wamv_v1')
        self.veh = self.get_parameter('veh').get_parameter_value().string_value

        self.suber = {   
            'goal': self.create_subscription(
            PoseStamped,
            f'/{self.veh}/goal_pose',
            self.__goal_callback,
            10),
            'pose': self.create_subscription(
            Pose,
            f'/model/{self.veh}/pose',
            self.__pose_callback,
            10),
            'auto': self.create_subscription(
            Bool,
            f'/{self.veh}/auto',
            self.__auto_callback,
            10),
        }

        self.obs = {
            'goal': None,
            'auto': False,
            'lidar' : np.zeros((4, 241)),
            'goal_diff': np.zeros((10, 3)),
            'velocity': np.zeros((10, 1)),
            'last_time': None,
            'last_pose': None,
        }
        
        self.cmd_vel_puber = self.create_publisher(
            TwistStamped,
            f'/{self.veh}/cmd_vel',
            1)
        print("Loading model...")
        device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256)
        )
        self.model = TD3.load(
            f"/home/arg/ros2_gazebo/ros2_ws/install/veh_model/share/veh_model/models/wamv_v1/forest_td3_2024-05-22_2900000_steps.zip",
            policy_kwargs=policy_kwargs,
            device=device
            )
        print("Model loaded.")

    def eval(self):
        while rclpy.ok():
            if self.obs['auto'] is True:
                obs = {
                    'laser': self.obs['lidar'],
                    'track': self.obs['goal_diff'].flatten(),
                    'velocity': self.obs['velocity'].flatten(),
                }
                action, _ = self.model.predict(obs)
                action = self.__remap_action(action)
                cmd_vel = TwistStamped()
                cmd_vel.header.stamp = self.get_clock().now().to_msg()
                cmd_vel.header.frame_id = 'base_link'
                cmd_vel.twist.linear.x = action[0]
                cmd_vel.twist.angular.z = action[1]
                self.cmd_vel_puber.publish(cmd_vel)
                rclpy.spin_once(self, timeout_sec=0.02)

    def __goal_callback(self, msg):
        self.obs['goal'] = msg.pose
        
    def __pose_callback(self, msg):
        q_goal = np.array([self.obs['goal'].orientation.x, self.obs['goal'].orientation.y, self.obs['goal'].orientation.z, self.obs['goal'].orientation.w])
        _, _, yaw_goal = R.from_quat(q_goal).as_euler('xyz')
        q = np.array([msg.orintation.x, msg.orintation.y, msg.orintation.z, msg.orintation.w])
        _, _, yaw = R.from_quat(q).as_euler('xyz')
        angle = yaw_goal - yaw
        if angle >= np.pi:
            angle -= 2*np.pi
        elif angle <= -np.pi:
            angle += 2*np.pi
        goal_x_prime, goal_y_prime = map_to_model_frame(np.array([self.obs['goal'].position.x, self.obs['goal'].position.y]), np.array([msg.position.x, msg.position.y]), angle)
        pos_diff = np.array([goal_x_prime, goal_y_prime, angle])
        self.obs['goal_diff'] = np.roll(self.obs['goal_diff'], 1, axis=0)
        self.obs['goal_diff'][0] = pos_diff
        self.obs['velocity'] = np.roll(self.obs['velocity'], 1, axis=0)
        if self.obs['last_time'] is None or self.obs['last_pose'] is None:
            self.obs['last_time'] = sys.get_clock().now()
            self.obs['last_pose'] = msg
        else:
            dt = (sys.get_clock().now() - self.obs['last_time']).nanoseconds / 1e9
            self.obs['last_time'] = sys.get_clock().now()
            distance = np.sqrt((msg.position.x - self.obs['last_pose'].position.x)**2 + (msg.position.y - self.obs['last_pose'].position.y)**2)
            vel = distance / dt
            self.obs['velocity'][0] = vel
    
    def __auto_callback(self, msg):
        if self.obs['goal'] is None and msg.data is True:
            self.obs['goal'] = self.obs['last_pose']
        else:
            self.obs['goal'] = None
        self.obs['auto'] = msg.data

    def __remap_action(self, action):
        # remap linear action[0] : [0~1] to [-1~1]
        # keep angular action[1] : [-1~1]
        action[0] = 2 * action[0] - 1
        return action    

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

def main(args=None):
    rclpy.init(args=args)
    node = SB3_DP()
    node.eval()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
