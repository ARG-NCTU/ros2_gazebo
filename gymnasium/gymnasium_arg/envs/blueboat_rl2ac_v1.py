import time, math, random, sys, os, queue
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from typing import Optional
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float32
from ros_gz_interfaces.srv import ControlWorld

class GZ_model(Node):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.srv = {}
        self.cli = {}
        self.pub = {}
        self.sub = {}

class BlueBoat_RL2AC_V1(gym.Env, Node):
    
    def __init__(self, 
                 render_mode: Optional[str] = None, 
                 veh='blueboat', 
                 world='blueboat_waves', 
                 num_envs=1,
                 seed=0
                 ):
        self.info = {
            'node_name': f'blueboat_rl2ac_v1',
            'veh': veh,
            'world': world,
            'num_envs': num_envs,
        }
        super().__init__(self.info['node_name'])
        ################ ROS2 params ################
        self.cli = {
            'gz_control': self.create_client(ControlWorld, f'/world/{world}/control'),
        }
        self.srv = {}
        self.sub = {}
        self.pub = {}

        ################ GYM params #################
        self.info['maxstep'] = 4096
        hist_frame = 5
        self.__obs_shape = {
            'ang': (hist_frame, 4),
            'cmd_vel': (hist_frame, 7),
            'pos_vel': (hist_frame, 3),
            'ang_vel': (hist_frame, 3),
            'pos_acc': (hist_frame, 3),
            'ang_acc': (hist_frame, 3),
        }
        self.action_space = gym.space.Box(low=-1.0, high=1.0, shape=(7, ), dtype=np.float, seed=seed)
        self.observation_space = gym.space.Dict({
            'ang': gym.space.Box(low=-1.0, high=1.0, shape=self.__obs_shape['ang'], dtype=np.float32, seed=seed),
            'cmd_vel': gym.space.Box(low=-1.0, high=1.0, shape=self.__obs_shape['cmd_vel'], dtype=np.float32, seed=seed),
            'pos_vel': gym.space.Box(low=-1.0, high=1.0, shape=self.__obs_shape['pos_vel'], dtype=np.float32, seed=seed),
            'ang_vel': gym.space.Box(low=-1.0, high=1.0, shape=self.__obs_shape['ang_vel'], dtype=np.float32, seed=seed),
            'pos_acc': gym.space.Box(low=-1.0, high=1.0, shape=self.__obs_shape['pos_acc'], dtype=np.float32, seed=seed),
            'ang_acc': gym.space.Box(low=-1.0, high=1.0, shape=self.__obs_shape['ang_acc'], dtype=np.float32, seed=seed),
        })
        #############################################
    
    def reset(self, seed=None, options=None):
        while not self.cli['gz_control'].wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for GZ world: {self.info['world']} control service...')
        req = ControlWorld.Request()
        req.world_control.reset.all = True
        future = self.cli['gz_control'].call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'GZ world: {self.info['world']} reseted')
        else:
            self.get_logger().error(f'GZ world: {self.info['world']} failed to reset')


    def step(self, action):
        pass

    def close(self):
        self.destroy_node()

    ########################################## private funcs ####################################

    def __get_reward(self, action):
        pass

    def __get_observation(self):
        pass