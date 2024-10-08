import time, math, random, sys, os, queue, subprocess
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from typing import Optional
import numpy as np

import asyncio
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from ros_gz_interfaces.srv import ControlWorld
from gymnasium_arg.utils.gz_model import BlueBoat_GZ_MODEL


class BlueBoat_V1(gym.Env, Node):
    
    metadata = {
                "render_modes": ["rgb_array", "human"],
                }
    
    def __init__(self, 
                 veh='blueboat', 
                 world='waves',
                 num_envs=1,
                 headless=False,
                 maxstep=4096, 
                 max_thrust=10.0,
                 hist_frame=5,
                 seed=0
                 ):
        self.info = {
            'node_name': f'blueboat_v1',
            'veh': veh,
            'world': world,
            'num_envs': num_envs,
            'hist_frame': hist_frame,
            'maxstep': maxstep,
            'max_thrust': max_thrust,
            'headless': headless,
        }
        
        rclpy.init()
        Node.__init__(self, self.info['node_name'])
        ################ ROS2 params ################
        self.gz_world = self.create_client(ControlWorld, f'/world/{world}/control')
        self.__pause()
        ################ blueboats   ################
        self.vehs = []
        num_x = int(np.sqrt(num_envs))
        num_y = num_envs//num_x
        dis = 5
        for i in range(num_x):
            for j in range(num_y):
                self.vehs.append(BlueBoat_GZ_MODEL(
                    world=world, 
                    name=f'{veh}{i*num_y+j}', 
                    path='/home/arg/ros2_gazebo/Gazebo/models/blueboat', 
                    pose=Pose(
                        position=Point(x=float(i*dis-dis*num_x//2), y=float(j*dis-dis*num_y//2), z=0.8),
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    ),
                    info={'veh':'blueboat', 'maxstep': maxstep, 'max_thrust': max_thrust, 'hist_frame': hist_frame}
                ))
        for i in range(num_envs-num_x*num_y):
            self.vehs.append(BlueBoat_GZ_MODEL(
                world=world,
                name=f'{veh}{i+num_y*num_x}',
                path='/home/arg/ros2_gazebo/Gazebo/models/blueboat',
                pose=Pose(
                    position=Point(x=float(i*dis+dis*num_x//2), y=float(dis*num_y//2), z=0.8), 
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                ),
                info={'veh':'blueboat', 'maxstep': maxstep, 'max_thrust': max_thrust, 'hist_frame': hist_frame}
            )
        )
        ################ GYM params #################
        self.info['maxstep'] = 4096
        # self.__obs_shape = {
        #     'ang': (self.info['hist_frame'], 4),
        #     'cmd_vel': (self.info['hist_frame'], 6),
        #     'pos_acc': (self.info['hist_frame'], 3),
        #     'ang_vel': (self.info['hist_frame'], 3),
        # }
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7, ), dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_envs, 19), dtype=np.float32, seed=seed)
        # self.observation_space = gym.spaces.Dict({
        #     'cmd_vel': gym.spaces.Box(low=-1.0, high=1.0, shape=self.__obs_shape['cmd_vel'], dtype=np.float32, seed=seed),
        #     'ang': gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.__obs_shape['ang'], dtype=np.float32, seed=seed),
        #     'pos_acc': gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.__obs_shape['pos_acc'], dtype=np.float32, seed=seed),
        #     'ang_vel': gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.__obs_shape['ang_vel'], dtype=np.float32, seed=seed),
        # })
        #############################################
        self.__unpause()
        self.get_observation()
    
    def reset(self, seed=None, options=None):
        self.__pause()
        while not self.gz_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for GZ world: {self.info['world']} control service...")

        req = ControlWorld.Request()
        req.world_control.reset.all = True
        future = self.gz_world.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error(f"GZ world: {self.info['world']} failed to reset")
        for i in range(self.info['num_envs']):
            self.reset_idx(i)
        self.__unpause()
        return self.get_observation()

    def step(self, actions):
        self.actions = actions
        cmd_vel = TwistStamped()
        cmd_vel.header.stamp = self.get_clock().now().to_msg()
        if actions.shape[0] != self.info['num_envs']:
            cmd_vel.twist.linear.x = actions[0]
            cmd_vel.twist.linear.y = actions[1]
            cmd_vel.twist.linear.z = actions[2]
            cmd_vel.twist.angular.x = actions[3]
            cmd_vel.twist.angular.y = actions[4]
            cmd_vel.twist.angular.z = actions[5]
            self.vehs[0].step(cmd_vel)
        else:
            for i, action in enumerate(actions):
                cmd_vel.twist.linear.x = action[0]
                cmd_vel.twist.linear.y = action[1]
                cmd_vel.twist.linear.z = action[2]
                cmd_vel.twist.angular.x = action[3]
                cmd_vel.twist.angular.y = action[4]
                cmd_vel.twist.angular.z = action[5]
                self.vehs[i].step(cmd_vel)
        state = {
            'obs': self.get_observation(), # 2D array
            'reward': self.get_reward(actions), # 1D array
            'termination': self.get_termination(), # 1D array
            'truncation': self.get_truncation(), # 1D array
        }
        return state['obs'], state['reward'], state['termination'], state['truncation'], self.info

    def reset_idx(self, idx):
        self.vehs[idx].reset()
        return

    # def step_idx(self, action, idx):
    #     self.vehs[idx].pub['cmd_vel'].publish(action)

    def close(self):
        for veh in self.vehs:
            veh.close()
        self.destroy_node()

    def get_observation(self):
        obs = np.array([])
        for veh in self.vehs:
            pose = veh.obs['pose']
            twist = veh.obs['twist']
            veh_obs = np.array([veh.obs['action'].linear.x, veh.obs['action'].linear.y, veh.obs['action'].linear.z, 
                                veh.obs['action'].angular.x, veh.obs['action'].angular.y, veh.obs['action'].angular.z])
            veh_obs = np.hstack((veh_obs, np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])))
            veh_obs = np.hstack((veh_obs, np.array([twist.linear.x, twist.linear.y, twist.linear.z])))
            veh_obs = np.hstack((veh_obs, np.array([twist.angular.x, twist.angular.y, twist.angular.z])))
            # cmd_vel = np.array([veh.obs['action'].linear.x, veh.obs['action'].linear.y, veh.obs['action'].linear.z, 
            #                     veh.obs['action'].angular.x, veh.obs['action'].angular.y, veh.obs['action'].angular.z])
            # ang = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            # pos_acc = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
            # ang_vel = np.array([twist.angular.x, twist.angular.y, twist.angular.z])
            obs = np.vstack((obs, veh_obs)) if obs.size else np.hstack((obs, veh_obs))
        return obs

    def get_reward(self, actions):
        reward = []
        for i in range(self.info['num_envs']):
            reward.append(self.get_reward_idx(actions[i], i))
        return np.array(reward)

    def get_reward_idx(self, action, idx):
        rew = 0.0
        '''
        reward function here
        '''        
        return rew
    
    def get_termination(self):
        termination = []
        for veh in self.vehs:
            termination.append(veh.obs['termination'])
        return np.array(termination)
    
    def get_truncation(self):
        truncation = []
        for veh in self.vehs:
            truncation.append(veh.obs['truncation'])
        return np.array(truncation)

    ########################################## private funcs ####################################

    def __pause(self):
        while not self.gz_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for GZ world control service...')
        req = ControlWorld.Request()
        req.world_control.pause = True
        future = self.gz_world.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('GZ world paused')
        else:
            self.get_logger().error('Failed to pause GZ world')

    def __unpause(self):
        while not self.gz_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for GZ world control service...')
        req = ControlWorld.Request()
        req.world_control.pause = False
        future = self.gz_world.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('GZ world unpaused')
        else:
            self.get_logger().error('Failed to unpause GZ world')
