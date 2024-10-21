import time, math, random, sys, os, queue, subprocess, threading
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from typing import Optional
import numpy as np

import asyncio
import rclpy
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.srv import ControlWorld
from gymnasium_arg.utils.gz_model_wamv_v1 import WAMVV1_GZ_MODEL
from gymnasium_arg.utils.vae import VAE, vae_loss
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class GzClock(Node):
    def __init__(self):
        super().__init__('gz_clock')
        self.subscription = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )
        self.current_time = None

    def clock_callback(self, msg):
        self.current_time = msg.clock.sec + msg.clock.nanosec * 1e-9

class WAMV_V1(gym.Env):
    
    
    def __init__(self, 
                 veh='wamv_v1',
                 world='waves',
                 headless=False,
                 render_mode=Optional[str],
                 maxstep=4096, 
                 max_thrust=15*746/9.8,
                 hist_frame=10,
                 seed=0,
                 hz=50,
                 ):
        self.info = {
            'node_name': f'wamv_v1_gym',
            'veh': veh,
            'world': world,
            'hist_frame': hist_frame,
            'maxstep': maxstep,
            'max_rew': 100.0,
            'max_thrust': max_thrust,
            'headless': headless,
            'hz': hz,
            'last_clock_time': None,
            'period': 1.0 / hz,
            'latent_dim': 32,
            'total_step': 0,
        }
        
        
        ################ ROS2 params ################
        rclpy.init()
        self.excutor = MultiThreadedExecutor()
        self.node = rclpy.create_node(self.info['node_name'])
        self.gz_world = self.node.create_client(ControlWorld, f'/world/{world}/control')
        self.clock = GzClock()
        self.__reset_world()
        self.__pause()
        ################ blueboats   ################
        self.veh = WAMVV1_GZ_MODEL(
            world=world,
            name=veh,
            path='/home/arg/ros2_gazebo/Gazebo/models/wamv_v1',
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=0.8),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            info={'veh':'wamv_v1', 'maxstep': maxstep, 'max_thrust': max_thrust, 'hist_frame': hist_frame}
            )
        self.excutor.add_node(self.veh)
        self.excutor.add_node(self.node)
        self.excutor.add_node(self.clock)
        self.excutor_thread = threading.Thread(target=self.excutor.spin)
        self.excutor_thread.start()
        ################ GYM params #################
        self.info['maxstep'] = 4096
        self.__action_shape = (6, )
        self.__obs_shape = (
            self.info['hist_frame']*self.__action_shape[0] + # hist action: hist cmd_vel
            self.info['hist_frame']*10  # hist imu (ori + ang_vel + pos_acc)
        )
        i = 1
        tb_vae_name = f'./tb_vae/{self.info["node_name"]}_{i}'
        while os.path.exists(f'{tb_vae_name}'):
            i += 1
            tb_vae_name = f'./tb_vae/{self.info["node_name"]}_{i}'
        self.vae_writer = SummaryWriter(tb_vae_name)

        self.vae = VAE(obs_dim=self.__obs_shape, latent_dim=self.info['latent_dim'])
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)
        self.cmd_vel = {
            'forward': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'backward': np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'left': np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            'right': np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            'yaw_left': np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            'yaw_right': np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        }

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.__action_shape, dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.info['latent_dim']+6, ), dtype=np.float32, seed=seed)
        
        #############################################
        self.__unpause()
    
    def reset(self, seed=None, options=None):
        self.__pause()
        self.veh.reset()
        self.info['last_clock_time'] = None
        self.action = np.zeros(self.__action_shape)
        self.__unpause()
        return self.get_observation(np.zeros(self.__action_shape)), self.veh.info

    def step(self, action):
        self.__clock_sync()
        self.info['total_step'] += 1
        self.action = action
        cmd_vel = TwistStamped()
        cmd_vel.header.stamp = self.node.get_clock().now().to_msg()
        cmd_vel.twist.linear.x = float(action[0])
        cmd_vel.twist.linear.y = float(action[1])
        cmd_vel.twist.linear.z = float(action[2])
        cmd_vel.twist.angular.x = float(action[3])
        cmd_vel.twist.angular.y = float(action[4])
        cmd_vel.twist.angular.z = float(action[5])

        self.veh.step(cmd_vel)

        state = {
            'obs': self.get_observation(action),  # Using latent representation as the observation
            'reward': self.get_reward(action),
            'termination': self.get_termination(), # 1D array
            'truncation': self.get_truncation(), # 1D array
        }

        return state['obs'], state['reward'], state['termination'], state['truncation'], self.veh.info

    def close(self):
        self.veh.close()
        self.vae_writer.close()
        self.node.destroy_node()
        self.clock.destroy_node()
        self.excutor.shutdown()
        rclpy.shutdown()

    def get_observation(self, cmd_vel):
        veh_obs = self.veh.get_observation()
        veh_obs = np.hstack((veh_obs['action'], veh_obs['imu'])).flatten()

        obs_tensor = torch.FloatTensor(veh_obs).unsqueeze(0)
        recon_obs, mu, logvar = self.vae(obs_tensor)

        latent_obs = self.vae.reparameterize(mu, logvar).detach().numpy().flatten()

        obs = np.hstack((self.cmd_vel['forward'], latent_obs))

        self.vae_optimizer.zero_grad()
        loss = vae_loss(recon_obs, obs_tensor, mu, logvar)
        loss.backward()
        self.vae_writer.add_scalar('Loss/train', loss.item(), self.info['total_step'])

        self.vae_optimizer.step()

        return obs

    def get_reward(self, action):
        rew = 0
        '''
            apply reward function here
        '''
        rew += action[0]*self.cmd_vel['forward'][0] * 10
        return rew/self.info['max_rew']
    
    def get_termination(self):
        return self.veh.obs['termination']
    
    def get_truncation(self):
        return self.veh.obs['truncation']

    ########################################## private funcs ####################################
    def __pause(self):
        while not self.gz_world.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for GZ world control service...')
        req = ControlWorld.Request()
        req.world_control.pause = True
        future = self.gz_world.call_async(req)
        # Use a temporary executor
        temp_executor = rclpy.executors.SingleThreadedExecutor()
        temp_executor.add_node(self.node)
        temp_executor.spin_until_future_complete(future)
        temp_executor.shutdown()
        if future.result() is not None:
            self.node.get_logger().info('GZ world paused')
        else:
            self.node.get_logger().error('Failed to pause GZ world')
        time.sleep(0.1)

    def __unpause(self):
        while not self.gz_world.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for GZ world control service...')
        req = ControlWorld.Request()
        req.world_control.pause = False
        future = self.gz_world.call_async(req)
        # Wait for the future to complete without spinning the node
        while not future.done():
            rclpy.spin_once(self.node, timeout_sec=0)  # Non-blocking spin
            time.sleep(0.01)
        if future.result() is not None:
            self.node.get_logger().info('GZ world unpaused')
        else:
            self.node.get_logger().error('Failed to unpause GZ world')
        time.sleep(0.1)
    
    def __reset_world(self):
        while not self.gz_world.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f"Waiting for GZ world: {self.info['world']} control service...")

        req = ControlWorld.Request()
        req.world_control.reset.all = True
        future = self.gz_world.call_async(req)
        
        # Wait for the future to complete without blocking the executor
        while not future.done():
            rclpy.spin_once(self.node, timeout_sec=0)  # Non-blocking spin
            time.sleep(0.01)  # Sleep briefly to prevent busy waiting

        if future.result() is None:
            self.node.get_logger().error(f"GZ world: {self.info['world']} failed to reset")
        else:
            self.node.get_logger().info(f"GZ world: {self.info['world']} reset successfully")
        time.sleep(0.1)


    def __clock_sync(self):
        if self.info['last_clock_time'] is None:
            self.info['last_clock_time'] = self.clock.current_time
            return
        while self.clock.current_time - self.info['last_clock_time'] < self.info['period']:
            pass
        self.info['last_clock_time'] = self.clock.current_time

def quaternion_to_direction(q):
    """
    Convert a quaternion (orientation) to a forward direction vector.
    This assumes the forward direction of the vehicle is along the X-axis in local space.
    """
    x, y, z, w = q
    # Formula to convert quaternion to direction vector
    forward_vector = np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x**2 + y**2)
    ])
    return forward_vector