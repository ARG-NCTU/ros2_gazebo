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
from gymnasium_arg.utils.gz_model import BlueBoat_GZ_MODEL
from gymnasium_arg.utils.vae import VAE, vae_loss
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R


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

class BlueBoat_V3(gym.Env):
    
    
    def __init__(self, 
                 veh='blueboat',
                 world='waves',
                 headless=False,
                 render_mode=Optional[str],
                 maxstep=4096, 
                 max_thrust=10.0,
                 hist_frame=50, # 50Hz * 50 = 1s
                 seed=0,
                 hz=50,
                 ):
        self.info = {
            'node_name': f'blueboat_v3',
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
        self.veh = BlueBoat_GZ_MODEL(
            world=world,
            name=veh,
            path='/home/arg/ros2_gazebo/Gazebo/models/blueboat',
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=0.8),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            info={'veh':'blueboat', 'maxstep': maxstep, 'max_thrust': max_thrust, 'hist_frame': hist_frame}
            )
        self.excutor.add_node(self.veh)
        self.excutor.add_node(self.node)
        self.excutor.add_node(self.clock)
        self.excutor_thread = threading.Thread(target=self.excutor.spin)
        self.excutor_thread.start()
        ################ GYM params #################
        self.info['maxstep'] = 4096
        self.__action_shape = (2, )
        self.__obs_shape = {
            'imu': (hist_frame, 10),
            'action': (hist_frame, 6),
            'latent': self.info['latent_dim'],
            # 'rl_obs': 6+self.info['latent_dim'],
            'cmd_vel': (6, ),
        }
        # (
        #     self.info['hist_frame']*self.__action_shape[0] + # hist action: hist cmd_vel
        #     self.info['hist_frame']*10  # hist imu (ori + ang_vel + pos_acc)
        # )
        # i = 1
        # tb_vae_name = f'./tb_vae/{self.info["node_name"]}_{i}'
        # while os.path.exists(f'{tb_vae_name}'):
        #     i += 1
        #     tb_vae_name = f'./tb_vae/{self.info["node_name"]}_{i}'
        # self.vae_writer = SummaryWriter(tb_vae_name)

        # self.vae = VAE(imu_dim=self.__obs_shape['imu'], action_dim=self.__obs_shape['action'], latent_dim=self.__obs_shape['latent'])
        # self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=1e-5)
        self.cmd_vel = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.__action_shape, dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.prod(self.__obs_shape['imu'])+np.prod(self.__obs_shape['action'])+np.prod(self.__obs_shape['cmd_vel']),),
            dtype=np.float32,
            seed=seed
        )
        # gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.__obs_shape['rl_obs'], ), dtype=np.float32, seed=seed)
        
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
        cmd_vel.twist.linear.y = 0.0
        cmd_vel.twist.linear.z = 0.0
        cmd_vel.twist.angular.x = 0.0
        cmd_vel.twist.angular.y = 0.0
        cmd_vel.twist.angular.z = float(action[1])
        # cmd_vel.twist.linear.x = float(action[0])
        # cmd_vel.twist.linear.y = float(action[1])
        # cmd_vel.twist.linear.z = float(action[2])
        # cmd_vel.twist.angular.x = float(action[3])
        # cmd_vel.twist.angular.y = float(action[4])
        # cmd_vel.twist.angular.z = float(action[5])

        self.veh.step(cmd_vel)

        state = {
            'obs': self.get_observation(action),  # Using latent representation as the observation
            'termination': self.get_termination(),
            'truncation': self.get_truncation(),
        }
        state['reward'], state['constraint_costs'] = self.get_reward_constraint(self.cmd_vel, action)

        sgn_bool = lambda x: True if x >= 0 else False

        output = "\rstep:{:4d}, cmd: [x:{}, yaw:{}], rews: [{}, {}, {}]".format(
            self.veh.info['step_cnt'],
            " {:4.2f}".format(action[0]) if sgn_bool(action[0]) else "{:4.2f}".format(action[0]),
            " {:4.2f}".format(action[1]) if sgn_bool(action[1]) else "{:4.2f}".format(action[1]),
            # " {:4.2f}".format(action[2]) if sgn_bool(action[2]) else "{:4.2f}".format(action[2]),
            # " {:4.2f}".format(action[3]) if sgn_bool(action[3]) else "{:4.2f}".format(action[3]),
            # " {:4.2f}".format(action[4]) if sgn_bool(action[4]) else "{:4.2f}".format(action[4]),
            # " {:4.2f}".format(action[5]) if sgn_bool(action[5]) else "{:4.2f}".format(action[5]),
            " {:4.2f}".format(state['reward'][0]) if sgn_bool(state['reward'][0]) else "{:4.2f}".format(state['reward'][0]),
            " {:4.2f}".format(state['reward'][1]) if sgn_bool(state['reward'][1]) else "{:4.2f}".format(state['reward'][1]),
            " {:4.2f}".format(state['reward'][2]) if sgn_bool(state['reward'][2]) else "{:4.2f}".format(state['reward'][2]),
        )
        sys.stdout.write(output)
        sys.stdout.flush()

        info = self.veh.info
        info['constraint_costs'] = np.array(state['constraint_costs'], dtype=np.float32)
        
        return state['obs'], state['reward'].sum(), state['termination'], state['truncation'], info

    def close(self):
    # Stop the executor thread before shutting down nodes
        if self.excutor_thread.is_alive():
            self.excutor.shutdown()
            self.excutor_thread.join()  # Wait for the thread to finish

        self.veh.close()
        # If you have commented out self.vae_writer, ensure it's not called
        # self.vae_writer.close()
        self.node.destroy_node()
        self.clock.destroy_node()
        rclpy.shutdown()

    def get_observation(self, cmd_vel):
        veh_obs = self.veh.get_observation()
        # veh_obs = np.hstack((veh_obs['action'], veh_obs['imu'])).flatten()
        
        # imu_obs = torch.FloatTensor(veh_obs['imu']).unsqueeze(0)  # Shape (1, 50, 10)
        # action_obs = torch.FloatTensor(veh_obs['action']).unsqueeze(0)  # Shape (1, 50, 6)
        
        # imu_recon, action_recon, mu, logvar = self.vae(imu_obs.permute(0, 2, 1), action_obs.permute(0, 2, 1))  # Permute to (batch, channels, time)
        # loss = vae_loss(imu_recon, action_recon, imu_obs, action_obs, mu, logvar, beta=0.5)
        # self.vae_optimizer.zero_grad()
        # loss.backward()
        # self.vae_writer.add_scalar('Loss/train', loss.item(), self.info['total_step'])
        # self.vae_optimizer.step()

        # latent_obs = self.vae.encode(imu_obs.permute(0, 2, 1), action_obs.permute(0, 2, 1))[0].detach().numpy().flatten()
        # obs = np.hstack((self.cmd_vel, latent_obs))

        imu_obs = veh_obs['imu'].flatten()
        action_obs = veh_obs['action'].flatten()
        # cmd_obs = np.array([self.cmd_vel[0], self.cmd_vel[1], self.cmd_vel[2], self.cmd_vel[3], self.cmd_vel[4], self.cmd_vel[5]])
        obs = np.concatenate([imu_obs, action_obs, self.cmd_vel])
        return obs

    def get_reward_constraint(self, cmd_vel, action):
        k2 = 10
        k3 = 50
        # operator = lambda x: 1 if x >= 0 else -1
        # sigmoid = lambda x: 1/(1+np.exp(-x))
        # relu = lambda x: x if x >= 0 else 0
        '''
            1. reward of following cmd_vel
            2. reward of save energy
            3. reward of smooth action
        '''
        # reward of following cmd_vel
        vec_vel = cmd_vel[:1] # from 0 to 1
        ori_acc = cmd_vel[3:] # from 3 to 5

        local_pose_diff = relative_pose_tf(self.veh.obs['pose'][0], self.veh.obs['pose'][10])
        veh_vel = local_pose_diff/(self.info['period']*10) / self.veh.info['max_lin_velocity']
        rew_vel = np.log(1+np.exp(-10*abs(veh_vel - vec_vel)))/np.log(2) # 2
        rew_ori = np.log(1+np.exp(-10*abs(self.veh.obs['imu'][0][4:7]/self.veh.info['max_ang_velocity'] - ori_acc)))/np.log(2) # 3
        rew1 = self.info['max_rew']*(2*(np.hstack((rew_ori, rew_vel)))-1).sum() / 5
        # rew1 = self.info['max_rew']*(2*np.log(1+np.exp(-10*(vec_vel - action)**2))/np.log(2) -1).sum() /self.__action_shape[0]

        # reward of save energy
        # rew2 = -k2*np.linalg.norm(relu(np.array([])), ord=1) / self.__action_shape[0]
        rew2 = 0

        # reward of smooth action
        rew3 = -k3*np.linalg.norm(self.veh.obs['action'][0]-self.veh.obs['action'][1], ord=1) / self.__action_shape[0]

        ###################### constraint ######################
        const = []
        # # smooth moving constraint
        # state['constraint_costs'].append(
        #     np.linalg.norm(self.veh.obs['imu'][0][4:] - self.veh.obs['imu'][1][4:]) / 6
        # )
        
        # smooth action constraint
        # state['constraint_costs'].append(
        #     np.linalg.norm(action - np.array([self.veh.obs['action'][0][0], self.veh.obs['action'][0][-1]]), ord=1) / self.__action_shape[0]
        # )

        # moving tolerrance constraint
        dot_product = np.dot(self.cmd_vel[:1], veh_vel[:1])
        magnitude_cmd = np.linalg.norm(self.cmd_vel[:1])
        magnitude_vel = np.linalg.norm(veh_vel[:1])
        if magnitude_cmd==0 or magnitude_vel/self.veh.info['max_lin_velocity']<=1e-3:
            const.append(abs(magnitude_cmd - magnitude_vel/self.veh.info['max_lin_velocity']))
        else:
            cos_theta = dot_product / (magnitude_cmd * magnitude_vel)
            const.append(
                1- np.cos(cos_theta)
            )

        # stable constraint
        const.append(
            np.linalg.norm(self.veh.obs['imu'][0][:1], ord=1) / 2
        )

        return np.array([rew1, rew2, rew3])/self.info['max_rew'], const
    
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

# relatively pose transfer with respect to orintation
def relative_pose_tf(pose1, pose2):
    # Calculate position difference
    dx, dy = pose1[:2] - pose2[:2]

    # Convert quaternion to yaw (heading)
    r = R.from_quat([pose2[3], pose2[4], pose2[5], pose2[6]])
    yaw = r.as_euler('xyz', degrees=False)[2]  # Extract yaw

    # Rotate dx, dy into the boat's local frame based on yaw
    rotation_matrix = R.from_euler('z', -yaw).as_matrix()[:2, :2]
    local_pose_diff = np.dot(rotation_matrix, np.array([dx, dy]))

    return local_pose_diff