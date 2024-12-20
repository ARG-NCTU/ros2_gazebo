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
from gymnasium_arg.utils.gz_model_wamv_v1 import WAMVV1_GZ_MODEL
from gymnasium_arg.utils.gz_model_wamv_v2 import WAMVV2_GZ_MODEL
from gymnasium_arg.utils.gz_model_wamv_v3 import WAMVV3_GZ_MODEL
from gymnasium_arg.utils.vae import VAE, vae_loss
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R


vehs = {
    'wamv_v3': WAMVV3_GZ_MODEL,
}

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

class USV_V2(gym.Env):
    
    
    def __init__(self, 
                 veh='wamv_v3',
                 world='waves',
                 headless=False,
                 render_mode=Optional[str],
                 maxstep=4096, 
                 max_thrust=10, # 15*746/9.8 wamv or 10 blueboat
                 hist_frame=50, # 50Hz * 50 = 1s
                 seed=0,
                 hz=50,
                 ):
        self.info = {
            'node_name': f'{world}_{veh}_env',
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
        
        super(USV_V2, self).__init__()
        ################ ROS2 params ################
        rclpy.init()
        self.excutor = MultiThreadedExecutor()
        self.node = rclpy.create_node(self.info['node_name'])
        self.gz_world = self.node.create_client(ControlWorld, f'/world/{world}/control')
        self.clock = GzClock()
        self.__reset_world()
        self.__pause()
        ################ veh   ################
        self.veh = vehs[veh](
            world=world,
            name=veh,
            path=f'/home/arg/ros2_gazebo/Gazebo/models/{veh}',
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=0.8),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            info={'veh':veh, 'maxstep': maxstep, 'max_thrust': max_thrust, 'hist_frame': hist_frame}
            )
        self.excutor.add_node(self.veh)
        self.excutor.add_node(self.node)
        self.excutor.add_node(self.clock)
        self.excutor_thread = threading.Thread(target=self.excutor.spin)
        self.excutor_thread.start()
        ################ GYM params #################
        self.info['maxstep'] = 4096
        self.__action_shape = (4, )
        self.__obs_shape = {
            'imu': (hist_frame, 8),
            'action': (hist_frame, 4),
            'cmd_vel': (3, ),
        }
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.__action_shape, dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Box(
            low=np.hstack((
                    np.full((np.prod(self.__obs_shape['imu']), ), -np.inf), 
                    np.full((np.prod(self.__obs_shape['action']), ), -1.0), 
                    np.full((np.prod(self.__obs_shape['cmd_vel']), ), -1.0)
                    )),
            high=np.hstack((
                    np.full((np.prod(self.__obs_shape['imu']), ), np.inf), 
                    np.full((np.prod(self.__obs_shape['action']), ), 1.0), 
                    np.full((np.prod(self.__obs_shape['cmd_vel']), ), 1.0)
                    )),
            shape=(np.prod(self.__obs_shape['imu']) + np.prod(self.__obs_shape['action']) + np.prod(self.__obs_shape['cmd_vel']),),
            dtype=np.float32,
            seed=seed
        )
        #############################################
        self.cmd_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.refer_pose = np.array([0, 0, 0], dtype=np.float32)
        # self.__unpause()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.__reset_world()
        self.__pause()
        self.dp_cnt = 0
        # self.last_goal_diff = 0
        x = random.uniform(-1, 1)
        y = np.sqrt(1 - x**2)*random.uniform(-1, 1)
        yaw = random.uniform(-np.pi/4, np.pi/4)
        self.refer_pose = np.array([x, y, yaw], dtype=np.float32)
        self.refer_pose[:2] = self.refer_pose[:2]*random.uniform(0.8, 2)
        # self.cmd_vel = np.array([0.0, 0.0, 0.0])
        self.veh.reset()
        self.info['last_clock_time'] = None
        self.action = np.zeros(self.__action_shape)
        self.__unpause()
        return self.get_observation(), self.veh.info

    def step(self, action):
        self.__unpause()
        self.__clock_sync()
        self.info['total_step'] += 1
        self.action = action

        self.veh.step(action)

        state = {
            'obs': self.get_observation(),  # Using latent representation as the observation
            'termination': self.get_termination(),
            'truncation': self.get_truncation(),
        }
        state['reward'], state['constraint_costs'] = self.get_reward_constraint()

        sgn_bool = lambda x: True if x >= 0 else False
        output = "\rstep:{:4d}, cmd: [x:{}, y:{}, yaw:{}], action: [l_t:{}, r_t:{}, l_a:{}, r_a:{}], rews: [{}, {}, {}, {}] const:[{}]".format(
            self.veh.info['step_cnt'],
            " {:4.2f}".format(self.cmd_vel[0]) if sgn_bool(self.cmd_vel[0]) else "{:4.2f}".format(self.cmd_vel[0]),
            " {:4.2f}".format(self.cmd_vel[1]) if sgn_bool(self.cmd_vel[1]) else "{:4.2f}".format(self.cmd_vel[1]),
            " {:4.2f}".format(self.cmd_vel[2]) if sgn_bool(self.cmd_vel[2]) else "{:4.2f}".format(self.cmd_vel[2]),
            " {:4.2f}".format(action[0]) if sgn_bool(action[0]) else "{:4.2f}".format(action[0]),
            " {:4.2f}".format(action[1]) if sgn_bool(action[1]) else "{:4.2f}".format(action[1]),
            " {:4.2f}".format(action[2]) if sgn_bool(action[2]) else "{:4.2f}".format(action[2]),
            " {:4.2f}".format(action[3]) if sgn_bool(action[3]) else "{:4.2f}".format(action[3]),
            " {:4.2f}".format(state['reward'][0]) if sgn_bool(state['reward'][0]) else "{:4.2f}".format(state['reward'][0]),
            " {:4.2f}".format(state['reward'][1]) if sgn_bool(state['reward'][1]) else "{:4.2f}".format(state['reward'][1]),
            " {:4.2f}".format(state['reward'][2]) if sgn_bool(state['reward'][2]) else "{:4.2f}".format(state['reward'][2]),
            " {:4.2f}".format(state['reward'][3]) if sgn_bool(state['reward'][3]) else "{:4.2f}".format(state['reward'][3]),
            " {:4.2f}".format(state['constraint_costs'][0]) if sgn_bool(state['constraint_costs'][0]) else "{:4.2f}".format(state['constraint_costs'][0]),
        )
        sys.stdout.write(output)
        sys.stdout.flush()


        state['reward'] = state['reward'].sum()
        
        if self.veh.info['step_cnt'] >= self.info['maxstep']:
            state['truncation'] = True

        if state['reward'] >=0.9:
            if self.dp_cnt >= 100:
                state['termination'] = True
            else:
                self.dp_cnt += 1
        else:
            self.dp_cnt = 0

        info = self.veh.info
        info['constraint_costs'] = np.array(state['constraint_costs'], dtype=np.float32)

        if state['termination'] or state['truncation']:
            self.veh.step(np.zeros(self.__action_shape))

        self.__pause()
        
        return state['obs'], state['reward'], state['termination'], state['truncation'], info

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

    def get_observation(self):
        veh_obs = self.veh.get_observation()
        imu_obs = veh_obs['imu']
        r, p, y = R.from_quat(imu_obs[:, :4]).as_euler('xyz', degrees=False).T
        yaw = self.refer_pose[2]-y[0]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        imu_obs = np.hstack((np.array([r]).T, np.array([p]).T, imu_obs[:, 4:])).flatten()
        action_obs = veh_obs['action'].flatten()


        goal_pose = np.hstack((self.refer_pose[:2], 0, R.from_euler('xyz', [0, 0, self.refer_pose[2]]).as_quat()))
        goal_diff = relative_pose_tf(goal_pose, self.veh.obs['pose'][0])
        ang_goal_diff = np.arctan2(goal_diff[1], goal_diff[0])
        norm_goal_diff = np.linalg.norm(goal_diff[:2], ord=2)
        self.cmd_vel = np.array([np.cos(ang_goal_diff), np.sin(ang_goal_diff), yaw], dtype=np.float32)
        if norm_goal_diff < 1:
            self.cmd_vel[:2] = self.cmd_vel[:2]*norm_goal_diff

        obs = np.concatenate([
            imu_obs, 
            action_obs, 
            self.cmd_vel
        ], axis=0)

        return obs

    def get_reward_constraint(self):
        ## reward and constraints ##
        k1 = 50 # Reward weight for navigating to center
        k2 = 50 # Reward weight for maintaining heading
        k3 = 10 # Reward weight for smooth action
        k4 = 5 # Reward weight for reserving energy

        # Reward of navigating to center
        # rew1 = k1*torch.exp(-(np.linalg.norm(self.refer_pose[:2]-self.veh.obs['pose'][0][:2], p=2)**2/0.25))
        now_dis = np.linalg.norm(self.refer_pose[:2]-self.veh.obs['pose'][0][:2], ord=2)
        last_dis = np.linalg.norm(self.refer_pose[:2]-self.veh.obs['pose'][1][:2], ord=2)
        rew1 = k1*(1-now_dis)
        if rew1 <= -50:
            rew1 = -50
        rew1 += (last_dis-now_dis)*k1/4

        # Reward of maintaining heading
        veh_quat = self.veh.obs['pose'][0][3:7]
        ref_yaw = self.refer_pose[2]
        rew2 = k2*np.cos(ref_yaw - R.from_quat(veh_quat).as_euler('xyz', degrees=False)[2])
        # Reward of smooth action
        rew3 = -k3 * np.linalg.norm(self.veh.obs['action'][0] - self.veh.obs['action'][1], ord=1) / self.__action_shape[0]

        # Reward of reserving energy
        rew4 = -k4 * np.linalg.norm(self.action[:2], ord=1) / 2
        # Constraint
        const = []

        # Constraint of thrust
        cmd_vel_norm = np.linalg.norm(self.cmd_vel[:2], ord=2)
        cons1 = np.sum(abs((abs(self.action[:2]) - cmd_vel_norm)))/2

        const.append(cons1)

        return np.array([rew1, rew2, rew3, rew4])/self.info['max_rew'], const
    
    def get_termination(self):
        return self.veh.obs['termination']
    
    def get_truncation(self):
        return self.veh.obs['truncation']

    ########################################## private funcs ####################################
    def __pause(self):
        while not self.gz_world.wait_for_service(timeout_sec=1.0):
            pass
            # self.node.get_logger().info('Waiting for GZ world control service...')
        req = ControlWorld.Request()
        req.world_control.pause = True
        future = self.gz_world.call_async(req)
        while not future.done():
            rclpy.spin_once(self.node, timeout_sec=0)  # Non-blocking spin
            # time.sleep(0.01)
        # Use a temporary executor
        # temp_executor = rclpy.executors.SingleThreadedExecutor()
        # temp_executor.add_node(self.node)
        # temp_executor.spin_until_future_complete(future)
        # temp_executor.shutdown()
        if future.result() is None:
            self.node.get_logger().error('Failed to pause GZ world')
        # time.sleep(0.01)

    def __unpause(self):
        while not self.gz_world.wait_for_service(timeout_sec=1.0):
            pass
            # self.node.get_logger().info('Waiting for GZ world control service...')
        req = ControlWorld.Request()
        req.world_control.pause = False
        future = self.gz_world.call_async(req)
        # Wait for the future to complete without spinning the node
        while not future.done():
            rclpy.spin_once(self.node, timeout_sec=0)  # Non-blocking spin
            # time.sleep(0.01)
        if future.result() is None:
            self.node.get_logger().error('Failed to unpause GZ world')
        # time.sleep(0.01)
    
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
