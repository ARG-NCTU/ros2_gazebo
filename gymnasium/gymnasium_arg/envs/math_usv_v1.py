import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pygame
from scipy.spatial.transform import Rotation as R
import sys, os, random


class MATH_USV_V1(gym.Env):
    """
    Custom Gymnasium Environment for USV Dynamics Simulation with Pygame Rendering.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode="none", hist_frame=50, seed=0, device='cpu'):
        super(MATH_USV_V1, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        assert render_mode in self.metadata["render_modes"], "Invalid render mode"
        self.render_mode = render_mode
        self.info = {
            "hist_frame": hist_frame,
            "latent_dim": 32,
            'maxstep': 4096,
            'max_thrust': 15*746/9.8,
            'dt': 1 / 50,
            'max_rew': 100.0,
            'total_step': 0,
            'step_cnt': 0,
        }
        # Simulation parameters
        self.dt = 1 / 50  # Time step
        self.mass = 50.0  # USV mass
        self.drag_coefficient = 0.1  # Drag coefficient
        self.moment_of_inertia = 10.0  # Rotational inertia
        self.max_thrust = 100.0  # Max thrust for linear and angular controls
        self.veh_obs = {
            'imu': np.zeros((hist_frame, 10)),
            'action': np.zeros((hist_frame, 4)),
            'pose': np.zeros((hist_frame, 7)),
        }
        self.veh_body = {
            'length': 40,
            'width': 20,
            'mass': 50.0,
        }
        # Observation space: [local_acc_x, local_acc_y, angular_velocity, orientation]
        # self.observation_space = spaces.Box(
        #     low=np.array([-np.inf, -np.inf, -np.inf, -np.pi]),
        #     high=np.array([np.inf, np.inf, np.inf, np.pi]),
        #     dtype=np.float32,
        # )
        # Action space: [linear_thrust, angular_thrust]
        # self.action_space = spaces.Box(
        #     low=np.array([-self.max_thrust, -self.max_thrust]),
        #     high=np.array([self.max_thrust, self.max_thrust]),
        #     dtype=np.float32,
        # )
        self.__action_shape = (4, )
        self.__obs_shape = {
            'imu': (hist_frame, 10),
            'action': (hist_frame, 4),
            'latent': self.info['latent_dim'],
            # 'rl_obs': 6+self.info['latent_dim'],
            'cmd_vel': (3, ),
            'refer': (3, ),
        }

        self.cmd_vel = np.array([0.0, 0.0, 0.0])
        self.refer_pose = np.array([0, 0, 0], dtype=np.float32)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.__action_shape, dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.prod(self.__obs_shape['imu'])+np.prod(self.__obs_shape['action'])+np.prod(self.__obs_shape['cmd_vel'])+np.prod(self.__obs_shape['refer']),),
            dtype=np.float32,
            seed=seed
        )
        # Initialize state
        self.state = None  # [x, y, theta, v, omega]
        self.imu_data = None  # [local_acc_x, local_acc_y, angular_velocity, orientation]

        # World bounds
        self.world_size = np.array([800.0, 600.0])  # Width, Height

        # Pygame setup
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.world_size)
            pygame.display.set_caption("USV Simulator")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        
        # Reset state: [x, y, theta, v, omega]
        self.state = torch.tensor(
            # [x_new, y_new, theta_new, vx_new, vy_new, omega_new]
            [400.0, 300.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )  # Start in the center

        # Reset IMU data
        self.imu_data = self._calculate_imu_data()
        self.veh_obs = {
            'imu': np.full((self.info['hist_frame'], 10), self.imu_data.numpy()),
            'action': np.zeros((self.info['hist_frame'], 4)),
        }
        ori = self.veh_obs['imu'][0][:4]
        pose = np.array([self.state[0], self.state[1], 0.0, ori[0], ori[1], ori[2], ori[3]], dtype=np.float32)
        self.veh_obs['pose'] = np.full((self.info['hist_frame'], 7), pose)
        self.refer_pose = np.array([self.state[0], self.state[1], self.state[2]], dtype=np.float32)
        self.cmd_vel = np.array([0.0, 0.0, 0.0])
        self.action = np.zeros(self.__action_shape)
        self.info['step_cnt'] = 0

        if self.render_mode == "human":
            self._render_pygame()

        return self.get_observation(), {}

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        self.info['step_cnt'] += 1
        self.action = action

        # Extract action components
        mag_left, mag_right = action[0], action[1]
        angle_left, angle_right = action[2], action[3]

        # Update state
        self.state, self.imu_data = self._update_dynamics(
            self.state, mag_left, mag_right, angle_left, angle_right, self.dt
        )

        # Update observation buffers
        self.veh_obs['imu'] = np.roll(self.veh_obs['imu'], 1, axis=0)
        self.veh_obs['imu'][0] = self.imu_data.numpy()
        self.veh_obs['action'] = np.roll(self.veh_obs['action'], 1, axis=0)
        self.veh_obs['action'][0] = action
        pose = np.array([self.state[0], self.state[1], 0.0, self.imu_data[0], self.imu_data[1], self.imu_data[2], self.imu_data[3]], dtype=np.float32)
        self.veh_obs['pose'] = np.roll(self.veh_obs['pose'], 1, axis=0)
        self.veh_obs['pose'][0] = pose
        info = {}
        termination = False
        truncation = False
        obs = self.get_observation()

        ## reward and constraints ##
        k1 = 60
        k2 = 30
        k3 = 10
        operator = lambda x: 1 if x >= 0 else -1
        # sigmoid = lambda x: 1/(1+np.exp(-x))
        relu = lambda x: x if x >= 0 else 0
        '''
            1. reward of following dir vel
            2. reward of followning norm vel
            3. reward of smooth action
        '''
        # reward of following cmd_vel dir
        refer_ori = R.from_euler('xyz', [0, 0, self.refer_pose[2]]).as_quat()
        refer_pose = np.hstack((np.hstack((self.refer_pose[:2], 0)), refer_ori))
        local_pose_diff = relative_pose_tf(self.veh_obs['pose'][0], refer_pose, device=self.device)
        local_pose_norm = np.linalg.norm(local_pose_diff[:2], ord=2)
        cmd_dir = np.arctan2(self.cmd_vel[1], self.cmd_vel[0])
        veh_dir = np.arctan2(local_pose_diff[1], local_pose_diff[0])

        theta = np.cos(veh_dir - cmd_dir)
        rew1 = k1*theta-relu(-operator(theta)*local_pose_norm)

        # reward of thrust
        cmd_vel_norm = np.linalg.norm(self.cmd_vel[:2], ord=2)
        rew2 = 1 - 2*(abs(self.action[:2])-cmd_vel_norm)
        rew2 = k2*np.sum(rew2) / 2
        # rew2 = -k2*np.linalg.norm(relu(np.array([])), ord=1) / self.__action_shape[0]

        # reward of smooth action
        rew3 = -k3*np.linalg.norm(self.veh_obs['action'][0]-self.veh_obs['action'][1], ord=1) / self.__action_shape[0]
        ###################### constraint ######################
        const = []
        veh_yaw = R.from_quat([self.veh_obs['pose'][0][3], 
                               self.veh_obs['pose'][0][4], 
                               self.veh_obs['pose'][0][5], 
                               self.veh_obs['pose'][0][6]]).as_euler('xyz', degrees=False)[2]
        ref_yaw = self.refer_pose[2]
        const.append(
            (1-np.cos(veh_yaw - ref_yaw))/2
        )

        rew1 = rew1/self.info['max_rew']
        rew2 = rew2/self.info['max_rew']
        rew3 = rew3/self.info['max_rew']

        sgn_bool = lambda x: True if x >= 0 else False
        output = "\rstep:{:4d}, cmd: [x:{}, y:{}, yaw:{}], action: [l_t:{}, r_t:{}, l_a:{}, r_a:{}], rews: [{}, {}, {}] const:[{}]".format(
            self.info['step_cnt'],
            " {:4.2f}".format(self.cmd_vel[0]) if sgn_bool(self.cmd_vel[0]) else "{:4.2f}".format(self.cmd_vel[0]),
            " {:4.2f}".format(self.cmd_vel[1]) if sgn_bool(self.cmd_vel[1]) else "{:4.2f}".format(self.cmd_vel[1]),
            " {:4.2f}".format(self.cmd_vel[2]) if sgn_bool(self.cmd_vel[2]) else "{:4.2f}".format(self.cmd_vel[2]),
            " {:4.2f}".format(action[0]) if sgn_bool(action[0]) else "{:4.2f}".format(action[0]),
            " {:4.2f}".format(action[1]) if sgn_bool(action[1]) else "{:4.2f}".format(action[1]),
            " {:4.2f}".format(action[2]) if sgn_bool(action[2]) else "{:4.2f}".format(action[2]),
            " {:4.2f}".format(action[3]) if sgn_bool(action[3]) else "{:4.2f}".format(action[3]),
            " {:4.2f}".format(rew1) if sgn_bool(rew1) else "{:4.2f}".format(rew1),
            " {:4.2f}".format(rew2) if sgn_bool(rew2) else "{:4.2f}".format(rew2),
            " {:4.2f}".format(rew3) if sgn_bool(rew3) else "{:4.2f}".format(rew3),
            " {:4.2f}".format(const[0]) if sgn_bool(const[0]) else "{:4.2f}".format(const[0]),
            # " {:4.2f}".format(const[1]) if sgn_bool(const[1]) else "{:4.2f}".format(const[1]),
        )
        sys.stdout.write(output)
        sys.stdout.flush()

        rew = rew1 + rew2 + rew3
        if rew <= -1.5:
            termination = True
        if self.info['step_cnt'] % 1024 == 0:
            x = random.uniform(-1, 1)
            y = np.sqrt(1 - x**2)*random.uniform(-1, 1)
            self.cmd_vel = np.array([x, y, 0])
            yaw = R.from_quat([self.veh_obs['pose'][0][3], 
                               self.veh_obs['pose'][0][4], 
                               self.veh_obs['pose'][0][5], 
                               self.veh_obs['pose'][0][6]]).as_euler('xyz', degrees=False)[2]  # Extract yaw
            self.refer_pose = np.hstack((self.veh_obs['pose'][0][:2], yaw))
        
        info['constraint_costs'] = np.array(const, dtype=np.float32)

        if self.info['step_cnt'] >= self.info['maxstep']:
            truncation = True

        if self.render_mode == "human":
            self._render_pygame()

        return obs, rew, termination, truncation, info

    def get_observation(self):
        """
        Returns the current observation from the environment.
        """
        veh_pose = self.state.numpy()[:2]
        veh_pose = np.hstack((veh_pose, 0))
        veh_pose = np.hstack((veh_pose, self.imu_data.numpy()[:4]))
        imu_obs = self.veh_obs['imu'].flatten()
        action_obs = self.veh_obs['action'].flatten()
        ref_yaw = self.refer_pose[2]
        refer_ori = R.from_euler('xyz', [0, 0, self.refer_pose[2]]).as_quat()
        refer_pose = np.hstack((np.hstack((self.refer_pose[:2], 0)), refer_ori))
        local_pose_diff = relative_pose_tf(self.veh_obs['pose'][0], refer_pose, device=self.device)
        veh_yaw = R.from_quat([veh_pose[3], 
                               veh_pose[4], 
                               veh_pose[5], 
                               veh_pose[6]]).as_euler('xyz', degrees=False)[2]
        obs = np.concatenate([imu_obs, action_obs, self.cmd_vel, np.hstack((local_pose_diff[:2], veh_yaw-ref_yaw))])
        return obs

    def render(self):
        """
        Placeholder for Gymnasium render call.
        """
        if self.render_mode == "human":
            self._render_pygame()

    def close(self):
        """
        Closes the Pygame window if open.
        """
        if self.render_mode == "human":
            pygame.quit()

    def _update_dynamics(self, state, mag_left, mag_right, angle_left, angle_right, dt):
        """
        Updates the USV dynamics based on thruster inputs.
        """
        x, y, theta, vx, vy, omega = state

        # Convert angles to tensors
        angle_left = torch.tensor(angle_left, dtype=torch.float32, device=self.device)
        angle_right = torch.tensor(angle_right, dtype=torch.float32, device=self.device)

        # Thruster positions relative to the center of mass
        left_thruster_pos = torch.tensor([-self.veh_body['length'] / 2, self.veh_body['width'] / 2], device=self.device)
        right_thruster_pos = torch.tensor([-self.veh_body['length'] / 2, -self.veh_body['width'] / 2], device=self.device)

        # Forces in local frame
        force_left_local = torch.tensor([
            mag_left * torch.sqrt(1 - torch.pow(angle_left, 2)), 
            mag_left * angle_left
        ], device=self.device)
        force_right_local = torch.tensor([
            mag_right * torch.sqrt(1 - torch.pow(angle_right, 2)), 
            mag_right * angle_right
        ], device=self.device)

        # Rotate forces to the global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=self.device)

        force_left_global = rotation_matrix @ force_left_local
        force_right_global = rotation_matrix @ force_right_local

        # Net force and torque
        net_force = force_left_global + force_right_global

        # Torques (cross-product: r x F)
        torque_left = -(left_thruster_pos[0] * force_left_local[1]) + (left_thruster_pos[1] * force_left_local[0])
        torque_right = -(right_thruster_pos[0] * force_right_local[1]) + (right_thruster_pos[1] * force_right_local[0])
        net_torque = torque_left + torque_right

        # Accelerations
        ax = net_force[0] / self.mass
        ay = net_force[1] / self.mass
        alpha = net_torque / self.moment_of_inertia

        # Update velocities and positions
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        omega_new = omega + alpha * dt

        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        theta_new = theta + omega_new * dt
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi  # Wrap angle between [-pi, pi]

        # IMU data: x, y, z, w, ax, ay, az, wx, wy, wz
        # theta_new is yaw and make imu_data[:4] to be orientation of it
        orientation = R.from_euler('xyz', [0, 0, theta_new]).as_quat()
        imu_data = torch.tensor([orientation[0], orientation[1], orientation[2], orientation[3], ax, ay, -9.8, 0, 0, omega_new], dtype=torch.float32, device=self.device)
        return torch.tensor([x_new, y_new, theta_new, vx_new, vy_new, omega_new], dtype=torch.float32, device=self.device), imu_data

    def _calculate_imu_data(self):
        """
        Calculates the initial IMU data from the initial state.
        """
        orientation = R.from_euler('xyz', [0, 0, self.state[2]]).as_quat()
        return torch.tensor([orientation[0], orientation[1], orientation[2], orientation[3], 0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

    def _check_bounds(self):
        """
        Checks if the USV is within the world bounds.
        """
        x, y = self.state[0], self.state[1]
        return not (0 <= x <= self.world_size[0] and 0 <= y <= self.world_size[1])

    def _render_pygame(self):
        """
        Renders the environment using Pygame.
        """
        self.screen.fill((255, 255, 255))  # Clear the screen with white

        # Draw USV
        self._draw_usv(self.state[0].item(), self.state[1].item(), self.state[2].item())

        # Update display
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_usv(self, x, y, theta):
        """
        Draw the USV on the screen based on its position and orientation.
        """
        length = 40  # USV length
        width = 20  # USV width

        # Ensure theta is a tensor for torch operations
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, device=self.device)

        # Calculate corner points based on rotation
        corners = torch.tensor([
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2]
        ], device=self.device)

        # Rotation matrix
        rotation = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ], device=self.device)

        # Apply rotation and translation
        rotated_corners = (corners @ rotation.T) + torch.tensor([x, y], device=self.device)

        # Convert to tuple for Pygame
        points = rotated_corners.tolist()
        pygame.draw.polygon(self.screen, (0, 0, 255), points)


def relative_pose_tf(pose1, pose2, device='cpu'):
    """
    Calculate the relative pose of pose1 with respect to pose2 in the local frame using PyTorch.

    Parameters:
    - pose1: Tensor of shape (7,) [x, y, z, qx, qy, qz, qw]
    - pose2: Tensor of shape (7,) [x, y, z, qx, qy, qz, qw]

    Returns:
    - local_pose_diff: Tensor of shape (2,) representing the position difference in the local frame.
    """
    # Ensure pose1 and pose2 are tensors
    if not isinstance(pose1, torch.Tensor):
        pose1 = torch.tensor(pose1, dtype=torch.float32, device=device)
    if not isinstance(pose2, torch.Tensor):
        pose2 = torch.tensor(pose2, dtype=torch.float32, device=device)

    # Extract position and quaternion
    dx, dy = pose1[:2] - pose2[:2]

    # Convert quaternion to yaw
    qx, qy, qz, qw = pose2[3], pose2[4], pose2[5], pose2[6]
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)

    # Ensure arguments to atan2 are tensors
    siny_cosp = torch.tensor(siny_cosp, dtype=torch.float32, device=device)
    cosy_cosp = torch.tensor(cosy_cosp, dtype=torch.float32, device=device)

    yaw = torch.atan2(siny_cosp, cosy_cosp)  # Extract yaw from quaternion

    # Rotation matrix for the inverse rotation (local frame transformation)
    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)
    rotation_matrix = torch.tensor([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ], dtype=torch.float32, device=device)

    # Transform dx, dy into the local frame
    local_pose_diff = torch.matmul(rotation_matrix, torch.tensor([dx, dy], dtype=torch.float32, device=device))

    return local_pose_diff




# Testing the Environment
# if __name__ == "__main__":
#     env = MATH_USV_V1(render_mode="human")

#     obs, _ = env.reset()
#     print("Initial Observation:", obs)

#     for _ in range(200):
#         action = env.action_space.sample()  # Random action
#         obs, reward, done, _, _ = env.step(action)
#         print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")

#         if done:
#             print("USV out of bounds. Resetting environment...")
#             env.reset()

#     env.close()
