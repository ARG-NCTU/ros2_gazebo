import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pygame
from scipy.spatial.transform import Rotation as R
import sys, os, random
from scipy.integrate import dblquad


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
            'dt': 1 / 50,
            'max_rew': 100.0,
            'total_step': 0,
            'step_cnt': 0,
        }
        self.veh_obs = {}  # Initialize as empty dict; will be set in reset()

        # Simulation parameters
        self.veh_body = {
            'length': 3.96, # meter
            'width': 2.44, # meter
            'mass': 136.7, # USV mass (kg)
            'drag_coefficient': 0.1, # Drag coefficient
            'moment_of_inertia': 10.0, # Rotational inertia
            'max_thrust': 15 * 746 / 9.8,
            'linear_damping_coefficient': 100.0, # Linear damping coefficient
            'rotational_damping_coefficient': 100.0, # Angular damping coefficient
        }
        # Calculate moment of inertia
        def integrand(y, x):
            return np.sqrt(x**2 + y**2)
        
        result, error = dblquad(integrand, 0, self.veh_body['length'], lambda x: 0, lambda x: self.veh_body['width'])
        self.veh_body['moment_of_inertia'] = result * self.veh_body['mass']/(self.veh_body['length'] * self.veh_body['width'])

        self.__action_shape = (4,)
        self.__obs_shape = {
            'imu': (hist_frame, 10),
            'action': (hist_frame, 4),
            'latent': self.info['latent_dim'],
            'cmd_vel': (3,),
            # 'refer': (3,),
            'refer': (1,),
        }

        # Initialize cmd_vel and refer_pose as tensors
        self.cmd_vel = torch.zeros(3, device=self.device)
        self.refer_pose = torch.zeros(3, device=self.device)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.__action_shape, dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.prod(self.__obs_shape['imu']) + np.prod(self.__obs_shape['action']) + np.prod(self.__obs_shape['cmd_vel']) + np.prod(self.__obs_shape['refer']),),
            dtype=np.float32,
            seed=seed
        )
        # Initialize state
        self.state = None  # [x, y, theta, vx, vy, omega]
        self.imu_data = None  # [orientation (4,), acceleration (3,), angular_velocity (3,)]

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

        # Reset state: [x, y, theta, vx, vy, omega]
        self.state = torch.tensor(
            [self.world_size[0]/2, self.world_size[1]/2, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )  # Start in the center

        # Reset IMU data
        self.imu_data = self._calculate_imu_data()
        self.veh_obs = {
            'imu': self.imu_data.repeat(self.info['hist_frame'], 1).to(self.device),  # Repeat imu_data for history
            'action': torch.zeros((self.info['hist_frame'], 4), device=self.device),  # Use torch for action buffer
        }
        ori = self.veh_obs['imu'][0][:4]
        pose = torch.cat([self.state[:2], torch.tensor([0.0], device=self.device), ori], dim=0)
        self.veh_obs['pose'] = pose.repeat(self.info['hist_frame'], 1).to(self.device)
        self.cmd_vel = torch.zeros(3, device=self.device)
        self.refer_pose = torch.tensor([self.state[0], self.state[1], self.state[2]], device=self.device)
        self.action = torch.zeros(self.__action_shape, device=self.device)
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
            self.state, mag_left, mag_right, angle_left, angle_right, self.info['dt']
        )

        # Update observation buffers using torch.roll
        self.veh_obs['imu'] = torch.roll(self.veh_obs['imu'], shifts=1, dims=0)
        self.veh_obs['imu'][0] = self.imu_data
        self.veh_obs['action'] = torch.roll(self.veh_obs['action'], shifts=1, dims=0)
        self.veh_obs['action'][0] = torch.tensor(action, device=self.device, dtype=torch.float32)

        pose = torch.cat([self.state[:2], torch.tensor([0.0], device=self.device), self.imu_data[:4]], dim=0)
        self.veh_obs['pose'] = torch.roll(self.veh_obs['pose'], shifts=1, dims=0)
        self.veh_obs['pose'][0] = pose

        info = {}
        termination = False
        truncation = False
        obs = self.get_observation()

        ## reward and constraints ##
        k1 = 60
        k2 = 30
        k3 = 10

        # Replace operator and relu with tensor-compatible functions
        operator = lambda x: torch.where(x >= 0, torch.tensor(1.0, device=self.device), torch.tensor(-1.0, device=self.device))
        relu = lambda x: torch.clamp(x, min=0)

        # Reward of following cmd_vel direction
        ref_yaw = self.refer_pose[2]
        refer_ori = self.quat_from_angle_z(ref_yaw)
        refer_ori_tensor = torch.tensor(refer_ori, device=self.device, dtype=torch.float32)
        # refer_pose = torch.cat([
        #     self.refer_pose[:2],
        #     torch.tensor([0.0], device=self.device),
        #     refer_ori_tensor
        # ])
        refer_pose = torch.cat([
            self.veh_obs['pose'][1][:3],
            refer_ori_tensor
        ])

        local_pose_diff = self.relative_pose_tf(self.veh_obs['pose'][0], refer_pose)
        local_pose_norm = torch.norm(local_pose_diff[:2], p=2)
        cmd_dir = torch.atan2(self.cmd_vel[1], self.cmd_vel[0])
        veh_dir = torch.atan2(local_pose_diff[1], local_pose_diff[0])

        cmd_vel_norm = torch.norm(self.cmd_vel[:2], p=2)


        if cmd_vel_norm == 0:
            rew1 = k1*(1 - relu(local_pose_norm))
        else:
            theta = torch.cos(veh_dir - cmd_dir)
            rew1 = k1 * theta - relu(-operator(theta) * local_pose_norm)
        # Reward of thrust
        action_tensor = torch.tensor(self.action[:2], device=self.device, dtype=torch.float32)
        rew2 = 1 - 2 * (torch.abs(action_tensor) - cmd_vel_norm)
        rew2 = k2 * torch.sum(rew2) / 2

        # Reward of smooth action
        rew3 = -k3 * torch.norm(self.veh_obs['action'][0] - self.veh_obs['action'][1], p=1) / self.__action_shape[0]

        # Constraint
        const = []
        veh_quat = self.veh_obs['pose'][0][3:7].cpu().numpy()
        veh_yaw = R.from_quat(veh_quat).as_euler('xyz', degrees=False)[2]
        ref_yaw = self.refer_pose[2].cpu().item()
        const_value = (1 - np.cos(veh_yaw - ref_yaw)) / 2
        const.append(const_value)
        # Normalize rewards
        rew1 = rew1 / self.info['max_rew']
        rew2 = rew2 / self.info['max_rew']
        rew3 = rew3 / self.info['max_rew']
        # Convert rewards to scalars for logging and further processing
        rew1_value = rew1.item()
        rew2_value = rew2.item()
        rew3_value = rew3.item()
        rew = rew1_value + rew2_value + rew3_value

        # Output formatting
        sgn_bool = lambda x: True if x >= 0 else False
        output = "\rstep:{:4d}, cmd: [x:{}, y:{}, yaw:{}], action: [l_t:{}, r_t:{}, l_a:{}, r_a:{}], rews: [{}, {}, {}] const:[{}]".format(
            self.info['step_cnt'],
            " {:4.2f}".format(self.cmd_vel[0].item()) if sgn_bool(self.cmd_vel[0].item()) else "{:4.2f}".format(self.cmd_vel[0].item()),
            " {:4.2f}".format(self.cmd_vel[1].item()) if sgn_bool(self.cmd_vel[1].item()) else "{:4.2f}".format(self.cmd_vel[1].item()),
            " {:4.2f}".format(self.cmd_vel[2].item()) if sgn_bool(self.cmd_vel[2].item()) else "{:4.2f}".format(self.cmd_vel[2].item()),
            " {:4.2f}".format(action[0]) if sgn_bool(action[0]) else "{:4.2f}".format(action[0]),
            " {:4.2f}".format(action[1]) if sgn_bool(action[1]) else "{:4.2f}".format(action[1]),
            " {:4.2f}".format(action[2]) if sgn_bool(action[2]) else "{:4.2f}".format(action[2]),
            " {:4.2f}".format(action[3]) if sgn_bool(action[3]) else "{:4.2f}".format(action[3]),
            " {:4.2f}".format(rew1_value) if sgn_bool(rew1_value) else "{:4.2f}".format(rew1_value),
            " {:4.2f}".format(rew2_value) if sgn_bool(rew2_value) else "{:4.2f}".format(rew2_value),
            " {:4.2f}".format(rew3_value) if sgn_bool(rew3_value) else "{:4.2f}".format(rew3_value),
            " {:4.2f}".format(const[0]) if sgn_bool(const[0]) else "{:4.2f}".format(const[0]),
        )
        sys.stdout.write(output)
        sys.stdout.flush()

        # Update for termination
        if rew <= -1.5:
            termination = True

        # Update cmd_vel and refer_pose every 1024 steps
        if self.info['step_cnt'] % 1024 == 0:
            x = random.uniform(-1, 1)
            y = np.sqrt(1 - x ** 2) * random.uniform(-1, 1)
            self.cmd_vel = torch.tensor([x, y, 0.0], device=self.device, dtype=torch.float32)
            yaw_quat = self.veh_obs['pose'][0][3:7].cpu().numpy()
            yaw = R.from_quat(yaw_quat).as_euler('xyz', degrees=False)[2]
            self.refer_pose = torch.cat([self.veh_obs['pose'][0][:2], torch.tensor([yaw], device=self.device, dtype=torch.float32)])

        info['constraint_costs'] = np.array(const, dtype=np.float32)

        if self.info['step_cnt'] >= self.info['maxstep']:
            truncation = True

        if self.render_mode == "human":
            self._render_pygame()

        # Return values
        return obs, rew, termination, truncation, info


    def get_observation(self):
        """
        Returns the current observation from the environment.
        """
        # Convert state to CPU and NumPy
        veh_pose = self.state.cpu().numpy()[:2]
        veh_pose = np.hstack((veh_pose, 0))
        veh_pose = np.hstack((veh_pose, self.imu_data.cpu().numpy()[:4]))  # Convert IMU data as well

        imu_obs = self.veh_obs['imu'].cpu().numpy().flatten()  # Convert to NumPy
        action_obs = self.veh_obs['action'].cpu().numpy().flatten()  # Convert to NumPy

        ref_yaw = self.refer_pose[2]
        refer_ori = self.quat_from_angle_z(self.refer_pose[2])

        # Construct refer_pose as a tensor
        refer_pose = torch.cat([
            self.refer_pose[:2],
            torch.tensor([0.0], device=self.device, dtype=torch.float32),
            refer_ori
        ])

        # Compute local pose difference
        local_pose_diff = self.relative_pose_tf(self.veh_obs['pose'][0], refer_pose)
        local_pose_diff_np = local_pose_diff.cpu().numpy()  # Convert to NumPy for concatenation

        veh_yaw = R.from_quat([
            veh_pose[3],
            veh_pose[4],
            veh_pose[5],
            veh_pose[6]
        ]).as_euler('xyz', degrees=False)[2]

        # Convert all parts to NumPy-compatible formats before concatenation
        obs = np.concatenate([
            imu_obs,
            action_obs,
            self.cmd_vel.cpu().numpy(),  # Convert cmd_vel to NumPy
            # np.hstack((local_pose_diff_np[:2], np.array([veh_yaw - ref_yaw.cpu().item()]))),  # Ensure NumPy compatibility
            np.array([veh_yaw - ref_yaw.cpu().item()]),  # Ensure NumPy compatibility
        ])
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
        mag_left = torch.tensor(mag_left*self.veh_body['max_thrust'], dtype=torch.float32, device=self.device)
        mag_right = torch.tensor(mag_right*self.veh_body['max_thrust'], dtype=torch.float32, device=self.device)
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
        # **Add damping forces**
        damping_force = -self.veh_body['linear_damping_coefficient'] * torch.tensor([vx, vy], device=self.device)

        # Net force and torque
        net_force = force_left_global + force_right_global + damping_force

        # Torques (cross-product: r x F)
        torque_left = -(left_thruster_pos[0] * force_left_local[1]) + (left_thruster_pos[1] * force_left_local[0])
        torque_right = -(right_thruster_pos[0] * force_right_local[1]) + (right_thruster_pos[1] * force_right_local[0])
        # **Add damping torque**
        damping_torque = -self.veh_body['rotational_damping_coefficient'] * omega

        net_torque = torque_left + torque_right + damping_torque

        # Accelerations
        mass = torch.tensor(self.veh_body['mass'], device=self.device)
        moment_of_inertia = torch.tensor(self.veh_body['moment_of_inertia'], device=self.device)
        ax = net_force[0] / mass
        ay = net_force[1] / mass
        alpha = net_torque / moment_of_inertia

        # Update velocities and positions
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        omega_new = omega + alpha * dt

        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        theta_new = theta + omega_new * dt
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi  # Wrap angle between [-pi, pi]

        # IMU data: [qx, qy, qz, qw, ax, ay, az, wx, wy, wz]
        orientation = self.quat_from_angle_z(theta_new)
        imu_data = torch.tensor([orientation[0], orientation[1], orientation[2], orientation[3], ax, ay, -9.8, 0, 0, omega_new], dtype=torch.float32, device=self.device)
        return torch.tensor([x_new, y_new, theta_new, vx_new, vy_new, omega_new], dtype=torch.float32, device=self.device), imu_data

    def _calculate_imu_data(self):
        """
        Calculates the initial IMU data from the initial state.
        """
        orientation = self.quat_from_angle_z(self.state[2])
        return torch.tensor([orientation[0], orientation[1], orientation[2], orientation[3], 0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

    def _render_pygame(self):
        """
        Renders the environment using Pygame.
        """
        self.screen.fill((255, 255, 255))  # Clear the screen with white

        # Draw USV
        self._draw_usv(self.state[0].item(), self.state[1].item(), self.state[2].item())

        # Update display
        pygame.display.flip()
        self.clock.tick(int(1 / self.info['dt']))

    def _draw_usv(self, x, y, theta):
        """
        Draw the USV on the screen based on its position and orientation.
        """
        length = self.veh_body['length']  # USV length
        width = self.veh_body['width']  # USV width

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
        points = rotated_corners.cpu().numpy().tolist()
        pygame.draw.polygon(self.screen, (0, 0, 255), points)

    def quat_from_angle_z(self, theta):
        """
        Compute quaternion from rotation around z-axis by angle theta.
        """
        half_theta = theta / 2.0
        sin_half_theta = torch.sin(half_theta)
        cos_half_theta = torch.cos(half_theta)
        return torch.tensor([0.0, 0.0, sin_half_theta, cos_half_theta], device=self.device, dtype=torch.float32)

    def yaw_from_quaternion(self, q):
        """
        Extract yaw angle from a quaternion.
        """
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(siny_cosp, cosy_cosp)

    def relative_pose_tf(self, pose1, pose2):
        """
        Calculate the relative pose of pose1 with respect to pose2 in the local frame using PyTorch.

        Parameters:
        - pose1: Tensor of shape (7,) [x, y, z, qx, qy, qz, qw]
        - pose2: Tensor of shape (7,) [x, y, z, qx, qy, qz, qw]

        Returns:
        - local_pose_diff: Tensor of shape (2,) representing the position difference in the local frame.
        """
        # Ensure pose1 and pose2 are tensors of type float32
        pose1 = torch.as_tensor(pose1, dtype=torch.float32, device=self.device)
        pose2 = torch.as_tensor(pose2, dtype=torch.float32, device=self.device)

        # Extract position and quaternion
        dx, dy = pose1[:2] - pose2[:2]

        # Convert quaternion to yaw
        qx, qy, qz, qw = pose2[3], pose2[4], pose2[5], pose2[6]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)

        # Ensure siny_cosp and cosy_cosp are float32
        siny_cosp = siny_cosp.to(dtype=torch.float32)
        cosy_cosp = cosy_cosp.to(dtype=torch.float32)

        yaw = torch.atan2(siny_cosp, cosy_cosp)  # Extract yaw from quaternion

        # Rotation matrix for the inverse rotation (local frame transformation)
        cos_yaw = torch.cos(-yaw)
        sin_yaw = torch.sin(-yaw)

        rotation_matrix = torch.stack([
            torch.stack([cos_yaw, -sin_yaw]),
            torch.stack([sin_yaw, cos_yaw])
        ])

        # Ensure rotation_matrix is float32
        rotation_matrix = rotation_matrix.to(dtype=torch.float32)

        # Transform dx, dy into the local frame
        local_pose_diff = torch.matmul(rotation_matrix, torch.stack([dx, dy]))
        return local_pose_diff


if __name__ == "__main__":
    env = MATH_USV_V1(render_mode="human", device='cuda')
    obs = env.reset()
    done = False
    while not done:
        keyboard_input = pygame.key.get_pressed()
        action = np.array([1.0, 1.0, 0.0, 0.0])
        # action = gym.spaces.Box.sample(env.action_space)
        obs, rew, done, _, _ = env.step(action)
        env.render()
    env.close()