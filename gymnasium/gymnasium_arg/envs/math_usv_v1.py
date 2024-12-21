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
            'imu': (hist_frame, 8),
            'action': (hist_frame, 4),
            'cmd_vel': (3,),
        }

        # Initialize cmd_vel and refer_pose as tensors
        self.cmd_vel = torch.zeros(3, device=self.device)
        self.refer_pose = torch.zeros(3, device=self.device)

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
        # Initialize state
        self.state = None  # [x, y, theta, vx, vy, omega]
        self.imu_data = None  # [orientation (4,), acceleration (3,), angular_velocity (3,)]

        # World bounds
        self.world_size = np.array([800.0, 600.0])  # Width, Height
        self.refer_pose[0] = self.world_size[0] / 2
        self.refer_pose[1] = self.world_size[1] / 2
        self.refer_pose[2] = (torch.rand(1, device=self.device) * 2 - 1) * np.pi
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
        self.dp_cnt = 0
        # Reset state: [x, y, theta, vx, vy, omega]
        d_pose = torch.rand(3, device=self.device)*2-1
        d_pose[1] = torch.sqrt(1-d_pose[0]**2)
        d_pose[2] = d_pose[2] * np.pi
        d_pose[:2] = d_pose[:2] * 0.5
        self.state = torch.tensor(
            [self.world_size[0]/2, self.world_size[1]/2, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )  # Start in the center
        self.state[:3] += d_pose

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
        self.refer_pose[2] = (torch.rand(1, device=self.device) * 2 - 1) * np.pi
        # self.refer_pose = torch.tensor([self.state[0], self.state[1], self.state[2]], device=self.device)
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
        base_action = self._calculate_motor_control(self.cmd_vel[0], self.cmd_vel[1])
        # Extract action components
        thrust_array = torch.clamp(base_action[:2]+2*torch.tensor(action[:2], device=self.device), -1.0, 1.0)
        ang_array = torch.clamp(base_action[2:]+torch.tensor(action[2:], device=self.device)*torch.pi/2, -torch.pi/4, torch.pi/4)
        mag_left, mag_right = thrust_array[0], thrust_array[1]
        angle_left, angle_right = torch.sin(ang_array[2]), torch.sin(ang_array[3])

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
        k1 = 50 # Reward weight for navigating to center
        k2 = 50 # Reward weight for maintaining heading
        k3 = 20 # Reward weight for smooth action
        k4 = 20 # Reward weight for reserving energy

        # Reward of navigating to center
        # rew1 = k1*torch.exp(-(torch.norm(self.refer_pose[:2]-self.veh_obs['pose'][0][:2], p=2)**2/0.25))
        rew1 = k1*(1-(torch.norm(self.refer_pose[:2]-self.veh_obs['pose'][0][:2], p=2)))

        # Reward of maintaining heading
        veh_quat = self.veh_obs['pose'][0][3:7]
        ref_yaw = self.refer_pose[2]
        rew2 = k2*torch.cos(ref_yaw - self.yaw_from_quaternion(veh_quat))
        # Reward of smooth action
        rew3 = -k3 * torch.norm(self.veh_obs['action'][0] - self.veh_obs['action'][1], p=1) / self.__action_shape[0]

        # Reward of reserving energy
        rew4 = -k4 * torch.norm(self.veh_obs['action'][0][:2], p=1) / 2
        # Constraint
        const = []

        # Constraint of thrust
        cmd_vel_norm = torch.norm(self.cmd_vel[:2], p=2)
        action_tensor = torch.tensor(self.action[:2], device=self.device, dtype=torch.float32)
        cons1 = torch.sum(torch.abs((torch.abs(action_tensor) - cmd_vel_norm)))/2

        const.append(cons1.item())

        # Normalize rewards
        rew1 = rew1 / self.info['max_rew']
        rew2 = rew2 / self.info['max_rew']
        rew3 = rew3 / self.info['max_rew']
        rew4 = rew4 / self.info['max_rew']
        # Convert rewards to scalars for logging and further processing
        rew1_value = rew1.item()
        rew2_value = rew2.item()
        rew3_value = rew3.item()
        rew4_value = rew4.item()
        rew = rew1_value + rew2_value + rew3_value + rew4_value

        # Output formatting
        sgn_bool = lambda x: True if x >= 0 else False
        output = "\rstep:{:4d}, cmd: [x:{}, y:{}, yaw:{}], action: [l_t:{}, r_t:{}, l_a:{}, r_a:{}], rews: [{}, {}, {}, {}] const:[{}]".format(
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
            " {:4.2f}".format(rew4_value) if sgn_bool(rew4_value) else "{:4.2f}".format(rew4_value),
            " {:4.2f}".format(const[0]) if sgn_bool(const[0]) else "{:4.2f}".format(const[0]),
        )
        sys.stdout.write(output)
        sys.stdout.flush()
        # Update for termination
        
        info['constraint_costs'] = np.array(const, dtype=np.float32)

        if self.info['step_cnt'] >= self.info['maxstep']:
            truncation = True

        if self.render_mode == "human":
            self._render_pygame()

        if rew1.item() <= -0.5:
            termination = True

        if rew >=0.9:
            if self.dp_cnt >= 100:
                termination = True
            else:
                self.dp_cnt += 1
        else:
            self.dp_cnt = 0

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

        imu_obs = self.veh_obs['imu'].cpu().numpy()   # Convert to NumPy
        r, p, y = R.from_quat(imu_obs[:, :4]).as_euler('xyz', degrees=False).T
        yaw = self.refer_pose[2].item()-y[0]
        # yaw = {-np.pi~np.pi}
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        imu_obs = np.hstack((np.array([r]).T, np.array([p]).T, imu_obs[:, 4:])) 
        imu_obs = imu_obs.flatten()
        action_obs = self.veh_obs['action'].cpu().numpy().flatten()  # Convert to NumPy


        goal_pose = torch.concat([self.refer_pose[:2], torch.tensor([0.0], device=self.device), self.quat_from_angle_z(self.refer_pose[2])], dim=0)
        goal_diff = self.relative_pose_tf(goal_pose, self.veh_obs['pose'][0])
        ang_goal_diff = torch.atan2(goal_diff[1], goal_diff[0])
        norm_goal_diff = torch.norm(goal_diff, p=2)
        self.cmd_vel = torch.tensor([torch.cos(ang_goal_diff), torch.sin(ang_goal_diff), yaw], device=self.device, dtype=torch.float32)
        if norm_goal_diff < 1:
            self.cmd_vel[:2] = self.cmd_vel[:2]*norm_goal_diff 

        # Convert all parts to NumPy-compatible formats before concatenation
        obs = np.concatenate([
            imu_obs,
            action_obs,
            self.cmd_vel.cpu().numpy(),  # Convert cmd_vel to NumPy
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


    def _calculate_motor_control(self, x, y):
        """
        Calculate thrust and yaw angles for left and right motors to achieve the desired velocity (x, y),
        with normalized inputs and outputs.

        Parameters:
            x (float): Desired velocity in x-direction (normalized such that norm2(x, y) <= 1).
            y (float): Desired velocity in y-direction (normalized such that norm2(x, y) <= 1).

        Returns:
            tuple: (T_L, theta_L, T_R, theta_R)
                T_L, T_R: Normalized thrust for left and right motors.
                theta_L, theta_R: Yaw angles for left and right motors (in degrees).
        """
        # Motor positions relative to the center of mass
        x_L = -self.veh_body['length'] / 2
        x_R = -self.veh_body['length'] / 2
        y_L = self.veh_body['width'] / 2
        y_R = -y_L

        # Convert inputs to PyTorch tensors
        F_x = torch.tensor(x, device=self.device, dtype=torch.float32)
        F_y = torch.tensor(y, device=self.device, dtype=torch.float32)

        # Initialize results
        T_L = torch.tensor(0.0, device=self.device)
        T_R = torch.tensor(0.0, device=self.device)
        theta_L = torch.tensor(0.0, device=self.device)
        theta_R = torch.tensor(0.0, device=self.device)

        # If there's no force to generate, return zero thrust and angles
        if F_x == 0 and F_y == 0:
            return torch.tensor([T_L.item(), T_R.item(), theta_L.item(), theta_R.item()], device=self.device)

        # Calculate the required yaw angles
        theta_L = torch.atan2(F_y, F_x)
        theta_R = -theta_L  # Opposite direction for lateral force balancing

        # Ensure angles are within the physical limits of [-45, 45] degrees
        angle_limit = torch.tensor(torch.pi / 4, device=self.device)
        theta_L = torch.clamp(theta_L, -angle_limit, angle_limit)
        theta_R = torch.clamp(theta_R, -angle_limit, angle_limit)

        # Calculate the thrust values
        denominator = (y_R * x_L - y_L * x_R)
        if denominator == 0:
            raise ValueError("Motor positions cannot produce torque balance.")

        T_L = (F_y * x_R - F_x * y_R) / denominator
        T_R = (F_x * y_L - F_y * x_L) / denominator

        # Normalize thrust values based on the input vector norm
        input_norm = torch.linalg.norm(torch.tensor([F_x, F_y], device=self.device), ord=2)
        if input_norm > 1:
            raise ValueError("Input velocities must be normalized such that norm2(x, y) <= 1.")

        max_T = torch.max(torch.abs(T_L), torch.abs(T_R))
        if max_T > 0:
            T_L = (T_L / max_T) * input_norm
            T_R = (T_R / max_T) * input_norm

        return torch.tensor([T_L.item(), T_R.item(), theta_L.item(), theta_R.item()], device=self.device)

    
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