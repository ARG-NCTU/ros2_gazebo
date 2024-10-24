import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Bool
from stable_baselines3 import PPO, TD3, SAC
from sensor_msgs.msg import Imu
import numpy as np
from scipy.spatial.transform import Rotation as R
import os, sys
import torch

sys.path.append("/home/arg/ros2_gazebo/ros2_ws/install/veh_model/share/veh_model/models/wamv_v1/")
from multiInput_featureExtractor import CustomFeatureExtractor

class SB3_DP(Node):
    def __init__(self):
        super().__init__('sb3_dp', namespace='/wamv_v1')
        self.declare_parameter('veh', 'wamv_v1')
        self.veh = self.get_parameter('veh').get_parameter_value().string_value

        self.obs = {
            'goal': Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            'auto': False,
            'lidar': np.full((4, 241), 10),
            # 'lidar': np.zeros((4, 241)),
            # 'goal_diff': None,
            # 'velocity': None,
            'goal_diff': np.zeros((10, 3)),
            'velocity': np.zeros((10, 1)),
            'last_time': None,
            'last_pose': None,
        }

        self.sub_goal = self.create_subscription(
            PoseStamped,
            f'/{self.veh}/goal_pose',
            self.__goal_callback,
            10)
        self.sub_pose = self.create_subscription(
            PoseArray,
            f'/model/{self.veh}/pose',
            self.__pose_callback,
            10)
        self.sub_auto = self.create_subscription(
            Bool,
            f'/{self.veh}/auto',
            self.__auto_callback,
            10)

        self.cmd_vel_puber = self.create_publisher(
            TwistStamped,
            f'/{self.veh}/cmd_vel',
            1)

        self.get_logger().info("Loading model...")
        device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.model = PPO.load(
            f"/home/arg/ros2_gazebo/ros2_ws/install/veh_model/share/veh_model/models/wamv_v1/vrx_ppo_2024-10-23_500000_steps.zip",
            device=device,
        )
        self.get_logger().info("Model loaded.")

        # Create a timer for the control loop
        self.timer = self.create_timer(0.02, self.control_loop)

    def control_loop(self):
        if self.obs['auto'] is True:
            # self.get_logger().info("Auto mode is on.")
            while self.obs['goal'] is None:
                self.get_logger().warning("Goal pose is None")
            while self.obs['goal_diff'] is None:
                self.get_logger().warning("Goal difference is None")            
            obs = {
                'laser': self.obs['lidar'],
                'track': self.obs['goal_diff'].reshape(-1),
                'vel': self.obs['velocity'].reshape(-1),
            }
            # self.get_logger().info(f"Observation: {obs}")
            action, _ = self.model.predict(obs)
            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], -1, 1)
            # action = self.__remap_action(action)
            cmd_vel = TwistStamped()
            cmd_vel.header.stamp = self.get_clock().now().to_msg()
            cmd_vel.header.frame_id = 'base_link'
            cmd_vel.twist.linear.x = float(action[0])
            cmd_vel.twist.angular.z = float(action[1])
            self.cmd_vel_puber.publish(cmd_vel)
            self.get_logger().info(f"Action: [{action[0]:.4f}, {action[1]:.4f}]")

    def __goal_callback(self, msg):
        # self.get_logger().info("Received goal pose")
        self.obs['goal'] = msg.pose

    def __pose_callback(self, msg):
        try:
            if not msg.poses:
                self.get_logger().warning("Received empty PoseArray")
                return

            pose = msg.poses[0]

            if self.obs['goal'] is None:
                self.get_logger().warning("Goal pose is None")
                return

            # Extract orientations and positions
            q_goal = np.array([
                self.obs['goal'].orientation.x,
                self.obs['goal'].orientation.y,
                self.obs['goal'].orientation.z,
                self.obs['goal'].orientation.w
            ])
            _, _, yaw_goal = R.from_quat(q_goal).as_euler('xyz')

            q = np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])
            _, _, yaw = R.from_quat(q).as_euler('xyz')
            angle = yaw_goal - yaw
            if angle >= np.pi:
                angle -= 2 * np.pi
            elif angle <= -np.pi:
                angle += 2 * np.pi

            goal_x_prime, goal_y_prime = self.map_to_model_frame(
                np.array([self.obs['goal'].position.x, self.obs['goal'].position.y]),
                np.array([pose.position.x, pose.position.y]),
                yaw
            )
            # goal_x_prime = self.obs['goal'].position.x - pose.position.x
            # goal_y_prime = self.obs['goal'].position.y - pose.position.y
            pos_diff = np.array([goal_x_prime, goal_y_prime, angle])

            current_time = self.get_clock().now()
            vel = 0.0
            if self.obs['last_time'] is None or self.obs['last_pose'] is None:
                self.obs['last_time'] = current_time
                self.obs['last_pose'] = pose
            else:
                dt = (current_time - self.obs['last_time']).nanoseconds / 1e9
                self.obs['last_time'] = current_time
                distance = np.sqrt(
                    (pose.position.x - self.obs['last_pose'].position.x) ** 2 +
                    (pose.position.y - self.obs['last_pose'].position.y) ** 2
                )
                vel = distance / dt if dt > 0 else 0.0
                # self.obs['velocity'][0] = vel
                self.obs['last_pose'] = pose

            self.obs['goal_diff'] = np.roll(self.obs['goal_diff'], -1, axis=0)
            self.obs['goal_diff'][-1] = pos_diff
            self.obs['velocity'] = np.roll(self.obs['velocity'], -1, axis=0)
            self.obs['velocity'][-1] = vel

        except Exception as e:
            self.get_logger().error(f"Error in pose callback: {e}")

    def __auto_callback(self, msg):
        # self.get_logger().info(f"Auto mode: {msg.data}")
        self.obs['auto'] = msg.data

    def __remap_action(self, action):
        # Remap linear action[0]: [0~1] to [-1~1]
        # Keep angular action[1]: [-1~1]
        action[0] = 2 * action[0] - 1
        return action

    def map_to_model_frame(self, map_goal_pose, map_robot_pose, map_robot_orientation):
        # Get relative position
        x, y = map_goal_pose - map_robot_pose
        # Get rotation matrix
        rotation_matrix = R.from_euler('z', -map_robot_orientation).as_matrix()[:2, :2]
        model_point = np.dot(rotation_matrix, np.array([x, y]))
        x_prime, y_prime = model_point
        return x_prime, y_prime

def main(args=None):
    rclpy.init(args=args)
    node = SB3_DP()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
