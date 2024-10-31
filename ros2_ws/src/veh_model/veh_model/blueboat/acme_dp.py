import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
import numpy as np
from scipy.spatial.transform import Rotation as R
import os, sys
import tensorflow as tf

class acme_DP(Node):
    def __init__(self):
        super().__init__('acme_dp', namespace='/blueboat')
        self.declare_parameter('veh', 'blueboat')
        self.veh = self.get_parameter('veh').get_parameter_value().string_value

        self.obs = {
            'goal': Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            'auto': False,
            'lidar': np.tile(np.full(241, 10), (4, 1)),
            # 'lidar': np.zeros((4, 241)),
            'goal_diff': None,
            'velocity': None,
            # 'goal_diff': np.zeros((10, 3)),
            # 'velocity': np.zeros((10, 1)),
            'last_time': None,
            'last_pose': None,
        }

        self.sub_goal = self.create_subscription(
            PoseStamped,
            f'/model/{self.veh}/goal_pose',
            self.__goal_callback,
            10)
        self.sub_pose = self.create_subscription(
            PoseArray,
            f'/model/{self.veh}/pose',
            self.__pose_callback,
            10)
        self.sub_auto = self.create_subscription(
            Bool,
            f'/model/{self.veh}/auto',
            self.__auto_callback,
            10)

        self.cmd_vel_puber = self.create_publisher(
            TwistStamped,
            f'/model/{self.veh}/thrust_calculator/cmd_vel',
            1)

        self.get_logger().info("Loading model...")

        # gpu = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpu[0], True)
        self.policy_network = tf.saved_model.load(f"/home/arg/ros2_gazebo/ros2_ws/install/veh_model/share/veh_model/models/{self.veh}/0.15_dp_model/snapshots/policy")

        self.get_logger().info("Model loaded.")

        # Create a timer for the control loop
        self.timer = self.create_timer(0.02, self.control_loop)

    def control_loop(self):
        if self.obs['auto'] is True:
            # self.get_logger().info("Auto mode is on.")
            if self.obs['goal'] is None:
                self.get_logger().warning("Goal pose is None")
                return
            if self.obs['goal_diff'] is None:
                self.get_logger().warning("Goal difference is None")
                return
            obs = np.append(self.obs['lidar'].reshape(-1), self.obs['goal_diff'].reshape(-1))
            obs = np.append(obs, self.obs['velocity'].reshape(-1))
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
            obs = tf.expand_dims(obs, axis=0)
            action = self.policy_network(obs)[0].numpy()
            
            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], -1, 1)

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
            q_goal = np.array([
                self.obs['goal'].orientation.x,
                self.obs['goal'].orientation.y,
                self.obs['goal'].orientation.z,
                self.obs['goal'].orientation.w
            ])
            yaw_goal = R.from_quat(q_goal).as_euler('zyx')[0]

            q = np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])
            yaw = R.from_quat(q).as_euler('zyx')[0]

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
            if self.obs['goal_diff'] is None:
                self.obs['goal_diff'] = np.tile(pos_diff, (10, 1))
            else:
                self.obs['goal_diff'] = np.roll(self.obs['goal_diff'], -1, axis=0)
                self.obs['goal_diff'][-1] = pos_diff

            if self.obs['velocity'] is None:
                self.obs['velocity'] = np.full((10, 1), vel)
            else:
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
    node = acme_DP()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
