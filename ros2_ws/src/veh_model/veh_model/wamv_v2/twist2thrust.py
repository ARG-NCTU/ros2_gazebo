import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
import numpy as np


class ThrustCalculatorNode(Node):
    def __init__(self):
        super().__init__('thrust_calculator')

        # Declare and get parameters
        self.declare_parameter('name', 'wamv_v2')
        self.declare_parameter('max_thrust', 15*746/9.8)

        self.name = self.get_parameter('name').get_parameter_value().string_value
        self.max_thrust = self.get_parameter('max_thrust').get_parameter_value().double_value

        # Subscriber to cmd_vel
        self.subscription = self.create_subscription(
            TwistStamped,
            f'/{self.name}/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Publishers for port and starboard motor thrusts
        self.left_thrust_publisher = self.create_publisher(
            Float64, f'/{self.name}/joint/left/thruster/cmd_thrust', 1)
        # self.left_front_thrust_publisher = self.create_publisher(
        #     Float64, f'/{self.name}/joint/left_front/thruster/cmd_thrust', 1)
        self.right_thrust_publisher = self.create_publisher(
            Float64, f'/{self.name}/joint/right/thruster/cmd_thrust', 1)
        # self.right_front_thrust_publisher = self.create_publisher(
        #     Float64, f'/{self.name}/joint/right_front/thruster/cmd_thrust', 1)
        self.left_thrust_pose_publisher = self.create_publisher(
            Float64, f'/{self.name}/joint/left/thruster/cmd_pos', 1)
        self.right_thrust_pose_publisher = self.create_publisher(
            Float64, f'/{self.name}/joint/right/thruster/cmd_pos', 1)

    def cmd_vel_callback(self, msg):
        left_msg = Float64()
        right_msg = Float64()
        left_pose_msg = Float64()
        right_pose_msg = Float64()
        left_pos = np.array([-2.3, 1.027135])
        right_pos = np.array([-2.3, -1.027135])
        operator = lambda x: 1 if x > 0 else -1
        # Compute the distance between engines (width of the boat)
        engine_distance = np.linalg.norm(left_pos - right_pos)
        thrust = msg.twist.linear.x * self.max_thrust
        differential = msg.twist.angular.z * engine_distance * self.max_thrust / 2

        left_thrust = thrust - differential
        right_thrust = thrust + differential

        angle_sync = 0.5* np.arctan2(abs(differential), abs(thrust)) * operator(differential)
        

        left_msg.data = left_thrust
        right_msg.data = right_thrust
        left_pose_msg.data = angle_sync
        right_pose_msg.data = angle_sync

        
        self.left_thrust_publisher.publish(left_msg)
        # self.left_front_thrust_publisher.publish(left_front_msg)
        self.right_thrust_publisher.publish(right_msg)
        # self.right_front_thrust_publisher.publish(right_front_msg)
        self.left_thrust_pose_publisher.publish(left_pose_msg)
        self.right_thrust_pose_publisher.publish(right_pose_msg)

        # self.get_logger().info(
        #     f"left thrust: {left_msg.data:.4f} rad/s, right thrust: {right_msg.data:.4f} rad/s, left ang thrust: {left_pose_msg.data:.4f} rad/s right ang thrust: {right_pose_msg.data:.4f} rad/s"
        # )

def main(args=None):
    rclpy.init(args=args)
    node = ThrustCalculatorNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()