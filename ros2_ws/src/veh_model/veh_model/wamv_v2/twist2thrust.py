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
        left_msg.data = msg.twist.linear.x * self.max_thrust
        right_msg.data = msg.twist.linear.x * self.max_thrust
        dir = -1 if msg.twist.linear.x > 0 else 1
        dir = dir * np.pi / 4
        left_pose_msg.data = msg.twist.angular.z * dir
        right_pose_msg.data = msg.twist.angular.z * dir
        
        self.left_thrust_publisher.publish(left_msg)
        # self.left_front_thrust_publisher.publish(left_front_msg)
        self.right_thrust_publisher.publish(right_msg)
        # self.right_front_thrust_publisher.publish(right_front_msg)
        self.left_thrust_pose_publisher.publish(left_pose_msg)
        self.right_thrust_pose_publisher.publish(right_pose_msg)

        # self.get_logger().info(
        #     f"left thrust: {left_msg.data:.4f} rad/s, right thrust: {right_msg.data:.4f} rad/s, left front thrust: {left_front_msg.data:.4f} rad/s right front thrust: {right_front_msg.data:.4f} rad/s"
        # )

def main(args=None):
    rclpy.init(args=args)
    node = ThrustCalculatorNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()