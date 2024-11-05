import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64

class ThrustCalculatorNode(Node):
    def __init__(self):
        super().__init__('thrust_calculator')

        # Declare and get parameters
        self.declare_parameter('name', 'default')
        self.declare_parameter('max_thrust', 10.0)

        self.name = self.get_parameter('name').get_parameter_value().string_value
        self.max_thrust = self.get_parameter('max_thrust').get_parameter_value().double_value

        # Subscriber to cmd_vel
        self.subscription = self.create_subscription(
            TwistStamped,
            f'/model/{self.name}/thrust_calculator/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Publishers for port and starboard motor thrusts
        self.port_thrust_publisher = self.create_publisher(
            Float64, f'/model/{self.name}/joint/motor_port_joint/cmd_thrust', 1)
        self.stbd_thrust_publisher = self.create_publisher(
            Float64, f'/model/{self.name}/joint/motor_stbd_joint/cmd_thrust', 1)

    def cmd_vel_callback(self, msg):
        # Get the normalized linear and angular commands
        v_x = msg.twist.linear.x  # normalized between -1 and 1
        omega_z = msg.twist.angular.z  # normalized between -1 and 1

        # Compute the port and starboard command values
        v_P = (v_x - omega_z) / 2
        v_S = (v_x + omega_z) / 2

        # Clamp the command values to be between -1 and 1
        v_P = max(min(v_P, 1.0), -1.0)
        v_S = max(min(v_S, 1.0), -1.0)

        # Compute the thrust commands
        T_P = v_P * self.max_thrust
        T_S = v_S * self.max_thrust

        # Publish the thrust values
        port_thrust_msg = Float64()
        port_thrust_msg.data = T_P
        self.port_thrust_publisher.publish(port_thrust_msg)

        stbd_thrust_msg = Float64()
        stbd_thrust_msg.data = T_S
        self.stbd_thrust_publisher.publish(stbd_thrust_msg)

        # Log the results for debugging
        # self.get_logger().info(
        #     f'Port Thrust: {T_P:.4f} N, Starboard Thrust: {T_S:.4f} N')

def main(args=None):
    rclpy.init(args=args)
    node = ThrustCalculatorNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
