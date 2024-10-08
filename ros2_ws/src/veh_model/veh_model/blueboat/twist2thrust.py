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

        self.C_T_port = -0.02  # Thrust coefficient for port motor
        self.C_T_stbd = 0.02   # Thrust coefficient for starboard motor
        self.rho = 1025        # Fluid density (kg/m^3)
        self.D = 0.112         # Propeller diameter (meters)
        self.W = 0.6           # Distance between motors (meters)

        # Subscriber to cmd_vel
        self.subscription = self.create_subscription(
            TwistStamped,
            f'/model/{self.name}/thrust_calculator/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Publishers for port and starboard motor thrusts
        self.port_thrust_publisher = self.create_publisher(Float64, f'/model/{self.name}/joint/motor_port_joint/cmd_thrust', 1)
        self.stbd_thrust_publisher = self.create_publisher(Float64, f'/model/{self.name}/joint/motor_stbd_joint/cmd_thrust', 1)

    def cmd_vel_callback(self, msg):
        v_x = msg.twist.linear.x
        omega_z = msg.twist.angular.z

        omega_port = (v_x / self.D) - (omega_z * self.W / 2)
        omega_stbd = (v_x / self.D) + (omega_z * self.W / 2)

        T_P = self.C_T_port * self.rho * (self.D**4) * (omega_port**2)
        T_S = self.C_T_stbd * self.rho * (self.D**4) * (omega_stbd**2)

        port_thrust_msg = Float64()
        port_thrust_msg.data = T_P * self.max_thrust
        self.port_thrust_publisher.publish(port_thrust_msg)

        stbd_thrust_msg = Float64()
        stbd_thrust_msg.data = T_S * self.max_thrust
        self.stbd_thrust_publisher.publish(stbd_thrust_msg)

        self.get_logger().info(f'Port Thrust: {T_P * self.max_thrust:.4f} N, Starboard Thrust: {T_S * self.max_thrust:.4f} N')

def main(args=None):
    rclpy.init(args=args)
    node = ThrustCalculatorNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
