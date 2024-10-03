import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

class ThrustCalculatorNode(Node):
    def __init__(self):
        super().__init__('thrust_calculator')
        
        # Parameters for the thrust calculation
        self.C_T_port = -0.02  # Thrust coefficient for port motor
        self.C_T_stbd = 0.02   # Thrust coefficient for starboard motor
        self.rho = 1025        # Fluid density (kg/m^3)
        self.D = 0.112         # Propeller diameter (meters)
        self.W = 0.6           # Distance between motors (meters)
        
        # Subscriber to cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        
        # Publishers for port and starboard motor thrusts
        self.port_thrust_publisher = self.create_publisher(Float64, 'port_motor/thrust', 10)
        self.stbd_thrust_publisher = self.create_publisher(Float64, 'stbd_motor/thrust', 10)

    def cmd_vel_callback(self, msg):
        # Get the linear and angular velocity from the cmd_vel message
        v_x = msg.linear.x
        omega_z = msg.angular.z

        # Calculate angular velocities for port and starboard motors
        omega_port = (v_x / self.D) - (omega_z * self.W / 2)
        omega_stbd = (v_x / self.D) + (omega_z * self.W / 2)

        # Calculate thrust for port motor
        T_P = self.C_T_port * self.rho * (self.D**4) * (omega_port**2)

        # Calculate thrust for starboard motor
        T_S = self.C_T_stbd * self.rho * (self.D**4) * (omega_stbd**2)

        # Publish the thrust values
        port_thrust_msg = Float64()
        port_thrust_msg.data = T_P
        self.port_thrust_publisher.publish(port_thrust_msg)

        stbd_thrust_msg = Float64()
        stbd_thrust_msg.data = T_S
        self.stbd_thrust_publisher.publish(stbd_thrust_msg)

        # Log the results for debugging
        self.get_logger().info(f'Port Thrust: {T_P:.4f} N, Starboard Thrust: {T_S:.4f} N')

def main(args=None):
    rclpy.init(args=args)
    node = ThrustCalculatorNode()
    rclpy.spin(node)

    # Shutdown ROS 2
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
