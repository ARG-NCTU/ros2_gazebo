import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pymavlink import mavutil
import time

# ROS2 node that subscribes to /cmd_vel and sends RC override using MAVLink
class RoverControllerNode(Node):
    def __init__(self):
        super().__init__('rover_controller')

        # Establish MAVLink connection to the rover
        self.device_file = 'udp:127.0.0.1:14551'
        self.master = mavutil.mavlink_connection(self.device_file)

        # Wait for heartbeat
        self.get_logger().info("Waiting for the vehicle heartbeat...")
        self.master.wait_heartbeat()
        self.get_logger().info("Heartbeat received from system.")

        # Arm the rover and set to ACRO mode
        self.send_arm_command()
        self.check_arm_status()
        self.send_acro_mode_command()

        # Create subscriber for Twist messages
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.twist_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Create a timer to send heartbeat at a regular interval
        self.timer = self.create_timer(1.0, self.send_heartbeat)

    def twist_callback(self, msg):
        """Callback function that receives Twist messages and sends RC override."""
        # Map Twist linear.x to throttle (Channel 3)
        # Map Twist angular.z to steering (Channel 1)
        throttle = self.map_value(msg.linear.x, -1.0, 1.0, 1000, 2000)  # Map from -1 to 1 to 1000 to 2000 (Throttle)
        steering = self.map_value(msg.angular.z, -1.0, 1.0, 1000, 2000)  # Map from -1 to 1 to 1000 to 2000 (Steering)

        # Create the RC override command
        cmd = [int(steering), 0, int(throttle), 0, 0, 0, 0, 0]  # Channel 1: Steering, Channel 3: Throttle
        self.send_rc_override(cmd)

    def send_acro_mode_command(self):
        """Send a command to set the rover to ACRO mode."""
        mode_id = self.master.mode_mapping().get("ACRO")
        if mode_id is None:
            self.get_logger().info("ACRO mode not available.")
            return
        self.master.set_mode(mode_id)
        self.get_logger().info("ACRO mode set!")

    def send_arm_command(self):
        """Send a command to arm the rover."""
        self.master.mav.command_long_send(self.master.target_system,
                                          self.master.target_component,
                                          mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                          0, 1, 0, 0, 0, 0, 0, 0)
        self.get_logger().info("Arm command sent!")

    def check_arm_status(self):
        """Check if the rover is armed."""
        while True:
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 1, 1
            )
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True)
            arm_state = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if arm_state:
                self.get_logger().info("Rover is armed!")
                break
            else:
                self.get_logger().info("Waiting for arming...")
            time.sleep(1)

    def send_rc_override(self, cmd):
        """Send RC override command with the given channel values."""
        self.master.mav.rc_channels_override_send(
            self.master.target_system, self.master.target_component,
            *cmd  # Unpack the list so that all 8 channels are passed
        )
        self.get_logger().info(f"RC Override sent: {cmd}")

    def send_heartbeat(self):
        """Send a heartbeat to maintain connection with the vehicle."""
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        self.get_logger().info("Heartbeat sent.")

    def map_value(self, value, input_min, input_max, output_min, output_max):
        """Maps a value from one range to another."""
        return (value - input_min) * (output_max - output_min) / (input_max - input_min) + output_min


def main(args=None):
    rclpy.init(args=args)

    rover_controller_node = RoverControllerNode()

    rclpy.spin(rover_controller_node)

    rover_controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
