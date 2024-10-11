import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix, Imu
from pymavlink import mavutil
import math
import time

class RoverControllerNode(Node):
    def __init__(self):
        super().__init__('rover_controller')

        self.device_file = 'udp:127.0.0.1:14551'
        self.master = mavutil.mavlink_connection(self.device_file)

        self.get_logger().info("Waiting for vehicle heartbeat...")
        self.master.wait_heartbeat()
        self.get_logger().info("Heartbeat received from system.")

        self.send_arm_command()
        self.check_arm_status()
        self.send_acro_mode_command()

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.twist_callback,
            10)

        self.gps_publisher = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.imu_publisher = self.create_publisher(Imu, '/imu/data', 10)

        self.timer = self.create_timer(0.1, self.publish_sensor_data)  # 10 Hz
        self.heartbeat_timer = self.create_timer(1.0, self.send_heartbeat)  # 1 Hz

    def twist_callback(self, msg):
        """Callback function that receives Twist messages and sends RC override."""
        throttle = self.map_value(msg.linear.x, -1.0, 1.0, 1000, 2000)  # Map -1 to 1 to 1000 to 2000 (Throttle)
        steering = self.map_value(msg.angular.z, -1.0, 1.0, 1000, 2000)  # Map -1 to 1 to 1000 to 2000 (Steering)

        cmd = [int(steering), 0, int(throttle), 0, 0, 0, 0, 0]  # Channel 1: Steering, Channel 3: Throttle
        self.send_rc_override(cmd)

    def send_acro_mode_command(self):
        mode_id = self.master.mode_mapping().get("ACRO")
        if mode_id is None:
            self.get_logger().info("ACRO mode not available.")
            return
        self.master.set_mode(mode_id)
        self.get_logger().info("ACRO mode set!")

    def send_arm_command(self):
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

    def map_value(self, value, input_min, input_max, output_min, output_max):
        """Maps a value from one range to another."""
        return (value - input_min) * (output_max - output_min) / (input_max - input_min) + output_min

    def publish_sensor_data(self):
        """Fetch and publish GPS and IMU data from the vehicle."""
        # Request GPS data
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION, 1, 1)

        # Request IMU data
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 1, 1)

        # Fetch messages (non-blocking)
        msg = self.master.recv_match(type=['GPS_RAW_INT', 'SCALED_IMU'], blocking=False)

        if msg:
            if msg.get_type() == 'GPS_RAW_INT':
                # GPS Data
                gps_msg = NavSatFix()
                gps_msg.latitude = msg.lat * 1e-7  # Convert from micro-degrees
                gps_msg.longitude = msg.lon * 1e-7  # Convert from micro-degrees
                gps_msg.altitude = msg.alt * 1e-3  # Convert from mm to meters
                gps_msg.header.stamp = self.get_clock().now().to_msg()
                gps_msg.header.frame_id = 'gps'
                self.gps_publisher.publish(gps_msg)
                # self.get_logger().info(f"Published GPS: {gps_msg.latitude}, {gps_msg.longitude}, {gps_msg.altitude}")

            elif msg.get_type() == 'SCALED_IMU':
                # IMU Data
                imu_msg = Imu()
                imu_msg.linear_acceleration.x = msg.xacc * 9.81 / 1000  # Convert from mg to m/s^2
                imu_msg.linear_acceleration.y = msg.yacc * 9.81 / 1000  # Convert from mg to m/s^2
                imu_msg.linear_acceleration.z = msg.zacc * 9.81 / 1000  # Convert from mg to m/s^2
                imu_msg.angular_velocity.x = msg.xgyro * (math.pi / 180) / 1000  # Convert from millirad/s to rad/s
                imu_msg.angular_velocity.y = msg.ygyro * (math.pi / 180) / 1000  # Convert from millirad/s to rad/s
                imu_msg.angular_velocity.z = msg.zgyro * (math.pi / 180) / 1000  # Convert from millirad/s to rad/s
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = 'imu'
                self.imu_publisher.publish(imu_msg)
                self.get_logger().info(f"Published IMU: Acc: ({imu_msg.linear_acceleration.x}, {imu_msg.linear_acceleration.y}, {imu_msg.linear_acceleration.z}), Gyro: ({imu_msg.angular_velocity.x}, {imu_msg.angular_velocity.y}, {imu_msg.angular_velocity.z})")

def main(args=None):
    rclpy.init(args=args)
    rover_controller_node = RoverControllerNode()
    rclpy.spin(rover_controller_node)
    rover_controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
