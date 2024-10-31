import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped

class JoyToCmdVel(Node):
    def __init__(self):
        super().__init__('joy_to_cmd_vel')
        self.subscription = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.puber = {
            'cmd_vel': self.create_publisher(TwistStamped, '/model/blueboat/thrust_calculator/cmd_vel', 10),
            'auto': self.create_publisher(Bool, '/model/blueboat/auto', 10),
        }
        self.auto = False
        self.get_logger().info('Joy to cmd_vel node has started.')

    def joy_callback(self, joy_msg):
        self.auto = True if joy_msg.buttons[7] == 1 else self.auto
        self.auto = False if joy_msg.buttons[6] == 1 else self.auto
        if self.auto is not True:
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = 'base_link'
            twist.twist.linear.x = joy_msg.axes[1]  # forward/backward
            twist.twist.angular.z = joy_msg.axes[3]  # left/right
            # 6 back, 7 start
            self.puber['cmd_vel'].publish(twist)
            self.get_logger().info(f'Published cmd_vel: linear.x={twist.twist.linear.x}, angular.z={twist.twist.angular.z}')

    def publish_loop(self):
        while rclpy.ok():
            auto_msg = Bool()
            auto_msg.data = self.auto
            self.puber['auto'].publish(auto_msg)
            # if self.auto is True:
            #     self.get_logger().info(f'Published auto: {auto_msg.data}')
            rclpy.spin_once(self, timeout_sec=0.02)
        
def main(args=None):
    rclpy.init(args=args)
    node = JoyToCmdVel()
    node.publish_loop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
