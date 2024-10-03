import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import ControlWorld

class ResetWorldClient(Node):
    def __init__(self):
        super().__init__('reset_world_client')
        self.client = self.create_client(ControlWorld, '/world/blueboat_waves/control')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for world control service...')
        self.send_request()

    def send_request(self):
        request = ControlWorld.Request()
        request.world_control.reset.all = True  # Reset the entire world
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('World reset successfully')
        else:
            self.get_logger().error('Failed to reset world')

def main(args=None):
    rclpy.init(args=args)
    reset_world_client = ResetWorldClient()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
