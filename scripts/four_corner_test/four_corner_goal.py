import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy


class GoalPosePublisher(Node):
    def __init__(self):
        super().__init__('goal_pose_publisher')
        
        # Define the subscriber to the current pose array
        self.pose_subscriber = self.create_subscription(
            PoseArray,
            '/model/wamv_v2/pose',
            self._pose_callback,
            10
        )
        
        # Define the publisher for the goal pose
        self.goal_publisher = self.create_publisher(
            PoseStamped,
            '/wamv_v2/goal_pose',
            10
        )
        
        # Define goal tolerance thresholds
        self.goal_tolerance = 0.5  # distance tolerance
        self.heading_tolerance = 0.1  # heading tolerance in radians
        
        # Initialize a list of goals with positions and headings
        self.goals = [
            {'position': (0.0, 0.0), 'heading': 0.0},
            {'position': (10.0, 0.0), 'heading': 0.5},
            {'position': (10.0, 10.0), 'heading': -0.5},
            {'position': (0.0, 10.0), 'heading': 1.0},
            {'position': (0.0, 0.0), 'heading': -1.0},
            # Add more goals as needed
        ]
        self.goal_index = 0  # Tracks the current goal index
        
        # Set up a timer to keep publishing the goal pose at regular intervals
        self.timer = self.create_timer(0.1, self.publish_goal)  # Publish every second

    def _pose_callback(self, msg):
        if not msg.poses:
            self.get_logger().warn("PoseArray is empty.")
            return
        
        # Assuming the first pose in the PoseArray is the WAM-V’s current pose
        current_pose = msg.poses[0]
        
        # Calculate the distance to the current goal
        goal_position = self.goals[self.goal_index]['position']
        distance = math.sqrt(
            (current_pose.position.x - goal_position[0]) ** 2 +
            (current_pose.position.y - goal_position[1]) ** 2
        )
        
        # Calculate the heading difference
        current_heading = self.get_heading_from_orientation(current_pose.orientation)
        goal_heading = self.goals[self.goal_index]['heading']
        heading_difference = abs(current_heading - goal_heading)
        
        # Check if within tolerance to move to the next goal
        if distance < self.goal_tolerance and heading_difference < self.heading_tolerance:
            self.goal_index += 1  # Move to the next goal
            if self.goal_index < len(self.goals):
                self.get_logger().info("Reached goal, moving to the next one.")
                self.get_logger().info(f"Goal position: {self.goals[self.goal_index]['position']}")
            else:
                self.get_logger().info("All goals reached.")
                self.timer.cancel()  # Stop publishing once all goals are reached

    def publish_goal(self):
        # Check if all goals are reached
        if self.goal_index >= len(self.goals):
            return
        
        # Get the next goal position and heading
        goal = self.goals[self.goal_index]
        goal_position = goal['position']
        goal_heading = goal['heading']
        
        # Create and populate the PoseStamped message
        goal_pose = PoseStamped()
        goal_pose.pose.position.x = goal_position[0]
        goal_pose.pose.position.y = goal_position[1]
        goal_pose.pose.orientation = self.heading_to_orientation(goal_heading)
        
        # Publish the goal
        self.goal_publisher.publish(goal_pose)
        # self.get_logger().info(
        #     f'Published goal at x: {goal_position[0]}, y: {goal_position[1]}, heading: {goal_heading}'
        # )
    
    def get_heading_from_orientation(self, orientation):
        # Convert quaternion to heading in radians
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def heading_to_orientation(self, heading):
        # Convert heading in radians to quaternion
        from geometry_msgs.msg import Quaternion
        orientation = Quaternion()
        orientation.z = math.sin(heading / 2.0)
        orientation.w = math.cos(heading / 2.0)
        return orientation

def main(args=None):
    rclpy.init(args=args)
    node = GoalPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
