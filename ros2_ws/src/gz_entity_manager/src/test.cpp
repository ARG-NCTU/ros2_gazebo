#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <gz/msgs/pose_v.pb.h>
#include <gz/transport/Node.hh>
#include <iostream>

class GzPoseSubscriber : public rclcpp::Node
{
public:
    GzPoseSubscriber() : Node("gz_pose_subscriber")
    {
        subscription_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/waves/pose/info", 10, std::bind(&GzPoseSubscriber::poseCallback, this, std::placeholders::_1)
        );
    }

private:
    void poseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(), "Received PoseArray with %zu poses", msg->poses.size());
        for (const auto &pose : msg->poses) {
            RCLCPP_INFO(this->get_logger(), "Pose: [x: %f, y: %f, z: %f]", 
                        pose.position.x, pose.position.y, pose.position.z);
        }
    }
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
    // Initialize ROS 2
    rclcpp::init(argc, argv);
    
    // Create the subscriber node
    auto node = std::make_shared<GzPoseSubscriber>();

    // Keep the node running
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
