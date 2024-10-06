#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <ignition/msgs/pose.pb.h>
#include <ignition/transport/Node.hh>
#include <ignition/math/Pose3.hh>

class MoveEntityNode : public rclcpp::Node
{
public:
  MoveEntityNode() : Node("move_entity")
  {
    RCLCPP_INFO(this->get_logger(), "Moving entity...");
    ignition::transport::Node ign_node;
    ignition::msgs::Pose req;
    req.set_name("box");
    ignition::math::Pose3d pose(1, 1, 0, 0, 0, 0);
    ignition::msgs::Set(req.mutable_position(), pose.Pos());
    ignition::msgs::Set(req.mutable_orientation(), pose.Rot());
    ign_node.Request("/world/default/set_pose", req,
      [this](const ignition::msgs::Boolean &rep, const bool result) {
        if (result && rep.data())
        {
          RCLCPP_INFO(this->get_logger(), "Entity moved successfully.");
        }
        else
        {
          RCLCPP_ERROR(this->get_logger(), "Failed to move entity.");
        }
      });
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveEntityNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}