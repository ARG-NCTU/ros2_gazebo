#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <ignition/msgs/entity.pb.h>
#include <ignition/transport/Node.hh>

class DeleteEntityNode : public rclcpp::Node
{
public:
  DeleteEntityNode() : Node("delete_entity")
  {
    RCLCPP_INFO(this->get_logger(), "Deleting entity...");
    ignition::transport::Node ign_node;
    ignition::msgs::Entity req;
    req.set_name("box");
    ign_node.Request("/world/default/remove", req,
      [this](const ignition::msgs::Boolean &rep, const bool result) {
        if (result && rep.data())
        {
          RCLCPP_INFO(this->get_logger(), "Entity deleted successfully.");
        }
        else
        {
          RCLCPP_ERROR(this->get_logger(), "Failed to delete entity.");
        }
      });
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DeleteEntityNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}