#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <ignition/msgs/entity_factory.pb.h>
#include <ignition/transport/Node.hh>
#include <ignition/math/Pose3.hh>

class SpawnEntityNode : public rclcpp::Node
{
public:
  SpawnEntityNode() : Node("spawn_entity")
  {
    RCLCPP_INFO(this->get_logger(), "Spawning entity...");
    ignition::transport::Node ign_node;
    ignition::msgs::EntityFactory req;
    req.set_sdf_filename("box.sdf");
    ignition::math::Pose3d pose(0, 0, 0, 0, 0, 0);
    ignition::msgs::Set(req.mutable_pose(), pose);
    ign_node.Request("/world/default/create", req,
      [this](const ignition::msgs::Boolean &rep, const bool result) {
        if (result && rep.data())
        {
          RCLCPP_INFO(this->get_logger(), "Entity spawned successfully.");
        }
        else
        {
          RCLCPP_ERROR(this->get_logger(), "Failed to spawn entity.");
        }
      });
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SpawnEntityNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}