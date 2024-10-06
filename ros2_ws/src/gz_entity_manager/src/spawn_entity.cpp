#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <gz/msgs/entity_factory.pb.h>
#include <gz/msgs/stringmsg_v.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/transport/Node.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Pose3.hh>
#include <sstream>
#include <string>

class SpawnEntityNode : public rclcpp::Node
{
public:
  SpawnEntityNode() : Node("spawn_entity")
  {
    this->declare_parameter<std::string>("world", "");
    this->declare_parameter<std::string>("sdf_string", "");
    this->declare_parameter<std::string>("entity_name", "");
    this->declare_parameter<double>("x", 0.0);
    this->declare_parameter<double>("y", 0.0);
    this->declare_parameter<double>("z", 0.0);
    this->declare_parameter<double>("roll", 0.0);
    this->declare_parameter<double>("pitch", 0.0);
    this->declare_parameter<double>("yaw", 0.0);

    spawn_entity();
  }

private:
  void spawn_entity()
  {
    std::string world_name;
    this->get_parameter("world", world_name);
    if (world_name.empty())
    {
      RCLCPP_ERROR(this->get_logger(), "World name is required.");
      return;
    }

    gz::transport::Node node;
    std::string service{"/world/" + world_name + "/create"};

    gz::msgs::EntityFactory req;
    std::string sdf_string;
    this->get_parameter("sdf_string", sdf_string);
    req.set_sdf(sdf_string);

    std::string entity_name;
    this->get_parameter("entity_name", entity_name);
    if (!entity_name.empty())
    {
      req.set_name(entity_name);
    }

    double x, y, z, roll, pitch, yaw;
    this->get_parameter("x", x);
    this->get_parameter("y", y);
    this->get_parameter("z", z);
    this->get_parameter("roll", roll);
    this->get_parameter("pitch", pitch);
    this->get_parameter("yaw", yaw);

    // Set position
    gz::msgs::Pose *pose_msg = req.mutable_pose();
    pose_msg->mutable_position()->set_x(x);
    pose_msg->mutable_position()->set_y(y);
    pose_msg->mutable_position()->set_z(z);

    // Set orientation using quaternion
    gz::math::Quaterniond quat(roll, pitch, yaw);
    pose_msg->mutable_orientation()->set_x(quat.X());
    pose_msg->mutable_orientation()->set_y(quat.Y());
    pose_msg->mutable_orientation()->set_z(quat.Z());
    pose_msg->mutable_orientation()->set_w(quat.W());

    gz::msgs::Boolean rep;
    bool result;
    unsigned int timeout = 5000;
    bool executed = node.Request(service, req, timeout, rep, result);

    if (executed)
    {
      if (result && rep.data())
      {
        RCLCPP_INFO(this->get_logger(), "Requested creation of entity.");
        RCLCPP_INFO(this->get_logger(), "Entity successfully created. Scheduling node shutdown...");
        schedule_shutdown();  // Schedule the shutdown
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Failed request to create entity.");
      }
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Request to create entity from service [%s] timed out.", service.c_str());
    }
  }

  void schedule_shutdown()
  {
    // Schedule a timer to shut down after 1 second
    shutdown_timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      []() {
        RCLCPP_INFO(rclcpp::get_logger("spawn_entity"), "Shutting down the node...");
        rclcpp::shutdown();
      });
  }

  rclcpp::TimerBase::SharedPtr shutdown_timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SpawnEntityNode>();
  rclcpp::spin(node);
  return 0;
}
