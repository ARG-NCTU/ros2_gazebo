#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <gz/msgs.hh>
#include <gz/math/Pose3.hh>
#include "gz/msgs/pose.pb.h"
#include <gz/transport/Node.hh>
#include <string>

class MoveEntityNode : public rclcpp::Node
{
public:
  MoveEntityNode() : Node("move_entity")
  {
    this->declare_parameter<std::string>("world", "waves");
    this->declare_parameter<std::string>("entity_name", "blueboat");
    this->declare_parameter<double>("x", 1.0);
    this->declare_parameter<double>("y", 1.0);
    this->declare_parameter<double>("z", 1.0);
    this->declare_parameter<double>("roll", 0.0);
    this->declare_parameter<double>("pitch", 0.0);
    this->declare_parameter<double>("yaw", 0.0);
    this->get_parameter("world", world_name);
    this->get_parameter("entity_name", entity_name);
    this->get_parameter("x", x);
    this->get_parameter("y", y);
    this->get_parameter("z", z);
    this->get_parameter("roll", roll);
    this->get_parameter("pitch", pitch);
    this->get_parameter("yaw", yaw);
    RCLCPP_INFO(this->get_logger(), "Moving entity...");
    gz::transport::Node node;
    gz::msgs::Pose req;
    req.set_name(entity_name);
    gz::math::Pose3d pose(x, y, z, roll, pitch, yaw);
    gz::msgs::Set( &req, pose);
    // gz::msgs::Set(req.mutable_orientation(), pose.Rot());
    // *(req.mutable_position()) = pose.Pos();
    // *(req.mutable_orientation()) = pose.Rot();
    std::string service{"/world/" + world_name + "/set_pose"};

    gz::msgs::Boolean rep;
    bool result;
    unsigned int timeout = 5000;
    bool executed = node.Request(service , req, timeout, rep, result);

    if(executed){
      if(result && rep.data()){
        // std::cout << result << std::endl << rep.data() << std::endl;
        RCLCPP_INFO(this->get_logger(), "Request to move entity sent successfully.");
        schedule_shutdown();
      }else{
        RCLCPP_ERROR(this->get_logger(), "Failed to send request.");
      }
    }else{
      RCLCPP_ERROR(this->get_logger(), "Request to move entity from service [%s] timed out.", service.c_str());
    }
  }
private:
  std::string world_name;
  std::string entity_name;
  double x, y, z, roll, pitch, yaw;
  rclcpp::TimerBase::SharedPtr shutdown_timer_;

  void schedule_shutdown()
  {
    // Schedule a timer to shut down after 1 second
    shutdown_timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      []() {
        RCLCPP_INFO(rclcpp::get_logger("move_entity"), "Shutting down the node...");
        rclcpp::shutdown();
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