#include <rclcpp/rclcpp.hpp>
#include <gz/msgs/entity.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <gz/transport/Node.hh>
#include <string>

class DeleteEntityNode : public rclcpp::Node
{
public:
  DeleteEntityNode() : Node("delete_entity")
  {
    this->declare_parameter<std::string>("world", "");
    this->declare_parameter<std::string>("entity_name", "");

    delete_entity();
  }

private:
  void delete_entity()
  {
    std::string world_name;
    this->get_parameter("world", world_name);
    if (world_name.empty())
    {
      RCLCPP_ERROR(this->get_logger(), "World name is required.");
      return;
    }
    std::string entity_name;
    this->get_parameter("entity_name", entity_name);

    if (entity_name.empty())
    {
      RCLCPP_ERROR(this->get_logger(), "Entity name is required.");
      return;
    }

    gz::transport::Node node;
    std::string service{"/world/" + world_name + "/remove"};

    // Try setting entity type explicitly
    gz::msgs::Entity req;
    req.set_name(entity_name);
    req.set_type(gz::msgs::Entity::MODEL);  // Explicitly set the type to MODEL

    RCLCPP_INFO(this->get_logger(), "Trying to delete entity with name: %s", entity_name.c_str());

    bool executed = node.Request(service, req, &DeleteEntityNode::on_response, this);
    if (!executed)
    {
      RCLCPP_ERROR(this->get_logger(), "Request to delete entity [%s] failed to send.", entity_name.c_str());
    }
    else
    {
      RCLCPP_INFO(this->get_logger(), "Request to delete entity [%s] sent successfully.", entity_name.c_str());
    }
  }

  // Callback function to handle the response from the server
  void on_response(const gz::msgs::Boolean &rep, bool result)
  {
    if (result)
    {
      if (rep.data())
      {
        RCLCPP_INFO(this->get_logger(), "Entity successfully deleted. Shutting down the node...");
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Service responded, but entity deletion was not successful.");
      }
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to get a valid response from the service.");
    }
    rclcpp::shutdown();
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DeleteEntityNode>();
  rclcpp::spin(node);
  return 0;
}
