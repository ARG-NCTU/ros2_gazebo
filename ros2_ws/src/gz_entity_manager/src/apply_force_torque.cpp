#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <gz/msgs/pose_v.pb.h>
#include <gz/transport/Node.hh>
#include <string>

class FindEntityBaseLink : public rclcpp::Node{
    public:
        void poseVCallBack(const gz::msgs::Pose_V &msg){
            std::cout << "enter call back function\n";
            RCLCPP_INFO(this->get_logger(), "Received Pose_V with %s poses.", std::to_string(msg.pose_size()).c_str());
            for(int i=0;i<msg.pose_size();i++){
                const auto &pose = msg.pose(i);
                std::cout << pose.name() << std::endl;
                if(pose.name() == ename){
                    target_id = pose.id();
                    RCLCPP_INFO(this->get_logger(), "Entity found with id : %s", std::to_string(target_id).c_str());
                    schedule_shutdown();
                }
            }
            RCLCPP_ERROR(this->get_logger(), "Entity not found.");
            schedule_shutdown();
        }

        FindEntityBaseLink(std::string world_name, std::string entity_name) : Node("FindEntityBaseLink"){
            wname = world_name;
            ename = entity_name;
            gz::transport::Node node;
            std::string topic{"/world/" + world_name + "/pose/info"};
            std::function<void(const gz::msgs::Pose_V&)> callback =
                std::bind(&FindEntityBaseLink::poseVCallBack, this, std::placeholders::_1);
            while(!node.Subscribe(topic, callback)){
                RCLCPP_ERROR(this->get_logger(), "Error subscribing %s", topic.c_str());
                node.Subscribe(topic, callback);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            RCLCPP_INFO(this->get_logger(), "Subscribed to topic : %s", topic.c_str());
            std::cout << target_id << std::endl;
        }

        int get_target_id(){return target_id;}

    private:
        int target_id;
        std::string wname;
        std::string ename;
        rclcpp::TimerBase::SharedPtr shutdown_timer_;

        void schedule_shutdown(){
            // Schedule a timer to shut down after 1 second
            shutdown_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            []() {
                RCLCPP_INFO(rclcpp::get_logger("FindEntityBaseLink"), "Shutting down the node...");
                rclcpp::shutdown();
            });
        }
};

class ApplyForceTorqueNode : public rclcpp::Node{
    public:
        ApplyForceTorqueNode() : Node("apply_force_torque"){
            this->declare_parameter<std::string>("world", "waves");
            this->declare_parameter<std::string>("entity_name", "blueboat");
            this->declare_parameter<double>("fx", 0.0);
            this->declare_parameter<double>("fy", 0.0);
            this->declare_parameter<double>("fz", 0.0);
            this->declare_parameter<double>("tx", 0.0);
            this->declare_parameter<double>("ty", 0.0);
            this->declare_parameter<double>("tz", 0.0);

            applyForceTorque();
        }

    private:
        double fx, fy, fz, tx, ty, tz;
        std::string world_name, entity_name;
        int _entity_base_link_id;
        rclcpp::TimerBase::SharedPtr shutdown_timer_;
        
        void schedule_shutdown(){
            // Schedule a timer to shut down after 1 second
            shutdown_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            []() {
                RCLCPP_INFO(rclcpp::get_logger("ApplyForceTorqueNode"), "Shutting down the node...");
                rclcpp::shutdown();
            });
        }

        void applyForceTorque(){
            this->get_parameter("world", world_name);
            this->get_parameter("entity_name", entity_name);
            this->get_parameter("fx", fx);
            this->get_parameter("fy", fy);
            this->get_parameter("fz", fz);
            this->get_parameter("tx", tx);
            this->get_parameter("ty", ty);
            this->get_parameter("tz", tz);

            RCLCPP_INFO(this->get_logger(), "Finding Target Entity(base_link)...");
            auto febl = std::make_shared<FindEntityBaseLink>(world_name, entity_name);
            rclcpp::spin(febl);
            _entity_base_link_id = febl->get_target_id();
            RCLCPP_INFO(this->get_logger(), "get entity id : %d", _entity_base_link_id);
        }
        
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ApplyForceTorqueNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

