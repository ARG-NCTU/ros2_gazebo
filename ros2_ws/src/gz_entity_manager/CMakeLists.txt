cmake_minimum_required(VERSION 3.8)
project(gz_entity_manager)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ignition-transport11 REQUIRED)
find_package(ignition-msgs8 REQUIRED)
find_package(ignition-math6 REQUIRED)

include_directories(include)

add_executable(spawn_entity src/spawn_entity.cpp)
ament_target_dependencies(spawn_entity rclcpp std_msgs ignition-transport11 ignition-msgs8 ignition-math6)

add_executable(delete_entity src/delete_entity.cpp)
ament_target_dependencies(delete_entity rclcpp std_msgs ignition-transport11 ignition-msgs8 ignition-math6)

add_executable(move_entity src/move_entity.cpp)
ament_target_dependencies(move_entity rclcpp std_msgs ignition-transport11 ignition-msgs8 ignition-math6)

install(TARGETS
  spawn_entity
  delete_entity
  move_entity
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()