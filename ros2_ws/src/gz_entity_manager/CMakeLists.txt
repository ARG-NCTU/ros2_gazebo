cmake_minimum_required(VERSION 3.8)
project(gz_entity_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gz-transport12 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-transport12)
find_package(gz-transport13 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-transport13)
find_package(gz-msgs9 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-msgs9)
find_package(gz-math7 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-math7)
find_package(gz-physics7 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-physics7)
find_package(gz-sim8 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-sim8)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(spawn_entity src/spawn_entity.cpp)
ament_target_dependencies(spawn_entity
  rclcpp
  std_msgs
  gz-transport12
  gz-msgs9
  gz-math7
)

add_executable(delete_entity src/delete_entity.cpp)
ament_target_dependencies(delete_entity
  rclcpp
  std_msgs
  gz-transport12
  gz-msgs9
  gz-math7
)

add_executable(move_entity src/move_entity.cpp)
ament_target_dependencies(move_entity
  rclcpp
  std_msgs
  gz-transport12
  gz-msgs9
  gz-math7
)

add_executable(apply_force_torque src/apply_force_torque.cpp)
ament_target_dependencies(apply_force_torque
  rclcpp
  std_msgs
  gz-transport12
  gz-msgs9
  gz-math7
  gz-physics7
)

add_executable(test_function src/test.cpp)
ament_target_dependencies(test_function
  rclcpp
  std_msgs
  gz-transport13
  gz-msgs9
  gz-math7
  gz-sim8
  geometry_msgs
)

install(TARGETS
  spawn_entity
  delete_entity
  move_entity
  apply_force_torque
  test_function
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()