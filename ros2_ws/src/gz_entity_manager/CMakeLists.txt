cmake_minimum_required(VERSION 3.8)
project(gz_entity_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gz-transport12 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-transport12)
find_package(gz-msgs9 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-msgs9)
find_package(gz-math7 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/gz-math7)
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

install(TARGETS
  spawn_entity
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()