cmake_minimum_required(VERSION 3.5)
project(f1_final_project)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  tf2
  tf2_ros)

  find_package(${PACKAGE} REQUIRED)

endforeach()

#Include Cpp "include" directory
include_directories(include)
include_directories(
 ${EIGEN3_INCLUDE_DIRS}
)

#Create Cpp executables
add_executable(pure_pursuit_node src/pure_pursuit_node.cpp)

ament_target_dependencies(pure_pursuit_node
 rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2_ros
)

# Install Cpp executables
install(TARGETS
 pure_pursuit_node
 DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/debug_node.py
  scripts/pure_pursuit_node.py
  scripts/waypoint_gen_node.py
  scripts/sensor_filtering.py
  scripts/waypoint_gen_node_mpc.py
  scripts/mpc_node.py
  scripts/utils.py
  scripts/debug_node_mpc.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
