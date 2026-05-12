cmake_minimum_required(VERSION 3.0.2)
project(ros_langchain_node)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/test_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
