cmake_minimum_required(VERSION 2.8.3)
project(red_ball_centering_node)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  bodyhub
  ros_vision_node
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/red_ball_centering_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
