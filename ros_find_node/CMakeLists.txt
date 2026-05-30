cmake_minimum_required(VERSION 2.8.3)
project(ros_find_node)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  bodyhub
  ros_vision_node
  ros_AIUI_node
  ros_langchain_node
  std_srvs
  message_generation
)

add_service_files(
  FILES
  StartFind.srv
)

generate_messages()

catkin_package(
  CATKIN_DEPENDS message_runtime
)

catkin_install_python(PROGRAMS
  scripts/red_ball_centering_node.py
  scripts/pointing_centering_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
