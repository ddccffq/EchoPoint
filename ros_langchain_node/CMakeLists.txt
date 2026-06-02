cmake_minimum_required(VERSION 3.0.2)
project(ros_langchain_node)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

## 声明 srv 文件
add_service_files(
  FILES
  DescribeImage.srv
)

## 生成消息/服务代码
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 安装 Python 可执行节点（按你当前 tree 里真实存在的文件）
catkin_install_python(PROGRAMS
  src/langchain_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
