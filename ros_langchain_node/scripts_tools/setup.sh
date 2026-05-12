#!/bin/bash
set -e

echo "========================================="
echo "🚀 一键配置 ROS + LangChain Docker 环境"
echo "========================================="

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装"
    exit 1
fi

# 检查镜像
if ! docker images | grep -q ros-langchain-env; then
    echo "❌ ros-langchain-env 镜像不存在"
    exit 1
fi

# 删除旧容器
if docker ps -a | grep -q ros-langchain-env-container; then
    echo "📦 删除旧容器..."
    docker rm -f ros-langchain-env-container 2>/dev/null || true
fi

# 创建新容器
echo "📦 创建新容器..."
docker run -itd \
    --network host \
    --name ros-langchain-env-container \
    -v ~/robot_ros_application/catkin_ws:/root/catkin_ws \
    ros-langchain-env:latest \
    bash

echo "✅ 容器创建成功"
sleep 3

# 安装 Python 依赖
echo "🐍 安装 Python 依赖..."
docker exec ros-langchain-env-container bash -c "
    pip3 install catkin_pkg empy rospkg pyyaml --quiet 2>/dev/null || true
"

# 创建测试节点
echo "📝 创建测试节点..."
mkdir -p ~/robot_ros_application/catkin_ws/src/ros_langchain_node/scripts

cat > ~/robot_ros_application/catkin_ws/src/ros_langchain_node/scripts/test_node.py << 'PYEOF'
#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

class TestNode:
    def __init__(self):
        rospy.init_node('test_node', anonymous=True)
        self.pub = rospy.Publisher('/test_topic', String, queue_size=10)
        rospy.Subscriber('/test_response', String, self.callback)
        rospy.loginfo("✅ 测试节点启动")
    
    def callback(self, msg):
        rospy.loginfo("📨 收到响应: %s", msg.data)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = TestNode()
    rospy.sleep(1)
    node.run()
PYEOF

# 配置 CMakeLists.txt
echo "⚙️ 配置 CMakeLists.txt..."
cat > ~/robot_ros_application/catkin_ws/src/ros_langchain_node/CMakeLists.txt << 'CMAKEEOF'
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
CMAKEEOF

# 只编译 ros_langchain_node 包
echo "🔨 编译 ros_langchain_node 包..."
docker exec ros-langchain-env-container bash -c "
    cd /root/catkin_ws && \
    source /opt/ros/noetic/setup.bash && \
    rm -rf build devel 2>/dev/null && \
    catkin_make -DCATKIN_WHITELIST_PACKAGES='ros_langchain_node'
"

echo "========================================="
echo "✅ 环境配置完成！"
echo "========================================="