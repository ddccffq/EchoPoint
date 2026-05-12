#!/bin/bash

echo "========================================="
echo "🧪 测试宿主机与 Docker 容器的 ROS 通信"
echo "========================================="

# 检查容器是否运行
if ! docker ps | grep -q ros-langchain-env-container; then
    echo "❌ 容器未运行"
    exit 1
fi

echo "✅ 容器正在运行"

# 清理容器内的所有 ROS 进程
echo "🧹 清理容器内的 ROS 进程..."
docker exec ros-langchain-env-container bash -c "
    pkill -f test_node 2>/dev/null || true
    pkill -f roscore 2>/dev/null || true
    pkill -f rosout 2>/dev/null || true
    sleep 2
"

# 启动 roscore（先 source 环境）
echo ""
echo "📡 启动 roscore..."
docker exec -d ros-langchain-env-container bash -c "
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    roscore
"
sleep 3

# 启动测试节点
echo "🚀 启动 test_node..."
docker exec -d ros-langchain-env-container bash -c "
    source /opt/ros/noetic/setup.bash && \
    source /root/catkin_ws/devel/setup.bash && \
    rosrun ros_langchain_node test_node.py
"
sleep 3

# 发布测试消息
echo ""
echo "📤 发送测试消息..."
docker exec ros-langchain-env-container bash -c "
    source /opt/ros/noetic/setup.bash && \
    rostopic pub /test_topic std_msgs/String 'Hello from Docker!' -1
"

sleep 2

# 查看节点是否运行
echo ""
echo "📋 查看运行中的节点..."
docker exec ros-langchain-env-container bash -c "
    source /opt/ros/noetic/setup.bash && \
    rosnode list
"

# 清理
echo ""
echo "🧹 清理..."
docker exec ros-langchain-env-container bash -c "
    pkill -f test_node 2>/dev/null || true
    pkill -f roscore 2>/dev/null || true
"

echo ""
echo "========================================="
echo "✅ 通信测试完成"
echo "========================================="