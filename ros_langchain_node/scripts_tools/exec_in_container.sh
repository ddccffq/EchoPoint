#!/bin/bash
# 在宿主机执行：在容器内执行命令

# 检查容器是否运行
if ! docker ps | grep -q ros-langchain-env-container; then
    echo "❌ 容器未运行，请先运行: bash scripts_tools/run_container.sh"
    exit 1
fi

# 执行传入的命令
if [ $# -eq 0 ]; then
    echo "📖 用法: bash scripts_tools/exec_in_container.sh <command>"
    echo ""
    echo "示例:"
    echo "  bash scripts_tools/exec_in_container.sh 'rostopic list'"
    echo "  bash scripts_tools/exec_in_container.sh 'rosnode list'"
    echo "  bash scripts_tools/exec_in_container.sh 'catkin_make'"
    exit 1
fi

docker exec ros-langchain-env-container bash -c "
    source /opt/ros/noetic/setup.bash
    source /root/catkin_ws/devel/setup.bash
    $@
"