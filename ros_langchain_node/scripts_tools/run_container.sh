#!/bin/bash
# 在宿主机执行：启动 Docker 容器

echo "========================================="
echo "🐳 启动 Docker 容器"
echo "========================================="

# 检查容器是否存在
if ! docker ps -a | grep -q ros-langchain-env-container; then
    echo "⚠️ 容器不存在，请先运行: bash scripts_tools/setup.sh"
    exit 1
fi

# 检查容器是否已运行
if docker ps | grep -q ros-langchain-env-container; then
    echo "✅ 容器已在运行中"
else
    echo "📦 启动容器..."
    docker start ros-langchain-env-container
    sleep 2
    echo "✅ 容器已启动"
fi

echo ""
echo "📋 容器信息:"
docker ps --filter name=ros-langchain-env-container --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🔧 进入容器命令:"
echo "  docker exec -it ros-langchain-env-container bash"