FROM docker.1panel.live/library/ubuntu:20.04

# 设置环境变量避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装 apt-utils 避免警告
RUN apt-get update && apt-get install -y \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

# 替换为阿里云 APT 源（加速系统软件包下载）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装基础工具
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 添加 ROS Noetic 源（使用清华镜像源加速）
RUN sh -c 'echo "deb http://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# 添加 ROS 密钥
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# 安装 ROS Noetic 基础包
RUN apt-get update && apt-get install -y \
    ros-noetic-ros-base \
    ros-noetic-rosbridge-server \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep（手动创建源列表，绕过 GitHub）
RUN mkdir -p /etc/ros/rosdep/sources.list.d && \
    echo "yaml https://mirrors.tuna.tsinghua.edu.cn/rosdistro/rosdep/osx-homebrew.yaml osx" > /etc/ros/rosdep/sources.list.d/20-default.list && \
    echo "yaml https://mirrors.tuna.tsinghua.edu.cn/rosdistro/rosdep/base.yaml" >> /etc/ros/rosdep/sources.list.d/20-default.list && \
    echo "yaml https://mirrors.tuna.tsinghua.edu.cn/rosdistro/rosdep/python.yaml" >> /etc/ros/rosdep/sources.list.d/20-default.list && \
    echo "yaml https://mirrors.tuna.tsinghua.edu.cn/rosdistro/rosdep/ruby.yaml" >> /etc/ros/rosdep/sources.list.d/20-default.list && \
    echo "gbpdistro https://mirrors.tuna.tsinghua.edu.cn/rosdistro/releases/fuerte.yaml fuerte" >> /etc/ros/rosdep/sources.list.d/20-default.list

# 设置环境变量并执行 rosdep update
RUN export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml && \
    for i in 1 2 3; do rosdep update && break || sleep 10; done

# --- 替换原有的 Python 3.10 安装步骤 ---
# 安装编译 Python 源码所需的依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 下载、编译并安装 Python 3.10
# 使用 --enable-optimizations 会进行性能优化，可能会稍微延长编译时间，但能获得更好的运行时性能
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations --prefix=/usr/local && \
    make -j "$(nproc)" && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.14*

# 创建软链接，方便使用 python3 和 pip3 命令
RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip3

# 验证安装
RUN python3 --version && pip3 --version

# 配置 pip 使用清华源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 升级 pip 并安装 LangChain
RUN pip3 install --upgrade pip
RUN pip3 install langchain langchain-community langchain-openai

# 设置环境变量
ENV ROS_DISTRO=noetic
ENV ROS_MASTER_URI=http://localhost:11311

CMD ["bash"]
