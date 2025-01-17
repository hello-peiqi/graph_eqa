# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

SHELL ["/bin/bash", "-c"]

# Update and install essential dependencies
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    git \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add ROS Noetic repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Install ROS Noetic Desktop Full
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full ros-noetic-gtsam \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN sh \
    -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" \
        > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-get update && apt-get install wget

RUN wget http://packages.ros.org/ros.key -O - | apt-key add -

RUN apt install -y python3-rosdep python3-catkin-tools python3-vcstool

# Initialize rosdep
RUN rosdep init && rosdep update

USER root

# Source ROS setup file automatically
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN mkdir -p catkin_ws/src \
    && cd catkin_ws \
	&& catkin init \
	&& catkin config -DCMAKE_BUILD_TYPE=Release \
	&& cd src \
	&& git clone https://github.com/blakerbuchanan/Hydra ./hydra \
	&& sed -i -E 's|git@github\.com:(.*)\.git|https://github.com/\1.git|g' hydra/install/hydra.rosinstall \
	&& vcs import . < hydra/install/hydra.rosinstall

WORKDIR catkin_ws

RUN source /opt/ros/noetic/setup.bash && cd src && rosdep install --from-paths . --ignore-src -r -y --rosdistro=noetic
RUN cd src && git clone https://github.com/ros/catkin.git
# RUN catkin config --cmake-args -Dcatkin_DIR=/opt/ros/noetic/share/catkin/cmake
RUN source /opt/ros/noetic/setup.bash && catkin build

WORKDIR /root

# Install python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Python3 dev
RUN apt-get update && apt-get install -y \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Stretch AI Pyaudio
RUN apt-get update && apt-get install -y \
    libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Mamba
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b
# Add mamba to the path
ENV PATH /root/miniforge3/bin:$PATH
RUN mamba create -n "grapheqa" python=3.10 -y \
    && mamba init

RUN source activate grapheqa \
    && git clone https://github.com/hello-peiqi/graph_eqa/ \
    && cd graph_eqa \
    && pip install -e . \
    && pip install numpy-quaternion

RUN source activate grapheqa \
    && git clone https://github.com/hello-robot/stretch_ai --branch hello-peiqi/grapheqa \
    && cd stretch_ai \
    && pip install setuptools==69.5.1 \
    && pip install -e ./src[dev] \
    && git submodule update --init --recursive \
    && cd third_party/detectron2 \
    && pip install -e . \
    && cd ../../src/stretch/perception/detection/detic/Detic \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt \
    && mkdir -p models \
    && wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

WORKDIR /catkin_ws
RUN source activate grapheqa \
    && source /opt/ros/noetic/setup.bash \
    && pip install "src/spark_dsg[viz]" 

WORKDIR /catkin_ws/src/hydra
RUN source /opt/ros/noetic/setup.bash \
    && source activate grapheqa \
    && pip install -r python/build_requirements.txt \
    && pip install git+https://github.com/ros/catkin.git@noetic-devel \
    && source ../../devel/setup.bash \
    && pip install -e .

# For debugging purpose
RUN apt-get update && apt-get install -y vim iputils-ping && rm -rf /var/lib/apt/lists/*

RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

WORKDIR /root/graph_eqa

# Install habitat
RUN source activate grapheqa && pip install git+https://github.com/facebookresearch/habitat-sim.git

RUN git pull origin main

# Default command: Launch bash
CMD ["bash"]
