FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04 AS main-setup


ARG DEBIAN_FRONTEND=noninteractive

ENV ROS_DISTRO=humble \
    GZ_VERSION=harmonic \
    SHELL=/bin/bash \
    NB_USER=arg \
    NB_UID=1000

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime

###################################### user #####################################
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} 

RUN echo "root:root" | chpasswd \
    && echo "${NB_USER}:111111" | chpasswd

ENV HOME=/home/${NB_USER}

###################################### basic #####################################
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    ccache \
    g++ \
    gcc \
    gdb \
    gawk \
    make \
    ninja-build \
    python3-numpy \
    python3-dev \
    python3-pyparsing \
    python3-serial \
    python-is-python3 \
    libpython3-stdlib \
    libtool-bin \
    zip \
    default-jre \
    socat \
    tzdata \
    pip \
    dirmngr \
    gnupg2 \
    git \
    gitk git-gui \
    vim \
    less \
    curl \
    wget \
    sudo \
    cmake \
    swig \
    net-tools \
    lsb-release \
    gnupg \
    build-essential \
    libcgal-dev \
    libfftw3-dev \
    rapidjson-dev \
    pkg-config \
    libboost-system-dev \
    libopencv-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

###################################### python pkg###############################
RUN apt-get update && apt-get upgrade pip -y && \
    pip install --upgrade pip && \
    pip install \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    scikit-image \
    pandas \
    opencv-python \
    opencv-python-headless \
    pyyaml \
    pyquaternion \
    pymavlink \
    MAVProxy \
    pyserial \
    pexpect \
    Pillow \
    tensorboard \
    tensorflow \
    future

################################## torch gymnasium sb3 #####################################
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --ignore-installed PyYAML && \
    python3 -m pip install \
    torch \
    torchaudio \
    torchvision \
    gymnasium \
    gymnasium[box2d] \
    stable-baselines3 \
    openai \
    trajectopy-core \
    huggingface-sb3 \
    huggingface_hub \
    transformers \
    datasets \
    hf_transfer \
    opencv-python-headless \
    timm

###################################### ros2 ####################################
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-joy* \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-mavros \
    ros-${ROS_DISTRO}-mavros-extras \
    ros-${ROS_DISTRO}-mavros-msgs \
    ros-${ROS_DISTRO}-rosbridge-server \
    ros-${ROS_DISTRO}-gps-msgs \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-ament-cmake \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-tf2-msgs \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-rviz-rendering \
    ros-${ROS_DISTRO}-rcl-interfaces \
    ros-${ROS_DISTRO}-urdf \
    ros-${ROS_DISTRO}-geographic-msgs \
    ros-${ROS_DISTRO}-topic-tools \
    ros-dev-tools \
    ros-${ROS_DISTRO}-launch-pytest \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-dev-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt && git clone -b ros2 https://github.com/mavlink/mavros.git && \
    cd mavros/mavros/scripts && \
    ./install_geographiclib_datasets.sh
###################################### gazebo ##################################
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && \
    apt-get install -y \
    libgz-sim8-dev \
    libgz-transport13-dev \
    libignition-utils1-dev \
    libsdformat14-dev \
    gz-${GZ_VERSION} \
    ros-${ROS_DISTRO}-ros-gz${GZ_VERSION} \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rosdep init && \
    rosdep update
###################################### microxrceddsgen ####################################
FROM eclipse-temurin:19-jdk-jammy AS dds-gen-builder
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    && apt-get clean \
    && apt-get -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN git clone -b master --recurse-submodules https://github.com/ArduPilot/Micro-XRCE-DDS-Gen.git --depth 1 --no-single-branch --branch master dds-gen \
    && cd dds-gen \
    && ./gradlew assemble

FROM main-setup

# Copy DDS Gen to final container
COPY --from=dds-gen-builder /dds-gen/scripts /usr/local/dds-gen/scripts/
COPY --from=dds-gen-builder /dds-gen/share /usr/local/dds-gen/share/

# Ensure microxrceddsgen is in PATH
ENV PATH="/usr/local/dds-gen/scripts:$PATH"

# Set the buildlogs directory into /tmp as other directories aren't accessible
ENV BUILDLOGS=/tmp/buildlogs
ENV TZ=UTC
###################################### sitl ################################################

ARG ARDU_WS=/opt/ardupilot_ws
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgps-dev && \
    mkdir -p ${ARDU_WS}/src && \
    cd ${ARDU_WS}/src && \
    git clone -b ${ROS_DISTRO} https://github.com/micro-ROS/micro_ros_msgs.git && \
    git clone https://github.com/ArduPilot/ardupilot.git && \
    git clone -b ros2 https://github.com/ros-perception/vision_msgs.git && \
    git clone https://github.com/rudislabs/actuator_msgs.git && \
    git clone https://github.com/swri-robotics/gps_umd.git -b ros2-devel && \
    cd ardupilot && git submodule update --init --recursive && \
    cd ${ARDU_WS} && vcs import --input https://raw.githubusercontent.com/ArduPilot/ardupilot_gz/main/ros2_gz.repos --recursive src && \
    cd ${ARDU_WS}/src/ardupilot && \
    ./waf configure --board sitl && \
    ./waf rover
    
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
cd ${ARDU_WS} && \
colcon build --packages-select actuator_msgs && \
colcon build --packages-skip actuator_msgs"


ENV PATH=$PATH:${ARDU_WS}/src/ardupilot/Tools/autotest
ENV PATH=/usr/lib/ccache:$PATH

###################################### asv water sim ################################
ARG GZ_ENV_WS=/opt/gzenv_ws
RUN mkdir -p ${GZ_ENV_WS}/src && \
    cd ${GZ_ENV_WS}/src && \
    git clone https://github.com/uwe77/asv_wave_sim.git && cd .. && \
    colcon build --symlink-install --merge-install --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_TESTING=ON \
    -DCMAKE_CXX_STANDARD=17 && \
    cd ${GZ_ENV_WS}/src/asv_wave_sim/gz-waves/src/gui/plugins/waves_control && \
    mkdir build && cd build && \
    cmake .. && make
    
ENV GZ_SIM_RESOURCE_PATH=${ARDU_WS}/src/ardupilot_gazebo/worlds:${ARDU_WS}/src/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_ENV_WS}/src/asv_wave_sim/gz-waves-models/models:${GZ_ENV_WS}/src/asv_wave_sim/gz-waves-models/world_models:${GZ_ENV_WS}/src/asv_wave_sim/gz-waves-models/worlds \
    GZ_SIM_SYSTEM_PLUGIN_PATH=${ARDU_WS}/install/ardupilot_gazebo/lib/ardupilot_gazebo/lib:${GZ_ENV_WS}/install/lib \
    GZ_GUI_PLUGIN_PATH=${GZ_ENV_WS}/src/asv_wave_sim/gz-waves/src/gui/plugins/waves_control/build

RUN echo "root ALL=(ALL)  ALL" >> /etc/sudoers \
    && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ${HOME}/.bashrc \
    && echo "source ${ARDU_WS}/install/setup.bash" >> ${HOME}/.bashrc \
    && echo "source ${GZ_ENV_WS}/install/setup.bash" >> ${HOME}/.bashrc

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV HF_HUB_ENABLE_HF_TRANSFER=1