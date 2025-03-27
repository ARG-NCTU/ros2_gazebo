# check REPO_NAME is set
if [ -z "$REPO_NAME" ]; then
    echo "REPO_NAME is not set. Please set REPO_NAME to the name of the repository."
    return
fi

# check GZ_SIM_RESOURCE_PATH contain ~/$REPO_NAME/Gazebo/models
if [[ ":$GZ_SIM_RESOURCE_PATH:" != *":$HOME/$REPO_NAME/Gazebo/models:"* ]]; then
    export GZ_SIM_RESOURCE_PATH="$HOME/$REPO_NAME/Gazebo/models:$HOME/$REPO_NAME/Gazebo/worlds:$GZ_SIM_RESOURCE_PATH"
fi
source ./ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
