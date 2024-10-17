# ros2 gazebo
## Clone REPO
```
git clone git@github.com:ARG-NCTU/ros2_gazebo.git --recurse-submodules
```
## Build ROS2 WorkSpace
```
source Docker/gpu/run.sh
source clean_ros2.sh
source build_ros2_ws.sh
```

## Start BlueBoat SITL

First Terminal
```
source Docker/gpu/run.sh
source environment.sh
ros2 launch blueboat_sitl blueboat_sitl.launch.py
```

Second Terminal
```
source Docker/gpu/run.sh
source environment.sh
ros2 launch mavros apm.launch fcu_url:=udp://:14551@127.0.0.1:14551 #gps_url:=udp:/<your ip> if needed
```

Third Terminal
Open Your QGC localy