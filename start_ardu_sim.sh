#!/bin/bash

mkdir -p ./sim_log && cd ./sim_log || exit 1

sim_vehicle.py -v Rover --add-param-file=../boat.parm --console --out=udp:192.168.0.110:14551 --out=udp:192.168.0.110:14550 -l 22.59920,120.298691,0,270

cd ..
# param set FRAME_CLASS 2
# param set FRAME_TYPE 0
