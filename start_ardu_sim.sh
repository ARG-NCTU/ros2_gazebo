#!/bin/bash

mkdir -p ./sim_log && cd ./sim_log || exit 1

sim_vehicle.py -v Rover --add-param-file=../boat.parm --console --out=udp:127.0.0.1:14551 --out=udp:127.0.0.1:14550 -l 22.59920,120.298691,0,270

cd ..
# param set FRAME_CLASS 2
# param set FRAME_TYPE 0
