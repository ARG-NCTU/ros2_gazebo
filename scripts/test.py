from pymavlink import mavutil
import time

# Establish MAVLink connection to the rover
device_file = 'udp:127.0.0.1:14551'
master = mavutil.mavlink_connection(device_file)

def send_acro_mode_command():
    """Send a command to set the rover to ACRO mode."""
    global master
    # Wait for the first heartbeat from the autopilot
    master.wait_heartbeat()
    mode_id = master.mode_mapping().get("ACRO")
    if mode_id is None:
        print("ACRO mode not available.")
        return
    master.set_mode(mode_id)
    print("ACRO mode set!")

def send_arm_command():
    """Send a command to arm the rover."""
    global master
    master.mav.command_long_send(master.target_system,
                                 master.target_component,
                                 mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                 0, 1, 0, 0, 0, 0, 0, 0)
    print("Arm command sent!")

def check_arm_status():
    """Check if the rover is armed."""
    global master
    while True:
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 1, 1
        )
        msg = master.recv_match(type='HEARTBEAT', blocking=True)
        arm_state = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        if arm_state:
            print("Rover is armed!")
            break
        else:
            print("Waiting for arming...")
        time.sleep(1)

def send_rc_override(cmd):
    """Send RC override command with the given channel values."""
    global master
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        *cmd  # Unpack the list so that all 8 channels are passed
    )
    print(f"RC Override sent: {cmd}")

def send_heartbeat():
    """Send a heartbeat to maintain connection with the vehicle."""
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    print("Heartbeat sent.")

# Initialize the system
print("Waiting for the vehicle heartbeat...")
master.wait_heartbeat()
print("Heartbeat received from system.")

# Send commands to arm the rover and set to ACRO mode
send_arm_command()
check_arm_status()
send_acro_mode_command()

# RC Override command (adjust based on your rover configuration)
# Channel 1: Steering, Channel 3: Throttle
cmd = [1500, 0, 1000, 0, 0, 0, 0, 0]  # Set appropriate values for your case

# Main loop for sending RC override and heartbeats
while True:
    send_rc_override(cmd)
    send_heartbeat()
    time.sleep(0.1)  # Sleep to avoid overloading the connection
