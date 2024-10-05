import gymnasium as gym
from gymnasium import error, spaces, utils
# from gymnasium.utils import seeding
import rospy
import time
import numpy as np
import math
import random
import sys
import os
import queue
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan, Imu
from std_srvs.srv import Empty
from std_msgs.msg import Int64, Header
from geometry_msgs.msg import Twist, Vector3, PoseStamped
from gazebo_msgs.msg import ModelState, ContactsState
from gazebo_msgs.srv import SetModelState, GetModelState, GetPhysicsProperties, SetPhysicsProperties, SetPhysicsPropertiesRequest
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Wrench, Point
import csv

# full_path = os.path.realpath(__file__)
# INITIAL_STATES = np.genfromtxt(os.path.dirname(
#     full_path) + '/wamv_heading_control2.csv', delimiter=',')

class obstacle():
    def __init__(self, name=None) -> None:
        self.name = name
        self.pose = np.zeros(3)
        self.force = np.zeros(3)
    
    def reset(self):
        r = np.random.uniform(0, np.pi)
        init_pose = 25*np.array([math.cos(r), math.sin(r), 1/25])
        self.reset_pose(init_pose)
        dir = r + np.random.normal(0, math.pi/60) + np.pi
        dir = dir % (2*np.pi)
        self.force = np.random.uniform(80, 100)*np.array([math.cos(dir), math.sin(dir), 0])

    def apply_force(self, force=np.zeros(3), torque=np.zeros(3)):
        f = Wrench()
        f.force = Vector3(x=force[0], y=force[1], z=force[2])
        f.torque = Vector3(x=torque[0], y=torque[1], z=torque[2])
        body_name_str = f"{self.name}" + "::" + "base_link"
        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        apply_wrench(body_name = body_name_str,
                        reference_frame = 'world', 
                        reference_point = Point(x=0.0, y=0.0, z=0.0),
                        wrench = f,
                        start_time = rospy.Time(0.0),
                        duration = rospy.Duration(0.04))
        
    def update_pose(self):
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)
        try:
            agent = get_model(self.name, '')
        except (rospy.ServiceException) as e:
            print(e)
        self.pose = np.array(
            [agent.pose.position.x, agent.pose.position.y, agent.pose.position.z])
        
    def reset_pose(self, pose=np.zeros(3)):
        def get_initail_obstacle_state(n, pose):
            state_msg = ModelState()
            state_msg.model_name = n
            state_msg.pose.position.x = pose[0]
            state_msg.pose.position.y = pose[1]
            state_msg.pose.position.z = pose[2]
            r = R.from_euler('z', 0)
            quat = r.as_quat()
            state_msg.pose.orientation.x = quat[0]
            state_msg.pose.orientation.y = quat[1]
            state_msg.pose.orientation.z = quat[2]
            state_msg.pose.orientation.w = quat[3]
            return state_msg
        reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        reset_model(get_initail_obstacle_state(self.name, pose))

    def apply_task(self):
        self.apply_force(self.force)

class WamvMotorAnchoringV6(gym.Env):
    # metadata = {'render.modes': ['laser']}

    def __init__(self):
        rospy.init_node('gym_subt')

        # env parameter
        self.max_dis = 10
        self.track_num = 10
        self.state_num = 4
        self.action_scale = {'linear': 0.8, 'angular': 0.8}

        # global variable
        self.total_step = 0
        self.step_count = 0
        self.epi = 0
        self.frame = 0
        self.last_dis = 0
        self.last_angle = 0
        self.velocity = 0
        self.time_diff = 0
        self.last_time = None
        self.total_dis = 0
        self.goal = np.array([0, 0, 0])
        self.last_action = [0, 0]
        self.angle = 0
        self.station_keeping_count = 0
        self.last_pos = None
        self.pos_track = None
        self.velocity_track = None
        self.action_track = None
        self.reset_robot = False
        self.contact = False
        self.state_stack = None
        self.maxstep = 4090
        self.sr_queue = queue.Queue(maxsize=self.maxstep)
        self.done_sf_flag = 0
        self.__stage = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        self.difficulty = 0
        self.save_sr_file = '/home/argrobotx/robotx-2022/acme_logs/success_rate/sr_output.csv'
        self.obstacle = obstacle()

        # ServiceProxy
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        self.get_model = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)
        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty)
        self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # publisher subscriber
        self.pub_twist = rospy.Publisher('/wamv/cmd_vel', Twist, queue_size=1)
        self.sub_laser_upper = rospy.Subscriber(
            '/wamv/RL/scan', LaserScan, self.cb_laser, queue_size=1)
        self.sub_contact = rospy.Subscriber('/wamv/bumper_states', ContactsState, self.cb_contact, queue_size = 1)
        # unpause physics
        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.pause_gym(False)

        # set real time factor
        os.system('gz physics -u 0')

        self.observation_space = gym.spaces.Dict({
            "laser": gym.spaces.Box(low=-10, high=10, shape=(4, 241), dtype=np.float64),
            "track": gym.spaces.Box(low=-10, high=10, shape=(30,), dtype=np.float64),
            "vel": gym.spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float64)
            # "action": gym.spaces.Box(low=-10, high=10, shape=(20,), dtype=np.int64)
        })
        self.action_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # state info
        self.info = {'laser_shape': (self.state_num, 241),
                     'goal_shape': (self.track_num, 4)}

        print("gym env: vrx-goal")
        print("obs_dim: ", self.observation_space['laser'].shape[0])
        print("act_dim: ", self.action_space.shape[0])
        print(self.info)
    
    # transform map goal to model frame by using rotation matrix
    def map_to_model_frame(self, map_goal_pose, map_robot_pose, map_robot_orientation):
        # get robot pose in map frame
        x, y = map_goal_pose - map_robot_pose
        # get robot orientation in map frame
        robot_orientation = - map_robot_orientation  
        # get rotation matrix
        rotation_matrix = R.from_euler('z', robot_orientation).as_matrix()
        map_point = np.array([x, y, 1.0])
        # get goal point in model frame 
        model_point = np.dot(rotation_matrix, map_point)
        x_prime, y_prime = model_point[:2]
        return x_prime, y_prime

    def save_sr(self, sr):
        with open(self.save_sr_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.difficulty, sr])
            
    def apply_force_wamv(self, force_x, force_y):
        
        force = Wrench()
        force.force = Vector3(x=force_x, y=force_y, z=0.0)
        force.torque = Vector3(x=0.0, y=0.0, z=0.0)

        body_name_str = "wamv" + "::" + "wamv/base_link"
        self.apply_wrench(body_name = body_name_str,
                        reference_frame = 'world', 
                        reference_point = Point(x=0.0, y=0.0, z=0.0),
                        wrench = force,
                        start_time = rospy.Time(0.0),
                        duration = rospy.Duration(0.04))

    def get_initial_state(self, name):
        # start position
        state_msg = ModelState()
        state_msg.model_name = name
        x_rand = np.random.uniform(-5, 5)
        y_rand = np.random.uniform(-5, 5)
        state_msg.pose.position.x = x_rand
        state_msg.pose.position.y = y_rand
        # state_msg.pose.position.x = 0
        # state_msg.pose.position.y = 0
        state_msg.pose.position.z = 0
        # r = R.from_euler('z', np.random.uniform(-np.pi, np.pi))
        # quat = r.as_quat()
        # diff = goal_position - init_position
        # self.angle = math.atan2(diff[1],diff[0])

        # angle = 1.57
        angle_rand = np.random.uniform(-0.35, 0.35)
        angle = 1.57 + angle_rand
        if angle >= np.pi:
            angle -= 2*np.pi
        elif angle <= -np.pi:
            angle += 2*np.pi
        r = R.from_euler('z', angle)
        quat = r.as_quat()
        state_msg.pose.orientation.x = quat[0]
        state_msg.pose.orientation.y = quat[1]
        state_msg.pose.orientation.z = quat[2]
        state_msg.pose.orientation.w = quat[3]
        return state_msg
    
    def cb_laser(self, msg):
        pass

    def cb_contact(self, msg):
        self.contact = False
        if msg.states != []: 
            self.contact = True
            # print("\033[33mcollision !!\033[0m")

    def set_max_dis(self, max_d):
        self.max_dis = max_d

    def scale_linear(self, n, bound):
        # bound
        return np.clip(n, self.action_space.low[0], self.action_space.high[0])*bound

    def scale_angular(self, n, bound):
        return np.clip(n, self.action_space.low[1], self.action_space.high[1])*bound

    def pause_gym(self, pause=True):
        srv_name = '/gazebo/pause_physics' if pause else '/gazebo/unpause_physics'
        rospy.wait_for_service(srv_name)
        try:
            if pause:
                self.pause_physics()
            else:
                self.unpause_physics()

        except (rospy.ServiceException) as e:
            print(e)

    def step(self, action):
        self.pause_gym(False)
        self.step_count += 1
        action[0] = self.scale_linear(action[0], 1)
        action[1] = self.scale_angular(action[1], 1)
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]

        if self.step_count % 200 == 1 and self.obstacle.name is None:
            ob = np.random.randint(2)
            if ob == 0:
                obstacle('polyform_red').reset_pose(np.array([-30, 30, 1]))
                self.obstacle = obstacle('polyform_black')
            elif ob == 1:
                obstacle('polyform_black').reset_pose(np.array([30, 30, 0]))
                self.obstacle = obstacle('polyform_red')

        wamv_obs_dis = 0
        if self.obstacle.name is not None:
            obsta_dis = np.linalg.norm(self.obstacle.pose[:2] - self.goal[:2])
            wamv_obs_dis = np.linalg.norm(self.last_pos[:2] - self.obstacle.pose[:2])
            if obsta_dis >= 35:
                self.obstacle.reset()
            else:
                self.obstacle.apply_task()
        elif self.obstacle.name is None and self.step_count % 400 == 201:
            ob = np.random.randint(2)
            if ob == 0:
                obstacle('polyform_red').reset_pose(np.array([-30, 30, 1]))
                self.obstacle = obstacle('polyform_black')
            elif ob == 1:
                obstacle('polyform_black').reset_pose(np.array([30, 30, 0]))
                self.obstacle = obstacle('polyform_red')
            self.obstacle.reset()
            wamv_obs_dis = 100

            
        if self.step_count % 100 == 1:
            self.wamv_force = [np.random.choice([-1, 1]) * np.random.normal(5, 8),
                              np.random.choice([-1, 1]) * np.random.normal(5, 8)]
        self.apply_force_wamv(self.wamv_force[0], self.wamv_force[1])

        self.pub_twist.publish(cmd_vel)
        self.last_action = [action[0], action[1]]
        state = self.get_observation()
        # calculate relative pos
        location_dif = self.goal[:2] - self.last_pos[:2]

        self.frame += 1
        # done = False
        terminated = False
        truncated = False
        reach_done = False
        stay_30_steps = False
        first_done = False
        max_step = 4090
        max_reward = 100
        wamv_boundry = 3
        goal_state = "reaching"
        ##################reward design##################
        
        reward = 0.0
        dis_to_g = np.linalg.norm(location_dif)
        
        
        r_location = math.exp(-0.15*dis_to_g**2)*math.exp(-8*math.tan(self.angle/2)**2)
        r_energy = -0.05 * np.absolute(action[0]) - 0.05 * np.absolute(action[1])
        r_obstacle = math.exp(-0.05*(wamv_obs_dis-wamv_boundry)**2) if self.obstacle.name is not None else 0

        if dis_to_g < 4 and np.absolute(self.angle) < 0.25:
            goal_state = "\033[33mgoal!!       \033[0m"
            self.sr_queue.put(1)

        if self.step_count >= max_step:
            truncated = True

        if dis_to_g > 10 :
            goal_state = "\033[33moutOfBounds!!\033[0m"
            self.station_keeping_count = 0
            reward = -max_reward
            terminated = True
            self.done_sf_flag = 0
        elif (wamv_obs_dis <= 3 or self.contact) and self.obstacle.name is not None:
            goal_state = "\033[33mcollision!!  \033[0m"
            reward = -max_reward
            terminated = True
            self.done_sf_flag = 0
        else:
            reward = (
                + max_reward*r_location
                + r_energy
                - 0.3*max_reward*r_obstacle
            )
        output = "\rstep:{:4d}, force on wamv: [x:{}, y:{}], reward:{}, state:{}".format(
            self.step_count,
            " {:4.2f}".format(self.wamv_force[0]) if self.wamv_force[0] >= 0 else "{:4.2f}".format(self.wamv_force[0]),
            " {:4.2f}".format(self.wamv_force[1]) if self.wamv_force[1] >= 0 else "{:4.2f}".format(self.wamv_force[1]),
            " {:4.2f}".format(reward) if reward >= 0 else "{:4.2f}".format(reward),
            goal_state
        )
        reward /= max_step

        ##################reward design###################

        self.last_dis = dis_to_g
        self.last_angle = self.angle
        sys.stdout.write(output)
        sys.stdout.flush()
        # self.total_step += 1
        self.pause_gym(True)
        
        return state, reward, terminated, truncated, self.info

    def reset(self, seed=None, options=None):
        print()
        self.step_count = 0
        self.wamv_forc = [0,0]
        ##################calculate success rate###################
        sr = sum(self.sr_queue.queue)/self.maxstep
        print(f"prev SR: {sr}")
        self.sr_queue.queue.clear()

        ##########################################################
        
        self.goal = [0, 0, 1.57]

        pose = PoseStamped()
        pose.header = Header()
        pose.header.frame_id = "map"
        pose.pose.position.x = self.goal[0]
        pose.pose.position.y = self.goal[1]

        r = R.from_euler('z', self.goal[2])
        quat = r.as_quat()
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        # self.pub_goal.publish(pose)
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_model(self.get_initial_state('wamv'))
        except(rospy.ServiceException) as e:
            print(e)

        # obstacle('totem').reset_pose(np.array([30, -30, 0]))
        
        
        ob = np.random.randint(3)
        if ob == 0:
            print("obstacle: polyform_red")
            obstacle('polyform_red').reset_pose(np.array([-30, 30, 1]))
            self.obstacle = obstacle('polyform_black')
        elif ob == 1:
            print("obstacle: polyform_black")
            obstacle('polyform_black').reset_pose(np.array([30, 30, 0]))
            self.obstacle = obstacle('polyform_red')
        else:
            print("obstacle: None")
            obstacle('polyform_red').reset_pose(np.array([-30, 30, 1]))
            obstacle('polyform_black').reset_pose(np.array([30, 30, 0]))
            self.obstacle = obstacle(None)

        if self.obstacle.name is not None:
            self.obstacle.reset()

        

        self.pause_gym(False)

        self.reward = 0
        self.last_dis = 0
        self.last_angle = 0
        self.total_dis = 0
        self.frame = 0
        self.epi += 1
        self.last_pos = None
        self.last_time = None
        self.state_stack = None
        self.pos_track = None
        self.velocity_track = None
        self.action_track = None
        self.velocity = 0
        self.reset_robot = True
        state = self.get_observation()
        
        return state, self.info

    def scan_once(self):
        data = LaserScan()
        try:
            data = rospy.wait_for_message(
                '/wamv/RL/scan', LaserScan, timeout=5)
        except:
            print('fail to receive message')

        ranges = np.array(data.ranges)
        ranges = np.clip(ranges, 0, self.max_dis)

        return ranges

    def get_observation(self):
        # obtain
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            agent = self.get_model('wamv', '')
        except (rospy.ServiceException) as e:
            print(e)
        new_pos = np.array(
            [agent.pose.position.x, agent.pose.position.y, agent.pose.position.z])
        if self.obstacle.name is not None:
            self.obstacle.update_pose()
        time = rospy.get_rostime()
        
        # add travel distance
        # if self.last_pos is not None:
        #     self.total_dis += np.linalg.norm(new_pos-self.last_pos)

        if self.last_time is not None and self.last_pos is not None and self.reset_robot==False:
            self.time_diff = (time.to_nsec()-self.last_time.to_nsec())/1000000000
            if self.time_diff == 0:
                self.time_diff = 0.1
            distance = math.sqrt((new_pos[0]-self.last_pos[0])**2+(new_pos[1]-self.last_pos[1])**2)
            self.velocity = distance/self.time_diff
            # print("velocity: ", self.velocity)

        self.last_time = time
        self.last_pos = new_pos

        self.reset_robot = False
        # caculate angle diff
        diff = self.goal[0:2] - self.last_pos[:2]
        r = R.from_quat([agent.pose.orientation.x,
                         agent.pose.orientation.y,
                         agent.pose.orientation.z,
                         agent.pose.orientation.w])
        yaw = r.as_euler('zyx')[0]
        self.angle = self.goal[2] - yaw
        if self.angle >= np.pi:
            self.angle -= 2*np.pi
        elif self.angle <= -np.pi:
            self.angle += 2*np.pi

        # update pose tracker
        goal_x_prime, goal_y_prime = self.map_to_model_frame(self.goal[:2], self.last_pos[:2], yaw)
        diff = np.array([goal_x_prime, goal_y_prime])
        track_pos = np.append(diff, self.angle)
        # track_pos = np.append(track_pos)
        if self.pos_track is None:
            self.pos_track = np.tile(track_pos, (self.track_num, 1))
        else:
            self.pos_track[:-1] = self.pos_track[1:]
            self.pos_track[-1] = track_pos
        
        self.velocity = np.array([self.velocity])
        # print("velocity", self.velocity)
        if self.velocity_track is None:
            self.velocity_track = np.tile(float(self.velocity), (self.track_num, 1))
        else:
            self.velocity_track[:-1] = self.velocity_track[1:]
            self.velocity_track[-1] = float(self.velocity)
        # print(self.velocity_track)
        # prepare laser scan stack
        last_action = np.array(self.last_action)
        if self.action_track is None:
            self.action_track = np.tile(last_action, (self.track_num, 1))
        else:
            self.action_track[:-1] = self.action_track[1:]
            self.action_track[-1] = last_action

        scan = self.scan_once()

        if self.state_stack is None:
            self.state_stack = np.tile(scan, (self.state_num, 1))
        else:
            self.state_stack[:-1] = self.state_stack[1:]
            self.state_stack[-1] = scan
        
        # print("vel:", self.velocity_track)
        # reshape
        # image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        # track = np.random.uniform(-1, 1, (10,))
        # print("laser shape: ", self.state_stack.shape)
        # print("track shape: ", self.pos_track.shape)
        # print("vel shape: ", self.velocity_track.shape)
        # print("action shape: ", self.action_track.shape)
        laser = self.state_stack # self.state_stack.reshape(-1)
        track = self.pos_track.reshape(-1)
        vel = self.velocity_track.reshape(-1)
        # action = self.action_track.reshape(-1)
        # print("after reshape")
        # print("laser shape: ", laser, end="\n\n")
        # print("track shape: ", track, end="\n\n")
        # print("vel shape: ", vel, end="\n\n")
        # print("action shape: ", action, end="\n\n")
        return {"laser": laser, "track": track, "vel": vel}

        # state = laser
        # state = np.append(state, track)
        # state = np.append(state, vel)
        # state = np.append(state, action)


        # return state

    def close(self):
        self.pause_gym(False)
        rospy.signal_shutdown('WTF')