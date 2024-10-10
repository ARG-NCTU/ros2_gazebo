import sys, os, time, random
import pygame
import gymnasium as gym
import warnings
import numpy as np
import torch
import rclpy

warnings.filterwarnings("ignore")

env = gym.make("gymnasium_arg:blueboat-v1", world="waves", veh='blueboat', max_thrust=100.0)
env.reset()
np.random.seed(0)
while True:
    action = np.random.rand(6)
    print("Actions: ", action)
    obs, reward, terminated, truncated, info = env.step(action)
    # print('obs:', obs)
    # time.sleep(0.1)
