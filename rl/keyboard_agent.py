import sys, os, time
import pygame
import gymnasium as gym
import warnings
import numpy as np
import torch
import rclpy

warnings.filterwarnings("ignore")

num_envs = 10
env = gym.make("gymnasium_arg:blueboat-v1", num_envs=num_envs, world="waves", veh='blueboat', max_thrust=100.0)
env.reset()
while True:
    np.random.seed(0)
    actions = np.random.rand(num_envs, 6)
    # print("Actions: ", actions)
    obs, reward, terminated, truncated, info = env.step(actions)
    print('obs:', obs)
    time.sleep(0.1)
