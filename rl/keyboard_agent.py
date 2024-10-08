import sys, os, time
import pygame
import gymnasium as gym
import warnings
import numpy as np
import torch
import rclpy


warnings.filterwarnings("ignore")

# Initialize pygame and set up a window for receiving input
# pygame.init()
# screen = pygame.display.set_mode((400, 300))  # Pygame window (400x300)
# pygame.display.set_caption("Control Lunar Lander")

# Set up task and environment
# task = task_config()
# task.headless = False
num_envs = 2
env = gym.make("gymnasium_arg:blueboat-v1", num_envs=num_envs, world="waves", veh='blueboat', max_thrust=100.0)
env.reset()
while True:
    np.random.seed(0)
    # actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    # actions = np.random.rand(6)
                        # [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    actions = np.random.rand(num_envs, 6)
    # print("Actions: ", actions)
    obs, reward, terminated, truncated, info = env.step(actions)
    time.sleep(0.1)
    # print("Reward: ", reward)
    # print("Terminated: ", terminated)
    # print("Truncated: ", truncated)
    # print("Info: ", info)
    # print("Observation: ", obs)