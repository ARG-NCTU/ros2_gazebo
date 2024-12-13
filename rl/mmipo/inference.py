import numpy as np
import pygame
import gymnasium as gym
from sb3_arg.policy.mmipo import MMIPO
from stable_baselines3 import PPO
from sb3_arg.FeatureExtractor import USVFeatureExtractor, USVGRUExtractor, USVCNNExtractor


if __name__ == "__main__":
    env = gym.make("gymnasium_arg:mathusv-v1", render_mode="human", device='cuda')
    obs, info = env.reset()
    terminated = truncation = False
    model = MMIPO.load("uwe_2024-12-10_7500000_steps")
    while True:
        action, _states = model.predict(obs)
        obs, rew, terminated, truncation, info = env.step(action)
        env.render()
        if terminated or truncation:
            env.reset()
    env.close()