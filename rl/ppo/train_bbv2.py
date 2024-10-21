import sys, os, time, torch
import threading
import gymnasium as gym
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from datetime import date
import torch

if torch.cuda.is_available():
    torch.cuda.set_device(0)
warnings.filterwarnings("ignore")
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[128, 256, 128],)
today = date.today()
checkpoint_callback = CheckpointCallback(
  save_freq=100000,
  save_path="./logs/",
  name_prefix="bb_"+str(today),
  save_replay_buffer=True,
  save_vecnormalize=True,
)
num_envs = 1
headless = False
vec_env = make_vec_env("gymnasium_arg:blueboat-v2", n_envs=num_envs)
model = PPO("MlpPolicy", vec_env,
            verbose=1, 
            policy_kwargs=policy_kwargs, 
            learning_rate=1e-6,
            batch_size=128,
            n_steps=4096,
            n_epochs=10,
            ent_coef=0.01,
            device='cuda',
            tensorboard_log='tb_ppo')

model.learn(total_timesteps=20_000_000, tb_log_name='tb_ppo', callback=checkpoint_callback)
model.save("ppo_blueboat_v2")