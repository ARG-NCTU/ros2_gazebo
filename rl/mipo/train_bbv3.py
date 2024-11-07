import sys, os, time, torch
import threading
import gymnasium as gym
import warnings
from sb3_arg.policy.mipo import MIPO
from sb3_arg.FeatureExtractor import BlueBoatFeatureExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from datetime import date
import numpy as np
import torch as th


warnings.filterwarnings("ignore")
# policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                      net_arch=[128, 256, 128],)
env = gym.make("gymnasium_arg:blueboat-v3", world='waves', veh='blueboat', max_thrust=10.0)
env = DummyVecEnv([lambda: env])

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],
    features_extractor_class=BlueBoatFeatureExtractor,
    features_extractor_kwargs=dict(hist_frame=50, imu_size=10, action_size=6, cmd_size=6, latent_dim=32),
)

today = date.today()
checkpoint_callback = CheckpointCallback(
  save_freq=100000,
  save_path="./logs/",
  name_prefix="dp_"+str(today),
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = MIPO(
    env=env,
    verbose=1, 
    policy_kwargs=policy_kwargs, 
    learning_rate=1e-6,
    batch_size=128,
    n_steps=4096,
    num_constraints=2,  # Pass the number of constraints to the policy
    constraint_thresholds=np.array([0.1, 0.1]),  # Initial thresholds d_k
    barrier_coefficient=100.0,                      # Hyperparameter t
    alpha=0.02,                                   # Hyperparameter α
    n_epochs=10,
    ent_coef=0.01,
    device='cuda',
    tensorboard_log='tb_mipo')

model.learn(total_timesteps=20_000_000, tb_log_name='tb_mipo', callback=checkpoint_callback)
model.save("ppo_blueboat_v3")
env.close()