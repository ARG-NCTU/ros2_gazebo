import sys, os, time, torch
import threading
import gymnasium as gym
import warnings
from sb3_arg.policy.mmipo import MMIPO
from sb3_arg.FeatureExtractor import USVFeatureExtractor, USVGRUExtractor, USVCNNExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium_arg.envs import MATH_USV_V2
from datetime import date
import numpy as np
import torch as th
import subprocess


def make_env(render_mode="none"):
    """
    Helper function to create a new environment instance.
    """
    def _init():
        return MATH_USV_V2(render_mode=render_mode, hist_frame=2, device='cuda')
    return _init

def linear_schedule(initial_lr: float):
    """
    Linear learning rate schedule function.
    Args:
        initial_lr (float): The starting learning rate.
    Returns:
        Callable: A function that calculates the learning rate.
    """
    def lr_schedule(progress_remaining: float) -> float:
        # Decrease learning rate linearly
        return progress_remaining * initial_lr
    return lr_schedule

# Update learning rate to use the linear schedule
initial_learning_rate = 1e-5
learning_rate_schedule = linear_schedule(initial_learning_rate)

warnings.filterwarnings("ignore")

n_envs = 50

vec_env = make_vec_env(make_env(render_mode="none"), n_envs=n_envs)

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],
    features_extractor_class=USVFeatureExtractor,
    features_extractor_kwargs=dict(hist_frame=2, imu_size=8, action_size=4, cmd_size=3, latent_dim=128),
)

today = date.today()
checkpoint_callback = CheckpointCallback(
  save_freq=100000,
  save_path="./logs/",
  name_prefix="mmipo_math_usv"+str(today),
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = MMIPO(
    env=vec_env,
    verbose=1, 
    policy_kwargs=policy_kwargs, 
    learning_rate=learning_rate_schedule,
    batch_size=128,
    n_steps=4096,
    num_constraints=1,  # Pass the number of constraints to the policy
    constraint_thresholds=np.array([0.5]),  # Initial thresholds d_k
    barrier_coefficient=100.0,                      # Hyperparameter t
    alpha=0.02,                                   # Hyperparameter α
    n_epochs=10,
    ent_coef=0.01,
    device='cuda',
    tensorboard_log='tb_mmipo')

pid = os.getpid()
print(f"{os.getpid()}  : Press Enter to continue...")
# sudo taskset -cp 0-5 <pid>
subprocess.run(["sudo", "taskset", "-cp", "0-5", str(pid)])

model.learn(total_timesteps=100_000_000, tb_log_name='tb_mmipo', callback=checkpoint_callback)
model.save("mmipo_math_usv")
vec_env.close()