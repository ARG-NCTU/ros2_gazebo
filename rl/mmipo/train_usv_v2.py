import sys, os, time, torch
import threading
import gymnasium as gym
import warnings
from sb3_arg.policy.mmipo import MMIPO
from sb3_arg.FeatureExtractor import USVFeatureExtractor, USVGRUExtractor, USVCNNExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from datetime import date
import numpy as np
import torch as th


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
env = gym.make("gymnasium_arg:usv-v2", world='lake', veh='wamv_v3', max_thrust=15*746/9.8, hist_frame=50)
# env = gym.make("gymnasium_arg:usv-v1", world='waves', veh='blueboat', max_thrust=10)
env = DummyVecEnv([lambda: env])

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],
    features_extractor_class=USVFeatureExtractor,
    features_extractor_kwargs=dict(hist_frame=50, imu_size=10, action_size=4, cmd_size=3, refer_size=3, latent_dim=6+128),
)

today = date.today()
checkpoint_callback = CheckpointCallback(
  save_freq=100000,
  save_path="./logs/",
  name_prefix="dp_"+str(today),
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = MMIPO(
    env=env,
    verbose=1, 
    policy_kwargs=policy_kwargs, 
    learning_rate=learning_rate_schedule,
    batch_size=128,
    n_steps=4096,
    num_constraints=1,  # Pass the number of constraints to the policy
    constraint_thresholds=np.array([0.3]),  # Initial thresholds d_k
    barrier_coefficient=1000.0,                      # Hyperparameter t
    alpha=0.02,                                   # Hyperparameter α
    n_epochs=10,
    ent_coef=0.01,
    device='cuda',
    tensorboard_log='tb_mmipo')

model.learn(total_timesteps=20_000_000, tb_log_name='tb_mmipo', callback=checkpoint_callback)
model.save("mmipo_wamv_v3")
env.close()