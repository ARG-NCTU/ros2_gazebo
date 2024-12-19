import gymnasium as gym
import warnings
from sb3_arg.policy.mmipo import MMIPO
from sb3_arg.FeatureExtractor import USVFeatureExtractor, USVGRUExtractor, USVCNNExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium_arg.envs import MATH_USV_V1
from datetime import date
import numpy as np
import torch
import subprocess, os


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
initial_learning_rate = 1e-6  # Slightly lower learning rate for fine-tuning
learning_rate_schedule = linear_schedule(initial_learning_rate)

warnings.filterwarnings("ignore")

# Create vectorized environments for fine-tuning
env = gym.make("gymnasium_arg:usv-v2", world='lake', veh='wamv_v3', max_thrust=15*746/9.8, hist_frame=2)
env = DummyVecEnv([lambda: env])

# Define policy architecture
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],
    features_extractor_class=USVFeatureExtractor,
    features_extractor_kwargs=dict(hist_frame=2, imu_size=8, action_size=4, cmd_size=3, latent_dim=6+128),
)

# Callback for saving checkpoints during training
today = date.today()
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./logs/",
    name_prefix="mmipo_fine_tune_from_math_to_usv2_"+str(today),
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# Load the pre-trained model and attach the environment
model = MMIPO.load("./logs/mmipo_math_usv2024-12-18_13300000_steps.zip", env=env)

# Update parameters for fine-tuning (if necessary)
model.learning_rate = learning_rate_schedule
model.n_epochs = 5  # Adjust epochs if needed
model.batch_size = 128
model.tensorboard_log = 'tb_mmipo_fine_tuned'

pid = os.getpid()
print(f"{os.getpid()}  : Press Enter to continue...")
# sudo taskset -cp 0-5 <pid>
subprocess.run(["sudo", "taskset", "-cp", "0-5", str(pid)])

# Fine-tune the model
model.learn(
    total_timesteps=5_000_000, 
    tb_log_name='tb_mmipo_fine_tuned', 
    callback=checkpoint_callback
    )

# Save the fine-tuned model
model.save("mmipo_wamv_v3_fine_tuned")
env.close()
