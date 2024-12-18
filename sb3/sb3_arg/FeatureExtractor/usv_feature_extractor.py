import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box


class USVFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, hist_frame: int = 50, imu_size: int = 10, action_size: int = 6, cmd_size: int = 6, latent_dim: int = 32):
        super(USVFeatureExtractor, self).__init__(observation_space, features_dim=latent_dim)
        
        # Save parameters as attributes
        self.hist_frame = hist_frame
        self.imu_size = imu_size
        self.action_size = action_size
        self.cmd_size = cmd_size
        self.latent_dim = latent_dim
        
        # Calculate lengths for slicing the flat observation vector
        self.imu_length = hist_frame * imu_size
        self.action_length = hist_frame * action_size
        self.cmd_length = cmd_size
        
        # IMU processing
        self.imu_extractor = nn.Sequential(
            nn.Linear(self.imu_length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Action processing
        self.action_extractor = nn.Sequential(
            nn.Linear(self.action_length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Final linear layer to produce latent representation
        self.fc = nn.Sequential(
            nn.Linear(64 + 64 + self.cmd_length, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        # Extract IMU and action observations from the flat observation vector
        imu_end = self.imu_length
        action_end = imu_end + self.action_length
        cmd_end = action_end + self.cmd_length

        imu_obs = observations[:, :imu_end]
        action_obs = observations[:, imu_end:action_end]
        cmd_obs = observations[:, action_end:cmd_end]
        
        imu_features = self.imu_extractor(imu_obs)
        action_features = self.action_extractor(action_obs)
        cmd_features = cmd_obs  # If you have processing layers for cmd_vel, apply them here
        
        # Concatenate and pass through final layer
        features = torch.cat((imu_features, action_features, cmd_features), dim=1)
        latent_obs = self.fc(features)
        
        return latent_obs
