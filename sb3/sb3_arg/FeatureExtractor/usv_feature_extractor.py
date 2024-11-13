import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box

class USVFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, hist_frame: int = 50, imu_size: int = 10, action_size: int = 6, cmd_size: int = 6, refer_size: int = 4, latent_dim: int = 32):
        super(USVFeatureExtractor, self).__init__(observation_space, features_dim=latent_dim)
        
        # Save parameters as attributes
        self.hist_frame = hist_frame
        self.imu_size = imu_size
        self.action_size = action_size
        self.cmd_size = cmd_size
        self.refer_size = refer_size
        self.latent_dim = latent_dim
        
        # Calculate lengths for slicing the flat observation vector
        self.imu_length = hist_frame * imu_size
        self.action_length = hist_frame * action_size
        self.cmd_length = cmd_size
        self.refer_length = refer_size
        
        # IMU processing
        self.imu_extractor = nn.Sequential(
            nn.Conv1d(self.imu_size, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        imu_input_shape = (self.hist_frame, self.imu_size)
        imu_flattened_dim = self._get_flattened_dim(imu_input_shape, self.imu_extractor)

        # Action processing
        self.action_extractor = nn.Sequential(
            nn.Conv1d(self.action_size, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        action_input_shape = (self.hist_frame, self.action_size)
        action_flattened_dim = self._get_flattened_dim(action_input_shape, self.action_extractor)
        
        # Final linear layer to produce latent representation
        self.fc = nn.Sequential(
            nn.Linear(imu_flattened_dim + action_flattened_dim + self.cmd_length + self.refer_length, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        )
    
    def _get_flattened_dim(self, input_shape, extractor):
        sample_input = torch.zeros(1, *input_shape).permute(0, 2, 1)  # Permute for Conv1d compatibility
        with torch.no_grad():
            return extractor(sample_input).view(-1).shape[0]

    def forward(self, observations):
        # Extract IMU and action observations from the flat observation vector
        imu_end = self.imu_length
        action_end = imu_end + self.action_length
        cmd_end = action_end + self.cmd_length
        refer_end = cmd_end + self.refer_length

        imu_obs = observations[:, :imu_end].view(-1, self.hist_frame, self.imu_size).permute(0, 2, 1)
        action_obs = observations[:, imu_end:action_end].view(-1, self.hist_frame, self.action_size).permute(0, 2, 1)
        cmd_obs = observations[:, action_end:cmd_end]
        refer_obs = observations[:, cmd_end:refer_end]
        
        imu_features = self.imu_extractor(imu_obs)
        action_features = self.action_extractor(action_obs)
        cmd_features = cmd_obs  # If you have processing layers for cmd_vel, apply them here
        refer_features = refer_obs
        
        # Concatenate and pass through final layer
        features = torch.cat((imu_features, action_features, cmd_features, refer_features), dim=1)
        latent_obs = self.fc(features)
        
        return latent_obs
