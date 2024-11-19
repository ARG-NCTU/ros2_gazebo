import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box


class USVFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, hist_frame: int = 50, imu_size: int = 10, action_size: int = 6, cmd_size: int = 6, refer_size: int = 4, latent_dim: int = 32, gru_hidden_size: int = 64):
        super(USVFeatureExtractor, self).__init__(observation_space, features_dim=latent_dim)
        
        # Save parameters as attributes
        self.hist_frame = hist_frame
        self.imu_size = imu_size
        self.action_size = action_size
        self.cmd_size = cmd_size
        self.refer_size = refer_size
        self.latent_dim = latent_dim
        self.gru_hidden_size = gru_hidden_size
        
        # Calculate lengths for slicing the flat observation vector
        self.imu_length = hist_frame * imu_size
        self.action_length = hist_frame * action_size
        self.cmd_length = cmd_size
        self.refer_length = refer_size
        
        # IMU GRU-based extractor
        self.imu_extractor = nn.GRU(input_size=imu_size, hidden_size=gru_hidden_size, batch_first=True)
        self.action_extractor = nn.GRU(input_size=action_size, hidden_size=gru_hidden_size, batch_first=True)

        # Ensure GRU layers are initialized in training mode
        self.imu_extractor.train()
        self.action_extractor.train()

        # Final linear layer to produce latent representation
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size * 2 + self.cmd_length + self.refer_length, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Force GRU to stay in training mode
        self.imu_extractor.train()
        self.action_extractor.train()
        # Extract IMU and action observations from the flat observation vector
        imu_end = self.imu_length
        action_end = imu_end + self.action_length
        cmd_end = action_end + self.cmd_length
        refer_end = cmd_end + self.refer_length

        imu_obs = observations[:, :imu_end].view(-1, self.hist_frame, self.imu_size)
        action_obs = observations[:, imu_end:action_end].view(-1, self.hist_frame, self.action_size)
        cmd_obs = observations[:, action_end:cmd_end]
        refer_obs = observations[:, cmd_end:refer_end]
        
        # Process IMU and action data using GRU
        _, imu_hidden = self.imu_extractor(imu_obs)  # imu_hidden shape: [1, batch_size, gru_hidden_size]
        _, action_hidden = self.action_extractor(action_obs)  # action_hidden shape: [1, batch_size, gru_hidden_size]
        
        imu_features = imu_hidden.squeeze(0)  # [batch_size, gru_hidden_size]
        action_features = action_hidden.squeeze(0)  # [batch_size, gru_hidden_size]

        # Concatenate features
        features = torch.cat((imu_features, action_features, cmd_obs, refer_obs), dim=1)

        # Pass through final layer
        latent_obs = self.fc(features)
        
        return latent_obs
