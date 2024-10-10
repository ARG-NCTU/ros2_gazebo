from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BB_V1_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 86):
        super(BB_V1_Extractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        hist = features_dim - 6
        extractors['cmd_vel'] = nn.Sequential(
            nn.ReLU(),
        )
        extractors['hist'] = nn.Sequential(
            nn.vae(),
        )
        extractors['observations'] = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
        )
        total_concat_size += 256
        # for key, subspace in observation_space.spaces.items():
        #     # if key == 'laser':
        #     #     # We will just downsample one channel of the laser by 4x241 and flatten.
        #     #     # Assume the laser is single-channel (subspace.shape[0] == 0)
        #     #     extractors[key] = nn.Sequential(
        #     #             nn.Conv1d(subspace.shape[0], 32, kernel_size=3, stride=1),
        #     #             nn.ReLU(),
        #     #             nn.Conv1d(32, 32, kernel_size=3, stride=1),
        #     #             nn.ReLU(),
        #     #             nn.Flatten(),
        #     #         )
        #     #     # print("laser", subspace.shape)
        #     #     total_concat_size += 7584
        #     if key == 'observations' or key == 'vel' or key == 'action':
        #         # Run through nothing
        #         extractors[key] = nn.Sequential()
        #         total_concat_size += subspace.shape[0]
        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        # self._features_dim = total_concat_size
        self.mlp_network = nn.Sequential(
            nn.Linear(total_concat_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
        )
        self._features_dim = 64



    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        cmd_vel = observations[:5]
        hist = observations[5:]
        encoded_tensor_list.append(self.extractors['observations'](observations))
        # for key, extractor in self.extractors.items():
        #     encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        features = th.cat(encoded_tensor_list, dim=1)
        return self.mlp_network(features)
    