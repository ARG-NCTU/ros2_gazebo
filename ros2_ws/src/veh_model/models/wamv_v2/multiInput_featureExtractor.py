from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == 'laser':
                # dimensions of laser scan is (4*241,) which is 1D feature vector.
                # print("laser shape: ", subspace.shape)
                # after conv1d, the shape should be 1D feature vector of size 64
                extractors[key] = nn.Sequential(
                        nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                total_concat_size += 32*241
            elif key == 'track' or key == 'velocity':
                # Run through nothing
                extractors[key] = nn.Sequential()
                total_concat_size += subspace.shape[0]
                #####print(f"key: {key}, shape: {subspace.shape}")
        ######print("Total concat size: ", total_concat_size)
        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        # self._features_dim = total_concat_size

        self.mlp_network = nn.Sequential(
            nn.Linear(total_concat_size, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self._features_dim = 256
        #####print("Total features dim: ", self._features_dim)



    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # print("start forward")
        # print(f"observations laser shape: {observations['laser'].shape}")
        # print(f"observations track shape: {observations['track'].shape}")
        # print(f"observations velocity shape: {observations['velocity'].shape}")
        # observations['laser'] = observations['laser'].reshape(964, 1)
        # observations['track'] = observations['track'].reshape(30, 1)
        # observations['velocity'] = observations['velocity'].reshape(10, 1)
        ## print(f"Observations['laser'].shape: {observations['laser'].shape}")
        ## print(f"Observations['track'].shape: {observations['track'].shape}")
        ## print(f"Observations['velocity'].shape: {observations['velocity'].shape}")
        ## self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            #####print("key: ", key)
            #####print("extractor: ", extractor)
            # if key = 'laser', the original size is (964, ) then reshap e to [1, 1, 964]
            tensor_append = extractor(observations[key])
            # if key == 'laser':
            #     tensor_append = tensor_append.view(128, 1)
            encoded_tensor_list.append(tensor_append)
            #####print("extracted_tensor shape: ", encoded_tensor_list[-1].shape)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        #####print("encoded_tensor_list: ", encoded_tensor_list)
        features = th.cat(encoded_tensor_list, dim=1)
        #####print("Features shape: ", features.shape)
        return self.mlp_network(features)
    