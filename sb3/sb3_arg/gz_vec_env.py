import torch
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
import numpy as np
import gymnasium as gym


class GzVecWnv(VecEnv):
    def __init__(self, env):
        """
        Initialize the custom vectorized environment with tensor-based actions and observations.
        :param env: Single environment instance, which internally handles multiple agents.
        """
        self.env = env  # Single environment
        self.num_envs = env.sim_env.num_envs  # Number of agents in the environment
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # self.dones = torch.zeros(self.num_envs, dtype=torch.bool)
        self.device = env.device  # Assume all environments share the same device

        super().__init__(self.num_envs, self.observation_space, self.action_space)

    def step_async(self, actions: np.ndarray):
        """
        Step asynchronously with tensor actions.
        :param actions: Torch tensor actions to take for each environment (agent).
        """
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for all agents in the environment to finish their step and return the results as tensors.
        """
        # Perform the step with actions for all agents
        # if isinstance(self.actions, np.ndarray):
            # self.actions = torch.as_tensor(self.actions, device=self.device)
        state = self.env.step(actions=self.actions)

        # Get observations, rewards, terminations & truncation (done), and infos
        dones = (state[2] or state[3])
        for i in range(len(dones)):
            if dones[i]:
                self.env.reset_idx(i)
        observations = state[0]
        rewards = state[1]
        infos = state[4]

        # Convert observations to a dictionary format if observation space is a Dict
        # if isinstance(observations, torch.Tensor):
        #     observations = observations.cpu().numpy()
        # observations = {'observations': observations}

        # rewards = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
        # dones = dones.cpu().numpy() if isinstance(dones, torch.Tensor) else dones

        return observations, rewards, dones, infos


    def reset(self) -> dict:
        """
        Reset the environment for all agents and return the initial observations as tensors.
        """
        # self.dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        obs = self.env.reset()
        # for key, value in obs.items():
        #     if isinstance(value, torch.Tensor):
        #         print("Value is already a tensor")
        #         obs[key] = value.cpu().numpy()
            # obs[key] = torch.as_tensor(value, device=self.device)
        return obs

    def close(self):
        """
        Close the environment.
        """
        self.env.close()

    def render(self, mode="human"):
        """
        Render the environment.
        :param mode: Render mode.
        """
        self.env.render()

    def env_is_wrapped(self, wrapper_class, indices=None):
        """
        Check if the environment is wrapped with a specific wrapper class.
        Since there is only one environment, this always returns False.
        """
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call a method on the environment.
        Since there's only one environment, this simply calls the method on self.env.
        """
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]

    def get_attr(self, attr_name, indices=None):
        """
        Get an attribute from the environment.
        Since there's only one environment, this returns the attribute from self.env.
        """
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        """
        Set an attribute on the environment.
        Since there's only one environment, this sets the attribute on self.env.
        """
        setattr(self.env, attr_name, value)