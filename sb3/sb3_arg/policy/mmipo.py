import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from typing import Optional, Union, Type, List, Dict, Tuple, Any
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from collections import namedtuple


# Custom policy with shared value network and multiple cost heads
class MMIPOActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule,
        net_arch: Optional[Union[List[Union[int, Dict[str, List[int]]]], Dict[str, List[int]]]] = None,
        num_constraints: int = 2,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        self.num_constraints = num_constraints
        super(MMIPOActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the shared feature extractor, policy network, and value network with multiple heads.
        """
        self.mlp_extractor = SharedValueNetwork(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            num_constraints=self.num_constraints,
        )
        self.latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.latent_dim_vf = self.mlp_extractor.latent_dim_vf

    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.
        """
        # Call the base class _build method
        super()._build(lr_schedule)

        # Overwrite value_net to use our custom reward head
        self.value_net = self.mlp_extractor.reward_head  # Use the reward head as value_net

    def get_cost_values(self, obs: th.Tensor) -> List[th.Tensor]:
        """
        Compute the cost values for given observations.

        :param obs: Observations
        :return: List of cost values
        """
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        cost_values = [head(latent_vf) for head in self.mlp_extractor.cost_heads]
        return cost_values

# Custom shared value network with multiple heads
class SharedValueNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Optional[Union[List[Union[int, Dict[str, List[int]]]], Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        num_constraints: int = 2,
    ):
        super(SharedValueNetwork, self).__init__()
        self.num_constraints = num_constraints

        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        # Build shared network
        shared_layers = []
        last_layer_dim_shared = feature_dim

        policy_layers_sizes = []
        value_layers_sizes = []

        if isinstance(net_arch, dict):
            # net_arch is a dict with 'pi' and 'vf' keys
            policy_layers_sizes = net_arch.get('pi', [])
            value_layers_sizes = net_arch.get('vf', [])
        elif isinstance(net_arch, list):
            for idx, layer in enumerate(net_arch):
                if isinstance(layer, int):
                    shared_layers.append(nn.Linear(last_layer_dim_shared, layer))
                    shared_layers.append(activation_fn())
                    last_layer_dim_shared = layer
                elif isinstance(layer, dict):
                    policy_layers_sizes = layer.get('pi', [])
                    value_layers_sizes = layer.get('vf', [])
                    break
                else:
                    raise ValueError(f"Invalid layer type: {type(layer)}")
        else:
            raise ValueError(f"Invalid net_arch type: {type(net_arch)}")

        self.shared_net = nn.Sequential(*shared_layers)

        # Build policy network
        last_layer_dim_pi = last_layer_dim_shared
        policy_layers = []
        for layer_size in policy_layers_sizes:
            policy_layers.append(nn.Linear(last_layer_dim_pi, layer_size))
            policy_layers.append(activation_fn())
            last_layer_dim_pi = layer_size

        self.policy_net = nn.Sequential(*policy_layers)

        # Build value network with multiple heads (one for reward, others for costs)
        last_layer_dim_vf = last_layer_dim_shared
        value_layers = []
        for layer_size in value_layers_sizes:
            value_layers.append(nn.Linear(last_layer_dim_vf, layer_size))
            value_layers.append(activation_fn())
            last_layer_dim_vf = layer_size

        self.value_net = nn.Sequential(*value_layers)
        # Heads for reward and cost values
        self.reward_head = nn.Linear(last_layer_dim_vf, 1)
        self.cost_heads = nn.ModuleList(
            [nn.Linear(last_layer_dim_vf, 1) for _ in range(self.num_constraints)]
        )

        # Set latent dimensions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        shared_features = self.shared_net(features)
        policy_features = self.policy_net(shared_features)
        value_features = self.value_net(shared_features)
        return policy_features, value_features

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Extract policy features."""
        policy_features, _ = self.forward(features)
        return policy_features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Extract value features."""
        _, value_features = self.forward(features)
        return value_features

# Custom sample data structure
ConstrainedRolloutBufferSamples = namedtuple(
    'ConstrainedRolloutBufferSamples',
    [
        'observations',
        'actions',
        'old_values',
        'old_log_prob',
        'advantages',
        'returns',
        'constraint_returns',
        'old_cost_values',
        'cost_advantages',
    ]
)

# Custom rollout buffer
class ConstrainedRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        gae_lambda,
        gamma,
        n_envs,
        num_constraints,
    ):
        # Set num_constraints and cost_gae_lambda before calling super().__init__()
        self.num_constraints = num_constraints
        self.cost_gae_lambda = gae_lambda

        # Initialize cost-related buffers to None
        self.cost_rewards = None
        self.cost_values = None
        self.cost_returns = None
        self.cost_advantages = None

        # Call the base class constructor
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

    def reset(self):
        super().reset()
        if self.cost_rewards is None:
            # Initialize cost buffers
            self.cost_rewards = np.zeros((self.buffer_size, self.n_envs, self.num_constraints), dtype=np.float32)
            self.cost_values = np.zeros((self.buffer_size, self.n_envs, self.num_constraints), dtype=np.float32)
            self.cost_returns = np.zeros((self.buffer_size, self.n_envs, self.num_constraints), dtype=np.float32)
            self.cost_advantages = np.zeros((self.buffer_size, self.n_envs, self.num_constraints), dtype=np.float32)
        else:
            # Reset cost buffers
            self.cost_rewards.fill(0)
            self.cost_values.fill(0)
            self.cost_returns.fill(0)
            self.cost_advantages.fill(0)

    def add(
        self,
        *args,
        cost_rewards=None,
        cost_values=None,
        **kwargs,
    ):
        super().add(*args, **kwargs)
        self.cost_rewards[self.pos - 1] = cost_rewards.copy()
        self.cost_values[self.pos - 1] = cost_values.copy()

    def compute_returns_and_advantage(self, last_values, dones):
        super().compute_returns_and_advantage(last_values, dones)
        last_cost_values = self.cost_values[-1]
        for idx in range(self.num_constraints):
            last_cost_value = last_cost_values[:, idx]
            # Convert to numpy
            last_cost_value = last_cost_value.copy()
            # Compute GAE for cost functions
            gae = np.zeros_like(last_cost_value)
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_values = last_cost_value
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_values = self.cost_values[step + 1, :, idx]
                delta = (
                    self.cost_rewards[step][:, idx]
                    + self.gamma * next_values * next_non_terminal
                    - self.cost_values[step][:, idx]
                )
                gae = delta + self.gamma * self.cost_gae_lambda * next_non_terminal * gae
                self.cost_advantages[step][:, idx] = gae
                self.cost_returns[step][:, idx] = gae + self.cost_values[step][:, idx]

    def get(self, batch_size: Optional[int] = None):
        assert self.full, "Rollout buffer must be full before sampling"

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Prepare the data
        observations = self.observations.reshape((-1, *self.observations.shape[2:]))
        actions = self.actions.reshape((-1, self.actions.shape[-1]))
        old_values = self.values.reshape(-1)
        old_log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)

        # Cost-related data
        cost_advantages = self.cost_advantages.reshape((-1, self.num_constraints))
        cost_returns = self.cost_returns.reshape((-1, self.num_constraints))
        old_cost_values = self.cost_values.reshape((-1, self.num_constraints))

        # Convert to torch tensors
        observations = th.as_tensor(observations).to(self.device)
        actions = th.as_tensor(actions).to(self.device)
        old_values = th.as_tensor(old_values).to(self.device)
        old_log_probs = th.as_tensor(old_log_probs).to(self.device)
        advantages = th.as_tensor(advantages).to(self.device)
        returns = th.as_tensor(returns).to(self.device)
        cost_advantages = th.as_tensor(cost_advantages).to(self.device)
        cost_returns = th.as_tensor(cost_returns).to(self.device)
        old_cost_values = th.as_tensor(old_cost_values).to(self.device)

        batch_size = self.buffer_size * self.n_envs if batch_size is None else batch_size

        for start_idx in range(0, self.buffer_size * self.n_envs, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            data = ConstrainedRolloutBufferSamples(
                observations=observations[batch_indices],
                actions=actions[batch_indices],
                old_values=old_values[batch_indices],
                old_log_prob=old_log_probs[batch_indices],
                advantages=advantages[batch_indices],
                returns=returns[batch_indices],
                cost_advantages=cost_advantages[batch_indices],
                constraint_returns=cost_returns[batch_indices],
                old_cost_values=old_cost_values[batch_indices],
            )
            yield data

# Custom PPO algorithm implementing Modified IPO
class MMIPO(PPO):
    def __init__(
        self,
        *args,
        policy = MMIPOActorCriticPolicy,
        num_constraints: int = 0,
        alpha: float = 0.1,
        barrier_coefficient: float = 100.0,
        constraint_thresholds: Optional[np.ndarray] = None,
        **kwargs
    ):
        '''
        policy_kwargs:
            num_constraints: int
                Number of constraints
            alpha: float
                Constraint threshold update rate
            barrier_coefficient: float
                Barrier coefficient for the barrier function
            constraint_thresholds: np.ndarray
                Initial constraint thresholds
        '''
        self.num_constraints = num_constraints
        self.alpha = alpha
        self.barrier_coefficient = barrier_coefficient
        constraint_thresholds = constraint_thresholds

        # Pass num_constraints to policy_kwargs
        policy_kwargs = kwargs.get("policy_kwargs", {})
        policy_kwargs["num_constraints"] = self.num_constraints
        kwargs["policy_kwargs"] = policy_kwargs

        super(MMIPO, self).__init__(*args, policy=MMIPOActorCriticPolicy, **kwargs)
        # Set default thresholds if not provided
        if constraint_thresholds is None:
            constraint_thresholds = np.array([0.1] * self.num_constraints)
        self.initial_constraint_thresholds = constraint_thresholds  # d_k
        self.dynamic_constraint_thresholds = np.copy(constraint_thresholds)  # d_k^i
        # Override the rollout buffer with the custom one
        self.rollout_buffer = ConstrainedRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
            num_constraints=self.num_constraints,
        )
        # Ensure policy is our custom policy
        assert isinstance(self.policy, MMIPOActorCriticPolicy), "Policy must be MIPOActorCriticPolicy"

    def collect_rollouts(self, env: VecEnv, callback, rollout_buffer, n_rollout_steps):
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
                # Get cost value predictions
                cost_values = self.policy.get_cost_values(obs_tensor)
                cost_values_cpu = th.cat(cost_values, dim=1).cpu().numpy()  # Shape: (n_envs, num_constraints)

            actions_cpu = actions.cpu().numpy()

            # Clip actions (if needed)
            clipped_actions = actions_cpu
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions_cpu, self.action_space.low, self.action_space.high)

            # Step environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # Handle constraint costs
            cost_rewards = np.array([info.get('constraint_costs', np.zeros(self.num_constraints)) for info in infos])

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            # Add to rollout buffer
            rollout_buffer.add(
                self._last_obs,
                actions_cpu,
                rewards,
                self._last_episode_starts,
                values,       # Pass values as PyTorch tensor
                log_probs,    # Pass log_probs as PyTorch tensor
                cost_rewards=cost_rewards,
                cost_values=cost_values_cpu,  # Pass cost_values as NumPy array
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            n_steps += 1

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)
            # Get last cost values
            cost_values = self.policy.get_cost_values(obs_tensor)
            cost_values = th.cat(cost_values, dim=1)

        rollout_buffer.compute_returns_and_advantage(values, dones)
        callback.on_rollout_end()
        return True


    def train(self) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Get current learning rate
        current_lr = self.policy.optimizer.param_groups[0]['lr']

        # Get current clip range
        clip_range = self.clip_range
        if callable(clip_range):
            clip_range = clip_range(self._current_progress_remaining)

        entropy_losses = []
        all_kl_divs = []
        pg_losses, value_losses = [], []
        barrier_losses = []
        cost_value_losses = []

        # Compute cumulative constraint costs J_C_k(π_i) over the entire buffer
        cost_rewards_buffer = self.rollout_buffer.cost_rewards.reshape(-1, self.num_constraints)
        mean_cost_rewards = cost_rewards_buffer.mean(axis=0)  # Shape: [num_constraints]

        # Compute J_C_k(π_i) = E[C_k(s,a,s')]/(1 - gamma)
        J_C_k_pi_i = mean_cost_rewards / (1 - self.gamma)

        # Update dynamic constraint thresholds
        for k in range(self.num_constraints):
            self.dynamic_constraint_thresholds[k] = max(
                self.initial_constraint_thresholds[k],
                J_C_k_pi_i[k] + self.alpha * self.initial_constraint_thresholds[k]
            )

        # Train for n_epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                # Evaluate actions
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                # Get cost value predictions
                cost_value_preds = self.policy.get_cost_values(rollout_data.observations)
                cost_values = th.cat(cost_value_preds, dim=1)  # Shape: [batch_size, num_constraints]

                values = values.flatten()
                advantages = rollout_data.advantages
                cost_advantages = rollout_data.cost_advantages  # Shape: [batch_size, num_constraints]

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                for i in range(self.num_constraints):
                    cost_advantages[:, i] = (cost_advantages[:, i] - cost_advantages[:, i].mean()) / (
                        cost_advantages[:, i].std() + 1e-8
                    )

                # Policy loss
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                surr1 = ratio * advantages
                surr2 = th.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
                policy_loss = -th.min(surr1, surr2).mean()

                # Value loss
                values_pred = values
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Cost value losses
                cost_value_loss = 0
                for i in range(self.num_constraints):
                    cost_values_pred = cost_values[:, i]
                    cost_returns = rollout_data.constraint_returns[:, i]
                    cost_value_loss += F.mse_loss(cost_returns, cost_values_pred)

                # Barrier function
                # Compute cumulative cost returns for the batch
                cumulative_constraint_costs = rollout_data.constraint_returns.mean(dim=0)  # Shape: [num_constraints]
                barrier_terms = []
                for i in range(self.num_constraints):
                    # d_k^i - J_C_k(π_i)
                    barrier_argument = self.dynamic_constraint_thresholds[i] - cumulative_constraint_costs[i]
                    # To prevent log(0) or negative values, ensure the argument is positive
                    epsilon = 1e-8
                    barrier_argument = th.clamp(barrier_argument, min=epsilon)
                    # log(d_k^i - J_C_k(π_i)) / t
                    barrier_term = th.log(barrier_argument)/self.barrier_coefficient
                    barrier_terms.append(barrier_term)
                # Sum up barrier terms
                barrier_loss = -th.sum(th.stack(barrier_terms))

                # Entropy loss
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -log_prob.mean()
                else:
                    entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.vf_coef * cost_value_loss + barrier_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Logging
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                cost_value_losses.append(cost_value_loss.item())
                barrier_losses.append(barrier_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kl_divs.append(th.mean(log_prob - rollout_data.old_log_prob).detach().cpu().numpy())

            self._n_updates += self.n_epochs

            # Log training information
            explained_var = explained_variance(
                self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
            )
            self.logger.record("train/barrier_loss", np.mean(barrier_losses))
            self.logger.record("train/policy_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/mean_reward", np.mean(self.rollout_buffer.rewards))
            self.logger.record("train/learning_rate", current_lr)
            for i in range(self.num_constraints):
                self.logger.record(f"train/dynamic_threshold_{i}", self.dynamic_constraint_thresholds[i])

    def _update_learning_rate(self, optimizer):
        # Update the optimizer's learning rate
        lr = self.lr_schedule(self._current_progress_remaining)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr