from gymnasium import spaces
import numpy as np
import torch as th
from torch.nn import functional as F
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.buffers import RolloutBuffer
from collections import namedtuple


# Custom sample data structure
ConstrainedRolloutBufferSamples = namedtuple(
    'ConstrainedRolloutBufferSamples',
    ['observations', 'actions', 'old_values', 'old_log_prob', 'advantages', 'returns', 'constraint_costs']
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
        num_constraints
    ):
        # Set self.num_constraints before calling super().__init__()
        self.num_constraints = num_constraints

        # Call the parent constructor
        super(ConstrainedRolloutBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

        # Initialize constraint_costs
        self.constraint_costs = np.zeros(
            (self.buffer_size, self.n_envs, self.num_constraints), dtype=np.float32
        )

        self.obs_shape = observation_space.shape
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = 1
        else:
            self.action_dim = action_space.shape[0]

    def reset(self):
        super(ConstrainedRolloutBuffer, self).reset()
        # Re-initialize constraint_costs
        self.constraint_costs = np.zeros(
            (self.buffer_size, self.n_envs, self.num_constraints), dtype=np.float32
        )

    def add(self, *args, **kwargs):
        # Remove 'constraint_costs' from kwargs to prevent passing it to the base class
        constraint_costs = kwargs.pop('constraint_costs', None)
        # Call the base class add method with the remaining arguments
        super(ConstrainedRolloutBuffer, self).add(*args, **kwargs)
        # Store the constraint costs
        if constraint_costs is not None:
            self.constraint_costs[self.pos - 1] = constraint_costs

    def get(self, batch_size: Optional[int] = None):
        assert self.full, "Rollout buffer must be full before sampling"

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Prepare the data
        observations = self.observations.reshape((-1, *self.obs_shape))
        actions = self.actions.reshape((-1, self.action_dim))
        values = self.values.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        constraint_costs = self.constraint_costs.reshape(-1, self.num_constraints)

        # Convert to torch tensors
        observations = self.to_torch(observations)
        actions = self.to_torch(actions)
        values = self.to_torch(values)
        log_probs = self.to_torch(log_probs)
        advantages = self.to_torch(advantages)
        returns = self.to_torch(returns)
        constraint_costs = self.to_torch(constraint_costs)

        batch_size = self.buffer_size * self.n_envs if batch_size is None else batch_size

        for start_idx in range(0, self.buffer_size * self.n_envs, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            data = ConstrainedRolloutBufferSamples(
                observations=observations[batch_indices],
                actions=actions[batch_indices],
                old_values=values[batch_indices],
                old_log_prob=log_probs[batch_indices],
                advantages=advantages[batch_indices],
                returns=returns[batch_indices],
                constraint_costs=constraint_costs[batch_indices],
            )
            yield data

# Custom PPO algorithm
class IPO(PPO):
    def __init__(
        self,
        *args,
        constraint_threshold=0.1,
        lambda_constraint=0.1,
        num_constraints=2,
        **kwargs
    ):
        super(IPO, self).__init__(*args, **kwargs)
        self.constraint_threshold = constraint_threshold
        self.lambda_constraint = np.full(num_constraints, lambda_constraint, dtype=np.float32)
        self.num_constraints = num_constraints
        # Override the rollout buffer with the custom one
        self.rollout_buffer = ConstrainedRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
            num_constraints=self.num_constraints,  # Pass num_constraints here
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        # Initialize variables
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            # Sample action
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)

            actions = actions.cpu().numpy()
            # Rescale and clip action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Step the environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            # Retrieve constraint costs from infos
            constraint_costs = np.array([info.get('constraint_costs', np.zeros(self.num_constraints)) for info in infos])
            # Shape: (n_envs, num_constraints)

            # Store data in the buffer
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                constraint_costs=constraint_costs,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            n_steps += 1

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Get current clip range
        clip_range = self.clip_range
        if callable(clip_range):
            clip_range = clip_range(self._current_progress_remaining)

        entropy_losses = []
        all_kl_divs = []
        pg_losses, value_losses = [], []
        constraint_losses = []

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
                values = values.flatten()
                advantages = rollout_data.advantages

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Constraint costs (shape: [batch_size, num_constraints])
                constraint_costs = rollout_data.constraint_costs

                # Compute discounted constraint returns for each constraint
                constraint_returns = []
                for i in range(self.num_constraints):
                    returns_i = self._compute_discounted_returns(constraint_costs[:, i])
                    # Normalize constraint returns
                    returns_i = (returns_i - returns_i.mean()) / (returns_i.std() + 1e-8)
                    constraint_returns.append(returns_i)
                # Stack constraint returns (shape: [batch_size, num_constraints])
                constraint_returns = th.stack(constraint_returns, dim=1)

                # Policy loss
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                surr1 = ratio * advantages
                surr2 = th.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
                policy_loss = -th.min(surr1, surr2).mean()

                # Constraint loss
                # Compute constraint loss for each constraint
                constraint_loss = 0
                for i in range(self.num_constraints):
                    constraint_loss += self.lambda_constraint[i] * (ratio * constraint_returns[:, i]).mean()

                # Value loss
                values_pred = values
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + constraint_loss - self.ent_coef * entropy.mean()

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Update lambda_constraint for each constraint
                average_constraints = constraint_returns.mean(dim=0).detach().cpu().numpy()
                for i in range(self.num_constraints):
                    if average_constraints[i] > self.constraint_threshold:
                        self.lambda_constraint[i] += 0.01
                    else:
                        self.lambda_constraint[i] -= 0.01
                        self.lambda_constraint[i] = max(0.0, self.lambda_constraint[i])

                # Logging
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                constraint_losses.append(constraint_loss.item())
                entropy_losses.append(entropy.mean().item())
                approx_kl_divs.append(th.mean(log_prob - rollout_data.old_log_prob).detach().cpu().numpy())

        self._n_updates += self.n_epochs

        # Log training information
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        # self.logger.record("train/lambda_constraint", self.lambda_constraint)
        # self.logger.record("train/constraint_violation", average_constraints)
        for i in range(self.num_constraints):
            self.logger.record(f"train/constraint_{i}_lambda", self.lambda_constraint[i])
            self.logger.record(f"train/constraint_{i}_violations", average_constraints[i])
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/constraint_loss", np.mean(constraint_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/explained_variance", explained_var)


    def _compute_discounted_returns(self, rewards):
        discounted_returns = []
        r = 0
        gamma = self.gamma
        for reward in reversed(rewards.cpu().numpy()):
            r = reward + gamma * r
            discounted_returns.insert(0, r)
        return th.tensor(discounted_returns, dtype=th.float32, device=self.device)