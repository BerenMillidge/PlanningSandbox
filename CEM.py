
import numpy as np
import torch
import torch.nn as nn

class CEMPlanner(nn.Module):
    def __init__(
        self,
        ensemble,
        reward_model,
        action_size,
        plan_horizon,
        optimisation_iters,
        n_candidates,
        top_candidates,
        use_reward=True,
        use_exploration=True,
        use_reward_info_gain=False,
        expl_scale=1,
        clamp_deltas=20,
        return_stats = False,
        device="cpu",
    ):
        super().__init__()
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.action_size = action_size
        self.ensemble_size = ensemble.ensemble_size

        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.n_candidates = n_candidates
        self.top_candidates = top_candidates

        self.use_reward = use_reward
        self.use_exploration = use_exploration
        self.expl_scale = expl_scale
        self.use_reward_info_gain = use_reward_info_gain
        self.device = device

        self.info_list = []
        self.reward_list = []
        self.reward_IG_list = []
        self.measure = InformationGain(self.ensemble)
        self.clamp_deltas = clamp_deltas
        self.return_stats = return_stats

    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.state_size = state.size(0)
        state = state.unsqueeze(dim=0).unsqueeze(dim=0)
        state = state.repeat(self.ensemble_size, self.n_candidates, 1)

        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(
            self.device
        )

        for _ in range(self.optimisation_iters):
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )

            states, delta_vars, delta_means = self.perform_rollout(state, actions)
            returns = torch.zeros(self.n_candidates).float().to(self.device)

            if self.use_exploration:
                expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale
                expl_bonus = expl_bonus.sum(dim=0)
                returns += expl_bonus

            if self.use_reward:
                if self.reward_model.ensemble_reward_model:
                    states = states.view(self.ensemble_size, -1, self.state_size)
                else:
                    states = states.view(-1, self.state_size)
                rewards = self.reward_model(states)

                self.rewards = rewards.view(
                    self.plan_horizon, self.ensemble_size, self.n_candidates
                )

                rewards = self.rewards.mean(dim=1).sum(dim=0)
                returns += rewards

            if self.use_reward_info_gain:
                if not self.use_reward:
                    if self.reward_model.ensemble_reward_model:
                        states = states.view(self.ensemble_size, -1, self.state_size)
                    else:
                        states = states.view(-1, self.state_size)
                    rewards = self.reward_model(states)

                    self.rewards = rewards.view(
                        self.plan_horizon, self.ensemble_size, self.n_candidates
                    )

                self.rewards = self.rewards.unsqueeze(3)
                reward_info_gain = torch.zeros([self.plan_horizon, self.n_candidates]).to(self.device)
                for t in range(self.plan_horizon):
                  reward_info_gain[t,:] = self.measure.entropy_of_average(self.rewards[t,:,:])
                #print("reward_info_gain: ", torch.sum(reward_info_gain,dim=0))
                reward_IG = torch.sum(reward_info_gain,dim=0)
                returns += reward_IG

            if self.return_stats:
                if self.use_reward:
                    self.reward_list.append(rewards)
                if self.use_exploration:
                    self.info_list.append(expl_bonus)
                if self.use_reward_info_gain:
                    self.reward_IG_list.append(reward_IG)


            returns = torch.where(
                torch.isnan(returns), torch.zeros_like(returns), returns
            )

            _, topk = returns.topk(
                self.top_candidates, dim=0, largest=True, sorted=False
            )

            best_actions = actions[:, topk.view(-1)].reshape(
                self.plan_horizon, self.top_candidates, self.action_size
            )

            action_mean, action_std_dev = (
                best_actions.mean(dim=1, keepdim=True),
                best_actions.std(dim=1, unbiased=False, keepdim=True),
            )

        if self.return_stats:
            reward_stats, info_stats,reward_IG_stats = self.get_stats()
            return action_mean[0].squeeze(dim=0), reward_stats, info_stats,reward_IG_stats

        return action_mean[0].squeeze(dim=0)

    def perform_rollout(self, current_state, actions):
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T
        states[0] = current_state

        actions = actions.unsqueeze(0)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.ensemble(states[t], actions[t])
            if self.clamp_deltas is not None:
                delta_mean = delta_mean.clamp(-self.clamp_deltas,self.clamp_deltas)
            states[t + 1] = states[t] + self.ensemble.sample(delta_mean, delta_var)
            # states[t + 1] = mean
            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        return states, delta_vars, delta_means

    def get_stats(self):
        reward_stats = {
        "max":0,
        "mean":0,
        "min":0,
        "std":0,
        }
        info_stats = {
        "max":0,
        "mean":0,
        "min":0,
        "std":0,
        }
        reward_IG_stats = {
        "max":0,
        "mean":0,
        "min":0,
        "std":0,
        }
        if self.use_reward:
            self.reward_list = torch.stack(self.reward_list).view(-1)
            reward_stats = {
                "max": self.reward_list.max().item(),
                "mean": self.reward_list.mean().item(),
                "min": self.reward_list.min().item(),
                "std": self.reward_list.std().item(),
            }
        if self.use_exploration:
            self.info_list = torch.stack(self.info_list).view(-1) * self.expl_scale
            info_stats = {
                "max": self.info_list.max().item(),
                "mean": self.info_list.mean().item(),
                "min": self.info_list.min().item(),
                "std": self.info_list.std().item(),
            }
        if self.use_reward_info_gain:
            self.reward_IG_list = torch.stack(self.reward_IG_list).view(-1) * self.expl_scale
            reward_IG_stats = {
                "max": self.reward_IG_list.max().item(),
                "mean": self.reward_IG_list.mean().item(),
                "min": self.reward_IG_list.min().item(),
                "std": self.reward_IG_list.std().item(),
            }
        self.info_list = []
        self.reward_list = []
        self.reward_IG_list = []
        return reward_stats, info_stats,reward_IG_stats
