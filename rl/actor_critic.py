# rl/actor_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim=256, action_dim=3):
        super().__init__()

        # actor 输出动作均值
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Sigmoid()  # 限制在 0~1
        )

        # critic 输出状态值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 动作方差参数（可学习 or 固定）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def act(self, state):
        """输出一个连续动作 + log_prob"""
        mean = self.actor_mean(state)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(0, 1), log_prob

    def evaluate(self, state):
        """返回状态值"""
        return self.critic(state)

    def save(self, path="rl_actor_critic.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path="rl_actor_critic.pt", map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()
