import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim=256, action_dim=3):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Sigmoid()   # 输出范围 0~1
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def act(self, state):
        probs = self.actor(state)
        dist = torch.distributions.Normal(probs, torch.tensor([0.1, 0.1, 0.1]))
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.clamp(0, 1), log_prob

    def evaluate(self, state):
        return self.critic(state)
