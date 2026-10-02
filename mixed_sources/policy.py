"""Small permutation-equivariant candidate actor and pooled value network."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .environment import ACTION_DIM, GLOBAL_DIM, MAX_ACTIONS


def tensor_observations(observations):
    if isinstance(observations, dict):
        observations = [observations]
    return {
        key: torch.from_numpy(np.stack([o[key] for o in observations]))
        for key in ("global_features", "candidates", "mask")
    }


class CandidateActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = int(hidden)
        self.encoder = nn.Sequential(
            nn.Linear(ACTION_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden + GLOBAL_DIM, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden + GLOBAL_DIM, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def forward(self, observation):
        mask = observation["mask"].bool()
        encoded = self.encoder(observation["candidates"].float())
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp(min=1)
        context = torch.cat((observation["global_features"].float(), pooled), dim=-1)
        actor_input = torch.cat(
            (encoded, context[:, None, :].expand(-1, MAX_ACTIONS, -1)), -1
        )
        logits = self.actor(actor_input).squeeze(-1).masked_fill(~mask, -1e9)
        value = self.critic(context).squeeze(-1)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def choose(self, observation, deterministic=True):
        distribution, value = self(tensor_observations(observation))
        action = (
            distribution.logits.argmax(-1) if deterministic else distribution.sample()
        )
        return int(action[0]), float(value[0])


def save_policy(path, model, metadata):
    torch.save(
        dict(
            state_dict=model.state_dict(),
            hidden=model.hidden,
            metadata=metadata,
            schema=4,
        ),
        path,
    )


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != 4:
        raise ValueError("Q4 requires a Q4 checkpoint (schema 4), not a Q3 model")
    model = CandidateActorCritic(checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["metadata"]
