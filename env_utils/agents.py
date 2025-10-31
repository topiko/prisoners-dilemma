import numpy as np
import torch
from torch import nn

JACK = "jack"
JOY = "joy"


class Prisoner:
    def __init__(self, name: str, Qnet: nn.Module):
        if name not in {JACK, JOY}:
            raise ValueError(f"Invalid prisoner name: {name}")
        self.name = name
        self._observations: list[np.ndarray] = []
        self._rewards: list[float] = []
        self.qnet = Qnet
        self._h = None
        self.qvals = []

        self.reset()

    def action(self) -> int:
        # Opponent's last action is at index 1
        x = torch.Tensor([self.observations()[-1][1]]).long()
        x = nn.functional.one_hot(x, num_classes=3).unsqueeze(0).float()

        qval, self._h = self.qnet(x, self._h)

        self.qvals.append(qval)

        action = torch.argmax(qval, dim=-1).detach().item()

        return action

    def cur_obs(self, observation: list[np.ndarray]) -> None:
        self._observations.append(observation)

    def observations(self) -> list[np.ndarray]:
        return self._observations

    def cur_reward(self, reward: float) -> None:
        self._rewards.append(reward)

    def rewards(self) -> list[float]:
        return self._rewards

    def get_returns(self, gamma: float = 0.8) -> torch.Tensor:
        rewards = np.array(self.rewards())
        returns = torch.zeros(len(rewards))
        G = 0.0
        for i, r in enumerate(rewards[::-1]):
            G = r + gamma * G
            returns[-i - 1] = G

        return returns

    def reset(self):
        self._observations = []
        self._rewards = []
        self._h = None
        self.qvals = []
