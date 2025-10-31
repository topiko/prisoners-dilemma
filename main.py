import torch
from torch import nn

from env_utils.agents import JACK, JOY, Prisoner
from env_utils.env import get_pd_env
from env_utils.rollout import rollout


class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        hsize = 64
        self.rnn = nn.LSTM(3, hidden_size=hsize, batch_first=True)
        self.linear = nn.Linear(hsize, 2)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, h


if __name__ == "__main__":
    jack = Prisoner(JACK, Qnet())
    joy = Prisoner(JOY, Qnet())

    env = get_pd_env(max_cycles=10)

    rollout(env, jack, joy, max_cycles=10)

    qvals = torch.cat(jack.qvals, dim=1)

    rewards = torch.Tensor(jack.rewards())
    print(qvals, qvals.shape, jack.qvals[0].shape)
    print(rewards, rewards.shape)
    print(jack.get_returns(), jack.get_returns().shape)
