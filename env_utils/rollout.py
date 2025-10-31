"""Utility helpers for running policy rollouts in the Prisoner's Dilemma env."""

from __future__ import annotations

from pettingzoo.utils.env import ParallelEnv

from env_utils.agents import Prisoner


def rollout(
    env: ParallelEnv,
    jack: Prisoner,
    joy: Prisoner,
    max_cycles: int | None = None,
):
    obs, _ = env.reset()

    for _ in range(max_cycles):
        jack.cur_obs(obs[jack.name])

        # The observation is [jack, joy], so reverse for joy
        joy.cur_obs(obs[joy.name][::-1])

        actions = {
            jack.name: jack.action(),
            joy.name: joy.action(),
        }
        obs, rewards, terminations, truncations, _ = env.step(actions)
        jack.cur_reward(rewards[jack.name])
        joy.cur_reward(rewards[joy.name])

        if terminations[jack.name] or truncations[jack.name]:
            break
