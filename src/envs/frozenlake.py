from typing import Optional, Any

import numpy as np
from sklearn.utils import check_random_state

import gym
from gym.spaces import Space


class PartiallyObservableFrozenLake(gym.Env):
    def __init__(
        self,
        memory_length: int,
        epsilon: float = 0.0,
        version: int = 1,
        random_state: Optional[int] = None,
    ):
        self.memory_length = memory_length
        self.memory = None
        self.epsilon = epsilon
        self.env = gym.make(f"FrozenLake-v{version}")
        self.action_space = self.env.action_space
        self.original_observation_space = self.env.observation_space
        self.observation_space = Space(shape=(self.memory_length + 1,), dtype=int)
        self.random_ = check_random_state(random_state)

    def noisy_mapping(self, state):
        x, y = self.decode(state)
        # add noise on taxi location
        x_ = (x + 1) % self.env.nrow
        y_ = (y + 1) % self.env.ncol
        return self.encode(x_, y_)

    def obs(self, state):
        return (
            self.noisy_mapping(state) if self.is_exploration[self.t % 1000] else state
        )

    def step(self, action, return_state: bool = False):
        self.t += 1
        state, reward, done, truncated, info = self.env.step(action)
        obs = self.obs(state)

        # engineering the reward
        if done and reward == 0:  # fallen into a hole
            reward = -10
        elif done and reward == 1:  # reach the goal
            reward = 5

        self.memory = np.roll(self.memory, -1)
        self.memory[-1] = obs

        if return_state:
            return (self.memory, state), reward, done, info
        else:
            return self.memory, reward, done, info

    def reset(
        self,
        seed: Optional[int] = None,
        reset_memory: bool = False,
        return_state: bool = False,
    ):
        self.random_ = check_random_state(seed)
        self.is_exploration = self.random_.binomial(1, self.epsilon, size=1000)
        self.t = 0

        state, info = self.env.reset(seed=seed)
        obs = self.obs(state)

        if reset_memory or self.memory is None:
            self.memory = np.full(self.memory_length + 1, obs)
        else:
            self.memory = np.roll(self.memory, -1)
            self.memory[-1] = obs

        if return_state:
            return (self.memory, state)
        else:
            return self.memory

    def encode(self, row, col):
        return row * self.env.ncol + col

    def decode(self, state_id):
        col = state_id % self.env.ncol
        row = (state_id - col) // self.env.ncol
        return row, col

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, random_state):
        self.random_ = check_random_state(random_state)

    def __getattr__(self, key) -> Any:
        return object.__getattribute__(self.env, key)
