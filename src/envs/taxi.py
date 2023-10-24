from typing import Optional, Any

import numpy as np
from sklearn.utils import check_random_state

import gym
from gym.spaces import Space


class PartiallyObservableTaxi(gym.Env):
    def __init__(
        self,
        memory_length: int,
        epsilon: float = 0.0,
        version: int = 3,
        random_state: Optional[int] = None,
    ):
        self.memory_length = memory_length
        self.memory = None
        self.epsilon = epsilon
        self.env = gym.make(f"Taxi-v{version}")
        self.action_space = self.env.action_space
        self.original_observation_space = self.env.observation_space
        self.observation_space = Space(shape=(self.memory_length + 1,), dtype=int)
        self.random_ = check_random_state(random_state)

    def noisy_mapping(self, state):
        x, y, passenger_location, destination = self.decode(state)
        # add noise on taxi location
        x_ = (x + 1) % 5
        y_ = (y + 1) % 5
        return self.encode(x_, y_, passenger_location, destination)

    def obs(self, state):
        return (
            self.noisy_mapping(state) if self.is_exploration[self.t % 1000] else state
        )

    def step(self, action, return_state: bool = False):
        self.t += 1
        state, reward, done, truncated, info = self.env.step(action)
        obs = self.obs(state)

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

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        return self.env.encode(taxi_row, taxi_col, pass_loc, dest_idx)

    def decode(self, state_id):
        return self.env.decode(state_id)

    def action_mask(self, state_id):
        return self.env.action_mask(state_id)

    def render(self):
        self.env.render()

    def get_surf_loc(self, map_loc):
        self.env.get_surf_loc(map_loc)

    def close(self):
        self.env.close()

    def seed(self, random_state):
        self.random_ = check_random_state(random_state)

    def __getattr__(self, key) -> Any:
        return object.__getattribute__(self.env, key)
