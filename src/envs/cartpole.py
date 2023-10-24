import math
from typing import Any, Optional

import gym
from gym.spaces import Box
import numpy as np
from sklearn.utils import check_random_state


class PartiallyObservableCartpole(gym.Env):
    def __init__(
        self,
        memory_length: int,
        noise: float = 0.10,
        version: int = 0,
        apply_reward_engineering: bool = False,
        random_state: Optional[int] = None,
    ):
        self.memory_length = memory_length
        self.memory = None
        self.noise_param = noise
        self.env = gym.make(f"CartPole-v{version}")
        self.original_observation_space = self.env.observation_space
        low = np.tile(self.env.observation_space.low, self.memory_length + 1)
        high = np.tile(self.env.observation_space.high, self.memory_length + 1)
        self.observation_space = Box(low, high)
        self.apply_reward_engineering = apply_reward_engineering

        self.random_ = check_random_state(random_state)
        self.seed(random_state)

    def obs(self, state):
        return state + self.noise[self.t % 1000]

    def reward_engineering(
        self,
        state,
        action,
        done,
    ):
        x, x_dot, theta, theta_dot = state

        force = self.env.force_mag if action == 1 else -self.env.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.env.polemass_length * theta_dot * theta_dot * sintheta) / (
            self.env.masspole + self.env.masscart
        )
        thetaacc = (self.env.gravity * sintheta - costheta * temp) / (
            self.env.length
            * (
                4.0 / 3.0
                - self.env.masspole * costheta * costheta / self.env.total_mass
            )
        )
        xacc = (
            temp - self.env.polemass_length * thetaacc * costheta / self.env.total_mass
        )

        if self.env.kinematics_integrator == "euler":
            x = x + self.env.tau * x_dot
            x_dot = x_dot + self.env.tau * xacc
            theta = theta + self.env.tau * theta_dot
            theta_dot = theta_dot + self.env.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.env.tau * xacc
            x = x + self.env.tau * x_dot
            theta_dot = theta_dot + self.env.tau * thetaacc
            theta = theta + self.env.tau * theta_dot

        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4

        if not done:
            reward = (
                (2 - theta / theta_threshold_radians) * (2 - x / x_threshold) - 1
            ) / 2
        else:
            reward = -10.0
        return reward

    def step(self, action, return_memory_2d: bool = False, return_state: bool = False):
        self.t += 1
        state, reward, done, truncated, info = self.env.step(action)
        obs = self.obs(state)

        self.memory = np.roll(self.memory, -1, axis=0)
        self.memory[-1] = obs

        memory = self.memory if return_memory_2d else self.memory.flatten()

        # engineering the reward
        if self.apply_reward_engineering:
            reward = self.reward_engineering(state, action, done)

        # if done and self.t < 20:
        #     reward = -20.0
        # elif done:
        #     reward = 5.0
        # else:
        #     reward = 0.0

        # if done:
        #     memory = self.reset()

        if return_state:
            return (memory, state), reward, done, info
        else:
            return memory, reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        reset_memory: bool = False,
        return_memory_2d: bool = False,
        return_state: bool = False,
        return_info: bool = False,
    ):
        self.t = 0
        self.noise = self.random_.normal(loc=0.0, scale=self.noise_param, size=1000)

        state, info = self.env.reset(seed=seed)
        obs = self.obs(state)

        if reset_memory or self.memory is None:
            self.memory = np.tile(obs, (self.memory_length + 1, 1))
        else:
            self.memory = np.roll(self.memory, -1, axis=0)
            self.memory[-1] = obs

        memory = self.memory if return_memory_2d else self.memory.flatten()

        if return_state and return_info:
            return (memory, state), info
        elif return_state and not return_info:
            return (memory, state)
        elif not return_state and return_info:
            return memory, info
        elif not return_info and not return_state:
            return memory

    def seed(self, random_state: Optional[int] = None):
        self.random_ = check_random_state(random_state)

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def __getattr__(self, key) -> Any:
        return object.__getattribute__(self.env, key)
