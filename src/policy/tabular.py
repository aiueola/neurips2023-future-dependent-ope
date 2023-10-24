from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class BaseTabularPolicy(metaclass=ABCMeta):
    @property
    @abstractmethod
    def policy(self):
        raise NotImplementedError

    def save(self, path: Path):
        np.save(path, self.q_table)

    def load(self, path: Path):
        self.q_table = np.load(path)

    def update(self, state: int, action: int, reward: int, next_state: int):
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[
            state, action
        ] + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]))

    def sample_action(self, state: int):
        return self.random_.choice(self.n_actions, p=self.policy[state])


@dataclass
class TabularEpsilonGreedyPolicy(BaseTabularPolicy):
    n_states: int
    n_actions: int
    epsilon: float = 0.0
    alpha: float = 0.9
    gamma: float = 1.0
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)
        self.action_matrix = np.eye(self.n_actions)
        self.q_table = self.random_.normal(size=(self.n_states, self.n_actions))

    @property
    def policy(self):
        greedy_action = self.q_table.argmax(axis=1)
        return (1 - self.epsilon) * self.action_matrix[
            greedy_action
        ] + self.epsilon / self.n_actions


@dataclass
class TabularSoftmaxPolicy(BaseTabularPolicy):
    n_states: int
    n_actions: int
    tau: float = 1.0
    alpha: float = 0.9
    gamma: float = 1.0
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)
        self.action_matrix = np.eye(self.n_actions)
        self.q_table = self.random_.normal(size=(self.n_states, self.n_actions))

    @property
    def policy(self):
        logit = self.tau * self.q_table
        logit = logit - logit.max()
        return np.exp(logit) / np.sum(np.exp(logit), axis=1, keepdims=True)
