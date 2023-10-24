from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from sklearn.utils import check_random_state

from d3rlpy.algos import AlgoBase

from .tabular import BaseTabularPolicy


@dataclass
class BasePolicy(metaclass=ABCMeta):
    @abstractmethod
    def sample_action(self, x: np.ndarray, is_batch: bool = False):
        raise NotImplementedError

    @abstractmethod
    def calc_action_choice_prob(self, x: np.ndarray, is_batch: bool = False):
        raise NotImplementedError

    def sample_action_with_action_choice_prob(
        self, x: np.ndarray, is_batch: bool = False
    ):
        action = self.sample_action(x, is_batch=is_batch)
        action_choice_prob = self.calc_action_choice_prob_given_action(
            x, action, is_batch=is_batch
        )
        return action, action_choice_prob

    def calc_action_choice_prob_given_action(
        self, x: np.ndarray, action: np.ndarray, is_batch: bool = False
    ):
        x = x if is_batch else x.reshape((1, -1))
        action = action if is_batch else action.reshape((1, -1))
        action_choice_prob = self.calc_action_choice_prob(x, is_batch=True)
        action_choice_prob = action_choice_prob[
            np.arange(len(action)), action.flatten()
        ]
        return action_choice_prob

    def predict(self, x: np.ndarray, is_batch: bool = False):
        x = x if is_batch else x.reshape((1, -1))
        action = self.base_policy.predict(x)
        return action if is_batch else action[0]

    def predict_value(
        self,
        x: np.ndarray,
        action: np.ndarray,
        with_std: bool = False,
        is_batch: bool = False,
    ):
        x = x if is_batch else x.reshape((1, -1))
        values = self.base_policy.predict_value(x, action, with_std)
        return values if is_batch else values[0]

    @property
    def encoder(self):
        return self.base_policy._impl._imitator._encoder.state_encoder


@dataclass
class EpsilonGreedyPolicy(BasePolicy):
    base_policy: Union[AlgoBase, BaseTabularPolicy]
    n_actions: int
    epsilon: float
    random_state: Optional[int] = None

    def __post_init__(self):
        self.action_matrix = np.eye(self.n_actions)
        self.random_ = check_random_state(self.random_state)

    def sample_action(self, x: np.ndarray, is_batch: bool = False):
        x = x if is_batch else x.reshape((1, -1))
        greedy_action = self.predict(x, is_batch=True)
        random_action = self.random_.randint(self.n_actions, size=len(x))
        greedy_mask = self.random_.binomial(1, 1 - self.epsilon, size=len(x))
        action = greedy_action * greedy_mask + random_action * (1 - greedy_mask)
        return action if is_batch else action[0]

    def calc_action_choice_prob(self, x: np.ndarray, is_batch: bool = False):
        x = x if is_batch else x.reshape((1, -1))
        greedy_action = self.predict(x, is_batch=True)
        greedy_action_matrix = self.action_matrix[greedy_action]
        uniform_matrix = np.ones_like(greedy_action_matrix, dtype=float)
        action_choice_prob = (1 - self.epsilon) * greedy_action_matrix + (
            self.epsilon / self.n_actions
        ) * uniform_matrix
        return action_choice_prob if is_batch else action_choice_prob[0]


@dataclass
class SoftmaxPolicy(BasePolicy):
    base_policy: AlgoBase
    n_actions: int
    tau: float = 1.0
    random_state: Optional[int] = None

    def __post_init__(self):
        self.tau = self.tau + 1e-10 if self.tau == 0 else self.tau
        self.random_ = check_random_state(self.random_state)

    def _softmax(self, x: np.ndarray):
        x = x - np.tile(np.max(x, axis=1), (x.shape[1], 1)).T  # to avoid overflow
        return np.exp(x / self.tau) / (
            np.sum(np.exp(x / self.tau), axis=1, keepdims=True)
        )

    def _gumble_max_trick(self, x: np.ndarray):
        gumble_variable = -np.log(-np.log(self.random_.rand(len(x), self.n_actions)))
        return np.argmax(x / self.tau + gumble_variable, axis=1).astype(int)

    def sample_action(self, x: np.ndarray, is_batch: bool = False):
        x = x if is_batch else x.reshape((1, -1))
        predicted_value = self.predict_value(x, is_batch=True)
        action = self._gumble_max_trick(predicted_value)
        return action if is_batch else action[0]

    def calc_action_choice_prob(self, x: np.ndarray, is_batch: bool = False):
        x = x if is_batch else x.reshape((1, -1))
        predicted_value = self.predict_value(x, is_batch=True)
        action_choice_prob = self._softmax(predicted_value)
        return action_choice_prob if is_batch else action_choice_prob[0]
