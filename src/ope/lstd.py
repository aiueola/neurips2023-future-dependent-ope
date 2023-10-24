"""Value-Based Estimator."""
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Tuple, Optional

import numpy as np

from policy.policy import BasePolicy
from .base import BaseLinearValueBasedOffPolicyEstimator


@dataclass
class TabularFutureIndependentLSTD(BaseLinearValueBasedOffPolicyEstimator):
    """This estimator is for the case of no-memory and MDP setting.

    e.g., :math:`M = \\phi, S = O`.

    """

    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    n_unique_states: int
    gamma: float = 1.0
    batch_size: int = 100

    def __post_init__(self):
        self.n_state_ids = self.n_unique_states
        self.state_embedding = np.eye(self.n_state_ids)

    def _encode(self, state_id: np.ndarray):
        return self.state_embedding[state_id]

    def _mu(self, state_id: np.ndarray, action_id: np.ndarray):
        state_id = state_id.reshape((-1, 1))
        action_id = action_id.reshape((-1, 1))
        behavior_policy = self.behavior_policy.calc_action_choice_prob_given_action(
            state_id,
            action_id,
            is_batch=True,
        )
        evaluation_policy = self.evaluation_policy.calc_action_choice_prob_given_action(
            state_id,
            action_id,
            is_batch=True,
        )
        return evaluation_policy / behavior_policy

    def _M1(
        self,
        state_id: np.ndarray,  # S
        action_id: np.ndarray,  # A
        reward: np.ndarray,  # R
    ):
        state = self._encode(state_id)
        mu = self._mu(state_id, action_id)

        M1 = (mu * reward)[:, np.newaxis] * state
        return M1.sum(axis=0)  # shape (1, d_S)

    def _M2(
        self,
        state_id: np.ndarray,  # S
        action_id: np.ndarray,  # A
        next_state_id: np.ndarray,  # S'
    ):
        state = self._encode(state_id)
        next_state = self._encode(next_state_id)
        mu = self._mu(state_id, action_id)

        M2 = (
            state[:, :, np.newaxis]
            @ (state - self.gamma * mu[:, np.newaxis] * next_state)[:, np.newaxis, :]
        )
        return M2.sum(axis=0)  # shape (d_S, d_S)

    def fit(
        self,
        state_id: np.ndarray,  # S
        action_id: np.ndarray,  # A
        next_state_id: np.ndarray,  # S'
        reward: np.ndarray,  # R
        **kwargs,
    ):
        n_samples = len(state_id)
        M1 = np.zeros((1, self.n_state_ids))
        M2 = np.zeros((self.n_state_ids, self.n_state_ids))

        for i in tqdm(
            np.arange((n_samples - 1) // self.batch_size + 1),
            desc="[fitting_value_bridge_function]",
            total=(n_samples - 1) // self.batch_size + 1,
        ):
            M1 = (
                M1
                + self._M1(
                    state_id=state_id[i * self.batch_size : (i + 1) * self.batch_size],
                    action_id=action_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                    reward=reward[i * self.batch_size : (i + 1) * self.batch_size],
                )
                / n_samples
            )

            M2 = (
                M2
                + self._M2(
                    state_id=state_id[i * self.batch_size : (i + 1) * self.batch_size],
                    action_id=action_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                    next_state_id=next_state_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                )
                / n_samples
            )

        self.value_bridge_function_coef = self._pseudo_inverse(M2) @ M1.T

    def _estimate_individual_value(
        self,
        initial_state_id: np.ndarray,  # F_0
    ):
        initial_state = self._encode(initial_state_id)
        return initial_state @ self.value_bridge_function_coef

    def estimate_policy_value(
        self,
        initial_state_id: np.ndarray,  # F_0
        **kwargs,
    ) -> float:
        return self._estimate_individual_value(
            initial_state_id=initial_state_id,
        ).mean()

    def estimate_confidence_interval(
        self,
        initial_state_id: np.ndarray,  # F_0
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        return_mean: bool = False,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[float]:
        estimates = self._estimate_individual_value(
            initial_state_id=initial_state_id,
        )
        return self.bootstrap_estimate(
            estimates=estimates,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            return_mean=return_mean,
            random_state=random_state,
        )


@dataclass
class TabularFutureDependentLSTD(BaseLinearValueBasedOffPolicyEstimator):
    """This estimator is for the case of no-memory setting.

    e.g., :math:`M_H = 1, M_F = 1, M = \\phi`.

    """

    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    n_unique_states: int
    n_unique_actions: int
    gamma: float = 1.0
    batch_size: int = 10

    def __post_init__(self):
        self.n_history_ids = self.n_unique_states * self.n_unique_actions
        self.n_state_ids = self.n_unique_states
        self.history_embedding = np.eye(self.n_history_ids)
        self.state_embedding = np.eye(self.n_state_ids)

    def _history_encode(
        self, history_state_id: np.ndarray, history_action_id: np.ndarray
    ):
        history_id = history_state_id * self.n_unique_actions + history_action_id
        return self.history_embedding[history_id]

    def _state_encode(self, state_id: np.ndarray):
        return self.state_embedding[state_id]

    def _mu(self, state_id: np.ndarray, action_id: np.ndarray):
        state_id = state_id.reshape((-1, 1))
        action_id = action_id.reshape((-1, 1))
        behavior_policy = self.behavior_policy.calc_action_choice_prob_given_action(
            state_id,
            action_id,
            is_batch=True,
        )
        evaluation_policy = self.evaluation_policy.calc_action_choice_prob_given_action(
            state_id,
            action_id,
            is_batch=True,
        )
        return evaluation_policy / behavior_policy

    def _M1(
        self,
        history_state_id: np.ndarray,  # H
        history_action_id: np.ndarray,  # H
        state_id: np.ndarray,  # O
        action_id: np.ndarray,  # A
        reward: np.ndarray,  # R
    ):
        history = self._history_encode(history_state_id, history_action_id)
        mu = self._mu(state_id, action_id)

        M1 = (mu * reward)[:, np.newaxis] * history
        return M1.sum(axis=0)  # shape (1, d_H)

    def _M2(
        self,
        history_state_id: np.ndarray,  # H
        history_action_id: np.ndarray,  # H
        state_id: np.ndarray,  # O
        action_id: np.ndarray,  # A
        next_state_id: np.ndarray,  # O'
    ):
        history = self._history_encode(history_state_id, history_action_id)
        state = self._state_encode(state_id)
        next_state = self._state_encode(next_state_id)
        mu = self._mu(state_id, action_id)

        M2 = (
            history[:, :, np.newaxis]
            @ (state - self.gamma * mu[:, np.newaxis] * next_state)[:, np.newaxis, :]
        )
        return M2.sum(axis=0)  # shape (d_H, d_F)

    def fit(
        self,
        history_state_id: np.ndarray,  # H
        history_action_id: np.ndarray,  # H
        state_id: np.ndarray,  # O
        action_id: np.ndarray,  # A
        next_state_id: np.ndarray,  # O'
        reward: np.ndarray,  # R
        **kwargs,
    ):
        n_samples = len(state_id)
        M1 = np.zeros((1, self.n_history_ids))
        M2 = np.zeros((self.n_history_ids, self.n_state_ids))

        for i in tqdm(
            np.arange((n_samples - 1) // self.batch_size + 1),
            desc="[fitting_value_bridge_function]",
            total=(n_samples - 1) // self.batch_size + 1,
        ):
            M1 = (
                M1
                + self._M1(
                    history_state_id=history_state_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                    history_action_id=history_action_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                    state_id=state_id[i * self.batch_size : (i + 1) * self.batch_size],
                    action_id=action_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                    reward=reward[i * self.batch_size : (i + 1) * self.batch_size],
                )
                / n_samples
            )

            M2 = (
                M2
                + self._M2(
                    history_state_id=history_state_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                    history_action_id=history_action_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                    state_id=state_id[i * self.batch_size : (i + 1) * self.batch_size],
                    action_id=action_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                    next_state_id=next_state_id[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ],
                )
                / n_samples
            )

        self.value_bridge_function_coef = self._pseudo_inverse(M2) @ M1.T

    def _estimate_individual_value(
        self,
        initial_state_id: np.ndarray,  # F_0
    ):
        initial_state = self._state_encode(initial_state_id)
        return initial_state @ self.value_bridge_function_coef

    def estimate_policy_value(
        self,
        initial_state_id: np.ndarray,  # F_0
        **kwargs,
    ) -> float:
        return self._estimate_individual_value(
            initial_state_id=initial_state_id,
        ).mean()

    def estimate_confidence_interval(
        self,
        initial_state_id: np.ndarray,  # F_0
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        return_mean: bool = False,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[float]:
        estimates = self._estimate_individual_value(
            initial_state_id=initial_state_id,
        )
        return self.bootstrap_estimate(
            estimates=estimates,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            return_mean=return_mean,
            random_state=random_state,
        )
