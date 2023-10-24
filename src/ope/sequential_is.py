"""Sequential IS Estimator."""
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from policy.policy import BasePolicy
from .base import BaseOffPolicyEstimator


@dataclass
class TabularSequentialImportanceSampling(BaseOffPolicyEstimator):
    """This estimator is for the case of no-memory setting for Taxi.

    e.g., :math:`M_H = 1, M = 0, M_F = 1`.

    Otherwise, non-tabular setting should be used.

    """

    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    gamma: float = 1.0

    def _estimate_individual_value(
        self,
        trajectory_state_ids: np.ndarray,
        trajectory_action_ids: np.ndarray,
        trajectory_rewards: np.ndarray,
    ):
        n_trajectories = len(trajectory_state_ids)
        behavior_policy_ = self.behavior_policy.calc_action_choice_prob_given_action(
            trajectory_state_ids.reshape((-1, 1)),
            trajectory_action_ids.reshape((-1, 1)),
            is_batch=True,
        ).reshape((n_trajectories, -1))
        evaluation_policy_ = (
            self.evaluation_policy.calc_action_choice_prob_given_action(
                trajectory_state_ids.reshape((-1, 1)),
                trajectory_action_ids.reshape((-1, 1)),
                is_batch=True,
            ).reshape((n_trajectories, -1))
        )
        iw = (self.gamma * (evaluation_policy_ / behavior_policy_)).cumprod(
            axis=1
        ) / self.gamma
        return (iw * trajectory_rewards).sum(axis=1)

    def estimate_policy_value(
        self,
        trajectory_state_ids: np.ndarray,
        trajectory_action_ids: np.ndarray,
        trajectory_rewards: np.ndarray,
        **kwargs,
    ) -> float:
        return self._estimate_individual_value(
            trajectory_state_ids=trajectory_state_ids,
            trajectory_action_ids=trajectory_action_ids,
            trajectory_rewards=trajectory_rewards,
        ).mean()

    def estimate_confidence_interval(
        self,
        trajectory_state_ids: np.ndarray,
        trajectory_action_ids: np.ndarray,
        trajectory_rewards: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        return_mean: bool = False,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[float]:
        estimates = self._estimate_individual_value(
            trajectory_state_ids=trajectory_state_ids,
            trajectory_action_ids=trajectory_action_ids,
            trajectory_rewards=trajectory_rewards,
        )
        return self.bootstrap_estimate(
            estimates=estimates,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            return_mean=return_mean,
            random_state=random_state,
        )


@dataclass
class NonTabularSequentialImportanceSampling(BaseOffPolicyEstimator):
    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    gamma: float = 1.0

    def _estimate_individual_value(
        self,
        trajectory_memory_states: np.ndarray,
        trajectory_actions: np.ndarray,
        trajectory_rewards: np.ndarray,
    ):
        if len(trajectory_memory_states.shape) == 2:
            n_trajectories, n_steps = trajectory_memory_states.shape
            input_dim = 1
        else:
            n_trajectories, n_steps, input_dim = trajectory_memory_states.shape

        behavior_policy_ = self.behavior_policy.calc_action_choice_prob_given_action(
            trajectory_memory_states.reshape((-1, input_dim)),
            trajectory_actions.reshape((-1, 1)),
            is_batch=True,
        ).reshape((n_trajectories, n_steps))

        evaluation_policy_ = (
            self.evaluation_policy.calc_action_choice_prob_given_action(
                trajectory_memory_states.reshape((-1, input_dim)),
                trajectory_actions.reshape((-1, 1)),
                is_batch=True,
            ).reshape((n_trajectories, n_steps))
        )

        iw = (self.gamma * (evaluation_policy_ / behavior_policy_)).cumprod(axis=1)
        return (iw * trajectory_rewards).sum(axis=1) / self.gamma

    def estimate_policy_value(
        self,
        trajectory_memory_states: np.ndarray,
        trajectory_actions: np.ndarray,
        trajectory_rewards: np.ndarray,
        **kwargs,
    ) -> float:
        return self._estimate_individual_value(
            trajectory_memory_states=trajectory_memory_states,
            trajectory_actions=trajectory_actions,
            trajectory_rewards=trajectory_rewards,
        ).mean()

    def estimate_confidence_interval(
        self,
        trajectory_memory_states: np.ndarray,
        trajectory_actions: np.ndarray,
        trajectory_rewards: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        return_mean: bool = False,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[float]:
        estimates = self._estimate_individual_value(
            trajectory_memory_states=trajectory_memory_states,
            trajectory_actions=trajectory_actions,
            trajectory_rewards=trajectory_rewards,
        )
        return self.bootstrap_estimate(
            estimates=estimates,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            return_mean=return_mean,
            random_state=random_state,
        )
