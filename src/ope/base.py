"""Base Off-Policy Estimator."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Optional
from pathlib import Path

import torch
import numpy as np
from sklearn.utils import check_random_state


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_individual_value(self) -> Union[np.ndarray, torch.Tensor]:
        """Estimate the value of each data point."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_confidence_interval(self) -> Tuple[float]:
        """Estimate the confidence interval."""
        raise NotImplementedError

    def bootstrap_estimate(
        estimates: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        return_mean: bool = False,
        random_state: Optional[int] = None,
    ) -> Tuple[float]:
        random_ = check_random_state(random_state)

        bootstrap_estimates = np.zeros(n_bootstrap_samples)
        for i in range(n_bootstrap_samples):
            bootstrap_estimates = random_.choice(
                estimates, size=len(estimates), replace=True
            ).mean()

        mean = bootstrap_estimates.mean()
        lower_bound = np.percentile(bootstrap_estimates, 100 * (alpha / 2))
        upper_bound = np.percentile(bootstrap_estimates, 100 * (1.0 - alpha / 2))

        if return_mean:
            return lower_bound, upper_bound, mean
        else:
            return lower_bound, upper_bound


@dataclass
class BaseValueBasedOffPolicyEstimator(BaseOffPolicyEstimator):
    """Base class for Value-based OPE estimators."""

    @abstractmethod
    def fit(self) -> None:
        """Fit the value bridge function."""
        raise NotImplementedError()


@dataclass
class BaseLinearValueBasedOffPolicyEstimator(BaseValueBasedOffPolicyEstimator):
    """Base class for linear value-based OPE estimators."""

    def save(self, path: Path):
        np.save(path, self.value_bridge_function_coef)

    def load(self, path: Path):
        self.value_bridge_function_coef = np.load(path)

    def _inverse(self, symmetric_matrix: np.ndarray, regularization: float = 1e-50):
        symmetric_matrix = symmetric_matrix + regularization * np.identity(
            len(symmetric_matrix)
        )
        return np.linalg.pinv(symmetric_matrix, hermitian=True)

    def _pseudo_inverse(
        self,
        asymmetic_matrix: np.ndarray,
    ):
        symmetric_matrix = asymmetic_matrix.T @ asymmetic_matrix
        return self._inverse(symmetric_matrix) @ asymmetic_matrix.T


@dataclass
class BaseNeuralValueBasedOffPolicyEstimator(BaseValueBasedOffPolicyEstimator):
    """Base class for neural value-based OPE estimators."""

    def save(self, path: Path):
        torch.save(self.v_function.state_dict(), path)

    def save_learning_process(self, path: Path):
        np.save(path + "_prediction", self.predictions)
        np.save(path + "_test_loss", self.losses)

    def load(self, path: Path):
        self.v_function.load_state_dict(torch.load(path))

    def load_learning_process(self, path: Path):
        self.predictions = np.load(path + "_prediction.npy")
        self.losses = np.load(path + "_test_loss.npy")

    def _mu(
        self,
        memory_states: Optional[np.ndarray],  # Z
        state: np.ndarray,  # O
        action: np.ndarray,  # A
    ):
        if state.ndim == 1:
            if memory_states is None:
                states = state.reshape((-1, 1))
            else:
                states = np.concatenate((memory_states, state.reshape((-1, 1))), axis=1)
        else:
            state_dim = state.shape[1]
            if memory_states is None:
                states = state.reshape((-1, state_dim))
            else:
                memory_length = memory_states.shape[1]
                states = np.concatenate(
                    (memory_states, state.reshape((-1, 1, state_dim))), axis=1
                )
                states = states.reshape((-1, (memory_length + 1) * state_dim))

        behavior_policy = self.behavior_policy.calc_action_choice_prob_given_action(
            states,
            action,
            is_batch=True,
        )
        evaluation_policy = self.evaluation_policy.calc_action_choice_prob_given_action(
            states,
            action,
            is_batch=True,
        )
        return evaluation_policy / behavior_policy

    def _inverse(
        self,
        symmetric_matrix: torch.Tensor,
        regularization: float = 1e-50,
    ):
        symmetric_matrix = symmetric_matrix + regularization * torch.eye(
            len(symmetric_matrix)
        )
        return torch.linalg.pinv(symmetric_matrix, hermitian=True)

    def _gaussian_kernel(
        self,
        states: torch.Tensor,  # H
        actions: Optional[torch.Tensor] = None,  # H
    ):
        # (x - x') ** 2 = x ** 2 + x' ** 2 - 2 x x'
        with torch.no_grad():
            if actions is not None:
                emb = self.v_function._predict_state(
                    states, actions, input_type="history"
                )
            elif states.ndim == 1:
                emb = self.v_function._encode_state(states)
            else:
                emb = states

            x_2 = (emb**2).sum(dim=1)
            x_y = emb @ emb.T
            distance = x_2[:, None] + x_2[None, :] - 2 * x_y
            kernel = torch.exp(-distance / self.sigma)

        return kernel  # shape (n_samples, n_samples)

    def _onehot_kernel(
        self,
        states: torch.Tensor,  # H
        actions: Optional[torch.Tensor] = None,  # H
    ):
        with torch.no_grad():
            if actions is not None:
                if states.shape[1] > 1:
                    emb = self.v_function._predict_state_action(
                        states, actions, input_type="history"
                    )  # one-hot
                else:
                    emb = self.v_function._encode_state_action(states, actions)[:, 0, :]

            else:
                emb = self.v_function._encode_state(states)  # one-hot

        emb = emb.to(torch.float32)
        return emb @ emb.T  # shape (n_samples, n_samples)

    def _middle_term(
        self,
        kernel: torch.Tensor,
    ):
        n_samples = len(kernel)
        inverse_kernel = self._inverse(
            self.alpha * torch.eye(n_samples) + self.lambda_ * kernel
        )
        return kernel @ inverse_kernel  # shape (n_samples, n_samples)
