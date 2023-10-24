"""Value-Based Estimator."""
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

from policy.policy import BasePolicy
from .v_func import DiscreteStateLSTMVfunction, ContinuousStateLSTMVfunction
from .base import BaseNeuralValueBasedOffPolicyEstimator
from utils import to_tensor


@dataclass
class NeuralFutureDependentValueBasedOPE(BaseNeuralValueBasedOffPolicyEstimator):
    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    v_function: Union[DiscreteStateLSTMVfunction, ContinuousStateLSTMVfunction]
    gamma: float = 1.0
    sigma: float = 1.0
    alpha: float = 0.5
    lambda_: float = 0.5
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        self.is_onehot = isinstance(self.v_function, DiscreteStateLSTMVfunction)
        self.v_function.to(self.device)
        self.random_ = check_random_state(self.random_state)

    def _save_learning_curve(self, predictions: np.ndarray, losses: np.ndarray):
        epochs = np.arange(len(predictions))
        # predictions = np.convolve(predictions, np.ones(100) / 100, mode="same")
        # losses = np.convolve(losses, np.ones(100) / 100, mode="same")

        # losses = np.clip(losses / (predictions**2 + 1e-5), 0, 1e2)

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(epochs, predictions, color=color[0], label="value pred.")

        ax2 = ax1.twinx()
        ln2 = ax2.plot(epochs, losses, color=color[1], alpha=0.3, label="loss fn.")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

        ax1.set_xlabel("epoch")
        ax1.set_ylabel("stationary value prediction")
        ax2.set_ylabel("weighted temporal difference loss")
        ax2.set_yscale("log")
        plt.savefig("learning_curve_future_dependent.png", dpi=300, bbox_inches="tight")

    def _objective_function(
        self,
        history_states: torch.Tensor,  # H
        history_actions: torch.Tensor,  # H
        memory_states: Optional[torch.Tensor],  # Z
        memory_actions: Optional[torch.Tensor],  # Z
        state: torch.Tensor,  # O
        action: torch.Tensor,  # A
        importance_weight: torch.Tensor,  # \\mu(Z, O, A)
        reward: torch.Tensor,  # R
        future_states: Optional[torch.Tensor],  # F \ O
        future_actions: Optional[torch.Tensor],  # F
        next_memory_states: Optional[torch.Tensor],  # Z'
        next_memory_actions: Optional[torch.Tensor],  # Z'
        next_state: torch.Tensor,  # O'
        next_action: torch.Tensor,  # A'
        next_future_states: Optional[torch.Tensor],  # F' \ O'
        next_future_actions: Optional[torch.Tensor],  # F'
        epoch: int,
    ):
        if self.is_onehot:
            kernel = self._onehot_kernel(history_states, history_actions)
        else:
            kernel = self._gaussian_kernel(history_states, history_actions)

        current_q = self.v_function(
            memory_states=memory_states,
            memory_actions=memory_actions,
            state=state,
            future_states=future_states,
            future_actions=future_actions,
        )  # q(\\bar{F})

        with torch.no_grad():
            next_q = self.v_function(
                memory_states=next_memory_states,
                memory_actions=next_memory_actions,
                state=next_state,
                future_states=next_future_states,
                future_actions=next_future_actions,
            )  # q(\\bar{F}')

        td_error = importance_weight * (reward + self.gamma * next_q) - current_q

        if self.lambda_ > 0:
            objective_function = td_error.T @ self._middle_term(kernel) @ td_error
        else:
            objective_function = td_error.T @ kernel @ td_error

        return objective_function

    def fit(
        self,
        history_states: np.ndarray,  # H
        history_actions: np.ndarray,  # H
        memory_states: Optional[np.ndarray],  # Z
        memory_actions: Optional[np.ndarray],  # Z
        state: np.ndarray,  # O
        action: np.ndarray,  # A
        reward: np.ndarray,  # R
        future_states: Optional[np.ndarray],  # F \ O
        future_actions: Optional[np.ndarray],  # F
        next_memory_states: Optional[np.ndarray],  # Z'
        next_memory_actions: Optional[np.ndarray],  # Z'
        next_state: np.ndarray,  # O'
        next_action: np.ndarray,  # A'
        next_future_states: Optional[np.ndarray],  # F' \ O'
        next_future_actions: Optional[np.ndarray],  # F'
        initial_memory_states: Optional[np.ndarray],  # Z_0
        initial_memory_actions: Optional[np.ndarray],  # Z_0
        initial_state: np.ndarray,  # O_0
        initial_action: np.ndarray,  # A_0
        initial_future_states: Optional[np.ndarray],  # F_0 \ O_0
        initial_future_actions: Optional[np.ndarray],  # F_0
        n_epoch: int = 20000,
        n_step_per_epoch: int = 10,
        # batch_size: int = 1026,
        batch_size: int = 32,
        lr: float = 1e-4,
        delta: float = 1.0,
        **kwargs,
    ):
        torch.manual_seed(self.random_state)
        importance_weight = self._mu(memory_states, state, action)

        n_samples = len(history_states)
        test_ids = self.random_.choice(n_samples, size=1000, replace=False)
        train_ids = np.setdiff1d(np.arange(n_samples), test_ids)

        if state.ndim == 1:
            test_history_states = to_tensor(
                history_states[test_ids], dtype=int, device=self.device
            )
            test_state = to_tensor(state[test_ids], dtype=int, device=self.device)
            test_next_state = to_tensor(
                next_state[test_ids], dtype=int, device=self.device
            )
            history_states = to_tensor(
                history_states[train_ids], dtype=int, device=self.device
            )
            state = to_tensor(state[train_ids], dtype=int, device=self.device)
            next_state = to_tensor(next_state[train_ids], dtype=int, device=self.device)
        else:
            test_history_states = to_tensor(
                history_states[test_ids], device=self.device
            )
            test_state = to_tensor(state[test_ids], device=self.device)
            test_next_state = to_tensor(next_state[test_ids], device=self.device)
            history_states = to_tensor(history_states[train_ids], device=self.device)
            state = to_tensor(state[train_ids], device=self.device)
            next_state = to_tensor(next_state[train_ids], device=self.device)

        test_importance_weight = to_tensor(
            importance_weight[test_ids], device=self.device
        )
        test_history_actions = to_tensor(
            history_actions[test_ids], dtype=int, device=self.device
        )
        test_action = to_tensor(action[test_ids], dtype=int, device=self.device)
        test_reward = to_tensor(reward[test_ids], device=self.device)
        test_next_action = to_tensor(
            next_action[test_ids], dtype=int, device=self.device
        )

        importance_weight = to_tensor(importance_weight[train_ids], device=self.device)
        history_actions = to_tensor(
            history_actions[train_ids], dtype=int, device=self.device
        )
        action = to_tensor(action[train_ids], dtype=int, device=self.device)
        reward = to_tensor(reward[train_ids], device=self.device)
        next_action = to_tensor(next_action[train_ids], dtype=int, device=self.device)

        # for debugging
        if state.ndim == 1:
            initial_state = to_tensor(initial_state, dtype=int, device=self.device)
        else:
            initial_state = to_tensor(initial_state, device=self.device)

        initial_action = to_tensor(initial_action, dtype=int, device=self.device)

        if memory_states is not None:
            if state.ndim == 1:
                test_memory_states = to_tensor(
                    memory_states[test_ids], dtype=int, device=self.device
                )
                test_next_memory_states = to_tensor(
                    next_memory_states[test_ids], dtype=int, device=self.device
                )
                memory_states = to_tensor(
                    memory_states[train_ids], dtype=int, device=self.device
                )
                next_memory_states = to_tensor(
                    next_memory_states[train_ids], dtype=int, device=self.device
                )
                initial_memory_states = to_tensor(
                    initial_memory_states,
                    dtype=int,
                    device=self.device,
                )
            else:
                test_memory_states = to_tensor(
                    memory_states[test_ids], device=self.device
                )
                test_next_memory_states = to_tensor(
                    next_memory_states[test_ids], device=self.device
                )
                memory_states = to_tensor(memory_states[train_ids], device=self.device)
                next_memory_states = to_tensor(
                    next_memory_states[train_ids], device=self.device
                )
                initial_memory_states = to_tensor(
                    initial_memory_states,
                    device=self.device,
                )

            test_memory_actions = to_tensor(
                memory_actions[test_ids], dtype=int, device=self.device
            )
            test_next_memory_actions = to_tensor(
                next_memory_actions[test_ids], dtype=int, device=self.device
            )
            memory_actions = to_tensor(
                memory_actions[train_ids], dtype=int, device=self.device
            )
            next_memory_actions = to_tensor(
                next_memory_actions[train_ids], dtype=int, device=self.device
            )
            initial_memory_actions = to_tensor(
                initial_memory_actions,
                dtype=int,
                device=self.device,
            )

        else:
            test_memory_states = None
            test_memory_actions = None
            test_next_memory_states = None
            test_next_memory_actions = None
            initial_memory_states = None
            initial_memory_actions = None

        if future_states is not None:
            if state.ndim == 1:
                test_future_states = to_tensor(
                    future_states[test_ids], dtype=int, device=self.device
                )
                test_next_future_states = to_tensor(
                    next_future_states[test_ids], dtype=int, device=self.device
                )
                future_states = to_tensor(
                    future_states[train_ids], dtype=int, device=self.device
                )
                next_future_states = to_tensor(
                    next_future_states[train_ids], dtype=int, device=self.device
                )
                initial_future_states = to_tensor(
                    initial_future_states, dtype=int, device=self.device
                )
            else:
                test_future_states = to_tensor(
                    future_states[test_ids], device=self.device
                )
                test_next_future_states = to_tensor(
                    next_future_states[test_ids], device=self.device
                )
                future_states = to_tensor(future_states[train_ids], device=self.device)
                next_future_states = to_tensor(
                    next_future_states[train_ids], device=self.device
                )
                initial_future_states = to_tensor(
                    initial_future_states, device=self.device
                )

            test_future_actions = to_tensor(
                future_actions[test_ids], dtype=int, device=self.device
            )
            test_next_future_actions = to_tensor(
                next_future_actions[test_ids], dtype=int, device=self.device
            )
            future_actions = to_tensor(
                future_actions[train_ids], dtype=int, device=self.device
            )
            next_future_actions = to_tensor(
                next_future_actions[train_ids], dtype=int, device=self.device
            )
            initial_future_actions = to_tensor(
                initial_future_actions, dtype=int, device=self.device
            )

        else:
            test_future_states = None
            test_future_actions = None
            test_next_future_states = None
            test_next_future_actions = None
            initial_future_states = None
            initial_future_actions = None

        # optimizer = optim.SGD(self.v_function.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(self.v_function.parameters(), lr=lr)
        predictions = np.zeros(n_epoch)
        losses = np.zeros(n_epoch)

        memory_states_ = None
        memory_actions_ = None
        next_memory_states_ = None
        next_memory_actions_ = None
        future_states_ = None
        future_actions_ = None
        next_future_states_ = None
        next_future_actions_ = None

        # pre-train lstm
        for grad_step in range(10000):
            idx_ = torch.randint(len(state), size=(batch_size,))

            if memory_states is not None:
                memory_states_ = memory_states[idx_]
                memory_actions_ = memory_actions[idx_]

            if future_states is not None:
                future_states_ = future_states[idx_]
                future_actions_ = future_actions[idx_]

            state_prediction_loss_ = self.v_function.state_prediction_loss(
                history_states=history_states[idx_],
                history_actions=history_actions[idx_],
                memory_states=memory_states_,
                memory_actions=memory_actions_,
                state=state[idx_],
                future_states=future_states_,
                future_actions=future_actions_,
            )
            optimizer.zero_grad()
            state_prediction_loss_.backward()
            optimizer.step()

            if grad_step % 1000 == 0:
                print(
                    f"grad_step={grad_step: >4}, state_prediction_loss={state_prediction_loss_.item():.3f}"
                )

        for epoch in range(n_epoch):
            for grad_step in range(n_step_per_epoch):
                idx_ = torch.randint(len(state), size=(batch_size,))

                if memory_states is not None:
                    memory_states_ = memory_states[idx_]
                    memory_actions_ = memory_actions[idx_]
                    next_memory_states_ = next_memory_states[idx_]
                    next_memory_actions_ = next_memory_actions[idx_]

                if future_states is not None:
                    future_states_ = future_states[idx_]
                    future_actions_ = future_actions[idx_]
                    next_future_states_ = next_future_states[idx_]
                    next_future_actions_ = next_future_actions[idx_]

                objective_loss_ = (
                    self._objective_function(
                        history_states=history_states[idx_],
                        history_actions=history_actions[idx_],
                        memory_states=memory_states_,
                        memory_actions=memory_actions_,
                        state=state[idx_],
                        action=action[idx_],
                        importance_weight=importance_weight[idx_],
                        reward=reward[idx_],
                        future_states=future_states_,
                        future_actions=future_actions_,
                        next_memory_states=next_memory_states_,
                        next_memory_actions=next_memory_actions_,
                        next_state=next_state[idx_],
                        next_action=next_action[idx_],
                        next_future_states=next_future_states_,
                        next_future_actions=next_future_actions_,
                        epoch=epoch,
                    )
                    / batch_size
                )

                optimizer.zero_grad()
                objective_loss_.backward()
                optimizer.step()

            with torch.no_grad():
                value_prediction = self.v_function(
                    memory_states=initial_memory_states,
                    memory_actions=initial_memory_actions,
                    state=initial_state,
                    future_states=initial_future_states,
                    future_actions=initial_future_actions,
                )
                test_loss = self._objective_function(
                    history_states=test_history_states,
                    history_actions=test_history_actions,
                    memory_states=test_memory_states,
                    memory_actions=test_memory_actions,
                    state=test_state,
                    action=test_action,
                    importance_weight=test_importance_weight,
                    reward=test_reward,
                    future_states=test_future_states,
                    future_actions=test_future_actions,
                    next_memory_states=test_next_memory_states,
                    next_memory_actions=test_next_memory_actions,
                    next_state=test_next_state,
                    next_action=test_next_action,
                    next_future_states=test_next_future_states,
                    next_future_actions=test_next_future_actions,
                    epoch=epoch,
                ) / len(test_state)

            predictions[epoch] = value_prediction.mean()
            losses[epoch] = test_loss.item()

            if epoch % 10 == 0:
                print(
                    f"epoch={epoch: >4}, value_prediction={value_prediction.mean():.3f}, test_loss={test_loss.item():.3f}"
                )

        self.predictions = predictions
        self.losses = losses
        # self._save_learning_curve(predictions, losses)

    def _estimate_individual_value(
        self,
        initial_memory_states: Optional[np.ndarray],  # Z_0
        initial_memory_actions: Optional[np.ndarray],  # Z_0
        initial_state: np.ndarray,  # O_0
        initial_action: np.ndarray,  # A_0
        initial_future_states: Optional[np.ndarray],  # F_0 \ O_0
        initial_future_actions: Optional[np.ndarray],  # F_0
    ) -> torch.Tensor:
        if initial_state.ndim == 1:
            initial_state = to_tensor(initial_state, dtype=int, device=self.device)
        else:
            initial_state = to_tensor(initial_state, device=self.device)

        initial_action = to_tensor(initial_action, dtype=int, device=self.device)

        if initial_memory_states is not None:
            if initial_state.ndim == 1:
                initial_memory_states = to_tensor(
                    initial_memory_states, dtype=int, device=self.device
                )
            else:
                initial_memory_states = to_tensor(
                    initial_memory_states, device=self.device
                )

            initial_memory_actions = to_tensor(
                initial_memory_actions, dtype=int, device=self.device
            )

        if initial_future_states is not None:
            if initial_state.ndim == 1:
                initial_future_states = to_tensor(
                    initial_future_states, dtype=int, device=self.device
                )
            else:
                initial_future_states = to_tensor(
                    initial_future_states, device=self.device
                )

            initial_future_actions = to_tensor(
                initial_future_actions, dtype=int, device=self.device
            )

        with torch.no_grad():
            value_prediction = (
                self.v_function(
                    memory_states=initial_memory_states,
                    memory_actions=initial_memory_actions,
                    state=initial_state,
                    future_states=initial_future_states,
                    future_actions=initial_future_actions,
                )
                .cpu()
                .detach()
                .numpy()
            )
        return value_prediction

    def estimate_policy_value(
        self,
        initial_memory_states: Optional[np.ndarray],  # Z_0
        initial_memory_actions: Optional[np.ndarray],  # Z_0
        initial_state: np.ndarray,  # O_0
        initial_action: np.ndarray,  # A_0
        initial_future_states: Optional[np.ndarray],  # F_0 \ O_0
        initial_future_actions: Optional[np.ndarray],  # F_0
        **kwargs,
    ) -> float:
        return self._estimate_individual_value(
            initial_memory_states=initial_memory_states,
            initial_memory_actions=initial_memory_actions,
            initial_state=initial_state,
            initial_action=initial_action,
            initial_future_states=initial_future_states,
            initial_future_actions=initial_future_actions,
        ).mean()

    def estimate_confidence_interval(
        self,
        initial_memory_states: Optional[np.ndarray],  # Z_0
        initial_memory_actions: Optional[np.ndarray],  # Z_0
        initial_state: np.ndarray,  # O_0
        initial_action: np.ndarray,  # A_0
        initial_future_states: Optional[np.ndarray],  # F_0 \ O_0
        initial_future_actions: Optional[np.ndarray],  # F_0
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        return_mean: bool = False,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[float]:
        estimates = self._estimate_individual_value(
            initial_memory_states=initial_memory_states,
            initial_memory_actions=initial_memory_actions,
            initial_state=initial_state,
            initial_action=initial_action,
            initial_future_states=initial_future_states,
            initial_future_actions=initial_future_actions,
        )
        return self.bootstrap_estimate(
            estimates=estimates,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            return_mean=return_mean,
            random_state=random_state,
        )


@dataclass
class NeuralStateValueBasedOPE(BaseNeuralValueBasedOffPolicyEstimator):
    """This estimator assumes that the environment is MDP."""

    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    v_function: Union[DiscreteStateLSTMVfunction, ContinuousStateLSTMVfunction]
    gamma: float = 1.0
    sigma: float = 1.0
    alpha: float = 0.5
    lambda_: float = 0.5
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        self.is_onehot = isinstance(self.v_function, DiscreteStateLSTMVfunction)
        self.v_function.to(self.device)
        self.random_ = check_random_state(self.random_state)

    def _save_learning_curve(self, predictions: np.ndarray, losses: np.ndarray):
        epochs = np.arange(len(predictions))
        # predictions = np.convolve(predictions, np.ones(50) / 50, mode="same")
        # losses = np.convolve(losses, np.ones(50) / 50, mode="same")

        # losses = np.clip(losses / (predictions**2 + 1e-5), 0, 1e2)

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(epochs, predictions, color=color[0], label="value pred.")

        ax2 = ax1.twinx()
        ln2 = ax2.plot(epochs, losses, color=color[1], alpha=0.3, label="loss fn.")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

        ax1.set_xlabel("epoch")
        ax1.set_ylabel("stationary value prediction")
        ax2.set_ylabel("weighted temporal difference loss")
        ax2.set_yscale("log")
        plt.savefig("learning_curve_baseline.png", dpi=300, bbox_inches="tight")

    def _objective_function(
        self,
        state: torch.Tensor,  # O
        importance_weight: torch.Tensor,  # \\mu(Z, O, A)
        reward: torch.Tensor,  # R
        next_state: torch.Tensor,  # O'
        epoch: int,
    ):
        if self.is_onehot:
            kernel = self._onehot_kernel(state)
        else:
            kernel = self._gaussian_kernel(state)

        current_q = self.v_function(state)  # q(O)

        with torch.no_grad():
            next_q = self.v_function(next_state)  # q(O')

        td_error = importance_weight * (reward + self.gamma * next_q) - current_q

        if self.lambda_ > 0:
            objective_function = td_error.T @ self._middle_term(kernel) @ td_error
        else:
            objective_function = td_error.T @ kernel @ td_error

        return objective_function

    def fit(
        self,
        initial_state: np.ndarray,  # O_0
        initial_action: np.ndarray,  # A_0
        memory_states: Optional[np.ndarray],  # Z
        memory_actions: Optional[np.ndarray],  # Z
        state: np.ndarray,  # O
        action: np.ndarray,  # A
        reward: np.ndarray,  # R
        next_state: np.ndarray,  # O'
        next_action: np.ndarray,  # A'
        n_epoch: int = 20000,
        n_step_per_epoch: int = 10,
        # batch_size: int = 1026,
        batch_size: int = 32,
        lr: float = 1e-4,
        **kwargs,
    ):
        torch.manual_seed(self.random_state)
        importance_weight = self._mu(memory_states, state, action)

        n_samples = len(state)
        test_ids = self.random_.choice(n_samples, size=1000, replace=False)
        train_ids = np.setdiff1d(np.arange(n_samples), test_ids)

        if state.ndim == 1:
            test_state = to_tensor(state[test_ids], dtype=int, device=self.device)
            test_next_state = to_tensor(
                next_state[test_ids], dtype=int, device=self.device
            )
            state = to_tensor(state[train_ids], dtype=int, device=self.device)
            next_state = to_tensor(next_state[train_ids], dtype=int, device=self.device)
        else:
            test_state = to_tensor(state[test_ids], device=self.device)
            test_next_state = to_tensor(next_state[test_ids], device=self.device)
            state = to_tensor(state[train_ids], device=self.device)
            next_state = to_tensor(next_state[train_ids], device=self.device)

        test_importance_weight = to_tensor(
            importance_weight[test_ids], device=self.device
        )
        test_action = to_tensor(action[test_ids], dtype=int, device=self.device)
        test_reward = to_tensor(reward[test_ids], device=self.device)

        importance_weight = to_tensor(importance_weight[train_ids], device=self.device)
        reward = to_tensor(reward[train_ids], device=self.device)

        # for debugging
        if state.ndim == 1:
            initial_state = to_tensor(initial_state, dtype=int, device=self.device)
        else:
            initial_state = to_tensor(initial_state, device=self.device)

        # optimizer = optim.SGD(self.v_function.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(self.v_function.parameters(), lr=lr)
        predictions = np.zeros(n_epoch)
        losses = np.zeros(n_epoch)

        for epoch in range(n_epoch):
            for grad_step in range(n_step_per_epoch):
                idx_ = torch.randint(len(state), size=(batch_size,))
                objective_loss_ = (
                    self._objective_function(
                        state=state[idx_],
                        importance_weight=importance_weight[idx_],
                        reward=reward[idx_],
                        next_state=next_state[idx_],
                        epoch=epoch,
                    )
                    / batch_size
                )
                optimizer.zero_grad()
                objective_loss_.backward()
                optimizer.step()

                # if grad_step % target_update_interval_step == 0:
                #     self.target_v_function = deepcopy(self.v_function)

            with torch.no_grad():
                value_prediction = self.v_function(initial_state)
                test_loss = self._objective_function(
                    state=test_state,
                    importance_weight=test_importance_weight,
                    reward=test_reward,
                    next_state=test_next_state,
                    epoch=epoch,
                ) / len(test_state)

            predictions[epoch] = value_prediction.mean()
            losses[epoch] = test_loss.item()

            if epoch % 10 == 0:
                print(
                    f"epoch={epoch: >4}, value_prediction={value_prediction.mean():.3f}, test_loss={test_loss.item():.3f}"
                )

        self.predictions = predictions
        self.losses = losses
        # self._save_learning_curve(predictions, losses)

    def _estimate_individual_value(
        self,
        initial_state: np.ndarray,  # O_0,
    ) -> torch.Tensor:
        if initial_state.ndim == 1:
            initial_state = to_tensor(initial_state, dtype=int, device=self.device)
        else:
            initial_state = to_tensor(initial_state, device=self.device)

        with torch.no_grad():
            value_prediction = (
                self.v_function(
                    initial_state,
                )
                .cpu()
                .detach()
                .numpy()
            )
        return value_prediction

    def estimate_policy_value(
        self,
        initial_state: np.ndarray,  # O_0
        **kwargs,
    ) -> float:
        return self._estimate_individual_value(
            initial_state=initial_state,
        ).mean()

    def estimate_confidence_interval(
        self,
        initial_state: np.ndarray,  # (Z_0, O_0)
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        return_mean: bool = False,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[float]:
        estimates = self._estimate_individual_value(
            initial_state=initial_state,
        )
        return self.bootstrap_estimate(
            estimates=estimates,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            return_mean=return_mean,
            random_state=random_state,
        )
