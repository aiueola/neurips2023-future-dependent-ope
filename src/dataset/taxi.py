from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm

import numpy as np
from sklearn.utils import check_random_state

from envs.taxi import PartiallyObservableTaxi
from policy.policy import BasePolicy


@dataclass
class TabularTaxiDataset:
    """This dataset class is for the case of no-memory setting.

    e.g., :math:`M_H = 1, M = 0, M_F = 1`.

    Otherwise, non-tabular setting should be used.

    """

    env: PartiallyObservableTaxi
    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    random_state: Optional[int] = None

    def __post_init__(self):
        self.env.seed(self.random_state)
        self.random_ = check_random_state(self.random_state)

    def obtain_logged_data_for_value_based(
        self,
        n_trajectories: int,
    ):
        trajectory_state_ids = np.zeros((n_trajectories, 102), dtype=int)
        trajectory_action_ids = np.zeros((n_trajectories, 102), dtype=int)
        trajectory_rewards = np.zeros((n_trajectories, 102))

        history_state_id = np.zeros(n_trajectories * 100, dtype=int)
        history_action_id = np.zeros(n_trajectories * 100, dtype=int)
        state_id = np.zeros(n_trajectories * 100, dtype=int)
        action_id = np.zeros(n_trajectories * 100, dtype=int)
        next_state_id = np.zeros(n_trajectories * 100, dtype=int)
        reward = np.zeros(n_trajectories * 100)

        # burn-in
        # to obtain data from a stationary distribution
        next_state = self.env.reset()
        for t in range(100):
            state = next_state
            action = self.behavior_policy.sample_action(state)
            next_state, reward_, done, _ = self.env.step(action)

            if done:
                next_state = self.env.reset()

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_logged_dataset]",
            total=n_trajectories,
        ):
            for t in range(102):
                state = next_state
                action = self.behavior_policy.sample_action(state)
                next_state, reward_, done, _ = self.env.step(action)

                trajectory_state_ids[i, t] = state
                trajectory_action_ids[i, t] = action
                trajectory_rewards[i, t] = reward_

                if done:
                    next_state = self.env.reset()

            history_state_id[i * 100 : (i + 1) * 100] = trajectory_state_ids[i, :-2]
            history_action_id[i * 100 : (i + 1) * 100] = trajectory_action_ids[i, :-2]
            state_id[i * 100 : (i + 1) * 100] = trajectory_state_ids[i, 1:-1]
            action_id[i * 100 : (i + 1) * 100] = trajectory_action_ids[i, 1:-1]
            reward[i * 100 : (i + 1) * 100] = trajectory_rewards[i, 1:-1]
            next_state_id[i * 100 : (i + 1) * 100] = trajectory_state_ids[i, 2:]

        logged_dataset = {
            "history_state_id": history_state_id,
            "history_action_id": history_action_id,
            "state_id": state_id,
            "action_id": action_id,
            "next_state_id": next_state_id,
            "reward": reward,
        }
        return logged_dataset

    def obtain_logged_data_for_sis(
        self,
        n_trajectories: int,
    ):
        trajectory_state_ids = np.zeros((n_trajectories, 100), dtype=int)
        trajectory_action_ids = np.zeros((n_trajectories, 100), dtype=int)
        trajectory_rewards = np.zeros((n_trajectories, 100))

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_logged_dataset]",
            total=n_trajectories,
        ):
            next_state = self.env.reset()

            # burn-in
            # to obtain initial distribution under evaluation policy
            for t in range(100):
                state = next_state
                action = self.evaluation_policy.sample_action(state)
                next_state, reward_, done, _ = self.env.step(action)

                if done:
                    next_state = self.env.reset()

            for t in range(100):
                state = next_state
                action = self.behavior_policy.sample_action(state)
                next_state, reward_, done, _ = self.env.step(action)

                trajectory_state_ids[i, t] = state
                trajectory_action_ids[i, t] = action
                trajectory_rewards[i, t] = reward_

                if done:
                    next_state = self.env.reset()

        logged_dataset = {
            "trajectory_state_ids": trajectory_state_ids,
            "trajectory_action_ids": trajectory_action_ids,
            "trajectory_rewards": trajectory_rewards,
        }
        return logged_dataset

    def obtain_initial_logged_data(
        self,
        n_trajectories: int,
    ):
        initial_state_id = np.zeros(n_trajectories, dtype=int)

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_initial_logged_dataset]",
            total=n_trajectories,
        ):
            next_state = self.env.reset()

            for t in range(100):
                state = next_state
                action = self.evaluation_policy.sample_action(state)
                next_state, _, done, _ = self.env.step(action)

                if done:
                    next_state = self.env.reset()

            initial_state_id[i] = state

        initial_logged_dataset = {
            "initial_state_id": initial_state_id,
        }
        return initial_logged_dataset


@dataclass
class NonTabularTaxiDataset:
    env: PartiallyObservableTaxi
    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    history_length: int
    memory_length: int  # |Z|
    future_length: int  # |F \ O|
    random_state: Optional[int] = None

    def __post_init__(self):
        self.env.seed(self.random_state)
        self.random_ = check_random_state(self.random_state)

    def obtain_logged_data_for_value_based(
        self,
        n_trajectories: int,
    ):
        trajectory_memory_state_ids = np.zeros(
            (
                n_trajectories,
                100 + self.history_length + self.future_length + 1,
                self.memory_length + 1,
            ),
            dtype=int,
        )
        trajectory_action_ids = np.zeros(
            (n_trajectories, 100 + self.history_length + self.future_length + 1),
            dtype=int,
        )
        trajectory_rewards = np.zeros(
            (n_trajectories, 100 + self.history_length + self.future_length + 1)
        )

        history_action_ids = np.zeros(
            (n_trajectories * 100, self.history_length), dtype=int
        )
        history_state_ids = np.zeros(
            (n_trajectories * 100, self.history_length), dtype=int
        )
        state_id = np.zeros(n_trajectories * 100, dtype=int)
        action_id = np.zeros(n_trajectories * 100, dtype=int)
        next_state_id = np.zeros(n_trajectories * 100, dtype=int)
        next_action_id = np.zeros(n_trajectories * 100, dtype=int)
        reward = np.zeros(n_trajectories * 100)

        if self.memory_length > 0:
            memory_state_ids = np.zeros(
                (n_trajectories * 100, self.memory_length), dtype=int
            )
            memory_action_ids = np.zeros(
                (n_trajectories * 100, self.memory_length), dtype=int
            )
            next_memory_state_ids = np.zeros(
                (n_trajectories * 100, self.memory_length), dtype=int
            )
            next_memory_action_ids = np.zeros(
                (n_trajectories * 100, self.memory_length), dtype=int
            )

        if self.future_length > 0:
            future_state_ids = np.zeros(
                (n_trajectories * 100, self.future_length), dtype=int
            )
            future_action_ids = np.zeros(
                (n_trajectories * 100, self.future_length), dtype=int
            )
            next_future_state_ids = np.zeros(
                (n_trajectories * 100, self.future_length), dtype=int
            )
            next_future_action_ids = np.zeros(
                (n_trajectories * 100, self.future_length), dtype=int
            )

        next_memory = self.env.reset()
        for t in range(100):
            memory = next_memory
            action = self.behavior_policy.sample_action(memory)
            next_memory, reward_, done, _ = self.env.step(action)

            if done:
                next_memory = self.env.reset()

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_logged_dataset]",
            total=n_trajectories,
        ):
            for t in range(100 + self.history_length + self.future_length + 1):
                memory = next_memory
                action = self.behavior_policy.sample_action(memory)
                next_memory, reward_, done, _ = self.env.step(action)

                trajectory_memory_state_ids[i, t] = memory
                trajectory_action_ids[i, t] = action
                trajectory_rewards[i, t] = reward_

                if done:
                    next_memory = self.env.reset()

            for t in range(100):
                history_state_ids[i * 100 + t] = trajectory_memory_state_ids[
                    i, t : t + self.history_length, -1
                ]
                history_action_ids[i * 100 + t] = trajectory_action_ids[
                    i,
                    t : t + self.history_length,
                ]
                state_id[i * 100 + t] = trajectory_memory_state_ids[
                    i, t + self.history_length, -1
                ]
                action_id[i * 100 + t] = trajectory_action_ids[
                    i, t + self.history_length
                ]
                next_state_id[i * 100 + t] = trajectory_memory_state_ids[
                    i, t + self.history_length + 1, -1
                ]
                next_action_id[i * 100 + t] = trajectory_action_ids[
                    i, t + self.history_length + 1
                ]
                reward[i * 100 + t] = trajectory_rewards[i, t + self.history_length]

            if self.memory_length > 0:
                memory_state_ids[i * 100 + t] = trajectory_memory_state_ids[
                    i,
                    t
                    + self.history_length
                    - self.memory_length : t
                    + self.history_length,
                    -1,
                ]
                memory_action_ids[i * 100 + t] = trajectory_action_ids[
                    i,
                    t
                    + self.history_length
                    - self.memory_length : t
                    + self.history_length,
                ]
                next_memory_state_ids[i] = trajectory_memory_state_ids[
                    i,
                    t
                    + self.history_length
                    - self.memory_length
                    + 1 : t
                    + self.history_length
                    + 1,
                    -1,
                ]
                next_memory_action_ids[i] = trajectory_action_ids[
                    i,
                    t
                    + self.history_length
                    - self.memory_length
                    + 1 : t
                    + self.history_length
                    + 1,
                ]

            if self.future_length > 0:
                future_state_ids[i * 100 + t] = trajectory_memory_state_ids[
                    i,
                    t
                    + self.history_length
                    + 1 : t
                    + self.history_length
                    + self.future_length
                    + 1,
                    -1,
                ]
                future_action_ids[i * 100 + t] = trajectory_action_ids[
                    i,
                    t
                    + self.history_length : t
                    + self.history_length
                    + self.future_length,
                ]
                next_future_state_ids[i * 100 + t] = trajectory_memory_state_ids[
                    i,
                    t
                    + self.history_length
                    + 2 : t
                    + self.history_length
                    + self.future_length
                    + 2,
                    -1,
                ]
                next_future_action_ids[i * 100 + t] = trajectory_action_ids[
                    i,
                    t
                    + self.history_length
                    + 1 : t
                    + self.history_length
                    + self.future_length
                    + 1,
                ]

        if self.memory_length == 0:
            memory_state_ids = None
            memory_action_ids = None
            next_memory_state_ids = None
            next_memory_action_ids = None

        if self.future_length == 0:
            future_state_ids = None
            future_action_ids = None
            next_future_state_ids = None
            next_future_action_ids = None

        logged_dataset = {
            "history_states": history_state_ids,
            "history_actions": history_action_ids,
            "memory_states": memory_state_ids,
            "memory_actions": memory_action_ids,
            "state": state_id,
            "action": action_id,
            "future_states": future_state_ids,
            "future_actions": future_action_ids,
            "next_memory_states": next_memory_state_ids,
            "next_memory_actions": next_memory_action_ids,
            "next_state": next_state_id,
            "next_action": next_action_id,
            "next_future_states": next_future_state_ids,
            "next_future_actions": next_future_action_ids,
            "reward": reward,
        }
        return logged_dataset

    def obtain_logged_data_for_sis(
        self,
        n_trajectories: int,
    ):
        trajectory_memory_state_ids = np.zeros(
            (n_trajectories, 100, self.memory_length + 1), dtype=int
        )
        trajectory_action_ids = np.zeros((n_trajectories, 100), dtype=int)
        trajectory_rewards = np.zeros((n_trajectories, 100))

        next_memory = self.env.reset()
        for t in range(100):
            memory = next_memory
            action = self.evaluation_policy.sample_action(memory)
            next_memory, reward_, done, _ = self.env.step(action)

            if done:
                next_memory = self.env.reset()

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_logged_dataset]",
            total=n_trajectories,
        ):
            for t in range(100):
                memory = next_memory
                action = self.behavior_policy.sample_action(memory)
                next_memory, reward_, done, _ = self.env.step(action)

                trajectory_memory_state_ids[i, t] = memory
                trajectory_action_ids[i, t] = action
                trajectory_rewards[i, t] = reward_

                if done:
                    next_memory = self.env.reset()

        logged_dataset = {
            "trajectory_memory_states": trajectory_memory_state_ids,
            "trajectory_actions": trajectory_action_ids,
            "trajectory_rewards": trajectory_rewards,
        }
        return logged_dataset

    def obtain_initial_logged_data(
        self,
        n_trajectories: int,
    ):
        initial_state_id = np.zeros((n_trajectories,), dtype=int)
        initial_action_id = np.zeros((n_trajectories,), dtype=int)

        if self.memory_length > 0:
            initial_memory_state_ids = np.zeros(
                (n_trajectories, self.memory_length), dtype=int
            )
            initial_memory_action_ids = np.zeros(
                (n_trajectories, self.memory_length), dtype=int
            )

        if self.future_length > 0:
            initial_future_state_ids = np.zeros(
                (n_trajectories, self.future_length), dtype=int
            )
            initial_future_action_ids = np.zeros(
                (n_trajectories, self.future_length), dtype=int
            )

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_initial_logged_dataset]",
            total=n_trajectories,
        ):
            next_memory = self.env.reset()
            for t in range(100):
                memory = next_memory
                action = self.evaluation_policy.sample_action(memory)
                next_memory, _, done, _ = self.env.step(action)

                if done:
                    next_memory = self.env.reset()

            # memory
            if self.memory_length > 0:
                for t in range(self.memory_length):
                    memory = next_memory
                    action = self.behavior_policy.sample_action(memory)
                    next_memory, _, done, _ = self.env.step(action)

                    if done:
                        next_memory = self.env.reset()

                    initial_memory_state_ids[i, t] = memory[-1]
                    initial_memory_action_ids[i, t] = action

            # observation
            memory = next_memory
            action = self.behavior_policy.sample_action(memory)
            next_memory, _, done, _ = self.env.step(action)

            if done:
                next_memory = self.env.reset()

            initial_state_id[i] = memory[-1]
            initial_action_id[i] = action

            # future
            if self.future_length > 0:
                for t in range(self.future_length):
                    initial_future_action_ids[i, t] = action

                    memory = next_memory
                    action = self.behavior_policy.sample_action(memory)
                    next_memory, _, done, _ = self.env.step(action)

                    if done:
                        next_memory = self.env.reset()

                    initial_future_state_ids[i, t] = memory[-1]
                    initial_future_action_ids[i, t] = action

        if self.memory_length == 0:
            initial_memory_state_ids = None
            initial_memory_action_ids = None

        if self.future_length == 0:
            initial_future_state_ids = None
            initial_future_action_ids = None

        initial_logged_dataset = {
            "initial_memory_states": initial_memory_state_ids,
            "initial_memory_actions": initial_memory_action_ids,
            "initial_state": initial_state_id,
            "initial_action": initial_action_id,
            "initial_future_states": initial_future_state_ids,
            "initial_future_actions": initial_future_action_ids,
        }
        return initial_logged_dataset
