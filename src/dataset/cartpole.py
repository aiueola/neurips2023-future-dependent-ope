from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm

import numpy as np
from sklearn.utils import check_random_state

from policy.policy import BasePolicy
from envs.cartpole import PartiallyObservableCartpole


@dataclass
class CartpoleDataset:
    env: PartiallyObservableCartpole
    behavior_policy: BasePolicy
    evaluation_policy: BasePolicy
    memory_length: int
    history_length: int
    future_length: int
    random_state: Optional[int] = None

    def __post_init__(self):
        self.env.seed(self.random_state)
        self.random_ = check_random_state(self.random_state)

    def obtain_logged_data_for_value_based(
        self,
        n_trajectories: int,
    ):
        trajectory_memory_states = np.zeros(
            (
                n_trajectories,
                100 + self.history_length + self.future_length + 1,
                self.memory_length + 1,
                4,
            )
        )
        trajectory_actions = np.zeros(
            (n_trajectories, 100 + self.history_length + self.future_length + 1),
            dtype=int,
        )
        trajectory_rewards = np.zeros(
            (n_trajectories, 100 + self.history_length + self.future_length + 1)
        )

        history_states = np.zeros((n_trajectories * 100, self.history_length, 4))
        history_actions = np.zeros(
            (n_trajectories * 100, self.history_length), dtype=int
        )
        state = np.zeros((n_trajectories * 100, 4))
        action = np.zeros(n_trajectories * 100, dtype=int)
        next_state = np.zeros((n_trajectories * 100, 4))
        next_action = np.zeros(n_trajectories * 100, dtype=int)
        reward = np.zeros(n_trajectories * 100)

        if self.memory_length > 0:
            memory_states = np.zeros(
                (n_trajectories * 100, self.memory_length, 4),
            )
            memory_actions = np.zeros(
                (n_trajectories * 100, self.memory_length), dtype=int
            )
            next_memory_states = np.zeros(
                (n_trajectories * 100, self.memory_length, 4),
            )
            next_memory_actions = np.zeros(
                (n_trajectories * 100, self.memory_length), dtype=int
            )

        if self.future_length > 0:
            future_states = np.zeros(
                (n_trajectories * 100, self.future_length, 4),
            )
            future_actions = np.zeros(
                (n_trajectories * 100, self.future_length), dtype=int
            )
            next_future_states = np.zeros(
                (n_trajectories * 100, self.future_length, 4),
            )
            next_future_actions = np.zeros(
                (n_trajectories * 100, self.future_length), dtype=int
            )

        next_memory = self.env.reset(return_memory_2d=True)
        for t in range(100):
            memory = next_memory
            action_ = self.behavior_policy.sample_action(memory.flatten())
            next_memory, reward_, done, _ = self.env.step(
                action_, return_memory_2d=True
            )

            if done:
                next_memory = self.env.reset()

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_logged_dataset]",
            total=n_trajectories,
        ):
            for t in range(100 + self.history_length + self.future_length + 1):
                memory = next_memory
                action_ = self.behavior_policy.sample_action(memory.flatten())
                next_memory, reward_, done, _ = self.env.step(
                    action_, return_memory_2d=True
                )

                if done:
                    next_memory = self.env.reset(return_memory_2d=True)

                trajectory_memory_states[i, t] = memory
                trajectory_actions[i, t] = action_
                trajectory_rewards[i, t] = reward_

            for t in range(100):
                history_states[i * 100 + t] = trajectory_memory_states[
                    i, t : t + self.history_length, -1
                ]
                history_actions[i * 100 + t] = trajectory_actions[
                    i,
                    t : t + self.history_length,
                ]
                state[i * 100 + t] = trajectory_memory_states[
                    i, t + self.history_length, -1
                ]
                action[i * 100 + t] = trajectory_actions[i, t + self.history_length]
                next_state[i * 100 + t] = trajectory_memory_states[
                    i, t + self.history_length + 1, -1
                ]
                next_action[i * 100 + t] = trajectory_actions[
                    i, t + self.history_length + 1
                ]
                reward[i * 100 + t] = trajectory_rewards[i, t + self.history_length]

            if self.memory_length > 0:
                memory_states[i * 100 + t] = trajectory_memory_states[
                    i,
                    t
                    + self.history_length
                    - self.memory_length : t
                    + self.history_length,
                    -1,
                ]
                memory_actions[i * 100 + t] = trajectory_actions[
                    i,
                    t
                    + self.history_length
                    - self.memory_length : t
                    + self.history_length,
                ]
                next_memory_states[i] = trajectory_memory_states[
                    i,
                    t
                    + self.history_length
                    - self.memory_length
                    + 1 : t
                    + self.history_length
                    + 1,
                    -1,
                ]
                next_memory_actions[i] = trajectory_actions[
                    i,
                    t
                    + self.history_length
                    - self.memory_length
                    + 1 : t
                    + self.history_length
                    + 1,
                ]

            if self.future_length > 0:
                future_states[i * 100 + t] = trajectory_memory_states[
                    i,
                    t
                    + self.history_length
                    + 1 : t
                    + self.history_length
                    + self.future_length
                    + 1,
                    -1,
                ]
                future_actions[i * 100 + t] = trajectory_actions[
                    i,
                    t
                    + self.history_length : t
                    + self.history_length
                    + self.future_length,
                ]
                next_future_states[i * 100 + t] = trajectory_memory_states[
                    i,
                    t
                    + self.history_length
                    + 2 : t
                    + self.history_length
                    + self.future_length
                    + 2,
                    -1,
                ]
                next_future_actions[i * 100 + t] = trajectory_actions[
                    i,
                    t
                    + self.history_length
                    + 1 : t
                    + self.history_length
                    + self.future_length
                    + 1,
                ]

        if self.memory_length == 0:
            memory_states = None
            memory_actions = None
            next_memory_states = None
            next_memory_actions = None

        if self.future_length == 0:
            future_states = None
            future_actions = None
            next_future_states = None
            next_future_actions = None

        logged_dataset = {
            "history_states": history_states,
            "history_actions": history_actions,
            "memory_states": memory_states,
            "memory_actions": memory_actions,
            "state": state,
            "action": action,
            "future_states": future_states,
            "future_actions": future_actions,
            "next_memory_states": next_memory_states,
            "next_memory_actions": next_memory_actions,
            "next_state": next_state,
            "next_action": next_action,
            "next_future_states": next_future_states,
            "next_future_actions": next_future_actions,
            "reward": reward,
        }
        return logged_dataset

    def obtain_logged_data_for_sis(
        self,
        n_trajectories: int,
    ):
        trajectory_memory_states = np.zeros(
            (n_trajectories, 100, self.memory_length + 1, 4)
        )
        trajectory_actions = np.zeros((n_trajectories, 100), dtype=int)
        trajectory_rewards = np.zeros((n_trajectories, 100))

        next_memory = self.env.reset(return_memory_2d=True)
        for t in range(100):
            memory = next_memory
            action = self.evaluation_policy.sample_action(memory.flatten())
            next_memory, reward_, done, _ = self.env.step(action, return_memory_2d=True)

            if done:
                next_memory = self.env.reset(return_memory_2d=True)

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_logged_dataset]",
            total=n_trajectories,
        ):
            for t in range(100):
                memory = next_memory
                action_ = self.behavior_policy.sample_action(memory.flatten())
                next_memory, reward_, done, _ = self.env.step(
                    action_, return_memory_2d=True
                )

                if done:
                    next_memory = self.env.reset(return_memory_2d=True)

                trajectory_memory_states[i, t] = memory
                trajectory_actions[i, t] = action_
                trajectory_rewards[i, t] = reward_

        logged_dataset = {
            "trajectory_memory_states": trajectory_memory_states.reshape(
                (-1, 100, (self.memory_length + 1) * 4)
            ),
            "trajectory_actions": trajectory_actions,
            "trajectory_rewards": trajectory_rewards,
        }
        return logged_dataset

    def obtain_initial_logged_data(
        self,
        n_trajectories: int,
    ):
        initial_state = np.zeros((n_trajectories, 4))
        initial_action = np.zeros((n_trajectories,), dtype=int)

        if self.memory_length > 0:
            initial_memory_states = np.zeros(
                (n_trajectories, self.memory_length, 4),
            )
            initial_memory_actions = np.zeros(
                (n_trajectories, self.memory_length), dtype=int
            )

        if self.future_length > 0:
            initial_future_states = np.zeros(
                (n_trajectories, self.future_length, 4),
            )
            initial_future_actions = np.zeros(
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

                    initial_memory_states[i, t] = memory[-1]
                    initial_memory_actions[i, t] = action

            # observation
            memory = next_memory
            action = self.behavior_policy.sample_action(memory)
            next_memory, _, done, _ = self.env.step(action)

            if done:
                next_memory = self.env.reset()

            initial_state[i] = memory[-1]
            initial_action[i] = action

            # future
            if self.future_length > 0:
                for t in range(self.future_length):
                    initial_future_actions[i, t] = action

                    memory = next_memory
                    action = self.behavior_policy.sample_action(memory)
                    next_memory, _, done, _ = self.env.step(action)

                    if done:
                        next_memory = self.env.reset()

                    initial_future_states[i, t] = memory[-1]
                    initial_future_actions[i, t] = action

        if self.memory_length == 0:
            initial_memory_states = None
            initial_memory_actions = None

        if self.future_length == 0:
            initial_future_states = None
            initial_future_actions = None

        initial_logged_dataset = {
            "initial_memory_states": initial_memory_states,
            "initial_memory_actions": initial_memory_actions,
            "initial_state": initial_state,
            "initial_action": initial_action,
            "initial_future_states": initial_future_states,
            "initial_future_actions": initial_future_actions,
        }
        return initial_logged_dataset
