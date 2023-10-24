from copy import deepcopy
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
from sklearn.utils import check_random_state

import torch
from torch import nn, optim
import torch.nn.functional as F

from envs import PartiallyObservableCartpole
from utils import to_tensor


class QFunction(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.n_actions = n_actions

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        with torch.no_grad():
            action_onehot = F.one_hot(action, num_classes=self.n_actions)

        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return (x * action_onehot).sum(dim=1)

    def values(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x

    def max(self, state: torch.Tensor):
        x = self.values(state)
        return x.max(dim=1)[0]

    def argmax(self, state: torch.Tensor):
        x = self.values(state)
        return x.max(dim=1)[1]


@dataclass
class ReplayBuffer:
    env: PartiallyObservableCartpole
    buffer_size: int
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_state)

        state_dim = self.env.original_observation_space.shape[0]

        self.state_buffer = torch.zeros((self.buffer_size, state_dim))
        self.action_buffer = torch.zeros((self.buffer_size,), dtype=int)
        self.reward_buffer = torch.zeros((self.buffer_size,))
        self.next_state_buffer = torch.zeros((self.buffer_size, state_dim))
        self.done_buffer = torch.zeros((self.buffer_size,))

        self.idx = 0
        self.is_full = False

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ):
        state = to_tensor(state, device=self.device)
        next_state = to_tensor(next_state, device=self.device)

        self.state_buffer[self.idx] = state
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.next_state_buffer[self.idx] = next_state
        self.done_buffer[self.idx] = done

        self.idx = (self.idx + 1) % self.buffer_size

        if self.idx % self.buffer_size == 0:
            self.is_full = True

    def sample(self, batch_size: int = 64):
        max_idx = self.buffer_size if self.is_full else self.idx
        idxes = torch.randint(max_idx, size=(batch_size,))

        state = self.state_buffer[idxes]
        action = self.action_buffer[idxes]
        reward = self.reward_buffer[idxes]
        next_state = self.next_state_buffer[idxes]
        done = self.done_buffer[idxes]

        return state, action, reward, next_state, done


@dataclass
class OnlinePolicy:
    env: PartiallyObservableCartpole
    hidden_dim: int = 32
    buffer_size: int = 10000
    batch_size: int = 128
    target_update_interval: int = 10
    gamma: float = 1.0
    lr: float = 1e-4
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        self.n_actions = self.env.action_space.n

        self.q_function = QFunction(
            state_dim=self.env.original_observation_space.shape[0],
            n_actions=self.env.action_space.n,
            hidden_dim=self.hidden_dim,
        )
        self.target_q_function = deepcopy(self.q_function)

        self.q_function.to(self.device)
        self.target_q_function.to(self.device)

        self.buffer = ReplayBuffer(
            env=self.env,
            buffer_size=self.buffer_size,
            device=self.device,
            random_state=self.random_state,
        )
        self.update_count = 0

        self.optimizer = optim.SGD(
            self.q_function.parameters(), lr=self.lr, momentum=0.9
        )

        self.random_ = check_random_state(self.random_state)

    def save(self, path: Path):
        torch.save(self.q_function.state_dict(), path)

    def load(self, path: Path):
        self.q_function.load_state_dict(torch.load(path))
        self.target_q_function.load_state_dict(torch.load(path))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.add(state, action, reward, next_state, done)
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        current_q = self.q_function(state, action)
        with torch.no_grad():
            target_q = (
                reward + self.gamma * self.target_q_function.max(next_state) * done
            )

        td_loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        if self.update_count % self.target_update_interval == 0:
            self.target_q_function.load_state_dict(self.q_function.state_dict())
        self.update_count += 1

    def sample_action(self, state: np.ndarray, epsilon: float = 0.0):
        greedy_action = self.predict(state)
        random_action = self.random_.choice(self.n_actions)
        return random_action if self.random_.binomial(1, epsilon) else greedy_action

    def sample_action_softmax(self, state: np.ndarray, tau: float = 1.0):
        logits = self.predict_values(state) / tau
        logits = logits - logits.max()
        action_prob = np.exp(logits) / np.exp(logits).sum()
        return self.random_.choice(self.n_actions, p=action_prob)

    def predict(self, state: np.ndarray):
        state = state.reshape((1, -1))
        with torch.no_grad():
            action = self.q_function.argmax(to_tensor(state, device=self.device))[0]
        return int(action)

    def predict_values(self, state: np.ndarray):
        state = state.reshape((1, -1))
        with torch.no_grad():
            values = self.q_function.values(to_tensor(state, device=self.device))
        return values.to("cpu").detach().numpy()


@dataclass
class CloneOnlinePolicy:
    env: PartiallyObservableCartpole
    path: Path
    hidden_dim: int = 32
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        self.n_actions = self.env.action_space.n

        self.q_function = QFunction(
            state_dim=self.env.original_observation_space.shape[0],
            n_actions=self.env.action_space.n,
            hidden_dim=self.hidden_dim,
        )
        self.q_function.load_state_dict(torch.load(self.path))
        self.q_function.to(self.device)

        self.random_ = check_random_state(self.random_state)

    def sample_action(self, state: np.ndarray, epsilon: float = 0.0):
        greedy_action = self.predict(state)
        random_action = self.random_.choice(self.n_actions)
        return random_action if self.random_.binomial(1, epsilon) else greedy_action

    def sample_action_softmax(self, state: np.ndarray, tau: float = 1.0):
        logits = self.predict_values(state) / tau
        logits = logits - logits.max()
        action_prob = np.exp(logits) / np.exp(logits).sum()
        return self.random_.choice(self.n_actions, p=action_prob)

    def predict(self, state: np.ndarray):
        state = state.reshape((1, -1))
        with torch.no_grad():
            action = self.q_function.argmax(to_tensor(state, device=self.device))[0]
        return int(action)

    def predict_values(self, state: np.ndarray):
        state = state.reshape((1, -1))
        with torch.no_grad():
            values = self.q_function.values(to_tensor(state, device=self.device))
        return values.to("cpu").detach().numpy()
