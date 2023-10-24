from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class DiscreteStateLSTMVfunction(nn.Module):
    def __init__(
        self,
        n_states: int = 500,
        n_actions: int = 6,
        memory_length: int = 0,
        future_length: int = 0,
        lstm_hidden_dim: int = 10,
        linear_hidden_dim: int = 100,
    ):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.is_memory_dependent = memory_length > 0
        self.is_future_dependent = future_length > 0

        self.lstm = nn.LSTM(
            (n_states * n_actions),
            lstm_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.history_based_state_predictor = nn.Linear(lstm_hidden_dim * 2, n_states)
        self.memory_based_state_predictor = nn.Linear(lstm_hidden_dim * 2, n_states)
        self.future_based_state_predictor = nn.Linear(lstm_hidden_dim * 2, n_states)

        self.fc1 = nn.Linear(
            n_states * (1 + self.is_memory_dependent + self.is_future_dependent),
            linear_hidden_dim,
        )
        self.fc2 = nn.Linear(linear_hidden_dim, 1)

        self.ce_loss = nn.CrossEntropyLoss()

    def _encode_state(
        self,
        state: torch.Tensor,
    ):
        return F.one_hot(state, num_classes=self.n_states)

    def _encode_state_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        indexes = states * self.n_actions + actions
        return F.one_hot(indexes, num_classes=self.n_states * self.n_actions)

    def _encode_sequence(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        input = self._encode_state_action(states, actions).to(torch.float32)
        out = self.lstm(input)[1][0]
        return torch.cat([out[0], out[1]], dim=1)

    def _predict_state(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        input_type: str,
    ):
        input = self._encode_sequence(states, actions)

        if input_type == "history":
            x = self.history_based_state_predictor(input)
        elif input_type == "memory":
            x = self.memory_based_state_predictor(input)
        else:
            x = self.future_based_state_predictor(input)
        return F.softmax(x)

    def forward(
        self,
        state: torch.Tensor,  # O
        memory_states: Optional[torch.Tensor] = None,  # Z
        memory_actions: Optional[torch.Tensor] = None,  # Z
        future_states: Optional[torch.Tensor] = None,  # F \ O
        future_actions: Optional[torch.Tensor] = None,  # F
    ):
        state = self._encode_state(state).to(torch.float32)

        if self.is_memory_dependent:
            with torch.no_grad():
                state_ = self._predict_state(
                    memory_states, memory_actions, input_type="memory"
                )
            state = torch.cat((state, state_), dim=1)

        if self.is_future_dependent:
            with torch.no_grad():
                state_ = self._predict_state(
                    future_states, future_actions, input_type="future"
                )
            state = torch.cat((state, state_), dim=1)

        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x.squeeze()

    def state_prediction_loss(
        self,
        history_states: torch.Tensor,  # H
        history_actions: torch.Tensor,  # H
        memory_states: torch.Tensor,  # M
        memory_actions: torch.Tensor,  # M
        state: torch.Tensor,  # O
        future_states: Optional[torch.Tensor] = None,  # F \ O
        future_actions: Optional[torch.Tensor] = None,  # F
    ):
        state_history = self._predict_state(
            history_states, history_actions, input_type="history"
        )
        loss = self.ce_loss(state_history, state)

        if self.is_memory_dependent:
            state_memory = self._predict_state(
                memory_states, memory_actions, input_type="memory"
            )
            loss += self.ce_loss(state_memory, state)

        if self.is_future_dependent:
            state_future = self._predict_state(
                future_states, future_actions, input_type="future"
            )
            loss += self.ce_loss(state_future, state)

        return loss


class ContinuousStateLSTMVfunction(nn.Module):
    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 2,
        memory_length: int = 0,
        future_length: int = 0,
        emb_dim: int = 4,
        lstm_hidden_dim: int = 10,
        linear_hidden_dim: int = 100,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.is_memory_dependent = memory_length > 0
        self.is_future_dependent = future_length > 0

        # self.emb = nn.Embedding(num_embeddings=n_actions, embedding_dim=emb_dim)

        self.lstm = nn.LSTM(
            (state_dim + n_actions),
            lstm_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.history_based_state_predictor = nn.Linear(lstm_hidden_dim * 2, state_dim)
        self.memory_based_state_predictor = nn.Linear(lstm_hidden_dim * 2, state_dim)
        self.future_based_state_predictor = nn.Linear(lstm_hidden_dim * 2, state_dim)

        # self.fc1 = nn.Linear(
        #     state_dim * (1 + self.is_memory_dependent + self.is_future_dependent),
        #     linear_hidden_dim,
        # )
        self.fc1 = nn.Linear(
            state_dim + (state_dim + n_actions) * (memory_length + future_length),
            linear_hidden_dim,
        )
        self.fc2 = nn.Linear(linear_hidden_dim, 1)

        self.mse_loss = nn.MSELoss()

    def _encode_state_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        # action_embs = self.emb(actions)
        action_embs = F.one_hot(actions, num_classes=self.n_actions)
        return torch.cat([states, action_embs], dim=2)

    def _encode_sequence(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        use_lstm: bool = False,
    ):
        input = self._encode_state_action(states, actions)

        if use_lstm:
            out = self.lstm(input)[1][0]
            out = torch.cat([out[0], out[1]], dim=1)
        else:
            out = input.reshape((input.shape[0], -1))

        return out

    def _predict_state(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        input_type: str,
    ):
        input = self._encode_sequence(states, actions, use_lstm=True)

        if input_type == "history":
            x = self.history_based_state_predictor(input)
        elif input_type == "memory":
            x = self.memory_based_state_predictor(input)
        else:
            x = self.future_based_state_predictor(input)

        return F.softmax(x)

    def forward(
        self,
        state: torch.Tensor,  # O
        memory_states: Optional[torch.Tensor] = None,  # Z
        memory_actions: Optional[torch.Tensor] = None,  # Z
        future_states: Optional[torch.Tensor] = None,  # F \ O
        future_actions: Optional[torch.Tensor] = None,  # F
    ):
        if self.is_memory_dependent:
            with torch.no_grad():
                state_ = self._encode_sequence(
                    memory_states,
                    memory_actions,
                )
            state = torch.cat((state, state_), dim=1)

        if self.is_future_dependent:
            with torch.no_grad():
                state_ = self._encode_sequence(
                    future_states,
                    future_actions,
                )
            state = torch.cat((state, state_), dim=1)

        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x.squeeze()

    def state_prediction_loss(
        self,
        history_states: torch.Tensor,  # H
        history_actions: torch.Tensor,  # H
        memory_states: torch.Tensor,  # M
        memory_actions: torch.Tensor,  # M
        state: torch.Tensor,  # O
        future_states: Optional[torch.Tensor] = None,  # F \ O
        future_actions: Optional[torch.Tensor] = None,  # F
    ):
        state_history = self._predict_state(
            history_states, history_actions, input_type="history"
        )
        loss = self.mse_loss(state_history, state)

        # if self.is_memory_dependent:
        #     state_memory = self._predict_state(
        #         memory_states, memory_actions, input_type="memory"
        #     )
        #     loss += self.mse_loss(state_memory, state)

        # if self.is_future_dependent:
        #     state_future = self._predict_state(
        #         future_states, future_actions, input_type="future"
        #     )
        #     loss += self.mse_loss(state_future, state)

        return loss
