import time
import pickle
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Union

import hydra
from omegaconf import DictConfig

import torch
import numpy as np

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteBC as BC
from d3rlpy.models.encoders import VectorEncoderFactory

from envs.taxi import PartiallyObservableTaxi
from envs.frozenlake import PartiallyObservableFrozenLake
from envs.cartpole import PartiallyObservableCartpole
from policy.encoder import EmbeddingEncoderFactory, LinearEncoderFactory
from policy.tabular import BaseTabularPolicy, TabularEpsilonGreedyPolicy
from policy.online import OnlinePolicy, CloneOnlinePolicy
from utils import format_runtime, LoggedDataset


def train_guide_policy_online_taxi(
    env: PartiallyObservableTaxi,
    epsilon: float = 0.3,
    alpha: float = 0.9,
    gamma: float = 1.0,
    random_state: Optional[int] = None,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    policy = TabularEpsilonGreedyPolicy(
        n_states=env.original_observation_space.n,
        n_actions=env.action_space.n,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        random_state=random_state,
    )

    path = Path(save_path + "expert" + save_name + ".npy")
    if path.exists():
        path = Path(save_path + "poor" + save_name + ".npy")
        policy.load(path)
        poor_policy = deepcopy(policy)

        path = Path(save_path + "medium" + save_name + ".npy")
        policy.load(path)
        medium_policy = deepcopy(policy)

        path = Path(save_path + "expert" + save_name + ".npy")
        policy.load(path)
        expert_policy = deepcopy(policy)

    else:
        path_ = Path(save_path)
        path_.mkdir(exist_ok=True, parents=True)

        done = True
        for i in tqdm(
            np.arange(50000),
            desc="[online_policy_learning_taxi]",
            total=50000,
        ):
            if done:
                memory_state = env.reset(return_state=True)
                next_memory, next_state = memory_state

            memory, state = next_memory, next_state
            action = policy.sample_action(state)
            memory_state, reward, done, _ = env.step(action, return_state=True)
            next_memory, next_state = memory_state

            policy.update(state, action, reward, next_state)

            if i + 1 == 10000:
                policy.save(save_path + "poor" + save_name)
                poor_policy = deepcopy(policy)

            elif i + 1 == 30000:
                policy.save(save_path + "medium" + save_name)
                medium_policy = deepcopy(policy)

            elif i + 1 == 50000:
                policy.save(save_path + "expert" + save_name)
                expert_policy = deepcopy(policy)

    return poor_policy, medium_policy, expert_policy


def train_guide_policy_online_frozenlake(
    env: PartiallyObservableFrozenLake,
    epsilon: float = 0.3,
    alpha: float = 0.9,
    gamma: float = 1.0,
    random_state: Optional[int] = None,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    policy = TabularEpsilonGreedyPolicy(
        n_states=env.original_observation_space.n,
        n_actions=env.action_space.n,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        random_state=random_state,
    )

    path = Path(save_path + "expert" + save_name + ".npy")
    if path.exists():
        path = Path(save_path + "poor" + save_name + ".npy")
        policy.load(path)
        poor_policy = deepcopy(policy)

        path = Path(save_path + "medium" + save_name + ".npy")
        policy.load(path)
        medium_policy = deepcopy(policy)

        path = Path(save_path + "expert" + save_name + ".npy")
        policy.load(path)
        expert_policy = deepcopy(policy)

    else:
        path_ = Path(save_path)
        path_.mkdir(exist_ok=True, parents=True)

        done = True
        for i in tqdm(
            np.arange(50000),
            desc="[online_policy_learning_frozenlake]",
            total=50000,
        ):
            if done:
                memory_state = env.reset(return_state=True)
                next_memory, next_state = memory_state

            memory, state = next_memory, next_state
            action = policy.sample_action(state)
            memory_state, reward, done, _ = env.step(action, return_state=True)
            next_memory, next_state = memory_state

            policy.update(state, action, reward, next_state)

            if i + 1 == 10000:
                policy.save(save_path + "poor" + save_name)
                poor_policy = deepcopy(policy)

            elif i + 1 == 30000:
                policy.save(save_path + "medium" + save_name)
                medium_policy = deepcopy(policy)

            elif i + 1 == 50000:
                policy.save(save_path + "expert" + save_name)
                expert_policy = deepcopy(policy)

    return poor_policy, medium_policy, expert_policy


def train_guide_policy_online_cartpole(
    env: PartiallyObservableCartpole,
    n_train_steps: int = 100000,
    final_epsilon: float = 0.3,
    gamma: float = 1.0,
    buffer_size: int = 10000,
    batch_size: int = 128,
    target_update_interval: int = 10,
    n_warmup_steps: int = 1000,
    lr: float = 1e-4,
    device: str = "cuda:0",
    random_state: Optional[int] = None,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    policy = OnlinePolicy(
        env=env,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_interval=target_update_interval,
        lr=lr,
        device=device,
        random_state=random_state,
    )

    path = Path(save_path + "expert" + save_name + ".pt")
    if path.exists():
        path = Path(save_path + "poor" + save_name + ".pt")
        policy.load(path)
        poor_policy = deepcopy(policy)

        path = Path(save_path + "medium" + save_name + ".pt")
        policy.load(path)
        medium_policy = deepcopy(policy)

        path = Path(save_path + "expert" + save_name + ".pt")
        policy.load(path)
        expert_policy = deepcopy(policy)

    else:
        path_ = Path(save_path)
        path_.mkdir(exist_ok=True, parents=True)

        poor_policy, medium_policy, expert_policy = None, None, None

        done = True
        for i in tqdm(
            np.arange(n_train_steps),
            desc="[online_policy_learning_cartpole]",
            total=n_train_steps,
        ):
            if done:
                memory_state = env.reset(return_state=True)
                next_memory, next_state = memory_state
                t = 0

            epsilon = max(1.0 - i / n_train_steps, final_epsilon)
            memory, state = next_memory, next_state
            action = policy.sample_action(state, epsilon=epsilon)
            memory_state, _, done, _ = env.step(action, return_state=True)
            next_memory, next_state = memory_state

            if done:
                if t < 30:
                    reward = -10
                elif t > 195:
                    reward = 10
                else:
                    reward = -1
            if t > 100:
                reward = 1

            if i > n_warmup_steps:
                policy.update(state, action, reward, next_state, done)

            if (i + 1) % 1000 == 0:
                cumulative_reward = np.zeros(10)

                for j in range(10):
                    memory_state = env.reset(return_state=True)
                    memory, next_state = memory_state
                    done = False

                    while not done:
                        state = next_state
                        action = policy.sample_action(state)
                        memory_state, reward, done, _ = env.step(
                            action, return_state=True
                        )
                        cumulative_reward[j] += reward
                        memory, next_state = memory_state

                if 60 < cumulative_reward.mean() < 80 and poor_policy is None:
                    path = Path(save_path + "poor" + save_name + ".pt")
                    policy.save(path)
                    poor_policy = CloneOnlinePolicy(
                        env=env,
                        path=path,
                        device=device,
                        random_state=random_state,
                    )

                elif 120 < cumulative_reward.mean() < 150 and medium_policy is None:
                    path = Path(save_path + "medium" + save_name + ".pt")
                    policy.save(path)
                    medium_policy = CloneOnlinePolicy(
                        env=env,
                        path=path,
                        device=device,
                        random_state=random_state,
                    )

                elif 195 < cumulative_reward.mean() and expert_policy is None:
                    path = Path(save_path + "expert" + save_name + ".pt")
                    policy.save(path)
                    expert_policy = CloneOnlinePolicy(
                        env=env,
                        path=path,
                        device=device,
                        random_state=random_state,
                    )

            if (
                poor_policy is not None
                and medium_policy is not None
                and expert_policy is not None
            ):
                break

    return poor_policy, medium_policy, expert_policy


def obtain_guide_policy_dataset(
    env: Union[
        PartiallyObservableTaxi,
        PartiallyObservableCartpole,
        PartiallyObservableFrozenLake,
    ],
    env_name: str,
    guide_policy: Union[BaseTabularPolicy, CloneOnlinePolicy],
    memory_length: int,
    n_steps: int = 100000,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    if env_name in ["taxi", "frozenlake"]:
        memory_states_buffer = np.zeros((n_steps, memory_length + 1))
    else:  # "cartpole"
        memory_states_buffer = np.zeros((n_steps, (memory_length + 1) * 4))

    action_buffer = np.zeros(n_steps, dtype=int)
    reward_buffer = np.zeros(n_steps)
    done_buffer = np.zeros(n_steps)

    path = Path(save_path + save_name + ".pkl")
    if path.exists():
        with open(path, "rb") as f:
            logged_dataset = pickle.load(f)

    else:
        path_ = Path(save_path)
        path_.mkdir(exist_ok=True, parents=True)

        done = True
        for i in tqdm(
            np.arange(n_steps),
            desc="[obtain_guide_policy_dataset]",
            total=n_steps,
        ):
            if done:
                memory_state = env.reset(return_state=True)
                next_memory, next_state = memory_state
                done = False

            memory, state = next_memory, next_state
            action = guide_policy.sample_action(state)
            memory_state, reward, done, _ = env.step(action, return_state=True)
            next_memory, next_state = memory_state

            memory_states_buffer[i] = memory
            action_buffer[i] = action
            reward_buffer[i] = reward
            done_buffer[i] = done

        logged_dataset = {
            "memory_states": memory_states_buffer,
            "action": action_buffer,
            "reward": reward_buffer,
            "done": done_buffer,
        }

        with open(path, "wb") as f:
            pickle.dump(logged_dataset, f)

    return logged_dataset


def train_memory_policy_behavior_cloning(
    env: Union[
        PartiallyObservableTaxi,
        PartiallyObservableCartpole,
        PartiallyObservableFrozenLake,
    ],
    env_name: str,
    logged_dataset: LoggedDataset,
    n_train_steps: int = 50000,
    is_linear: bool = False,
    device: str = "cuda:0",
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    use_gpu = not (device == "cpu")

    if env_name in ["taxi", "frozenlake"]:
        if is_linear:
            bc = BC(
                encoder_factory=LinearEncoderFactory(
                    n_unique_states=env.original_observation_space.n,
                ),
                use_gpu=use_gpu,
            )
        else:
            bc = BC(
                encoder_factory=EmbeddingEncoderFactory(
                    n_unique_states=env.original_observation_space.n,
                    hidden_units=[32],
                ),
                use_gpu=use_gpu,
            )
    else:  # "cartpole"
        bc = BC(
            encoder_factory=VectorEncoderFactory(
                hidden_units=[32],
            ),
            use_gpu=use_gpu,
        )

    logged_dataset = MDPDataset(
        observations=logged_dataset["memory_states"],
        actions=logged_dataset["action"],
        rewards=logged_dataset["reward"],
        terminals=logged_dataset["done"],
        discrete_action=True,
    )

    if is_linear:
        path = Path(save_path + save_name + "_linear.pt")
    else:
        path = Path(save_path + save_name + ".pt")

    if path.exists():
        bc.build_with_env(env)
        bc.load_model(path)

    else:
        path_ = Path(save_path)
        path_.mkdir(exist_ok=True, parents=True)

        bc.fit(
            logged_dataset,
            n_steps=n_train_steps,
            n_steps_per_epoch=1000,
        )
        bc.save_model(path)


def process(
    env_name: str,
    memory_length: int,
    noise_param: float,
    gamma: float,
    is_linear: bool,
    device: str,
    random_state: int,
):
    policy_path = f"policies/{env_name}/"
    config = f"_{memory_length}_{noise_param}_{random_state}"

    if env_name == "taxi":
        env = PartiallyObservableTaxi(
            memory_length=memory_length,
            epsilon=noise_param,
            random_state=random_state,
        )
        guide_policies = train_guide_policy_online_taxi(
            env=env,
            gamma=gamma,
            random_state=random_state,
            save_path=policy_path + "guide_policy/",
            save_name=config,
        )

    elif env_name == "frozenlake":
        env = PartiallyObservableFrozenLake(
            memory_length=memory_length,
            epsilon=noise_param,
            random_state=random_state,
        )
        guide_policies = train_guide_policy_online_frozenlake(
            env=env,
            gamma=gamma,
            random_state=random_state,
            save_path=policy_path + "guide_policy/",
            save_name=config,
        )

    else:  # "cartpole"
        env = PartiallyObservableCartpole(
            memory_length=memory_length,
            noise=noise_param,
            random_state=random_state,
        )
        guide_policies = train_guide_policy_online_cartpole(
            env=env,
            gamma=gamma,
            device=device,
            random_state=random_state,
            save_path=policy_path + "guide_policy/",
            save_name=config,
        )

    guide_policy_names = ["poor", "medium", "expert"]
    for guide_policy, guide_policy_name in zip(guide_policies, guide_policy_names):
        logged_dataset = obtain_guide_policy_dataset(
            env=env,
            env_name=env_name,
            guide_policy=guide_policy,
            memory_length=memory_length,
            save_path=policy_path + "guide_dataset/",
            save_name=guide_policy_name + config,
        )
        train_memory_policy_behavior_cloning(
            env=env,
            env_name=env_name,
            logged_dataset=logged_dataset,
            is_linear=is_linear,
            device=device,
            save_path=policy_path + "base_policy/",
            save_name=guide_policy_name + config,
        )


def assert_configuration(cfg: DictConfig):
    env = cfg.online_learning.env
    assert env in ["taxi", "frozenlake", "cartpole"]

    memory_length = cfg.online_learning.memory_length
    assert isinstance(memory_length, int) and 0 <= memory_length

    noise_param = cfg.online_learning.noise_param
    if env in ["taxi", "frozenlake"]:
        assert isinstance(noise_param, float) and 0 <= noise_param <= 1
    else:
        assert isinstance(noise_param, float) and 0 <= noise_param

    gamma = cfg.online_learning.gamma
    assert isinstance(gamma, float) and 0 < gamma <= 0.95

    base_random_state = cfg.online_learning.base_random_state
    assert isinstance(base_random_state, int) and 0 <= base_random_state


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    # configurations
    assert_configuration(cfg)
    conf = {
        "env_name": cfg.online_learning.env,
        "memory_length": cfg.online_learning.memory_length,
        "noise_param": cfg.online_learning.noise_param,
        "gamma": cfg.online_learning.gamma,
        "is_linear": cfg.online_learning.is_linear,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "random_state": cfg.online_learning.base_random_state,
    }
    process(**conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
