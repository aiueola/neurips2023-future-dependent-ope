import time
import pickle
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, Dict, List, Optional

import hydra
from omegaconf import DictConfig

import numpy as np
from pandas import DataFrame

import gym
import torch

from d3rlpy.algos import DiscreteBC as BC

from dataset.taxi import TabularTaxiDataset
from envs import (
    PartiallyObservableTaxi, 
    PartiallyObservableFrozenLake,
    PartiallyObservableCartpole,
)
from policy.policy import BasePolicy, EpsilonGreedyPolicy
from policy.encoder import EmbeddingEncoderFactory

from ope.sequential_is import TabularSequentialImportanceSampling
from ope.tabular import (
    TabularStateValueBasedOPE,
    TabularFutureDependentValueBasedOPE,
)

from utils import format_runtime, LoggedDataset


def load_policy(
    env: Union[PartiallyObservableTaxi, PartiallyObservableCartpole],
    epsilon: str,
    device: str = "cuda:0",
    random_state: Optional[int] = None,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    use_gpu = not (device == "cpu")

    base_policy = BC(
        encoder_factory=EmbeddingEncoderFactory(
            n_unique_states=env.original_observation_space.n,
            hidden_units=[32],
        ),
        use_gpu=use_gpu,
    )

    path = Path(save_path + save_name + ".pt")

    if path.exists():
        base_policy.build_with_env(env)
        base_policy.load_model(path)

    policy = EpsilonGreedyPolicy(
        base_policy=base_policy,
        n_actions=env.action_space.n,
        epsilon=epsilon,
        random_state=random_state,
    )
    return policy


def obtain_logged_dataset(
    env: Union[PartiallyObservableTaxi, PartiallyObservableCartpole],
    behavior_policy: BasePolicy,
    evaluation_policy: BasePolicy,
    n_trajectories: int,
    n_trajectories_initial_datasets: int,
    random_state: Optional[int] = None,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    path = Path(save_path + save_name + ".pkl")
    if path.exists():
        with open(path, "rb") as f:
            logged_datasets = pickle.load(f)
            (
                train_logged_dataset,
                sis_logged_dataset,
                initial_logged_dataset,
            ) = logged_datasets
    else:
        path_ = Path(save_path)
        path_.mkdir(exist_ok=True, parents=True)

        dataset = TabularTaxiDataset(
            env=env,
            behavior_policy=behavior_policy,
            evaluation_policy=evaluation_policy,
            random_state=random_state,
        )

        train_logged_dataset = dataset.obtain_logged_data_for_value_based(
            n_trajectories
        )
        sis_logged_dataset = dataset.obtain_logged_data_for_sis(n_trajectories)
        initial_logged_dataset = dataset.obtain_initial_logged_data(
            n_trajectories_initial_datasets
        )

        with open(path, "wb") as f:
            pickle.dump(
                (train_logged_dataset, sis_logged_dataset, initial_logged_dataset), f
            )

    return train_logged_dataset, sis_logged_dataset, initial_logged_dataset


def off_policy_evaluation(
    env: gym.Env,
    estimator_name: str,
    gamma: float,
    behavior_policy: BasePolicy,
    evaluation_policy: BasePolicy,
    train_logged_dataset: LoggedDataset,
    initial_logged_dataset: LoggedDataset,
    sis_logged_dataset: LoggedDataset,
    sigma: float,  # ope_config
    alpha: float,  # ope_config
    alpha_p: float,  # ope_config
    lambda_: float,  # ope_config
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    if estimator_name == "future_dependent":
        estimator = TabularFutureDependentValueBasedOPE(
            behavior_policy=behavior_policy,
            evaluation_policy=evaluation_policy,
            n_unique_states=env.original_observation_space.n,
            n_unique_actions=env.action_space.n,
            gamma=gamma,
            alpha=alpha,
            alpha_p=alpha_p,
            lambda_=lambda_,
        )
    elif estimator_name == "value_based":
        estimator = TabularStateValueBasedOPE(
            behavior_policy=behavior_policy,
            evaluation_policy=evaluation_policy,
            n_unique_states=env.original_observation_space.n,
            gamma=gamma,
            alpha=alpha,
            alpha_p=alpha_p,
            lambda_=lambda_,
        )
    else:  # "sequential_is"
        estimator = TabularSequentialImportanceSampling(
            behavior_policy=behavior_policy,
            evaluation_policy=evaluation_policy,
            gamma=gamma,
        )

    if estimator_name in ["value_based", "future_dependent"]:
        path = Path(save_path + save_name + f"_{estimator_name}.npy")

        if path.exists():
            estimator.load(path)

        else:
            path_ = Path(save_path)
            path_.mkdir(exist_ok=True, parents=True)

            estimator.fit(**train_logged_dataset)
            estimator.save(save_path + save_name + f"_{estimator_name}")

    return estimator.estimate_policy_value(
        **initial_logged_dataset, **sis_logged_dataset
    )


def estimate_on_policy_policy_value(
    env: gym.Env,
    evaluation_policy: BasePolicy,
    gamma: float,
    n_trajectories: int,
    random_state: Optional[int] = None,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    path = Path(save_path + save_name + ".npy")
    if path.exists():
        on_policy_policy_value = np.load(path)

    else:
        path_ = Path(save_path)
        path_.mkdir(exist_ok=True, parents=True)

        env.seed(random_state)
        on_policy_policy_value = np.zeros(n_trajectories)

        # burn-in
        next_memory = env.reset()
        for t in range(100):
            memory = next_memory
            action = evaluation_policy.sample_action(memory)
            (
                next_memory,
                reward,
                done,
                _,
            ) = env.step(action)

            if done:
                next_memory = env.reset()

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[calc_on_policy_policy_value]",
            total=n_trajectories,
        ):
            cumulative_reward = 0
            discount = 1.0

            # as we set gamma <= 0.95, the discount rate at step = 100 is less than 0.01
            # therefore, we rollout 100 steps for each trajectory
            for t in range(100):
                memory = next_memory
                action = evaluation_policy.sample_action(memory)
                (
                    next_memory,
                    reward,
                    done,
                    _,
                ) = env.step(action)

                cumulative_reward += discount * reward
                discount *= gamma

                if done:
                    next_memory = env.reset()

            on_policy_policy_value[i] = cumulative_reward

        np.save(save_path + save_name, on_policy_policy_value)

    on_policy_policy_value = float(on_policy_policy_value.mean())
    return on_policy_policy_value


def load_estimation_result(
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    path = Path(save_path + save_name + ".npy")
    return np.load(path)


def save_estimation_result(
    estimation_result: np.ndarray,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    path_ = Path(save_path)
    path_.mkdir(exist_ok=True, parents=True)

    path = Path(save_path + save_name)
    np.save(path, estimation_result)


def aggregate_estimators_performance(
    on_policy_policy_value: float,
    estimation_result: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
):
    estimation_df = DataFrame()
    for estimator_name, estimation_arr in estimation_result.items():
        estimation_df[estimator_name] = estimation_arr
    estimation_df["on_policy_policy_value"] = on_policy_policy_value

    path_ = Path(save_path)
    path_.mkdir(exist_ok=True, parents=True)

    path = Path(save_path + save_name + ".csv")
    estimation_df.to_csv(path, index=False)


def process(
    experiment: str,
    env_name: str,
    compared_estimators: List[str],
    n_trajectories: int,
    history_length: int,
    memory_length: int,
    future_length: int,
    noise_param: float,
    behavior_epsilon: float,
    evaluation_epsilon: float,
    behavior_guide_policy_type: str,
    evaluation_guide_policy_type: str,
    gamma: float,
    device: str,
    start_random_state: int,
    n_random_state: int,
    base_random_state: int,
    ope_config: Dict[str, float],
):
    policy_path = f"policies/{env_name}/base_policy/"
    base_policy_config = f"_{memory_length}_{noise_param}_{base_random_state}"
    base_behavior_policy_config = f"_{memory_length}_{0.0}_{base_random_state}"

    log_path = f"logs/{experiment}/"
    model_path = f"models/{experiment}/"
    dataset_path = f"datasets/{experiment}/"
    config = f"{n_trajectories}_{history_length}_{memory_length}_{future_length}_{noise_param}_{behavior_guide_policy_type}_{behavior_epsilon}_{evaluation_guide_policy_type}_{evaluation_epsilon}_{gamma}_{base_random_state}"
    config_ = f"{memory_length}_{noise_param}_{evaluation_guide_policy_type}_{evaluation_epsilon}_{gamma}_{base_random_state}"

    if env_name == "taxi":
        env = PartiallyObservableTaxi(
            memory_length=memory_length,
            epsilon=noise_param,
            random_state=base_random_state,
        )
    elif env_name == "frozenlake":
        env = PartiallyObservableFrozenLake(
            memory_length=memory_length,
            epsilon=noise_param,
            random_state=base_random_state,
        )
    else:
        env = PartiallyObservableCartpole(
            memory_length=memory_length,
            noise=noise_param,
            random_state=base_random_state,
        )

    behavior_policy = load_policy(
        env=env,
        epsilon=behavior_epsilon,
        device=device,
        random_state=base_random_state,
        save_path=policy_path,
        save_name=f"{behavior_guide_policy_type}" + base_behavior_policy_config,
    )
    evaluation_policy = load_policy(
        env=env,
        device=device,
        epsilon=evaluation_epsilon,
        random_state=base_random_state,
        save_path=policy_path,
        save_name=f"{evaluation_guide_policy_type}" + base_policy_config,
    )
    on_policy_policy_value = estimate_on_policy_policy_value(
        env=env,
        evaluation_policy=evaluation_policy,
        gamma=gamma,
        n_trajectories=10000,
        random_state=base_random_state,
        save_path=log_path + "on_policy_policy_value/",
        save_name=config_,
    )

    estimation_result = {}
    for estimator in compared_estimators:
        estimation_result[estimator] = np.zeros(n_random_state)
        if start_random_state > 0:
            estimation_result[estimator][:start_random_state] = load_estimation_result(
                save_path=log_path + f"estimation_result/{estimator}/",
                save_name=config,
            )

    for random_state in range(start_random_state, n_random_state):
        (
            train_logged_dataset,
            sis_logged_dataset,
            initial_logged_dataset,
        ) = obtain_logged_dataset(
            env=env,
            behavior_policy=behavior_policy,
            evaluation_policy=evaluation_policy,
            n_trajectories=n_trajectories,
            n_trajectories_initial_datasets=10000,
            random_state=random_state,
            save_path=dataset_path,
            save_name=config + f"_{random_state}",
        )
        for estimator in compared_estimators:
            estimation_result[estimator][random_state] = off_policy_evaluation(
                env=env,
                estimator_name=estimator,
                gamma=gamma,
                behavior_policy=behavior_policy,
                evaluation_policy=evaluation_policy,
                train_logged_dataset=train_logged_dataset,
                initial_logged_dataset=initial_logged_dataset,
                sis_logged_dataset=sis_logged_dataset,
                save_path=model_path + "v_function/",
                save_name=config + f"_{random_state}",
                **ope_config,
            )

    for estimator in compared_estimators:
        save_estimation_result(
            estimation_result=estimation_result[estimator],
            save_path=log_path + f"estimation_result/{estimator}/",
            save_name=config,
        )
    aggregate_estimators_performance(
        on_policy_policy_value=on_policy_policy_value,
        estimation_result=estimation_result,
        save_path=log_path + "estimation_result/",
        save_name=f"n_trajectories={n_trajectories},evaluation_epsilon={evaluation_epsilon},noise_param={noise_param}",
    )


def assert_configuration(cfg: DictConfig):
    experiment = cfg.setting.experiment
    assert experiment in ["taxi_tabular", "frozenlake_tabular"]

    env = cfg.setting.env
    assert env in ["taxi", "frozenlake"]

    model = cfg.setting.model
    assert model == "tabular"

    assert experiment == env + "_" + model

    compared_estimators = cfg.setting.compared_estimators
    for estimator in compared_estimators:
        assert estimator in ["future_dependent", "value_based", "sis"]

    n_trajectories = cfg.setting.n_trajectories
    assert isinstance(n_trajectories, int) and 0 < n_trajectories

    memory_length = cfg.setting.memory_length
    assert isinstance(memory_length, int) and 0 <= memory_length
    if model == "tabular":
        assert memory_length == 0

    history_length = cfg.setting.history_length
    assert isinstance(history_length, int) and max(1, memory_length) <= history_length
    if model == "tabular":
        assert history_length == 1

    future_length = cfg.setting.future_length
    assert isinstance(future_length, int) and 0 <= future_length
    if model == "tabular":
        assert future_length == 0

    noise_param = cfg.setting.noise_param
    if env == "taxi":
        assert isinstance(noise_param, float) and 0 <= noise_param <= 1
    else:
        assert isinstance(noise_param, float) and 0 <= noise_param

    behavior_epsilon = cfg.setting.behavior_epsilon
    assert isinstance(behavior_epsilon, float) and 0 < behavior_epsilon <= 1

    evaluation_epsilon = cfg.setting.evaluation_epsilon
    for value in evaluation_epsilon:
        assert isinstance(value, float) and 0 <= value <= 1

    behavior_guide_policy_type = cfg.setting.behavior_guide_policy_type
    assert behavior_guide_policy_type in ["poor", "medium", "expert"]

    evaluation_guide_policy_type = cfg.setting.evaluation_guide_policy_type
    assert evaluation_guide_policy_type in ["poor", "medium", "expert"]

    gamma = cfg.setting.gamma
    assert isinstance(gamma, float) and 0 < gamma <= 0.95

    start_random_state = cfg.setting.start_random_state
    assert isinstance(start_random_state, int) and 0 <= start_random_state

    n_random_state = cfg.setting.n_random_state
    assert isinstance(n_random_state, int) and start_random_state < n_random_state

    base_random_state = cfg.setting.base_random_state
    assert isinstance(base_random_state, int) and 0 <= base_random_state

    sigma = cfg.ope_config.sigma
    assert isinstance(sigma, float) and 0 < sigma

    alpha = cfg.ope_config.alpha
    assert isinstance(alpha, float) and 0 <= alpha

    alpha_p = cfg.ope_config.alpha_p
    assert isinstance(alpha_p, float) and 0 <= alpha_p

    lambda_ = cfg.ope_config.lambda_
    assert isinstance(lambda_, float) and 0 <= lambda_

    if alpha_p > 0:
        assert alpha > 0

    # if model in ["future_dependent", "neural_baseline"]:
    #     assert alpha > 0 and lambda_ > 0


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    # configurations
    assert_configuration(cfg)
    conf = {
        "experiment": cfg.setting.experiment,
        "env_name": cfg.setting.env,
        "compared_estimators": cfg.setting.compared_estimators,
        "n_trajectories": cfg.setting.n_trajectories,
        "history_length": cfg.setting.history_length,
        "memory_length": cfg.setting.memory_length,
        "future_length": cfg.setting.future_length,
        "noise_param": cfg.setting.noise_param,
        "behavior_epsilon": cfg.setting.behavior_epsilon,
        "evaluation_epsilon": cfg.setting.evaluation_epsilon,
        "behavior_guide_policy_type": cfg.setting.behavior_guide_policy_type,
        "evaluation_guide_policy_type": cfg.setting.evaluation_guide_policy_type,
        "gamma": cfg.setting.gamma,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "start_random_state": cfg.setting.start_random_state,
        "n_random_state": cfg.setting.n_random_state,
        "base_random_state": cfg.setting.base_random_state,
        "ope_config": cfg.ope_config,
    }
    for value in conf["evaluation_epsilon"]:
        conf_ = deepcopy(conf)
        conf_["evaluation_epsilon"] = value
        process(**conf_)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
