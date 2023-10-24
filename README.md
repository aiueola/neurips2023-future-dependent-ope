## Future-Dependent Value-Based Off-Policy Evaluation in POMDPs

---

### About
This repository contains the code to replicate the semi-synthetic experiments conducted in the paper "[Future-Dependent Value-Based Off-Policy Evaluation in POMDPs](https://arxiv.org/abs/2207.13081)" by [Masatoshi Uehara](https://www.masatoshiuehara.com/), [Haruka Kiyohara](https://sites.google.com/view/harukakiyohara), Andrew Bennett, Victor Chernozhukov, Nan Jiang, Nathan Kallus, Chengchun Shi, and Wen Sun, which has been accepted to [NeurIPS2023](https://nips.cc/Conferences/2023) as a Spotlight. 

<details>
<summary><strong>Click here to show the abstract </strong></summary>

We study off-policy evaluation (OPE) for partially observable MDPs (POMDPs) with general function approximation. Existing methods such as sequential importance sampling estimators suffer from the curse of horizon in POMDPs. To circumvent this problem, we develop a novel model-free OPE method by introducing future-dependent value functions that take future proxies as inputs. Future-dependent value functions play and perform a similar role to that of classical value functions in fully-observable MDPs. We derive a new off-policy Bellman equation for future-dependent value functions as conditional moment equations that use history proxies as instrumental variables. We further propose a minimax learning method to learn future-dependent value functions using the new Bellman equation. We obtain the PAC result, which implies our OPE estimator is close to the true policy value as long as futures and histories contain sufficient information about latent states, and the Bellman completeness. 

</details>

If you find this code useful in your research then please site:
```
@artile{uehara2023future,
  author = {Masatoshi Uehara, Haruka Kiyohara, Andrew Bennett, Victor Chernozhukov, Nan Jiang, Nathan Kallus, Chengchun Shi, and Wen Sun},
  title = {Future-Dependent Value-Based Off-Policy Evaluation in POMDPs},
  journal = {Advances in Neural Information Processing Systems},
  volume = {xxx},
  pages = {xxx -- xxx},
  year = {2023},
}
```

### Dependencies
This repository supports Python 3.7 or newer.

- numpy==1.22.4
- pandas==1.5.3
- scikit-learn==1.0.2
- matplotlib==3.7.1
- torch==2.0.0
- d3rlpy==1.1.1
- hydra-core==1.3.2

### Running the code
To conduct the experiments with CartPole, run the following command.

(i) learning policies
```bash
python src/online_learning.py online_learning.noise_param=0.0
python src/online_learning.py online_learning.noise_param=0.3
```

(ii) OPE
```bash
python src/evaluation_neural.py
```
