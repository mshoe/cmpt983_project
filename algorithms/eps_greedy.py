import numpy as np
import random
from util.bandit_util import BanditMachine
from util.bandit_algorithm import BanditAlgorithm
import copy
import abc

class eps_greedy(BanditAlgorithm):
    def __init__(self, _bandit_machine: BanditMachine, initial_eps):
        self._initial_eps = initial_eps
        self._bandit_machine = copy.deepcopy(_bandit_machine)
        return
    
    @abc.abstractmethod
    def _get_eps(self, t):
        # called once a round
        return self._initial_eps
    
    def run_alg(self, T: int):
        bandit_machine = copy.deepcopy(self._bandit_machine)

        init_num_arms = bandit_machine.num_arms
        
        sample_means = np.zeros(shape=(init_num_arms,), dtype=float)
        sample_pulls = np.zeros(shape=(init_num_arms,), dtype=int)
        
        total_reward_per_round = np.zeros(shape=(T,), dtype=float)
        total_reward = 0.0

        best_mean, best_arm = bandit_machine.get_max_mean()
        
        total_best_exp_reward_per_round = np.zeros(shape=(T,), dtype=float)
        total_best_exp_reward = 0.0

        total_exp_reward_per_round = np.zeros(shape=(T,), dtype=float)
        total_exp_reward = 0.0

        # select arms until max rounds
        num_arms = init_num_arms
        all_arms_pulled_once = False
        for t in range(0, T):

            # check if arms are acquired this round
            arms_added = bandit_machine.acquire_arms()
            if arms_added:
                num_arms = bandit_machine.num_arms
                sample_means.resize(num_arms)
                sample_pulls.resize(num_arms)
                best_mean, best_arm = bandit_machine.get_max_mean()
                all_arms_pulled_once = False

            # exploit or explore
            val = random.random()
            eps = self._get_eps(t)
            if val <= 1.0 - eps:
                a = sample_means.argmax()
            else:
                a = random.randint(0, num_arms-1)

            # mandatory exploration of unpulled arms
            if not all_arms_pulled_once:
                for arm_index in range(num_arms):
                    if sample_pulls[arm_index] == 0:
                        a = arm_index
                        break
                if arm_index >= num_arms - 1:
                    all_arms_pulled_once = True

            # update sample means and number of pulls
            sample_reward = bandit_machine.arms[a].pull()
            sample_means[a] = (sample_pulls[a] * sample_means[a] + sample_reward) / (sample_pulls[a] + 1.0)
            sample_pulls[a] += 1

            # update reward and best reward for plots
            total_reward += sample_reward
            total_reward_per_round[t] = total_reward

            total_exp_reward += bandit_machine.arms[a].expected_reward()
            total_exp_reward_per_round[t] = total_exp_reward

            total_best_exp_reward += best_mean
            total_best_exp_reward_per_round[t] = total_best_exp_reward

        return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round


class decay_eps_greedy(eps_greedy):
    def __init__(self, _bandit_machine: BanditMachine, initial_eps, decay_factor):
        super().__init__(_bandit_machine, initial_eps)
        self._curr_eps = initial_eps
        self._decay_factor = decay_factor
        return
    
    def _get_eps(self, t):
        eps = self._curr_eps
        self._curr_eps *= self._decay_factor
        return eps