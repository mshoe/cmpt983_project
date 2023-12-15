import numpy as np
import random
from util.bandit_util import BanditMachine
from util.bandit_algorithm import MVBanditAlgorithm
import copy
import abc

class etc_mv(MVBanditAlgorithm):
    def __init__(self, _bandit_machine: BanditMachine, rho=1, cutoffs=0):
        self._bandit_machine = copy.deepcopy(_bandit_machine)
        self._rho = rho
        self._cutoffs = cutoffs
        return
    
    def run_alg(self, T: int, bandit_machine=None):
        if  bandit_machine is None:
            bandit_machine = copy.deepcopy(self._bandit_machine)

        init_num_arms = bandit_machine.num_arms
        
        sample_means = np.zeros(shape=(init_num_arms,), dtype=float)
        sample_vars = np.zeros(shape=(init_num_arms,), dtype=float)
        sample_goodness = np.zeros(shape=(init_num_arms,), dtype=float)
        sample_pulls = np.zeros(shape=(init_num_arms,), dtype=int)
        
        total_reward_per_round = np.zeros(shape=(T,), dtype=float)
        total_reward = 0.0

        best_mean, best_arm = bandit_machine.get_max_mean()
        
        total_best_exp_reward_per_round = np.zeros(shape=(T,), dtype=float)
        total_best_exp_reward = 0.0

        total_exp_reward_per_round = np.zeros(shape=(T,), dtype=float)
        total_exp_reward = 0.0

        total_mv = 0
        total_mv_per_round = np.zeros(T)

        total_best_reward = 0.0
        total_best_exp_reward_per_round = np.zeros(T)

        # Measures the amount the algorithm is below the mean of the optimal MEAN arm.
        # This is a measure of bad results that we want to minimize
        total_below_best_mean = 0.0
        total_below_best_mean_per_round = np.zeros(T)
        
        total_best_mv = 0
        total_best_mv_per_round = np.zeros(T)

        # select arms until max rounds
        num_arms = init_num_arms
        
        best_mv, _ = bandit_machine.get_max_mv(self._rho)
        count_below_cutoff = np.zeros(len(self._cutoffs))
        count_below_cutoff_per_round = np.zeros(shape=(len(self._cutoffs), T))

        for t in range(0, T):

            # check if arms are acquired this round
            arms_added = bandit_machine.acquire_arms(t)
            if arms_added:
                num_arms = bandit_machine.num_arms
                sample_means.resize(num_arms)
                sample_pulls.resize(num_arms)
                sample_vars.resize(num_arms)
                sample_goodness.resize(num_arms)
                best_mean, best_arm = bandit_machine.get_max_mean()
                best_mv, _ = bandit_machine.get_max_mv(self._rho)

            if np.min(sample_pulls) < self._m:
                a = sample_pulls.argmin()
            else:
                a = sample_goodness.argmax()

            # update sample means and number of pulls
            sample_reward = bandit_machine.arms[a].pull()
            sample_means[a] = (sample_pulls[a] * sample_means[a] + sample_reward) / (sample_pulls[a] + 1.0)
            sample_vars[a] = (sample_vars[a] * sample_pulls[a] + (sample_reward - sample_means[a]) ** 2) / (sample_pulls[a] + 1.0)
            sample_goodness[a] = sample_means[a] - self._rho * sample_vars[a]
            sample_pulls[a] += 1

            # update reward and best reward for plots
            total_reward += sample_reward
            total_reward_per_round[t] = total_reward

            total_exp_reward += bandit_machine.arms[a].expected_reward()
            total_exp_reward_per_round[t] = total_exp_reward

            total_mv += bandit_machine.arms[a].expected_mv_reward(self._rho)
            total_mv_per_round[t] = total_mv
            total_best_mv += best_mv
            total_best_mv_per_round[t] = total_best_mv
            total_best_reward += best_mean
            total_best_exp_reward_per_round[t] = total_best_reward

            for idx,c in enumerate(self._cutoffs):
                count_below_cutoff[idx] += 1 if sample_reward < c else 0
                count_below_cutoff_per_round[idx][t] = count_below_cutoff[idx]

            total_below_best_mean += max(0, best_mean - sample_reward)
            total_below_best_mean_per_round[t] = total_below_best_mean

        regret = total_best_exp_reward_per_round - total_exp_reward_per_round
        regret_mv = total_best_mv_per_round - total_mv_per_round

        return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round, count_below_cutoff_per_round, total_best_mv_per_round, total_mv_per_round, regret, regret_mv, total_below_best_mean_per_round


class etc_mv_alg(etc_mv):
    def __init__(self, _bandit_machine: BanditMachine, m=10, rho=1, **kwargs):
        super().__init__(_bandit_machine, **kwargs)
        self._m = m
        self._rho = rho
        return
    