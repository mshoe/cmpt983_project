import numpy as np
import random
from util.bandit_util import BanditMachine
from util.bandit_algorithm import MVBanditAlgorithm
import copy
import abc

class ucb(MVBanditAlgorithm):
    def __init__(self, _bandit_machine: BanditMachine, rho=1, cutoffs=[0]):
        self._bandit_machine = copy.deepcopy(_bandit_machine)
        self._rho = rho
        self._cutoffs = cutoffs
        return
    
    @abc.abstractmethod
    def _get_std_coeff(self, t): return

    def run_alg(self, T: int, bandit_machine=None):
        if  bandit_machine is None:
            bandit_machine = copy.deepcopy(self._bandit_machine)
        else:
            delta = 1.0 / T**2
            max_var = max([a.variance() for a in self._bandit_machine.arms])
            self._std_coeff = np.sqrt(2 * max_var * np.log(1.0 / delta))

        K = bandit_machine.num_arms

        U = np.ones(K) * np.inf
        mu = np.zeros(K)
        var = np.zeros(K)
        goodness = np.zeros(K)
        N = np.zeros(K)
        total_reward = 0.0
        total_reward_per_round = np.zeros(T)

        total_mv = 0
        total_mv_per_round = np.zeros(T)

        total_exp_reward = 0.0
        total_exp_reward_per_round = np.zeros(T)

        total_best_reward = 0.0
        total_best_exp_reward_per_round = np.zeros(T)

        # Measures the amount the algorithm is below the mean of the optimal MEAN arm.
        # This is a measure of bad results that we want to minimize
        total_below_best_mean = 0.0
        total_below_best_mean_per_round = np.zeros(T)
        
        total_best_mv = 0
        total_best_mv_per_round = np.zeros(T)

        best_mean, best_arm = bandit_machine.get_max_mean()
        best_mv, _ = bandit_machine.get_max_mv(self._rho)

        count_below_cutoff = np.zeros(len(self._cutoffs))

        for t in range(0, T):

            # check if arms are acquired this round
            arms_added = bandit_machine.acquire_arms()
            if arms_added:
                old_K = K
                K = bandit_machine.num_arms
                U.resize(K)
                for i in range(old_K, K):
                    U[i] = np.inf
                mu.resize(K)
                var.resize(K)
                N.resize(K)
                goodness.resize(K)
                best_mean, best_arm = bandit_machine.get_max_mean()
                best_mv, _ = bandit_machine.get_max_mv(self._rho)

            a = np.argmax(U) # in case of ties, argmax returns the index of first occurance
            reward = bandit_machine.arms[a].pull()
            
            mu[a] = (mu[a] * N[a] + reward) / (N[a] + 1.0)
            var[a] = (var[a] * N[a] + (reward - mu[a]) ** 2) / (N[a] + 1.0)
            goodness[a] = mu[a] - self._rho * var[a]
            N[a] += 1.0
            
            std_coeff = self._get_std_coeff(t)
            U[a] = goodness[a] + std_coeff / np.sqrt(N[a])

            total_reward += reward
            total_reward_per_round[t] = total_reward

            total_exp_reward += bandit_machine.arms[a].expected_reward()
            total_exp_reward_per_round[t] = total_exp_reward

            total_mv += bandit_machine.arms[a].expected_mv_reward(self._rho)
            total_best_mv += best_mv
            total_best_reward += best_mean
            total_best_exp_reward_per_round[t] = total_best_reward
            total_best_mv_per_round[t] = total_best_mv
            total_mv_per_round[t] = total_mv

            for idx,c in enumerate(self._cutoffs):
                count_below_cutoff[idx] += 1 if reward < c else 0

            total_below_best_mean += max(0, best_mean - reward)
            total_below_best_mean_per_round[t] = total_below_best_mean

        regret = total_best_exp_reward_per_round - total_exp_reward_per_round
        regret_mv = total_best_mv_per_round - total_mv_per_round

        return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round, count_below_cutoff, total_best_mv_per_round, total_mv_per_round, regret, regret_mv, total_below_best_mean_per_round
    
class ucb_mv_basic(ucb):
    def __init__(self, _bandit_machine: BanditMachine, T: int, **kwargs):
        super().__init__(_bandit_machine, **kwargs)
        self.T = T
        return
    
    def _get_std_coeff(self, t): return self._std_coeff