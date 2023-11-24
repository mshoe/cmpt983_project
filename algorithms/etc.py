import numpy as np
import random
from util.bandit_util import BanditMachine
from util.bandit_algorithm import BanditAlgorithm
import copy
import abc

class etc_classic(BanditAlgorithm):
    def __init__(self, _bandit_machine: BanditMachine, explore_rounds):
        self.explore_rounds = explore_rounds 
        self._bandit_machine = copy.deepcopy(_bandit_machine)
        return
    
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


            a = sample_means.argmax()

            # mandatory exploration of unpulled arms
            if not all_arms_pulled_once:
                for arm_index in range(num_arms):
                    if sample_pulls[arm_index] < self.explore_rounds:
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

class etc(BanditAlgorithm):
    def __init__(self, _bandit_machine: BanditMachine, explore_rounds):
        self.explore_rounds = explore_rounds 
        self._bandit_machine = copy.deepcopy(_bandit_machine)
        return
    
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


            a = sample_means.argmax()

            # mandatory exploration of unpulled arms
            if not all_arms_pulled_once:
                for arm_index in range(num_arms):
                    if sample_pulls[arm_index] < self.explore_rounds:
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