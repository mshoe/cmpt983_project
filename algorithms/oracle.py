import numpy as np
import random
from util.bandit_util import BanditMachine
from util.bandit_algorithm import BanditAlgorithm
import copy

class oracle(BanditAlgorithm):
    def __init__(self, _bandit_machine: BanditMachine):
        self._bandit_machine = copy.deepcopy(_bandit_machine)
        return
    
    def run_alg(self, T: int):
        bandit_machine = copy.deepcopy(self._bandit_machine)

        K = bandit_machine.num_arms
        N = np.zeros(K)

        total_reward = 0.0
        total_reward_per_round = np.zeros(T)

        total_exp_reward = 0.0
        total_exp_reward_per_round = np.zeros(T)

        total_best_reward = 0.0
        total_best_exp_reward_per_round = np.zeros(T)

        best_mean, best_arm = bandit_machine.get_max_mean()

        for t in range(0, T):

            # check if arms are acquired this round
            arms_added = bandit_machine.acquire_arms()
            if arms_added:
                K = bandit_machine.num_arms
                N.resize(K)
                #print("arm added at t =", t)


            # decide the arm based on best mean
            best_mean, a = bandit_machine.get_max_mean()

            # decision is overwritten if there is any arm with 0 pulls
            sampling_new_arm = False
            for i in range(K):
                if N[i] == 0:
                    a = i
                    sampling_new_arm = True
                    break

            N[a] += 1
            reward = bandit_machine.arms[a].pull()
            # if sampling_new_arm:
            #     print(t)

            total_reward += reward
            total_reward_per_round[t] = total_reward

            total_exp_reward += bandit_machine.arms[a].expected_reward()
            total_exp_reward_per_round[t] = total_exp_reward

            total_best_reward += best_mean
            total_best_exp_reward_per_round[t] = total_best_reward

        return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round

            
