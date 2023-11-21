import numpy as np
import random
from bandit_util import BanditMachine
import copy

def ucb(_bandit_machine: BanditMachine, T):
    
    bandit_machine = copy.deepcopy(_bandit_machine)
    K = bandit_machine.num_arms

    U = np.ones(K) * np.inf
    mu = np.zeros(K)
    N = np.zeros(K)
    total_reward = 0.0
    total_reward_per_round = np.zeros(T)

    total_exp_reward = 0.0
    total_exp_reward_per_round = np.zeros(T)

    total_best_reward = 0.0
    total_best_exp_reward_per_round = np.zeros(T)

    
    delta = 1.0 / T**2
    std_coeff = np.sqrt(2.0 * np.log(1.0/delta))

    best_mean, best_arm = bandit_machine.get_max_mean()

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
            N.resize(K)
            best_mean, best_arm = bandit_machine.get_max_mean()

        a = np.argmax(U) # in case of ties, argmax returns the index of first occurance
        reward = bandit_machine.arms[a].pull()
        mu[a] = (mu[a] * N[a] + reward) / (N[a] + 1.0)
        N[a] += 1.0
        U[a] = mu[a] + std_coeff / np.sqrt(N[a])

        
        total_reward += reward
        total_reward_per_round[t] = total_reward

        total_exp_reward += bandit_machine.arms[a].expected_reward()
        total_exp_reward_per_round[t] = total_exp_reward

        total_best_reward += best_mean
        total_best_exp_reward_per_round[t] = total_best_reward



    return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round