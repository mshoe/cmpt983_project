import numpy as np
import random
from bandit_util import BanditMachine
import copy

def const_eps_greedy(_bandit_machine: BanditMachine, num_rounds, eps):
    bandit_machine = copy.deepcopy(_bandit_machine)

    init_num_arms = bandit_machine.num_arms
    
    sample_means = np.zeros(shape=(init_num_arms,), dtype=float)
    sample_pulls = np.zeros(shape=(init_num_arms,), dtype=int)
    
    total_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
    total_reward = 0.0

    best_mean, best_arm = bandit_machine.get_max_mean()
    
    total_best_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
    total_best_exp_reward = 0.0

    total_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
    total_exp_reward = 0.0

    # select arms until max rounds
    num_arms = init_num_arms
    all_arms_pulled_once = False
    for t in range(0, num_rounds):

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


def decay_eps_greedy(_bandit_machine: BanditMachine, num_rounds, eps):
    bandit_machine = copy.deepcopy(_bandit_machine)

    init_num_arms = bandit_machine.num_arms
    
    sample_means = np.zeros(shape=(init_num_arms,), dtype=float)
    sample_pulls = np.zeros(shape=(init_num_arms,), dtype=int)
    
    total_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
    total_reward = 0.0

    best_mean, best_arm = bandit_machine.get_max_mean()
    
    total_best_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
    total_best_exp_reward = 0.0

    total_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
    total_exp_reward = 0.0

    initial_eps = eps
    # select arms until max rounds
    num_arms = init_num_arms
    all_arms_pulled_once = False
    for t in range(0, num_rounds):

        # check if arms are acquired this round
        arms_added = bandit_machine.acquire_arms()
        if arms_added:
            num_arms = bandit_machine.num_arms
            sample_means.resize(num_arms)
            sample_pulls.resize(num_arms)
            best_mean, best_arm = bandit_machine.get_max_mean()
            all_arms_pulled_once = False
            eps = initial_eps

        # exploit or explore
        val = random.random()
        if val <= 1.0 - eps:
            a = sample_means.argmax()
        else:
            a = random.randint(0, num_arms-1)

        # decay epsilon
        eps *= 0.99

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