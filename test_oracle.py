from util.bandit_util import DeterministicArmAcquiringMachine, StandardBanditMachine, GaussianBanditArm, BernoulliBanditArm
from util.bandit_util import BernoulliArmAcquiringMachine
from algorithms.eps_greedy import eps_greedy, decay_eps_greedy, decay_eps_greedy2
from algorithms.ucb import ucb_basic, auer, ucb_AO
from algorithms.etc import etc
from algorithms.oracle import oracle
import matplotlib.pyplot as plt
import numpy as np
import random

num_rounds = 10000

num_rounds_between_adding_arms = 1000

total_num_arms = num_rounds // num_rounds_between_adding_arms

# initial_arms = []
# additional_arms = []
# addition_times = []
# for i in range(0, total_num_arms):
#     mean = random.random()
#     arm = BernoulliBanditArm(mean)
#     additional_arms.append(arm)
#     addition_times.append(i * num_rounds_between_adding_arms)
# bandit_machine = DeterministicArmAcquiringMachine(initial_arms, additional_arms, addition_times)
bandit_machine = BernoulliArmAcquiringMachine(num_rounds_between_adding_arms, 0, 1)


num_trials = 10000
# # UCB test
# print("Running UCB tests")
# ucb_alg = ucb_basic(bandit_machine, num_rounds)
# ucb_regret, ucb_reward, ucb_exp_reward, ucb_best_exp_reward = ucb_alg.run_experiment(num_trials, num_rounds)

# # AUER test
# print("Running AUER tests")
# auer_alg = auer(bandit_machine)
# auer_regret, auer_reward, auer_exp_reward, auer_best_exp_reward = auer_alg.run_experiment(num_trials, num_rounds)

# # UCB - Asymptotically Optimal
# print("Running UCB AO tests")
# ucbao_alg = ucb_AO(bandit_machine)
# ucbao_regret, ucbao_reward, ucbao_exp_reward, ucbao_best_exp_reward = ucbao_alg.run_experiment(num_trials, num_rounds)

# # eps-greedy test
# print("Running eps greedy tests")
# initial_eps = 0.1
# epsg_alg = eps_greedy(bandit_machine, initial_eps)
# epsg_regret, epsg_reward, epsg_exp_reward, epsg_best_exp_reward = epsg_alg.run_experiment(num_trials, num_rounds)

# # decay eps-greedy test
# print("Running decay eps greedy tests")
# decay_factor = 0.995
# depsg_alg = decay_eps_greedy(bandit_machine, initial_eps, decay_factor)
# depsg_regret, depsg_reward, depsg_exp_reward, depsg_best_exp_reward = depsg_alg.run_experiment(num_trials, num_rounds)

# # decay eps-greedy test 2 (eps(t) = C / t)
# print("Running decay eps greedy tests")
# decay_constant = 10
# depsg2_alg = decay_eps_greedy2(bandit_machine, initial_eps, decay_constant)
# depsg2_regret, depsg2_reward, depsg2_exp_reward, depsg2_best_exp_reward = depsg2_alg.run_experiment(num_trials, num_rounds)


# # etc
# explore_rounds = 10
# etc_alg = etc(bandit_machine, explore_rounds)
# etc_regret, etc_reward, etc_exp_reward, etc_best_exp_reward = etc_alg.run_experiment(num_trials, num_rounds)

# # etc
# explore_rounds = 100
# etc_100_alg = etc(bandit_machine, explore_rounds)
# etc_100_regret, etc_100_reward, etc_100_exp_reward, etc_100_best_exp_reward = etc_100_alg.run_experiment(num_trials, num_rounds)

# # etc
# explore_rounds = 1000
# etc_1000_alg = etc(bandit_machine, explore_rounds)
# etc_1000_regret, etc_1000_reward, etc_1000_exp_reward, etc_1000_best_exp_reward = etc_1000_alg.run_experiment(num_trials, num_rounds)

# oracle
oracle_alg = oracle(bandit_machine)
oracle_regret, oracle_reward, oracle_exp_reward, oracle_best_exp_reward = oracle_alg.run_experiment(num_trials, num_rounds)

print("regret =", oracle_regret[-1])

# plt.plot(ucb_regret, 'b', label="ucb")
# plt.plot(auer_regret, 'purple', label="auer")
# plt.plot(ucbao_regret, 'orange', label="ucb-ao")
# plt.plot(epsg_regret, 'r', label="eps-greedy")
# plt.plot(depsg_regret, 'g', label="decay-eps-greedy")
# plt.plot(depsg2_regret, 'black', label="decay-eps-greedy-2")
# plt.plot(etc_regret, 'pink', label="etc-10")
# plt.plot(etc_100_regret, 'm', label="etc-100")
# plt.plot(etc_1000_regret, 'y', label="etc-1000")
plt.plot(oracle_regret, 'gray', label='oracle')
plt.xlabel('t')
plt.ylabel('Regret')
plt.title('Regret vs t')
plt.legend()
plt.grid(True)
plt.show()