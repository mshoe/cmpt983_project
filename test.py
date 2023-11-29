from util.bandit_util import StochasticGaussianArmAcquiringMachine, StochasticBernoulliArmAcquiringMachine
from algorithms.eps_greedy import eps_greedy, decay_eps_greedy
from algorithms.etc import etc
from algorithms.ucb import ucb_basic, auer, ucb_AO
from algorithms.oracle import oracle
import matplotlib.pyplot as plt
import numpy as np

num_rounds = 10000

acquire_probability = 0.002
min_mean = 0.1
max_mean = 0.9
min_var = 0.1
max_var = 0.5

# bandit_machine = StochasticGaussianArmAcquiringMachine(acquire_probability, min_mean, max_mean, min_var, max_var)
# bandit_machine.insert_arm(bandit_machine.generate_arm())
# bandit_machine.insert_arm(bandit_machine.generate_arm())
bandit_machine = StochasticBernoulliArmAcquiringMachine(acquire_probability, min_mean, max_mean)
bandit_machine.insert_arm(bandit_machine.generate_arm())
bandit_machine.insert_arm(bandit_machine.generate_arm())

num_trials = 100
# # UCB test
# print("Running UCB tests")
# ucb_alg = ucb_basic(bandit_machine, num_rounds)
# ucb_regret, ucb_reward, ucb_exp_reward, ucb_best_exp_reward = ucb_alg.run_experiment(num_trials, num_rounds)

# # # AUER test
# print("Running AUER tests")
# auer_alg = auer(bandit_machine)
# auer_regret, auer_reward, auer_exp_reward, auer_best_exp_reward = auer_alg.run_experiment(num_trials, num_rounds)

# # # UCB - Asymptotically Optimal
# print("Running UCB AO tests")
# ucbao_alg = ucb_AO(bandit_machine)
# ucbao_regret, ucbao_reward, ucbao_exp_reward, ucbao_best_exp_reward = ucbao_alg.run_experiment(num_trials, num_rounds)

# # # eps-greedy test
# print("Running eps greedy tests")
# initial_eps = 1.0
# epsg_alg = eps_greedy(bandit_machine, initial_eps)
# epsg_regret, epsg_reward, epsg_exp_reward, epsg_best_exp_reward = epsg_alg.run_experiment(num_trials, num_rounds)

# # decay eps-greedy test
# print("Running decay eps greedy tests")
# decay_factor = 0.99
# depsg_alg = decay_eps_greedy(bandit_machine, initial_eps, decay_factor)
# depsg_regret, depsg_reward, depsg_exp_reward, depsg_best_exp_reward = depsg_alg.run_experiment(num_trials, num_rounds)

# # etc
# explore_rounds = 10
# etc_alg = etc(bandit_machine, explore_rounds)
# etc_regret, etc_reward, etc_exp_reward, etc_best_exp_reward = etc_alg.run_experiment(num_trials, num_rounds)

# oracle
oracle_alg = oracle(bandit_machine)
oracle_regret, oracle_reward, oracle_exp_reward, oracle_best_exp_reward = oracle_alg.run_experiment(num_trials, num_rounds)

# plt.plot(ucb_regret, 'b', label="ucb")
# plt.plot(auer_regret, 'purple', label="auer")
# plt.plot(ucbao_regret, 'orange', label="ucb-ao")
# plt.plot(epsg_regret, 'r', label="eps-greedy")
# plt.plot(depsg_regret, 'g', label="decay-eps-greedy")
# plt.plot(etc_regret, 'pink', label="etc-10")
plt.plot(oracle_regret, 'black', label='oracle')

plt.xlabel('t')
plt.ylabel('Regret')
plt.title('Regret vs t')
plt.legend()
plt.grid(True)
plt.show()