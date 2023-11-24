from util.bandit_util import StochasticGaussianArmAcquiringMachine
from algorithms.eps_greedy import eps_greedy, decay_eps_greedy, decay_eps_greedy2
from algorithms.ucb import ucb_basic, auer, ucb_AO
import matplotlib.pyplot as plt
import numpy as np

acquire_probability = 0.002
min_mean = 0.1
max_mean = 0.9
min_var = 0.3
max_var = 0.5

bandit_machine = StochasticGaussianArmAcquiringMachine(acquire_probability, min_mean, max_mean, min_var, max_var)
bandit_machine.insert_arm(bandit_machine.generate_arm())
bandit_machine.insert_arm(bandit_machine.generate_arm())

num_rounds = 10000
eps = 0.1

num_trials = 20
num_trials = 20
# UCB test
print("Running UCB tests")
ucb_alg = ucb_basic(bandit_machine, num_rounds)
ucb_regret, ucb_reward, ucb_exp_reward, ucb_best_exp_reward = ucb_alg.run_experiment(num_trials, num_rounds)

# AUER test
print("Running AUER tests")
auer_alg = auer(bandit_machine)
auer_regret, auer_reward, auer_exp_reward, auer_best_exp_reward = auer_alg.run_experiment(num_trials, num_rounds)

# UCB - Asymptotically Optimal
print("Running UCB AO tests")
ucbao_alg = ucb_AO(bandit_machine)
ucbao_regret, ucbao_reward, ucbao_exp_reward, ucbao_best_exp_reward = ucbao_alg.run_experiment(num_trials, num_rounds)

# eps-greedy test
print("Running eps greedy tests")
initial_eps = 0.1
epsg_alg = eps_greedy(bandit_machine, initial_eps)
epsg_regret, epsg_reward, epsg_exp_reward, epsg_best_exp_reward = epsg_alg.run_experiment(num_trials, num_rounds)

# decay eps-greedy test
print("Running decay eps greedy tests")
decay_factor = 0.995
depsg_alg = decay_eps_greedy(bandit_machine, initial_eps, decay_factor)
depsg_regret, depsg_reward, depsg_exp_reward, depsg_best_exp_reward = depsg_alg.run_experiment(num_trials, num_rounds)

# decay eps-greedy test 2 (eps(t) = C / t)
print("Running decay eps greedy tests")
decay_constant = 10
depsg2_alg = decay_eps_greedy2(bandit_machine, initial_eps, decay_constant)
depsg2_regret, depsg2_reward, depsg2_exp_reward, depsg2_best_exp_reward = depsg2_alg.run_experiment(num_trials, num_rounds)


plt.plot(ucb_regret, 'b', label="ucb")
plt.plot(auer_regret, 'purple', label="auer")
plt.plot(ucbao_regret, 'orange', label="ucb-ao")
plt.plot(epsg_regret, 'r', label="eps-greedy")
plt.plot(depsg_regret, 'g', label="decay-eps-greedy")
plt.plot(depsg2_regret, 'black', label="decay-eps-greedy-2")
plt.xlabel('t')
plt.ylabel('Regret')
plt.title('Regret vs t')
plt.legend()
plt.grid(True)
plt.show()