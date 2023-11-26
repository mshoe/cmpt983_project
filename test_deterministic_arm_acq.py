from util.bandit_util import DeterministicArmAcquiringMachine, StandardBanditMachine, GaussianBanditArm, BernoulliBanditArm
from algorithms.eps_greedy import eps_greedy, decay_eps_greedy
from algorithms.ucb import ucb_basic, auer, ucb_AO, ucb_faster, ucb_fast
from algorithms.etc import etc
import matplotlib.pyplot as plt
import numpy as np

num_rounds = 10000
num_trials = 10

initial_arms = [BernoulliBanditArm(0.5), BernoulliBanditArm(0.3)]
additional_arms = [BernoulliBanditArm(0.6), BernoulliBanditArm(0.2), BernoulliBanditArm(0.8)]
#initial_arms = [GaussianBanditArm(0.5, 0.5), GaussianBanditArm(0.3, 0.5)]
#additional_arms = [GaussianBanditArm(0.6, 0.5), GaussianBanditArm(0.2, 0.5), GaussianBanditArm(0.8, 0.5)]
addition_times = [num_rounds // 4, num_rounds // 2, 3*num_rounds//4]
bandit_machine = DeterministicArmAcquiringMachine(initial_arms, additional_arms, addition_times)

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
initial_eps = 0.5
epsg_alg = eps_greedy(bandit_machine, initial_eps)
epsg_regret, epsg_reward, epsg_exp_reward, epsg_best_exp_reward = epsg_alg.run_experiment(num_trials, num_rounds)

# decay eps-greedy test
print("Running decay eps greedy tests")
decay_factor = 0.99
depsg_alg = decay_eps_greedy(bandit_machine, initial_eps, decay_factor)
depsg_regret, depsg_reward, depsg_exp_reward, depsg_best_exp_reward = depsg_alg.run_experiment(num_trials, num_rounds)


# etc
explore_rounds = 10
etc_alg = etc(bandit_machine, explore_rounds)
etc_regret, etc_reward, etc_exp_reward, etc_best_exp_reward = etc_alg.run_experiment(num_trials, num_rounds)

print("Running UCB Fast tests")
ucbfaster_algo = ucb_faster(bandit_machine)
ucbfaster_regret, ucbfaster_reward, ucbfaster_exp_reward, ucbfaster_best_exp_reward = ucbfaster_algo.run_experiment(num_trials, num_rounds)


# +
plt.plot(ucb_regret, 'b', label="ucb")
plt.plot(auer_regret, 'purple', label="auer")
plt.plot(ucbao_regret, 'orange', label="ucb-ao")
plt.plot(epsg_regret, 'r', label="eps-greedy")
plt.plot(depsg_regret, 'g', label="decay-eps-greedy")
plt.plot(etc_regret, 'pink', label="etc")
plt.plot(ucbfaster_regret, 'black', label="ucb-faster")

plt.xlabel('t')
plt.ylabel('Regret')
plt.title('Regret vs t')
plt.legend()
plt.grid(True)
plt.show()
# -

# plt.plot(eps_avg_total_best_exp_reward_per_round, 'b', label="eps")
# plt.plot(ucb_avg_total_best_exp_reward_per_round, 'orange', label="ucb")
# plt.xlabel('t')
# plt.ylabel('reward')
# plt.title('Best expected reward per round (sanity check)')
# plt.legend()
# plt.grid(True)
# plt.show()
