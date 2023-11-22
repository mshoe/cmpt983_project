from util.bandit_util import DeterministicArmAcquiringMachine, StandardBanditMachine, GaussianBanditArm, BernoulliBanditArm
from algorithms.eps_greedy import eps_greedy, decay_eps_greedy, decay_eps_greedy2
from algorithms.ucb import ucb_basic, auer, ucb_AO
import matplotlib.pyplot as plt
import numpy as np

num_rounds = 100000

initial_arms = [BernoulliBanditArm(0.5), BernoulliBanditArm(0.3)]
additional_arms = [BernoulliBanditArm(0.6), BernoulliBanditArm(0.2), BernoulliBanditArm(0.8)]
#initial_arms = [GaussianBanditArm(0.5, 0.5), GaussianBanditArm(0.3, 0.5)]
#additional_arms = [GaussianBanditArm(0.6, 0.5), GaussianBanditArm(0.2, 0.5), GaussianBanditArm(0.8, 0.5)]
addition_times = [num_rounds // 4, num_rounds // 2, 3*num_rounds//4]
bandit_machine = DeterministicArmAcquiringMachine(initial_arms, additional_arms, addition_times)

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

# plt.plot(ucb_best_exp_reward, 'b', label="ucb")
# plt.plot(auer_best_exp_reward, 'orange', label="auer")
# plt.plot(ucbao_best_exp_reward, 'orange', label="ucbao")
# plt.plot(epsg_best_exp_reward, 'orange', label="epsg")
# plt.plot(depsg_best_exp_reward, 'orange', label="depsg")
# plt.xlabel('t')
# plt.ylabel('reward')
# plt.title('Best expected reward per round (sanity check, all curves should be the same)')
# plt.legend()
# plt.grid(True)
# plt.show()