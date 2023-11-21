from bandit_util import DeterministicArmAcquiringMachine, StandardBanditMachine, GaussianBanditArm, BernoulliBanditArm
from eps_greedy import const_eps_greedy, decay_eps_greedy
from ucb import ucb
from auer import auer
import matplotlib.pyplot as plt
import numpy as np

num_rounds = 100000
eps = 0.1

initial_arms = [BernoulliBanditArm(0.5), BernoulliBanditArm(0.3)]
additional_arms = [BernoulliBanditArm(0.6), BernoulliBanditArm(0.2), BernoulliBanditArm(0.8)]
#initial_arms = [GaussianBanditArm(0.5, 0.5), GaussianBanditArm(0.3, 0.5)]
#additional_arms = [GaussianBanditArm(0.6, 0.5), GaussianBanditArm(0.2, 0.5), GaussianBanditArm(0.8, 0.5)]
addition_times = [num_rounds // 4, num_rounds // 2, 3*num_rounds//4]
bandit_machine = DeterministicArmAcquiringMachine(initial_arms, additional_arms, addition_times)

num_trials = 20
eps_avg_regret_per_round = np.zeros(shape=(num_rounds,), dtype=float)
eps_avg_total_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
eps_avg_total_best_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
for trial in range(num_trials):
    print("trial:", trial)
    total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round = const_eps_greedy(bandit_machine, num_rounds, eps)

    regret_per_round = total_best_exp_reward_per_round - total_exp_reward_per_round

    #eps_avg_total_reward_per_round = (trial * eps_avg_total_reward_per_round + total_reward_per_round) / (trial + 1.0)
    eps_avg_total_best_exp_reward_per_round = (trial * eps_avg_total_best_exp_reward_per_round + total_best_exp_reward_per_round) / (trial + 1.0)
    eps_avg_regret_per_round = (trial * eps_avg_regret_per_round + regret_per_round) / (trial + 1.0)

# UCB test
ucb_avg_regret_per_round = np.zeros(shape=(num_rounds,), dtype=float)
ucb_avg_total_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
ucb_avg_total_best_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
for trial in range(num_trials):
    print("trial:", trial)
    total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round = ucb(bandit_machine, num_rounds)

    regret_per_round = total_best_exp_reward_per_round - total_exp_reward_per_round

    #ucb_avg_total_reward_per_round = (trial * ucb_avg_total_reward_per_round + total_reward_per_round) / (trial + 1.0)
    ucb_avg_total_best_exp_reward_per_round = (trial * ucb_avg_total_best_exp_reward_per_round + total_best_exp_reward_per_round) / (trial + 1.0)
    ucb_avg_regret_per_round = (trial * ucb_avg_regret_per_round + regret_per_round) / (trial + 1.0)


# decaying eps-greedy test
eps2_avg_regret_per_round = np.zeros(shape=(num_rounds,), dtype=float)
eps2_avg_total_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
eps2_avg_total_best_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
for trial in range(num_trials):
    print("trial:", trial)
    total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round = decay_eps_greedy(bandit_machine, num_rounds, 0.1)

    regret_per_round = total_best_exp_reward_per_round - total_exp_reward_per_round

    #ucb_avg_total_reward_per_round = (trial * ucb_avg_total_reward_per_round + total_reward_per_round) / (trial + 1.0)
    eps2_avg_total_best_exp_reward_per_round = (trial * eps2_avg_total_best_exp_reward_per_round + total_best_exp_reward_per_round) / (trial + 1.0)
    eps2_avg_regret_per_round = (trial * eps2_avg_regret_per_round + regret_per_round) / (trial + 1.0)


# AUER test
auer_avg_regret_per_round = np.zeros(shape=(num_rounds,), dtype=float)
auer_avg_total_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
auer_avg_total_best_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
for trial in range(num_trials):
    print("trial:", trial)
    total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round = auer(bandit_machine, num_rounds)

    regret_per_round = total_best_exp_reward_per_round - total_exp_reward_per_round

    #ucb_avg_total_reward_per_round = (trial * ucb_avg_total_reward_per_round + total_reward_per_round) / (trial + 1.0)
    auer_avg_total_best_exp_reward_per_round = (trial * ucb_avg_total_best_exp_reward_per_round + total_best_exp_reward_per_round) / (trial + 1.0)
    auer_avg_regret_per_round = (trial * auer_avg_regret_per_round + regret_per_round) / (trial + 1.0)


# Note: this is not a fair comparison because avg_total_best_exp_reward_per_round2 > avg_total_best_exp_reward_per_round,
# since the standard bandit machine has all arms at the start.
# So this comparison is kinda pointless.
plt.plot(eps_avg_regret_per_round, 'r', label="eps-greedy")
plt.plot(ucb_avg_regret_per_round, 'b', label="ucb")
plt.plot(eps2_avg_regret_per_round, 'g', label="decay-eps-greedy")
plt.plot(auer_avg_regret_per_round, 'purple', label="auer")
plt.xlabel('t')
plt.ylabel('Regret')
plt.title('Regret vs t')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(ucb_avg_total_best_exp_reward_per_round, 'b', label="eps")
plt.plot(ucb_avg_total_best_exp_reward_per_round, 'orange', label="ucb")
plt.xlabel('t')
plt.ylabel('reward')
plt.title('Best expected reward per round (sanity check)')
plt.legend()
plt.grid(True)
plt.show()