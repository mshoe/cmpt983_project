from util.bandit_util import StochasticGaussianArmAcquiringMachine
from algorithms.eps_greedy import const_eps_greedy
import matplotlib.pyplot as plt
import numpy as np

#NOTE: this file is out of date and will not run properly

acquire_probability = 0.002
min_mean = 0.1
max_mean = 0.9
min_var = 0.05
max_var = 0.2

bandit_machine = StochasticGaussianArmAcquiringMachine(acquire_probability, min_mean, max_mean, min_var, max_var)
bandit_machine.insert_arm(bandit_machine.generate_arm())
bandit_machine.insert_arm(bandit_machine.generate_arm())

num_rounds = 10000
eps = 0.1

num_trials = 20
avg_regret_per_round = np.zeros(shape=(num_rounds,), dtype=float)
avg_total_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
avg_total_best_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
for trial in range(num_trials):
    print("trial:", trial)
    total_reward_per_round, total_best_exp_reward_per_round = const_eps_greedy(bandit_machine, num_rounds, eps)

    regret_per_round = total_best_exp_reward_per_round - total_reward_per_round

    avg_total_reward_per_round = (trial * avg_total_reward_per_round + total_reward_per_round) / (trial + 1.0)
    avg_total_best_exp_reward_per_round = (trial * avg_total_best_exp_reward_per_round + total_best_exp_reward_per_round) / (trial + 1.0)
    avg_regret_per_round = (trial * avg_regret_per_round + regret_per_round) / (trial + 1.0)

plt.plot(avg_regret_per_round, 'r', label="avg regret")
plt.xlabel('t')
plt.ylabel('Regret')
plt.title('Regret vs t')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(avg_total_reward_per_round, 'r', label="avg total reward")
plt.plot(avg_total_best_exp_reward_per_round, 'b', label="avg best total reward")
plt.xlabel('t')
plt.ylabel('Reward')
plt.title('Reward vs t')
plt.legend()
plt.grid(True)
plt.show()