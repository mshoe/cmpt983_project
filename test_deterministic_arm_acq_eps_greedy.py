from bandit_util import DeterministicArmAcquiringMachine, StandardBanditMachine, GaussianBanditArm
from eps_greedy import const_eps_greedy
import matplotlib.pyplot as plt
import numpy as np

num_rounds = 10000
eps = 0.1

initial_arms = [GaussianBanditArm(0.5, 0.1), GaussianBanditArm(0.3, 0.1)]
additional_arms = [GaussianBanditArm(0.6, 0.1), GaussianBanditArm(0.2, 0.1), GaussianBanditArm(0.8, 0.1)]
addition_times = [num_rounds // 4, num_rounds // 2, 3*num_rounds//4]
bandit_machine = DeterministicArmAcquiringMachine(initial_arms, additional_arms, addition_times)

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

# Now test what happens when all arms are available right away
bandit_machine2 = StandardBanditMachine()
for arm in initial_arms:
    bandit_machine2.insert_arm(arm)
for arm in additional_arms:
    bandit_machine2.insert_arm(arm)
avg_regret_per_round2 = np.zeros(shape=(num_rounds,), dtype=float)
avg_total_reward_per_round2 = np.zeros(shape=(num_rounds,), dtype=float)
avg_total_best_exp_reward_per_round2 = np.zeros(shape=(num_rounds,), dtype=float)
for trial in range(num_trials):
    print("trial:", trial)
    total_reward_per_round, total_best_exp_reward_per_round = const_eps_greedy(bandit_machine2, num_rounds, eps)

    regret_per_round = total_best_exp_reward_per_round - total_reward_per_round

    avg_total_reward_per_round2 = (trial * avg_total_reward_per_round2 + total_reward_per_round) / (trial + 1.0)
    avg_total_best_exp_reward_per_round2 = (trial * avg_total_best_exp_reward_per_round2 + total_best_exp_reward_per_round) / (trial + 1.0)
    avg_regret_per_round2 = (trial * avg_regret_per_round2 + regret_per_round) / (trial + 1.0)


# Note: this is not a fair comparison because avg_total_best_exp_reward_per_round2 > avg_total_best_exp_reward_per_round,
# since the standard bandit machine has all arms at the start.
# So this comparison is kinda pointless.
plt.plot(avg_regret_per_round, 'r', label="arm acquiring")
plt.plot(avg_regret_per_round2, 'b', label="standard")
plt.xlabel('t')
plt.ylabel('Regret')
plt.title('Regret vs t')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(avg_total_reward_per_round, 'r', label="avg total reward (arm-acquiring)")
plt.plot(avg_total_best_exp_reward_per_round, 'b', label="avg best total reward (arm-acquiring)")
plt.plot(avg_total_reward_per_round2, 'g', label="avg total reward (standard)")
plt.plot(avg_total_best_exp_reward_per_round2, 'orange', label="avg best total reward (standard)")
plt.xlabel('t')
plt.ylabel('Reward')
plt.title('Reward vs t')
plt.legend()
plt.grid(True)
plt.show()