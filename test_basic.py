from util.bandit_util import StandardBanditMachine, GaussianBanditArm
from algorithms.eps_greedy import const_eps_greedy
import matplotlib.pyplot as plt

#NOTE: this file is out of date and will not run properly

bandit_machine = StandardBanditMachine()
bandit_machine.insert_arm(GaussianBanditArm(0.5, 0.2))
bandit_machine.insert_arm(GaussianBanditArm(0.3, 0.2))

best_arm_mean = 0.5
num_rounds = 10000
eps = 0.1
total_reward_per_round, total_best_exp_reward_per_round = const_eps_greedy(bandit_machine, num_rounds, eps)

regret_per_round = total_best_exp_reward_per_round - total_reward_per_round

plt.plot(regret_per_round, 'r')
plt.xlabel('t')
plt.ylabel('Regret')
plt.title('Regret vs t')
plt.legend()
plt.grid(True)
plt.show()