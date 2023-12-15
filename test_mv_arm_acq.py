import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random
from algorithms.ucb_mv import ucb_mv_basic as ucb_mv
from algorithms.eps_greedy_mv import decay_eps_greedy as decay_eps_greedy_mv
from util.bandit_util import StochasticGaussianArmAcquiringMachine, GaussianBanditArm
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from itertools import product

class SeedManager:
    def __init__(self, seed=0): self.seed = seed
    def __enter__(self):
        self.old_seed = SeedManager.get_seed_dict()
        SeedManager.set_seed(self.seed)
    def __exit__(self, type, value, traceback):
        SeedManager.set_seed(self.old_seed)
    @staticmethod
    def set_seed(seed):
        if isinstance(seed, int):
            random.seed(seed)
            np.random.seed(seed)
        elif isinstance(seed, dict):
            random.setstate(seed["random_seed"])
            np.random.set_state(seed["numpy_seed"])
    @staticmethod
    def get_seed_dict():
        return {"random_seed": random.getstate(), "numpy_seed": np.random.get_state()}

def get_new_bandit_machine(args, seed=0):
    """Returns a new bandit machine given [args] with [seed] used to get the arms."""
    with SeedManager(seed=seed):
        bandit_machine = StochasticGaussianArmAcquiringMachine(args.p_acc, args.min_mu, args.max_mu, args.min_var, args.max_var)
        for _ in range(args.num_arms):
            bandit_machine.insert_arm(GaussianBanditArm(np.random.uniform(args.min_mu, args.max_mu), np.random.uniform(args.min_var, args.max_var)))
        return bandit_machine

P = argparse.ArgumentParser()
P.add_argument("--num_arms", type=int, default=2)
P.add_argument("--num_rounds", type=int, default=10000)
P.add_argument("--num_trials", type=int, default=20)
P.add_argument("--rhos", type=float, default=[0, 0.5, 2], nargs="+")
P.add_argument("--cutoffs", type=float, default=[0.3, .5, .7], nargs="+")
P.add_argument("--min_mu", type=float, default=0.1)
P.add_argument("--max_mu", type=float, default=0.9)
P.add_argument("--min_var", type=float, default=0.05)
P.add_argument("--max_var", type=float, default=0.2)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--p_acc", type=float, default=0.02)
P.add_argument("--algs", choices=["ucb", "epsilon-greedy"], default=["ucb"], nargs="+")
args = P.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

colors = mcolors.TABLEAU_COLORS

bandit_machine = get_new_bandit_machine(args)

rho2alg2results = defaultdict(lambda: {})
for rho in tqdm(args.rhos):
    for alg in tqdm(args.algs):

        if alg == "ucb":
            algorithm = ucb_mv(bandit_machine, args.num_rounds,
                rho=rho,
                cutoffs=args.cutoffs)
            result = algorithm.run_experiment(args.num_trials, args.num_rounds,
                trial_to_bandit_machine=partial(get_new_bandit_machine, args))
            rho2alg2results[rho]["ucb"] = list(result)
        elif alg == "epsilon-greedy":
            algorithm = decay_eps_greedy_mv(bandit_machine, args.num_rounds,
                rho=rho,
                cutoffs=args.cutoffs)
            result = algorithm.run_experiment(args.num_trials, args.num_rounds,
                trial_to_bandit_machine=partial(get_new_bandit_machine, args))
            rho2alg2results[rho]["epsilon-greedy"] = list(result)

# Plot normal regret
for c,(rho,alg) in zip(colors, product(args.rhos, args.algs)):
    _, _, _, count_below_cutoff, _, _, regret, regret_mv, total_below_best_mean_per_round = rho2alg2results[rho][alg]
    plt.plot(regret, color=c, label=f"{alg} - rho={rho}")

plt.xlabel("Rounds")
plt.ylabel("Regret - unadjusted for variance")
plt.title('Unadjusted Regret vs Round')
plt.legend()
plt.grid(True)
plt.savefig("fig-regret.png")
plt.close()

# Plot MV regret
for c,(rho,alg) in zip(colors, product(args.rhos, args.algs)):
    _, _, _, count_below_cutoff, _, _, regret, regret_mv, total_below_best_mean_per_round = rho2alg2results[rho][alg]
    plt.plot(regret_mv, color=c, label=f"{alg} - rho={rho}")

plt.xlabel("Rounds")
plt.ylabel("Mean-Variance Regret")
plt.title('Mean-Variance Regret vs Round')
plt.legend()
plt.grid(True)
plt.savefig("fig-mv_regret.png")
plt.close()

for c,(rho,alg) in zip(colors, product(args.rhos, args.algs)):
    _, _, _, count_below_cutoff, _, _, regret, regret_mv, total_below_best_mean_per_round = rho2alg2results[rho][alg]
    print(count_below_cutoff)
    plt.plot(total_below_best_mean_per_round, color=c, label=f"{alg} - rho={rho}")

plt.xlabel("Rounds")
plt.ylabel("Cumulative Reward below Best Arm Mean")
plt.title('Cumulative Reward below Best Arm Mean vs Round')
plt.legend()
plt.grid(True)
plt.savefig("fig-mv_reward_below_mean.png")
plt.close()