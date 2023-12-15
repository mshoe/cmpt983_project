from util.bandit_util import BanditArm, BanditMachine
import abc
import numpy as np
from tqdm import tqdm

class BanditAlgorithm:

    def __init__(self):
        return
    
    @abc.abstractmethod
    def run_alg(self, T: int):
        return
    
    def run_experiment(self, num_trials: int, num_rounds: int, verbose=True):
        avg_regret_per_round = np.zeros(shape=(num_rounds,), dtype=float)
        avg_total_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
        avg_total_best_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
        avg_total_exp_reward_per_round = np.zeros(shape=(num_rounds,), dtype=float)
        for trial in range(num_trials):
            if verbose:
                print("trial:", trial)
            total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round = self.run_alg(num_rounds)

            regret_per_round = total_best_exp_reward_per_round - total_exp_reward_per_round

            avg_total_reward_per_round = (trial * avg_total_reward_per_round + total_reward_per_round) / (trial + 1.0)
            avg_total_exp_reward_per_round = (trial * avg_total_exp_reward_per_round + total_exp_reward_per_round) / (trial + 1.0)
            avg_total_best_exp_reward_per_round = (trial * avg_total_best_exp_reward_per_round + total_best_exp_reward_per_round) / (trial + 1.0)
            avg_regret_per_round = (trial * avg_regret_per_round + regret_per_round) / (trial + 1.0)

        return avg_regret_per_round, avg_total_reward_per_round, avg_total_exp_reward_per_round, avg_total_best_exp_reward_per_round
    

def update_average(old_value, new_value, trial):
    """Returns [old_value], the mean of something after [trial] trials, updated to
    include data from one more trial given with [new_value].
    """
    return (old_value * trial + new_value) / (trial + 1)

class MVBanditAlgorithm:
    def __init__(self): return
    
    @abc.abstractmethod
    def run_alg(self, T: int): return
    
    def run_experiment(self, num_trials: int, num_rounds: int, verbose=True, trial_to_bandit_machine=None):
        for trial in tqdm(range(num_trials)):
            if not trial_to_bandit_machine is None:
                bandit_machine = trial_to_bandit_machine(trial)
            trial_results = self.run_alg(num_rounds, bandit_machine)
            avg_results = trial_results if trial == 0 else [update_average(a, r, trial) for a,r in zip(avg_results, trial_results)]
        return avg_results