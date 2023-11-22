from util.bandit_util import BanditArm, BanditMachine
import abc
import numpy as np

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