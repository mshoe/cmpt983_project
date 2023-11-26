import numpy as np
import random
from util.bandit_util import BanditMachine
from util.bandit_algorithm import BanditAlgorithm
import copy
import abc
class ucb(BanditAlgorithm):
    def __init__(self, _bandit_machine: BanditMachine):
        self._bandit_machine = copy.deepcopy(_bandit_machine)
        return
    
    @abc.abstractmethod
    def _get_std_coeff(self, t):
        return

    def run_alg(self, T: int):
        bandit_machine = copy.deepcopy(self._bandit_machine)
        K = bandit_machine.num_arms

        U = np.ones(K) * np.inf
        mu = np.zeros(K)
        N = np.zeros(K)
        total_reward = 0.0
        total_reward_per_round = np.zeros(T)

        total_exp_reward = 0.0
        total_exp_reward_per_round = np.zeros(T)

        total_best_reward = 0.0
        total_best_exp_reward_per_round = np.zeros(T)

        
        #delta = 1.0 / T**2
        #std_coeff = np.sqrt(2.0 * np.log(1.0/delta))

        best_mean, best_arm = bandit_machine.get_max_mean()

        for t in range(0, T):

            # check if arms are acquired this round
            arms_added = bandit_machine.acquire_arms()
            if arms_added:
                old_K = K
                K = bandit_machine.num_arms
                U.resize(K)
                for i in range(old_K, K):
                    U[i] = np.inf
                mu.resize(K)
                N.resize(K)
                best_mean, best_arm = bandit_machine.get_max_mean()

            a = np.argmax(U) # in case of ties, argmax returns the index of first occurance
            reward = bandit_machine.arms[a].pull()
            mu[a] = (mu[a] * N[a] + reward) / (N[a] + 1.0)
            N[a] += 1.0
            std_coeff = self._get_std_coeff(t)
            U[a] = mu[a] + (std_coeff / (np.sqrt(N[a])))

            
            total_reward += reward
            total_reward_per_round[t] = total_reward

            total_exp_reward += bandit_machine.arms[a].expected_reward()
            total_exp_reward_per_round[t] = total_exp_reward

            total_best_reward += best_mean
            total_best_exp_reward_per_round[t] = total_best_reward



        return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round
    
class ucb_basic(ucb):
    def __init__(self, _bandit_machine: BanditMachine, T: int):
        super().__init__(_bandit_machine)
        delta = 1.0 / T**2
        self._std_coeff = np.sqrt(2.0 * np.log(1.0/delta))
        return
    
    def _get_std_coeff(self, t):
        return self._std_coeff
    
class auer(ucb):
    def __init__(self, _bandit_machine: BanditMachine):
        super().__init__(_bandit_machine)
        return
    
    def _get_std_coeff(self, t):
        # + 1.0 since t starts at 0
        t = t + 1.0
        return np.sqrt(8.0 * np.log(t))
    
class ucb_AO(ucb):
    def __init__(self, _bandit_machine: BanditMachine):
        super().__init__(_bandit_machine)
        return
    
    def _get_std_coeff(self, t):
        # + 1.0 since t starts at 0
        t = t + 1.0
        f = 1.0 + t * (np.log(t))**2
        return np.sqrt(2.0 * np.log(f))
    
class ucb_fast(ucb):
    def __init__(self, _bandit_machine: BanditMachine, power: float):
        super().__init__(_bandit_machine)
        self._power = power
        return
    
    def _get_std_coeff(self, t):
        # + 1.0 since t starts at 0
        t = t + 1.0
        f = 1.0 + t * (np.log(t))**2
        return np.sqrt(2.0 * np.log(f))
    
    def run_alg(self, T: int):
        bandit_machine = copy.deepcopy(self._bandit_machine)
        K = bandit_machine.num_arms

        U = np.ones(K) * np.inf
        mu = np.zeros(K)
        N = np.zeros(K)
        total_reward = 0.0
        total_reward_per_round = np.zeros(T)

        total_exp_reward = 0.0
        total_exp_reward_per_round = np.zeros(T)

        total_best_reward = 0.0
        total_best_exp_reward_per_round = np.zeros(T)

        
        #delta = 1.0 / T**2
        #std_coeff = np.sqrt(2.0 * np.log(1.0/delta))

        best_mean, best_arm = bandit_machine.get_max_mean()

        for t in range(0, T):

            # check if arms are acquired this round
            arms_added = bandit_machine.acquire_arms()
            if arms_added:
                old_K = K
                K = bandit_machine.num_arms
                U.resize(K)
                for i in range(old_K, K):
                    U[i] = np.inf
                mu.resize(K)
                N.resize(K)
                best_mean, best_arm = bandit_machine.get_max_mean()

            a = np.argmax(U) # in case of ties, argmax returns the index of first occurance
            reward = bandit_machine.arms[a].pull()
            mu[a] = (mu[a] * N[a] + reward) / (N[a] + 1.0)
            N[a] += 1.0
            std_coeff = self._get_std_coeff(t)
            U[a] = mu[a] + (std_coeff / (np.sqrt(N[a]))**self._power)

            
            total_reward += reward
            total_reward_per_round[t] = total_reward

            total_exp_reward += bandit_machine.arms[a].expected_reward()
            total_exp_reward_per_round[t] = total_exp_reward

            total_best_reward += best_mean
            total_best_exp_reward_per_round[t] = total_best_reward



        return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round
    
     

class ucb_faster(ucb):
    def __init__(self, _bandit_machine: BanditMachine):
        super().__init__(_bandit_machine)
        self._power = 1
        return
    
    def _get_std_coeff(self, t):
        # + 1.0 since t starts at 0
        t = t + 1.0
        f = 1.0 + t * (np.log(t))**2
        return np.sqrt(2.0 * np.log(f))
    
    def run_alg(self, T: int):
        bandit_machine = copy.deepcopy(self._bandit_machine)
        K = bandit_machine.num_arms

        U = np.ones(K) * np.inf
        mu = np.zeros(K)
        N = np.zeros(K)
        total_reward = 0.0
        total_reward_per_round = np.zeros(T)

        total_exp_reward = 0.0
        total_exp_reward_per_round = np.zeros(T)

        total_best_reward = 0.0
        total_best_exp_reward_per_round = np.zeros(T)

        
        #delta = 1.0 / T**2
        #std_coeff = np.sqrt(2.0 * np.log(1.0/delta))

        best_mean, best_arm = bandit_machine.get_max_mean()

        self._power = 1.0

        for t in range(0, T):

            # check if arms are acquired this round
            arms_added = bandit_machine.acquire_arms()
            if arms_added:
                old_K = K
                K = bandit_machine.num_arms
                U.resize(K)
                for i in range(old_K, K):
                    U[i] = np.inf
                mu.resize(K)
                N.resize(K)
                self._power += 1.0
                best_mean, best_arm = bandit_machine.get_max_mean()

            a = np.argmax(U) # in case of ties, argmax returns the index of first occurance
            reward = bandit_machine.arms[a].pull()
            mu[a] = (mu[a] * N[a] + reward) / (N[a] + 1.0)
            N[a] += 1.0
            std_coeff = self._get_std_coeff(t)
            U[a] = mu[a] + (std_coeff / (np.sqrt(N[a]))**self._power)

            
            total_reward += reward
            total_reward_per_round[t] = total_reward

            total_exp_reward += bandit_machine.arms[a].expected_reward()
            total_exp_reward_per_round[t] = total_exp_reward

            total_best_reward += best_mean
            total_best_exp_reward_per_round[t] = total_best_reward



        return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round
    

class linucb(BanditAlgorithm):
    def __init__(self, _bandit_machine: BanditMachine):
        self._bandit_machine = copy.deepcopy(_bandit_machine)
        self._lambda = 1e-4
        return
    
    @abc.abstractmethod
    def _get_std_coeff(self, t):
        return

    def sherman_morrison_update(A_inv, x):
        """
        Perform the Sherman-Morrison update to the inverse of a matrix A_inv
        when a rank-one perturbation u * v^T is added to A_inv.

        Args:
        A_inv (numpy.ndarray): The original inverse matrix.
        u (numpy.ndarray): The vector u in the rank-one perturbation u * v^T.
        v (numpy.ndarray): The vector v in the rank-one perturbation u * v^T.

        Returns:
        numpy.ndarray: The updated inverse matrix after the perturbation.
        """

        # Calculate the numerator and denominator of the Sherman-Morrison formula
        numerator = np.outer(A_inv @ x, x @ A_inv)
        denominator = 1 + x @ A_inv @ x

        # Perform the update
        updated_A_inv = A_inv - numerator / denominator

        return updated_A_inv

    def run_alg(self, T: int):
        bandit_machine = copy.deepcopy(self._bandit_machine)
        K = bandit_machine.num_arms

        U = np.ones(K) * np.inf
        mu = np.zeros(K)
        N = np.zeros(K)

        dim = bandit_machine.arms[0].feature_vector().shape[0]

        # V_inv = np.identity(dim) * 1.0/self._lambda
        # b = np.zeros_like(bandit_machine.arms[0].feature_vector())
        
        total_reward = 0.0
        total_reward_per_round = np.zeros(T)

        total_exp_reward = 0.0
        total_exp_reward_per_round = np.zeros(T)

        total_best_reward = 0.0
        total_best_exp_reward_per_round = np.zeros(T)

        
        #delta = 1.0 / T**2
        #std_coeff = np.sqrt(2.0 * np.log(1.0/delta))

        best_mean, best_arm = bandit_machine.get_max_mean()

        for t in range(0, T):

            # check if arms are acquired this round
            arms_added = bandit_machine.acquire_arms()
            if arms_added:
                old_K = K
                K = bandit_machine.num_arms
                U.resize(K)
                for i in range(old_K, K):
                    U[i] = np.inf
                mu.resize(K)
                N.resize(K)
                best_mean, best_arm = bandit_machine.get_max_mean()

            a = np.argmax(U) # in case of ties, argmax returns the index of first occurance
            reward = bandit_machine.arms[a].pull()

            x = bandit_machine.arms[a].feature_vector()
            V_inv = linucb.sherman_morrison_update(V_inv, x)
            b += reward * x

            theta_hat = V_inv @ b

            beta_sqrt = np.sqrt(dim * np.log((self._lambda * dim + t)/(self._lambda * dim)) + 2*np.log(T)) + np.sqrt(self._lambda)

            mu[a] = (mu[a] * N[a] + reward) / (N[a] + 1.0)
            N[a] += 1.0
            std_coeff = self._get_std_coeff(t)

            U[a] = np.dot(x,theta_hat) + beta_sqrt * np.sqrt(x.T @ V_inv @ x)

            total_reward += reward
            total_reward_per_round[t] = total_reward

            total_exp_reward += bandit_machine.arms[a].expected_reward()
            total_exp_reward_per_round[t] = total_exp_reward

            total_best_reward += best_mean
            total_best_exp_reward_per_round[t] = total_best_reward



        return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round
    


# def ucb(_bandit_machine: BanditMachine, T):
    
#     bandit_machine = copy.deepcopy(_bandit_machine)
#     K = bandit_machine.num_arms

#     U = np.ones(K) * np.inf
#     mu = np.zeros(K)
#     N = np.zeros(K)
#     total_reward = 0.0
#     total_reward_per_round = np.zeros(T)

#     total_exp_reward = 0.0
#     total_exp_reward_per_round = np.zeros(T)

#     total_best_reward = 0.0
#     total_best_exp_reward_per_round = np.zeros(T)

    
#     delta = 1.0 / T**2
#     std_coeff = np.sqrt(2.0 * np.log(1.0/delta))

#     best_mean, best_arm = bandit_machine.get_max_mean()

#     for t in range(0, T):

#         # check if arms are acquired this round
#         arms_added = bandit_machine.acquire_arms()
#         if arms_added:
#             old_K = K
#             K = bandit_machine.num_arms
#             U.resize(K)
#             for i in range(old_K, K):
#                 U[i] = np.inf
#             mu.resize(K)
#             N.resize(K)
#             best_mean, best_arm = bandit_machine.get_max_mean()

#         a = np.argmax(U) # in case of ties, argmax returns the index of first occurance
#         reward = bandit_machine.arms[a].pull()
#         mu[a] = (mu[a] * N[a] + reward) / (N[a] + 1.0)
#         N[a] += 1.0
#         U[a] = mu[a] + std_coeff / np.sqrt(N[a])

        
#         total_reward += reward
#         total_reward_per_round[t] = total_reward

#         total_exp_reward += bandit_machine.arms[a].expected_reward()
#         total_exp_reward_per_round[t] = total_exp_reward

#         total_best_reward += best_mean
#         total_best_exp_reward_per_round[t] = total_best_reward



#     return total_reward_per_round, total_exp_reward_per_round, total_best_exp_reward_per_round