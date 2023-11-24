import numpy as np
import abc
import random

class BanditArm:
    def __init__(self):
        return
    
    @abc.abstractmethod
    def expected_reward(self):
        return

    @abc.abstractmethod
    def pull(self):
        return

class BernoulliBanditArm(BanditArm):
    def __init__(self, mean):
        self._mean = mean

    def expected_reward(self):
        return self._mean

    def pull(self):
        return np.random.binomial(1, self._mean, None)
    
class GaussianBanditArm(BanditArm):
    def __init__(self, mean, variance):
        self._mean = mean
        self._variance = variance

    def expected_reward(self):
        return self._mean
    
    def variance(self):
        return self._variance

    def pull(self):
        return np.random.normal(self._mean, self._variance, None)
    
class BanditMachine:
    def __init__(self):
        self.arms = []
        self.num_arms = 0 # update this when arms is updated
        return
    
    def insert_arm(self, arm: BanditArm):
        self.arms.append(arm)
        self.num_arms += 1
        return

    def get_max_mean(self):
        best_mean = 0
        best_arm = 0
        for i in range(self.num_arms):
            arm_mean = self.arms[i].expected_reward()
            if arm_mean > best_mean:
                best_mean = arm_mean
                best_arm = i

        return best_mean, best_arm
    
    @abc.abstractmethod
    def acquire_arms(self):
        # call after every round of a bandit algorithm
        # return True if any arm has been acquired
        return False

class StandardBanditMachine(BanditMachine):
    def acquire_arms(self):
        return False
    
class StochasticArmAcquiringMachine(BanditMachine):
    def __init__(self, acquire_probability):
        super().__init__()
        self._acquire_probability = acquire_probability
        return
    
    @abc.abstractmethod
    def generate_arm(self):
        return
    
    def acquire_arms(self):
        val = random.random()
        if val < self._acquire_probability:
            self.insert_arm(self.generate_arm())
            return True
        else:
            return False
    
class StochasticGaussianArmAcquiringMachine(StochasticArmAcquiringMachine):
    def __init__(self, acquire_probability, min_mean, max_mean, min_var, max_var):
        super().__init__(acquire_probability)
        self._min_mean = min_mean
        self._max_mean = max_mean
        self._min_var = min_var
        self._max_var = max_var
        return
    
    def generate_arm(self):
        new_arm_mean = random.random() * (self._max_mean - self._min_mean) + self._min_mean
        new_arm_var = random.random() * (self._max_var - self._min_var) + self._min_var
        return GaussianBanditArm(new_arm_mean, new_arm_var)
    
class StochasticBernoulliArmAcquiringMachine(StochasticArmAcquiringMachine):
    def __init__(self, acquire_probability, min_mean, max_mean):
        super().__init__(acquire_probability)
        self._min_mean = min_mean
        self._max_mean = max_mean
        return
    
    def generate_arm(self):
        new_arm_mean = random.random() * (self._max_mean - self._min_mean) + self._min_mean
        return BernoulliBanditArm(new_arm_mean)
    
        
class DeterministicArmAcquiringMachine(BanditMachine):
    def __init__(self, initial_arms: list[BanditArm], additional_arms: list[BanditArm], 
                 addition_times: list[int]):
        
        super().__init__()
        for arm in initial_arms:
            self.insert_arm(arm)

        # addition times is a list of times in increasing order that arms get added.
        # These two lists must be the same length:
        self._additional_arms = additional_arms
        self._addition_times = addition_times
        
        self._t = 0
        self._num_additional_arms = len(self._additional_arms)
        self._curr_arm_index = 0
        return
    
    def acquire_arms(self):
        
        arm_acquired = False
        if self._curr_arm_index < self._num_additional_arms:
            if self._t >= self._addition_times[self._curr_arm_index]:
                self.insert_arm(self._additional_arms[self._curr_arm_index])
                self._curr_arm_index += 1
                arm_acquired = True
        
        self._t += 1
        return arm_acquired