bandit_util.py contains classes for bandit arms and "bandit machines".
eps_greedy.py contains an epsilon greedy function for constant epsilon.

BanditMachine is the base arm-acquiring bandit class. It holds the bandit arms, some utility functions, and has an abstract function "acquire_arms" which gets called every round of your bandit algorithm. You can define acquire_arms in whichever way you want in a derived class of BanditMachine. See StochasticGaussianArmAcquiringMachine and DeterministicArmAcquiringMachine for examples.

To write a bandit algorithm, make it inherit the base class BanditAlgorithm in util/bandit_algorithm.py
This class contains a function "run_alg" which must be redefined in your derived class, and a function "run_experiment" which runs the algorithm multiple times and returns the averaged results.

Refer to "test_deterministic_arm_acq.py" for a script that tests ucb, auer, ucb AO, eps greedy, and decaying eps greedy.

Environment setup:
```
conda create --name cmpt983 python=3.10
conda activate cmpt983
pip install -r requirements.txt
```