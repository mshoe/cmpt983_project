bandit_util.py contains classes for bandit arms and "bandit machines".
eps_greedy.py contains an epsilon greedy function for constant epsilon.

BanditMachine is the base arm-acquiring bandit class. It holds the bandit arms, some utility functions, and has an abstract function "acquire_arms" which gets called every round of your bandit algorithm. You can define acquire_arms in whichever way you want in a derived class of BanditMachine. See StochasticGaussianArmAcquiringMachine and DeterministicArmAcquiringMachine for examples.

The "test_....py" scripts test and plot regret from using epsilon greedy for different types of arm-acquiring bandit machines.

Environment setup:
```
conda create --name cmpt983 python=3.10
conda activate cmpt983
pip install -r requirements.txt
```