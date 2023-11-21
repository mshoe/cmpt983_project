from bandit_util import GaussianBanditArm
import numpy as np
import matplotlib.pyplot as plt
arm = GaussianBanditArm(0, 1)

N = 1000
num_trials = 10000

bins = N / 10
mu_hats = np.zeros(shape=(num_trials,), dtype=float)
for trial in range(num_trials):
    mu_hat = 0
    for i in range(N):
        r = arm.pull()
        mu_hat += r

    mu_hat = mu_hat / N
    mu_hats[trial] = mu_hat

# Compute histogram
hist, bin_edges = np.histogram(mu_hats, bins=10)

# Plot the histogram
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()