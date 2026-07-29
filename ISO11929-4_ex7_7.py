import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

np.random.seed(41)

# Number of MC trials
n_M = int(1e6)
# For the characteristic limits
alpha = 0.05
beta = 0.05

# Model
def model(r_g, r_0, w):
    y = w * (r_g - r_0)
    return y

# Input quantities and data
n_g = 4
t_g = 1200
n_0 = 1
t_0 = 1200
w = 4.1
u_w = 0.6
# Intermediate values

# Determining the detection threshold (n_g = n_0)
w_samples = np.random.normal(w, u_w, size=n_M)
r_g_samples = np.random.gamma(n_0, scale=1/t_g, size=n_M)
r_0_samples = np.random.gamma(n_0, scale=1/t_0, size=n_M)
y_samples = model(r_g_samples, r_0_samples, w_samples)
y_star = np.quantile(y_samples, 1 - alpha)
print(y_star)

# Determining the detection limit with a root finder
def objective_function(n_g_value):
    r_g_samples = np.random.gamma(n_g_value, scale=1/t_g, size=n_M)
    y_hash_samples = model(r_g_samples, r_0_samples, w_samples)
    beta_candidate = np.mean(y_hash_samples < y_star)
    loss = beta_candidate - beta
    return loss

n_g_hash = brentq(objective_function, 1*n_0, 10*n_0)
r_g_samples = np.random.gamma(n_g_hash, scale=1/t_g, size=n_M)
y_hash_samples = model(r_g_samples, r_0_samples, w_samples)
y_hash = np.mean(y_hash_samples)
print(y_hash)

# Make histograms
y_0_histo, ex_0 = np.histogram(y_samples, bins=500)
y_0_histo_norm = y_0_histo / n_M

y_hash_histo, ex_hash = np.histogram(y_hash_samples, bins=500)
y_hash_histo_norm = y_hash_histo / n_M

# Plot
inch_to_mm = 25.4
fig, ax = plt.subplots(1, 1, figsize=(88/inch_to_mm, 60/inch_to_mm))
ax.step(ex_0[:-1], y_0_histo_norm, where="post")
ax.axvline(x=y_star, ymax=0.9, color='blue', ls='--')
ax.step(ex_hash[:-1], y_hash_histo_norm, where="post")
ax.axvline(x=y_hash, ymax=0.9, color='red', ls='--')
ax.set_xlabel("y (Bq)", fontsize=8)
ax.set_ylabel("pdf", fontsize=8)
ax.set_xlim([-0.05, 0.10])
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)

plt.show()
