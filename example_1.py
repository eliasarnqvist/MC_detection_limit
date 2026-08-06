import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import brentq

np.random.seed(42)

inch_to_mm = 25.4



# From MC background simulation
n = 1e9
x = 20
t_s = 1e5

# From intended experiment design
t_m = 1e5

# Simple estimate of efficiency
eff_naive = x / n
rate_naive = eff_naive * n / t_s
lambda_naive = rate_naive * t_m

# Beta function for efficiency according to Bayesian inference
# Parameters
a = x + 1/2
b = n - x + 1/2
pdf_eff_x = np.linspace(0, 5e-8, 1000)
pdf_eff = beta.pdf(pdf_eff_x, a, b)

# Rate determined from efficiency
pdf_rate = pdf_eff / (n/t_s)
pdf_rate_x = pdf_eff_x * (n/t_s)

# Pseudo-experiment counts
N_ex = int(1e6)
eff_samples = np.random.beta(a, b, size=N_ex)
rate_samples = eff_samples * (n/t_s)
lambda_samples = rate_samples * t_m
x_samples = np.random.poisson(lambda_samples, size=N_ex)
x_naive_samples = np.random.poisson(lambda_naive, size=N_ex)
# Make histograms
eff_samples_histo, ex_eff_samples_histo = np.histogram(eff_samples, bins=20)
eff_samples_histo = eff_samples_histo / max(eff_samples_histo) * max(pdf_eff)
rate_samples_histo, ex_rate_samples_histo = np.histogram(rate_samples, bins=20)
rate_samples_histo = rate_samples_histo / max(rate_samples_histo) * max(pdf_rate)
x_bin_edges = np.arange(min(x_samples), max(x_samples) + 2)
x_samples_histo, ex_x_samples_histo = np.histogram(x_samples, bins=x_bin_edges)
x_samples_histo = x_samples_histo / len(x_samples)
x_naive_bin_edges = np.arange(min(x_naive_samples), max(x_naive_samples) + 2)
x_naive_samples_histo, ex_x_naive_samples_histo = np.histogram(x_naive_samples, bins=x_naive_bin_edges)
x_naive_samples_histo = x_naive_samples_histo / len(x_naive_samples)

# Plot pdf for efficiency
fig, ax = plt.subplots(1, 1, figsize=(88/inch_to_mm, 60/inch_to_mm))
ax.plot(pdf_eff_x, pdf_eff, label=r"$f_E(\tilde{\varepsilon} \mid n, x)$")
ax.axvline(x=eff_naive, color='gray', ls='--', label=r"$\tilde{\varepsilon}=x/n$")
ax.step(ex_eff_samples_histo[:-1], eff_samples_histo, where="post", label="MC PDF")
ax.set_xlabel(r"$\tilde{\varepsilon}$")
ax.set_ylabel(r"$f_E(\tilde{\varepsilon})$")
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
# plt.savefig("example_1_fig1.jpg", dpi=600)

# Plot pdf for source rate
fig, ax = plt.subplots(1, 1, figsize=(88/inch_to_mm, 60/inch_to_mm))
ax.plot(pdf_rate_x, pdf_rate, label=r"$f_R(\tilde{r})$")
ax.step(ex_rate_samples_histo[:-1], rate_samples_histo, where="post", label="MC PDF")
ax.set_xlabel(r"$\tilde{r}$")
ax.set_ylabel(r"$f_R(\tilde{r})$")
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)

# Plot pmf for pseduo-experiment counts
fig, ax = plt.subplots(1, 1, figsize=(88/inch_to_mm, 60/inch_to_mm))
ax.step(ex_x_samples_histo[:-1], x_samples_histo, where="post", label="MC PDF (with bayes)")
ax.step(ex_x_naive_samples_histo[:-1], x_naive_samples_histo, where="post", label="MC PDF (no bayes)")
ax.set_xlabel(r"$\tilde{x}$")
ax.set_ylabel(r"$f_X(\tilde{x})$")
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)

print(x_samples)

# Number of pseudo experiment coutns to use
N_x = int(1e3)
# Number of MC samples to use for determining the detection limit
N_mc = int(1e4)

# Now treat the real experimental situation
y_stars = np.array([])
y_hashs = np.array([])

def model(r_g, r_0, w):
    y = w * (r_g - r_0)
    return y

# Weighting factor
w = 4.1
u_w = 0.6
# Acceptible false positive and false negative rates
alpha = 0.05
beta = 0.05

# For determining the detection limit with a root finder
def objective_function(n_g_value):
    w_samples = np.random.normal(w, u_w, size=N_mc)
    if n_0 != 0:
        r_g_samples = np.random.gamma(n_0, scale=1/t_m, size=N_mc)
        r_0_samples = np.random.gamma(n_0, scale=1/t_m, size=N_mc)
    else:
        r_g_samples = np.random.gamma(n_0+1, scale=1/t_m, size=N_mc)
        r_0_samples = np.random.gamma(n_0+1, scale=1/t_m, size=N_mc)
    r_g_samples = np.random.gamma(n_g_value, scale=1/t_m, size=N_mc)

    y_hash_samples = model(r_g_samples, r_0_samples, w_samples)
    beta_candidate = np.mean(y_hash_samples < y_star)
    loss = beta_candidate - beta
    return loss

x_selection = np.random.choice(x_samples, size=N_x, replace=False)
for x_selected in x_selection:
    n_0 = x_selected
    print(n_0)

    w_samples = np.random.normal(w, u_w, size=N_mc)
    if n_0 != 0:
        r_g_samples = np.random.gamma(n_0, scale=1/t_m, size=N_mc)
        r_0_samples = np.random.gamma(n_0, scale=1/t_m, size=N_mc)
    else:
        r_g_samples = np.random.gamma(n_0+1, scale=1/t_m, size=N_mc)
        r_0_samples = np.random.gamma(n_0+1, scale=1/t_m, size=N_mc)
    y_samples = model(r_g_samples, r_0_samples, w_samples)
    y_star = np.quantile(y_samples, 1 - alpha)

    y_stars = np.append(y_stars, y_star)

    n_g_hash = brentq(objective_function, 1*n_0, 10*n_0)
    r_g_samples = np.random.gamma(n_g_hash, scale=1/t_m, size=N_mc)
    y_hash_samples = model(r_g_samples, r_0_samples, w_samples)
    y_hash = np.mean(y_hash_samples)

    y_hashs = np.append(y_hashs, y_hash)


print(y_stars)
print(y_hashs)

y_hash_histo, ex_y_hash = np.histogram(y_hashs, bins=30)
# y_hashs_norm = y_hash_histo / n_MC

fig, ax = plt.subplots(1, 1, figsize=(88/inch_to_mm, 60/inch_to_mm))
ax.step(ex_y_hash[:-1], y_hash_histo, where="post")
ax.set_xlabel("y# (Bq)")
ax.set_ylabel("frequency")
# ax.set_xlim([-0.05, 0.10])
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)

y_samples_histo, ex_y_samples = np.histogram(y_samples, bins=100)
y_hash_samples_histo, ex_y_hash_samples = np.histogram(y_hash_samples, bins=100)

fig, ax = plt.subplots(1, 1, figsize=(88/inch_to_mm, 60/inch_to_mm))
ax.step(ex_y_samples[:-1], y_samples_histo, where="post")
ax.axvline(x=y_star, ymax=1, color='blue', ls='--', label="y*")
ax.step(ex_y_hash_samples[:-1], y_hash_samples_histo, where="post")
ax.axvline(x=y_hash, ymax=1, color='red', ls='--', label="y#")
ax.legend(frameon=False)
ax.set_xlabel("y (Bq)")
ax.set_ylabel("pdf")
# ax.set_xlim([-0.05, 0.10])
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)

plt.show()