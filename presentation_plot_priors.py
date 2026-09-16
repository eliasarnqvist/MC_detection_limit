import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, poisson, binom
from scipy.optimize import brentq

np.random.seed(42)
inch_to_mm = 25.4

# Poisson rate prior likelihood posterior

n = 2
t = 10

shape = n
scale = 1/t
rate_x = np.linspace(0, 1, 100)
n_x = np.linspace(0, 10, 11)
posterior_rate = gamma.pdf(rate_x, shape, scale=scale)
prior_rate = 1/rate_x[1:]
r_guess = n/t
likelihood_rate = poisson.pmf(n_x, r_guess*t)

posterior_info_text = r"$n=$" + f"{n}, " + r"$t=$" + f"{t} s"
likelihood_info_text = r"$\tilde{r}=$" + f"{r_guess} (guess), \n" + r"$t=$" + f"{t} s"

fig, ax = plt.subplots(1, 1, figsize=(70/inch_to_mm, 50/inch_to_mm))
ax.plot(rate_x, posterior_rate, label=r"$f_R(\tilde{r} | n, t)$")
ax.set_xlabel(r"$\tilde{r}$ [1/s]")
ax.set_ylabel(r"$f_R$ [s]")
ax.legend(title=posterior_info_text, frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/rate_posterior.jpg", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(70/inch_to_mm, 50/inch_to_mm))
ax.plot(rate_x[1:], prior_rate, label=r"$f_R(\tilde{r})$")
ax.set_xlabel(r"$\tilde{r}$ [1/s]")
ax.set_ylabel(r"$f_R$ [s]")
ax.set_ylim([None, 50])
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/rate_prior.jpg", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(70/inch_to_mm, 50/inch_to_mm))
ax.plot(n_x, likelihood_rate, label=r"$p_N(n | \tilde{r}, t)$", ls="--", marker="o")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$p_N$")
ax.legend(title=likelihood_info_text, frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/rate_likelihood.jpg", dpi=300)

# Poisson rate posterior examples

ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
t = 100
rate_x = np.linspace(0, 0.2, 100)

fig, ax = plt.subplots(1, 1, figsize=(100/inch_to_mm, 80/inch_to_mm))
for n in ns:
    shape = n
    scale = 1/t
    posterior_rate = gamma.pdf(rate_x, shape, scale=scale)
    ax.plot(rate_x, posterior_rate, label=r"$n=$" + f"{n}, " + r"$t=$" + f"{t} s")
ax.set_xlabel(r"$\tilde{r}$ [1/s]")
ax.set_ylabel(r"$f_R(\tilde{r} | n, t)$ [s]")
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/rate_posterior_examples_1.jpg", dpi=300)

ns = [50, 100, 150]
t = 100
rate_x = np.linspace(0, 2, 100)

fig, ax = plt.subplots(1, 1, figsize=(100/inch_to_mm, 80/inch_to_mm))
for n in ns:
    shape = n
    scale = 1/t
    posterior_rate = gamma.pdf(rate_x, shape, scale=scale)
    ax.plot(rate_x, posterior_rate, label=r"$n=$" + f"{n}, " + r"$t=$" + f"{t} s")
ax.set_xlabel(r"$\tilde{r}$ [1/s]")
ax.set_ylabel(r"$f_R(\tilde{r} | n, t)$ [s]")
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/rate_posterior_examples_2.jpg", dpi=300)

# Binomial efficiency prior posterior likelihood

k = 2
s = 100

a = k + 1/2
b = s - k + 1/2
efficiency_x = np.linspace(0, 0.08, 100)
efficiency_x_prior = np.linspace(0.01, 0.99, 100)
k_x = np.linspace(0, 10, 11)
posterior_rate = beta.pdf(efficiency_x, a, b)
prior_rate = 1/np.sqrt((efficiency_x_prior*(1-efficiency_x_prior)))
e_guess = k/s
likelihood_rate = binom.pmf(k_x, s, e_guess)

posterior_info_text = r"$k=$" + f"{k}, " + r"$s=$" + f"{s}"
likelihood_info_text = r"$\tilde{\varepsilon}=$" + f"{e_guess} (guess), \n" + r"$s=$" + f"{s}"

fig, ax = plt.subplots(1, 1, figsize=(70/inch_to_mm, 50/inch_to_mm))
ax.plot(efficiency_x, posterior_rate, label=r"$f_E(\tilde{\varepsilon} | k, s)$")
ax.set_xlabel(r"$\tilde{\varepsilon}$")
ax.set_ylabel(r"$f_E$")
ax.legend(title=posterior_info_text, frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/efficiency_posterior.jpg", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(70/inch_to_mm, 50/inch_to_mm))
ax.plot(efficiency_x_prior, prior_rate, label=r"$f_R(\tilde{\varepsilon})$")
ax.set_xlabel(r"$\tilde{r}$")
ax.set_ylabel(r"$f_E$")
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/efficiency_prior.jpg", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(70/inch_to_mm, 50/inch_to_mm))
ax.plot(k_x, likelihood_rate, label=r"$p_K(k | \tilde{\varepsilon}, s)$", ls="--", marker="o")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$p_K$")
ax.legend(title=likelihood_info_text, frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/efficiency_likelihood.jpg", dpi=300)

# Binomial effiecni posterior examples

ks = [0, 1, 2, 5]
ss = [100, 100, 100, 100]
efficiency_x = np.linspace(0, 0.08, 100)

fig, ax = plt.subplots(1, 1, figsize=(100/inch_to_mm, 80/inch_to_mm))
for k, s in zip(ks, ss):
    a = k + 1/2
    b = s - k + 1/2
    posterior_rate = beta.pdf(efficiency_x, a, b)
    ax.plot(efficiency_x, posterior_rate, label=r"$k=$" + f"{k}, " + r"$s=$" + f"{s}")
ax.set_xlabel(r"$\tilde{\varepsilon}$")
ax.set_ylabel(r"$f_E(\tilde{\varepsilon} | k, s)$")
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/efficiency_posterior_examples_1.jpg", dpi=300)

ks = [5, 10, 20, 40]
ss = [100, 200, 400, 800]
efficiency_x = np.linspace(0, 0.14, 100)

fig, ax = plt.subplots(1, 1, figsize=(100/inch_to_mm, 80/inch_to_mm))
for k, s in zip(ks, ss):
    a = k + 1/2
    b = s - k + 1/2
    posterior_rate = beta.pdf(efficiency_x, a, b)
    ax.plot(efficiency_x, posterior_rate, label=r"$k=$" + f"{k}, " + r"$s=$" + f"{s}")
ax.set_xlabel(r"$\tilde{\varepsilon}$")
ax.set_ylabel(r"$f_E(\tilde{\varepsilon} | k, s)$")
ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/efficiency_posterior_examples_2.jpg", dpi=300)




plt.show()