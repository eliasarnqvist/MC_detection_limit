import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm
from scipy.optimize import brentq

np.random.seed(42)
inch_to_mm = 25.4

# DID NOT HAVE TIME TO EDIT THIS


# Number of primary MC trials
m = int(1e6)
# For the characteristic limits
alpha = 0.05
beta = 0.05

# Input quantities
n_G = 10
n_0 = 4
t_G = 1e3
t_0 = 1e3
k = 1e5
s = 1e7
i = 0.3
u_i = 0.03

# Intermediate quantities
e_samples = np.random.beta(k+1/2, s-k+1/2, size=m)
r_G_samples = np.random.gamma(n_G, 1/t_G, size=m)
r_0_samples = np.random.gamma(n_0, 1/t_0, size=m)
i_samples = np.random.normal(i, u_i, size=m)

# Plot intermediate distributions
e_histo, ex_e = np.histogram(e_samples, bins=50)
e_histo_norm = e_histo / (m * (ex_e[1]-ex_e[0]))
fig, ax = plt.subplots(1, 1, figsize=(80/inch_to_mm, 50/inch_to_mm))
ax.step(ex_e[:-1], e_histo_norm, where="post", label="MC samples")
# xmin, xmax = ax.get_xlim()
# xx = np.linspace (xmin, xmax, 100)
# ax.plot(xx, beta.pdf(xx, k+1/2, s-k+1/2), ls="--", label="Beta distr.")
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel("PDF")
# ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_efficiency_pdf.jpg", dpi=300)

r_G_histo, ex_r_G = np.histogram(r_G_samples, bins=50)
r_G_histo_norm = r_G_histo / (m * (ex_r_G[1]-ex_r_G[0]))
fig, ax = plt.subplots(1, 1, figsize=(80/inch_to_mm, 50/inch_to_mm))
ax.step(ex_r_G[:-1], r_G_histo_norm, where="post", label="MC samples")
# xmin, xmax = ax.get_xlim()
# xx = np.linspace (xmin, xmax, 100)
# ax.plot(xx, gamma.pdf(xx, n_G, 1/t_G), ls="--", label="Gamma distr.")
ax.set_xlabel(r"$r_G$")
ax.set_ylabel("PDF")
# ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_gross_rate_pdf.jpg", dpi=300)

r_0_histo, ex_r_0 = np.histogram(r_0_samples, bins=50)
r_0_histo_norm = r_0_histo / (m * (ex_r_0[1]-ex_r_0[0]))
fig, ax = plt.subplots(1, 1, figsize=(80/inch_to_mm, 50/inch_to_mm))
ax.step(ex_r_0[:-1], r_0_histo_norm, where="post", label="MC samples")
ax.set_xlabel(r"$r_0$")
ax.set_ylabel("PDF")
# ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_background_rate_pdf.jpg", dpi=300)

i_histo, ex_i = np.histogram(i_samples, bins=50)
i_histo_norm = i_histo / (m * (ex_i[1]-ex_i[0]))
fig, ax = plt.subplots(1, 1, figsize=(80/inch_to_mm, 50/inch_to_mm))
ax.step(ex_i[:-1], i_histo_norm, where="post", label="MC samples")
ax.set_xlabel(r"$i$")
ax.set_ylabel("PDF")
# ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_intensity_pdf.jpg", dpi=300)

# Apply model equation
a_samples = 1/(i_samples*e_samples) * (r_G_samples - r_0_samples)

a_pe = np.mean(a_samples)
u_a_pe = np.std(a_samples)

a_histo, ex_a = np.histogram(a_samples, bins=50)
a_histo_norm = a_histo / (m * (ex_a[1]-ex_a[0]))
fig, ax = plt.subplots(1, 1, figsize=(120/inch_to_mm, 80/inch_to_mm))
ax.step(ex_a[:-1], a_histo_norm, where="post", label="MC samples")
ax.axvline(x=a_pe, ymax=1, color='green', ls='-', label=r"$a_{pe}=$"+f"{a_pe:.2f} ({u_a_pe:.2f})")
ax.axvline(x=a_pe-u_a_pe, ymax=1, color='green', ls='--')
ax.axvline(x=a_pe+u_a_pe, ymax=1, color='green', ls='--')
ax.set_xlabel(r"$a$")
ax.set_ylabel("PDF")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_activity_pdf.jpg", dpi=300)

print(a_pe, u_a_pe)



# Determine detection threshold
r_G_prime_samples = np.random.gamma(n_0, 1/t_0, size=m)
a_prime_samples = 1/(i_samples*e_samples) * (r_G_prime_samples - r_0_samples)
a_prime_pe = np.mean(a_prime_samples)
print(a_prime_pe, "should be close to 0...")

a_star = np.quantile(a_prime_samples, 1 - alpha)
print(a_star)

a_prime_histo, ex_a_prime = np.histogram(a_prime_samples, bins=50)
a_prime_histo_norm = a_prime_histo / (m * (ex_a_prime[1]-ex_a_prime[0]))
fig, ax = plt.subplots(1, 1, figsize=(120/inch_to_mm, 80/inch_to_mm))
ax.step(ex_a_prime[:-1], a_prime_histo_norm, where="post", label="MC samples")
ax.axvline(x=a_prime_pe, ymax=1, color='gray', ls='-', label=r"$a_{pe}=$"+f"{a_prime_pe:.2f} (want 0)")
ax.axvline(x=a_star, ymax=1, color='red', ls='-', label=r"$a^\ast=$"+f"{a_star:.2f}")
ax.set_xlabel(r"$a$")
ax.set_ylabel("PDF")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_activity_prime_pdf.jpg", dpi=300)



# Best estimate




# Determine detection limit
def objective_function(n_G_value):
    r_G_primeprime_samples = np.random.gamma(n_G_value, 1/t_G, size=m)
    a_primeprime_samples = 1/(i_samples*e_samples) * (r_G_primeprime_samples - r_0_samples)
    # beta_candidate = np.mean(y_hash_samples < y_star)
    a_star_candidate = np.quantile(a_primeprime_samples, beta)
    # loss = beta_candidate - beta
    loss = a_star_candidate - a_star
    return loss

n_G_hash = brentq(objective_function, 1*n_0, 10*n_0)
r_G_primeprime_samples = np.random.gamma(n_G_hash, 1/t_G, size=m)
a_primeprime_samples = 1/(i_samples*e_samples) * (r_G_primeprime_samples - r_0_samples)
a_hash = np.mean(a_primeprime_samples)
print(a_hash)

a_primeprime_histo, ex_a_primeprime = np.histogram(a_primeprime_samples, bins=50)
a_primeprime_histo_norm = a_primeprime_histo / (m * (ex_a_primeprime[1]-ex_a_primeprime[0]))
fig, ax = plt.subplots(1, 1, figsize=(120/inch_to_mm, 80/inch_to_mm))
ax.step(ex_a_prime[:-1], a_prime_histo_norm, where="post", label="MC samples")
ax.axvline(x=a_prime_pe, ymax=1, color='gray', ls='-', label=r"$a_{pe}=$"+f"{a_prime_pe:.2f} (want 0)")
ax.axvline(x=a_star, ymax=1, color='red', ls='-', label=r"$a^\ast=$"+f"{a_star:.2f}")
ax.step(ex_a_primeprime[:-1], a_primeprime_histo_norm, where="post", label="MC samples")
ax.axvline(x=a_hash, ymax=1, color='blue', ls='-', label=r"$a^\#=$"+f"{a_hash:.2f}")
ax.set_xlabel(r"$a$")
ax.set_ylabel("PDF")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_activity_primeprime_pdf.jpg", dpi=300)



plt.show()