import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm
from scipy.optimize import brentq

np.random.seed(42)
inch_to_mm = 25.4

# Background simulation
s_sim = 1e9
t_sim = 1e5
k_sim_1 = 10
k_sim_2 = 3
k_sim_3 = 0

m_sim = int(1e3)

P_sim_1 = np.random.beta(k_sim_1+1/2, s_sim-k_sim_1+1/2, size=m_sim)
P_sim_2 = np.random.beta(k_sim_2+1/2, s_sim-k_sim_2+1/2, size=m_sim)
P_sim_3 = np.random.beta(k_sim_3+1/2, s_sim-k_sim_3+1/2, size=m_sim)

R_sim_1 = P_sim_1 * s_sim / t_sim
R_sim_2 = P_sim_2 * s_sim / t_sim
R_sim_3 = P_sim_3 * s_sim / t_sim

R_sim_samples = R_sim_1 + R_sim_2 + R_sim_3

n_G_samples = np.random.poisson(R_sim_samples*t_sim, size=m_sim)

R_sim_histo, ex_R_sim = np.histogram(R_sim_samples, bins=50)
R_sim_histo_norm = R_sim_histo / (m_sim * (ex_R_sim[1]-ex_R_sim[0]))
fig, ax = plt.subplots(1, 1, figsize=(80/inch_to_mm, 50/inch_to_mm))
ax.step(ex_R_sim[:-1], R_sim_histo_norm, where="post", label="MC samples")
# xmin, xmax = ax.get_xlim()
# xx = np.linspace (xmin, xmax, 100)
# ax.plot(xx, beta.pdf(xx, k+1/2, s-k+1/2), ls="--", label="Beta distr.")
ax.set_xlabel(r"$r_{sim}$")
ax.set_ylabel("PDF")
# ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_pt2_simulated_rate_pdf.jpg", dpi=300)

bins = np.arange(np.floor(n_G_samples.min()) - 0.5, np.ceil(n_G_samples.max()) + 1.5, 1)
n_G_histo, ex_n_G = np.histogram(n_G_samples, bins=bins)
n_G_histo_norm = n_G_histo / (m_sim * (ex_n_G[1]-ex_n_G[0]))
fig, ax = plt.subplots(1, 1, figsize=(80/inch_to_mm, 50/inch_to_mm))
ax.step(ex_n_G[:-1], n_G_histo_norm, where="post", label="MC samples")
# xmin, xmax = ax.get_xlim()
# xx = np.linspace (xmin, xmax, 100)
# ax.plot(xx, beta.pdf(xx, k+1/2, s-k+1/2), ls="--", label="Beta distr.")
ax.set_xlabel(r"$n_{sim}$")
ax.set_ylabel("PDF")
# ax.legend(frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_pt2_simulated_background_counts_pdf.jpg", dpi=300)

###

# Number of primary MC trials
m = int(1e4)
# For the characteristic limits
alpha = 0.05
beta = 0.05

a_star_samples = []
a_hash_samples = []

for n_0 in n_G_samples:
    # Input quantities
    n_G = 10
    # n_0 = 4
    t_G = 1e3
    t_0 = 1e3
    k = 1e5
    s = 1e7
    i = 0.3
    u_i = 0.03

    # Intermediate quantities
    e_samples = np.random.beta(k+1/2, s-k+1/2, size=m)
    # r_G_samples = np.random.gamma(n_G, 1/t_G, size=m)
    r_0_samples = np.random.gamma(n_0, 1/t_0, size=m)
    i_samples = np.random.normal(i, u_i, size=m)

    # # Apply model equation
    # a_samples = 1/(i_samples*e_samples) * (r_G_samples - r_0_samples)

    # a_pe = np.mean(a_samples)
    # u_a_pe = np.std(a_samples)

    # Determine detection threshold
    r_G_prime_samples = np.random.gamma(n_0, 1/t_0, size=m)
    a_prime_samples = 1/(i_samples*e_samples) * (r_G_prime_samples - r_0_samples)
    a_prime_pe = np.mean(a_prime_samples)
    # print(a_prime_pe, "should be close to 0...")

    a_star = np.quantile(a_prime_samples, 1 - alpha)
    # print(a_star)

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
    # print(a_hash)

    a_star_samples.append(a_star)
    a_hash_samples.append(a_hash)

a_star_samples = np.array(a_star_samples)
a_hash_samples = np.array(a_hash_samples)

a_star_pe = np.mean(a_star_samples)
u_a_star_pe = np.std(a_star_samples)

a_hash_pe = np.mean(a_hash_samples)
u_a_hash_pe = np.std(a_hash_samples)



a_star_histo, ex_a_star = np.histogram(a_star_samples, bins=50)
a_star_histo_norm = a_star_histo / (m * (ex_a_star[1]-ex_a_star[0]))
fig, ax = plt.subplots(1, 1, figsize=(120/inch_to_mm, 80/inch_to_mm))
ax.step(ex_a_star[:-1], a_star_histo_norm, where="post", label="MC samples")
ax.axvline(x=a_star_pe, ymax=1, color='red', ls='-', label=r"$a^\ast_{pe}=$"+f"{a_star_pe:.2f} ({u_a_star_pe:.2f})")
ax.axvline(x=a_star_pe-u_a_star_pe, ymax=1, color='red', ls='--')
ax.axvline(x=a_star_pe+u_a_star_pe, ymax=1, color='red', ls='--')
ax.set_xlabel(r"$a^\ast$")
ax.set_ylabel("PDF")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_pt2_simulated_a_star_pdf.jpg", dpi=300)

a_hash_histo, ex_a_hash = np.histogram(a_hash_samples, bins=50)
a_hash_histo_norm = a_hash_histo / (m * (ex_a_hash[1]-ex_a_hash[0]))
fig, ax = plt.subplots(1, 1, figsize=(120/inch_to_mm, 80/inch_to_mm))
ax.step(ex_a_hash[:-1], a_hash_histo_norm, where="post", label="MC samples")
ax.axvline(x=a_hash_pe, ymax=1, color='red', ls='-', label=r"$a^\#_{pe}=$"+f"{a_hash_pe:.2f} ({u_a_hash_pe:.2f})")
ax.axvline(x=a_hash_pe-u_a_hash_pe, ymax=1, color='red', ls='--')
ax.axvline(x=a_hash_pe+u_a_hash_pe, ymax=1, color='red', ls='--')
ax.set_xlabel(r"$a^\#$")
ax.set_ylabel("PDF")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/example_pt2_simulated_a_hash_pdf.jpg", dpi=300)





plt.show()