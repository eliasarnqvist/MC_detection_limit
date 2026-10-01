import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

np.random.seed(42)
inch_to_mm = 25.4
colors = plt.get_cmap('tab10').colors

# Number of primary MC trials
m = int(1e7)
# For the characteristic limits
alpha = 0.05
beta = 0.05

# Input quantities
n_G = 6
n_0 = 2
t_G = 1e2
t_0 = 1e2
w = 5
u_w = 0.6

# Input quantities
if n_G > 0:
    r_G_samples = np.random.gamma(n_G, 1/t_G, size=m)
else:
    r_G_samples = np.random.gamma(n_G + 1, 1/t_G, size=m)
if n_0 > 0:
    r_0_samples = np.random.gamma(n_0, 1/t_0, size=m)
else:
    r_0_samples = np.random.gamma(n_0 + 1, 1/t_0, size=m)
w_samples = np.random.normal(w, u_w, size=m)

# Apply model equation for measurand
y_samples = w_samples * (r_G_samples - r_0_samples)

# Primary estimate of measurand
y_pe = np.mean(y_samples)
u_y_pe = np.std(y_samples, ddof=1)
print("Primary estimate", y_pe, u_y_pe)

# Determine detection threshold
if n_0 > 0:
    r_G_ast_samples = np.random.gamma(n_0, 1/t_0, size=m)
else:
    r_G_ast_samples = np.random.gamma(n_0 + 1, 1/t_0, size=m)
y_ast_samples = w_samples * (r_G_ast_samples - r_0_samples)
y_ast_pe = np.mean(y_ast_samples)
print(y_ast_pe, "should be close to 0")

# Determine the detection threshold
y_ast = np.quantile(y_ast_samples, 1 - alpha)
print("y*", y_ast)

# Determine detection limit
def objective_function(n_G_value):
    r_G_hash_samples = np.random.gamma(n_G_value, 1/t_G, size=m)
    y_hash_samples = w_samples * (r_G_hash_samples - r_0_samples)
    y_ast_candidate = np.quantile(y_hash_samples, beta)
    loss = y_ast_candidate - y_ast
    return loss
n_G_hash = brentq(objective_function, 1*n_0, 10*n_0)
r_G_hash_samples = np.random.gamma(n_G_hash, 1/t_G, size=m)
y_hash_samples = w_samples * (r_G_hash_samples - r_0_samples)
y_hash = np.mean(y_hash_samples)
print("y#", y_hash)

# Best estimate of measurand
y_be_samples = y_samples[y_samples >= 0]
m_be = y_be_samples.size
y_be = np.mean(y_be_samples)
u_y_be = np.std(y_be_samples, ddof=1)
print("Best estimate", y_be, u_y_be)

# Plot the characteristic limits

fig, ax = plt.subplots(2, 1, figsize=(83/inch_to_mm, 90/inch_to_mm), sharex=True)

# Make histograms
histo_range = [-0.4, 1.3]
histo_bins = 200

y_histo, ex_y = np.histogram(y_samples, bins=histo_bins, range=histo_range)
y_histo_norm = y_histo / (m * (ex_y[1]-ex_y[0]))

# y_be_histo, ex_y_be = np.histogram(y_be_samples, bins=histo_bins, range=histo_range)
y_be_histo, ex_y_be = np.histogram(y_samples, bins=histo_bins, range=histo_range)
y_be_histo_norm = y_be_histo / (m_be * (ex_y_be[1]-ex_y_be[0]))

y_ast_histo, ex_y_ast = np.histogram(y_ast_samples, bins=histo_bins, range=histo_range)
y_ast_histo_norm = y_ast_histo / (m * (ex_y_ast[1]-ex_y_ast[0]))

y_hash_histo, ex_y_hash = np.histogram(y_hash_samples, bins=histo_bins, range=histo_range)
y_hash_histo_norm = y_hash_histo / (m * (ex_y_hash[1]-ex_y_hash[0]))

# Plot

ax[0].plot(ex_y[:-1], y_histo_norm, color=colors[0], label=r"$f_{Y}(\tilde{y}|I)$")
ax[0].axvline(x=y_pe, ymax=1, color=colors[0], ls='--', label=r"$y_{pe}$")

horizonal_values = np.concatenate([ex_y_be[:-1][ex_y_be[:-1]<0], [0, 0], ex_y_be[:-1][ex_y_be[:-1]>=0]])
vertical_value_at_0 = np.interp(0, ex_y_ast[:-1], y_be_histo_norm)
vertical_values = np.concatenate([np.zeros((ex_y_be[:-1]<0).sum()), [0, vertical_value_at_0], y_be_histo_norm[ex_y_be[:-1]>=0]])
ax[0].plot(horizonal_values, vertical_values, color=colors[1], label=r"$f_{Y}(\tilde{y}|I,\tilde{y}\geq0)$")
ax[0].axvline(x=y_be, ymax=1, color=colors[1], ls='--', label=r"$\hat{y}$")

ax[1].plot(ex_y_ast[:-1], y_ast_histo_norm, color=colors[2], label=r"$f_{Y}(\tilde{y}|I',\tilde{y}=0)$")
ax[1].axvline(x=y_ast, ymax=1, color=colors[2], ls='--', label=r"$y^\ast$")
horizonal_value_at_y_ast = np.interp(y_ast, ex_y_ast[:-1], y_ast_histo_norm)
horizonal_values = np.append(y_ast, ex_y_ast[:-1][ex_y_ast[:-1] >= y_ast])
vertical_values = np.append(horizonal_value_at_y_ast, y_ast_histo_norm[ex_y_ast[:-1] >= y_ast])
ax[1].fill_between(horizonal_values, 0, vertical_values, color=colors[2], alpha=0.25)

ax[1].plot(ex_y_hash[:-1], y_hash_histo_norm, color=colors[3], label=r"$f_{Y}(\tilde{y}|I',\tilde{y}=y^\#)$")
ax[1].axvline(x=y_hash, ymax=1, color=colors[3], ls='--', label=r"$y^\#$")
horizonal_value_at_y_ast = np.interp(y_ast, ex_y_hash[:-1], y_hash_histo_norm)
horizonal_values = np.append(ex_y_hash[:-1][ex_y_hash[:-1] <= y_ast], y_ast)
vertical_values = np.append(y_hash_histo_norm[ex_y_hash[:-1] <= y_ast], horizonal_value_at_y_ast)
ax[1].fill_between(horizonal_values, 0, vertical_values, color=colors[3], alpha=0.25)

ax[1].text(0.28, 0.16, r"$\beta$", transform=ax[1].transAxes, va="bottom", size=8)
ax[1].annotate("", xy=(0.33, 0.05), xytext=(.30, 0.17), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
ax[1].text(0.39, 0.16, r"$\alpha$", transform=ax[1].transAxes, va="bottom", size=8)
ax[1].annotate("", xy=(0.37, 0.05), xytext=(.40, 0.17), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))

ax[1].set_xlabel(r"$\tilde{y}$", size=8)
ax[0].set_ylabel(r"$f_Y(\tilde{y})$", size=8)
ax[1].set_ylabel(r"$f_Y(\tilde{y})$", size=8)

ax[0].legend(frameon=False, fontsize=8, handlelength=1.8)
ax[1].legend(frameon=False, fontsize=8, handlelength=1.8)
ax[0].set_yticks([0])
ax[0].set_yticklabels([0])
ax[1].set_yticks([0])
ax[1].set_yticklabels([0])
ax[1].set_xticks([0])
ax[1].set_xticklabels([0])
ax[0].grid()
ax[1].grid()

plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figures/figure_1.jpg", dpi=300)
plt.savefig("figures/figure_1.pdf")

plt.show()
