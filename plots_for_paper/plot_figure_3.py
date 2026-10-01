import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

np.random.seed(42)
inch_to_mm = 25.4
colors = plt.get_cmap('tab10').colors
k_values = [1, 1, 5, 25, 100]
s_values = [50, 100, 50, 250, 1000]
efficiency_x = np.linspace(0, 0.2, int(1e3))

fig, ax = plt.subplots(1, 1, figsize=(83/inch_to_mm, 60/inch_to_mm))
for k, s in zip(k_values, s_values):
    a = k + 1/2
    b = s - k + 1/2
    posterior_rate = beta.pdf(efficiency_x, a, b)
    ax.plot(efficiency_x, posterior_rate, label=r"$k$=" + f"{k}, " + r"$s$=" + f"{s}")
ax.set_xlabel(r"$\tilde{\varepsilon}$", size=8)
ax.set_ylabel(r"$f_E(\tilde{\varepsilon} | k, s)$", size=8)
ax.legend(frameon=False, fontsize=8)
ax.tick_params(axis='both', labelsize=8)

plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figures/figure_3.jpg", dpi=300)
plt.savefig("figures/figure_3.pdf")

plt.show()