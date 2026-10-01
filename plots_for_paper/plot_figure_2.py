import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

np.random.seed(42)
inch_to_mm = 25.4
colors = plt.get_cmap('tab10').colors
n_values = [1, 1, 5, 15, 50]
t_values = [20, 40, 50, 150, 500]
rate_x = np.linspace(0, 0.3, int(1e3))

fig, ax = plt.subplots(1, 1, figsize=(83/inch_to_mm, 60/inch_to_mm))
for n, t in zip(n_values, t_values):
    shape = n
    scale = 1/t
    posterior_rate = gamma.pdf(rate_x, shape, scale=scale)
    ax.plot(rate_x, posterior_rate, label=r"$n$=" + f"{n}, " + r"$t$=" + f"{t} s")
ax.set_xlabel(r"$\tilde{r}$ (1/s)")
ax.set_ylabel(r"$f_R(\tilde{r} | n, t)$ (s)")
ax.legend(frameon=False)

plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figures/figure_2.jpg", dpi=300)
plt.savefig("figures/figure_2.pdf")

plt.show()