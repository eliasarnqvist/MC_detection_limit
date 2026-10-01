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
plt.savefig("figures/figure_2.jpg", dpi=300)

plt.showfig()