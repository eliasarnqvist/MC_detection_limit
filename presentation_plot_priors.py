import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma
from scipy.optimize import brentq

np.random.seed(42)
inch_to_mm = 25.4

# Poisson rate

# Plot prior
# Input values
n = 2
t = 10
# Posterior
shape = n
scale = 1/t
pdf_rate_x = np.linspace(0, 2, 100)
pdf_rate = gamma.pdf(pdf_rate_x, shape, scale=scale)
info_text = r"$n=$" + f"{n}, " + r"$t=$" + f"{t}"

fig, ax = plt.subplots(1, 1, figsize=(60/inch_to_mm, 50/inch_to_mm))
ax.plot(pdf_rate_x, pdf_rate, label=r"$f_R(\tilde{r} \mid n, t)$")
ax.set_xlabel(r"$\tilde{r}$")
ax.set_ylabel(r"$f_R(\tilde{r})$")
ax.legend(title=info_text, frameon=False)
plt.tight_layout(pad = 0.2)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figs/rate_posterior.jpg", dpi=300)


# a = x + 1/2
# b = n - x + 1/2
# pdf_eff_x = np.linspace(0, 5e-8, 1000)
# pdf_eff = beta.pdf(pdf_eff_x, a, b)



plt.show()