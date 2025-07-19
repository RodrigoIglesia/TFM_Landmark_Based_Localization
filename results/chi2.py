import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Grados de libertad y nivel de significancia
df = 3
alpha = 0.01

# Rango de valores Chi²
x = np.linspace(0, 15, 500)
y = chi2.pdf(x, df)

# Valor crítico para el nivel de confianza (1 - alpha)
chi2_critical = chi2.ppf(1 - alpha, df)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=f"Chi² distribution (df={df})")
plt.fill_between(x, y, where=(x >= chi2_critical), color='red', alpha=0.5, label=f"Rejection region (α={alpha})")
plt.axvline(chi2_critical, color='red', linestyle='--', label=f"Critical value = {chi2_critical:.3f}")
plt.title("Chi² Distribution with 3 Degrees of Freedom")
plt.xlabel("Chi² value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
