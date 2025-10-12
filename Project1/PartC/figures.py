import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend suitable for file output
import matplotlib.pyplot as plt


# Predictor names
predictors = ["Cement", "Slag", "Fly Ash", "Water", "Superplasticizer",
              "Coarse Agg.", "Fine Agg.", "Age"]

# Example raw coefficients from Part B (approximate)
coefficients = [0.133, 0.125, 0.107, -0.133, 0.116, 0.029, 0.034, 0.119]

# Corresponding p-values (raw)
p_values = [6.97e-41, 4.15e-26, 1.89e-13, 1.92e-3, 2.46e-1,
            5.44e-3, 2.74e-3, 9.52e-74]

# Convert p-values to -log10(p)
log_p = -np.log10(p_values)

# Set up figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart for coefficients
bar1 = ax1.bar(predictors, coefficients, color='steelblue')
ax1.set_ylabel('Coefficient Value')
ax1.set_title('OLS Coefficients and Significance (Raw Predictors)')
ax1.axhline(0, color='black', linewidth=0.8)

# Secondary axis for -log10(p)
ax2 = ax1.twinx()
bar2 = ax2.plot(predictors, log_p, color='darkred', marker='o', linestyle='-')
ax2.set_ylabel(r'$-\log_{10}(p)$')

plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('figure1_coefficients_significance.png', dpi=300)
plt.show()

# p-values before (raw)
p_raw = np.array([6.97e-41, 4.15e-26, 1.89e-13, 1.92e-3, 2.46e-1,
                  5.44e-3, 2.74e-3, 9.52e-74])

# p-values after log transform
p_log = np.array([1.30e-86, 4.52e-36, 2.61e-1, 1.86e-20, 1.04e-5,
                  4.08e-1, 9.48e-2, 1.75e-192])

log_p_raw = -np.log10(p_raw)
log_p_log = -np.log10(p_log)

x = np.arange(len(predictors))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, log_p_raw, width, label='Raw')
bars2 = ax.bar(x + width/2, log_p_log, width, label='Log-Transformed')

ax.set_xticks(x)
ax.set_xticklabels(predictors, rotation=30, ha='right')
ax.set_ylabel(r'$-\log_{10}(p)$')
ax.set_title('Comparison of Predictor Significance (Raw vs Log Transformed)')
ax.legend()

plt.tight_layout()
plt.savefig('figure2_pvalues_comparison.png', dpi=300)
plt.show()
