# Correlation Analysis
plt.figure(figsize=(15, 12))

# Calculate correlation
correlation = train[numeric_cols + [target_col]].corr()

# 1. Correlation Heatmap
plt.subplot(2, 1, 1)
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"size": 10})
plt.title('Correlation Heatmap (Numerical Variables)', fontsize=15)

# 2. Correlation with Target (Bar Plot)
plt.subplot(2, 1, 2)
target_corr = correlation[target_col].drop(target_col).sort_values(ascending=False)
colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in target_corr]
target_corr.plot(kind='barh', color=colors)
plt.title(f'Correlation of Features with {target_col}', fontsize=15)
plt.xlabel('Correlation Coefficient')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
