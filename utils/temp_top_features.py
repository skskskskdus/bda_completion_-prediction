# Analysis of Top Correlated Features
top_features = target_corr.abs().sort_values(ascending=False).head(4).index.tolist()

print(f"Top 4 features correlated with '{target_col}': {top_features}")

plt.figure(figsize=(15, 10))
for i, col in enumerate(top_features):
    plt.subplot(2, 2, i+1)
    # Using boxplot as it's better for numerical vs binary target
    sns.boxplot(x=target_col, y=col, data=train, palette='husl')
    plt.title(f'{col} Distribution by {target_col}', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
