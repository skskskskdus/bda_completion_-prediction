import json
import os

notebook_path = r'c:\Users\dell\bda\baseline.ipynb'
after_id = 'viz_corr'
new_cell = {
 "cell_type": "code",
 "execution_count": None,
 "id": "viz_top_features",
 "metadata": {},
 "outputs": [],
 "source": [
  "# Analysis of Top Correlated Features\n",
  "top_features = target_corr.abs().sort_values(ascending=False).head(4).index.tolist()\n",
  "\n",
  "print(f\"Top 4 features correlated with '{target_col}': {top_features}\")\n",
  "\n",
  "plt.figure(figsize=(15, 10))\n",
  "for i, col in enumerate(top_features):\n",
  "    plt.subplot(2, 2, i+1)\n",
  "    # Using boxplot as it's better for numerical vs binary target\n",
  "    sns.boxplot(x=target_col, y=col, data=train, palette='husl')\n",
  "    plt.title(f'{col} Distribution by {target_col}', fontsize=12)\n",
  "    plt.grid(axis='y', linestyle='--', alpha=0.3)\n",
  "\n",
  "plt.tight_layout()\n",
  "plt.show()\n"
 ]
}

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
found = False
for cell in nb.get('cells', []):
    new_cells.append(cell)
    if cell.get('id') == after_id:
        new_cells.append(new_cell)
        found = True

if not found:
    print(f"Error: Cell ID '{after_id}' not found")
else:
    nb['cells'] = new_cells
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully added cell after '{after_id}'")
