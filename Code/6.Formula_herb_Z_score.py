import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['font.family'] = 'Arial'

file_path = os.path.join(PROJECT_ROOT, "Data", "Formula herb", "Formula herb-Dynasty and Frequency.xlsx")
output_path = os.path.join(PROJECT_ROOT, "Results", "Formula herb")
os.makedirs(output_path, exist_ok=True)

df = pd.read_excel(file_path)

# === 2. Dynasty column names ===
columns = [
    "before the end of Han Dynasty",
    "Wei, Jin, Southern and Northern Dynasties",
    "Sui, Tang and Five Dynasties",
    "Song, Jin and Yuan Dynasties",
    "Ming Dynasty",
    "Qing Dynasty",
    "Modern Times"
]

# === 3. Extract herb-frequency data and merge ===
data = {}
for i, col in enumerate(columns):
    herb_col = col
    freq_col = f"Unnamed: {i*2+1}"
    temp = df[[herb_col, freq_col]].iloc[1:].dropna()
    temp.columns = ["Herb", col]
    temp[col] = pd.to_numeric(temp[col], errors='coerce').astype(float).fillna(0.0)
    data[col] = temp

herb_list = sorted(set(h for t in data.values() for h in t["Herb"]))
matrix = pd.DataFrame(index=herb_list, columns=columns).fillna(0.0)

for col in columns:
    temp = data[col]
    for _, row in temp.iterrows():
        matrix.loc[row["Herb"], col] = row[col]

matrix = matrix.apply(pd.to_numeric, errors='coerce').astype(float).fillna(0.0)

# === 4. Select Top20 herbs ===
matrix["Total"] = matrix.sum(axis=1)
top20 = matrix.sort_values("Total", ascending=False).head(20).drop(columns="Total")

# === 5. Column-wise Z-score standardization ===
zscore_matrix = top20.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# === 6. Plot heatmap and save directly as TIFF ===
plt.figure(figsize=(12, 12))
ax = sns.heatmap(
    zscore_matrix,
    annot=True,
    fmt=".3f",
    annot_kws={"size": 14},
    cmap="RdYlBu_r",
    center=0,
    cbar_kws={'label': 'Z-score'}
)
plt.xlabel("Dynasty", fontname="Arial", fontsize=20, fontweight='bold')
plt.ylabel("Herb", fontname="Arial", fontsize=20, fontweight='bold')

plt.xticks(rotation=30, ha='right', fontname="Arial", fontsize=16, fontweight='bold')
plt.yticks(fontname="Arial", fontsize=16, fontweight='bold')

# Colorbar font size
cbar = ax.collections[0].colorbar
cbar.set_label("Z-score", fontsize=16)  # label font
cbar.ax.tick_params(labelsize=16)  # tick font

plt.tight_layout()

final_tiff = os.path.join(output_path, "Top20_Formula herbs_Zscore_Heatmap.tiff")
plt.savefig(final_tiff, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close()

print(f"TIFF image saved at: {final_tiff}")