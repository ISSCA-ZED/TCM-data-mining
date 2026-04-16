import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_data = os.path.join(PROJECT_ROOT, "Data", "Entity-Normalized.xlsx")
file_pinyin = os.path.join(PROJECT_ROOT, "Data", "Entity network", "Pinyin name.xlsx")
output_dir = os.path.join(PROJECT_ROOT, "Results", "Symptom")
os.makedirs(output_dir, exist_ok=True)

output_heatmap = os.path.join(output_dir, "Symptom_Frequencies_Heatmap_by_Dynasty.tiff")
output_table = os.path.join(output_dir, "Symptom_Frequency_Distribution_by_Dynasty.xlsx")
output_bar = os.path.join(output_dir, "Symptom-Top20.tiff")

# ===========================
# Load data
# ===========================
df = pd.read_excel(file_data)
pinyin_df = pd.read_excel(file_pinyin)

# ===========================
# Reshape symptom columns
# ===========================
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
df_melted = df.melt(
    id_vars=["Dynasty"],
    value_vars=symptom_cols,
    var_name="Symptom_Type",
    value_name="Symptom"
)
df_melted = df_melted.dropna(subset=["Symptom"])

# ===========================
# Count frequency
# ===========================
symptom_counts = df_melted.groupby(["Dynasty", "Symptom"]).size().reset_index(name="Count")

# ===========================
# Merge with pinyin normalization
# ===========================
symptom_counts_std = symptom_counts.merge(
    pinyin_df, left_on="Symptom", right_on="Chinese name", how="left"
)
symptom_counts_std["Display_Name"] = symptom_counts_std["Pinyin name"].fillna(symptom_counts_std["Symptom"])

# ===========================
# Define dynasty order
# ===========================
dynasty_order = [
    "before the end of Han Dynasty",
    "Wei, Jin, Southern and Northern Dynasties",
    "Sui, Tang and Five Dynasties",
    "Song, Jin and Yuan Dynasties",
    "Ming Dynasty",
    "Qing Dynasty",
    "Modern Times"
]
symptom_counts_std["Dynasty"] = pd.Categorical(
    symptom_counts_std["Dynasty"], categories=dynasty_order, ordered=True
)

# ===========================
# Pivot table for heatmap
# ===========================
heatmap_data = symptom_counts_std.pivot_table(
    index="Display_Name", columns="Dynasty", values="Count", fill_value=0
)
heatmap_data = heatmap_data.astype(int)

# ===========================
# Draw heatmap
# ===========================
plt.figure(figsize=(18, 22))
ax = sns.heatmap(
    heatmap_data, cmap="YlGnBu", linewidths=0.5, annot=False,
    cbar_kws={"label": "Frequency"}
)

plt.xlabel("Dynasty", fontsize=20, fontweight='bold', labelpad=10)
plt.ylabel("Symptom", fontsize=20, fontweight='bold', labelpad=10)
plt.xticks(rotation=30, ha="right", fontsize=13)
plt.yticks(rotation=0, fontsize=12)
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
plt.tight_layout()

cbar = ax.collections[0].colorbar
cbar.set_label("Frequency", fontsize=16)
cbar.ax.tick_params(labelsize=16)

plt.savefig(output_heatmap, dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ===========================
# Generate and export frequency table
# ===========================
total_counts = symptom_counts_std.groupby("Display_Name")["Count"].sum().reset_index(name="Total_Frequency")
dynasty_pivot = symptom_counts_std.pivot_table(
    index="Display_Name", columns="Dynasty", values="Count", fill_value=0
)
freq_df = total_counts.merge(dynasty_pivot, on="Display_Name", how="left")
freq_df.to_excel(output_table, index=False)

# ===========================
# Draw Top 20 bar chart
# ===========================
top20 = total_counts.sort_values(by="Total_Frequency", ascending=False).head(20)
sym_names = top20["Display_Name"].tolist()
sym_counts = top20["Total_Frequency"].tolist()

# Generate color mapping (reverse viridis palette)
colors = list(reversed(sns.color_palette("viridis", len(sym_names))))

plt.figure(figsize=(10, 8))
plt.barh(sym_names, sym_counts, color=colors)
plt.xlabel("Frequency", fontsize=15, fontweight='bold')
plt.ylabel("Symptom", fontsize=15, fontweight='bold')
plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()  # Show highest frequency at top

# Add value labels
for index, value in enumerate(sym_counts):
    plt.text(value + 0.5, index, str(value), va='center', fontsize=10)

# Save figure
plt.savefig(output_bar, dpi=600, format="tiff", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

print(f"✅ Heatmap saved to: {output_heatmap}")
print(f"✅ Symptom frequency table exported to: {output_table}")
print(f"✅ Top 20 symptom frequency bar chart saved to: {output_bar}")
