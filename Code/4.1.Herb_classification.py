import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pattern_file = os.path.join(PROJECT_ROOT, "Results", "Herb", "herb_evolution_patterns.csv")
dynasty_file = os.path.join(PROJECT_ROOT, "Data", "Herb", "Herb-Dynasty and Frequency.xlsx")
output_path = os.path.join(PROJECT_ROOT, "Results", "Herb", "Herb_Pattern_Distribution.tiff")

# ===============================
# 2. Read CSV (Pattern information)
# ===============================
pattern_df = pd.read_csv(pattern_file)
pattern_df.columns = [col.strip() for col in pattern_df.columns]

# ===============================
# 3. Read Excel (two columns per dynasty)
# ===============================
df = pd.read_excel(dynasty_file, header=1)
dynasty_row = pd.read_excel(dynasty_file, header=None, nrows=1).iloc[0]

# ===============================
# 4. Construct long-format table
# ===============================
long_df_list = []
for i in range(0, df.shape[1], 2):
    dynasty_name = dynasty_row[i]
    sub_df = df.iloc[:, [i, i + 1]].copy()
    sub_df.columns = ["Herb", "Frequency"]
    sub_df["Dynasty"] = dynasty_name
    long_df_list.append(sub_df)

dynasty_df = pd.concat(long_df_list, ignore_index=True)

# Clean data
dynasty_df = dynasty_df.dropna(subset=["Herb", "Frequency"])
dynasty_df["Frequency"] = pd.to_numeric(dynasty_df["Frequency"], errors="coerce")
dynasty_df = dynasty_df.dropna(subset=["Frequency"])

# ===============================
# 5. Merge pattern information
# ===============================
merged_df = pd.merge(dynasty_df, pattern_df, on="Herb", how="left")

# ===============================
# 6. Select all herbs per dynasty
# ===============================
All_per_dynasty = merged_df.copy()

# ===============================
# 7. Calculate pattern percentage
# ===============================
pattern_count = (
    All_per_dynasty.groupby(["Dynasty", "Pattern"]).size().reset_index(name="Count")
)

pattern_count["Percent"] = (
    pattern_count.groupby("Dynasty")["Count"].transform(lambda x: x / x.sum() * 100)
)

pattern_pivot = pattern_count.pivot(
    index="Dynasty", columns="Pattern", values="Percent"
).fillna(0)

# ===============================
# 8. Set dynasty order
# ===============================
dynasty_order = [
    'before the end of Han Dynasty',
    'Wei, Jin, Southern and Northern Dynasties',
    'Sui, Tang and Five Dynasties',
    'Song, Jin and Yuan Dynasties',
    'Ming Dynasty',
    'Qing Dynasty',
    'Modern Times'
]

dynasty_order_existing = [d for d in dynasty_order if d in pattern_pivot.index]
pattern_pivot = pattern_pivot.loc[dynasty_order_existing]

# ===============================
# 8.1 Rename x-axis labels with numbering
# ===============================
dynasty_label_map = {
    'before the end of Han Dynasty': 'before the end of Han Dynasty (1)',
    'Wei, Jin, Southern and Northern Dynasties': 'Wei, Jin, Southern and Northern Dynasties (2)',
    'Sui, Tang and Five Dynasties': 'Sui, Tang and Five Dynasties (3)',
    'Song, Jin and Yuan Dynasties': 'Song, Jin and Yuan Dynasties (4)',
    'Ming Dynasty': 'Ming Dynasty (5)',
    'Qing Dynasty': 'Qing Dynasty (6)',
    'Modern Times': 'Modern Times (7)'
}

pattern_pivot.index = [dynasty_label_map.get(d, d) for d in pattern_pivot.index]

# ===============================
# 9. Define fixed pattern–color mapping (light warm tones)
# ===============================
color_map = {
    "Classical herb": "#A3E4D7",  # light cyan-green
    "Common herb": "#AED6F1",     # light blue
    "Early herb": "#F9E79F",      # light yellow
    "Inherited herb": "#F5B7B1",  # light pink
    "Later herb": "#D7BDE2",      # light purple
    "Stage herb": "#FAD7A0"       # light orange
}

# ===============================
# 10. Draw stacked bar chart
# ===============================
ax = pattern_pivot.plot(
    kind="bar",
    stacked=True,
    edgecolor="black",
    figsize=(12, 7),
    color=[color_map[p] for p in pattern_pivot.columns]
)

plt.xlabel("Dynasty", fontsize=16, fontweight='bold')
plt.ylabel("Percentage (%)", fontsize=16, fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=12, fontweight='bold')
plt.yticks(rotation=0, fontsize=12, fontweight='bold')
plt.legend(title="Pattern", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
plt.tight_layout()

# ===============================
# 11. Save high-resolution TIFF
# ===============================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()

print(f"✅ Figure saved to: {output_path}")