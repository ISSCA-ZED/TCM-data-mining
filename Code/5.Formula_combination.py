import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

formula_path = os.path.join(PROJECT_ROOT, "Data", "Entity-Normalized.xlsx")
pinyin_path = os.path.join(PROJECT_ROOT, "Data", "Entity network", "Pinyin name.xlsx")
output_dir = os.path.join(PROJECT_ROOT, "Results", "Formula")
output_file = os.path.join(output_dir, "Formula_combination_statistics.xlsx")

# === Step 1. Load data ===
formula_df = pd.read_excel(formula_path)
pinyin_df = pd.read_excel(pinyin_path)

# === Step 2. Detect relevant columns ===
formula_col = "Formula name"

possible_dynasty_cols = ["Dynasty", "朝代", "时代"]
dynasty_col = None
for col in possible_dynasty_cols:
    if col in formula_df.columns:
        dynasty_col = col
        break

if dynasty_col is None:
    raise KeyError(
        "No dynasty column found. Please check whether the file contains 'Dynasty', '朝代', or '时代'."
    )

herb_cols = [col for col in formula_df.columns if "Formula herb" in str(col)]
if not herb_cols:
    raise KeyError("No columns found containing 'Formula herb'.")

print(f"Detected herb columns: {herb_cols}")
print(f"Detected dynasty column: {dynasty_col}")

# === Step 3. Build Pinyin mapping ===
pinyin_mapping = dict(zip(pinyin_df.iloc[:, 0], pinyin_df.iloc[:, 1]))

# === Step 4. Combine herb columns into one list per formula ===
def combine_herbs(row):
    herbs = []
    for col in herb_cols:
        val = row[col]
        if pd.notna(val):
            val = str(val).strip()
            if val not in ["", "nan", "None"]:
                herbs.append(val)
    return herbs

formula_df["Herb List CN"] = formula_df.apply(combine_herbs, axis=1)

# === Step 5. Map Chinese herb names to Pinyin ===
def map_to_pinyin(herb_list):
    return [pinyin_mapping.get(h, h) for h in herb_list]

formula_df["Herb List PY"] = formula_df["Herb List CN"].apply(map_to_pinyin)

# === Step 6. Create normalized herb combination keys ===
formula_df["Formula herb_combination"] = formula_df["Herb List PY"].apply(
    lambda x: ";".join(sorted(set(x))) if len(x) > 0 else ""
)

formula_df["Formula herb_combination_CN"] = formula_df["Herb List CN"].apply(
    lambda x: ";".join(sorted(set(x))) if len(x) > 0 else ""
)

# === Step 7. Calculate herb count per formula ===
formula_df["Herb Count"] = formula_df["Herb List PY"].apply(
    lambda x: len(set(x)) if len(x) > 0 else 0
)

# === Step 8. Remove formulas with no valid herb combination ===
formula_df = formula_df[formula_df["Formula herb_combination"].str.strip() != ""].copy()

if formula_df.empty:
    raise ValueError("No valid herb combinations found. Please check the input data.")

# === Step 9. Aggregate formulas by identical herb combinations ===
def merge_formulas(series):
    unique_names = sorted(
        set([str(s) for s in series if pd.notna(s) and str(s).strip() != ""])
    )
    return ";".join(unique_names)

result_df = (
    formula_df.groupby("Formula herb_combination", dropna=False)
    .agg({
        formula_col: merge_formulas,
        "Formula herb_combination_CN": "first",
        "Herb Count": "first"
    })
    .reset_index()
)

# === Step 10. Add total frequency ===
result_df["Frequency"] = result_df["Formula herb_combination"].map(
    formula_df["Formula herb_combination"].value_counts()
)

# === Step 11. Rename columns ===
result_df.rename(columns={
    "Formula herb_combination_CN": "Formula herb_combination_Chinese name",
    formula_col: "Formula name"
}, inplace=True)

# === Step 12. Organize columns ===
result_df = result_df[
    [
        "Formula herb_combination",
        "Formula herb_combination_Chinese name",
        "Formula name",
        "Herb Count",
        "Frequency"
    ]
].sort_values(by=["Frequency", "Herb Count"], ascending=[False, False]).reset_index(drop=True)

# =========================================================
# Additional Output 1:
# Frequency of each formula combination in each dynasty
# (non-deduplicated frequency)
# =========================================================
dynasty_comb_freq_long = (
    formula_df.groupby([dynasty_col, "Formula herb_combination", "Formula herb_combination_CN"])
    .size()
    .reset_index(name="Frequency")
)

dynasty_comb_freq_long = dynasty_comb_freq_long.sort_values(
    by=[dynasty_col, "Frequency"], ascending=[True, False]
).reset_index(drop=True)

# Wide-format table: frequency of each combination across dynasties
dynasty_comb_freq_wide = dynasty_comb_freq_long.pivot_table(
    index=["Formula herb_combination", "Formula herb_combination_CN"],
    columns=dynasty_col,
    values="Frequency",
    aggfunc="sum",
    fill_value=0
).reset_index()

dynasty_freq_cols = [
    c for c in dynasty_comb_freq_wide.columns
    if c not in ["Formula herb_combination", "Formula herb_combination_CN"]
]

dynasty_comb_freq_wide["Total Frequency"] = dynasty_comb_freq_wide[dynasty_freq_cols].sum(axis=1)

dynasty_comb_freq_wide.rename(columns={
    "Formula herb_combination_CN": "Formula herb_combination_Chinese name"
}, inplace=True)

dynasty_comb_freq_wide = dynasty_comb_freq_wide.sort_values(
    by="Total Frequency", ascending=False
).reset_index(drop=True)

# =========================================================
# Additional Output 2:
# Number of unique formula combinations in each dynasty
# =========================================================
dynasty_unique_comb_count = (
    formula_df[[dynasty_col, "Formula herb_combination"]]
    .drop_duplicates()
    .groupby(dynasty_col)
    .size()
    .reset_index(name="Unique Formula Combination Count")
)

# =========================================================
# Additional Output 3:
# Deduplicated formula combination list for each dynasty,
# with the frequency of each combination within that dynasty
# Each row = one unique formula combination in one dynasty
# =========================================================
dynasty_unique_comb_with_freq = (
    formula_df.groupby(
        [dynasty_col, "Formula herb_combination", "Formula herb_combination_CN"],
        as_index=False
    )
    .agg(
        Frequency=("Formula herb_combination", "size"),
        Herb_Count=("Herb Count", "first"),
        Formula_Name=(formula_col, merge_formulas)
    )
)

dynasty_unique_comb_with_freq.rename(columns={
    "Formula herb_combination_CN": "Formula herb_combination_Chinese name",
    dynasty_col: "Dynasty"
}, inplace=True)

dynasty_unique_comb_with_freq = dynasty_unique_comb_with_freq.sort_values(
    by=["Dynasty", "Frequency", "Herb_Count"],
    ascending=[True, False, False]
).reset_index(drop=True)

# =========================================================
# Additional Output 4:
# Average herb count per formula in each dynasty
# Calculated based on original records
# =========================================================
avg_herb_count_by_dynasty = (
    formula_df.groupby(dynasty_col)
    .agg(
        Formula_Count=("Formula herb_combination", "count"),
        Mean_Herb_Count=("Herb Count", "mean"),
        Median_Herb_Count=("Herb Count", "median"),
        Min_Herb_Count=("Herb Count", "min"),
        Max_Herb_Count=("Herb Count", "max"),
        SD_Herb_Count=("Herb Count", "std")
    )
    .reset_index()
)

avg_herb_count_by_dynasty["Mean_Herb_Count"] = avg_herb_count_by_dynasty["Mean_Herb_Count"].round(2)
avg_herb_count_by_dynasty["Median_Herb_Count"] = avg_herb_count_by_dynasty["Median_Herb_Count"].round(2)
avg_herb_count_by_dynasty["SD_Herb_Count"] = avg_herb_count_by_dynasty["SD_Herb_Count"].round(2)
avg_herb_count_by_dynasty.rename(columns={dynasty_col: "Dynasty"}, inplace=True)

# =========================================================
# Additional Output 5:
# Summary statistics for each dynasty:
# total frequency + unique combination count + mean herb count
# =========================================================
dynasty_summary = (
    formula_df.groupby(dynasty_col)
    .agg(
        Total_Frequency=("Formula herb_combination", "count"),
        Unique_Combination_Count=("Formula herb_combination", pd.Series.nunique),
        Mean_Herb_Count=("Herb Count", "mean")
    )
    .reset_index()
)

dynasty_summary["Mean_Herb_Count"] = dynasty_summary["Mean_Herb_Count"].round(2)
dynasty_summary.rename(columns={dynasty_col: "Dynasty"}, inplace=True)

# === Step 13. Export results ===
os.makedirs(output_dir, exist_ok=True)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    result_df.to_excel(writer, sheet_name="Formula Combination Stats", index=False)

    # Wide-format table: frequency of each combination across dynasties
    dynasty_comb_freq_wide.to_excel(writer, sheet_name="Dynasty Comb Frequency Wide", index=False)

    # Long-format table: unique combinations and their frequency within each dynasty
    dynasty_unique_comb_with_freq.to_excel(writer, sheet_name="Dynasty Unique Comb Freq", index=False)

    # Number of unique combinations in each dynasty
    dynasty_unique_comb_count.rename(columns={dynasty_col: "Dynasty"}).to_excel(
        writer, sheet_name="Dynasty Unique Comb Count", index=False
    )

    # Average herb count per dynasty
    avg_herb_count_by_dynasty.to_excel(writer, sheet_name="Avg Herb Count by Dynasty", index=False)

    # Summary statistics for each dynasty
    dynasty_summary.to_excel(writer, sheet_name="Dynasty Summary", index=False)

print(f"Formula combination statistics have been saved to:\n{output_file}")
print(f"Total valid combinations counted: {len(result_df)}")
print("Additional outputs generated:")
print("  - Unique formula combination count by dynasty")
print("  - Frequency of each unique combination within each dynasty")
print("  - Average herb count per dynasty")