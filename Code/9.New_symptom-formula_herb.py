import os
import re
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(PROJECT_ROOT, "Data", "Entity-Normalized.xlsx")
pinyin_file = os.path.join(PROJECT_ROOT, "Data", "Entity network", "Pinyin name.xlsx")
output_dir = os.path.join(PROJECT_ROOT, "Results", "Symptom-Formula herb")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "01_New_Pair_Tables"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "02_Node_Edge_Tables"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "03_Network_Figures"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "04_Summary_Figures"), exist_ok=True)

# ==============================
# 2. Dynasty order
# ==============================
dynasty_order = [
    "before the end of Han Dynasty",
    "Wei, Jin, Southern and Northern Dynasties",
    "Sui, Tang and Five Dynasties",
    "Song, Jin and Yuan Dynasties",
    "Ming Dynasty",
    "Qing Dynasty",
    "Modern Times"
]

dynasty_alias_map = {
    "before the end of Han Dynasty": "before the end of Han Dynasty",
    "Before the end of Han Dynasty": "before the end of Han Dynasty",
    "Wei, Jin, Southern and Northern Dynasties": "Wei, Jin, Southern and Northern Dynasties",
    "Sui, Tang and Five Dynasties": "Sui, Tang and Five Dynasties",
    "Song, Jin and Yuan Dynasties": "Song, Jin and Yuan Dynasties",
    "Ming Dynasty": "Ming Dynasty",
    "Qing Dynasty": "Qing Dynasty",
    "Modern Times": "Modern Times"
}

# ==============================
# 3. Parameters
# ==============================
TOP_N = 20
SYMPTOM_COLOR = "#fc8d62"
HERB_COLOR = "#66c2a5"

# network node size: fixed size for both symptoms and formula herbs
FIXED_NODE_SIZE = 1200

# 4:3 ratio for summary figures
FIG_W_4_3 = 10
FIG_H_4_3 = 7.5

# ==============================
# 4. Helper functions
# ==============================
def clean_cell(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "" or x.lower() in ["nan", "none", "null"]:
        return None
    return x

def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def safe_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def wrap_text(text, width=18):
    """
    Wrap text by spaces when possible.
    Do not break a single word into pieces.
    This avoids cases like 'Ying' -> 'Yi\\nng'.
    """
    if text is None:
        return ""

    text = str(text).strip()
    if len(text) <= width:
        return text

    if " " in text:
        words = text.split()
        lines = []
        current = words[0]

        for w in words[1:]:
            if len(current) + 1 + len(w) <= width:
                current += " " + w
            else:
                lines.append(current)
                current = w
        lines.append(current)
        return "\n".join(lines)

    # if there is no space, keep as one whole token
    return text

def normalize_dynasty(x):
    x = clean_cell(x)
    if x is None:
        return None
    return dynasty_alias_map.get(x, x)

# ==============================
# 5. Read input data
# ==============================
df = pd.read_excel(input_file)
df.columns = [str(c).strip() for c in df.columns]

if "Dynasty" not in df.columns:
    raise KeyError("Input file must contain column 'Dynasty'.")

df["Dynasty_std"] = df["Dynasty"].apply(normalize_dynasty)

# detect symptom columns and formula herb columns
symptom_cols = [c for c in df.columns if re.match(r"^Symptom\.\d+$", str(c).strip())]
herb_cols = [c for c in df.columns if re.match(r"^Formula herb\.\d+$", str(c).strip())]

if not symptom_cols:
    raise KeyError("No columns found like 'Symptom.1', 'Symptom.2', ...")
if not herb_cols:
    raise KeyError("No columns found like 'Formula herb.1', 'Formula herb.2', ...")

print("Detected symptom columns:", symptom_cols)
print("Detected formula herb columns:", herb_cols)

# ==============================
# 6. Read pinyin mapping
# ==============================
pinyin_df = pd.read_excel(pinyin_file)
pinyin_df.columns = [str(c).strip() for c in pinyin_df.columns]

required_pinyin_cols = {"Chinese name", "Pinyin name"}
if not required_pinyin_cols.issubset(set(pinyin_df.columns)):
    raise KeyError("Pinyin mapping file must contain 'Chinese name' and 'Pinyin name'.")

pinyin_map = dict(
    zip(
        pinyin_df["Chinese name"].astype(str).str.strip(),
        pinyin_df["Pinyin name"].astype(str).str.strip()
    )
)

def map_to_pinyin(x):
    if x is None:
        return None
    return pinyin_map.get(x, x)

# ==============================
# 7. Extract all Symptom–Formula herb pairs by dynasty
# Each row:
# - deduplicate symptoms first
# - deduplicate formula herbs first
# - then Cartesian product
# - same pair in one row counted only once
# ==============================
pair_count_by_dynasty = defaultdict(lambda: defaultdict(int))
symptoms_by_dynasty = defaultdict(set)
herbs_by_dynasty = defaultdict(set)

for _, row in df.iterrows():
    dynasty = row["Dynasty_std"]
    if dynasty not in dynasty_order:
        continue

    symptoms = [clean_cell(row[c]) for c in symptom_cols]
    herbs = [clean_cell(row[c]) for c in herb_cols]

    symptoms = unique_preserve_order([x for x in symptoms if x])
    herbs = unique_preserve_order([x for x in herbs if x])

    if not symptoms or not herbs:
        continue

    symptoms_by_dynasty[dynasty].update(symptoms)
    herbs_by_dynasty[dynasty].update(herbs)

    row_pairs = set()
    for s in symptoms:
        for h in herbs:
            row_pairs.add((s, h))

    for pair in row_pairs:
        pair_count_by_dynasty[dynasty][pair] += 1

# ==============================
# 8. Calculate novel pairs/symptoms/herbs cumulatively
# ==============================
cumulative_pairs = set()
cumulative_symptoms = set()
cumulative_herbs = set()

summary_records = []
new_pair_count_by_dynasty = {}
new_pairs_by_dynasty = {}
new_symptoms_by_dynasty = {}
new_herbs_by_dynasty = {}

for d in dynasty_order:
    current_pair_count = pair_count_by_dynasty.get(d, {})
    current_pairs = set(current_pair_count.keys())
    current_symptoms = symptoms_by_dynasty.get(d, set())
    current_herbs = herbs_by_dynasty.get(d, set())

    newly_added_pairs = current_pairs - cumulative_pairs
    newly_added_pair_count = {p: current_pair_count[p] for p in newly_added_pairs}

    newly_added_symptoms = current_symptoms - cumulative_symptoms
    newly_added_herbs = current_herbs - cumulative_herbs

    new_pairs_by_dynasty[d] = newly_added_pairs
    new_pair_count_by_dynasty[d] = newly_added_pair_count
    new_symptoms_by_dynasty[d] = newly_added_symptoms
    new_herbs_by_dynasty[d] = newly_added_herbs

    summary_records.append({
        "Dynasty": d,
        "Total unique symptom-formula herb pairs in dynasty": len(current_pairs),
        "New unique symptom-formula herb pairs": len(newly_added_pairs),
        "Total pair occurrences in dynasty": sum(current_pair_count.values()),
        "Occurrences of new pairs in dynasty": sum(newly_added_pair_count.values()),
        "Total symptoms in dynasty": len(current_symptoms),
        "New symptoms": len(newly_added_symptoms),
        "Total formula herbs in dynasty": len(current_herbs),
        "New formula herbs": len(newly_added_herbs),
        "Cumulative pairs after dynasty": len(cumulative_pairs | current_pairs),
        "Cumulative symptoms after dynasty": len(cumulative_symptoms | current_symptoms),
        "Cumulative herbs after dynasty": len(cumulative_herbs | current_herbs)
    })

    cumulative_pairs |= current_pairs
    cumulative_symptoms |= current_symptoms
    cumulative_herbs |= current_herbs

summary_df = pd.DataFrame(summary_records)

# ==============================
# 9. Export summary workbook
# ==============================
summary_excel_path = os.path.join(output_dir, "Novel_Symptom_FormulaHerb_Summary.xlsx")

with pd.ExcelWriter(summary_excel_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    for d in dynasty_order:
        safe_d = safe_filename(d)
        pair_count_dict = new_pair_count_by_dynasty[d]

        top_pairs = sorted(
            pair_count_dict.items(),
            key=lambda x: (-x[1], x[0][0], x[0][1])
        )[:TOP_N]

        pair_records = []
        for (s, h), freq in top_pairs:
            pair_records.append({
                "Dynasty": d,
                "Symptom_Chinese": s,
                "FormulaHerb_Chinese": h,
                "Symptom_Pinyin": map_to_pinyin(s),
                "FormulaHerb_Pinyin": map_to_pinyin(h),
                "Frequency": freq
            })

        pair_df = pd.DataFrame(pair_records)
        if pair_df.empty:
            pair_df = pd.DataFrame(columns=[
                "Dynasty", "Symptom_Chinese", "FormulaHerb_Chinese",
                "Symptom_Pinyin", "FormulaHerb_Pinyin", "Frequency"
            ])
        pair_df.to_excel(writer, sheet_name=f"{safe_d[:26]}_top20", index=False)

        sym_list = sorted(new_symptoms_by_dynasty[d])
        sym_df = pd.DataFrame({
            "Dynasty": [d] * len(sym_list),
            "Symptom_Chinese": sym_list,
            "Symptom_Pinyin": [map_to_pinyin(x) for x in sym_list]
        })
        if sym_df.empty:
            sym_df = pd.DataFrame(columns=["Dynasty", "Symptom_Chinese", "Symptom_Pinyin"])
        sym_df.to_excel(writer, sheet_name=f"{safe_d[:27]}_sym", index=False)

        herb_list = sorted(new_herbs_by_dynasty[d])
        herb_df = pd.DataFrame({
            "Dynasty": [d] * len(herb_list),
            "FormulaHerb_Chinese": herb_list,
            "FormulaHerb_Pinyin": [map_to_pinyin(x) for x in herb_list]
        })
        if herb_df.empty:
            herb_df = pd.DataFrame(columns=["Dynasty", "FormulaHerb_Chinese", "FormulaHerb_Pinyin"])
        herb_df.to_excel(writer, sheet_name=f"{safe_d[:26]}_herb", index=False)

print(f"Summary workbook saved to: {summary_excel_path}")

# ==============================
# 10. Export Top20 table for each dynasty only
# ==============================
for d in dynasty_order:
    safe_d = safe_filename(d)
    pair_count_dict = new_pair_count_by_dynasty[d]

    top_pairs = sorted(
        pair_count_dict.items(),
        key=lambda x: (-x[1], x[0][0], x[0][1])
    )[:TOP_N]

    top_records = []
    for (s, h), freq in top_pairs:
        top_records.append({
            "Dynasty": d,
            "Symptom_Chinese": s,
            "FormulaHerb_Chinese": h,
            "Symptom_Pinyin": map_to_pinyin(s),
            "FormulaHerb_Pinyin": map_to_pinyin(h),
            "Frequency": freq
        })

    top_df = pd.DataFrame(top_records)
    if top_df.empty:
        top_df = pd.DataFrame(columns=[
            "Dynasty", "Symptom_Chinese", "FormulaHerb_Chinese",
            "Symptom_Pinyin", "FormulaHerb_Pinyin", "Frequency"
        ])

    top_df.to_excel(
        os.path.join(output_dir, "01_New_Pair_Tables", f"{safe_d}_new_pairs_top20.xlsx"),
        index=False
    )

# ==============================
# 11. Export node-edge tables for Top20 network of each dynasty
# ==============================
for d in dynasty_order:
    safe_d = safe_filename(d)
    pair_count_dict = new_pair_count_by_dynasty[d]

    top_pairs = sorted(pair_count_dict.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:TOP_N]

    node_rows = []
    edge_rows = []

    symptom_nodes = sorted({s for (s, h), freq in top_pairs})
    herb_nodes = sorted({h for (s, h), freq in top_pairs})

    for s in symptom_nodes:
        node_rows.append({
            "Node_Chinese": s,
            "Node_Pinyin": map_to_pinyin(s),
            "Type": "Symptom"
        })

    for h in herb_nodes:
        node_rows.append({
            "Node_Chinese": h,
            "Node_Pinyin": map_to_pinyin(h),
            "Type": "Formula herb"
        })

    for (s, h), freq in top_pairs:
        edge_rows.append({
            "Source_Chinese": s,
            "Source_Pinyin": map_to_pinyin(s),
            "Source_Type": "Symptom",
            "Target_Chinese": h,
            "Target_Pinyin": map_to_pinyin(h),
            "Target_Type": "Formula herb",
            "Weight": freq
        })

    node_df = pd.DataFrame(node_rows)
    edge_df = pd.DataFrame(edge_rows)

    if node_df.empty:
        node_df = pd.DataFrame(columns=["Node_Chinese", "Node_Pinyin", "Type"])
    if edge_df.empty:
        edge_df = pd.DataFrame(columns=[
            "Source_Chinese", "Source_Pinyin", "Source_Type",
            "Target_Chinese", "Target_Pinyin", "Target_Type", "Weight"
        ])

    node_df.to_excel(
        os.path.join(output_dir, "02_Node_Edge_Tables", f"{safe_d}_nodes_top20.xlsx"),
        index=False
    )
    edge_df.to_excel(
        os.path.join(output_dir, "02_Node_Edge_Tables", f"{safe_d}_edges_top20.xlsx"),
        index=False
    )

# ==============================
# 12. Plot network for each dynasty
# Top20 only
# Fixed node size
# No edge labels
# Better label display
# TIFF: 600 dpi + LZW compression
# ==============================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

for d in dynasty_order:
    safe_d = safe_filename(d)
    pair_count_dict = new_pair_count_by_dynasty[d]

    fig_path_png = os.path.join(output_dir, "03_Network_Figures", f"{safe_d}_new_pair_network_top20.png")
    fig_path_tiff = os.path.join(output_dir, "03_Network_Figures", f"{safe_d}_new_pair_network_top20.tiff")

    if len(pair_count_dict) == 0:
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        ax.text(
            0.5, 0.5,
            f"No newly introduced symptom-formula herb pairs\n{d}",
            ha="center", va="center", fontsize=16
        )
        ax.axis("off")
        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)
        plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
        try:
            plt.savefig(
                fig_path_tiff,
                dpi=600,
                bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"}
            )
        except Exception as e:
            print(f"TIFF save failed for {d}: {e}")
        plt.close()
        continue

    top_pairs = sorted(
        pair_count_dict.items(),
        key=lambda x: (-x[1], x[0][0], x[0][1])
    )[:TOP_N]

    G = nx.Graph()

    symptom_nodes = sorted({s for (s, h), freq in top_pairs})
    herb_nodes = sorted({h for (s, h), freq in top_pairs})

    for s in symptom_nodes:
        G.add_node(s, bipartite=0, label=map_to_pinyin(s), node_type="Symptom")

    for h in herb_nodes:
        G.add_node(h, bipartite=1, label=map_to_pinyin(h), node_type="Formula herb")

    for (s, h), freq in top_pairs:
        G.add_edge(s, h, weight=freq)

    # bipartite layout with larger side margins
    pos = {}
    n_left = len(symptom_nodes)
    n_right = len(herb_nodes)

    left_y = np.linspace(1, 0, max(n_left, 1))
    right_y = np.linspace(1, 0, max(n_right, 1))

    LEFT_X = 0.22
    RIGHT_X = 0.78

    for i, s in enumerate(symptom_nodes):
        pos[s] = (LEFT_X, left_y[i] if n_left > 1 else 0.5)

    for i, h in enumerate(herb_nodes):
        pos[h] = (RIGHT_X, right_y[i] if n_right > 1 else 0.5)

    # fixed node size
    node_sizes = [FIXED_NODE_SIZE for _ in G.nodes()]

    node_colors = []
    for n in G.nodes():
        if G.nodes[n]["node_type"] == "Symptom":
            node_colors.append(SYMPTOM_COLOR)
        else:
            node_colors.append(HERB_COLOR)

    edge_widths = []
    for _, _, data in G.edges(data=True):
        edge_widths.append(0.8 + data["weight"] * 0.6)

    labels = {n: wrap_text(G.nodes[n]["label"], width=18) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths, alpha=0.55, edge_color="#8A8A8A"
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes, node_color=node_colors,
        edgecolors="black", linewidths=0.6
    )

    # draw labels manually to avoid clipping and overlap
    for n, (x, y) in pos.items():
        label = labels[n]
        if G.nodes[n]["node_type"] == "Symptom":
            ax.text(
                x - 0.035, y, label,
                ha="right", va="center",
                fontsize=12,
                fontweight='bold'
            )
        else:
            ax.text(
                x + 0.035, y, label,
                ha="left", va="center",
                fontsize=12,
                fontweight='bold'
            )


    ax.text(0.02, 1.02, "Symptoms", fontsize=16, ha="left", va="bottom", transform=ax.transAxes)
    ax.text(0.98, 1.02, "Formula herbs", fontsize=16, ha="right", va="bottom", transform=ax.transAxes)
    ax.axis("off")

    plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.05)

    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")

    try:
        plt.savefig(
            fig_path_tiff,
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"}
        )
    except Exception as e:
        print(f"TIFF save failed for {d}: {e}")

    plt.close()

# ==============================
# 13. Scatter plot: new unique pairs per dynasty
# Full dynasty names + 4:3
# ==============================
scatter_df = summary_df.copy()
scatter_df["x"] = np.arange(len(scatter_df))
scatter_df["y"] = scatter_df["New unique symptom-formula herb pairs"]
sizes = scatter_df["New unique symptom-formula herb pairs"].fillna(0).values
bubble_sizes = 60 + sizes * 18

fig, ax = plt.subplots(figsize=(FIG_W_4_3, FIG_H_4_3), dpi=300)

ax.scatter(
    scatter_df["x"],
    scatter_df["y"],
    s=bubble_sizes,
    alpha=0.78,
    edgecolors="black",
    linewidths=0.8,
    color="#F28E2B"
)

for _, row in scatter_df.iterrows():
    ax.text(
        row["x"],
        row["y"],
        str(int(row["New unique symptom-formula herb pairs"])),
        ha="center",
        va="center",
        fontsize=12
    )

ax.set_xticks(scatter_df["x"])
ax.set_xticklabels(
    dynasty_order,
    rotation=30,
    ha="right",
    fontweight="bold",
    fontsize=11
)
ax.set_ylabel("New symptom–formula herb pairs (n)", fontsize=14)
ax.set_xlabel("Dynasty", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "04_Summary_Figures", "Novel_Pairs_Scatter.png"),
    dpi=300, bbox_inches="tight"
)
try:
    plt.savefig(
        os.path.join(output_dir, "04_Summary_Figures", "Novel_Pairs_Scatter.tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
except Exception as e:
    print("TIFF save failed for scatter plot:", e)
plt.close()

# ==============================
# 14. Line plot: new unique pairs per dynasty
# Full dynasty names + 4:3
# ==============================
line_df = summary_df.copy()
line_df["x"] = np.arange(len(line_df))
line_df["y"] = line_df["New unique symptom-formula herb pairs"]

fig, ax = plt.subplots(figsize=(FIG_W_4_3, FIG_H_4_3), dpi=300)

ax.plot(
    line_df["x"],
    line_df["y"],
    marker="o",
    linewidth=2.0,
    markersize=8,
    color="#4C78A8"
)

offset = max(line_df["y"].max() * 0.02, 0.3) if len(line_df) > 0 else 0.3

for _, row in line_df.iterrows():
    ax.text(
        row["x"],
        row["y"] + offset,
        str(int(row["New unique symptom-formula herb pairs"])),
        ha="center",
        va="bottom",
        fontsize=12
    )

ax.set_xticks(line_df["x"])
ax.set_xticklabels(
    dynasty_order,
    rotation=30,
    ha="right",
    fontweight="bold",
    fontsize=11
)
ax.set_ylabel("New symptom–formula herb pairs (n)", fontsize=14)
ax.set_xlabel("Dynasty", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "04_Summary_Figures", "Novel_Pairs_Line.png"),
    dpi=300, bbox_inches="tight"
)
try:
    plt.savefig(
        os.path.join(output_dir, "04_Summary_Figures", "Novel_Pairs_Line.tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
except Exception as e:
    print("TIFF save failed for line plot:", e)
plt.close()

# ==============================
# 15. Grouped bar chart: new symptoms and new formula herbs
# ==============================
bar_df = summary_df.copy()
bar_df["x"] = np.arange(len(bar_df))
width = 0.38

fig, ax = plt.subplots(figsize=(FIG_W_4_3, FIG_H_4_3), dpi=300)

ax.bar(
    bar_df["x"] - width / 2,
    bar_df["New symptoms"],
    width=width,
    label="New symptoms",
    color=SYMPTOM_COLOR,
    edgecolor="black",
    linewidth=0.6
)

ax.bar(
    bar_df["x"] + width / 2,
    bar_df["New formula herbs"],
    width=width,
    label="New formula herbs",
    color=HERB_COLOR,
    edgecolor="black",
    linewidth=0.6
)

for _, row in bar_df.iterrows():
    ax.text(
        row["x"] - width / 2,
        row["New symptoms"] + 0.3,
        str(int(row["New symptoms"])),
        ha="center", va="bottom", fontsize=12
    )
    ax.text(
        row["x"] + width / 2,
        row["New formula herbs"] + 0.3,
        str(int(row["New formula herbs"])),
        ha="center", va="bottom", fontsize=12
    )

ax.set_xticks(bar_df["x"])
ax.set_xticklabels(
    dynasty_order,
    rotation=30,
    ha="right",
    fontweight="bold",
    fontsize=10
)
ax.set_ylabel("Count (n)", fontsize=14)
ax.set_xlabel("Dynasty", fontsize=14)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "04_Summary_Figures", "New_Symptoms_New_FormulaHerbs_Bar.png"),
    dpi=300, bbox_inches="tight"
)
try:
    plt.savefig(
        os.path.join(output_dir, "04_Summary_Figures", "New_Symptoms_New_FormulaHerbs_Bar.tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
except Exception as e:
    print("TIFF save failed for bar plot:", e)
plt.close()

# ==============================
# 16. Export CSV summary
# ==============================
summary_df.to_csv(
    os.path.join(output_dir, "Novel_Symptom_FormulaHerb_Summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("All done.")
print(f"Results saved to: {output_dir}")