import os
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

base_path = os.path.join(PROJECT_ROOT, "Data", "Network Pharmacology")
output_path = os.path.join(PROJECT_ROOT, "Results", "Network Pharmacology")
os.makedirs(output_path, exist_ok=True)

dynasties = [
    ("Han", "before the end of Han Dynasty"),
    ("Wei", "Wei, Jin, Southern and Northern Dynasties"),
    ("Sui", "Sui, Tang and Five Dynasties"),
    ("Song", "Song, Jin and Yuan Dynasties"),
    ("Ming", "Ming Dynasty"),
    ("Qing", "Qing Dynasty"),
    ("Modern", "Modern Times")
]

# -------------------------------
# Load multiple sheets in one file; retain raw pathway names for display
# -------------------------------
def build_herb_pathway(prefix):
    herb_comp_path = os.path.join(base_path, "Herb-Compound.xlsx")
    comp_target_path = os.path.join(base_path, "Compound-Target.xlsx")
    target_pathway_path = os.path.join(base_path, "Target-Pathway.xlsx")

    hc = pd.read_excel(herb_comp_path, sheet_name=prefix)
    ct = pd.read_excel(comp_target_path, sheet_name=prefix)
    tp = pd.read_excel(target_pathway_path, sheet_name=prefix)

    hc.columns = ["Herb", "Compound"]
    ct.columns = ["Compound", "Target"]
    tp.columns = [c.strip().capitalize() for c in tp.columns]
    if "Targets" in tp.columns:
        tp.rename(columns={"Targets": "Target"}, inplace=True)
    if "Pathways" in tp.columns:
        tp.rename(columns={"Pathways": "Pathway"}, inplace=True)

    if not {"Target", "Pathway"}.issubset(tp.columns):
        raise ValueError(f"❌ {prefix}: mismatched column names in target_pathway Sheet: {tp.columns}")

    tp["Pathway_raw"] = tp["Pathway"]

    for df in [hc, ct, tp]:
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    hc["Compound_upper"] = hc["Compound"].str.upper()
    ct["Compound_upper"] = ct["Compound"].str.upper()
    ct["Target_upper"] = ct["Target"].str.upper()
    tp["Target_upper"] = tp["Target"].str.upper()

    herb_pathway = (
        hc.merge(ct, on="Compound_upper", how="inner")
          .merge(tp, on="Target_upper", how="inner")
    )

    herb_pathway = herb_pathway[["Herb", "Pathway_raw"]].drop_duplicates()
    herb_pathway.rename(columns={"Pathway_raw": "Pathway"}, inplace=True)

    print(f"📘 {prefix}: {len(herb_pathway)} Herb–Pathway pairs")
    return herb_pathway

# -------------------------------
# Build Herb–Pathway co-occurrence matrices by dynasty
# -------------------------------
era_matrices = {}
for prefix, name in dynasties:
    herb_pathway = build_herb_pathway(prefix)
    if herb_pathway.empty:
        print(f"⚠️ {prefix} data is empty, skipping this dynasty.")
        continue

    matrix = pd.crosstab(herb_pathway["Herb"], herb_pathway["Pathway"])
    era_matrices[name] = matrix

    out_csv = os.path.join(output_path, f"{prefix}_Herb_Pathway_CoMatrix.csv")
    matrix.to_csv(out_csv, encoding="utf-8-sig")
    print(f"✅ {prefix} co-occurrence matrix exported to {out_csv}")

if not era_matrices:
    raise ValueError("❌ All dynasty data are empty. Please check input file paths or sheet names.")

# -------------------------------
# Construct Pathway × Dynasty matrix (number of herbs per pathway)
# -------------------------------
pathway_union = sorted(set().union(*[m.columns for m in era_matrices.values()]))
era_names = [n for _, n in dynasties]

era_matrix = pd.DataFrame(index=pathway_union, columns=era_names).fillna(0)
n_per_era = {}

for name in era_matrix.columns:
    if name not in era_matrices:
        continue
    mat = era_matrices[name]
    era_matrix[name] = mat.sum(axis=0).reindex(era_matrix.index).fillna(0)
    n_per_era[name] = int(mat.shape[0])

era_matrix = era_matrix.infer_objects(copy=False)
n_per_era = pd.Series(n_per_era).reindex(era_matrix.columns).fillna(0).astype(int)
print(f"📊 Total pathways: {len(era_matrix)}; unique herbs per dynasty:\n{n_per_era}")

# -------------------------------
# Similarity matrices
# -------------------------------
jaccard_sim = pd.DataFrame(index=era_matrix.columns, columns=era_matrix.columns, dtype=float)
for e1 in era_matrix.columns:
    for e2 in era_matrix.columns:
        vec1 = (era_matrix[e1] > 0).astype(int)
        vec2 = (era_matrix[e2] > 0).astype(int)
        jaccard_sim.loc[e1, e2] = jaccard_score(vec1, vec2, zero_division=0)

cos_sim = pd.DataFrame(
    cosine_similarity(era_matrix.T),
    index=era_matrix.columns,
    columns=era_matrix.columns
)

# -------------------------------
# ① Normalize by proportion
# -------------------------------
era_prev = era_matrix.copy().astype(float)
for era in era_prev.columns:
    denom = n_per_era[era]
    era_prev[era] = (era_prev[era] / denom) if denom > 0 else 0.0

# -------------------------------
# ② Enrichment + significance test (Fisher’s exact)
# -------------------------------
N = int(n_per_era.sum())
K_per_path = era_matrix.sum(axis=1).astype(int)
records = []
eps = 1e-9

for era in era_matrix.columns:
    n = int(n_per_era[era])
    for path in era_matrix.index:
        k = int(era_matrix.at[path, era])
        K = int(K_per_path[path])
        table = [[k, n - k],
                 [K - k, (N - n) - (K - k)]]
        _, p = fisher_exact(table, alternative='two-sided')
        rate_era = k / n if n > 0 else 0
        rate_bg = (K - k) / (N - n) if (N - n) > 0 else 0
        l2fc = np.log2((rate_era + eps) / (rate_bg + eps))
        records.append((path, era, k, n, rate_era, rate_bg, l2fc, p))

enrich_df = pd.DataFrame(records, columns=[
    "Pathway", "Era", "k", "n", "rate_era", "rate_bg", "log2FC", "p"
])
enrich_df["FDR"] = multipletests(enrich_df["p"].values, method="fdr_bh")[1]

# -------------------------------
# ③ Temporal trend modeling: GLM binomial regression
# -------------------------------
era_index_map = {era: i for i, era in enumerate(era_matrix.columns)}
trend_rows = []

for path in era_matrix.index:
    y = np.array([era_matrix.at[path, e] for e in era_matrix.columns], dtype=float)
    n = np.array([n_per_era[e] for e in era_matrix.columns], dtype=float)
    X = np.array([era_index_map[e] for e in era_matrix.columns], dtype=float)
    X = sm.add_constant(X)
    model = sm.GLM(y / np.where(n == 0, 1, n), X,
                   family=sm.families.Binomial(),
                   freq_weights=n)
    res = model.fit()
    beta = res.params[1]
    conf_int = res.conf_int()
    if isinstance(conf_int, np.ndarray):
        lcl, ucl = conf_int[1]
    else:
        lcl, ucl = conf_int.iloc[1].tolist()
    p_beta = res.pvalues[1]
    trend_rows.append((path, beta, lcl, ucl, p_beta))

trend_df = pd.DataFrame(trend_rows, columns=["Pathway", "beta", "beta_lcl", "beta_ucl", "p_beta"])
trend_df["FDR_beta"] = multipletests(trend_df["p_beta"].values, method="fdr_bh")[1]

# -------------------------------
# Plotting functions
# -------------------------------
def draw_heatmap(df, title, fname, y_font=8.5):
    fig_height = len(df) * 0.20
    fig_height = max(8, min(fig_height, 60))
    plt.figure(figsize=(10, fig_height))
    ax = sns.heatmap(
        df,
        cmap="crest",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={'label': 'Pathway Count' if "Normalized" not in title else 'Pathway Prevalence'}
    )
    ax.set_title(title, fontsize=16, pad=18, weight='bold')
    ax.set_xlabel("Dynasty", fontsize=13, fontweight='bold')
    ax.set_ylabel("Pathway", fontsize=13, fontweight='bold')
    ax.set_yticks(np.arange(len(df.index)) + 0.5)
    ax.set_yticklabels(df.index, fontsize=y_font, va="center")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.tick_params(axis="y", length=0)
    plt.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.08)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Normalized pathway prevalence' if "Normalized" not in title else 'Prevalence (k / n)', fontsize=11)
    vmin = float(np.nanmin(df.values))
    vmax = float(np.nanmax(df.values))
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fname), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.show()

def draw_volcano_for_era(era_name, df_enrich, fname):
    sub = df_enrich[df_enrich["Era"] == era_name].copy()
    sub["neglog10FDR"] = -np.log10(np.clip(sub["FDR"], 1e-300, 1))
    plt.figure(figsize=(7.5, 6))
    ax = sns.scatterplot(
        data=sub, x="log2FC", y="neglog10FDR",
        hue=sub["FDR"] < 0.05, palette={True: "crimson", False: "gray"}, legend=False, s=18
    )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Enrichment Volcano — {era_name}", fontsize=14, weight='bold')
    ax.set_xlabel(r"log$_2$(rate$_{era}$/rate$_{bg}$)")
    ax.set_ylabel(r"$-\log_{10}$(FDR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fname), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.show()

def draw_trend_bars(trend_df, fname):
    sig = trend_df[trend_df["FDR_beta"] < 0.05].copy()
    up = sig.sort_values("beta", ascending=False).head(10)
    down = sig.sort_values("beta", ascending=True).head(10)
    plt.figure(figsize=(8.5, 6))
    sns.barplot(y="Pathway", x="beta", data=up, color="tab:blue")
    plt.title("Top Increasing Pathways (β > 0, FDR < 0.05)", fontsize=13, weight='bold')
    plt.xlabel("Temporal Trend (β)")
    plt.ylabel("Pathway")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fname.replace(".tiff", "_up.tiff")), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    plt.figure(figsize=(8.5, 6))
    sns.barplot(y="Pathway", x="beta", data=down, color="tab:orange")
    plt.title("Top Decreasing Pathways (β < 0, FDR < 0.05)", fontsize=13, weight='bold')
    plt.xlabel("Temporal Trend (β)")
    plt.ylabel("Pathway")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fname.replace(".png", "_down.png")), dpi=450, bbox_inches="tight")
    plt.show()

# -------------------------------
# Visualization: Heatmap + Enrichment + Trend
# -------------------------------
draw_heatmap(era_matrix, "All Pathways Across Dynasties (Raw Counts)", "All_Pathway_Heatmap_raw.tiff")
draw_heatmap(era_prev, "All Pathways Across Dynasties (Normalized by Herb Count)", "All_Pathway_Heatmap_normalized.tiff")

for era in era_matrix.columns:
    draw_volcano_for_era(era, enrich_df, f"Volcano_{era}.png")

draw_trend_bars(trend_df, "Temporal_Trends.tiff")

# >>> Insert: Significant temporal trend pathway filtering and heatmap output <<<
trend_cutoff = 0.05
trend_sig = trend_df[trend_df["FDR_beta"] < trend_cutoff]
sig_paths = trend_sig["Pathway"].unique()
era_prev_trend = era_prev.loc[era_prev.index.isin(sig_paths)]

print(f"📈 A total of {len(sig_paths)} pathways show significant temporal trends (FDR_beta < {trend_cutoff})")

top_n = 30
trend_sig_sorted = trend_sig.reindex(
    trend_sig["beta"].abs().sort_values(ascending=False).index
).drop_duplicates(subset=["Pathway"], keep="first")
top_paths = trend_sig_sorted["Pathway"].head(top_n)
era_prev_toptrend = era_prev_trend.loc[top_paths]

draw_heatmap(
    era_prev_trend,
    f"Pathways with Significant Temporal Trends (FDR_beta < {trend_cutoff})",
    "All_Pathway_Heatmap_temporal_trend_all.tiff",
    y_font=8.5
)

draw_heatmap(
    era_prev_toptrend,
    f"Top {top_n} Pathways with Strong Temporal Trends (|β| sorted)",
    "All_Pathway_Heatmap_temporal_trend_top.tiff",
    y_font=9
)

trend_sig_sorted.to_csv(
    os.path.join(output_path, "Significant_Temporal_Trend_Pathways.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"✅ Significant temporal trend pathway heatmaps and result tables exported:\n"
      f"   - All_Pathway_Heatmap_temporal_trend_all.tiff\n"
      f"   - All_Pathway_Heatmap_temporal_trend_top.tiff\n"
      f"   - Significant_Temporal_Trend_Pathways.csv\n")

# -------------------------------
# Write results to Excel
# -------------------------------
output_xlsx = os.path.join(output_path, "Pathway_Temporal_Similarity_AllPathways_Analyzed.xlsx")
with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
    era_matrix.to_excel(writer, sheet_name="Raw_Frequency")
    era_prev.to_excel(writer, sheet_name="Normalized_Frequency")
    cos_sim.to_excel(writer, sheet_name="Cosine_Similarity")
    jaccard_sim.to_excel(writer, sheet_name="Jaccard_Similarity")
    enrich_df.to_excel(writer, sheet_name="Enrichment_log2FC_FDR", index=False)
    trend_df.to_excel(writer, sheet_name="Temporal_Trends_GLM", index=False)

print(f"\n✅ Analysis complete. Results saved to: {output_xlsx}\n📁 Image output directory: {output_path}")