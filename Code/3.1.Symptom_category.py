import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(PROJECT_ROOT, "Data", "Symptom", "Symptom-Terminology List.xlsx")
output_dir = os.path.join(PROJECT_ROOT, "Results", "Symptom")
os.makedirs(output_dir, exist_ok=True)

# Sunburst chart output (SVG only)
output_svg_global = os.path.join(output_dir, "Symptom_Sunburst_GlobalPercent.svg")

# Category frequency plots (TIFF only)
output_tiff_bar = os.path.join(output_dir, "Symptom_Category_Frequency_Panels.tiff")
output_tiff_panel_a = os.path.join(output_dir, "Symptom_Category_TermCount.tiff")
output_tiff_panel_b = os.path.join(output_dir, "Symptom_Category_CumulativeFrequency.tiff")

# Table outputs
output_table = os.path.join(output_dir, "Symptom_Percentage_Table.xlsx")
output_summary_table = os.path.join(output_dir, "Symptom_Category_Summary.xlsx")

# ===============================
# 2. Read Excel file
# ===============================
df = pd.read_excel(input_file)
df.columns = df.columns.str.strip()

required_cols = ["Classification", "Symptom ID", "Frequency"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# ===============================
# 3. Data cleaning
# ===============================
plot_df = df[required_cols].copy()
plot_df = plot_df.dropna(subset=["Classification", "Symptom ID", "Frequency"])

plot_df["Classification"] = plot_df["Classification"].astype(str).str.strip()
plot_df["Symptom ID"] = plot_df["Symptom ID"].astype(str).str.strip()
plot_df["Frequency"] = pd.to_numeric(plot_df["Frequency"], errors="coerce")

plot_df = plot_df.dropna(subset=["Frequency"])
plot_df = plot_df[plot_df["Frequency"] > 0]

# Merge duplicated rows with the same Classification + Symptom ID
plot_df = (
    plot_df.groupby(["Classification", "Symptom ID"], as_index=False)["Frequency"]
    .sum()
)

if plot_df.empty:
    raise ValueError("No valid data available after cleaning.")

# ===============================
# 4. Calculate global percentage
# ===============================
total_freq = plot_df["Frequency"].sum()
plot_df["Global_Percent"] = plot_df["Frequency"] / total_freq * 100

plot_df["Label_Global"] = (
    plot_df["Symptom ID"]
    + " ("
    + plot_df["Global_Percent"].round(2).map(lambda x: f"{x:.2f}")
    + "%)"
)

# ===============================
# 5. Color palette
# ===============================
top_classes = plot_df["Classification"].drop_duplicates().tolist()

pastel_palette = [
    "#B9C7D6",
    "#E6D6D8",
    "#D8E7DB",
    "#DDD7EA",
    "#F1DFC8",
    "#F3E6C8",
    "#D8E3EE",
    "#E7DCCF",
    "#DCE8E3",
    "#EFE2E2",
]

color_map = {
    cls: pastel_palette[i % len(pastel_palette)]
    for i, cls in enumerate(top_classes)
}

# ===============================
# 6. Helper: draw and export sunburst (SVG only)
# ===============================
def draw_and_export_sunburst_svg(
    data,
    label_col,
    output_svg,
    title_text
):
    fig = px.sunburst(
        data,
        path=["Classification", label_col],
        values="Frequency",
        color="Classification",
        color_discrete_map=color_map,
    )

    hovertemplate = (
        "<b>%{label}</b><br>"
        "Frequency: %{value}<br>"
        "Percent of total: %{percentRoot:.2%}<br>"
        "Percent of parent: %{percentParent:.2%}<extra></extra>"
    )

    fig.update_traces(
        sort=False,
        branchvalues="total",
        insidetextorientation="radial",
        textinfo="label",
        textfont=dict(family="Arial Bold", color="black"),
        marker=dict(line=dict(color="white", width=2)),
        hovertemplate=hovertemplate,
        root=dict(color="white")
    )

    fig.update_layout(
        width=3200,
        height=3200,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=80, l=40, r=40, b=40),
        font=dict(family="Arial Bold", size=24, color="black"),
        uniformtext=dict(minsize=10, mode="hide"),
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=32, family="Arial Bold", color="black")
        )
    )

    fig.write_image(output_svg, format="svg", width=3200, height=3200)

# ===============================
# 7. Build summary table
# ===============================
def build_category_summary(data):
    summary_df = (
        data.groupby("Classification", as_index=False)
        .agg(
            Term_Count=("Symptom ID", "nunique"),
            Cumulative_Frequency=("Frequency", "sum")
        )
    )

    summary_df = summary_df.sort_values(
        by=["Term_Count", "Cumulative_Frequency"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return summary_df

# ===============================
# 8. Helper: export a single horizontal bar chart (TIFF only)
# ===============================
def export_single_barh_tiff(
    summary_df,
    y_col,
    x_col,
    title_text,
    xlabel,
    output_tiff
):
    categories = summary_df[y_col].tolist()
    colors = [color_map[c] for c in categories]

    n_cat = len(categories)
    fig_height = max(6, 0.7 * n_cat + 1.8)

    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    bars = ax.barh(
        summary_df[y_col],
        summary_df[x_col],
        color=colors,
        edgecolor="none"
    )

    ax.invert_yaxis()
    ax.set_title(title_text, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    max_val = summary_df[x_col].max()
    ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 1)

    for bar, val in zip(bars, summary_df[x_col]):
        label_text = f"{int(val):,}" if x_col == "Cumulative_Frequency" else f"{int(val)}"
        ax.text(
            bar.get_width() + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha="left",
            fontsize=10
        )

    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    handles = [mpatches.Patch(color=color_map[c], label=c) for c in categories]
    ncol = 2 if len(handles) <= 8 else 3

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.00),
        handlelength=1.0,
        columnspacing=1.2,
        handletextpad=0.4
    )

    plt.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.18)

    fig.savefig(
        output_tiff,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)

# ===============================
# 9. Helper: export combined panels (TIFF only)
# ===============================
def export_combined_panels_tiff(summary_df, output_tiff):
    categories = summary_df["Classification"].tolist()
    colors = [color_map[c] for c in categories]

    n_cat = len(categories)
    fig_height = max(6, 0.7 * n_cat + 2.2)

    plt.rcParams["font.family"] = "Arial"

    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height), constrained_layout=False)
    ax1, ax2 = axes

    # Panel A
    bars1 = ax1.barh(
        summary_df["Classification"],
        summary_df["Term_Count"],
        color=colors,
        edgecolor="none"
    )
    ax1.invert_yaxis()
    ax1.set_title("PANEL A — TERM COUNT PER CATEGORY", fontsize=15, fontweight="bold", pad=12)
    ax1.set_xlabel("Number of terms", fontsize=11)
    ax1.set_ylabel("")
    ax1.grid(axis="x", linestyle="-", alpha=0.2)
    ax1.set_axisbelow(True)

    max_a = summary_df["Term_Count"].max()
    ax1.set_xlim(0, max_a * 1.15 if max_a > 0 else 1)

    for bar, val in zip(bars1, summary_df["Term_Count"]):
        ax1.text(
            bar.get_width() + max_a * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(val)}",
            va="center",
            ha="left",
            fontsize=10
        )

    # Panel B
    bars2 = ax2.barh(
        summary_df["Classification"],
        summary_df["Cumulative_Frequency"],
        color=colors,
        edgecolor="none"
    )
    ax2.invert_yaxis()
    ax2.set_title("PANEL B — CUMULATIVE FREQUENCY PER CATEGORY", fontsize=15, fontweight="bold", pad=12)
    ax2.set_xlabel("Cumulative frequency (n)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("")
    ax2.grid(axis="x", linestyle="-", alpha=0.2)
    ax2.set_axisbelow(True)

    max_b = summary_df["Cumulative_Frequency"].max()
    ax2.set_xlim(0, max_b * 1.18 if max_b > 0 else 1)

    for bar, val in zip(bars2, summary_df["Cumulative_Frequency"]):
        ax2.text(
            bar.get_width() + max_b * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(val):,}",
            va="center",
            ha="left",
            fontsize=10
        )

    for ax in axes:
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")

    handles = [mpatches.Patch(color=color_map[c], label=c) for c in categories]
    ncol = 3 if len(handles) >= 6 else 2

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=1.0,
        columnspacing=1.2,
        handletextpad=0.4
    )

    plt.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.20, wspace=0.35)

    fig.savefig(
        output_tiff,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)

# ===============================
# 10. Draw the global percentage sunburst chart
# ===============================
draw_and_export_sunburst_svg(
    data=plot_df,
    label_col="Label_Global",
    output_svg=output_svg_global,
    title_text="Symptom Distribution Sunburst (Global Percentage)"
)

# ===============================
# 11. Draw category summary figures
# ===============================
summary_df = build_category_summary(plot_df)

# Combined panels
export_combined_panels_tiff(
    summary_df=summary_df,
    output_tiff=output_tiff_bar
)

# Panel A only
export_single_barh_tiff(
    summary_df=summary_df,
    y_col="Classification",
    x_col="Term_Count",
    title_text="PANEL A — TERM COUNT PER CATEGORY",
    xlabel="Number of terms",
    output_tiff=output_tiff_panel_a
)

# Panel B only
export_single_barh_tiff(
    summary_df=summary_df,
    y_col="Classification",
    x_col="Cumulative_Frequency",
    title_text="PANEL B — CUMULATIVE FREQUENCY PER CATEGORY",
    xlabel="Cumulative frequency (n)",
    output_tiff=output_tiff_panel_b
)

# ===============================
# 12. Export tables
# ===============================
export_df = plot_df[
    [
        "Classification",
        "Symptom ID",
        "Frequency",
        "Global_Percent",
        "Label_Global"
    ]
].copy()

export_df["Global_Percent"] = export_df["Global_Percent"].round(2)

export_df.to_excel(output_table, index=False)
summary_df.to_excel(output_summary_table, index=False)

# ===============================
# 13. Print results
# ===============================
print("All figures and tables were exported successfully:")
print(f"[Sunburst SVG] {output_svg_global}")
print(f"[Panels TIFF ] {output_tiff_bar}")
print(f"[Panel A TIFF] {output_tiff_panel_a}")
print(f"[Panel B TIFF] {output_tiff_panel_b}")
print(f"[Table XLSX  ] {output_table}")
print(f"[Summary XLSX] {output_summary_table}")