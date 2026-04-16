import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILE_HERB_COMP = os.path.join(PROJECT_ROOT, "Data", "Network Pharmacology", "Isolated path", "Herb_Compound.xlsx")
FILE_COMP_TGT = os.path.join(PROJECT_ROOT, "Data", "Network Pharmacology", "Isolated path", "Compound_Target.xlsx")
FILE_DISEASE = os.path.join(PROJECT_ROOT, "Data", "Network Pharmacology", "Isolated path", "Target-Thyroid cancer.xlsx")
FILE_SYM_HERB = os.path.join(PROJECT_ROOT, "Data", "Network Pharmacology", "Isolated path", "Symptom_Herb.xlsx")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Results", "Symptom-Target", "Isolated path")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "isolated_paths.xlsx")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# =========================
# 2. Global parameters
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_DPI = 600
TIFF_KWARGS = {"compression": "tiff_lzw"}

LAYER_COLOR = {
    "symptom": "#E3F2FD",
    "herb": "#E8F5E9",
    "compound": "#FFF3E0",
    "target": "#FCE4EC"
}

LAYER_BORDER = {
    "symptom": "#1565C0",
    "herb": "#2E7D32",
    "compound": "#E65100",
    "target": "#C62828"
}

LAYER_LABEL = {
    "symptom": "Symptom",
    "herb": "Herb",
    "compound": "Compound",
    "target": "Target"
}


# =========================
# 3. Utility functions
# =========================
def safe_filename(x):
    x = str(x)
    x = re.sub(r'[\\/*?:"<>|]', "_", x)
    x = x.replace("\n", "_").replace("\r", "_").strip()
    return x


def safe_w(row, col="Weight"):
    return float(row[col]) if col in row.index and pd.notna(row[col]) else 1.0


def save_tiff(fig, out_path):
    fig.savefig(
        out_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pil_kwargs=TIFF_KWARGS
    )
    plt.close(fig)


def get_isolation_style(iso):
    iso_map = {
        0: "Low isolation",
        1: "Mild isolation",
        2: "High isolation",
        3: "Fully isolated"
    }

    facecolor = "#FFF8E1" if iso < 3 else "#FFEBEE"
    edgecolor = "#FFC107" if iso < 3 else "#C62828"

    return iso_map.get(iso, "Unknown"), facecolor, edgecolor


# =========================
# 4. Read data and build network
# =========================
def load_and_build(f_sh, f_hc, f_ct, f_d):
    print("=" * 60)
    print("Step 1 | Read data and build the four-layer network")
    print("=" * 60)

    df_sh = pd.read_excel(f_sh).dropna(how="all")
    df_hc = pd.read_excel(f_hc).dropna(how="all")
    df_ct = pd.read_excel(f_ct).dropna(how="all")
    df_d = pd.read_excel(f_d).dropna(how="all")

    df_sh.columns = [str(c).strip() for c in df_sh.columns]
    df_hc.columns = [str(c).strip() for c in df_hc.columns]
    df_ct.columns = [str(c).strip() for c in df_ct.columns]
    df_d.columns = [str(c).strip() for c in df_d.columns]

    required_cols = {
        "Symptom_Herb.xlsx": (df_sh, ["Symptom", "Herb"]),
        "Herb_Compound.xlsx": (df_hc, ["Herb", "Compound"]),
        "Compound_Target.xlsx": (df_ct, ["Compound", "Target"]),
        "Target-Thyroid cancer-434.xlsx": (df_d, ["Target"]),
    }

    for file_name, (df, cols) in required_cols.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{file_name} is missing required columns: {missing}")

    disease_targets = set(df_d["Target"].dropna().astype(str).str.strip())

    G = nx.Graph()

    # Symptom-Herb
    for _, r in df_sh.iterrows():
        s = str(r["Symptom"]).strip()
        h = str(r["Herb"]).strip()
        if not s or not h or s == "nan" or h == "nan":
            continue
        G.add_node(s, layer="symptom")
        G.add_node(h, layer="herb")
        G.add_edge(s, h, edge_type="sym_herb", weight=safe_w(r))

    # Herb-Compound
    for _, r in df_hc.iterrows():
        h = str(r["Herb"]).strip()
        c = str(r["Compound"]).strip()
        if not h or not c or h == "nan" or c == "nan":
            continue
        G.add_node(h, layer="herb")
        G.add_node(c, layer="compound")
        G.add_edge(h, c, edge_type="herb_compound", weight=safe_w(r))

    # Compound-Target
    for _, r in df_ct.iterrows():
        c = str(r["Compound"]).strip()
        t = str(r["Target"]).strip()
        if not c or not t or c == "nan" or t == "nan":
            continue
        G.add_node(c, layer="compound")
        G.add_node(t, layer="target")
        G.add_edge(c, t, edge_type="compound_tgt", weight=safe_w(r))

    layer_count = defaultdict(int)
    for _, d in G.nodes(data=True):
        layer_count[d.get("layer", "?")] += 1

    print(f"Number of nodes in network: {G.number_of_nodes()}")
    print(f"Number of edges in network: {G.number_of_edges()}")
    for layer, count in sorted(layer_count.items()):
        print(f"  {layer}: {count}")
    print(f"Number of disease targets: {len(disease_targets)}")

    return G, disease_targets


# =========================
# 5. Find isolated unique paths
# =========================
def find_isolated_paths(G, disease_targets):
    print("\n" + "=" * 60)
    print("Step 2 | Identify isolated unique paths")
    print("=" * 60)

    symptoms = [n for n, d in G.nodes(data=True) if d.get("layer") == "symptom"]
    herbs = {n for n, d in G.nodes(data=True) if d.get("layer") == "herb"}
    compounds = {n for n, d in G.nodes(data=True) if d.get("layer") == "compound"}

    isolated = []

    for sym in symptoms:
        sym_herbs = [h for h in G.neighbors(sym) if h in herbs]
        all_paths = []

        for h in sym_herbs:
            herb_compounds = [c for c in G.neighbors(h) if c in compounds]
            for comp in herb_compounds:
                for tgt in G.neighbors(comp):
                    if tgt in disease_targets:
                        all_paths.append((sym, h, comp, tgt))

        if not all_paths:
            continue

        st_paths = defaultdict(list)
        for p in all_paths:
            st_paths[(p[0], p[3])].append(p)

        for (s, t), paths in st_paths.items():
            if len(paths) != 1:
                continue

            _, h, comp, _ = paths[0]

            h_syms = [n for n in G.neighbors(h) if G.nodes[n].get("layer") == "symptom"]
            c_herbs = [n for n in G.neighbors(comp) if G.nodes[n].get("layer") == "herb"]
            t_comps = [n for n in G.neighbors(t) if G.nodes[n].get("layer") == "compound"]

            h_excl = (len(h_syms) == 1 and h_syms[0] == s)
            c_excl = (len(c_herbs) == 1 and c_herbs[0] == h)
            t_excl = (len(t_comps) == 1 and t_comps[0] == comp)

            iso = int(h_excl) + int(c_excl) + int(t_excl)

            isolated.append({
                "Symptom": s,
                "Herb": h,
                "Compound": comp,
                "Target": t,
                "W_sym_herb": round(G[s][h].get("weight", 1.0), 4),
                "W_herb_compound": round(G[h][comp].get("weight", 1.0), 4),
                "W_compound_tgt": round(G[comp][t].get("weight", 1.0), 4),
                "H_exclusive_to_S": h_excl,
                "C_exclusive_to_H": c_excl,
                "T_exclusive_to_C": t_excl,
                "Isolation_Score": iso,
                "Fully_Isolated": iso == 3
            })

    df = pd.DataFrame(isolated)

    if df.empty:
        print("No isolated unique paths were identified.")
        return df

    df = df.sort_values(
        ["Fully_Isolated", "Isolation_Score", "Symptom", "Target"],
        ascending=[False, False, True, True]
    ).reset_index(drop=True)

    print(f"Total isolated unique paths: {len(df)}")
    print(f"Number of fully isolated paths: {df['Fully_Isolated'].sum()}")

    return df


# =========================
# 6. Draw single-path figure
# =========================
def draw_single_path(row, G, idx, save_dir):
    s = row["Symptom"]
    h = row["Herb"]
    comp = row["Compound"]
    t = row["Target"]

    layers = ["symptom", "herb", "compound", "target"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-2.2, 2.5)
    ax.axis("off")

    ax.text(
        1.5, 2.22,
        f"Isolated unique path #{idx + 1}: {s} → {h} → {comp} → {t}",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold", color="black"
    )

    iso = int(row["Isolation_Score"])
    iso_label, iso_facecolor, iso_edgecolor = get_isolation_style(iso)

    ax.text(
        1.5, 1.88,
        f"Isolation score: {iso}/3   {iso_label}",
        ha="center", va="center",
        fontsize=8.8,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=iso_facecolor,
            edgecolor=iso_edgecolor,
            linewidth=1.2
        )
    )

    for xi, lyr in enumerate(layers):
        ax.text(
            xi, 1.52, LAYER_LABEL[lyr],
            ha="center", va="bottom",
            fontsize=8.6, color=LAYER_BORDER[lyr], fontweight="bold"
        )
        ax.axvline(xi, color="#f8f8f8", linewidth=45, zorder=0)

    def nbrs(node, lyr, excl=None):
        return [n for n in G.neighbors(node) if G.nodes[n].get("layer") == lyr and n != excl]

    def draw_nbrs(x, nodes, direction="up", max_show=4):
        if not nodes:
            return
        show = nodes[:max_show]
        ys = np.linspace(0.45, 1.25, len(show)) if direction == "up" else np.linspace(-0.45, -1.65, len(show))
        for yi, nm in zip(ys, show):
            ax.add_patch(plt.Circle((x, yi), 0.18, color="#e8e8e8", ec="#bbbbbb", lw=0.7, zorder=2))
            label = (nm[:10] + "…") if len(nm) > 11 else nm
            ax.text(x, yi, label, ha="center", va="center", fontsize=6, color="#888888", zorder=3)
        if len(nodes) > max_show:
            extra_y = ys[-1] + 0.22 if direction == "up" else ys[-1] - 0.22
            ax.text(x, extra_y, f"+{len(nodes) - max_show} more", ha="center", fontsize=5.5, color="#aaaaaa")

    draw_nbrs(0, nbrs(s, "herb", h), "up")
    draw_nbrs(1, nbrs(h, "symptom", s), "down")
    draw_nbrs(1, nbrs(h, "compound", comp), "up")
    draw_nbrs(2, nbrs(comp, "herb", h), "down")
    draw_nbrs(2, nbrs(comp, "target", t), "up")
    draw_nbrs(3, nbrs(t, "compound", comp), "down")

    for xi, (node, lyr) in enumerate(zip([s, h, comp, t], layers)):
        ax.add_patch(
            plt.Circle((xi, 0), 0.30, color=LAYER_COLOR[lyr], ec=LAYER_BORDER[lyr], lw=2.0, zorder=4)
        )
        lbl = node if len(node) <= 12 else node[:12] + "\n" + node[12:24]
        ax.text(
            xi, 0, lbl,
            ha="center", va="center",
            fontsize=8, fontweight="bold", color=LAYER_BORDER[lyr], zorder=5
        )

    edges_info = [
        (0, 1, row["W_sym_herb"], row["H_exclusive_to_S"], "#1565C0"),
        (1, 2, row["W_herb_compound"], row["C_exclusive_to_H"], "#2E7D32"),
        (2, 3, row["W_compound_tgt"], row["T_exclusive_to_C"], "#C62828")
    ]

    for x1, x2, w, excl, col in edges_info:
        ax.annotate(
            "",
            xy=(x2 - 0.31, 0), xytext=(x1 + 0.31, 0),
            arrowprops=dict(
                arrowstyle="->",
                color=col,
                lw=2.2 if excl else 1.2,
                linestyle="solid" if excl else "dashed"
            ),
            zorder=3
        )
        ax.text(
            (x1 + x2) / 2, 0.14,
            f"w={w:.3f}",
            ha="center", fontsize=7, color=col, style="italic"
        )

    out_name = f"path_{idx + 1:03d}_{safe_filename(s)}_to_{safe_filename(t)}.tiff"
    out_path = os.path.join(save_dir, out_name)
    save_tiff(fig, out_path)
    return out_path


# =========================
# 7. Draw standalone legend figure
# =========================
def draw_legend_only(save_dir):
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.axis("off")

    layers = ["symptom", "herb", "compound", "target"]

    legend_elems = [
        mpatches.Patch(color=LAYER_BORDER[l], alpha=0.9, label=LAYER_LABEL[l])
        for l in layers
    ]
    legend_elems += [
        plt.Line2D([0], [0], color="gray", lw=1.6, ls="-", label="Exclusive"),
        plt.Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Non-exclusive")
    ]

    ax.legend(
        handles=legend_elems,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.6
    )

    out_path = os.path.join(save_dir, "legend_only.tiff")
    save_tiff(fig, out_path)
    print("Standalone legend figure has been saved.")


# =========================
# 8. Draw summary heatmap
# =========================
def draw_summary_heatmap(df, save_dir):
    if df.empty:
        return

    pivot = df.pivot_table(
        index="Symptom",
        columns="Target",
        values="Isolation_Score",
        aggfunc="max"
    ).fillna(0)

    pivot = pivot.loc[pivot.max(axis=1) > 0, pivot.max(axis=0) > 0]

    fig_w = max(6, len(pivot.columns) * 0.5)
    fig_h = max(5, len(pivot.index) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = sns.color_palette(["#f5f5f5", "#FFF9C4", "#FFB300", "#C62828"], as_cmap=True)

    sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=0,
        vmax=3,
        linewidths=0.5,
        linecolor="#e0e0e0",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 8},
        ax=ax,
        cbar_kws={"label": "Isolation score (0–3)", "shrink": 0.6}
    )

    ax.set_title("Isolated unique paths: Symptom × disease target", fontsize=11, fontweight="bold")
    ax.set_xlabel("Disease target", fontsize=10)
    ax.set_ylabel("Symptom", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)

    out_path = os.path.join(save_dir, "summary_heatmap.tiff")
    save_tiff(fig, out_path)
    print("Summary heatmap has been saved.")


# =========================
# 9. Draw network overview
# =========================
def draw_network_overview(df, G, save_dir):
    if df.empty:
        return

    highlight = df[df["Isolation_Score"] >= 2].copy()
    if highlight.empty:
        highlight = df.copy()

    sub_nodes = set()
    for _, r in highlight.iterrows():
        for n in [r["Symptom"], r["Herb"], r["Compound"], r["Target"]]:
            sub_nodes.add(n)
            sub_nodes.update(G.neighbors(n))

    H = G.subgraph(sub_nodes).copy()
    pos = nx.spring_layout(H, seed=RANDOM_SEED, k=2.5)

    path_node_set = set(highlight[["Symptom", "Herb", "Compound", "Target"]].values.flatten())

    path_edges = set()
    for _, r in highlight.iterrows():
        path_edges.update([
            (r["Symptom"], r["Herb"]),
            (r["Herb"], r["Compound"]),
            (r["Compound"], r["Target"])
        ])

    node_colors = []
    node_sizes = []
    for n in H.nodes():
        if n in path_node_set:
            node_colors.append(LAYER_BORDER.get(H.nodes[n].get("layer", ""), "#888888"))
            node_sizes.append(600)
        else:
            node_colors.append("#d0d0d0")
            node_sizes.append(120)

    edge_colors = []
    edge_widths = []
    for u, v in H.edges():
        if (u, v) in path_edges or (v, u) in path_edges:
            edge_colors.append("#E24B4A")
            edge_widths.append(2.2)
        else:
            edge_colors.append("#d8d8d8")
            edge_widths.append(0.5)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.axis("off")

    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(H, pos, edge_color=edge_colors, width=edge_widths, ax=ax, alpha=0.8)

    labels = {n: n for n in H.nodes() if n in path_node_set}
    nx.draw_networkx_labels(H, pos, labels, font_size=6, font_weight="bold", ax=ax)

    legend = [
        mpatches.Patch(color="#1565C0", label="Symptom"),
        mpatches.Patch(color="#2E7D32", label="Herb"),
        mpatches.Patch(color="#E65100", label="Compound"),
        mpatches.Patch(color="#C62828", label="Target"),
        mpatches.Patch(color="#d0d0d0", label="Other nodes")
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(
        f"Network overview of isolated paths ({len(highlight)} highlighted paths)",
        fontsize=11,
        fontweight="bold"
    )

    out_path = os.path.join(save_dir, "network_overview.tiff")
    save_tiff(fig, out_path)
    print("Global network overview has been saved.")


# =========================
# 10. Main program
# =========================
if __name__ == "__main__":
    required_files = [FILE_SYM_HERB, FILE_HERB_COMP, FILE_COMP_TGT, FILE_DISEASE]
    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        print("Error: the following files do not exist:")
        for f in missing:
            print("  ", f)
        raise FileNotFoundError("Please check the input file paths.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    try:
        G, disease_targets = load_and_build(
            FILE_SYM_HERB,
            FILE_HERB_COMP,
            FILE_COMP_TGT,
            FILE_DISEASE
        )

        df = find_isolated_paths(G, disease_targets)

        if df.empty:
            print("\nNo isolated unique paths meeting the criteria were identified.")
        else:
            df.to_excel(OUTPUT_EXCEL, index=False)
            print(f"\nResult table exported to: {OUTPUT_EXCEL}")

            print("\n" + "=" * 60)
            print("Step 3 | Visualize each path")
            print("=" * 60)

            n_draw = min(len(df), 50)
            for i, (_, row) in enumerate(df.head(n_draw).iterrows()):
                draw_single_path(row, G, i, FIGURES_DIR)
                if (i + 1) % 10 == 0 or (i + 1) == n_draw:
                    print(f"Completed {i + 1}/{n_draw} path figures")

            print("\n" + "=" * 60)
            print("Step 4 | Summary visualizations")
            print("=" * 60)

            draw_legend_only(FIGURES_DIR)
            draw_summary_heatmap(df, FIGURES_DIR)
            draw_network_overview(df, G, FIGURES_DIR)

            print("\n" + "=" * 60)
            print(f"Completed. Total isolated paths: {len(df)} | Fully isolated: {df['Fully_Isolated'].sum()}")
            print("=" * 60)

    except Exception as e:
        import traceback
        print(f"\n[Error] {e}")
        traceback.print_exc()