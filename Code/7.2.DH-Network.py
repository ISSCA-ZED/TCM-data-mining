import os
import re
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.spatial import ConvexHull
import community as community_louvain
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tuple_base_path = os.path.join(PROJECT_ROOT, "Results", "Entity network", "Tuples")
cluster_output_path = os.path.join(PROJECT_ROOT, "Results", "Entity network", "Cluster-DH")
detail_output_dir = os.path.join(cluster_output_path, "Details")
top10_output_dir = os.path.join(cluster_output_path, "Top10")
os.makedirs(detail_output_dir, exist_ok=True)
os.makedirs(top10_output_dir, exist_ok=True)

symptom_term_path = os.path.join(PROJECT_ROOT, "Data", "Symptom", "Symptom-Terminology List.xlsx")
herb_term_path = os.path.join(PROJECT_ROOT, "Data", "Herb", "Herb-Terminology List.xlsx")
USE_ID_LABEL = True

dynasty_names = {
    "Han": "before the end of Han Dynasty",
    "Wei": "Wei, Jin, Southern and Northern Dynasties",
    "Sui": "Sui, Tang and Five Dynasties",
    "Song": "Song, Jin and Yuan Dynasties",
    "Ming": "Ming Dynasty",
    "Qing": "Qing Dynasty",
    "Modern": "Modern Times",
}

# ============================================================
# Font settings
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Parameter settings: control cluster separation
# ============================================================
GLOBAL_K = 2.8
GLOBAL_ITER = 300
LOCAL_K = 0.55
LOCAL_ITER = 250
CLUSTER_SCALE = 0.42
MIN_CENTER_DIST = 1.25
HULL_EXPAND = 1.08

LABEL_FORCE_TEXT = 0.18
LABEL_FORCE_POINTS = 0.12
LABEL_FORCE_PULL = 0.20
LABEL_MAX_MOVE = 18
LABEL_LINE_THRESHOLD = 0.10

# ============================================================
# Column name resolution function
# ============================================================
_HYPHENS = r"\s|\-|_|–|—|﹣|－"

def _norm(s: str) -> str:
    return re.sub(_HYPHENS, "", str(s)).lower()

def resolve_col(df: pd.DataFrame, aliases, contains=None, must=True) -> str:
    norm_map = {_norm(c): c for c in df.columns}
    for a in aliases:
        key = _norm(a)
        if key in norm_map:
            return norm_map[key]
    if contains:
        key_sub = _norm(contains)
        for k, v in norm_map.items():
            if key_sub in k:
                return v
    if must:
        raise KeyError(
            f"Cannot find {aliases} in columns (or a column containing {contains}); "
            f"available columns: {list(df.columns)}"
        )
    return ""

# ============================================================
# Read terminology table: Pinyin name -> ID
# Preserved according to the attached code structure
# ============================================================
def load_pinyin_id_map(file_path, id_aliases):
    if not os.path.exists(file_path):
        print(f"Warning: terminology file does not exist, skipping ID mapping: {file_path}")
        return {}

    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    pinyin_col = resolve_col(df, ["Pinyin name", "Pinyin", "Pinyin Name"], contains="pinyin")
    id_col = resolve_col(df, id_aliases)

    df = df[[pinyin_col, id_col]].copy()
    df[pinyin_col] = df[pinyin_col].astype(str).str.strip()
    df[id_col] = df[id_col].astype(str).str.strip()

    df = df[
        df[pinyin_col].notna() & (df[pinyin_col] != "") &
        df[id_col].notna() & (df[id_col] != "")
    ].drop_duplicates(subset=[pinyin_col], keep="first")

    return dict(zip(df[pinyin_col], df[id_col]))

# ============================================================
# Standardize entity types
# ============================================================
def normalize_type(x):
    s = str(x).strip().lower()
    if s in ["disease name", "disease"]:
        return "Disease"
    elif s in [
        "formula", "herb", "formula herb", "formula_herb",
        "prescription herb"
    ]:
        return "Herb"
    else:
        return str(x).strip()

# ============================================================
# Convert partition to labels (for robustness analysis)
# ============================================================
def partition_to_labels(partition_dict, node_order):
    return [partition_dict[n] for n in node_order]

# ============================================================
# Build cluster-level graph
# ============================================================
def build_cluster_graph(G, partition):
    CG = nx.Graph()

    communities = sorted(set(partition.values()))
    for c in communities:
        members = [n for n, cc in partition.items() if cc == c]
        subg = G.subgraph(members)
        internal_weight = sum(d.get("weight", 1.0) for _, _, d in subg.edges(data=True))
        CG.add_node(c, size=len(members), internal_weight=internal_weight)

    inter_weights = {}
    for u, v, d in G.edges(data=True):
        cu, cv = partition[u], partition[v]
        if cu == cv:
            continue
        key = tuple(sorted((cu, cv)))
        inter_weights[key] = inter_weights.get(key, 0.0) + float(d.get("weight", 1.0))

    for (cu, cv), w in inter_weights.items():
        CG.add_edge(cu, cv, weight=w)

    return CG

# ============================================================
# Repel cluster centers
# ============================================================
def repel_cluster_centers(center_pos, min_dist=1.2, n_iter=200, step=0.02):
    keys = list(center_pos.keys())
    pos = {k: np.array(v, dtype=float) for k, v in center_pos.items()}

    for _ in range(n_iter):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                diff = pos[b] - pos[a]
                dist = np.linalg.norm(diff)

                if dist < 1e-9:
                    diff = np.random.randn(2) * 1e-3
                    dist = np.linalg.norm(diff)

                if dist < min_dist:
                    direction = diff / dist
                    delta = (min_dist - dist) * step
                    pos[a] -= direction * delta
                    pos[b] += direction * delta
                    moved = True
        if not moved:
            break
    return pos

# ============================================================
# Expand convex hull
# ============================================================
def expand_polygon(points, scale=1.05):
    centroid = points.mean(axis=0)
    return centroid + (points - centroid) * scale

# ============================================================
# Two-stage layout
# ============================================================
def two_stage_cluster_layout(G, partition,
                             global_k=2.8,
                             global_iter=300,
                             local_k=0.55,
                             local_iter=250,
                             cluster_scale=0.42,
                             min_center_dist=1.2,
                             seed=42):
    communities = sorted(set(partition.values()))
    cluster_members = {
        c: [n for n in G.nodes() if partition[n] == c]
        for c in communities
    }

    CG = build_cluster_graph(G, partition)

    if len(CG.nodes()) == 1:
        c = communities[0]
        local_sub = G.subgraph(cluster_members[c]).copy()
        local_pos = nx.spring_layout(
            local_sub, weight="weight", k=local_k, iterations=local_iter, seed=seed
        )
        return local_pos, {c: np.array([0.0, 0.0])}

    cluster_centers = nx.spring_layout(
        CG,
        weight="weight",
        k=global_k,
        iterations=global_iter,
        seed=seed
    )
    cluster_centers = {
        c: np.array(cluster_centers[c], dtype=float)
        for c in cluster_centers
    }

    cluster_centers = repel_cluster_centers(
        cluster_centers,
        min_dist=min_center_dist,
        n_iter=300,
        step=0.06
    )

    pos = {}
    for c in communities:
        members = cluster_members[c]
        subg = G.subgraph(members).copy()

        if len(members) == 1:
            pos[members[0]] = cluster_centers[c]
            continue

        local_pos = nx.spring_layout(
            subg,
            weight="weight",
            k=local_k,
            iterations=local_iter,
            seed=seed
        )

        local_xy = np.array([local_pos[n] for n in members], dtype=float)
        local_xy = local_xy - local_xy.mean(axis=0, keepdims=True)
        max_norm = np.max(np.linalg.norm(local_xy, axis=1))
        if max_norm > 0:
            local_xy = local_xy / max_norm

        radius = cluster_scale * (0.85 + 0.18 * np.log1p(len(members)))
        local_xy = local_xy * radius

        for i, n in enumerate(members):
            pos[n] = cluster_centers[c] + local_xy[i]

    return pos, cluster_centers

# ============================================================
# Load ID mappings
# Notes:
# - In the current DH network, only herb_pinyin_to_id is actually used
# - symptom_pinyin_to_id is retained to keep the same structure as the attached code
# ============================================================
symptom_pinyin_to_id = {}
herb_pinyin_to_id = {}

if USE_ID_LABEL:
    symptom_pinyin_to_id = load_pinyin_id_map(
        symptom_term_path,
        ["Symptom ID", "SymptomID", "Symptom_Id"]
    )
    herb_pinyin_to_id = load_pinyin_id_map(
        herb_term_path,
        ["Herb ID", "HerbID", "Herb_Id"]
    )

# ============================================================
# Load all dynasty files
# ============================================================
available_dynasties = []
for short, full in dynasty_names.items():
    edge_path = os.path.join(tuple_base_path, f"Disease-Formula_herb_{short}_edges.xlsx")
    node_path = os.path.join(tuple_base_path, f"Disease-Formula_herb_{short}_nodes.xlsx")
    if os.path.exists(edge_path) and os.path.exists(node_path):
        available_dynasties.append((short, edge_path, node_path))

if not available_dynasties:
    raise FileNotFoundError(
        "No dynasty-specific Disease-Formula_herb nodes/edges files were found. "
        "Please check the paths and file names."
    )

# ============================================================
# Summary tables
# ============================================================
network_summary_all = []
sensitivity_all = []
top10_all = []

# ============================================================
# Main loop
# ============================================================
for short, edge_file, node_file in available_dynasties:
    dynasty = dynasty_names[short]
    print(f"\n===== Processing: {dynasty} =====")

    edges = pd.read_excel(edge_file)
    nodes = pd.read_excel(node_file)

    # Resolve column names
    entity_col = resolve_col(nodes, ["Entity"], contains="entity")
    type_col = resolve_col(nodes, ["Type"], contains="type")
    tfidf_col = resolve_col(nodes, ["TF-IDF", "TFIDF", "TF_IDF"], contains="tfidf")

    p1_col = resolve_col(edges, ["Pinyin1", "Source", "Entity1"], contains="pinyin1")
    p2_col = resolve_col(edges, ["Pinyin2", "Target", "Entity2"], contains="pinyin2")
    w_col = resolve_col(edges, ["Weight"], contains="weight")

    # ============================================================
    # Standardize Type column
    # ============================================================
    nodes[type_col] = nodes[type_col].apply(normalize_type)

    # ============================================================
    # Build mapping dictionaries
    # Here it is assumed that nodes[Entity] and edges[Pinyin1/Pinyin2]
    # use the same naming system
    # Following the attached code, Entity is directly treated as the key
    # for Pinyin-to-ID mapping
    # ============================================================
    entity_to_type = {r[entity_col]: r[type_col] for _, r in nodes.iterrows()}
    entity_to_tfidf = {r[entity_col]: r[tfidf_col] for _, r in nodes.iterrows()}
    entity_to_pinyin = {
        r[entity_col]: str(r[entity_col]).strip() if pd.notna(r[entity_col]) else ""
        for _, r in nodes.iterrows()
    }

    print("Node type distribution:")
    print(nodes[type_col].value_counts())

    # ============================================================
    # Build network
    # ============================================================
    G = nx.Graph()
    for _, r in nodes.iterrows():
        entity = str(r[entity_col]).strip()
        G.add_node(
            entity,
            Type=r[type_col],
            TFIDF=r[tfidf_col],
            PinyinName=entity
        )

    for _, r in edges.iterrows():
        n1 = r[p1_col]
        n2 = r[p2_col]
        w = r[w_col]

        if pd.isna(n1) or pd.isna(n2):
            continue
        if pd.isna(w):
            w = 0

        G.add_edge(str(n1).strip(), str(n2).strip(), weight=float(w))

    if len(G.nodes) < 3:
        print("Too few nodes. Skipped.")
        continue

    # ============================================================
    # Louvain community detection
    # ============================================================
    gamma_default = 1.0
    partition = community_louvain.best_partition(
        G, weight="weight", resolution=gamma_default, random_state=42
    )

    feature_df = pd.DataFrame({
        "Entity": list(partition.keys()),
        "Cluster": list(partition.values())
    })
    feature_df["Type"] = feature_df["Entity"].map(entity_to_type)
    feature_df["TFIDF"] = feature_df["Entity"].map(entity_to_tfidf)
    feature_df["Pinyin_name"] = feature_df["Entity"].map(entity_to_pinyin)

    # ============================================================
    # Generate display labels following the attached code structure
    # Disease keeps the original name
    # Herb is mapped to Herb ID
    # ============================================================
    def map_display_label(entity, entity_type, pinyin_name):
        pinyin_name = str(pinyin_name).strip()
        if not USE_ID_LABEL:
            return entity

        if entity_type == "Herb":
            return herb_pinyin_to_id.get(pinyin_name, entity)
        else:
            return entity

    feature_df["Display_Label"] = feature_df.apply(
        lambda r: map_display_label(r["Entity"], r["Type"], r["Pinyin_name"]),
        axis=1
    )

    print(
        "Herb ID mapped:",
        ((feature_df["Type"] == "Herb") &
         (feature_df["Display_Label"] != feature_df["Entity"])).sum()
    )

    n_clusters = len(set(partition.values()))
    print(f"Detected {n_clusters} structural communities")

    # ============================================================
    # Overall network metrics
    # ============================================================
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = np.mean([d for _, d in G.degree()]) if n_nodes > 0 else np.nan

    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc_size = len(largest_cc)
    largest_cc_ratio = largest_cc_size / n_nodes if n_nodes > 0 else np.nan

    modularity_q = community_louvain.modularity(partition, G, weight="weight")

    print(
        f"Nodes={n_nodes}, Edges={n_edges}, Density={density:.4f}, "
        f"Communities={n_clusters}, Modularity={modularity_q:.4f}"
    )

    network_summary_all.append({
        "Dynasty_Short": short,
        "Dynasty": dynasty,
        "Nodes": n_nodes,
        "Edges": n_edges,
        "Density": density,
        "Average_Degree": avg_degree,
        "Communities": n_clusters,
        "Modularity_Q": modularity_q,
        "Largest_CC_Size": largest_cc_size,
        "Largest_CC_Ratio": largest_cc_ratio
    })

    # ============================================================
    # Gamma sensitivity analysis
    # ============================================================
    gamma_list = [0.5, 0.75, 1.0, 1.5, 2.0]
    node_order = list(G.nodes())
    baseline_labels = partition_to_labels(partition, node_order)

    for gamma in gamma_list:
        part_gamma = community_louvain.best_partition(
            G, weight="weight", resolution=gamma, random_state=42
        )
        labels_gamma = partition_to_labels(part_gamma, node_order)

        k_gamma = len(set(labels_gamma))
        q_gamma = community_louvain.modularity(part_gamma, G, weight="weight")
        ari = adjusted_rand_score(baseline_labels, labels_gamma)
        nmi = normalized_mutual_info_score(baseline_labels, labels_gamma)

        sensitivity_all.append({
            "Dynasty_Short": short,
            "Dynasty": dynasty,
            "Gamma": gamma,
            "Communities": k_gamma,
            "Modularity_Q": q_gamma,
            "ARI_vs_gamma1.0": ari,
            "NMI_vs_gamma1.0": nmi
        })

    # ============================================================
    # Node centrality features
    # ============================================================
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clo = nx.closeness_centrality(G)

    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight") if len(G) > 10 else {n: 0 for n in G.nodes}
    except Exception:
        eig = {n: 0 for n in G.nodes}

    strength = dict(G.degree(weight="weight"))

    feature_df["Degree"] = feature_df["Entity"].map(deg)
    feature_df["Betweenness"] = feature_df["Entity"].map(bet)
    feature_df["Closeness"] = feature_df["Entity"].map(clo)
    feature_df["Eigenvector"] = feature_df["Entity"].map(eig)
    feature_df["Strength"] = feature_df["Entity"].map(strength)

    # ============================================================
    # Two-stage layout
    # ============================================================
    pos, cluster_centers = two_stage_cluster_layout(
        G,
        partition,
        global_k=GLOBAL_K,
        global_iter=GLOBAL_ITER,
        local_k=LOCAL_K,
        local_iter=LOCAL_ITER,
        cluster_scale=CLUSTER_SCALE,
        min_center_dist=MIN_CENTER_DIST,
        seed=42
    )

    feature_df["X"] = feature_df["Entity"].map(lambda x: pos[x][0])
    feature_df["Y"] = feature_df["Entity"].map(lambda x: pos[x][1])

    # ============================================================
    # Plotting
    # ============================================================
    plt.figure(figsize=(14, 11))
    colors = plt.cm.get_cmap("tab20", n_clusters)
    texts = []
    text_meta = []

    # Draw edges first
    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        plt.plot(
            [x1, x2], [y1, y2],
            color="lightgray",
            linewidth=0.35,
            alpha=0.32,
            zorder=1
        )

    # Nodes + convex hull + labels
    for c in range(n_clusters):
        sub = feature_df[feature_df["Cluster"] == c].copy()
        cluster_color = colors(c)

        if len(sub) >= 3:
            points = sub[["X", "Y"]].values
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = expand_polygon(hull_points, scale=HULL_EXPAND)

                plt.fill(
                    hull_points[:, 0], hull_points[:, 1],
                    color=cluster_color, alpha=0.10, zorder=0
                )
                plt.plot(
                    np.r_[hull_points[:, 0], hull_points[0, 0]],
                    np.r_[hull_points[:, 1], hull_points[0, 1]],
                    color=cluster_color, linewidth=1.0, alpha=0.35, zorder=0
                )
            except Exception:
                pass

        # Disease nodes: circles
        dis = sub[sub["Type"] == "Disease"]
        if len(dis) > 0:
            plt.scatter(
                dis["X"], dis["Y"],
                s=120,
                color=cluster_color,
                marker="o",
                edgecolor="black",
                linewidth=0.7,
                alpha=0.88,
                label=f"Cluster {c}",
                zorder=3
            )

        # Herb nodes: triangles
        herb = sub[sub["Type"] == "Herb"]
        if len(herb) > 0:
            plt.scatter(
                herb["X"], herb["Y"],
                s=120,
                color=cluster_color,
                marker="^",
                edgecolor="dimgray",
                linewidth=0.7,
                alpha=0.82,
                zorder=3
            )

        # Fallback for other node types
        other = sub[~sub["Type"].isin(["Disease", "Herb"])]
        if len(other) > 0:
            plt.scatter(
                other["X"], other["Y"],
                s=110,
                color=cluster_color,
                marker="s",
                edgecolor="gray",
                linewidth=0.7,
                alpha=0.78,
                zorder=3
            )

        # Labels
        for _, row in sub.iterrows():
            txt = plt.text(
                row["X"], row["Y"],
                row["Display_Label"],
                fontsize=8.0,
                zorder=4
            )
            texts.append(txt)
            text_meta.append((txt, row["X"], row["Y"]))

    # Adjust labels: weaker repulsion, keep them closer to nodes
    adjust_text(
        texts,
        expand_text=(1.01, 1.03),
        expand_points=(1.01, 1.03),
        force_text=LABEL_FORCE_TEXT,
        force_points=LABEL_FORCE_POINTS,
        force_pull=LABEL_FORCE_PULL,
        max_move=LABEL_MAX_MOVE
    )

    # Draw lines only for labels that move far away
    for txt, x0, y0 in text_meta:
        x1, y1 = txt.get_position()
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if dist > LABEL_LINE_THRESHOLD:
            plt.plot(
                [x0, x1], [y0, y1],
                color="gray",
                lw=0.30,
                alpha=0.35,
                zorder=2
            )

    plt.xlabel("Spring X", fontsize=16, fontweight='bold')
    plt.ylabel("Spring Y", fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc="best", frameon=False)
    plt.tight_layout()

    # ============================================================
    # Save high-resolution figures
    # ============================================================
    fig_base = os.path.join(cluster_output_path, f"{short}_Louvain_Spring_DH_clusters_separated")
    plt.savefig(fig_base + ".png", dpi=400, bbox_inches="tight")
    plt.savefig(fig_base + ".tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close()

    # ============================================================
    # Export cluster summary and details
    # ============================================================
    summary_list = []
    for c in range(n_clusters):
        sub = feature_df[feature_df["Cluster"] == c].copy()
        diseases = sub[sub["Type"] == "Disease"]["Entity"].tolist()
        herbs = sub[sub["Type"] == "Herb"]["Entity"].tolist()

        summary_list.append({
            "Cluster": c,
            "Disease_Count": len(diseases),
            "Herb_Count": len(herbs),
            "Top_Diseases": "、".join(diseases[:5]),
            "Top_Herbs": "、".join(herbs[:5])
        })

        detail_path = os.path.join(detail_output_dir, f"{short}_Cluster_{c}_Louvain_Spring_DH.xlsx")
        sub[[
            "Entity", "Pinyin_name", "Display_Label", "Type", "TFIDF", "Cluster",
            "Degree", "Betweenness", "Closeness", "Eigenvector", "Strength",
            "X", "Y"
        ]].to_excel(detail_path, index=False)

    pd.DataFrame(summary_list).to_excel(
        os.path.join(cluster_output_path, f"{short}_Louvain_Spring_DH_cluster_summary.xlsx"),
        index=False
    )

    # ============================================================
    # Export Top 10
    # ============================================================
    top10_df = feature_df.sort_values(
        by=["Degree", "Betweenness", "Closeness", "Eigenvector", "Strength"],
        ascending=False
    ).head(10).copy()

    top10_df.insert(0, "Rank", range(1, len(top10_df) + 1))
    top10_df.insert(0, "Dynasty", dynasty)

    top10_df[[
        "Dynasty", "Rank", "Entity", "Pinyin_name", "Display_Label", "Type", "TFIDF",
        "Degree", "Betweenness", "Closeness", "Eigenvector", "Strength", "Cluster"
    ]].to_excel(
        os.path.join(top10_output_dir, f"{short}_Top10_Centrality_DH.xlsx"),
        index=False
    )

    top10_all.append(top10_df[[
        "Dynasty", "Rank", "Entity", "Pinyin_name", "Display_Label", "Type", "TFIDF",
        "Degree", "Betweenness", "Closeness", "Eigenvector", "Strength", "Cluster"
    ]])

# ============================================================
# Export overall network metrics
# ============================================================
network_summary_df = pd.DataFrame(network_summary_all)
network_summary_df.to_excel(
    os.path.join(cluster_output_path, "Supplementary_Table_DH_Network_Summary.xlsx"),
    index=False
)

# ============================================================
# Export gamma sensitivity analysis
# ============================================================
sensitivity_df = pd.DataFrame(sensitivity_all)
sensitivity_df.to_excel(
    os.path.join(cluster_output_path, "Supplementary_Fig_DH_Sensitivity_Analysis.xlsx"),
    index=False
)

# ============================================================
# Plot gamma sensitivity analysis
# Output simultaneously:
# 1) A combined 4-panel figure
# 2) Four separate single-panel figures
# ============================================================
sens_df = pd.DataFrame(sensitivity_all)

if not sens_df.empty:
    # Dynasty order
    dynasty_order = [
        "before the end of Han Dynasty",
        "Wei, Jin, Northern and Southern Dynasties",
        "Sui, Tang and Five Dynasties",
        "Song, Jin and Yuan Dynasties",
        "Ming Dynasty",
        "Qing Dynasty",
        "Modern Times"
    ]
    sens_df["Dynasty"] = pd.Categorical(
        sens_df["Dynasty"],
        categories=dynasty_order,
        ordered=True
    )
    sens_df = sens_df.sort_values(["Dynasty", "Gamma"])

    metrics = [
        ("Communities", "Number of communities", "Communities"),
        ("Modularity_Q", "Modularity Q", "Modularity_Q"),
        ("ARI_vs_gamma1.0", "ARI vs γ = 1.0", "ARI_vs_gamma1.0"),
        ("NMI_vs_gamma1.0", "NMI vs γ = 1.0", "NMI_vs_gamma1.0")
    ]

    unique_dynasties = [d for d in dynasty_order if d in sens_df["Dynasty"].astype(str).unique()]

    # ========================================================
    # 1. Combined 4-panel figure
    # ========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric, ylabel, short_name) in zip(axes, metrics):
        for dynasty in unique_dynasties:
            sub = sens_df[sens_df["Dynasty"] == dynasty]
            ax.plot(
                sub["Gamma"],
                sub[metric],
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=dynasty
            )

        ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Resolution parameter γ", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        if metric in ["ARI_vs_gamma1.0", "NMI_vs_gamma1.0"]:
            ax.set_ylim(0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.suptitle(
        "Sensitivity analysis of Louvain community detection across different γ values",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    sens_fig_base = os.path.join(cluster_output_path, "Supplementary_Fig_DH_Sensitivity_Analysis")
    plt.savefig(sens_fig_base + ".png", dpi=400, bbox_inches="tight")
    plt.savefig(
        sens_fig_base + ".tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()

    # ========================================================
    # 2. Export four separate single-panel figures
    # ========================================================
    for metric, ylabel, short_name in metrics:
        plt.figure(figsize=(8, 6))

        for dynasty in unique_dynasties:
            sub = sens_df[sens_df["Dynasty"] == dynasty]
            plt.plot(
                sub["Gamma"],
                sub[metric],
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=dynasty
            )

        plt.axvline(x=1.0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Resolution parameter γ", fontsize=12, fontweight="bold")
        plt.ylabel(ylabel, fontsize=12, fontweight="bold")
        plt.title(ylabel, fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(fontsize=9, frameon=False, loc="best")

        if metric in ["ARI_vs_gamma1.0", "NMI_vs_gamma1.0"]:
            plt.ylim(0, 1.05)

        plt.tight_layout()

        single_fig_base = os.path.join(
            cluster_output_path,
            f"Supplementary_Fig_DH_Sensitivity_{short_name}"
        )
        plt.savefig(single_fig_base + ".png", dpi=400, bbox_inches="tight")
        plt.savefig(
            single_fig_base + ".tiff",
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"}
        )
        plt.close()

# ============================================================
# Export Top 10 summary for all dynasties
# ============================================================
if len(top10_all) > 0:
    top10_all_df = pd.concat(top10_all, ignore_index=True)
    top10_all_df.to_excel(
        os.path.join(cluster_output_path, "Supplementary_Table_DH_Top10_Centrality_All_Dynasties.xlsx"),
        index=False
    )

print("\n=== Structural disease-herb community analysis for all dynasties has been completed "
      "(with terminology ID mapping from the attached code included) ===")