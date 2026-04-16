import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(PROJECT_ROOT, "Data", "Formula herb", "Formula herb-Dynasty and Frequency.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Results", "Formula herb")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def setup_font():
    """Set Times New Roman font"""
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False


def load_and_process_data(excel_path):
    """Load dynasty-herb sets from Excel"""
    print("Reading Excel file...")
    df = pd.read_excel(excel_path, header=0, skiprows=[1])
    dynasties = [c for c in df.columns if "Unnamed" not in str(c)]
    dynasty_sets = {d: set(df[d].dropna().astype(str).str.strip().unique()) for d in dynasties}
    for d in dynasties:
        print(f"{d}: {len(dynasty_sets[d])} herbs")
    return dynasty_sets, dynasties


def calculate_adjacent_intersections(dynasty_sets, dynasties):
    """Calculate intersections of consecutive dynasties"""
    print("Calculating adjacent intersections...")
    intersection_data = []
    for length in range(2, min(8, len(dynasties) + 1)):
        for start_idx in range(len(dynasties) - length + 1):
            group = dynasties[start_idx:start_idx + length]
            inter = dynasty_sets[group[0]].copy()
            for d in group[1:]:
                inter &= dynasty_sets[d]
            if inter:
                intersection_data.append({
                    'combination': ' ∩ '.join(group),
                    'herb_count': len(inter),
                    'dynasty_count': len(group),
                    'type': f'{length}_adjacent_intersection',
                    'start_index': start_idx,
                    'dynasties': group,
                    'herbs': sorted(inter)
                })
    intersection_data.sort(key=lambda x: x['herb_count'], reverse=True)
    print(f"Generated {len(intersection_data)} intersection groups")
    return intersection_data


def create_upset_plot(intersection_data, dynasty_sets, dynasties, save_path=None, point_size=300):
    """Create upset plot"""
    print(f"Creating upset plot with {len(intersection_data)} sets...")

    fig = plt.figure(figsize=(26, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 0.1, 1], width_ratios=[1, 3],
                          hspace=0.05, wspace=0.05)

    # Top bar chart
    ax_bar = fig.add_subplot(gs[0, 1])
    sizes = [e['herb_count'] for e in intersection_data]
    if not sizes:
        raise ValueError("No intersections found.")
    bars = ax_bar.bar(range(len(intersection_data)), sizes, color='black', width=0.6)
    for bar, size in zip(bars, sizes):
        ax_bar.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    f'{size}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    ax_bar.set_ylabel('Intersection number', fontsize=22, fontweight='bold')
    ax_bar.set_xlim(-0.5, len(intersection_data) - 0.5)
    ax_bar.set_ylim(0, max(sizes) * 1.1)
    ax_bar.grid(True, alpha=0.3)
    ax_bar.set_xticks([])
    ax_bar.tick_params(axis='y', labelsize=18)

    # Left bar chart
    ax_left = fig.add_subplot(gs[2, 0])
    counts = [len(dynasty_sets[d]) for d in dynasties]
    y_pos = range(len(dynasties))
    ax_left.barh(y_pos, counts, color='black', height=0.6)
    ax_left.set_xlabel('Total number of herbs', fontsize=20, fontweight='bold')
    ax_left.set_ylim(-0.5, len(dynasties) - 0.5)
    ax_left.set_xlim(0, max(counts) * 1.3)
    ax_left.grid(True, alpha=0.3)
    ax_left.tick_params(axis='x', labelsize=16)
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(dynasties, fontsize=18, fontweight='bold')
    ax_left.invert_yaxis()
    ax_left.invert_xaxis()
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    # Matrix
    ax_matrix = fig.add_subplot(gs[2, 1])
    matrix = np.zeros((len(dynasties), len(intersection_data)))
    for j, e in enumerate(intersection_data):
        for d in e['dynasties']:
            i = dynasties.index(d)
            matrix[i, j] = 1
    for i in range(len(dynasties)):
        for j in range(len(intersection_data)):
            color = 'black' if matrix[i, j] else 'lightgray'
            ax_matrix.scatter(j, i, s=point_size, c=color, zorder=3)
    for j, e in enumerate(intersection_data):
        if len(e['dynasties']) > 1:
            idx = sorted([dynasties.index(d) for d in e['dynasties']])
            ax_matrix.plot([j, j], [idx[0], idx[-1]], 'k-', linewidth=3, zorder=1)
    ax_matrix.set_xlim(-0.5, len(intersection_data) - 0.5)
    ax_matrix.set_ylim(-0.5, len(dynasties) - 0.5)
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks([])
    ax_matrix.invert_yaxis()
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)
    for i in range(len(dynasties) + 1):
        ax_matrix.axhline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)
    for j in range(len(intersection_data) + 1):
        ax_matrix.axvline(j - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.2)

    if save_path:
        if not save_path.lower().endswith(".tiff"):
            save_path = save_path + ".tiff"
        plt.savefig(
            save_path,
            dpi=600,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pil_kwargs={"compression": "tiff_lzw"}
        )
        print(f"Figure saved: {save_path}")

    return fig


def save_intersection_results(intersection_data, save_path):
    """Save intersection results into one Excel file with two sheets"""
    print("Saving intersection results...")

    # Detailed info
    df = pd.DataFrame([
        {
            'Rank': i + 1,
            'Combination': e['combination'],
            'Herb count': e['herb_count'],
            'Dynasty count': e['dynasty_count'],
            'Type': e['type'],
            'Start index': e['start_index'],
            'Herbs': '、'.join(e['herbs'])
        }
        for i, e in enumerate(intersection_data)
    ])

    df.to_excel(save_path, sheet_name="Intersection results", index=False)
    print(f"Excel saved: {save_path}")


def main():
    setup_font()
    dynasty_sets, dynasties = load_and_process_data(INPUT_FILE)
    intersection_data = calculate_adjacent_intersections(dynasty_sets, dynasties)
    if intersection_data:
        create_upset_plot(
            intersection_data, dynasty_sets, dynasties,
            save_path=os.path.join(OUTPUT_DIR, "Upset_plot")
        )
        save_intersection_results(intersection_data, os.path.join(OUTPUT_DIR, "Upset_results.xlsx"))
        plt.show()
    else:
        print("No intersections found.")


if __name__ == "__main__":
    main()