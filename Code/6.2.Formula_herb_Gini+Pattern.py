import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import font_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

font_path_english = 'C:/Windows/Fonts/Arial.ttf'
font_prop_english = font_manager.FontProperties(fname=font_path_english, size=12)
plt.rcParams['font.sans-serif'] = [font_prop_english.get_name()]
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

file_path = os.path.join(PROJECT_ROOT, "Data", "Formula herb", "Formula herb-Dynasty and Frequency.xlsx")
output_path = os.path.join(PROJECT_ROOT, "Results", "Formula herb")

# ======= Dynasty Order =======
dynasty_order = [
    'before the end of Han Dynasty',
    'Wei, Jin, Southern and Northern Dynasties',
    'Sui, Tang and Five Dynasties',
    'Song, Jin and Yuan Dynasties',
    'Ming Dynasty',
    'Qing Dynasty',
    'Modern Times'
]


def load_and_preprocess_data(file_path):
    """Load and preprocess herb data from Excel file"""
    print("=== Data Loading and Preprocessing ===\n")

    df_herbs = pd.read_excel(file_path, sheet_name=0)

    dynasty_columns = [col for i, col in enumerate(df_herbs.columns) if i % 2 == 0]
    frequency_columns = [col for i, col in enumerate(df_herbs.columns) if i % 2 == 1]

    dynasties_data = []
    for dynasty_col, freq_col in zip(dynasty_columns, frequency_columns):
        dynasty_data = df_herbs[[dynasty_col, freq_col]].iloc[1:].copy()
        dynasty_data.columns = ['Herb', 'Frequency']
        dynasty_data = dynasty_data.dropna()
        dynasty_data['Frequency'] = pd.to_numeric(dynasty_data['Frequency'], errors='coerce')
        dynasty_data = dynasty_data.dropna()
        dynasty_data['Dynasty'] = dynasty_col
        dynasties_data.append(dynasty_data)

    all_herbs_data = pd.concat(dynasties_data, ignore_index=True)
    print(f"Combined data: {len(all_herbs_data)} records, {all_herbs_data['Herb'].nunique()} unique herbs")
    return all_herbs_data


def gini_analysis(herbs_data):
    """Calculate Gini index for each dynasty"""

    def calculate_gini(frequencies):
        frequencies = np.array(frequencies)
        frequencies = np.sort(frequencies)
        n = len(frequencies)
        cumsum = np.cumsum(frequencies)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    gini_stats = []
    for dynasty in herbs_data['Dynasty'].unique():
        subset = herbs_data[herbs_data['Dynasty'] == dynasty]
        gini = calculate_gini(subset['Frequency'].values)
        gini_stats.append({'Dynasty': dynasty, 'Gini': gini})

    gini_df = pd.DataFrame(gini_stats)
    print("\n=== Gini Index Analysis ===")
    print(gini_df.round(3))
    return gini_df


def similarity_matrix_analysis(herbs_data):
    """Calculate similarity matrix between dynasties (fixed order)"""
    herb_dynasty_matrix = herbs_data.pivot_table(
        index='Herb', columns='Dynasty', values='Frequency', fill_value=0
    )
    herb_dynasty_matrix = herb_dynasty_matrix[dynasty_order]

    similarity_matrix = np.zeros((len(dynasty_order), len(dynasty_order)))
    for i, d1 in enumerate(dynasty_order):
        for j, d2 in enumerate(dynasty_order):
            vec1, vec2 = herb_dynasty_matrix[d1].values, herb_dynasty_matrix[d2].values
            similarity = 1 - cosine(vec1, vec2)
            similarity_matrix[i, j] = similarity

    similarity_df = pd.DataFrame(similarity_matrix, index=dynasty_order, columns=dynasty_order)
    print("\n=== Dynasty Similarity Matrix (Fixed Order) ===")
    print(similarity_df.round(3))
    return similarity_df, herb_dynasty_matrix


def evolution_pattern_analysis(herb_dynasty_matrix):
    """Classify herbs into historical evolution patterns"""

    def analyze_herb(herb_name):
        herb_data = herb_dynasty_matrix.loc[herb_name]
        appearance_count = (herb_data > 0).sum()
        max_continuous, current = 0, 0
        for freq in herb_data:
            if freq > 0:
                current += 1
                max_continuous = max(max_continuous, current)
            else:
                current = 0
        first_appearance, last_appearance = None, None
        for dynasty, freq in herb_data.items():
            if freq > 0:
                if first_appearance is None:
                    first_appearance = dynasty
                last_appearance = dynasty
        return {
            'Herb': herb_name,
            'AppearanceCount': appearance_count,
            'MaxContinuous': max_continuous,
            'FirstAppearance': first_appearance,
            'LastAppearance': last_appearance,
            'TotalFrequency': herb_data.sum(),
            'PeakDynasty': herb_data.idxmax(),
            'PeakFrequency': herb_data.max()
        }

    continuity_df = pd.DataFrame([analyze_herb(h) for h in herb_dynasty_matrix.index])

    def classify_pattern(row):
        if row['AppearanceCount'] >= 6:
            return "Classical herb"
        elif row['AppearanceCount'] >= 4:
            return "Common herb"
        elif row['FirstAppearance'] in ['before the end of Han Dynasty',
                                        'Wei, Jin, Southern and Northern Dynasties'] and \
             row['LastAppearance'] in ['Qing Dynasty', 'Modern Times']:
            return "Inherited herb"
        elif row['FirstAppearance'] in ['Song, Jin and Yuan Dynasties',
                                        'Ming Dynasty', 'Qing Dynasty', 'Modern Times']:
            return "Later herb"
        elif row['LastAppearance'] in ['before the end of Han Dynasty',
                                       'Wei, Jin, Southern and Northern Dynasties',
                                       'Sui, Tang and Five Dynasties']:
            return "Early herb"
        else:
            return "Stage herb"

    continuity_df['Pattern'] = continuity_df.apply(classify_pattern, axis=1)
    print("\n=== Evolution Pattern Distribution ===")
    print(continuity_df['Pattern'].value_counts())
    return continuity_df


def create_figures(gini_df, similarity_df, output_path):
    """Generate Gini index bar chart and similarity matrix heatmap"""
    custom_colors = ['#D8D8D7', '#E4EBF5', '#ADD16B',
                     '#F5BDBD', '#F4E571', '#EBA165', '#BBB7D5']

    # ===== Gini Index Bar Chart =====
    gini_ordered = gini_df.set_index('Dynasty').loc[dynasty_order]
    plt.figure(figsize=(8, 7))
    bars = plt.bar(gini_ordered.index, gini_ordered['Gini'],
                   color=custom_colors[:len(dynasty_order)], edgecolor='black', linewidth=1)
    plt.ylabel("Gini index", fontsize=16, fontweight='bold')
    plt.xlabel("Dynasty", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, rotation=30, ha='right', fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    for bar, value in zip(bars, gini_ordered['Gini']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Gini_index.tif"),
                dpi=600, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close()

    # ===== Similarity Matrix Heatmap =====
    similarity_ordered = similarity_df.loc[dynasty_order, dynasty_order]
    plt.figure(figsize=(10, 9))
    sns.heatmap(similarity_ordered, annot=True, fmt=".3f",
                cmap="YlOrRd", cbar_kws={'label': 'Similarity', 'shrink': 0.7},
                square=True, linewidths=0.5, linecolor='gray')
    plt.ylabel("Dynasty", fontsize=16, fontweight='bold')
    plt.xlabel("Dynasty", fontsize=16, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=13, fontweight='bold')
    plt.yticks(rotation=0, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Dynasty_similarity_heatmap.tif"),
                dpi=600, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"\nFigures saved in: {output_path}")


def plot_top20_herbs(herbs_data, output_path):
    """Plot top 20 herbs by total frequency"""
    print("\n=== Plotting Top 20 Herbs ===")

    herb_total = herbs_data.groupby('Herb')['Frequency'].sum().reset_index()
    herb_total_sorted = herb_total.sort_values(by='Frequency', ascending=False).head(20)

    med_names = herb_total_sorted['Herb'].tolist()
    med_counts = herb_total_sorted['Frequency'].tolist()
    colors = list(reversed(sns.color_palette("viridis", len(med_names))))

    plt.figure(figsize=(10, 8))
    plt.barh(med_names, med_counts, color=colors)
    plt.xlabel("Frequency", fontsize=15, fontweight='bold')
    plt.ylabel("Herb name", fontsize=15, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()

    for index, value in enumerate(med_counts):
        plt.text(value + 0.5, index, str(int(value)), va='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Herb_Top20.tif"),
                dpi=600, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close()

    print(f"Top 20 herbs figure saved in: {output_path}")


def main():
    herbs_data = load_and_preprocess_data(file_path)
    gini_df = gini_analysis(herbs_data)
    similarity_df, herb_dynasty_matrix = similarity_matrix_analysis(herbs_data)
    continuity_df = evolution_pattern_analysis(herb_dynasty_matrix)

    os.makedirs(output_path, exist_ok=True)
    gini_df.to_csv(os.path.join(output_path, 'Gini_analysis.csv'), index=False)
    similarity_df.to_csv(os.path.join(output_path, 'Dynasty_similarity_matrix.csv'))
    continuity_df.to_csv(os.path.join(output_path, 'Formula herb_evolution_patterns.csv'), index=False)

    create_figures(gini_df, similarity_df, output_path)
    plot_top20_herbs(herbs_data, output_path)
    print(f"\nAnalysis completed. Results saved in: {output_path}")


if __name__ == "__main__":
    main()