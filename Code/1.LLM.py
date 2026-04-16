import pandas as pd
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(PROJECT_ROOT, "Data", "LLM")
output_dir = os.path.join(PROJECT_ROOT, "Results", "LLM")
os.makedirs(output_dir, exist_ok=True)

manual_path = os.path.join(input_dir, "Manual extraction.xlsx")
llm_path = os.path.join(input_dir, "LLM extraction.xlsx")

comparison_file = os.path.join(output_dir, "Detailed_comparison.xlsx")
summary_path = os.path.join(output_dir, "Summary.xlsx")
kappa_fig_path = os.path.join(output_dir, "Cohen's Kappa.tiff")
heatmap_path = os.path.join(output_dir, "Evaluation_heatmap.tiff")

# 2.Functions
def load_excel_files(manual_path, llm_path):
    """Load Excel files"""
    try:
        manual_df = pd.read_excel(manual_path)
        print(f"Manual extraction file columns: {manual_df.columns.tolist()} | Shape: {manual_df.shape}")

        llm_df = pd.read_excel(llm_path)
        print(f"LLM extraction file columns: {llm_df.columns.tolist()} | Shape: {llm_df.shape}")

        return manual_df, llm_df
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None


def extract_entities_by_type(df, entity_type):
    """Extract all related columns of a given entity type"""
    entity_columns = [col for col in df.columns if col.startswith(entity_type)]
    entities_by_no = {}

    for _, row in df.iterrows():
        no = row['NO.']
        entities = set()
        for col in entity_columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                entity = str(value).strip()
                if entity and entity.lower() != 'nan':
                    entities.add(entity)
        entities_by_no[no] = entities

    return entities_by_no


def calculate_kappa(manual_entities, llm_entities):
    """Calculate Cohen's Kappa for each record"""
    all_entities = set()
    for s in manual_entities.values():
        all_entities.update(s)
    for s in llm_entities.values():
        all_entities.update(s)
    all_entities = sorted(list(all_entities))

    kappa_scores = {}
    kappa_list = []

    for no in manual_entities.keys():
        manual_set = manual_entities.get(no, set())
        llm_set = llm_entities.get(no, set())

        manual_vec = [1 if e in manual_set else 0 for e in all_entities]
        llm_vec = [1 if e in llm_set else 0 for e in all_entities]

        if not manual_set and not llm_set:
            kappa = 1.0
        else:
            kappa = cohen_kappa_score(manual_vec, llm_vec)

        kappa_scores[no] = kappa
        kappa_list.append(kappa)

    avg_kappa = np.mean(kappa_list) if kappa_list else 0.0
    return kappa_scores, avg_kappa


def calculate_metrics(manual_entities, llm_entities, entity_type):
    """Calculate Precision, Recall, F1-Score, and Coverage"""
    results = {}
    is_flexible_precision = entity_type in ['Formula herb', 'Formula name']

    for no in manual_entities.keys():
        manual_set = manual_entities.get(no, set())
        llm_set = llm_entities.get(no, set())
        intersection = set()

        if is_flexible_precision:
            for m in manual_set:
                for l in llm_set:
                    if m in l or l in m:
                        intersection.add(m)
                        break
        else:
            intersection = manual_set & llm_set

        if is_flexible_precision and manual_set.issubset(llm_set):
            Precision = 1.0
        else:
            Precision = len(intersection) / len(llm_set) if llm_set else 0

        Recall = len(intersection) / len(manual_set) if manual_set else 0
        Coverage = 1.0 if manual_set and intersection else 0.0
        F1_Score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

        results[no] = {
            'Manual_count': len(manual_set),
            'LLM_count': len(llm_set),
            'Correct_count': len(intersection),
            'Precision': Precision,
            'Recall': Recall,
            'F1-Score': F1_Score,
            'Coverage': Coverage,
            'Manual_entities': manual_set,
            'LLM_entities': llm_set,
            'Correct_entities': intersection
        }

    return results


def analyze_entity_type(manual_df, llm_df, entity_type, entity_name_en):
    """Entity-level evaluation"""
    print(f"\nAnalyzing entity type: {entity_name_en}")
    print("=" * 50)

    manual_entities = extract_entities_by_type(manual_df, entity_type)
    llm_entities = extract_entities_by_type(llm_df, entity_type)
    metrics = calculate_metrics(manual_entities, llm_entities, entity_type)
    kappa_scores, avg_kappa = calculate_kappa(manual_entities, llm_entities)

    total_manual = sum(len(v) for v in manual_entities.values())
    total_llm = sum(len(v) for v in llm_entities.values())
    total_correct = sum(m['Correct_count'] for m in metrics.values())

    overall_Precision = total_correct / total_llm if total_llm else 0
    overall_Recall = total_correct / total_manual if total_manual else 0
    overall_F1_Score = 2 * overall_Precision * overall_Recall / (overall_Precision + overall_Recall) if (
                overall_Precision + overall_Recall) else 0
    overall_Coverage = sum(m['Coverage'] for m in metrics.values()) / len(metrics) if metrics else 0

    print(f"{entity_name_en} Avg. Kappa: {avg_kappa:.4f}")
    print(f"Total manual entities: {total_manual}")
    print(f"Total LLM entities: {total_llm}")
    print(f"Correctly extracted entities: {total_correct}")
    print(f"Overall Precision: {overall_Precision:.4f}")
    print(f"Overall Recall: {overall_Recall:.4f}")
    print(f"Overall F1-Score: {overall_F1_Score:.4f}")
    print(f"Overall Coverage: {overall_Coverage:.4f}")

    return {
        'entity_type': entity_type,
        'entity_name_en': entity_name_en,
        'metrics': metrics,
        'kappa': {
            'per_no': kappa_scores,
            'avg_kappa': avg_kappa
        },
        'overall': {
            'Total_manual': total_manual,
            'Total_LLM': total_llm,
            'Total_correct': total_correct,
            'Precision': overall_Precision,
            'Recall': overall_Recall,
            'F1-Score': overall_F1_Score,
            'Coverage': overall_Coverage
        }
    }


def create_comparison_table(results, filename):
    """Create detailed comparison table"""
    rows = []
    for entity_type, data in results.items():
        kappa_scores = data['kappa']['per_no']
        for no, metric in data['metrics'].items():
            missed = metric['Manual_entities'] - metric['Correct_entities']
            false = metric['LLM_entities'] - metric['Correct_entities']
            rows.append({
                'NO.': no,
                'Entity_type': data['entity_name_en'],
                'Manual_count': metric['Manual_count'],
                'LLM_count': metric['LLM_count'],
                'Correct_count': metric['Correct_count'],
                'Precision': metric['Precision'],
                'Recall': metric['Recall'],
                'F1-Score': metric['F1-Score'],
                'Coverage': metric['Coverage'],
                'Kappa': kappa_scores.get(no, 0),
                'Manual_entities': '|'.join(metric['Manual_entities']),
                'LLM_entities': '|'.join(metric['LLM_entities']),
                'Correct_entities': '|'.join(metric['Correct_entities']),
                'Missed_entities': '|'.join(missed),
                'False_entities': '|'.join(false)
            })
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    print(f"Detailed comparison table exported: {filename}")
    return df


def plot_avg_kappa_bar(summary_df, output_path):
    """Plot average Kappa bar chart"""
    plt.rcParams['font.family'] = 'Arial'
    entity_names = summary_df['entity_type']
    avg_kappa_values = summary_df['avg_kappa']

    plt.figure(figsize=(7,6))
    bars = plt.bar(entity_names, avg_kappa_values, color='lightblue')

    for bar, value in zip(bars, avg_kappa_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{value:.3f}", ha='center', va='bottom', fontsize=14)

    plt.ylim(0, 1.0)
    plt.xlabel("Entity", fontsize=14, fontweight='bold')
    plt.ylabel("Kappa", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=600, format='tiff', pil_kwargs={"compression":"tiff_lzw"})
    print(f"Kappa bar chart saved: {output_path}")
    plt.show()


def plot_metrics_heatmap(summary_df, output_path):
    """Plot heatmap of Precision, Recall, and F1-Score"""
    plt.rcParams['font.family'] = 'Arial'
    metrics = ['Precision', 'Recall', 'F1-Score']

    sorted_entities = summary_df.sort_values(by="F1-Score", ascending=False)['entity_type']
    heatmap_data = summary_df.set_index('entity_type').loc[sorted_entities, metrics].T

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        heatmap_data,
        annot=True, fmt=".3f", cmap="RdYlBu_r",
        cbar_kws={'label': 'Metric value'},
        linewidths=0.5, linecolor='white',
        annot_kws={"size": 14}
    )
    ax.set_xlabel("Entity", fontsize=12, fontweight='bold')
    ax.set_ylabel("Evaluation metric", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10, rotation=0, fontweight='bold')
    plt.yticks(fontsize=12, rotation=0, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, format='tiff', pil_kwargs={"compression":"tiff_lzw"})
    print(f"Heatmap saved: {output_path}")
    plt.show()


# 3.Main
if __name__ == "__main__":
    manual_df, llm_df = load_excel_files(manual_path, llm_path)
    if manual_df is None or llm_df is None:
        exit()

    entity_mapping = {
        'Disease name': 'Disease names',
        'Symptom': 'Symptoms',
        'Formula name': 'Formula names',
        'Formula herb': 'Formula herbs',
        'Herb': 'Herbs'
    }

    all_results = {}
    for et, en_name in entity_mapping.items():
        res = analyze_entity_type(manual_df, llm_df, et, en_name)
        all_results[et] = res

    comparison_df = create_comparison_table(all_results, comparison_file)

    summary_data = []
    for et, data in all_results.items():
        summary_data.append({
            'entity_type': data['entity_name_en'],
            **data['overall'],
            'avg_kappa': data['kappa']['avg_kappa']
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(summary_path, index=False)
    print(f"Summary table exported: {summary_path}")

    plot_avg_kappa_bar(summary_df, kappa_fig_path)
    plot_metrics_heatmap(summary_df, heatmap_path)