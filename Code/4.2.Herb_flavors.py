import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from matplotlib import font_manager as fm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

excel_path = os.path.join(PROJECT_ROOT, "Data", "Herb", "Herb-Terminology List.xlsx")
output_dir = os.path.join(PROJECT_ROOT, "Results", "Herb")
os.makedirs(output_dir, exist_ok=True)


# ========== 2. Read Excel data ==========
df = pd.read_excel(excel_path)
df.columns = df.columns.str.strip()

required_cols = ["Property", "Flavor", "Meridian tropism"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in Excel file. Found columns: {list(df.columns)}")

print("Using columns:", required_cols)


# ========== 3. Define function to split by ';' and count ==========
def split_and_count(series, sep=';'):
    """Split each cell by semicolon and count each unique term."""
    items = []
    for val in series.dropna():
        for sub in str(val).split(sep):
            sub = sub.strip()
            if sub:
                items.append(sub)
    count_series = pd.Series(items).value_counts()
    df_count = count_series.reset_index()
    df_count.columns = ['Items', 'Frequency']   # <-- compatible with all pandas versions
    return df_count


# ========== 4. Compute statistics ==========
property_count = split_and_count(df["Property"])
flavor_count = split_and_count(df["Flavor"])
meridian_count = split_and_count(df["Meridian tropism"])

# Save to Excel
save_path = os.path.join(output_dir, "Herb_Property_Flavor_Meridian_Statistics.xlsx")
with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
    property_count.to_excel(writer, sheet_name="Property_Statistics", index=False)
    flavor_count.to_excel(writer, sheet_name="Flavor_Statistics", index=False)
    meridian_count.to_excel(writer, sheet_name="Meridian_Statistics", index=False)

print(f"Statistical results saved to: {save_path}")


# ========== 5. Define radar chart function ==========
def plot_english_radar(data_df, title, save_name, max_items=10, rlabel_angle=45):
    data_dict = dict(zip(data_df["Items"], data_df["Frequency"]))
    sorted_items = sorted(data_dict.items(), key=lambda x: -x[1])[:max_items]
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Prepare radar data
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Plot radar chart
    ax.plot(angles, values, linewidth=3, color='blue')
    ax.fill(angles, values, alpha=0.25, color='skyblue')

    # Set radar chart style
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(rlabel_angle)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=20, weight='bold')

    # Adjust label position
    for label in ax.get_xticklabels():
        label.set_y(label.get_position()[1] - 0.10)

    # Integer y-axis ticks
    max_val = max(values)
    step = max(1, math.ceil(max_val / 5))
    y_ticks = np.arange(0, max_val + step, step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(tick)}" for tick in y_ticks])
    for label in ax.get_yticklabels():
        label.set_fontproperties(fm.FontProperties(family='Arial', size=16))

    # Save figure
    plt.savefig(os.path.join(output_dir, f"{save_name}.tiff"),
                dpi=600, format='tiff', bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close()


# ========== 6. Generate radar charts ==========
plot_english_radar(property_count, "Distribution of Medicinal Properties", "Property_Distribution", rlabel_angle=37)
plot_english_radar(flavor_count, "Distribution of Medicinal Flavors", "Flavor_Distribution", rlabel_angle=32)
plot_english_radar(meridian_count, "Distribution of Meridian Tropism", "Meridian_Distribution", rlabel_angle=32)

print(f"All radar charts have been saved to: {os.path.abspath(output_dir)}")