import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

font = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = font.get_name()

file_path = os.path.join(PROJECT_ROOT, "Data", "Disease name", "Disease name.xlsx")
excel_data = pd.ExcelFile(file_path)
sheet_data = excel_data.parse('Disease name-Time')

# 3.Data cleaning
cleaned_data = sheet_data.dropna(subset=['Time'], how='any')
cleaned_data['Time'] = pd.to_numeric(cleaned_data['Time'], errors='coerce')
cleaned_data = cleaned_data.dropna(subset=['Time'])
cleaned_data = cleaned_data.drop(columns=['No.'])
melted_data = cleaned_data.melt(id_vars=['Time'], var_name='Disease name', value_name='Value')
melted_data = melted_data.dropna(subset=['Value'])

# 4.Plot
plt.figure(figsize=(16, 20))

# 5.Background shading (by period)
plt.axvspan(-50 , 220, color='#DFE1E2', alpha=0.5)
plt.axvspan(220, 589, color='#CEE2F6', alpha=0.3)
plt.axvspan(581, 960, color='#CBE0BB', alpha=0.3)
plt.axvspan(960, 1368, color='#FDDDDC', alpha=0.3)
plt.axvspan(1368, 1644, color='#F5E09B', alpha=0.3)
plt.axvspan(1644, 1912, color='#FFBE7A', alpha=0.3)
plt.axvspan(1912, 1960, color='#DCD4E7', alpha=0.5)


# 6.Scatter plot
ax=sns.scatterplot(
    data=melted_data,
    x='Time', y='Disease name',
    hue='Value', palette="viridis",
    size='Value',
    sizes=(20, 500),
    legend=None
)

# 7.Colorbar
norm = plt.Normalize(melted_data['Value'].min(),melted_data['Value'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm,ax=ax, orientation="vertical", shrink=0.5,pad=0.02)
cbar.set_label('Value (Number)', fontsize=16,fontweight='bold',labelpad=15)
cbar.ax.tick_params(labelsize=12)

# 8.Axis format
plt.yticks(rotation=0, fontsize=12)
plt.xticks(rotation=45, fontsize=13)
plt.subplots_adjust(left=0.2, right=0.9, top=0.1, bottom=0.05)
plt.xlabel("Time", fontsize=20,fontweight='bold',labelpad=10)
plt.ylabel("Disease names", fontsize=20,fontweight='bold',labelpad=10)
plt.xticks(rotation=45)
plt.tight_layout()

# 9.Export
output_dir = os.path.join(PROJECT_ROOT, "Results", "Disease name")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'disease_distribution.tiff'), format='tiff', dpi=600, pil_kwargs={"compression":"tiff_lzw"})
plt.show()