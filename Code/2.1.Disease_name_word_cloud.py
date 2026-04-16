import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ch_font_path = 'C:/Windows/Fonts/msyh.ttc'
en_font_path = 'C:/Windows/Fonts/arial.ttf'

fm.fontManager.addfont(ch_font_path)
fm.fontManager.addfont(en_font_path)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

input_path = os.path.join(PROJECT_ROOT, "Data", "Disease name", "Disease name-Word cloud.xlsx")
output_dir = os.path.join(PROJECT_ROOT, "Results", "Disease name")
os.makedirs(output_dir, exist_ok=True)

# =====================
# Load All Sheets
# =====================
excel_data = pd.ExcelFile(input_path)
all_sheets = {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}

# =====================
# 1️⃣ Overall Disease Name Word Cloud
# =====================
if "Disease name-Frequency" in all_sheets:
    df_all = all_sheets["Disease name-Frequency"]
    print("Generating overall disease name word cloud...")

    # Data preparation
    word_freq = dict(zip(df_all["Disease name"], df_all["Frequency"]))

    # Word cloud generation
    wordcloud_all = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        font_path=ch_font_path,  # Ensure Chinese compatibility
        collocations=False
    ).generate_from_frequencies(word_freq)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_all, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    # Export
    output_file_all = os.path.join(output_dir, "Disease_WordCloud.tiff")
    plt.savefig(output_file_all, format='tiff', dpi=600, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"✅ Overall word cloud saved: {output_file_all}")

# =====================
# 2️⃣ Dynasty-based Disease Name Word Clouds
# =====================
if "Dynasty-Disease name-Frequency" in all_sheets:
    df_dynasty = all_sheets["Dynasty-Disease name-Frequency"]
    print("Generating dynasty-based word clouds...")

    if set(["Dynasty", "Disease name", "Frequency"]).issubset(df_dynasty.columns):
        for dynasty, group in df_dynasty.groupby("Dynasty"):
            print(f"  -> Processing: {dynasty}")

            # Data preparation
            word_freq = dict(zip(group["Disease name"], group["Frequency"]))

            # Word cloud generation
            wordcloud_dyn = WordCloud(
                width=1000,
                height=600,
                background_color='white',
                font_path=ch_font_path,
                collocations=False
            ).generate_from_frequencies(word_freq)

            # Plot
            plt.figure(figsize=(10, 8))
            plt.imshow(wordcloud_dyn, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()

            # Export
            output_file_dyn = os.path.join(output_dir, f"Disease_WordCloud_{dynasty}.tiff")
            plt.savefig(output_file_dyn, format='tiff', dpi=600, bbox_inches='tight',
                        pil_kwargs={"compression": "tiff_lzw"})
            plt.close()

        print("✅ All dynasty-based word clouds generated successfully!")

print("\n🎉 All word cloud generation tasks completed!")
plt.show()