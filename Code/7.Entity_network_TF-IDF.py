import pandas as pd
import numpy as np
import re
import os
import math
from collections import defaultdict
from itertools import product

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path_raw = os.path.join(PROJECT_ROOT, "Data", "Entity-Normalized.xlsx")
case_output_path = os.path.join(PROJECT_ROOT, "Data", "Entity network", "Case.xlsx")
tuples_output_dir = os.path.join(PROJECT_ROOT, "Results", "Entity network", "Tuples")
triplets_output_dir = os.path.join(PROJECT_ROOT, "Results", "Entity network", "Triplets")
pinyin_map_path = os.path.join(PROJECT_ROOT, "Data", "Entity network", "Pinyin Name.xlsx")

os.makedirs(tuples_output_dir, exist_ok=True)
os.makedirs(triplets_output_dir, exist_ok=True)

# ============================================================
# DICTIONARIES
# ============================================================
DYNASTY_MAP = {
    "before the end of Han Dynasty": "Han",
    "Wei, Jin, Southern and Northern Dynasties": "Wei",
    "Sui, Tang and Five Dynasties": "Sui",
    "Song, Jin and Yuan Dynasties": "Song",
    "Ming Dynasty": "Ming",
    "Qing Dynasty": "Qing",
    "Modern Times": "Modern",
}

PAIR_TYPE_MAP = {
    "Disease-Symptom": ("Disease", "Symptom"),
    "Disease-Formula herb": ("Disease", "Formula herb"),
    "Symptom-Formula herb": ("Symptom", "Formula herb"),
}

# ============================================================
# COMMON UTILITY FUNCTIONS
# ============================================================

def map_dynasty(name):
    for k, v in DYNASTY_MAP.items():
        if k in str(name):
            return v
    return str(name).strip()

def clean_and_split(text):
    """Clean and split text into individual terms."""
    if not isinstance(text, str):
        return []
    text = re.sub(r"[|\s、，,。；;]+", ",", text)
    text = re.sub(r"\([^)]*\)", "", text)
    return [w.strip() for w in text.split(",") if w.strip()]

def safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]+', "_", str(name).strip())

def read_pinyin_map(pinyin_path: str):
    """Read the Pinyin mapping table."""
    if os.path.exists(pinyin_path):
        try:
            df = pd.read_excel(pinyin_path)
            cn_col = None
            py_col = None

            for c in df.columns:
                lc = str(c).lower()
                if "chinese" in lc:
                    cn_col = c
                if "pinyin" in lc:
                    py_col = c

            if cn_col is None:
                cn_col = "Chinese Name" if "Chinese Name" in df.columns else df.columns[0]
            if py_col is None:
                py_col = "Pinyin Name" if "Pinyin Name" in df.columns else df.columns[1]

            mapping = dict(
                zip(
                    df[cn_col].astype(str).str.strip(),
                    df[py_col].astype(str).str.strip()
                )
            )
            return mapping
        except Exception as e:
            print(f"Warning: Failed to read the Pinyin mapping table: {e}")
    return {}

name2pinyin = read_pinyin_map(pinyin_map_path)

def to_pinyin(x: str) -> str:
    x = str(x).strip()
    return name2pinyin.get(x, x)

def detect_columns(df):
    """Detect disease, symptom, formula herb, and dynasty columns."""
    cols = list(df.columns)

    disease_col = None
    dynasty_col = None

    for c in cols:
        lc = str(c).strip().lower()
        if lc in ["disease name", "disease", "disease_name"]:
            disease_col = c
        if lc == "dynasty":
            dynasty_col = c

    symptom_cols = [c for c in cols if re.match(r"^Symptom(\.\d+)?$", str(c).strip(), flags=re.I)]
    formula_cols = [c for c in cols if re.match(r"^Formula herb(\.\d+)?$", str(c).strip(), flags=re.I)]

    return disease_col, symptom_cols, formula_cols, dynasty_col

# ============================================================
# PART 1: CLASSIFY DATA AND GENERATE Case.xlsx
# Scheme A: all four categories can overlap
# ============================================================

def classify_case_data(input_path, output_path):
    """
    Scheme A:
    - Disease-Symptom: any record containing both disease and symptom
    - Disease-Formula herb: any record containing both disease and formula herb
    - Symptom-Formula herb: any record containing both symptom and formula herb
    - Disease-Symptom-Formula herb: any record containing all three

    Note:
    The four categories are overlapping and not mutually exclusive.
    """
    df = pd.read_excel(input_path)

    disease_col, symptom_cols, formula_cols, dynasty_col = detect_columns(df)

    if disease_col is None:
        raise ValueError("Disease column not found.")
    if not symptom_cols:
        raise ValueError("Symptom columns not found.")
    if not formula_cols:
        raise ValueError("Formula herb columns not found.")
    if dynasty_col is None:
        raise ValueError("Dynasty column not found.")

    ds_rows = []
    dfh_rows = []
    sfh_rows = []
    dsfh_rows = []

    for _, row in df.iterrows():
        has_disease = pd.notna(row.get(disease_col)) and str(row.get(disease_col)).strip() != ""

        has_symptom = any(
            pd.notna(row.get(c)) and str(row.get(c)).strip() != ""
            for c in symptom_cols
        )

        has_formula = any(
            pd.notna(row.get(c)) and str(row.get(c)).strip() != ""
            for c in formula_cols
        )

        if has_disease and has_symptom:
            ds_rows.append(row.copy())

        if has_disease and has_formula:
            dfh_rows.append(row.copy())

        if has_symptom and has_formula:
            sfh_rows.append(row.copy())

        if has_disease and has_symptom and has_formula:
            dsfh_rows.append(row.copy())

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if ds_rows:
            pd.DataFrame(ds_rows).to_excel(writer, sheet_name="Disease-Symptom", index=False)
        if dfh_rows:
            pd.DataFrame(dfh_rows).to_excel(writer, sheet_name="Disease-Formula herb", index=False)
        if sfh_rows:
            pd.DataFrame(sfh_rows).to_excel(writer, sheet_name="Symptom-Formula herb", index=False)
        if dsfh_rows:
            pd.DataFrame(dsfh_rows).to_excel(writer, sheet_name="Disease-Symptom-Formula herb", index=False)

    print(f"Classification completed. Output file: {output_path}")
    print(f"   Disease-Symptom rows: {len(ds_rows)}")
    print(f"   Disease-Formula herb rows: {len(dfh_rows)}")
    print(f"   Symptom-Formula herb rows: {len(sfh_rows)}")
    print(f"   Disease-Symptom-Formula herb rows: {len(dsfh_rows)}")

# ============================================================
# TF-IDF CORE FUNCTIONS
# Global TF-IDF logic
# ============================================================

def compute_global_df(records, entity_keys):
    """
    DF(i) = number of records in which entity i appears globally
    """
    df_counts = defaultdict(int)
    for rec in records:
        seen = set()
        for key in entity_keys:
            seen.update(rec.get(key, []))
        for entity in seen:
            df_counts[entity] += 1
    return dict(df_counts)

def compute_idf(entity, df_counts, N):
    """
    IDF(i) = log(N / (DF_i + 1))
    """
    return math.log(N / (df_counts.get(entity, 0) + 1))

def compute_tf(entity, entity_list):
    """
    TF(i, j) = count(i in j) / len(entity_list)
    """
    if not entity_list:
        return 0.0
    return entity_list.count(entity) / len(entity_list)

# ============================================================
# RECORD EXTRACTION
# ============================================================

def extract_records_from_pair_sheet(df, sheet_name):
    """
    Extract structured records from a pairwise sheet.
    """
    disease_col, symptom_cols, formula_cols, dynasty_col = detect_columns(df)
    if dynasty_col is None:
        raise ValueError(f"{sheet_name} is missing the Dynasty column.")

    records = []

    for idx, row in df.iterrows():
        dynasty = map_dynasty(row.get(dynasty_col))
        if not dynasty or dynasty.lower() == "nan":
            continue

        disease = []
        symptom = []
        formula_herb = []

        if disease_col and pd.notna(row.get(disease_col)) and str(row.get(disease_col)).strip():
            disease = [str(row.get(disease_col)).strip()]

        for c in symptom_cols:
            if pd.notna(row.get(c)) and str(row.get(c)).strip():
                symptom.append(str(row.get(c)).strip())

        for c in formula_cols:
            if pd.notna(row.get(c)) and str(row.get(c)).strip():
                formula_herb.extend(clean_and_split(str(row.get(c))))

        rec = {
            "record_id": f"{safe_filename(sheet_name)}_{idx + 1}",
            "dynasty": dynasty,
            "disease": disease,
            "symptom": symptom,
            "formula_herb": formula_herb,
        }
        records.append(rec)

    return records

# ============================================================
# GENERATE NODE TABLE
# ============================================================

def build_node_table(tfidf_df):
    """
    In the node table, TF-IDF is calculated as the sum of TF-IDF values
    across all records within the same dynasty.
    """
    if tfidf_df.empty:
        return pd.DataFrame(columns=["Entity", "Type", "TF-IDF"])

    node_df = (
        tfidf_df.groupby(["entity", "entity_type"], as_index=False)["TFIDF"]
        .sum()
        .rename(columns={"entity": "Entity", "entity_type": "Type", "TFIDF": "TF-IDF"})
    )
    node_df["Entity"] = node_df["Entity"].map(to_pinyin)
    return node_df

# ============================================================
# PART 2: PROCESS GENERAL SHEETS (PAIRWISE NETWORKS)
# ============================================================

def process_general_sheets(input_path, output_edges_path):
    excel = pd.ExcelFile(input_path)

    for sheet_name in excel.sheet_names:
        if sheet_name.strip().lower() == "disease-symptom-formula herb":
            print(f"Skipping {sheet_name} (it will be processed in Part 3).")
            continue

        if sheet_name not in PAIR_TYPE_MAP:
            print(f"Skipping unknown pairwise sheet: {sheet_name}")
            continue

        print(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(input_path, sheet_name=sheet_name)
        if df.empty:
            print(f"Warning: Empty sheet: {sheet_name}")
            continue

        records = extract_records_from_pair_sheet(df, sheet_name)
        if not records:
            print(f"Warning: No valid records found in {sheet_name}")
            continue

        type1, type2 = PAIR_TYPE_MAP[sheet_name]

        key_map = {
            "Disease": "disease",
            "Symptom": "symptom",
            "Formula herb": "formula_herb"
        }

        key1 = key_map[type1]
        key2 = key_map[type2]

        # Compute global DF / IDF only for the entity types involved in this sheet
        entity_keys = [key1, key2]
        N = len(records)
        df_counts = compute_global_df(records, entity_keys)

        # 1. Record-level TF-IDF
        tfidf_rows = []
        for rec in records:
            rid = rec["record_id"]
            dyn = rec["dynasty"]

            for entity in set(rec[key1]):
                tf = compute_tf(entity, rec[key1])
                idf = compute_idf(entity, df_counts, N)
                tfidf_rows.append({
                    "record_id": rid,
                    "dynasty": dyn,
                    "entity": entity,
                    "entity_type": type1,
                    "TF": tf,
                    "IDF": idf,
                    "TFIDF": tf * idf
                })

            for entity in set(rec[key2]):
                tf = compute_tf(entity, rec[key2])
                idf = compute_idf(entity, df_counts, N)
                tfidf_rows.append({
                    "record_id": rid,
                    "dynasty": dyn,
                    "entity": entity,
                    "entity_type": type2,
                    "TF": tf,
                    "IDF": idf,
                    "TFIDF": tf * idf
                })

        tfidf_df = pd.DataFrame(tfidf_rows)
        dynasties = sorted(tfidf_df["dynasty"].dropna().unique().tolist())

        # Optional: export TF-IDF details for the whole sheet
        tfidf_detail_path = os.path.join(
            output_edges_path,
            f"{safe_filename(sheet_name)}_TFIDF_detail.xlsx"
        )
        tfidf_df.to_excel(tfidf_detail_path, index=False)

        for dynasty in dynasties:
            sub_tfidf = tfidf_df[tfidf_df["dynasty"] == dynasty].copy()
            if sub_tfidf.empty:
                continue

            # Node table
            node_df = build_node_table(sub_tfidf)

            # Edge accumulation
            edge_acc = defaultdict(float)
            co_count = defaultdict(int)

            dynasty_records = [r for r in records if r["dynasty"] == dynasty]

            for rec in dynasty_records:
                vals1 = {}
                vals2 = {}

                for e in set(rec[key1]):
                    tf = compute_tf(e, rec[key1])
                    idf = compute_idf(e, df_counts, N)
                    vals1[e] = tf * idf

                for e in set(rec[key2]):
                    tf = compute_tf(e, rec[key2])
                    idf = compute_idf(e, df_counts, N)
                    vals2[e] = tf * idf

                if not vals1 or not vals2:
                    continue

                # For same-type networks, avoid self-loops and duplicated directions
                if key1 == key2:
                    ents = sorted(set(vals1.keys()))
                    for i in range(len(ents)):
                        for j in range(i + 1, len(ents)):
                            a, b = ents[i], ents[j]
                            wa, wb = vals1[a], vals2[b]
                            if wa > 0 and wb > 0:
                                key = (a, b, type1, type2)
                                edge_acc[key] += math.sqrt(wa * wb)
                                co_count[key] += 1
                else:
                    for a, wa in vals1.items():
                        for b, wb in vals2.items():
                            if wa > 0 and wb > 0:
                                key = (a, b, type1, type2)
                                edge_acc[key] += math.sqrt(wa * wb)
                                co_count[key] += 1

            edges = []
            for (a, b, t1, t2), w in edge_acc.items():
                edges.append({
                    "Pinyin1": to_pinyin(a),
                    "Pinyin2": to_pinyin(b),
                    "Type1": t1,
                    "Type2": t2,
                    "Weight": w,
                    "Cooccurrence": co_count[(a, b, t1, t2)]
                })

            edge_df = pd.DataFrame(edges)

            dynasty_safe = safe_filename(dynasty)
            sheet_safe = safe_filename(sheet_name)
            node_path = os.path.join(output_edges_path, f"{sheet_safe}_{dynasty_safe}_nodes.xlsx")
            edge_path = os.path.join(output_edges_path, f"{sheet_safe}_{dynasty_safe}_edges.xlsx")

            node_df.to_excel(node_path, index=False)
            edge_df.to_excel(edge_path, index=False)

            print(f"    {dynasty} exported: nodes={len(node_df)}, edges={len(edge_df)}")

    print("All pairwise networks have been exported successfully.")

# ============================================================
# PART 3: TRIPLET HIERARCHICAL NETWORK
# ============================================================

def process_three_relation(input_path, output_dir):
    excel = pd.ExcelFile(input_path)

    for sheet_name in excel.sheet_names:
        if sheet_name.strip().lower() != "disease-symptom-formula herb":
            continue

        print(f"Processing triplet sheet: {sheet_name}")
        df = pd.read_excel(input_path, sheet_name=sheet_name)
        if df.empty:
            print(f"Warning: Empty sheet: {sheet_name}")
            continue

        records = extract_records_from_pair_sheet(df, sheet_name)
        if not records:
            print(f"Warning: No valid records found in {sheet_name}")
            continue

        entity_keys = ["disease", "symptom", "formula_herb"]
        N = len(records)
        df_counts = compute_global_df(records, entity_keys)

        # Record-level TF-IDF
        tfidf_rows = []
        for rec in records:
            rid = rec["record_id"]
            dyn = rec["dynasty"]

            for entity in set(rec["disease"]):
                tf = compute_tf(entity, rec["disease"])
                idf = compute_idf(entity, df_counts, N)
                tfidf_rows.append({
                    "record_id": rid,
                    "dynasty": dyn,
                    "entity": entity,
                    "entity_type": "Disease name",
                    "TF": tf,
                    "IDF": idf,
                    "TFIDF": tf * idf
                })

            for entity in set(rec["symptom"]):
                tf = compute_tf(entity, rec["symptom"])
                idf = compute_idf(entity, df_counts, N)
                tfidf_rows.append({
                    "record_id": rid,
                    "dynasty": dyn,
                    "entity": entity,
                    "entity_type": "Symptom",
                    "TF": tf,
                    "IDF": idf,
                    "TFIDF": tf * idf
                })

            for entity in set(rec["formula_herb"]):
                tf = compute_tf(entity, rec["formula_herb"])
                idf = compute_idf(entity, df_counts, N)
                tfidf_rows.append({
                    "record_id": rid,
                    "dynasty": dyn,
                    "entity": entity,
                    "entity_type": "Formula herb",
                    "TF": tf,
                    "IDF": idf,
                    "TFIDF": tf * idf
                })

        tfidf_df = pd.DataFrame(tfidf_rows)

        # Optional: export TF-IDF details for the whole triplet sheet
        tfidf_detail_path = os.path.join(
            output_dir,
            f"{safe_filename(sheet_name)}_TFIDF_detail.xlsx"
        )
        tfidf_df.to_excel(tfidf_detail_path, index=False)

        dynasties = sorted(tfidf_df["dynasty"].dropna().unique().tolist())

        for dynasty in dynasties:
            sub_tfidf = tfidf_df[tfidf_df["dynasty"] == dynasty].copy()
            if sub_tfidf.empty:
                continue

            # Node table
            node_df = build_node_table(sub_tfidf)

            # Edge accumulation
            edge_acc = defaultdict(float)
            co_count = defaultdict(int)

            dynasty_records = [r for r in records if r["dynasty"] == dynasty]

            for rec in dynasty_records:
                disease_vals = {}
                symptom_vals = {}
                herb_vals = {}

                for e in set(rec["disease"]):
                    tf = compute_tf(e, rec["disease"])
                    idf = compute_idf(e, df_counts, N)
                    disease_vals[e] = tf * idf

                for e in set(rec["symptom"]):
                    tf = compute_tf(e, rec["symptom"])
                    idf = compute_idf(e, df_counts, N)
                    symptom_vals[e] = tf * idf

                for e in set(rec["formula_herb"]):
                    tf = compute_tf(e, rec["formula_herb"])
                    idf = compute_idf(e, df_counts, N)
                    herb_vals[e] = tf * idf

                # Disease -> Symptom
                for d, wd in disease_vals.items():
                    for s, ws in symptom_vals.items():
                        if wd > 0 and ws > 0:
                            key = (d, s, "Disease name", "Symptom")
                            edge_acc[key] += math.sqrt(wd * ws)
                            co_count[key] += 1

                # Symptom -> Formula herb
                for s, ws in symptom_vals.items():
                    for h, wh in herb_vals.items():
                        if ws > 0 and wh > 0:
                            key = (s, h, "Symptom", "Formula herb")
                            edge_acc[key] += math.sqrt(ws * wh)
                            co_count[key] += 1

            edges = []
            for (a, b, t1, t2), w in edge_acc.items():
                edges.append({
                    "Pinyin1": to_pinyin(a),
                    "Pinyin2": to_pinyin(b),
                    "Type1": t1,
                    "Type2": t2,
                    "Weight": w,
                    "Cooccurrence": co_count[(a, b, t1, t2)]
                })

            edge_df = pd.DataFrame(edges)

            dynasty_safe = safe_filename(dynasty)
            sheet_safe = safe_filename(sheet_name)
            node_path = os.path.join(output_dir, f"{sheet_safe}_{dynasty_safe}_nodes.xlsx")
            edge_path = os.path.join(output_dir, f"{sheet_safe}_{dynasty_safe}_edges.xlsx")

            node_df.to_excel(node_path, index=False)
            edge_df.to_excel(edge_path, index=False)

            print(f"    {dynasty} exported: nodes={len(node_df)}, edges={len(edge_df)}")

    print("All triplet networks have been exported successfully.")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("====== Step 1: Data Classification ======")
    classify_case_data(input_path_raw, case_output_path)

    print("\n====== Step 2: Pairwise Network Construction ======")
    process_general_sheets(case_output_path, tuples_output_dir)

    print("\n====== Step 3: Triplet Network Construction ======")
    process_three_relation(case_output_path, triplets_output_dir)

    print("\nAll steps completed successfully.")