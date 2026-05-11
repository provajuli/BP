import os
import shutil
import pandas as pd
import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))
RAW_DIRECTORY = os.path.join(PATH, "raw")
PROCESSED_DIRECTORY = os.path.join(PATH, "processed")
EXCLUDED_DIRECTORY = os.path.join(PATH, "excluded")

REPORT_FILE = os.path.join(PATH, "processing_report.csv")

DROP_FILE_THRESHOLD = 0.30
STUCK_VALUE_THRESHOLD = 0.50
EDGE_RESPONSE_THRESHOLD = 0.30

BETWEEN_TOL = 5.0 


def validate_dataframe(df: pd.DataFrame, filename: str):

    if df.empty:
        return False, df, {}

    required_cols = {"sizeA", "sizeB", "sizeC"}
    if not required_cols.issubset(df.columns):
        return False, df, {}

    df = df.copy()

    for col in ["sizeA", "sizeB", "sizeC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["sizeA", "sizeB", "sizeC"])

    if df.empty:
        return False, df, {}

    mask_range = (
        df["sizeA"].between(1, 100) &
        df["sizeB"].between(1, 100) &
        df["sizeC"].between(1, 100)
    )

    df_clean = df.loc[mask_range].copy()

    if df_clean.empty:
        return False, df_clean, {}

    range_invalid_rate = 1.0 - mask_range.mean()

    lower = df_clean[["sizeA", "sizeC"]].min(axis=1) - BETWEEN_TOL
    upper = df_clean[["sizeA", "sizeC"]].max(axis=1) + BETWEEN_TOL

    mask_between = (df_clean["sizeB"] >= lower) & (df_clean["sizeB"] <= upper)
    between_violation_rate = 1.0 - mask_between.mean()

    edge_rate = df_clean["sizeB"].isin([1, 100]).mean()

    file_invalid = edge_rate > EDGE_RESPONSE_THRESHOLD

    report = {
        "file": filename,
        "n_trials": len(df_clean),
        "range_invalid_rate": round(range_invalid_rate, 3),
        "between_violation_rate": round(between_violation_rate, 3),
        "edge_rate": round(edge_rate, 3),
        "kept": not file_invalid
    }

    return (not file_invalid), df_clean, report


def sort_files():
    os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
    os.makedirs(EXCLUDED_DIRECTORY, exist_ok=True)

    reports = []

    for filename in os.listdir(RAW_DIRECTORY):
        if not filename.endswith(".csv"):
            continue

        raw_path = os.path.join(RAW_DIRECTORY, filename)
        processed_path = os.path.join(PROCESSED_DIRECTORY, filename)
        excluded_path = os.path.join(EXCLUDED_DIRECTORY, filename)

        try:
            df = pd.read_csv(raw_path)
        except Exception as e:
            print(f"[WARN] {filename}: nelze precist ({e})")
            shutil.move(raw_path, excluded_path)
            continue

        is_valid, clean_df, report = validate_dataframe(df, filename)
        reports.append(report)

        if not is_valid:
            print(f"[DROP] {filename} | edge={report['edge_rate']}")
            shutil.move(raw_path, excluded_path)
        else:
            print(
                f"[OK] {filename} | kept {len(clean_df)}/{report['n_trials']} | "
                f"between violations={report['between_violation_rate']}"
            )
            clean_df.to_csv(processed_path, index=False)
            os.remove(raw_path)

    if reports:
        df_report = pd.DataFrame(reports)
        df_report.to_csv(REPORT_FILE, index=False)
        print(f"\n[INFO] Report uložen: {REPORT_FILE}")

def merge_valid():
    merged_path = os.path.join(PATH, "merged_valid_data.csv")

    files = [f for f in os.listdir(PROCESSED_DIRECTORY) if f.endswith(".csv")]

    if not files:
        print("[INFO] Žádná validní data k sloučení.")
        return

    with open(merged_path, "w", encoding="utf-8") as out:
        first = True

        for filename in files:
            path = os.path.join(PROCESSED_DIRECTORY, filename)

            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"[WARN] {filename}: nelze načíst ({e})")
                continue

            out.write(f"# {filename}\n")

            df.to_csv(out, index=False, header=first)

            first = False

    print(f"[INFO] Sloučená data uložena: {merged_path}")
    print(f"[INFO] Počet validních souborů: {len(files)}")
    print(f"[INFO] Počet nevalidních souborů: {len(os.listdir(EXCLUDED_DIRECTORY))}")
    print(f"[INFO] Celkový počet řádků ve všech souborech: {sum(pd.read_csv(os.path.join(PROCESSED_DIRECTORY, f)).shape[0] for f in files) + sum(pd.read_csv(os.path.join(EXCLUDED_DIRECTORY, f)).shape[0] for f in os.listdir(EXCLUDED_DIRECTORY) if f.endswith('.csv'))}")
    print(f"[INFO] Celkový počet řádků v merged souboru: {sum(pd.read_csv(os.path.join(PROCESSED_DIRECTORY, f)).shape[0] for f in files)}")


if __name__ == "__main__":
    sort_files()
    merge_valid()