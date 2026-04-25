
import pandas as pd

input_file = "data/data_processing_input/filtered_results.csv"
output_file = "data/data_processing_input/filtered_results_no_close_AC.csv"

df = pd.read_csv(input_file)

def remove_close_AC(df, min_dist=10.0):
    middle = (df["sizeA"] + df["sizeC"]) / 2
    dist = abs(df["sizeC"] - df["sizeA"])
    return df[dist >= min_dist].copy()

df_cleaned = remove_close_AC(df, min_dist=10.0)
output_file = "data/data_processing_input/filtered_results_no_close_AC.csv"

df_cleaned.to_csv(output_file, index=False)
print(f"Hotovo. Původní počet řádků: {len(df)}, nový počet řádků: {len(df_cleaned)}")
print(f"Odstraněno {len(df) - len(df_cleaned)} řádků, které měly vzdálenost mezi A a C menší než 10, což je {((len(df) - len(df_cleaned)) / len(df) * 100):.2f} % původních dat.")
print(f"Uloženo do: {output_file}")