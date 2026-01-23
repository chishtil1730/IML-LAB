import pandas as pd
df = pd.read_csv("../ML_Experiments/data_sets/EnterpriseSurvey.csv")
numeric_cols = df.select_dtypes(include='number').columns
df_norm = df.copy()
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    if col_max == col_min:
        df_norm[col] = 0
    else:
        df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        df_norm[col] = df_norm[col] * 2 - 1
print("Means after normalization:\n", df_norm[numeric_cols].mean())
print("\nStandard deviations after normalization:\n", df_norm[numeric_cols].std())
df_norm.to_csv("INDIAVIX_normalized_-1_1.csv", index=False)