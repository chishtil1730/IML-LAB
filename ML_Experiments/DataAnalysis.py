import pandas as pd
df = pd.read_csv("EnterpriseSurvey.csv")
df.columns = df.columns.str.strip()
col = 'value'
if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=[col], inplace=True)
    col_min = df[col].min()
    col_max = df[col].max()
    col_mean = df[col].mean()
    col_std = df[col].std()
    print(f"Min {col}: ", col_min)
    print(f"Max {col}: ", col_max)
    print(f"Mean {col}: ", col_mean)
    print(f"Std Dev {col}: ", col_std)
