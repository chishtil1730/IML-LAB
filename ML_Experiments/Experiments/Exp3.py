import pandas as pd

# 1. Load the dataset
df = pd.read_csv('../framingham - framingham.csv')

# 2. Identify columns with missing values (NaN or "NA")
# pandas automatically treats "NA" in CSVs as NaN if not specified otherwise
columns_with_na = df.columns[df.isnull().any()].tolist()

print(f"Columns with missing values: {columns_with_na}")

# 3. Preprocessing loop
for col in columns_with_na:
    # Calculate Mean and Standard Deviation
    mean_val = df[col].mean()
    std_val = df[col].std()

    # Define the "good value" for replacement (Mean is the standard choice)
    good_value = mean_val

    print(f"Processing {col}: Mean={mean_val:.2f}, Std={std_val:.2f}. Replacing NAs with {good_value:.2f}")

    # 4. Replace all NA values in that column with the calculated good value
    df[col] = df[col].fillna(good_value)

# 5. Save the cleaned CSV
df.to_csv('framingham_cleaned.csv', index=False)
print("Preprocessing complete. Cleaned file saved as 'framingham_cleaned.csv'.")