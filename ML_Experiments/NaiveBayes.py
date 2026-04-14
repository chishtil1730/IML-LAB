import pandas as pd

# 1. PREPROCESSING
df = pd.read_csv('data_sets/framingham - framingham.csv')
df.columns = df.columns.str.strip()
df.fillna(df.mean().round(2), inplace=True)

data = df.values.tolist()
split = int(len(data) * 0.8)
train_data, test_data = data[:split], data[split:]

# 2. TRAINING (Calculate P(Class))
total = len(train_data)
# Count how many people are sick (1) vs healthy (0)
count_1 = sum(1 for row in train_data if row[-1] == 1)
count_0 = total - count_1

prob_1 = count_1 / total
prob_0 = count_0 / total


# 3. PREDICTION (The "Naive" part)
def predict(row):
    # In the simplest form, we look at the 'prior' probabilities.
    # To be more accurate, we check if specific features lean towards 1 or 0.
    score_1 = prob_1
    score_0 = prob_0

    # We compare the row's values to the averages of each class
    # (Simplified: if Age is high, it slightly increases score_1)
    for i in range(len(row) - 1):
        if row[i] > df.iloc[:, i].mean():
            score_1 *= 1.1  # Arbitrary weight for demonstration
        else:
            score_0 *= 1.1

    return 1 if score_1 > score_0 else 0


# 4. EXECUTION
actual = [r[-1] for r in test_data]
preds = [predict(r) for r in test_data]

# 5. ACCURACY
correct = sum(1 for a, p in zip(actual, preds) if a == p)
print(f"Simple Bayes Accuracy: {correct / len(actual):.2%}")