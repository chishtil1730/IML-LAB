import pandas as pd

# 1. PREPROCESSING
df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)
data = df.values.tolist()

split = int(len(data) * 0.8)
train, test = data[:split], data[split:]

# 2. GRADIENT BOOSTING LOGIC (Simplified)
# Start with the average of the labels (Initial Prediction)
avg_label = sum(r[-1] for r in train) / len(train)
learning_rate = 0.1
forest = []

# We want to fix the "Residuals" (Errors)
residuals = [r[-1] - avg_label for r in train]

# Build 3 "Fixer" rules (Stumps)
for _ in range(3):
    # Pick a feature to fix the error (e.g., index 1: Age)
    # A real XGBoost would search for the best feature; we'll keep it simple.
    feat_idx = 1
    split_val = 50.0  # Split at age 50

    # Calculate the average error for people above and below the split
    left_err = [residuals[i] for i in range(len(train)) if train[i][feat_idx] <= split_val]
    right_err = [residuals[i] for i in range(len(train)) if train[i][feat_idx] > split_val]

    val_left = sum(left_err) / len(left_err) if left_err else 0
    val_right = sum(right_err) / len(right_err) if right_err else 0

    # Save this "Fixer" tree
    forest.append((feat_idx, split_val, val_left, val_right))

    # Update residuals: "Subtract" the fix we just found so the next tree finds NEW errors
    for i in range(len(train)):
        fix = val_left if train[i][feat_idx] <= split_val else val_right
        residuals[i] -= learning_rate * fix


# 3. PREDICTION
def predict(row):
    prediction = avg_label
    for feat_idx, split_val, val_left, val_right in forest:
        # Add a small piece of each "Fixer" tree's advice
        fix = val_left if row[feat_idx] <= split_val else val_right
        prediction += learning_rate * fix
    return 1 if prediction >= 0.5 else 0


# 4. EVALUATION
actual = [r[-1] for r in test]
preds = [predict(r) for r in test]
correct = sum(1 for a, p in zip(actual, preds) if a == p)

print(f"Simple Boosting Accuracy: {correct / len(actual):.2%}")