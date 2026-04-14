import pandas as pd

# 1. PREPROCESSING
df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)
data = df.values.tolist()


# 2. SIMPLEST "STUMP" MODEL
def get_stump_accuracy(train, test, feature_idx):
    # Training: Find most common outcome for smokers (1) vs non-smokers (0)
    group_1 = [r[-1] for r in train if r[feature_idx] == 1.0]
    group_0 = [r[-1] for r in train if r[feature_idx] == 0.0]

    # Get the "Majority Vote" for each group
    pred_1 = max(set(group_1), key=group_1.count) if group_1 else 0
    pred_0 = max(set(group_0), key=group_0.count) if group_0 else 0

    # Testing
    correct = 0
    for row in test:
        prediction = pred_1 if row[feature_idx] == 1.0 else pred_0
        if prediction == row[-1]:
            correct += 1
    return correct / len(test)


# 3. K-FOLD LOGIC (Simple 3-Fold)
k = 3
fold_size = len(data) // k
total_acc = 0

for i in range(k):
    # Define boundaries for the "Test" fold
    start, end = i * fold_size, (i + 1) * fold_size

    test_fold = data[start:end]
    train_fold = data[:start] + data[end:]

    # Run our simple model on this fold (using index 3: currentSmoker)
    acc = get_stump_accuracy(train_fold, test_fold, 3)
    total_acc += acc
    print(f"Fold {i + 1} Accuracy: {acc:.2%}")

print(f"\nAverage Cross-Validation Accuracy: {total_acc / k:.2%}")