import pandas as pd
import random

# 1. PREPROCESSING
df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)
data = df.values.tolist()

split = int(len(data) * 0.8)
train, test = data[:split], data[split:]


# 2. SIMPLEST TREE (A "Stump")
def create_stump(train_sample):
    # Pick one random feature to look at
    feature_idx = random.randint(0, len(train_sample[0]) - 2)

    # Simple rule: What's the most common result for '1' vs '0' in this feature?
    group_1 = [r[-1] for r in train_sample if r[feature_idx] == 1.0]
    group_0 = [r[-1] for r in train_sample if r[feature_idx] == 0.0]

    # Get majority vote for each (default to 0 if group is empty)
    pred_1 = max(set(group_1), key=group_1.count) if group_1 else 0
    pred_0 = max(set(group_0), key=group_0.count) if group_0 else 0

    return (feature_idx, pred_1, pred_0)


# 3. FOREST LOGIC (Bagging)
forest = []
num_trees = 5

for _ in range(num_trees):
    # Bootstrapping: Create a random sample of the training data
    sample = [random.choice(train) for _ in range(len(train) // 10)]  # Smaller sample for speed
    forest.append(create_stump(sample))


# 4. PREDICTION (Voting)
def predict_forest(row, forest):
    votes = []
    for feat_idx, p1, p0 in forest:
        # Each tree votes based on its specific feature
        vote = p1 if row[feat_idx] == 1.0 else p0
        votes.append(vote)
    # Majority wins
    return max(set(votes), key=votes.count)


# 5. EXECUTION
actual = [r[-1] for r in test]
preds = [predict_forest(r, forest) for r in test]

correct = sum(1 for a, p in zip(actual, preds) if a == p)
print(f"Simple Random Forest Accuracy: {correct / len(actual):.2%}")