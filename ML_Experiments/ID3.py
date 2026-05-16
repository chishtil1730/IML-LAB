import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. PREPROCESSING
df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)
data = df.values.tolist()

split = int(len(data) * 0.8)
train, test = data[:split], data[split:]

# 2. "TRAINING" (Find the most common outcome for a simple rule)
# Let's pick a simple rule: "If you smoke (column 3), are you more likely to have heart disease?"
smokers = [row for row in train if row[3] == 1.0]
non_smokers = [row for row in train if row[3] == 0.0]

# Find the most common outcome (Mode) for each group
def get_mode(rows):
    labels = [r[-1] for r in rows]
    return max(set(labels), key=labels.count) if labels else 0

mode_smoker = get_mode(smokers)
mode_non_smoker = get_mode(non_smokers)

# 3. PREDICTION
def predict(row):
    # Rule: If smoker, predict what most smokers are. Else, predict for non-smokers.
    if row[3] == 1.0:
        return mode_smoker
    else:
        return mode_non_smoker

# 4. EVALUATION
threshold =100
actual = [r[-1] for r in test]
preds = [predict(r) for r in test]

for i in range(len(actual)):
    if actual[i]<threshold:
        actual[i] = 0
    else:
        actual[i]=1

for i in range(len(preds)):
    if preds[i]<threshold:
        preds[i] = 0
    else:
        preds[i]=1

cm = confusion_matrix(actual,preds)

plt.figure(figsize=(6,9))

sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')

plt.show()

correct = sum(1 for a, p in zip(actual, preds) if a == p)
print(f"Simplest Tree Accuracy: {correct/len(actual):.2%}")