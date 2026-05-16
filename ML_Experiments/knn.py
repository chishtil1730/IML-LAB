import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ML_Experiments.data_handling import get_dist

df = pd.read_csv('data_sets/framingham - framingham.csv')
df.columns = df.columns.str.strip()
df.fillna(df.mean(), inplace=True)

all_data = df.values.tolist()
split = int(len(all_data) * 0.8)
training_data = all_data[:split]
test_data = all_data[split:]

def get_distance(r1, r2):
    dis = 0.0
    for i in range(len(r1) - 1):
        dis += (r1[i] - r2[i])**2
    return math.sqrt(dis)

def get_pred(train, test_row, k):
    dist = []
    for row in train:
        distance = get_distance(row, test_row)
        dist.append((row, distance))
    dist.sort(key=lambda x: x[1])
    neighbours = [dist[m][0][-1] for m in range(k)]
    return max(set(neighbours), key=neighbours.count)

actual_res = []
pred_res = []

for test_row in test_data:
    actual_res.append(int(test_row[-1]))
    pred_res.append(int(get_pred(training_data, test_row, 5)))

cm = confusion_matrix(actual_res, pred_res)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['No CHD', 'CHD'],
            yticklabels=['No CHD', 'CHD'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()

tp = tn = fp = fn = 0
for a, p in zip(actual_res, pred_res):
    if a == 1 and p == 1: tp += 1
    elif a == 0 and p == 1: fp += 1
    elif a == 1 and p == 0: fn += 1
    elif a == 0 and p == 0: tn += 1

accuracy_val = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0

print(f"Accuracy: {accuracy_val:.2%}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")