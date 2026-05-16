import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)

# Keep labels for the confusion matrix
labels = df['TenYearCHD'].tolist()
data = df[['age', 'BMI']].values.tolist()

k = 2 # Set k=2 to match binary labels (No CHD vs CHD)
centroids = random.sample(data, k)

for _ in range(10):
    clusters = [[] for _ in range(k)]
    cluster_indices = [[] for _ in range(k)]

    for idx, row in enumerate(data):
        distances = [math.sqrt((row[0]-c[0])**2 + (row[1]-c[1])**2) for c in centroids]
        closest_idx = distances.index(min(distances))
        clusters[closest_idx].append(row)
        cluster_indices[closest_idx].append(idx)

    for i in range(k):
        if clusters[i]:
            avg_age = sum(p[0] for p in clusters[i]) / len(clusters[i])
            avg_bmi = sum(p[1] for p in clusters[i]) / len(clusters[i])
            centroids[i] = [avg_age, avg_bmi]

# Map clusters to actual labels based on majority vote
pred_labels = [0] * len(data)
for i in range(k):
    if cluster_indices[i]:
        actuals_in_cluster = [labels[idx] for idx in cluster_indices[i]]
        majority_label = max(set(actuals_in_cluster), key=actuals_in_cluster.count)
        for idx in cluster_indices[i]:
            pred_labels[idx] = majority_label

cm = confusion_matrix(labels, pred_labels)

plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix')
plt.show()

tp = sum(1 for a, p in zip(labels, pred_labels) if a == 1 and p == 1)
tn = sum(1 for a, p in zip(labels, pred_labels) if a == 0 and p == 0)
acc = (tp + tn) / len(labels)
print(f"Clustering Alignment Accuracy: {acc:.2%}")