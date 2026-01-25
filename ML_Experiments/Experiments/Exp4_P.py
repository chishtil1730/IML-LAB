import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

NAME = "24BCA8144 PARDHIV VEER"

data = pd.read_excel("ML_Experiments/data_sets/diabetes.xlsx")

features = data.iloc[:, :-1]
target = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=21, stratify=target
)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

pred = model.predict(x_test)
prob = model.predict_proba(x_test)[:, 1]

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

matrix = confusion_matrix(y_test, pred)

print("STUDENT:", NAME)
print("\nAccuracy:", acc)
print("\nReport:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens")
plt.title(f"Confusion Matrix — {NAME}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, _ = roc_curve(y_test, prob)
roc_score = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title(f"ROC Curve — {NAME}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [acc, prec, rec, f1]

plt.figure(figsize=(7, 5))
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title(f"Performance Metrics — {NAME}")
plt.ylabel("Score")
plt.show()

angles = np.linspace(0, 2 * math.pi, len(metrics), endpoint=False).tolist()
values_cycle = values + values[:1]
angles += angles[:1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values_cycle)
ax.fill(angles, values_cycle, alpha=0.3)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
plt.title(f"Radar Chart — {NAME}")
plt.show()

plt.figure(figsize=(12, 6))
data.drop("Outcome", axis=1).boxplot()
plt.xticks(rotation=45)
plt.title(f"Feature Distribution — {NAME}")
plt.show()

plt.figure(figsize=(7, 5))
sns.violinplot(x="Outcome", y="BMI", data=data)
plt.title(f"BMI vs Outcome — {NAME}")
plt.show()

errors = []

for i in range(1, 21):
    temp_model = KNeighborsClassifier(n_neighbors=i)
    temp_model.fit(x_train, y_train)
    temp_pred = temp_model.predict(x_test)
    errors.append(np.mean(temp_pred != y_test))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), errors, marker="o")
plt.title(f"K Optimization Curve — {NAME}")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.show()

print("\nOptimal K:", errors.index(min(errors)) + 1)
print("Lowest Error:", min(errors))
