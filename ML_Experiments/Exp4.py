# ===============================
# KNN CLASSIFICATION - DIABETES
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

STUDENT_NAME = "24BCA7027 SHAIK BARAKH CHISHTI"

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_excel("ML_Experiments/diabetes.xlsx")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# -------------------------------
# 2. Features & Target
# -------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5. KNN Model
# -------------------------------
k = 5
knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
knn.fit(X_train_scaled, y_train)

# -------------------------------
# 6. Predictions
# -------------------------------
y_pred = knn.predict(X_test_scaled)
y_prob = knn.predict_proba(X_test_scaled)[:, 1]

# -------------------------------
# 7. Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nKNN RESULTS (K = 5)")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", cm)

# ===============================
# VISUALIZATION SECTION
# ===============================

# 1️⃣ Box Plot
plt.figure(figsize=(12, 6))
df.drop("Outcome", axis=1).boxplot()
plt.title(f"Box Plot of Input Features — {STUDENT_NAME}")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 2️⃣ Violin Plot
plt.figure(figsize=(7, 5))
sns.violinplot(x="Outcome", y="BMI", data=df, inner="quartile")
plt.title(f"Violin Plot: BMI vs Outcome — {STUDENT_NAME}")
plt.xlabel("Outcome")
plt.ylabel("BMI")
plt.show()

# 3️⃣ Hexbin Plot
plt.figure(figsize=(6, 5))
plt.hexbin(df["Glucose"], df["BMI"], gridsize=30, cmap="Blues")
plt.colorbar(label="Density")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title(f"Hexbin Plot: Glucose vs BMI — {STUDENT_NAME}")
plt.show()

# 4️⃣ Raincloud Plot
plt.figure(figsize=(7, 5))
sns.violinplot(x="Outcome", y="Glucose", data=df, inner=None, color="lightgray")
sns.boxplot(x="Outcome", y="Glucose", data=df, width=0.2)
sns.stripplot(x="Outcome", y="Glucose", data=df, color="black", alpha=0.4)
plt.title(f"Raincloud Plot: Glucose vs Outcome — {STUDENT_NAME}")
plt.show()

# 5️⃣ Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {STUDENT_NAME}")
plt.show()

# 6️⃣ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {STUDENT_NAME}")
plt.legend()
plt.grid(True)
plt.show()

# 7️⃣ Radar Plot
labels = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [accuracy, precision, recall, f1]
values += values[:1]

angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title(f"Model Performance Radar Plot — {STUDENT_NAME}")
plt.show()

# 8️⃣ Bar Graph
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
scores = [accuracy, precision, recall, f1]

plt.figure(figsize=(7, 5))
bars = plt.bar(metrics, scores)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"Bar Graph of Model Performance — {STUDENT_NAME}")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 0.02,
             f"{height:.2f}",
             ha="center")

plt.show()

# -------------------------------
# 9. Elbow Method
# -------------------------------
error_rates = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rates.append(np.mean(pred_k != y_test))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), error_rates, marker="o")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.title(f"Elbow Method for Optimal K — {STUDENT_NAME}")
plt.grid(True)
plt.show()

best_k = error_rates.index(min(error_rates)) + 1
print("\nBest K Value:", best_k)
print("Minimum Error Rate:", min(error_rates))
