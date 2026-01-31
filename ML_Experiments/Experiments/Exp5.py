import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    roc_curve,
    auc
)

STUDENT_NAME = "24BCA7027 SHAIK-BARAKH-CHISHTI"

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("ML_Experiments/data_sets/weather_data.csv")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# -------------------------------
# 2. Encode Categorical Features
# -------------------------------
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# -------------------------------
# 3. Features & Target
# -------------------------------
X = df.drop("PlayTennis", axis=1)
y = df["PlayTennis"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 5. Decision Tree Model (ID3)
# -------------------------------
dt = DecisionTreeClassifier(
    criterion="entropy",   # ID3 uses entropy
    random_state=42
)

dt.fit(X_train, y_train)

# -------------------------------
# 6. Predictions
# -------------------------------
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]

# -------------------------------
# 7. Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

cm = confusion_matrix(y_test, y_pred)

print("\nDECISION TREE RESULTS (ID3)")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("MSE:", mse)
print("RMSE:", rmse)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", cm)

# ===============================
# VISUALIZATION SECTION
# ===============================


# -------------------------------
# Box Plot
# -------------------------------
plt.figure(figsize=(10, 6))
df.drop("PlayTennis", axis=1).boxplot()
plt.title(f"Box Plot of Weather Features — {STUDENT_NAME}")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -------------------------------
# Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title(f"Feature Correlation Heatmap — {STUDENT_NAME}")
plt.show()


# 1️⃣ Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {STUDENT_NAME}")
plt.show()

# 2️⃣ ROC Curve
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

# 3️⃣ Decision Tree Visualization
plt.figure(figsize=(16, 8))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)
plt.title(f"Decision Tree using ID3 — {STUDENT_NAME}")
plt.show()

# 4️⃣ Radar Plot
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

# 5️⃣ Bar Graph
metrics = ["Accuracy", "Precision", "Recall", "F1-score", "RMSE"]
scores = [accuracy, precision, recall, f1, rmse]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, scores)
plt.ylabel("Score")
plt.title(f"Model Performance Metrics — {STUDENT_NAME}")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f"{height:.3f}",
             ha="center",
             va="bottom")

plt.show()
