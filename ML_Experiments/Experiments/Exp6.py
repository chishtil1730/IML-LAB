# ==========================================
# RANDOM FOREST CLASSIFICATION - FRAMINGHAM
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

STUDENT_NAME = "24BCA7027 SHAIK-BARAKH-CHISHTI"

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\LENOVO\Dsa\IML_LAB\ML_Experiments\data_sets\framingham_cleaned.csv")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# -------------------------------
# 2. Features & Target
# -------------------------------
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

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
# 4. Random Forest Model
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# -------------------------------
# 6. Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", cm)

# ==========================================
# VISUALIZATION SECTION
# ==========================================

# 1️⃣ Box Plot
plt.figure(figsize=(12, 6))
df.drop("TenYearCHD", axis=1).boxplot()
plt.title(f"Box Plot of Input Features — {STUDENT_NAME}")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 2️⃣ Violin Plot (Age vs CHD)
plt.figure(figsize=(7, 5))
sns.violinplot(x="TenYearCHD", y="age", data=df, inner="quartile")
plt.title(f"Violin Plot: Age vs CHD — {STUDENT_NAME}")
plt.xlabel("TenYearCHD")
plt.ylabel("Age")
plt.show()

# 3️⃣ Hexbin Plot (SysBP vs BMI)
plt.figure(figsize=(6, 5))
plt.hexbin(df["sysBP"], df["BMI"], gridsize=30, cmap="Blues")
plt.colorbar(label="Density")
plt.xlabel("sysBP")
plt.ylabel("BMI")
plt.title(f"Hexbin Plot: sysBP vs BMI — {STUDENT_NAME}")
plt.show()

# 4️⃣ Raincloud Plot (Glucose vs CHD)
plt.figure(figsize=(7, 5))
sns.violinplot(x="TenYearCHD", y="glucose", data=df, inner=None, color="lightgray")
sns.boxplot(x="TenYearCHD", y="glucose", data=df, width=0.2)
sns.stripplot(x="TenYearCHD", y="glucose", data=df, color="black", alpha=0.4)
plt.title(f"Raincloud Plot: Glucose vs CHD — {STUDENT_NAME}")
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
plt.title(f"Random Forest Performance Radar — {STUDENT_NAME}")
plt.show()

# 8️⃣ Bar Graph
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
scores = [accuracy, precision, recall, f1]

plt.figure(figsize=(7, 5))
bars = plt.bar(metrics, scores)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"Random Forest Performance — {STUDENT_NAME}")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 0.02,
             f"{height:.2f}",
             ha="center")

plt.show()

# -------------------------------
# 9. Feature Importance Plot
# -------------------------------
importances = rf.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title(f"Feature Importance — {STUDENT_NAME}")
plt.show()

print("\nTop 5 Important Features:")
print(feat_df.head())
