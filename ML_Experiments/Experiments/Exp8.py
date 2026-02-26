import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, roc_curve, auc
)
from sklearn.neural_network import MLPClassifier

# -------------------------------
# 1. Configuration & Dataset Load
# -------------------------------
STUDENT_NAME = "24BCA7027 SHAIK-BARAKH-CHISHTI"
# Ensure the file path matches your local file
df = pd.read_csv("ML_Experiments/data_sets/Gaming and Mental Health - Gaming and Mental Health.csv")

# Cleaning column names (handling duplicate 'game_genre' columns)
new_cols = []
genre_count = 0
for col in df.columns:
    if 'game_genre' in col:
        new_cols.append('game_genre' if genre_count == 0 else 'game_title')
        genre_count += 1
    else:
        new_cols.append(col)
df.columns = [c.strip() for c in new_cols]

# -------------------------------
# 2. Preprocessing (Imputation & Encoding)
# -------------------------------
df['grades_gpa'] = df['grades_gpa'].fillna(df['grades_gpa'].median())
df['work_productivity_score'] = df['work_productivity_score'].fillna(df['work_productivity_score'].median())

# Encode categorical/boolean features to numeric for the Neural Network
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop("gaming_addiction_risk_level", axis=1)
y = df["gaming_addiction_risk_level"]

# -------------------------------
# 3. Train-Test Split & Feature Scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (Mean=0, Std=1) - Essential for Backpropagation efficiency
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. ANN Development using Backpropagation
# -------------------------------
# The MLPClassifier uses the Backpropagation algorithm to update weights
ann_backprop = MLPClassifier(
    hidden_layer_sizes=(64, 32), # Two hidden layers
    activation='relu',           # Activation function
    solver='adam',               # Optimizer that uses backpropagation gradients
    learning_rate_init=0.01,     # Step size for weight updates
    max_iter=500,                # Iterations (epochs)
    random_state=42
)

# Training the model (This step executes the Backpropagation process)
ann_backprop.fit(X_train_scaled, y_train)

# -------------------------------
# 5. Predictions & Evaluation
# -------------------------------
y_pred = ann_backprop.predict(X_test_scaled)
y_prob = ann_backprop.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nBACKPROPAGATION ANN RESULTS — {STUDENT_NAME}")
print(f"Accuracy:  {accuracy:.4f} | Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f} | F1-score:  {f1:.4f}")
print(f"MSE:       {mse:.4f} | RMSE:      {rmse:.4f}")
print(f"R2 Score:  {r2:.4f}")

# ===============================
# VISUALIZATION SECTION
# ===============================

# 1. Box Plot
plt.figure(figsize=(12, 6))
X.boxplot()
plt.title(f"Box Plot of Features — {STUDENT_NAME}")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('1_box_plot.png')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title(f"Feature Correlation Heatmap — {STUDENT_NAME}")
plt.tight_layout()
plt.savefig('2_correlation_heatmap.png')
plt.close()

# 3. Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Backprop) — {STUDENT_NAME}")
plt.tight_layout()
plt.savefig('3_confusion_matrix.png')
plt.close()

# 4. ROC Curve (Multi-class)
n_classes = len(np.unique(y))
y_test_bin = label_binarize(y_test, classes=np.unique(y))
plt.figure(figsize=(7, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {STUDENT_NAME}")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('4_roc_curve.png')
plt.close()

# 5. Radar Plot
labels = ["Accuracy", "Precision", "Recall", "F1-score"]
stats = [accuracy, precision, recall, f1]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
stats += stats[:1]; angles += angles[:1]
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, color='red', linewidth=2)
ax.fill(angles, stats, color='red', alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title(f"Model Radar Plot — {STUDENT_NAME}")
plt.savefig('5_radar_plot.png')
plt.close()

# 6. Bar Graph
metrics_list = ["Accuracy", "Precision", "Recall", "F1-score", "RMSE", "R2"]
scores_list = [accuracy, precision, recall, f1, rmse, r2]
plt.figure(figsize=(10, 5))
bars = plt.bar(metrics_list, scores_list, color='skyblue')
plt.title(f"Performance Metrics — {STUDENT_NAME}")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig('6_performance_bar_graph.png')
plt.close()

# 7. Hexbin Plot
plt.figure(figsize=(8, 6))
plt.hexbin(df['daily_gaming_hours'], df['sleep_hours'], gridsize=20, cmap='YlGnBu')
plt.colorbar(label='Density')
plt.xlabel('Daily Gaming Hours')
plt.ylabel('Sleep Hours')
plt.title(f"Hexbin: Gaming vs Sleep — {STUDENT_NAME}")
plt.tight_layout()
plt.savefig('7_hexbin_plot.png')
plt.close()

print("\nAll tasks complete. 7 plots saved as PNGs.")