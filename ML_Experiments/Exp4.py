# ===============================
# KNN CLASSIFICATION - DIABETES
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
# 4. Feature Scaling (MANDATORY)
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------
# 5. KNN Model (Initial K)
# -------------------------------
k = 5

knn = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean"
)

knn.fit(X_train_scaled, y_train)


# -------------------------------
# 6. Predictions
# -------------------------------
y_pred = knn.predict(X_test_scaled)


# -------------------------------
# 7. Evaluation
# -------------------------------
print("\nKNN RESULTS (K = 5)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# -------------------------------
# 8. Finding Best K (Elbow Method)
# -------------------------------
error_rates = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rates.append(np.mean(pred_k != y_test))


# -------------------------------
# 9. Plot Error vs K
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), error_rates, marker='o')
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.show()


# -------------------------------
# 10. Best K
# -------------------------------
best_k = error_rates.index(min(error_rates)) + 1
print("\nBest K Value:", best_k)
print("Minimum Error Rate:", min(error_rates))
