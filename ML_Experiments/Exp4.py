import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#loading the data set.
df = pd.read_excel("ML_Experiments/diabetes.xlsx")

print(df.head())
print(df.info())

#Splitting the features:
X = df.drop("Outcome", axis=1)   # Features
y = df["Outcome"]               # Target


#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Training KNN
k = 5  # start with 5 neighbors

knn = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean"
)

knn.fit(X_train_scaled, y_train)


#Prediction & Evaluation
y_pred = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Elbow method for it.
error_rates = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    error_rates.append(np.mean(pred != y_test))

import matplotlib.pyplot as plt
plt.plot(range(1, 21), error_rates, marker="o")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.title("Elbow Method for KNN")
plt.show()
