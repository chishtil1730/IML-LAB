import numpy as np
import pandas as pd

#Removing empty values in csv
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns")
    df = df.dropna(subset=[df.columns[-1]])
    df.iloc[:, :-1] = df.iloc[:, :-1].fillna(df.iloc[:, :-1].median())
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    return X, y

# Normalising function
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std

def normalize_y(y):
    mean = np.mean(y)
    std = np.std(y)
    if std < 1e-8:
        std = 1.0
    return (y - mean) / std, mean, std

# Gradient Descent function
def gradient_descent(X, y, lr=0.01, epochs=3000, reg=0.001):
    m, n = X.shape
    X = np.c_[np.ones(m), X]
    theta = np.zeros(n + 1)
    prev_cost = float("inf")
    for i in range(epochs):
        preds = X @ theta
        errors = preds - y
        # L2 regularization (except bias)
        gradients = (X.T @ errors) / m
        gradients[1:] += reg * theta[1:]
        # Gradient clipping (KEY FIX)
        gradients = np.clip(gradients, -1, 1)
        theta -= lr * gradients
        cost = np.mean(errors ** 2)
        # Early stopping
        if abs(prev_cost - cost) < 1e-8:
            break
        prev_cost = cost
    return theta

#Prediction function
def predict(X, theta, y_mean, y_std):
    X = np.c_[np.ones(X.shape[0]), X]
    preds = X @ theta
    return preds * y_std + y_mean

if __name__ == "__main__":
    X, y = load_csv("../ML_Experiments/data_lab2.csv")
    X = normalize(X)
    y_scaled, y_mean, y_std = normalize_y(y)
    theta = gradient_descent(X, y_scaled)
    preds = predict(X, theta, y_mean, y_std)
    print("\nFinal theta:")
    print(theta)
    print("\nFirst 10 predictions:")
    print(preds[:10])