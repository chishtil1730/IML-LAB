import pandas as pd
import numpy as np

# 1. Load and prepare data
df = pd.read_csv('data_lab2.csv')
y = df['y'].values.reshape(-1, 1)
X = df[['x1', 'x2', 'x3']].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_b = np.c_[np.ones((len(X), 1)), X]
learning_rate = 0.01
n_iterations = 1000
m = len(y)
theta = np.random.randn(4, 1)

# 3. Gradient Descent Loop
for i in range(n_iterations):
    y_pred = X_b.dot(theta)
    gradients = (1 / m) * X_b.T.dot(y_pred - y)
    theta = theta - learning_rate * gradients

final_pred = X_b.dot(theta)

# Calculate MSE
mse = ((y - final_pred) ** 2).mean()

# Calculate R-squared (R^2)
ssr = ((y - final_pred) ** 2).sum()
sst = ((y - y.mean()) ** 2).sum()
r2 = 1 - (ssr / sst)
print("24BCA7027 SHAIK BARAKH CHISHTI")
print(f"Final Weights: {theta.flatten()}")
print(f"MSE: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
