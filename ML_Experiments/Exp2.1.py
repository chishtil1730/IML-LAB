import pandas as pd
import numpy as np
df = pd.read_csv('data_lab2.csv')
y = df['y'].values
X = df[['x1', 'x2', 'x3']].values
X_b = np.c_[np.ones((len(X), 1)), X]  # Add bias term (x0 = 1)
# 3. Normal Equation: theta = (X^T * X)^-1 * X^T * y
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# 4. Calculate Predicted Values (y_hat)
y_pred = X_b.dot(theta)
# 5. Calculate MSE
mse = ((y - y_pred) ** 2).mean()
# 6. Calculate R-squared (R²)
ssr = ((y - y_pred) ** 2).sum()
sst = ((y - y.mean()) ** 2).sum()
r2 = 1 - (ssr / sst)
print("24BCA7027 SHAIK BARAKH CHISHTI")
print(f"Coefficients (Intercept, x1, x2, x3): {theta}")
print(f"MSE: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
