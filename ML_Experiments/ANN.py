import pandas as pd
import math
import random

# 1. PREPROCESSING
df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)

# Keep it super simple: only use Age and BMI
data = df[['age', 'BMI', 'TenYearCHD']].values.tolist()

# 2. RANDOM WEIGHTS (The "Guessing" setup)
# We have 2 inputs (Age, BMI), so we need 2 weights and 1 bias
w1, w2 = random.uniform(-1, 1), random.uniform(-1, 1)
bias = random.uniform(-1, 1)


# 3. FORWARD PROPAGATION
def predict(row):
    # z = (age * w1) + (bmi * w2) + bias
    z = (row[0] * w1) + (row[1] * w2) + bias

    # Activation: Sigmoid squashes the number between 0 and 1
    probability = 1 / (1 + math.exp(-z))

    return 1 if probability >= 0.5 else 0


# 4. EXECUTION
actual = [r[-1] for r in data[:100]]
preds = [predict(r) for r in data[:100]]

correct = sum(1 for a, p in zip(actual, preds) if a == p)
print(f"Simple Perceptron Accuracy: {correct / len(actual):.2%}")