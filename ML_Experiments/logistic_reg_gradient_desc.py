import pandas as pd
import math


df = pd.read_csv('data_sets/framingham - framingham.csv')
df.columns = df.columns.str.strip()

df = (df - df.min()) / (df.max() - df.min())

data = df.values.tolist()

split = int(len(data)*0.8)

train_dat = data[:split]
test_dat = data[split:]

def sigmoid(x):
    return 1/(1+math.exp(-x))

def training(train, lr = 0.01, epochs=100):
    no_of_features = len(train[0])-1
    weights = [0.0]*no_of_features
    bias = 0.0

    for _ in range(epochs):
        for row in train:
            z = sum(weights[i]*row[i] for i in range(no_of_features))+bias
            prediction = sigmoid(z)

            actual = row[-1]
            error = prediction-actual

            for i in range(no_of_features):
                weights[i] -= lr*error*row[i]
            bias -= lr*error

    return weights,bias

def predict_logistic(row, weights, bias):
    z = sum(weights[i]*row[i] for i in range(len(weights))) + bias
    return 1 if sigmoid(z)>=0.5 else 0

weights, bias =training(train_dat)

actual_res = []
pred_res = []

for row in test_dat:
    actual_res.append(row[-1])
    pred_res.append(predict_logistic(row, weights, bias))

# 4. Metrics (Same logic as your code)
tp = tn = fp = fn = 0
for a, p in zip(actual_res, pred_res):
    if a == 1 and p == 1: tp += 1
    elif a == 0 and p == 0: tn += 1
    elif a == 0 and p == 1: fp += 1
    elif a == 1 and p == 0: fn += 1

accuracy = (tp + tn) / len(actual_res)
print(f"Logistic Regression Accuracy: {accuracy:.2%}")
