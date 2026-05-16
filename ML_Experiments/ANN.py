import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def sigmoid(x):
    return 1/(1+math.exp(-x))

def train(training_data, lr=0.01, iterations=1000):
    no_of_features = len(training_data[0])-1
    weights = [0.0]*no_of_features
    bias = 0.0
    for _ in range(iterations):
        for row in training_data:
            z = sum(weights[i]*row[i] for i in range(no_of_features)) + bias
            pred = sigmoid(z)
            actual_val = row[-1]
            err = pred - actual_val
            for i in range(no_of_features):
                weights[i] -= lr * err * row[i]
            bias -= lr * err
    return weights, bias

def ANN(test_row, weights, bias):
    z = sum(weights[i]*test_row[i] for i in range(len(weights))) + bias
    prob = sigmoid(z)
    return 1 if prob >= 0.5 else 0

df = pd.read_csv('data_sets/framingham - framingham.csv')
df.columns = df.columns.str.strip()
df.fillna(df.mean(), inplace=True)
df = (df - df.min()) / (df.max() - df.min())

data = df.values.tolist()
split = int(len(data)*0.8)
training_data = data[:split]
test_data = data[split:]

weights, bias = train(training_data)

actual = []
pred = []

for i in test_data:
    actual.append(int(i[-1]))
    pred.append(ANN(i, weights, bias))

cm = confusion_matrix(actual, pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('ANN (Perceptron) Confusion Matrix')
plt.show()

fn = fp = tn = tp = 0
for a, p in zip(actual, pred):
    if a == 0 and p == 0: tn += 1
    elif a == 1 and p == 1: tp += 1
    elif a == 0 and p == 1: fp += 1
    elif a == 1 and p == 0: fn += 1

acc = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy for ANN: {acc:.2%}")