import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data_sets/framingham - framingham.csv')
df.columns = df.columns.str.strip()
df = (df - df.min()) / (df.max() - df.min())

data = df.values.tolist()
split = int(len(data)*0.8)
training_data = data[:split]
testing_data = data[split:]

def sigmoid(x):
    return 1/(1+math.exp(-x))

def train(train, lr=0.01, iterations=100):
    no_of_features = len(train[0])-1
    weights = [0.0]*no_of_features
    bias = 0.0
    for _ in range(iterations):
        for row in train:
            z = sum(weights[i]*row[i] for i in range(no_of_features)) + bias
            prediction = sigmoid(z)
            actual_val = row[-1]
            error = prediction - actual_val
            for i in range(no_of_features):
                weights[i] -= lr*error*row[i]
            bias -= error*lr
    return weights, bias

def logistic_pred(row, weights, bias):
    z = sum(weights[i]*row[i] for i in range(len(weights))) + bias
    return 1 if sigmoid(z) >= 0.5 else 0

weights, bias = train(training_data)

actual = []
pred = []
for i in testing_data:
    actual.append(int(i[-1]))
    pred.append(logistic_pred(i, weights, bias))

cm = confusion_matrix(actual, pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No CHD', 'CHD'],
            yticklabels=['No CHD', 'CHD'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

tp = tn = fp = fn = 0
for a, p in zip(actual, pred):
    if a == 1 and p == 1: tp += 1
    elif a == 0 and p == 0: tn += 1
    elif a == 0 and p == 1: fp += 1
    elif a == 1 and p == 0: fn += 1

accuracy = (tp + tn) / (tp+fp+tn+fn)
print(f"Logistic Regression Accuracy: {accuracy:.2%}")