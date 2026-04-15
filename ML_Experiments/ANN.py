import pandas as pd
import math


def sigmoid(x):
    return 1/(1+math.exp(-x))

df = pd.read_csv('data_sets/framingham - framingham.csv')

df = (df-df.min())/(df.max() - df.min())

data = df.values.tolist()

trnd = data[:int(len(data)*0.8)]
tstd = data[int(len(data)*0.8):]

features = len(data[0])-1
weights = [0.0]*features
bias=0.0
lr=0.01

for i in range(1000):
    for row in trnd:
        z = sum(weights[j]*row[j] for j in range(features))+bias
        pd = sigmoid(z)
        ac = row[-1]

        err = pd-ac

        for k in range(features):
            weights[k] -= row[k]*err*lr
            bias -= lr*err

def predict(row, wts, b):
    z = sum(row[i] * wts[i] for i in range(len(wts))) + b
    return 1 if sigmoid(z)>=0.5 else 0
actual = []
pred = []

for i in tstd:
    actual.append(i[-1])
    pred.append(predict(i,weights,bias))

tp = tn = fp = fn = 0
for a, p in zip(actual, pred):
    if a == 1 and p == 1: tp += 1
    elif a == 0 and p == 0: tn += 1
    elif a == 0 and p == 1: fp += 1
    elif a == 1 and p == 0: fn += 1

accuracy = (tp + tn) / (tp+fp+tn+fn) if (tp+tn+fn+fp)>0 else 0
print(f"ANN: {accuracy:.2%}")

