import pandas as pd
import math


df = pd.read_csv('data_sets/framingham - framingham.csv')
df.columns = df.columns.str.strip()

df = (df - df.min())/(df.max()-df.min())

data = df.values.tolist()

split = int(len(data)*0.8)

training_data = data[:split]
test_data = data[split:]

def sigmoid(x):
    return 1/(1+math.exp(-x))

def training(train,lr=0.01,iterations=100):
    no_of_features = len(train[0])-1
    weights=[0.0]*no_of_features
    bias = 0.0


    for _ in range(iterations):
        for x in train:
            z = sum(weights[i]*x[i] for i in range(no_of_features))+bias
            prediction = sigmoid(z)

            actual = x[-1]

            error = prediction-actual

            for i in range(no_of_features):
                weights[i] -= lr*error*x[i]
                bias -= error*lr
    return weights,bias

def logistic_reg(x,w,b):
    z = sum(w[i]*x[i] for i in range(len(w)))+b
    if sigmoid(z)>=0.5 : return 1
    else:return 0

weights,bias = training(training_data)

actual = []
pred =[]

for row in test_data:
    actual.append(row[-1])
    pred.append(logistic_reg(row,weights,bias))

tp = tn = fp = fn = 0
for a, p in zip(actual, pred):
    if a == 1 and p == 1: tp += 1
    elif a == 0 and p == 0: tn += 1
    elif a == 0 and p == 1: fp += 1
    elif a == 1 and p == 0: fn += 1

accuracy = (tp + tn) / (tp+fp+tn+fn)
print(f"Logistic Regression Accuracy: {accuracy:.2%}")