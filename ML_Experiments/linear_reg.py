import pandas as pd
import math

df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2),inplace=True)

x = df['age']
y = df['sysBP']

def mean(row1,row2):
    values1 = row1.values.tolist()
    values2 = row2.values.tolist()
    sum=0.0
    for x,y in zip(values1,values2):
        sum+=x*y
    return (sum/len(values1)).__round__(2)

x_mean,y_mean = x.mean().round(2),y.mean().round(2)
xy_mean = mean(x,y)
print(xy_mean)
print(x_mean,y_mean)

b = (xy_mean - x_mean*y_mean)/(mean(x,x) - x_mean*x_mean)

a = y_mean - (b*x_mean)

def predict(x):
    return a+(b*x)

print(predict(30))

actual = []
predicted = []

for i in range(len(x)):
    actual.append(y[i])
    predicted.append(predict(x[i]))

def get_error(actual,predicted):
    error=0.0
    for a,p in zip(actual,predicted):
        error += (p-a)**2
    return math.sqrt(error/(len(actual)))

print(f"Accuracy:{100-get_error(actual,predicted)}\nError: {get_error(actual,predicted)}")



