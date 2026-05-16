import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)

x = df['age']
y = df['sysBP']

def mean_product(row1, row2):
    values1 = row1.values.tolist()
    values2 = row2.values.tolist()
    total = 0.0
    for x_val, y_val in zip(values1, values2):
        total += x_val * y_val
    return (total / len(values1)).__round__(2)

x_mean, y_mean = x.mean().round(2), y.mean().round(2)
xy_mean = mean_product(x, y)
xx_mean = mean_product(x, x)

b = (xy_mean - (x_mean * y_mean)) / (xx_mean - (x_mean**2))
a = y_mean - (b * x_mean)

def predict(val):
    return a + (b * val)

actual_vals = y.tolist()
predicted_vals = [predict(val) for val in x]

threshold = 140
actual_labels = [1 if val >= threshold else 0 for val in actual_vals]
predicted_labels = [1 if val >= threshold else 0 for val in predicted_vals]

cm = confusion_matrix(actual_labels,predicted_labels)

plt.figure(figsize=(8,6))

sns.heatmap(cm,annot=True,fmt='d',cmap='Purples')
plt.xlabel('x-values')
plt.ylabel('y-label')
plt.title('Confusion Matrix')
plt.show()
def get_rmse(act, pre):
    error = 0.0
    for a_val, p_val in zip(act, pre):
        error += (p_val - a_val)**2
    return math.sqrt(error / len(act))

rmse = get_rmse(actual_vals, predicted_vals)
print(f"RMSE: {rmse}")