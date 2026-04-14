import math
import pandas as pd


df = pd.read_csv('data_sets/framingham - framingham.csv')
df.columns = df.columns.str.strip()
df.fillna(df.mean().round(2),inplace=True)

all_data = df.values.tolist()

split = int(len(all_data) * 0.8)

training_data = all_data[:split]
test_data = all_data[split:]

def get_dist(row1,row2):
    dist = 0.0
    for i in range(len(row1)-1):
        dist+= (row1[i]-row2[i])**2
    return math.sqrt(dist)

def get_prediction(train,test_row,k):
    distances = []

    for train_row in train:
        distance = get_dist(train_row, test_row)
        distances.append((train_row, distance))

    distances.sort(key=lambda x:x[1])

    neighbours = []
    for m in range(k):
        neighbours.append(distances[m][0][-1])

    return max(set(neighbours), key=neighbours.count)

result = get_prediction(training_data,test_data[5],10)

print(result)


actual_results = []
predicted_results = []

tp = tn = fp = fn = 0

for i in test_data:
    actual_results.append(i[-1])

    res = get_prediction(training_data,i,5)
    predicted_results.append(res)

for a,p in zip(actual_results,predicted_results):
    if a==0 and p==0 : tn+=1
    elif a==1 and p==1:tp+=1
    elif a==0 and p==1: fp+=1
    elif a==1 and p==0: fn+=1

accuracy = (tp + tn) / len(actual_results) if len(actual_results) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")

