import pandas as pd
import math

from ML_Experiments.data_handling import get_dist, accuracy

df = pd.read_csv('data_sets/framingham - framingham.csv')
df.columns=df.columns.str.strip()

all_data = df.values.tolist()

split = int(len(all_data)*0.8)

training_data = all_data[:split]
test_data = all_data[split:]

def get_distance(r1,r2):
    dis = 0.0
    for i in range(len(r1)-1):
        dis += (r1[i]-r2[i])**2
    return math.sqrt(dis)

def get_pred(train,test_row,k):
    distances = []
    for train_row in train:
        dist = get_dist(train_row,test_row)
        distances.append((train_row,dist))

    distances.sort(key = lambda x:x[1])

    neighbours = [distances[m][0][-1] for m in range(k)]

    return max(set(neighbours),key= neighbours.count)

res = get_pred(training_data,test_data[17],5)

print(res)

actual_res =[]
pred_res = []

for test_row in test_data:
    actual_res.append(test_row[-1])
    predicted = get_pred(training_data,test_row,5)
    pred_res.append(predicted)

tp = tn = fp = fn =0

for a,p in zip(actual_res,pred_res):
    if a==1 and p==1: tp+=1
    elif a==0 and p==1: fp+=1
    elif a==1 and p==0: fn+=1
    elif a==0 and p==0: tn+=1

accuracy = tp+tn /(tp+tn+fp+fn)
precision = tp/(tp+fp) if tp+fp>0 else 0
recall = tp/(tp+fn)  if tp+fn >0 else 0

f1 = (2*precision*recall)/(precision+recall)
