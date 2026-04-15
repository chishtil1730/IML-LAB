import pandas as pd
import math
import random

# 1. PREPROCESSING
df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)
# Keep it simple: use only 2 columns (Age and BMI) to make it easy to see
data = df[['age', 'BMI']].values.tolist()

k =10
centroids = random.sample(data,k)

for _ in range(5):
    clusters= [[] for _ in range(k)]

    for row in data:
        d1 = math.sqrt((row[0] - centroids[0][0]) ** 2 + (row[1] - centroids[0][1]) ** 2)
        d2 = math.sqrt((row[0] - centroids[1][0]) ** 2 + (row[1] - centroids[1][1]) ** 2)

        if d1<d2:
            clusters[0].append(row)
        else:
            clusters[1].append(row)

    for i in range(k):
        if clusters[i]:
            avg_age = sum(p[0] for p in clusters[i])/len(clusters[i])
            avg_bmi  = sum(p[1] for p in clusters[i])/len(clusters[i])

            centroid = [avg_age,avg_bmi]

    for i in centroids:
        print(i)
