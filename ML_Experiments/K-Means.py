import pandas as pd
import math
import random

# 1. PREPROCESSING
df = pd.read_csv('data_sets/framingham - framingham.csv')
df.fillna(df.mean().round(2), inplace=True)
# Keep it simple: use only 2 columns (Age and BMI) to make it easy to see
data = df[['age', 'BMI']].values.tolist()

# 2. STARTING POINT
k = 10
# Pick 2 random points to be our "centers"
centroids = random.sample(data, k)

# 3. THE "ONE-SHOT" CLUSTER (Simplified to 1 loop)
for _ in range(5):
    clusters = [[] for _ in range(k)]

    # Assignment: Which center is closer?
    for row in data:
        # Distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
        dist_0 = math.sqrt((row[0] - centroids[0][0]) ** 2 + (row[1] - centroids[0][1]) ** 2)
        dist_1 = math.sqrt((row[0] - centroids[1][0]) ** 2 + (row[1] - centroids[1][1]) ** 2)

        if dist_0 < dist_1:
            clusters[0].append(row)
        else:
            clusters[1].append(row)

    # Update: Move centers to the average of their cluster
    for i in range(k):
        if clusters[i]:
            avg_age = sum(p[0] for p in clusters[i]) / len(clusters[i])
            avg_bmi = sum(p[1] for p in clusters[i]) / len(clusters[i])
            centroids[i] = [avg_age, avg_bmi]

# 4. RESULTS
for i in range(k):
    print(f"Cluster{i} (Average Age/BMI): {centroids[i]}")