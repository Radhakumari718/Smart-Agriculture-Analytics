
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from time import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE, "dataset", "smart_agriculture_dataset.csv"))

features = [
    "latitude",
    "longitude",
    "banana_area",
    "banana_yield",
    "banana_production",
    "wheat_area",
    "wheat_yield",
    "wheat_production",
    "rice_area",
    "rice_yield",
    "rice_production",
    "maize_area",
    "maize_yield",
    "maize_production"
]

features = [f for f in features if f in data.columns]

X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

start = time()

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

execution_time = time() - start

silhouette = silhouette_score(X_scaled, labels)
db_index = davies_bouldin_score(X_scaled, labels)

data["Cluster"] = labels

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

data.to_csv(os.path.join(BASE, "results", "kmeans_clusters.csv"), index=False)

with open(os.path.join(BASE, "results", "kmeans_metrics.txt"), "w") as f:
    f.write(f"Algorithm: K-Means\n")
    f.write(f"Clusters: 3\n")
    f.write(f"Silhouette Score: {silhouette:.4f}\n")
    f.write(f"Davies-Bouldin Index: {db_index:.4f}\n")
    f.write(f"Execution Time: {execution_time:.2f} seconds\n")

plt.figure(figsize=(8, 5))
plt.scatter(data["longitude"], data["latitude"], c=labels, s=5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Agricultural Geographic Clusters - K-Means")
plt.savefig(os.path.join(BASE, "results", "kmeans_clusters.png"))
plt.close()

print("K-Means completed successfully!")
print("Silhouette Score:", round(silhouette, 4))
print("Davies-Bouldin Index:", round(db_index, 4))
print("Execution Time:", round(execution_time, 2), "seconds")
print("Cluster Distribution:")
print(data["Cluster"].value_counts().sort_index())
