import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE, "dataset", "smart_agriculture_dataset.csv"))

features = [
    "latitude", "longitude",
    "banana_area", "banana_yield", "banana_production",
    "wheat_area", "wheat_yield", "wheat_production",
    "rice_area", "rice_yield", "rice_production",
    "maize_area", "maize_yield", "maize_production"
]

features = [f for f in features if f in data.columns]

X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

start = time()

centers, membership, _, _, _, _, fpc = fuzz.cluster.cmeans(
    X_scaled.T,
    c=3,
    m=2,
    error=0.005,
    maxiter=1000,
    init=None,
    seed=42
)

labels = np.argmax(membership, axis=0)

execution_time = time() - start

silhouette = silhouette_score(X_scaled, labels)
db_index = davies_bouldin_score(X_scaled, labels)

data["Fuzzy_Cluster"] = labels
data["Membership_Strength"] = np.max(membership, axis=0)

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

data.to_csv(os.path.join(BASE, "results", "fuzzy_kmeans_clusters.csv"), index=False)

with open(os.path.join(BASE, "results", "fuzzy_kmeans_metrics.txt"), "w") as f:
    f.write("Algorithm: Fuzzy C-Means\n")
    f.write("Clusters: 3\n")
    f.write(f"Silhouette Score: {silhouette:.4f}\n")
    f.write(f"Davies-Bouldin Index: {db_index:.4f}\n")
    f.write(f"Fuzzy Partition Coefficient: {fpc:.4f}\n")
    f.write(f"Execution Time: {execution_time:.2f} seconds\n")

plt.figure(figsize=(8, 5))
plt.scatter(data["longitude"], data["latitude"], c=labels, s=5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Agricultural Geographic Clusters - Fuzzy C-Means")
plt.savefig(os.path.join(BASE, "results", "fuzzy_kmeans_clusters.png"))
plt.close()

print("Fuzzy C-Means completed successfully!")
print("Silhouette Score:", round(silhouette, 4))
print("Davies-Bouldin Index:", round(db_index, 4))
print("Fuzzy Partition Coefficient:", round(fpc, 4))
print("Execution Time:", round(execution_time, 2), "seconds")
print("Cluster Distribution:")
print(pd.Series(labels).value_counts().sort_index())
