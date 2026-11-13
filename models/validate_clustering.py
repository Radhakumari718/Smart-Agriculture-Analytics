import os
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from time import time
from sklearn.cluster import KMeans
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

activity_cols = [
    c for c in features
    if c.endswith("_area") or c.endswith("_production")
]

active = data[activity_cols].fillna(0).abs().sum(axis=1) > 0
filtered = data.loc[active].copy()

X = filtered[features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_scaled = StandardScaler().fit_transform(X)

start = time()
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = km.fit_predict(X_scaled)
km_time = time() - start

start = time()
centers, membership, _, _, _, _, fpc = fuzz.cluster.cmeans(
    X_scaled.T, c=3, m=2, error=0.005, maxiter=1000, seed=42
)
fuzzy_labels = np.argmax(membership, axis=0)
fuzzy_time = time() - start

results = []

for name, labels, elapsed in [
    ("K-Means", km_labels, km_time),
    ("Fuzzy C-Means", fuzzy_labels, fuzzy_time)
]:
    results.append({
        "Algorithm": name,
        "Silhouette": silhouette_score(X_scaled, labels),
        "Davies-Bouldin": davies_bouldin_score(X_scaled, labels),
        "Execution_Time": elapsed
    })
    print("\n", name)
    print("Cluster distribution:")
    print(pd.Series(labels).value_counts().sort_index())

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
pd.DataFrame(results).to_csv(
    os.path.join(BASE, "results", "active_clustering_comparison.csv"),
    index=False
)

print("\nActive agricultural locations:", len(filtered))
print(pd.DataFrame(results).to_string(index=False))
print("\nComparison saved successfully.")
