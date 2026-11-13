
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "results")
VIS_DIR = os.path.join(RESULT_DIR, "visualizations")

os.makedirs(VIS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# 1. Random Forest Evaluation Metrics
metrics = {
    "Accuracy": 79.70,
    "Precision": 79.88,
    "Recall": 79.70,
    "F1-score": 79.62
}

plt.figure(figsize=(9, 5))
ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.title("Random Forest Classification Performance", fontsize=15, weight="bold")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=4)

save_plot("random_forest_metrics.png")


# 2. Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df = pd.read_csv(os.path.join(DATA_DIR, "smart_agriculture_dataset.csv"))

active = df[df["banana_yield"] > 0].copy()
active["Yield_Category"] = pd.qcut(
    active["banana_yield"], q=3,
    labels=["Low", "Medium", "High"]
)

features = [
    c for c in active.columns
    if c.endswith("_area") or c.endswith("_production")
]
features += ["latitude", "longitude"]
features = [
    c for c in features
    if c not in ["banana_area", "banana_production"]
]

X = active[features]
y = active["Yield_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

cm = confusion_matrix(
    y_test, predictions, labels=["Low", "Medium", "High"]
)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlGn",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)
plt.title("Random Forest Confusion Matrix", weight="bold")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
save_plot("confusion_matrix.png")


# 3. Clustering Metrics Comparison
comparison = pd.read_csv(
    os.path.join(RESULT_DIR, "active_clustering_comparison.csv")
)

for metric, filename, title in [
    ("Silhouette", "silhouette_comparison.png", "Silhouette Score Comparison"),
    ("Davies-Bouldin", "davies_bouldin_comparison.png", "Davies-Bouldin Index Comparison")
]:
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=comparison, x="Algorithm", y=metric)
    plt.title(title, weight="bold")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=4)
    save_plot(filename)


# 4. Cluster Distribution
kmeans = pd.read_csv(os.path.join(RESULT_DIR, "kmeans_clusters.csv"))
fuzzy = pd.read_csv(os.path.join(RESULT_DIR, "fuzzy_kmeans_clusters.csv"))

distribution = pd.DataFrame({
    "K-Means": kmeans["Cluster"].value_counts().sort_index(),
    "Fuzzy C-Means": fuzzy["Fuzzy_Cluster"].value_counts().sort_index()
}).fillna(0)

distribution.plot(kind="bar", figsize=(9, 5))
plt.title("Cluster Distribution Comparison", weight="bold")
plt.xlabel("Cluster")
plt.ylabel("Number of Locations")
plt.xticks(rotation=0)
save_plot("cluster_distribution.png")


# 5. Top 10 Crops by Production
production_columns = [
    c for c in df.columns if c.endswith("_production")
]

top_crops = df[production_columns].sum().sort_values(ascending=False).head(10)
top_crops.index = [c.replace("_production", "").title() for c in top_crops.index]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_crops.values, y=top_crops.index)
plt.title("Top 10 Crops by Total Production", weight="bold")
plt.xlabel("Total Production")
plt.ylabel("Crop")
save_plot("top_10_crops.png")


# 6. Geographic Distribution of Active Agricultural Locations
geo = df[
    df[["banana_area", "rice_area", "wheat_area", "maize_area"]]
    .sum(axis=1) > 0
]

plt.figure(figsize=(12, 6))
plt.scatter(
    geo["longitude"], geo["latitude"],
    c=geo["banana_yield"], cmap="YlGn",
    s=8, alpha=0.6
)
plt.colorbar(label="Banana Yield")
plt.title("Geographic Distribution of Agricultural Locations", weight="bold")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
save_plot("geographic_distribution.png")

print("\nAll visualizations generated successfully!")
print("Output folder:", VIS_DIR)
