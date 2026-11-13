import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE, "dataset", "smart_agriculture_dataset.csv"))

data = data[data["banana_yield"] > 0].copy()

data["Yield_Category"] = pd.qcut(
    data["banana_yield"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

crop_features = [
    c for c in data.columns
    if c.endswith("_area") or c.endswith("_production")
]

features = ["latitude", "longitude"] + crop_features
features = [f for f in features if f not in [
    "banana_area", "banana_production", "banana_yield"
]]

X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = data["Yield_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

start = time()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

execution_time = time() - start

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

with open(os.path.join(BASE, "results", "random_forest_metrics.txt"), "w") as f:
    f.write("Algorithm: Random Forest\n")
    f.write("Target: Banana Yield Category\n")
    f.write(f"Training Records: {len(X_train)}\n")
    f.write(f"Testing Records: {len(X_test)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Execution Time: {execution_time:.2f} seconds\n")

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax)
plt.title("Random Forest - Banana Yield Classification")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "results", "random_forest_confusion_matrix.png"))
plt.close()

print("Random Forest completed successfully!")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Execution Time:", round(execution_time, 2), "seconds")
print(classification_report(y_test, predictions))
