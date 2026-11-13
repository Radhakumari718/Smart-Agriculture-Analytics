# Smart Agriculture Data Analytics Platform

## Project Overview
This project analyzes global agricultural crop patterns and yield using Big Data technologies and Machine Learning.

## Objectives
- Analyze harvested area, crop production and yield.
- Identify agricultural patterns using clustering.
- Classify crop yield categories using machine learning.
- Compare model performance.

## Dataset Details
- Dataset: Harvested Area and Yield for 175 Crops
- Source: EarthStat
- URL: https://www.earthstat.org/
- Original extracted dataset size: Approximately 2.6 GB
- Generated CSV: 54 MB
- Crops processed: 172
- Records: 23,328
- Columns: 518

## Technologies
- Python
- GDAL
- Hadoop
- HDFS
- Apache Mahout

## Machine Learning Models
1. K-Means Clustering
2. Fuzzy K-Means Clustering
3. Random Forest Classification

Note: Model implementation and evaluation are pending.

## Project Structure
- dataset/: Extracted CSV dataset
- preprocessing/: Dataset extraction script
- models/: Machine learning implementations
- results/: Model evaluation results
- screenshots/: Project execution screenshots

## Current Progress
- Dataset downloaded and extracted.
- GeoTIFF files inspected.
- 172 crop datasets processed.
- CSV generated and verified.

## Model Evaluation Results

### Clustering comparison
Evaluation was performed on 2,251 active agricultural locations.

| Metric | K-Means | Fuzzy C-Means |
|---|---:|---:|
| Silhouette Score | 0.3583 | 0.1599 |
| Davies-Bouldin Index | 1.6258 | 2.6376 |
| Execution Time (seconds) | 0.1964 | 0.0778 |

Best clustering model: K-Means.

### Classification results
Random Forest was evaluated separately for banana yield classification.

- Accuracy: 79.70%
- Precision: 79.88%
- Recall: 79.70%
- F1 Score: 79.62%
- Execution Time: 0.36 seconds

Note: Classification and clustering solve different tasks, so their scores are not directly compared.
