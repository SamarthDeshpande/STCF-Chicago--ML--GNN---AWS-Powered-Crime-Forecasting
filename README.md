# 🔎 Chicago Crime ML — End-to-End Pipelines  

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)  
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green.svg)](https://xgboost.ai/)  
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Glue%20%7C%20S3-orange.svg)](https://aws.amazon.com/)  
[![License](https://img.shields.io/badge/License-MIT-black.svg)](./LICENSE)  

⚡ *Production-ready machine learning pipelines for spatio-temporal crime forecasting in Chicago.*  
This repo unifies *big-data preprocessing (PySpark on AWS Glue)* with *three specialized models* — count forecasting, hotspot prediction (GNN), and crime type classification — all orchestrated in an AWS-native pipeline.  

---

## 📌 Pipelines Overview  

1️⃣ *Feature Engineering* → lags/rolls, histograms, temporal & spatial clusters, Node2Vec embeddings.  
2️⃣ *Count Forecasting* (Beat × Day) → HistGradientBoosting Poisson GBMs, quantile bands, 7-day forecasts.  
3️⃣ *Hotspot Prediction (Weekly)* → Graph Attention Network (PyTorch Geometric) with MLP fallback.  
4️⃣ *Crime Type Classification* → XGBoost multiclass with class-weight balancing and CV feature selection.  

All pipelines: *pure Python CLIs, fully **S3-aware*, with logging, manifests, and reproducible artifacts.  

---

## 🚀 Quick Start  

```bash
# Clone repository
git clone <your-repo>
cd <your-repo>

# Create environment (CPU by default; installs PyG too)
chmod +x setup_crime_ml_env.sh
./setup_crime_ml_env.sh

# If you don’t need PyTorch Geometric:
INSTALL_PYG=no ./setup_crime_ml_env.sh

# Activate environment
conda activate crime-ml

# Smoke test
python - <<'PY'
import numpy, pandas, pyarrow, s3fs, boto3, joblib, sklearn, xgboost, statsmodels, torch
print("Core OK")
PY

