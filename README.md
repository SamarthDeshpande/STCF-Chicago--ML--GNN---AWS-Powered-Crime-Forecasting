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

##Dataset Link - https://data.cityofchicago.org/Public-Safety/Crimes-2022/9hwr-2zxp/data_preview

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

🗂 S3 Layout and Required Permissions

Inputs

s3://dataset-chicago/project-folder/Processed/Final/ (curated raw table in Parquet)

s3://dataset-chicago/project-folder/Processed/Engineereddata/ (engineered dataset in Parquet)

Outputs

s3://dataset-chicago/project-folder/Processed/Models/ (count forecasting + GNN artifacts)

s3://dataset-chicago/project-folder/Processed/Models/xgb_multiclass/ (XGB artifacts and reports)

🔒 IAM role permissions

s3:GetObject, s3:PutObject

Optional: sagemaker:CreateProcessingJob, sagemaker:CreateTrainingJob

If using private ECR images: ecr:BatchGetImage

⚙ Environment

Python 3.10

Pinned versions for reproducibility:

numpy 1.26.4, pandas 2.2.2, scipy 1.12.0

pyarrow 15.0.2, s3fs 2024.5.0, boto3 1.34.122, joblib 1.4.2

scikit-learn 1.4.2, statsmodels 0.14.3, xgboost 2.0.3

matplotlib 3.8.4, seaborn 0.13.2

torch 2.2.2 (CPU or CUDA 12.1)

Optional: torch-geometric 2.5.3 (+ scatter/sparse/cluster/spline-conv)

Extras: networkx 3.2.1, gensim 4.3.2, node2vec 0.4.6, holidays 0.56

📜 See requirements.txt (auto-generated).

📂  Project Structure
.
├── setup_crime_ml_env.sh
├── feature_engineering_pipeline.py
├── count_forecasting_pipeline.py
├── gnn_hotspot_pipeline.py
├── xgb_multiclass_pipeline.py
├── chicago-feature-engineering-job.json   # optional SageMaker job spec
└── README.md


📊 Dataset Expectations

Base columns: Beat, Date_TS

Feature Engineering → requires Beat, Date_TS (+ optional CrimeType, Latitude, Longitude, Hour)

Count Forecasting → requires Beat, Date_TS (+ optional embeddings/hour clusters)

GNN Hotspot → engineered features (Crime_Count_Last7Days, Rolling_Avg_3Day_Beat, Arrest_Rate_*, top crime dummies, Hour, Month, IsWeekend, Is_Covid_Era, Lat_Bin, Lng_Bin)

XGB Multiclass → requires Date_TS, CrimeType (drops leakage-prone columns internally)

▶ Running Pipelines
1️⃣ Feature Engineering
python feature_engineering_pipeline.py \
  --s3-input  s3://dataset-chicago/project-folder/Processed/Final/ \
  --s3-output s3://dataset-chicago/project-folder/Processed/Engineereddata/part-00000.parquet

2️⃣ Count Forecasting (Beat × Day)
python count_forecasting_pipeline.py \
  --s3-input s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
  --s3-bucket dataset-chicago \
  --s3-prefix-out project-folder/Processed/Models


3️⃣ GNN Hotspot Prediction (Weekly)
python gnn_hotspot_pipeline.py \
  --s3-input s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
  --s3-bucket dataset-chicago \
  --s3-prefix-out project-folder/Processed/Models \
  --outdir /tmp/_out_gnn \
  --epochs 200 --dist-threshold 0.06
Outputs: gnn_model.pth, gnn_scaler.pkl, gnn_threshold.pkl, metrics.json, hotspot predictions.

4️⃣ XGBoost Multiclass (CrimeType)
python xgb_multiclass_pipeline.py \
  --s3-dataset-uri s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
  --s3-out-prefix  s3://dataset-chicago/project-folder/Processed/Models/xgb_multiclass \
  --split-date 2022-01-01 \
  --n-top-feats 30 \
  --outdir /tmp/_out_xgb_multiclass

☁ Run via SageMaker Job JSON (Optional)
aws sagemaker create-processing-job \
  --region us-east-1 \
  --cli-input-json file://chicago-feature-engineering-job.json

🔄 Reproducibility and Config

All scripts expose CLI flags for seeds, thresholds, and model sizes.

Seeds fixed (random_state=42) for comparability.

Manifests + metrics JSON written with every run for traceability.

⚡ Performance Tips

Instances

Feature Engineering & Count Forecasts: m5.2xlarge+ (CPU)

GNN + XGB: c5.4xlarge (CPU) or g5.xlarge (GPU)

Large Data: partition inputs, reduce DBSCAN sample size with --sample-n.

🔐 Security

Do not commit AWS secrets. Use IAM roles.

s3fs defaults to storage_options={"anon": False} → uses role creds.

Restrict write prefixes to minimize exposure.

🛠 Troubleshooting

AccessDenied / S3 auth → check IAM permissions.

PyG install issues → run with INSTALL_PYG=no (uses MLP fallback).

Missing columns → optional steps skipped gracefully.

Large data OOM → increase instance size or sample fewer rows.

🔮 Extending the Pipelines

Add new engineered features in feature_engineering_pipeline.py → auto-flow into GNN/XGB.

Add CatBoost/LightGBM models by following current template.

Extend manifests with additional metadata or metrics.

📜 License

MIT License (see LICENSE).
Third-party dependencies retain their original licenses.
