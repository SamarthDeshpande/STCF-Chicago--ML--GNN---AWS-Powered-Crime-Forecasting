CHICAGO CRIME ML — END-TO-END PIPELINES (TXT EDITION)
=====================================================

This repository contains four production-ready pipelines that read/write data on
Amazon S3 and run cleanly on AWS SageMaker:

  1) Feature Engineering — builds lags/rolls, histograms, spatial/temporal
     clusters, optional Node2Vec embeddings.
  2) Count Forecasting (Beat × Day) — Poisson GBMs (+ optional XGB) with
     quantile bands and 7-day forecasts.
  3) GNN Hotspot Prediction (Weekly) — Graph Attention Network (PyTorch
     Geometric) with MLP fallback when PyG is not installed.
  4) XGBoost Multiclass (CrimeType) — class-weighted training, CV feature
     selection, robust early-stopping handling.

All pipelines are pure Python CLIs with S3-aware I/O (pyarrow+s3fs), logging,
and manifest/metrics artifacts.



0) QUICK START
--------------

On SageMaker (or any machine with conda):

  git clone <your-repo>
  cd <your-repo>

  # Create environment (CPU by default; also installs PyG by default)
  chmod +x setup_crime_ml_env.sh
  ./setup_crime_ml_env.sh

  # If you do not need PyG (GNN will use MLP fallback):
  INSTALL_PYG=no ./setup_crime_ml_env.sh

  # Activate environment
  conda activate crime-ml

Smoke test:

  python - <<'PY'
import numpy, pandas, pyarrow, s3fs, boto3, joblib, sklearn, xgboost, statsmodels, torch
print("Core OK")
PY



1) S3 LAYOUT AND REQUIRED PERMISSIONS
-------------------------------------

Example paths used by defaults (you can override via CLI flags):

  INPUTS
    s3://dataset-chicago/project-folder/Processed/Final/
      (curated raw table in Parquet)

    s3://dataset-chicago/project-folder/Processed/Engineereddata/
      (engineered dataset in Parquet)

  OUTPUTS
    s3://dataset-chicago/project-folder/Processed/Models/
      (count forecasting + gnn artifacts)

    s3://dataset-chicago/project-folder/Processed/Models/xgb_multiclass/
      (xgb artifacts and reports)

IAM role for your SageMaker runtime (or EC2) should allow:
  - s3:GetObject and s3:PutObject for the above prefixes
  - Optional, if you submit SageMaker jobs directly: sagemaker:CreateProcessingJob
    and/or sagemaker:CreateTrainingJob
  - If using private ECR images (not required by these scripts): ecr:BatchGetImage



2) ENVIRONMENT
--------------

Use the provided setup script:

  setup_crime_ml_env.sh
    - Creates conda env "crime-ml" (Python 3.10).
    - Installs CPU or GPU builds of PyTorch 2.2.2 (CUDA 12.1 if GPU detected).
    - Optionally installs PyTorch Geometric (PyG) 2.5.3 + companions.
    - Writes a unified, pinned requirements.txt compatible with all pipelines.

Pinned versions (kept consistent across all four pipelines):
  - numpy 1.26.4, pandas 2.2.2, scipy 1.12.0
  - pyarrow 15.0.2, s3fs 2024.5.0, boto3 1.34.122, joblib 1.4.2
  - scikit-learn 1.4.2, statsmodels 0.14.3, xgboost 2.0.3
  - matplotlib 3.8.4, seaborn 0.13.2
  - torch 2.2.2 (CPU or CUDA 12.1)
  - Optional: torch-geometric 2.5.3 (+ torch-scatter 2.1.2, torch-sparse 0.6.18,
    torch-cluster 1.6.3, torch-spline-conv 1.2.2)
  - Feature-engineering extras: networkx 3.2.1, gensim 4.3.2, node2vec 0.4.6
  - holidays 0.56



3) SUGGESTED PROJECT STRUCTURE
------------------------------

.
├── setup_crime_ml_env.sh
├── feature_engineering_pipeline.py
├── count_forecasting_pipeline.py
├── gnn_hotspot_pipeline.py
├── xgb_multiclass_pipeline.py
├── chicago-feature-engineering-job.json      (optional SageMaker job spec)
└── readme.txt



4) DATASET EXPECTATIONS
-----------------------

Common base columns:
  - Beat       (string or integer convertible to string)
  - Date_TS    (timestamp; UTC/local consistent)

Feature Engineering:
  - Requires: Beat, Date_TS
  - Optional but recommended: CrimeType, Latitude, Longitude, Hour
    * If Hour missing, derived from Date_TS
    * If Latitude/Longitude missing, DBSCAN and Node2Vec steps are skipped

Count Forecasting:
  - Requires: Beat, Date_TS
  - Optional: Latitude, Longitude (for per-beat medians),
              any Embed_* or Hour_Cluster columns (used if present)

GNN Hotspot (Weekly):
  - Input is typically the engineered dataset from Feature Engineering
  - Requires: Beat, Date_TS (for weekly aggregation)
  - Uses these if present: Crime_Count_Last7Days, Rolling_Avg_3Day_Beat,
    Crime_Lag_*, Arrest_Rate_*, top crime one-hots, Hour, Weekday, Month,
    IsWeekend, Is_Covid_Era, Lat_Bin, Lng_Bin

XGB Multiclass:
  - Requires: Date_TS, CrimeType
  - The script drops leakage-prone and text columns internally



5) RUNNING EACH PIPELINE
------------------------

5.1 FEATURE ENGINEERING

Builds lags/rolls, beat-level crime histograms, DBSCAN spatial cluster,
KMeans hour cluster, and optional Node2Vec embeddings using beat proximity.

Command:

  python feature_engineering_pipeline.py \
    --s3-input  s3://dataset-chicago/project-folder/Processed/Final/ \
    --s3-output s3://dataset-chicago/project-folder/Processed/Engineereddata/part-00000.parquet

Output:
  Engineereddata/part-00000.parquet  (single Parquet file with engineered columns)


5.2 COUNT FORECASTING (BEAT × DAY)

Poisson models (Tweedie and HGB), optional XGB, quantile bands, test metrics,
and 7-day forecasts per beat. Artifacts uploaded to S3.

Command:

  python count_forecasting_pipeline.py \
    --s3-input s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
    --s3-bucket dataset-chicago \
    --s3-prefix-out project-folder/Processed/Models

Key outputs (in Models/):
  test_predictions_best.csv
  forecast_7d_per_beat.csv
  beat_day_test_actuals_preds.csv
  model_results_test.csv
  per_beat_metrics.csv
  actual_vs_pred_best.png
  error_distribution_best.png
  worst_beats_mae.png
  citywide_forecast.png
  count_models.pkl
  count_feature_cols.pkl
  beat_day_features.parquet
  per_beat_location.csv
  summary.txt
  MANIFEST.json


5.3 GNN HOTSPOT PREDICTION (WEEKLY)

Uses GAT (PyG). If PyG is not installed, falls back to an MLP using the same
tabular features.

Command:

  python gnn_hotspot_pipeline.py \
    --s3-input s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
    --s3-bucket dataset-chicago \
    --s3-prefix-out project-folder/Processed/Models \
    --outdir /tmp/_out_gnn \
    --epochs 200 --dist-threshold 0.06

Key outputs (in Models/):
  gnn_model.pth
  gnn_scaler.pkl
  gnn_threshold.pkl
  metrics.json
  training_loss.png
  gnn_hotspot_predictions.csv
  per_beat_location.csv  (if lat/lon available)


5.4 XGBOOST MULTICLASS (CRIMETYPE)

Class-weighted training, 3-fold cross-validated feature selection, robust
early-stopping handling (callbacks, then parameter style, then fallback).

Command:

  python xgb_multiclass_pipeline.py \
    --s3-dataset-uri s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
    --s3-out-prefix  s3://dataset-chicago/project-folder/Processed/Models/xgb_multiclass \
    --split-date 2022-01-01 \
    --n-top-feats 30 \
    --outdir /tmp/_out_xgb_multiclass

Key outputs:
  Artifacts/
    xgboost_crimetype_model.pkl
    xgboost_label_encoders.pkl
    xgboost_selected_features.pkl
    xgboost_label_mapping.pkl
  Reports/
    classification_report.csv
    confusion_matrix.png
    roc_curve.png
    feature_importance.png
    top30_features.csv
    accuracy_by_crime.csv
  Root:
    MANIFEST.json



6) RUN VIA SAGEMAKER JOB JSON (OPTIONAL)
----------------------------------------

A file like chicago-feature-engineering-job.json is a job specification, not
an executable. Submit it with AWS CLI:

  aws sagemaker create-processing-job \
    --region us-east-1 \
    --cli-input-json file://chicago-feature-engineering-job.json

GitHub Actions example (runs in your AWS account via OIDC and an assumable role):

  .github/workflows/run-feature-eng.yml
    - uses: actions/checkout@v4
    - uses: aws-actions/configure-aws-credentials@v4
    - run: aws sagemaker create-processing-job --region us-east-1 \
             --cli-input-json file://chicago-feature-engineering-job.json



7) REPRODUCIBILITY AND CONFIG
-----------------------------

- All scripts expose CLI flags for seeds, thresholds, model sizes, etc.
- Seeds are set (random_state=42), but numerical determinism may still vary
  slightly by hardware and library builds, especially for deep learning.
- Manifests and metrics JSON are written alongside artifacts for traceability.



8) PERFORMANCE TIPS
-------------------

- Instance choice:
    Feature Engineering and Count Forecasts: m5.2xlarge or larger (CPU).
    GNN (with PyG) and XGB: benefit from c5.4xlarge (CPU) or g5.xlarge (GPU).
- Reading Parquet:
    Reading directory layouts is supported; pyarrow handles partitioned files.
- Memory:
    For large datasets, process by date partitions or reduce DBSCAN sample size
    with --sample-n (feature engineering pipeline).



9) SECURITY AND SECRETS
-----------------------

- Do not commit AWS secrets. Prefer IAM roles attached to the runtime.
- The scripts default to storage_options={"anon": False} for s3fs, which uses
  instance/role credentials.
- Bucket names and ARNs are not secrets, but ensure write prefixes are scoped.



10) TROUBLESHOOTING
-------------------

- AccessDenied / S3 auth:
    Ensure the runtime role has s3:GetObject and s3:PutObject on the prefixes.

- pyarrow or s3fs errors:
    Confirm both are installed; the setup script pins compatible versions.

- PyG install issues:
    Run INSTALL_PYG=no ./setup_crime_ml_env.sh and re-run the GNN pipeline
    (it will use the MLP fallback).

- xgboost early-stopping flags not accepted:
    The Multiclass script tries callbacks, then early_stopping_rounds, then a
    no-early-stopping fallback. Training will proceed.

- Missing columns:
    Scripts validate required columns and skip optional steps if inputs are
    absent (e.g., embeddings if lat/lon missing).

- Large data:
    Increase instance size or reduce sampling for DBSCAN.



11) EXTENDING THE PIPELINES
---------------------------

- Add new engineered features to feature_engineering_pipeline.py and they will
  flow into GNN/XGB models automatically (if columns exist).
- Add alternative models (e.g., CatBoost/LightGBM) by following current
  training/evaluation patterns.
- Extend manifests with any extra metadata or metrics your workflows need.



12) LICENSES AND CREDITS
------------------------

- Repository code: your chosen license (e.g., MIT or Apache-2.0).
- Third-party dependencies are licensed by their authors (PyTorch, PyG,
  scikit-learn, XGBoost, pandas, etc.).



13) HANDY COMMANDS
------------------

Generate environment and see requirements:

  ./setup_crime_ml_env.sh
  cat requirements.txt

Run all four pipelines in sequence (example):

  # 1) Feature Engineering
  python feature_engineering_pipeline.py \
    --s3-input  s3://dataset-chicago/project-folder/Processed/Final/ \
    --s3-output s3://dataset-chicago/project-folder/Processed/Engineereddata/part-00000.parquet

  # 2) Count Forecasts
  python count_forecasting_pipeline.py \
    --s3-input s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
    --s3-bucket dataset-chicago \
    --s3-prefix-out project-folder/Processed/Models

  # 3) GNN Hotspots
  python gnn_hotspot_pipeline.py \
    --s3-input s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
    --s3-bucket dataset-chicago \
    --s3-prefix-out project-folder/Processed/Models \
    --outdir /tmp/_out_gnn

  # 4) XGB Multiclass
  python xgb_multiclass_pipeline.py \
    --s3-dataset-uri s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
    --s3-out-prefix  s3://dataset-chicago/project-folder/Processed/Models/xgb_multiclass \
    --split-date 2022-01-01 \
    --n-top-feats 30


END OF FILE
