#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiclass XGBoost (CrimeType) — AWS S3 VERSION
- Reads Parquet from S3 (dataset directory)
- Trains class-weighted XGBClassifier with CV-based feature selection
- Handles early stopping across different xgboost versions (callbacks / param / none)
- Saves ALL artifacts to S3 under project-folder/Processed/Models/xgb_multiclass/
- Also writes locally to --outdir (default: /tmp/_out_xgb_multiclass)
"""

import os
import json
import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SNS = True
    sns.set(rc={"figure.dpi": 120})
except Exception:
    _HAS_SNS = False

# AWS / S3
import boto3

# XGBoost + sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_curve, auc
)

warnings.filterwarnings("ignore", category=UserWarning, message=".use_label_encoder.")
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- Config -----------------------------

@dataclass
class RunConfig:
    s3_dataset_uri: str
    s3_out_prefix: str
    split_date: str
    n_top_feats: int
    outdir: Path
    seed: int = 42
    k_folds: int = 3
    test_min_year: int = 2010
    upload: bool = True

# ----------------------------- Logging -----------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )

# ----------------------------- S3 helpers -----------------------------

def parse_s3_uri(uri: str):
    u = urlparse(uri)
    return u.netloc, u.path.lstrip("/")

def s3_join(prefix: str, *parts: str):
    prefix = prefix.rstrip("/")
    tail = "/".join(p.strip("/") for p in parts)
    return f"{prefix}/{tail}"

def upload_file(local_path: str, s3_uri: str, s3_client=None):
    if s3_client is None:
        s3_client = boto3.client("s3")
    bucket, key = parse_s3_uri(s3_uri)
    s3_client.upload_file(local_path, bucket, key)
    logging.info("⬆  Uploaded: %s", s3_uri)

def save_df_to_s3(df: pd.DataFrame, s3_uri: str, outdir: Path, s3_client=None):
    # Write local then upload (most compatible)
    tmp = outdir / os.path.basename(urlparse(s3_uri).path)
    df.to_csv(tmp, index=False)
    upload_file(str(tmp), s3_uri, s3_client)

def save_fig_to_s3(fig, s3_uri: str, outdir: Path, dpi=300, s3_client=None):
    tmp = outdir / os.path.basename(urlparse(s3_uri).path)
    fig.tight_layout()
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    upload_file(str(tmp), s3_uri, s3_client)

def save_joblib_to_s3(obj, s3_uri: str, outdir: Path, s3_client=None):
    tmp = outdir / os.path.basename(urlparse(s3_uri).path)
    joblib.dump(obj, tmp)
    upload_file(str(tmp), s3_uri, s3_client)

# ----------------------------- Data I/O -----------------------------

def load_dataset(s3_dataset_uri: str, min_year: int) -> pd.DataFrame:
    logging.info("📥 Loading Parquet dataset from: %s", s3_dataset_uri)
    # Requires pyarrow + s3fs
    df = pd.read_parquet(
        s3_dataset_uri,
        engine="pyarrow",
        storage_options={"anon": False}
    )
    logging.info("📊 DataFrame shape: %s", df.shape)

    required = {"Date_TS", "CrimeType"}
    missing_req = required - set(df.columns)
    if missing_req:
        raise ValueError(f"Input is missing required columns: {missing_req}")

    df["Date_TS"] = pd.to_datetime(df["Date_TS"], errors="coerce")
    df = df[df["Date_TS"].notna()]
    df = df[df["Date_TS"].dt.year >= min_year].copy()
    df = df[df["CrimeType"].notna()].copy()
    return df

def basic_optimize(df: pd.DataFrame) -> pd.DataFrame:
    # Downcast numerics; cast small-cardinality objects to category
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique(dropna=False) < 1000:
            df[col] = df[col].astype("category")
    return df

# ----------------------------- Features & Labels -----------------------------

DROP_COLS = [
    # targets / duplicates / obvious temporal strings
    "label", "CrimeType", "PrimaryType", "TopCrimeType", "Crime_Context",
    "Date", "Updated On", "Date_TS", "Date_Only", "YearMonth",
    "Description", "Description_Grouped",
    # outcomes / leakage
    "FBI_Code", "FBI Code", "Arrest", "Arrest_Flag",
    "FBI_Code_Count", "FBI_Arrest_Rate",
    # raw strings (use engineered bins instead)
    "Location", "Location_Description", "Primary Type",
    # sometimes present one-hots — drop to avoid trivial shortcuts
    "ASSAULT", "BATTERY", "BURGLARY", "CRIMINAL DAMAGE", "DECEPTIVE PRACTICE",
    "MOTOR VEHICLE THEFT", "NARCOTICS", "OTHER", "ROBBERY", "THEFT",
    # occasional engineered dups
    "FBI_Category", "CrimeCountPerDay_x", "CrimeCountPerDay_y", "CrimeType_Count"
]

def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    crime_cat = df["CrimeType"].astype("category")
    df["label"] = crime_cat.cat.codes
    label_mapping = dict(enumerate(crime_cat.cat.categories))  # code -> name

    existing_cols = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=existing_cols)
    y = df["label"].copy()
    return X, y, label_mapping

def time_split(X: pd.DataFrame, y: pd.Series, df_ref: pd.DataFrame, split_date: str):
    train_mask = df_ref["Date_TS"] < split_date
    test_mask  = df_ref["Date_TS"] >= split_date
    X_train, X_test = X.loc[train_mask].copy(), X.loc[test_mask].copy()
    y_train, y_test = y.loc[train_mask].copy(), y.loc[test_mask].copy()
    logging.info("🧪 Train size: %s | Test size: %s", X_train.shape, X_test.shape)
    assert len(X_train) > 0 and len(X_test) > 0, "Empty train/test after split—check --split-date."
    return X_train, X_test, y_train, y_test

def fit_transform_label_encoders(X_train, X_test, na_token="_NA_"):
    label_encoders = {}
    cat_cols = X_train.select_dtypes(include=["category", "object"]).columns.tolist()
    for col in cat_cols:
        both = pd.concat([X_train[col], X_test[col]], axis=0)
        both = both.astype("string").fillna(na_token)
        le = LabelEncoder().fit(both)
        X_train.loc[:, col] = le.transform(X_train[col].astype("string").fillna(na_token))
        X_test.loc[:,  col] = le.transform(X_test[col].astype("string").fillna(na_token))
        label_encoders[col] = le
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test  = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X_train, X_test, label_encoders

def compute_weights(y_train: pd.Series) -> np.ndarray:
    classes = np.unique(y_train)
    cls_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    mapping = {c: w for c, w in zip(classes, cls_weights)}
    sw = y_train.map(mapping).astype(float).to_numpy()
    logging.info("🛡 Class weights applied. #Classes: %d | Example: %s",
                 len(classes), list(mapping.items())[:5])
    return sw

# ----------------------------- Feature Selection -----------------------------

def cv_select_top_features(X_train: pd.DataFrame, y_train: pd.Series,
                           sample_weight_train: np.ndarray, n_top: int, seed: int, k_folds: int) -> list[str]:
    logging.info("🔍 Selecting important features with cross-validation...")
    best_features = []
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    X_tr_np = X_train.values
    y_tr_np = y_train.values
    sw_tr_np = sample_weight_train

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_tr_np, y_tr_np), start=1):
        Xtr, Xva = X_tr_np[tr_idx], X_tr_np[va_idx]
        ytr = y_tr_np[tr_idx]
        swtr = sw_tr_np[tr_idx]

        selector = XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=seed,
            n_estimators=80,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            n_jobs=-1
        )
        selector.fit(Xtr, ytr, sample_weight=swtr)
        importances = pd.Series(selector.feature_importances_, index=X_train.columns)
        top_fold_features = importances.nlargest(n_top).index.tolist()
        best_features.extend(top_fold_features)
        logging.info("   Fold %d: captured %d features", fold, len(top_fold_features))

    feature_counts = pd.Series(best_features).value_counts()
    top_features = feature_counts.head(n_top).index.tolist()
    if not top_features:
        top_features = X_train.columns.tolist()
    logging.info("✅ Selected top %d features", len(top_features))
    return top_features

# ----------------------------- Training -----------------------------

def train_final_model(X_tr_final, y_tr_final, X_te_final, y_te_final, sample_weight_train):
    logging.info("🏋 Training final model with regularization + class weights...")
    final_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_tr_final)),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=200,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,        # L1
        reg_lambda=1.5,       # L2
        min_child_weight=3,
        gamma=0.5,
        n_jobs=-1
    )

    fit_succeeded = False
    # Try callback-based early stopping
    try:
        from xgboost.callback import EarlyStopping
        es = EarlyStopping(rounds=30, metric_name="mlogloss", save_best=True, maximize=False)
        final_model.fit(
            X_tr_final, y_tr_final,
            sample_weight=sample_weight_train,
            eval_set=[(X_te_final, y_te_final)],
            callbacks=[es],
            verbose=10
        )
        fit_succeeded = True
        logging.info("   ✅ Trained with callback-based early stopping.")
    except Exception as e:
        logging.info("   ⚠ Callback early stopping not available: %s", e)

    # Try parameter-based early stopping
    if not fit_succeeded:
        try:
            final_model.fit(
                X_tr_final, y_tr_final,
                sample_weight=sample_weight_train,
                eval_set=[(X_te_final, y_te_final)],
                early_stopping_rounds=30,
                verbose=10
            )
            fit_succeeded = True
            logging.info("   ✅ Trained with early_stopping_rounds.")
        except Exception as e:
            logging.info("   ⚠ early_stopping_rounds not supported: %s", e)

    # Final fallback: no ES
    if not fit_succeeded:
        final_model.fit(
            X_tr_final, y_tr_final,
            sample_weight=sample_weight_train,
            eval_set=[(X_te_final, y_te_final)],
            verbose=10
        )
        logging.info("   ✅ Trained without early stopping (version fallback).")

    return final_model

# ----------------------------- Evaluation & Plots -----------------------------

def evaluate_and_plots(final_model, X_te_final, y_te_final, top_features, label_mapping,
                       s3_reports_dir: str, outdir: Path, s3_client=None):
    logging.info("🔮 Generating predictions & reports...")
    y_pred       = final_model.predict(X_te_final)
    y_pred_proba = final_model.predict_proba(X_te_final)

    # Names for readability
    y_test_names = pd.Series(y_te_final).map(label_mapping)
    y_pred_names = pd.Series(y_pred, index=pd.Index(range(len(y_pred)))).map(label_mapping)

    # Classification report
    class_report = classification_report(
        y_test_names, y_pred_names,
        labels=list(label_mapping.values()),
        output_dict=True,
        zero_division=0
    )
    class_report_df = pd.DataFrame(class_report).transpose()
    save_df_to_s3(class_report_df.reset_index().rename(columns={"index":"metric"}),
                  s3_join(s3_reports_dir, "classification_report.csv"), outdir, s3_client)

    # Confusion matrix (Top 10 by support)
    logging.info("🎨 Creating confusion matrix...")
    top_crimes = y_test_names.value_counts().nlargest(10).index.tolist()
    idx_all = pd.Index(range(len(y_pred)))
    common_idx = idx_all  # both are aligned
    y_test_top = y_test_names.loc[common_idx]
    y_pred_top = y_pred_names.loc[common_idx]
    mask = y_test_top.isin(top_crimes)
    y_test_top = y_test_top[mask]
    y_pred_top = y_pred_top[mask]

    fig = plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_test_top, y_pred_top, labels=top_crimes)
    if _HAS_SNS:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=top_crimes, yticklabels=top_crimes)
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(top_crimes)), top_crimes, rotation=45, ha="right")
        plt.yticks(range(len(top_crimes)), top_crimes)
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center', fontsize=8)
    plt.title('Confusion Matrix (Top 10 Crime Types)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    save_fig_to_s3(fig, s3_join(s3_reports_dir, "confusion_matrix.png"), outdir, s3_client=s3_client)

    # ROC curves (Top 5 by support)
    logging.info("📈 Creating ROC curves...")
    classes_in_model = list(final_model.classes_)
    class_to_col = {c: i for i, c in enumerate(classes_in_model)}
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    crime_counts = y_test_names.value_counts()
    top_5_names = crime_counts.head(5).index.tolist()
    top_5_codes = [inv_label_mapping[n] for n in top_5_names if n in inv_label_mapping]

    if len(classes_in_model) >= 2:
        y_test_bin = label_binarize(y_te_final, classes=classes_in_model)
        fig = plt.figure(figsize=(10, 8))
        for name, code in zip(top_5_names, top_5_codes):
            if code not in class_to_col:
                continue
            col = class_to_col[code]
            if y_test_bin.shape[1] <= col or y_pred_proba.shape[1] <= col:
                continue
            fpr, tpr, _ = roc_curve(y_test_bin[:, col], y_pred_proba[:, col])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0,1],[0,1],'--', lw=1, color='navy')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve – Top 5 Crime Types')
        plt.legend(loc="lower right")
        save_fig_to_s3(fig, s3_join(s3_reports_dir, "roc_curve.png"), outdir, s3_client=s3_client)
    else:
        logging.info("⚠ Skipped ROC (need at least 2 classes).")

    # Feature importance
    logging.info("📊 Saving feature importance...")
    importances = final_model.feature_importances_
    sorted_idx_20 = np.argsort(importances)[-20:]
    fig = plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx_20)), importances[sorted_idx_20], align='center')
    plt.yticks(range(len(sorted_idx_20)), np.array(top_features)[sorted_idx_20])
    plt.xlabel('Feature Importance'); plt.title('Top 20 Feature Importances')
    save_fig_to_s3(fig, s3_join(s3_reports_dir, "feature_importance.png"), outdir, s3_client=s3_client)

    fi_df = pd.DataFrame({"Feature": top_features, "Importance": importances})
    fi_top30 = fi_df.sort_values("Importance", ascending=False).head(30).reset_index(drop=True)
    save_df_to_s3(fi_top30, s3_join(s3_reports_dir, "top30_features.csv"), outdir, s3_client)

    # Accuracy by crime type
    logging.info("📈 Creating accuracy by crime type...")
    rows = []
    for name, code in inv_label_mapping.items():
        idx = (y_te_final == code)
        n = int(idx.sum())
        if n > 0:
            acc = accuracy_score(y_te_final[idx], final_model.predict(X_te_final[idx]))
            rows.append({"Crime Type": name, "Accuracy": acc, "Count": n})
    accuracy_df = pd.DataFrame(rows).sort_values(['Accuracy','Count'], ascending=[False,False])
    save_df_to_s3(accuracy_df, s3_join(s3_reports_dir, "accuracy_by_crime.csv"), outdir, s3_client)

    # Console summary
    logging.info("\n" + "="*80)
    logging.info("📈 MODEL EVALUATION SUMMARY")
    logging.info("="*80)
    overall_accuracy = accuracy_score(y_te_final, y_pred)
    logging.info("🔢 Overall Accuracy: %.4f", overall_accuracy)
    logging.info("🔠 Top 10 Crime Type Distribution:\n%s",
                 pd.Series(y_te_final).map(label_mapping).value_counts().head(10).to_string())
    class_report_df = pd.DataFrame(class_report).transpose()
    precision = class_report_df.loc['weighted avg', 'precision']
    recall    = class_report_df.loc['weighted avg', 'recall']
    f1        = class_report_df.loc['weighted avg', 'f1-score']
    logging.info("📊 Weighted Avg: Precision=%.4f  Recall=%.4f  F1=%.4f", precision, recall, f1)

    if not accuracy_df.empty:
        best = accuracy_df.iloc[0]
        suff = accuracy_df[accuracy_df['Count'] >= 100]
        if not suff.empty:
            worst = suff.iloc[-1]
            logging.info("🏆 Best: %s (Acc: %.4f, N=%d)", best['Crime Type'], best['Accuracy'], int(best['Count']))
            logging.info("⚠ Worst: %s (Acc: %.4f, N=%d)", worst['Crime Type'], worst['Accuracy'], int(worst['Count']))
        else:
            logging.info("⚠ Not enough samples (>=100) to report worst class reliably.")
    else:
        logging.info("⚠ Accuracy dataframe is empty")

    # Return a few metrics for manifest
    return {
        "overall_accuracy": float(overall_accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
    }

# ----------------------------- Main -----------------------------

def run(cfg: RunConfig):
    setup_logging()
    np.random.seed(cfg.seed)

    outdir = cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    # Load & prep
    df = load_dataset(cfg.s3_dataset_uri, cfg.test_min_year)
    df = basic_optimize(df)
    X, y, label_mapping = build_xy(df)
    X_train, X_test, y_train, y_test = time_split(X, y, df, cfg.split_date)
    X_train, X_test, label_encoders = fit_transform_label_encoders(X_train, X_test)
    sample_weight_train = compute_weights(y_train)

    # Feature selection
    top_features = cv_select_top_features(
        X_train, y_train, sample_weight_train, cfg.n_top_feats, cfg.seed, cfg.k_folds
    )

    X_tr_final = X_train[top_features].values
    X_te_final = X_test[top_features].values
    y_tr_final = y_train.values
    y_te_final = y_test.values

    # Train
    final_model = train_final_model(X_tr_final, y_tr_final, X_te_final, y_te_final, sample_weight_train)

    # Save models/pickles (local + S3)
    logging.info("💾 Saving artifacts to S3...")
    S3_MODELS_DIR = s3_join(cfg.s3_out_prefix, "artifacts")
    S3_REPORTS_DIR = s3_join(cfg.s3_out_prefix, "reports")

    save_joblib_to_s3(final_model,    s3_join(S3_MODELS_DIR, "xgboost_crimetype_model.pkl"), outdir, s3)
    save_joblib_to_s3(label_encoders, s3_join(S3_MODELS_DIR, "xgboost_label_encoders.pkl"), outdir, s3)
    save_joblib_to_s3(top_features,   s3_join(S3_MODELS_DIR, "xgboost_selected_features.pkl"), outdir, s3)
    save_joblib_to_s3(label_mapping,  s3_join(S3_MODELS_DIR, "xgboost_label_mapping.pkl"), outdir, s3)

    # Eval + plots + CSVs
    metrics_core = evaluate_and_plots(
        final_model, X_te_final, y_te_final, top_features, label_mapping,
        S3_REPORTS_DIR, outdir, s3_client=s3
    )

    # Manifest
    manifest = {
        "pipeline": "XGB Multiclass (CrimeType)",
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "s3_dataset_uri": cfg.s3_dataset_uri,
        "s3_out_prefix": cfg.s3_out_prefix,
        "split_date": cfg.split_date,
        "n_top_feats": cfg.n_top_feats,
        "top_features": top_features,
        "label_mapping": label_mapping,
        "metrics": metrics_core
    }
    man_path = outdir / "MANIFEST.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    upload_file(str(man_path), s3_join(cfg.s3_out_prefix, "MANIFEST.json"), s3)

    logging.info("\n" + "="*80)
    logging.info("✅ Done! All artifacts written under: %s", cfg.s3_out_prefix)

# ----------------------------- CLI -----------------------------

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Multiclass XGBoost (CrimeType) — AWS S3 VERSION")
    p.add_argument("--s3-dataset-uri", required=True,
                   help="S3 path (or local path) to Engineereddata Parquet dataset directory.")
    p.add_argument("--s3-out-prefix",  required=True,
                   help="S3 prefix for outputs, e.g. s3://bucket/project-folder/Processed/Models/xgb_multiclass")
    p.add_argument("--split-date",     default="2022-01-01",
                   help="Train/Test split date (YYYY-MM-DD). Default: 2022-01-01")
    p.add_argument("--n-top-feats",    type=int, default=30,
                   help="Number of top features to keep after CV selection. Default: 30")
    p.add_argument("--outdir",         default="/tmp/_out_xgb_multiclass",
                   help="Local output directory. Default: /tmp/_out_xgb_multiclass")
    p.add_argument("--no-upload",      action="store_true",
                   help="(Kept for parity) — uploads always enabled in this script.")
    args = p.parse_args()

    return RunConfig(
        s3_dataset_uri=args.s3_dataset_uri,
        s3_out_prefix=args.s3_out_prefix,
        split_date=args.split_date,
        n_top_feats=args.n_top_feats,
        outdir=Path(args.outdir),
        upload=not args.no_upload
    )

if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
