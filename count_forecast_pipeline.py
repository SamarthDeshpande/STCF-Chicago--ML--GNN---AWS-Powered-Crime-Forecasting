#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count Forecasting (Beat x Day) — SageMaker + S3
- Reads Parquet from S3 (pyarrow/s3fs)
- Trains Poisson regressors (HGB, Tweedie, optional XGB), optional ZINB
- Evaluates on a time-based test split
- Produces 7-day per-beat forecasts with 20/80 quantile bands
- Saves artifacts to /tmp (or container's ephemeral storage)
- Uploads results to s3://<bucket>/<prefix>/

Usage:
    python count_forecast_pipeline.py \
        --s3-input s3://dataset-chicago/project-folder/Processed/Engineereddata/ \
        --s3-bucket dataset-chicago \
        --s3-prefix-out project-folder/Processed/Models \
        --split-date 2022-05-29
"""

import os
import json
import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.dpi": 130})

# ML
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
import joblib

# Optional deps
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    _HAS_SM_ZI = True
except Exception:
    _HAS_SM_ZI = False

try:
    import holidays
    US_HOLIDAYS = holidays.US()
    def _is_holiday(d): return d in US_HOLIDAYS
except Exception:
    US_HOLIDAYS = None
    def _is_holiday(d): return False

import boto3

# ------------------------- Config -------------------------

LAGS  = [1, 3, 7, 14, 21]
ROLLS = [3, 7, 14, 28]
SPLIT_DATE_DEFAULT = pd.Timestamp("2022-01-01")

@dataclass
class RunConfig:
    s3_input: str
    s3_bucket: str
    s3_prefix_out: str
    split_date: str | None
    outdir: Path
    random_seed: int = 42
    quantile_low: float = 0.2
    quantile_high: float = 0.8
    val_frac: float = 0.15

# ------------------------- Utils -------------------------

def setup_logging():
    fmt = "[%(asctime)s] %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1.0
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def season_from_month(m: int) -> str:
    return {12:"Winter",1:"Winter",2:"Winter",
            3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",
            9:"Fall",10:"Fall",11:"Fall"}[int(m)]

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Weekday"]   = df["Date_Only"].dt.weekday.astype(int)
    df["IsWeekend"] = (df["Weekday"] >= 5).astype(int)
    df["Month"]     = df["Date_Only"].dt.month.astype(int)
    df["Season"]    = df["Month"].map(season_from_month)
    df["DoW_sin"]   = np.sin(2*np.pi*df["Weekday"]/7.0)
    df["DoW_cos"]   = np.cos(2*np.pi*df["Weekday"]/7.0)
    df["Mon_sin"]   = np.sin(2*np.pi*(df["Month"]-1)/12.0)
    df["Mon_cos"]   = np.cos(2*np.pi*(df["Month"]-1)/12.0)
    df["IsHoliday"] = 0
    if US_HOLIDAYS is not None:
        df["IsHoliday"] = df["Date_Only"].dt.date.map(lambda d: int(_is_holiday(d))).astype(int)
    return df

def ensure_node2vec_features(daily: pd.DataFrame, df_row_level: pd.DataFrame) -> pd.DataFrame:
    merged = daily.copy()
    df_rl = df_row_level.copy()
    merged["Beat"] = merged["Beat"].astype(str)
    df_rl["Beat"]  = df_rl["Beat"].astype(str)

    # Node2Vec embeddings (Embed_*)
    embed_cols = [c for c in df_rl.columns if c.startswith("Embed_")]
    if embed_cols:
        emb = df_rl.groupby("Beat")[embed_cols].first().reset_index()
        emb[embed_cols] = emb[embed_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        merged = merged.merge(emb, on="Beat", how="left")

    # Hour_Cluster (mode per beat)
    if "Hour_Cluster" in df_rl.columns:
        hc = (df_rl.groupby("Beat")["Hour_Cluster"]
                      .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                      .reset_index())
        hc["Hour_Cluster"] = pd.to_numeric(hc["Hour_Cluster"], errors="coerce").fillna(0).astype(int)
        merged = merged.merge(hc, on="Beat", how="left")

    new_cols = [c for c in merged.columns if c not in daily.columns]
    if new_cols:
        merged[new_cols] = merged[new_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return merged

def build_daily_beat_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Daily Beat x Date counts + lags/rolls + calendar + Stage-3 statics, dtype/NaN-safe.
    Requires columns: ['Beat', 'Date_TS'] and will use ['Hour_Cluster','Lat_Bin','Lng_Bin','Embed_*'] if present.
    """
    if "Beat" not in df_raw.columns or "Date_TS" not in df_raw.columns:
        raise ValueError("Input df must contain at least 'Beat' and 'Date_TS' columns.")

    df_ = df_raw[(df_raw["Beat"].notna()) & (df_raw["Date_TS"].notna())].copy()
    df_["Beat"] = df_["Beat"].astype(str)
    df_["Date_TS"] = pd.to_datetime(df_["Date_TS"], errors="coerce")
    df_ = df_[df_["Date_TS"].notna()].copy()
    df_ = df_[df_["Date_TS"].dt.year >= 2010].copy()
    df_["Date_Only"] = pd.to_datetime(df_["Date_TS"].dt.date)

    # Daily counts per Beat
    daily = (df_.groupby(["Beat", "Date_Only"])
               .size()
               .reset_index(name="CrimeCountPerDay")
               .sort_values(["Beat","Date_Only"]))

    # Lags & rolling means
    for lag in LAGS:
        daily[f"Crime_Lag_{lag}Day_Beat"] = daily.groupby("Beat")["CrimeCountPerDay"].shift(lag)
    for w in ROLLS:
        daily[f"Rolling_Avg_{w}Day_Beat"] = (
            daily.groupby("Beat")["CrimeCountPerDay"]
                 .rolling(window=w, min_periods=1).mean()
                 .reset_index(level=0, drop=True)
        )

    daily = add_calendar_features(daily)
    daily = ensure_node2vec_features(daily, df_)

    # Optional Stage-1/2 bins if present (Lat_Bin/Lng_Bin)
    for c in ["Lat_Bin", "Lng_Bin"]:
        if c not in daily.columns and c in df_.columns:
            tmp = df_.dropna(subset=[c]).copy()
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            last_vals = tmp.groupby("Beat")[c].last().reset_index()
            daily = daily.merge(last_vals, on="Beat", how="left")

    feat_like = [col for col in daily.columns if col != "CrimeCountPerDay"]
    daily[feat_like] = daily[feat_like].apply(pd.to_numeric, errors="coerce").fillna(0)
    daily["CrimeCountPerDay"] = (
        pd.to_numeric(daily["CrimeCountPerDay"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    )
    daily["Beat"] = daily["Beat"].astype(str)
    return daily

def feature_columns_available(df_daily: pd.DataFrame) -> list[str]:
    base = [f"Crime_Lag_{lag}Day_Beat" for lag in LAGS] + \
           [f"Rolling_Avg_{w}Day_Beat" for w in ROLLS] + \
           ["Weekday","IsWeekend","Month","Season","DoW_sin","DoW_cos","Mon_sin","Mon_cos","IsHoliday"]
    opt = [c for c in ["Hour_Cluster","Lat_Bin","Lng_Bin"] if c in df_daily.columns]
    embed_cols = [c for c in df_daily.columns if c.startswith("Embed_")]
    cols = [c for c in base + opt + embed_cols if c in df_daily.columns]
    # De-dup while keeping order
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

def choose_split_date(df_daily: pd.DataFrame, split_date_arg: str | None) -> pd.Timestamp:
    if split_date_arg:
        return pd.to_datetime(split_date_arg)
    dates = np.array(sorted(df_daily["Date_Only"].unique()))
    if len(dates) == 0:
        raise ValueError("No dates found after preprocessing.")
    if SPLIT_DATE_DEFAULT in dates:
        return SPLIT_DATE_DEFAULT
    return dates[int(len(dates)*0.8)]

def compute_sample_weights(train_df: pd.DataFrame, gamma: float = 1.0) -> np.ndarray:
    tmp = train_df.copy()
    if "Rolling_Avg_7Day_Beat" not in tmp.columns:
        tmp["Rolling_Avg_7Day_Beat"] = tmp.groupby("Beat")["CrimeCountPerDay"]\
            .transform(lambda s: s.rolling(7, min_periods=1).mean())

    mae_beat = (tmp.assign(err=(tmp["CrimeCountPerDay"] - tmp["Rolling_Avg_7Day_Beat"]).abs())
                  .groupby("Beat")["err"].mean())
    med = float(np.median(mae_beat.values)) if len(mae_beat) else 1.0
    w_hard = (mae_beat / max(med, 1e-6)).pow(gamma).clip(0.5, 3.0)

    cnt = train_df.groupby("Beat").size().astype(float)
    inv = (1.0 / np.sqrt(cnt + 1e-6))
    inv = inv / inv.mean()

    w_map = 0.7 * w_hard + 0.3 * inv
    w_series = train_df["Beat"].map(w_map).astype(float)
    w_series = w_series.fillna(1.0).clip(0.25, 4.0)
    return w_series.values

# ------------------------- Model blocks -------------------------

def train_tweedie_poisson(Xtr, ytr, sample_weight=None):
    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        TweedieRegressor(power=1.0, alpha=1e-4, max_iter=2000)
    )
    model.fit(Xtr, ytr, tweedieregressor__sample_weight=sample_weight)
    return model

def train_hgb_poisson(Xtr, ytr, sample_weight=None):
    ytr_pos = np.clip(ytr, 0, None)
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.07,
        max_depth=14,
        max_iter=800,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    model.fit(Xtr, ytr_pos, sample_weight=sample_weight)
    return model

def train_xgb_poisson(Xtr, ytr, Xva=None, yva=None, sample_weight=None):
    if not _HAS_XGB:
        return None
    params = dict(
        objective="count:poisson",
        tree_method="hist",
        random_state=42,
        n_estimators=1200,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=1
    )
    model = xgb.XGBRegressor(**params)
    if Xva is not None and yva is not None:
        try:
            model.fit(Xtr, ytr, sample_weight=sample_weight,
                      eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=75)
        except TypeError:
            model.fit(Xtr, ytr, sample_weight=sample_weight, verbose=False)
    else:
        model.fit(Xtr, ytr, sample_weight=sample_weight, verbose=False)
    return model

def train_quantile_bands(Xtr, ytr, sample_weight=None, a_low=0.2, a_hi=0.8):
    qlow = GradientBoostingRegressor(loss="quantile", alpha=a_low,
                                     learning_rate=0.05, max_depth=3, n_estimators=600,
                                     subsample=0.9, random_state=42)
    qhi  = GradientBoostingRegressor(loss="quantile", alpha=a_hi,
                                     learning_rate=0.05, max_depth=3, n_estimators=600,
                                     subsample=0.9, random_state=42)
    qlow.fit(Xtr, ytr, sample_weight=sample_weight)
    qhi.fit(Xtr, ytr, sample_weight=sample_weight)
    return {"q_low": qlow, "q_hi": qhi, "alpha_low": a_low, "alpha_hi": a_hi}

def predict_safe(model, X):
    try:
        yhat = np.asarray(model.predict(X), dtype=float)
    except NotFittedError:
        return np.zeros(X.shape[0], dtype=float)
    yhat = np.where(np.isfinite(yhat), yhat, 0.0)
    return np.clip(yhat, 0.0, None)

def evaluate_regression(y_true, y_pred, name="", round_to_int=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)):
        return {"model": name, "rounded": round_to_int, "MAE": np.inf, "RMSE": np.inf, "SMAPE": np.inf}
    y_pred_use = np.rint(y_pred).astype(int) if round_to_int else y_pred
    mae  = mean_absolute_error(y_true, y_pred_use)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_use))
    s_m  = smape(y_true, y_pred_use)
    return {"model": name, "rounded": round_to_int, "MAE": mae, "RMSE": rmse, "SMAPE": s_m}

# ------------------------- Forecasting -------------------------

def forecast_next_days_count(best_model_name: str,
                             models_dict: dict,
                             df_daily: pd.DataFrame,
                             feat_cols: list[str],
                             days: int = 7,
                             q_models: dict | None = None) -> pd.DataFrame:
    df_sorted = df_daily.sort_values(["Beat","Date_Only"]).copy()
    df_sorted["Beat"] = df_sorted["Beat"].astype(str)

    # Prepare last state by beat
    last_state = {}
    for beat, g in df_sorted.groupby("Beat", sort=False):
        last_date = g["Date_Only"].iloc[-1]
        last_state[beat] = {
            "last_date": pd.to_datetime(last_date),
            "counts": g["CrimeCountPerDay"].iloc[-max(ROLLS + LAGS):].astype(float).tolist()
        }

    stat_cols = [c for c in ["Hour_Cluster","Lat_Bin","Lng_Bin"] if c in df_daily.columns]
    embed_cols = [c for c in df_daily.columns if c.startswith("Embed_")]
    static_cols = stat_cols + embed_cols

    def _prep_one_row(date, counts, static_vals):
        row = {}
        for lag in LAGS:
            row[f"Crime_Lag_{lag}Day_Beat"] = (counts[-lag] if len(counts) >= lag else 0.0)
        for w in ROLLS:
            row[f"Rolling_Avg_{w}Day_Beat"] = float(np.mean(counts[-w:])) if len(counts) else 0.0
        row["Date_Only"] = pd.to_datetime(date)
        row["Weekday"]   = int(row["Date_Only"].weekday())
        row["IsWeekend"] = int(row["Weekday"] >= 5)
        row["Month"]     = int(row["Date_Only"].month)
        row["Season"]    = season_from_month(row["Month"])
        row["DoW_sin"]   = float(np.sin(2*np.pi*row["Weekday"]/7.0))
        row["DoW_cos"]   = float(np.cos(2*np.pi*row["Weekday"]/7.0))
        row["Mon_sin"]   = float(np.sin(2*np.pi*(row["Month"]-1)/12.0))
        row["Mon_cos"]   = float(np.cos(2*np.pi*(row["Month"]-1)/12.0))
        row["IsHoliday"] = int(_is_holiday(row["Date_Only"].date())) if US_HOLIDAYS is not None else 0
        for c in static_cols:
            row[c] = static_vals.get(c, 0.0)
        df1 = pd.DataFrame([row])
        for c in feat_cols:
            if c not in df1.columns:
                df1[c] = 0.0
        df1[feat_cols] = df1[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return df1[feat_cols].astype(float).values

    model_obj = models_dict[best_model_name]
    forecasts = []
    for beat, g in df_sorted.groupby("Beat", sort=False):
        static_vals = {}
        if static_cols:
            last_static = g.iloc[-1][static_cols].to_dict()
            static_vals.update({k: (0.0 if pd.isna(v) else float(v)) for k, v in last_static.items()})

        state_date = pd.to_datetime(last_state[beat]["last_date"])
        counts = list(map(int, np.rint(last_state[beat]["counts"])))

        for _ in range(days):
            target_date = pd.to_datetime(state_date) + pd.Timedelta(days=1)
            Xrow = _prep_one_row(target_date, counts, static_vals)

            if isinstance(model_obj, tuple):  # ZINB stored as (res, True)
                try:
                    yhat = float(model_obj[0].predict(sm.add_constant(Xrow, has_constant="add"))[0])
                except Exception:
                    yhat = 0.0
            else:
                yhat = float(predict_safe(model_obj, Xrow)[0])

            yhat_int = int(np.rint(max(0.0, yhat)))
            ql = float(np.clip(q_models["q_low"].predict(Xrow)[0],  0, None)) if q_models else np.nan
            qh = float(np.clip(q_models["q_hi"].predict(Xrow)[0],   0, None)) if q_models else np.nan
            if np.isfinite(ql) and np.isfinite(qh) and ql > qh:
                ql, qh = qh, ql

            forecasts.append({
                "Date": pd.to_datetime(target_date),
                "Beat": beat,
                "CrimeCount_Pred": yhat_int,
                "q_low": ql, "q_hi": qh
            })
            counts.append(yhat_int)
            state_date = target_date
    return pd.DataFrame(forecasts)

# ------------------------- I/O helpers -------------------------

def save_fig(fig, outdir: Path, name: str):
    path = str(outdir / name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def save_csv(df: pd.DataFrame, outdir: Path, name: str):
    df.to_csv(outdir / name, index=False)

def save_parquet(df: pd.DataFrame, outdir: Path, name: str):
    df.to_parquet(outdir / name, index=False)

def upload_many_to_s3(file_names: list[str], outdir: Path, bucket: str, prefix: str):
    s3 = boto3.client("s3")
    logging.info("Uploading artifacts to S3...")
    for fname in file_names:
        fpath = outdir / fname
        if fpath.exists():
            s3_key = f"{prefix}/{fname}"
            s3.upload_file(str(fpath), bucket, s3_key)
            logging.info("  ✅ s3://%s/%s", bucket, s3_key)
        else:
            logging.info("  ⚠ Skipped missing: %s", fname)

# ------------------------- Main pipeline -------------------------

def run(cfg: RunConfig):
    np.random.seed(cfg.random_seed)
    warnings.filterwarnings("ignore")
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Local temp output dir: %s", cfg.outdir)

    # Load
    logging.info("Reading Parquet from: %s", cfg.s3_input)
    df = pd.read_parquet(cfg.s3_input, engine="pyarrow", storage_options={"anon": False})
    logging.info("Loaded shape: %s", df.shape)

    # Minimal schema
    req = {"Beat", "Date_TS"}
    missing = req - set(df.columns)
    if missing:
        raise AssertionError(f"Input file is missing columns: {missing}")

    df["Beat"] = df["Beat"].astype(str)

    # Build features (daily)
    logging.info("Building daily Beat×Date frame + features...")
    daily = build_daily_beat_frame(df)
    feat_cols = feature_columns_available(daily)
    joblib.dump(feat_cols, cfg.outdir / "count_feature_cols.pkl")
    save_parquet(daily, cfg.outdir, "beat_day_features.parquet")

    # Split
    split_date = choose_split_date(daily, cfg.split_date)
    logging.info("Split date: %s", pd.Timestamp(split_date).date())

    train_mask = daily["Date_Only"] < split_date
    test_mask  = daily["Date_Only"] >= split_date

    cols_for_data = feat_cols + ["CrimeCountPerDay","Beat","Date_Only"]
    data = daily[cols_for_data].copy()

    # Safety numeric conversion + fill
    for c in feat_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)

    # X/y
    X_train = data.loc[train_mask, feat_cols].astype(float).values
    y_train = data.loc[train_mask, "CrimeCountPerDay"].astype(float).values
    X_test  = data.loc[test_mask,  feat_cols].astype(float).values
    y_test  = data.loc[test_mask,  "CrimeCountPerDay"].astype(float).values
    beats_test = data.loc[test_mask, "Beat"].astype(str).values
    dates_test = data.loc[test_mask, "Date_Only"].values

    # Validation slice
    split_idx = max(1, int(len(X_train) * (1 - cfg.val_frac)))
    Xtr, Xva = X_train[:split_idx], X_train[split_idx:]
    ytr, yva = y_train[:split_idx], y_train[split_idx:]

    # Sample weights
    train_df = data.loc[train_mask, :].copy()
    sample_weight = compute_sample_weights(train_df, gamma=1.0)

    # Train models
    models = {}

    logging.info("Training Tweedie Poisson...")
    models["TweediePoisson"] = train_tweedie_poisson(Xtr, ytr, sample_weight=sample_weight[:len(Xtr)])

    logging.info("Training HGB Poisson (depth=14, leaf=20)...")
    models["HGBPoisson"] = train_hgb_poisson(Xtr, ytr, sample_weight=sample_weight[:len(Xtr)])

    if _HAS_XGB:
        logging.info("Training XGB Poisson...")
        xgb_model = train_xgb_poisson(Xtr, ytr, Xva, yva, sample_weight=sample_weight[:len(Xtr)])
        if xgb_model is not None:
            models["XGBPoisson"] = xgb_model
    else:
        logging.info("xgboost not installed — skipping XGB.")

    # Optional ZINB (can fail with singular matrices)
    if _HAS_SM_ZI:
        logging.info("Fitting ZINB (full)...")
        try:
            X_sm = sm.add_constant(X_train, has_constant="add")
            infl_cols = slice(0, min(X_sm.shape[1], 6))  # const + first 5 feats
            zinb = ZeroInflatedNegativeBinomialP(y_train, X_sm, exog_infl=X_sm[:, infl_cols], inflation='logit')
            zinb_res = zinb.fit(method="bfgs", maxiter=200, disp=0)
            models["ZINB_full"] = (zinb_res, True)
            logging.info("ZINB fit completed.")
        except Exception as e:
            logging.info("Skipping ZINB (reason: %s)", e)
    else:
        logging.info("statsmodels not installed — skipping ZINB.")

    logging.info("Training quantile bands (%.2f / %.2f)...", cfg.quantile_low, cfg.quantile_high)
    q_models = train_quantile_bands(Xtr, ytr, sample_weight=sample_weight[:len(Xtr)],
                                    a_low=cfg.quantile_low, a_hi=cfg.quantile_high)

    # Evaluate on TEST
    logging.info("Evaluating on TEST...")
    results, preds_store = [], {}

    for name, mdl in models.items():
        if name.startswith("ZINB"):
            try:
                Xte_sm = sm.add_constant(X_test, has_constant="add")
                yhat = np.asarray(mdl[0].predict(Xte_sm), dtype=float)
                yhat = np.clip(np.where(np.isfinite(yhat), yhat, 0.0), 0.0, None)
            except Exception:
                yhat = np.full_like(y_test, np.nan, dtype=float)
        else:
            yhat = predict_safe(mdl, X_test)

        preds_store[name] = yhat
        results.append(evaluate_regression(y_test, yhat, name=name, round_to_int=False))
        results.append(evaluate_regression(y_test, yhat, name=name, round_to_int=True))

    results_df = pd.DataFrame(results).sort_values(["rounded","MAE"])
    save_csv(results_df, cfg.outdir, "model_results_test.csv")
    logging.info("Test results (lowest MAE is best):\n%s", results_df.head(10).to_string(index=False))

    # Best by integer MAE
    best_row  = results_df[results_df["rounded"]==True].sort_values("MAE").iloc[0]
    best_name = str(best_row["model"])
    logging.info("Best model by integer MAE: %s | MAE=%.3f, RMSE=%.3f",
                 best_name, best_row["MAE"], best_row["RMSE"])

    # Save TEST predictions + quantiles
    best_pred_cont = np.clip(preds_store[best_name], 0, None)
    best_pred_int  = np.rint(best_pred_cont).astype(int)

    q_low  = np.clip(q_models["q_low"].predict(X_test),  0, None)
    q_hi   = np.clip(q_models["q_hi"].predict(X_test),   0, None)
    q_low, q_hi = np.minimum(q_low, q_hi), np.maximum(q_low, q_hi)

    df_pred = pd.DataFrame({
        "Date": pd.to_datetime(dates_test),
        "Beat": beats_test,
        "y_true": y_test.astype(int),
        "y_pred_cont": best_pred_cont,
        "y_pred_int": best_pred_int,
        "q_low": q_low,
        "q_hi": q_hi
    }).sort_values(["Date","Beat"])
    save_csv(df_pred, cfg.outdir, "test_predictions_best.csv")

    avg_actual = float(df_pred["y_true"].mean())
    med_actual = float(df_pred["y_true"].median())
    with open(cfg.outdir / "summary.txt", "w", encoding="utf-8", errors="ignore") as f:
        f.write(f"Average actual per beat/day: {avg_actual:.2f}\n")
        f.write(f"Median actual per beat/day: {med_actual:.2f}\n")
        f.write(f"Best model: {best_name}\n")
        f.write(f"Best int-MAE: {best_row['MAE']:.3f}\n")
        f.write(f"Best int-RMSE: {best_row['RMSE']:.3f}\n")

    # Visuals
    fig = plt.figure(figsize=(8,6))
    plt.scatter(df_pred["y_true"], df_pred["y_pred_int"], alpha=0.25, s=8)
    lim = max(df_pred["y_true"].max(), df_pred["y_pred_int"].max())
    plt.plot([0,lim],[0,lim],"r--")
    plt.xlabel("Actual"); plt.ylabel("Predicted (integer)")
    plt.title(f"Actual vs Predicted — {best_name}")
    save_fig(fig, cfg.outdir, "actual_vs_pred_best.png")

    err = df_pred["y_pred_int"] - df_pred["y_true"]
    fig = plt.figure(figsize=(8,5))
    sns.histplot(err, bins=60, kde=True)
    plt.axvline(0, color="red", lw=1)
    plt.title(f"Error Distribution — {best_name}")
    plt.xlabel("Error (Pred - Actual)")
    save_fig(fig, cfg.outdir, "error_distribution_best.png")

    per_beat = (df_pred.groupby("Beat")
                .apply(lambda g: pd.Series({
                    "MAE": mean_absolute_error(g["y_true"], g["y_pred_int"]),
                    "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred_int"])),
                    "N": len(g)
                }))
                .reset_index()
                .sort_values("MAE", ascending=False))
    save_csv(per_beat, cfg.outdir, "per_beat_metrics.csv")

    top_worst = per_beat[per_beat["N"]>=50].head(25)
    if not top_worst.empty:
        fig = plt.figure(figsize=(9,6))
        sns.barplot(data=top_worst, x="MAE", y="Beat")
        plt.title(f"Worst 25 Beats by MAE — {best_name} (min 50 samples)")
        save_fig(fig, cfg.outdir, "worst_beats_mae.png")

    # 7-day forecast
    logging.info("Forecasting next 7 days with best model + quantiles...")
    forecast_df = forecast_next_days_count(best_name, models, daily, feat_cols, days=7, q_models=q_models)
    forecast_df = forecast_df.sort_values(["Date","Beat"]).reset_index(drop=True)
    save_csv(forecast_df, cfg.outdir, "forecast_7d_per_beat.csv")

    # Citywide chart
    city = forecast_df.groupby("Date")["CrimeCount_Pred"].sum().reset_index()
    fig = plt.figure(figsize=(12,6))
    plt.plot(city["Date"], city["CrimeCount_Pred"], marker="o")
    plt.title(f"Citywide 7-Day Forecast — {best_name}")
    plt.xlabel("Date"); plt.ylabel("Predicted Count"); plt.grid(True)
    save_fig(fig, cfg.outdir, "citywide_forecast.png")

    # Persist models/artifacts locally
    joblib.dump(models, cfg.outdir / "count_models.pkl")

    manifest = {
        "pipeline": "Count forecasting (Poisson/GBM/XGB + quantiles)",
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "split_date": str(pd.Timestamp(split_date).date()),
        "lags": LAGS, "rolls": ROLLS,
        "features_used": feat_cols,
        "models_trained": list(models.keys()),
        "best_model": best_name,
        "best_int_mae": float(best_row["MAE"]),
        "avg_actual_per_beat_day": float(avg_actual),
        "median_actual_per_beat_day": float(med_actual),
        "quantiles": {"low": cfg.quantile_low, "high": cfg.quantile_high},
        "outputs_dir": str(cfg.outdir.resolve())
    }
    with open(cfg.outdir / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Extra dashboard-friendly exports
    city_test = (df_pred.groupby("Date")["y_true"].sum().reset_index()
                 .rename(columns={"y_true":"Citywide_Actual"}))
    save_csv(city_test, cfg.outdir, "citywide_actual_test.csv")
    save_csv(df_pred,   cfg.outdir, "beat_day_test_actuals_preds.csv")

    # Beat coordinates (median of raw lat/lon per Beat)
    coord_cols = [c for c in ["Latitude","Longitude"] if c in df.columns]
    if len(coord_cols) == 2:
        coords = (df.dropna(subset=coord_cols)
                    .groupby("Beat")[["Latitude","Longitude"]]
                    .median()
                    .reset_index())
        save_csv(coords, cfg.outdir, "per_beat_location.csv")
        logging.info("Saved per_beat_location.csv")
    else:
        logging.info("Raw data lacks Latitude/Longitude — per_beat_location.csv not created.")

    # Uploads
    uploads = [
        "test_predictions_best.csv",
        "forecast_7d_per_beat.csv",
        "beat_day_test_actuals_preds.csv",
        "model_results_test.csv",
        "per_beat_metrics.csv",
        "actual_vs_pred_best.png",
        "error_distribution_best.png",
        "worst_beats_mae.png",
        "citywide_forecast.png",
        "count_models.pkl",
        "count_feature_cols.pkl",
        "MANIFEST.json",
        "summary.txt",
        "beat_day_features.parquet",
        "per_beat_location.csv"
    ]
    upload_many_to_s3(uploads, cfg.outdir, cfg.s3_bucket, cfg.s3_prefix_out)
    logging.info("Done. All artifacts saved to %s and uploaded to s3://%s/%s/",
                 cfg.outdir, cfg.s3_bucket, cfg.s3_prefix_out)

# ------------------------- CLI -------------------------

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Beat×Day Count Forecasting (Poisson/GBM/XGB + Quantiles)")
    parser.add_argument("--s3-input", required=True,
                        help="S3 path (or local path) to Engineereddata Parquet (folder or file).")
    parser.add_argument("--s3-bucket", required=True, help="Target S3 bucket for uploads.")
    parser.add_argument("--s3-prefix-out", required=True,
                        help="Target S3 prefix (no trailing slash) for uploads, e.g. project-folder/Processed/Models")
    parser.add_argument("--split-date", default=None,
                        help="Optional split date (YYYY-MM-DD). If omitted, uses default or 80/20 time split.")
    parser.add_argument("--outdir", default="/tmp", help="Local output directory (default: /tmp).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    return RunConfig(
        s3_input=args.s3_input,
        s3_bucket=args.s3_bucket,
        s3_prefix_out=args.s3_prefix_out,
        split_date=args.split_date,
        outdir=outdir
    )

if __name__ == "__main__":
    setup_logging()
    cfg = parse_args()
    run(cfg)
