#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN Hotspot Prediction — SageMaker + S3 (Weekly Aggregation)
- Input:  s3://<bucket>/project-folder/Processed/Engineereddata/
- Output: s3://<bucket>/project-folder/Processed/Models/
- Local:  /tmp/_out_gnn (default)
- Model:  GAT (PyG) if available; MLP fallback otherwise
- Exports: gnn_hotspot_predictions.csv, metrics.json, training_loss.png,
           gnn_model.pth, gnn_scaler.pkl, gnn_threshold.pkl, (optional) per_beat_location.csv
"""

import os
import json
import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try PyG; fallback gracefully
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
    _HAS_PYG = True
except Exception as e:
    _HAS_PYG = False

# Plots (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    roc_auc_score, average_precision_score, confusion_matrix
)

import joblib
import boto3

warnings.filterwarnings("ignore")

# ----------------------------- Config -----------------------------

@dataclass
class RunConfig:
    s3_input: str
    s3_bucket: str
    s3_prefix_out: str
    outdir: Path
    epochs: int = 200
    lr: float = 0.002
    weight_decay: float = 4e-4
    hidden: int = 128
    dropout: float = 0.4
    dist_threshold: float = 0.06
    seed: int = 42
    test_size: float = 0.20
    log_every: int = 10
    upload: bool = True

# ----------------------------- Utils -----------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(t, device):
    return t.to(device) if hasattr(t, "to") else t

def upload_many_to_s3(file_names, outdir: Path, bucket: str, prefix: str):
    s3 = boto3.client("s3")
    logging.info("Uploading artifacts to S3...")
    for fname in file_names:
        fpath = outdir / fname
        if fpath.exists():
            key = f"{prefix}/{fname}"
            s3.upload_file(str(fpath), bucket, key)
            logging.info("  ✅ s3://%s/%s", bucket, key)
        else:
            logging.info("  ⏭ Skipped (not found): %s", fname)

# ----------------------------- Data -----------------------------

def load_engineered_df(s3_input: str) -> pd.DataFrame:
    logging.info("Reading Parquet from: %s", s3_input)
    df = pd.read_parquet(s3_input, engine="pyarrow", storage_options={"anon": False})
    logging.info("Loaded: %s", df.shape)

    # Basic cleanup for possible merged duplicates (_x/_y)
    for base in ['CrimeCountPerDay', 'Is_CrimeOutlier', 'CrimeCountPerDay_Capped']:
        for suffix in ['_x', '_y']:
            col = base + suffix
            if col in df.columns:
                if base in df.columns:
                    df[base] = df[base].combine_first(df[col])
                else:
                    df[base] = df[col]
                df.drop(col, axis=1, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()].drop_duplicates()

    required = {"Beat", "Date_TS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Beat"] = df["Beat"].astype(str)
    df["Date_TS"] = pd.to_datetime(df["Date_TS"], errors="coerce")
    df = df[df["Date_TS"].notna()]
    df = df[df["Date_TS"].dt.year >= 2010].copy()

    # Weekly key used as "Date_Only" parity
    df["Week"] = df["Date_TS"].dt.to_period("W").dt.start_time
    return df

def build_weekly_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Count per (Beat, Week) then per-beat 75th percentile threshold
    beat_weekly = df.groupby(["Beat", "Week"]).size().reset_index(name="crime_count")
    thresholds = beat_weekly.groupby("Beat")["crime_count"].quantile(0.75).reset_index()
    thresholds.columns = ["Beat", "threshold"]
    beat_weekly = beat_weekly.merge(thresholds, on="Beat", how="left")
    beat_weekly["is_hotspot"] = (beat_weekly["crime_count"] > beat_weekly["threshold"]).astype(int)

    # Use latest row per (Beat, Week) as carrier of row-level engineered features
    latest = df.sort_values("Date_TS").drop_duplicates(["Beat", "Week"], keep="last")
    latest = latest.merge(beat_weekly[["Beat", "Week", "is_hotspot"]], on=["Beat", "Week"], how="left")
    return latest

def select_features(latest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    # Only keep features that exist
    feature_cols = [
        'Crime_Count_Last7Days', 'Rolling_Avg_3Day_Beat', 'Crime_Lag_1Day_Beat', 'Crime_Lag_7Day_Beat',
        'Arrest_Rate_Percent', 'Domestic_Crime_Rate_Percent', 'Arrest_Rate_Roll3', 'Arrest_Rate_Roll7',
        # One-hot crime mix (if present)
        'ASSAULT', 'BATTERY', 'BURGLARY', 'CRIMINAL DAMAGE', 'DECEPTIVE PRACTICE',
        'MOTOR VEHICLE THEFT', 'NARCOTICS', 'OTHER', 'ROBBERY', 'THEFT',
        # Time/context
        'Hour', 'Weekday', 'Month', 'IsWeekend', 'Is_Covid_Era'
    ]
    feature_cols = [c for c in feature_cols if c in latest.columns]
    latest[feature_cols] = latest[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(latest[feature_cols])
    y = latest["is_hotspot"].fillna(0).astype(int).values
    return X, y, feature_cols, scaler

# ----------------------------- Graph -----------------------------

def build_edges_from_bins(latest: pd.DataFrame, dist_threshold: float) -> torch.Tensor:
    """
    Build undirected edges using (Lat_Bin, Lng_Bin) proximity with L2 distance <= threshold.
    Fallback: chain graph if none found or bins missing.
    """
    latest["beat_idx"] = latest["Beat"].astype("category").cat.codes
    if {"Lat_Bin", "Lng_Bin"}.issubset(latest.columns):
        coords = latest.groupby("beat_idx")[["Lat_Bin", "Lng_Bin"]].first().dropna()
        edges = []
        idxs = coords.index.to_list()
        arr = coords[["Lat_Bin", "Lng_Bin"]].values.astype(float)
        for i, ai in zip(idxs, arr):
            d = np.linalg.norm(arr - ai, axis=1)
            nbrs = [j for j, dist in zip(idxs, d) if (dist <= dist_threshold and j != i)]
            edges.extend([[i, j] for j in nbrs])
        if not edges:
            edges = [[i, i+1] for i in range(len(idxs)-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        n_nodes = latest["beat_idx"].nunique()
        edges = [[i, i+1] for i in range(max(0, n_nodes-1))]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

# ----------------------------- Models -----------------------------

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.9, gamma: float = 2.5, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

# GAT (PyG) or MLP fallback
if _HAS_PYG:
    class GAT(nn.Module):
        def __init__(self, in_ch, hid, out_ch, p=0.4):
            super().__init__()
            self.conv1 = GATConv(in_ch, hid, heads=4, concat=True)
            self.bn1   = nn.BatchNorm1d(hid * 4)
            self.conv2 = GATConv(hid * 4, hid, heads=1, concat=True)
            self.bn2   = nn.BatchNorm1d(hid)
            self.mlp   = nn.Sequential(
                nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(p),
                nn.Linear(hid//2, out_ch)
            )

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.bn1(x).relu()
            x = self.conv2(x, edge_index)
            x = self.bn2(x).relu()
            return self.mlp(x)
else:
    class MLP(nn.Module):
        def __init__(self, in_ch, hid, out_ch, p=0.4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_ch, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(p),
                nn.Linear(hid, hid//2), nn.BatchNorm1d(hid//2), nn.ReLU(), nn.Dropout(p),
                nn.Linear(hid//2, out_ch)
            )

        def forward(self, x):
            return self.net(x)

# ----------------------------- Training/Eval -----------------------------

def train_and_eval(X, y, latest, cfg: RunConfig, outdir: Path, device: torch.device):
    # Tensors
    x = torch.tensor(np.nan_to_num(X), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Graph (if PyG)
    if _HAS_PYG:
        edge_index = build_edges_from_bins(latest.copy(), cfg.dist_threshold)
        data = Data(x=x, edge_index=edge_index, y=y_tensor)
    else:
        data = None

    # Split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    train_mask = torch.zeros(len(y), dtype=torch.bool); train_mask[train_idx] = True
    test_mask  = torch.zeros(len(y), dtype=torch.bool);  test_mask[test_idx]  = True

    # Model
    in_ch = x.shape[1]; out_ch = 2
    if _HAS_PYG:
        model = GAT(in_ch, cfg.hidden, out_ch, p=cfg.dropout)
    else:
        model = MLP(in_ch, cfg.hidden, out_ch, p=cfg.dropout)
    model = model.to(device)

    # Class weights & optimizer
    classes = np.unique(y)
    cw = compute_class_weight('balanced', classes=classes, y=y)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    criterion = WeightedFocalLoss(alpha=0.9, gamma=2.5, weight=cw_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

    # Train
    logging.info("🧠 Training...")
    loss_history = []
    for epoch in range(cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        if _HAS_PYG:
            out = model(to_device(x, device), to_device(edge_index, device))
        else:
            out = model(to_device(x, device))
        loss = criterion(out[train_mask.to(device)], y_tensor[train_mask].to(device))
        loss.backward(); optimizer.step(); scheduler.step()
        loss_history.append(float(loss.item()))
        if epoch % cfg.log_every == 0:
            logging.info("Epoch %d, Loss: %.4f", epoch, loss.item())

    # Eval
    model.eval()
    with torch.no_grad():
        if _HAS_PYG:
            logits_all = model(to_device(x, device), to_device(edge_index, device))
        else:
            logits_all = model(to_device(x, device))
        probs_all = logits_all.softmax(dim=1)[:, 1].detach().cpu().numpy()
        y_all = y_tensor.cpu().numpy()
        p_test = probs_all[test_mask.cpu().numpy()]
        y_test = y_all[test_mask.cpu().numpy()]

    prec, rec, thr = precision_recall_curve(y_test, p_test)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1s)) if len(f1s) else 0
    best_thr = float(thr[best_idx]) if len(thr) else 0.5
    preds = (p_test >= best_thr).astype(int)
    report_dict = classification_report(y_test, preds, output_dict=True)
    report_txt  = classification_report(y_test, preds, target_names=['Not Hotspot', 'Hotspot'])

    logging.info("🔍 Best Threshold = %.3f | F1 = %.3f", best_thr, f1s[best_idx] if len(f1s) else float('nan'))
    logging.info("\n📋 Final Test Report:\n%s", report_txt)

    # Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "gnn_model.pth")
    plt.figure(figsize=(8,5))
    plt.plot(range(len(loss_history)), loss_history)
    plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "training_loss.png"); plt.close()

    metrics = {
        "classification_report": report_dict,
        "best_threshold": best_thr,
        "f1_score": float(f1s[best_idx]) if len(f1s) else float("nan"),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "avg_precision": float(average_precision_score(y_test, p_test)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "used_model": "GAT" if _HAS_PYG else "MLP_fallback"
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, probs_all, best_thr, metrics

# ----------------------------- Export -----------------------------

def export_predictions(latest: pd.DataFrame,
                       probs_all: np.ndarray,
                       best_thr: float,
                       df_raw: pd.DataFrame,
                       outdir: Path) -> Path:
    pred_df = latest[["Beat", "Week"]].copy()
    pred_df["prob_hotspot"] = probs_all
    pred_df["Prediction"]   = (pred_df["prob_hotspot"] >= best_thr).astype(int)
    pred_df = pred_df.rename(columns={"Week": "Date_Only"})
    pred_df["Date_Only"] = pd.to_datetime(pred_df["Date_Only"]).dt.date

    # Optional coords
    coord_cols = None
    if {"Latitude","Longitude"}.issubset(df_raw.columns):
        coord_cols = ["Latitude","Longitude"]
    elif {"Lat","Lng"}.issubset(df_raw.columns):
        df_raw = df_raw.rename(columns={"Lat":"Latitude","Lng":"Longitude"})
        coord_cols = ["Latitude","Longitude"]

    if coord_cols:
        coords = (df_raw.dropna(subset=coord_cols)
                        .groupby("Beat")[coord_cols]
                        .median()
                        .reset_index())
        coords.to_csv(outdir / "per_beat_location.csv", index=False)
        pred_df = pred_df.merge(coords, on="Beat", how="left")
        logging.info("📍 per_beat_location.csv saved.")
    else:
        logging.info("⚠ No raw Latitude/Longitude; per_beat_location.csv not created.")

    csv_path = outdir / "gnn_hotspot_predictions.csv"
    pred_df.to_csv(csv_path, index=False)

    if not pred_df.empty:
        logging.info("📆 Prediction weeks: %s → %s",
                     pred_df["Date_Only"].min(), pred_df["Date_Only"].max())
    return csv_path

# ----------------------------- Main -----------------------------

def run(cfg: RunConfig):
    setup_logging()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s | PyG: %s", device, _HAS_PYG)

    # I/O
    outdir = cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load & prepare
    df = load_engineered_df(cfg.s3_input)
    latest = build_weekly_labels(df)
    X, y, feat_cols, scaler = select_features(latest)

    # Train & evaluate
    model, probs_all, best_thr, metrics = train_and_eval(X, y, latest, cfg, outdir, device)

    # Save scaler & threshold
    joblib.dump(scaler, outdir / "gnn_scaler.pkl")
    joblib.dump(best_thr, outdir / "gnn_threshold.pkl")

    # Export predictions (CSV) and optional coordinates
    csv_path = export_predictions(latest, probs_all, best_thr, df, outdir)

    logging.info("✅ All local outputs in: %s", outdir.resolve())

    # Upload
    if cfg.upload:
        to_upload = [
            "gnn_model.pth",
            "gnn_scaler.pkl",
            "gnn_threshold.pkl",
            "training_loss.png",
            "metrics.json",
            "gnn_hotspot_predictions.csv",
            "per_beat_location.csv"  # may not exist
        ]
        upload_many_to_s3(to_upload, outdir, cfg.s3_bucket, cfg.s3_prefix_out)
        logging.info("🎉 Done. Uploaded to s3://%s/%s/", cfg.s3_bucket, cfg.s3_prefix_out)

# ----------------------------- CLI -----------------------------

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="GNN Hotspot Prediction (Weekly) — GAT w/ fallback MLP")
    p.add_argument("--s3-input",      required=True, help="S3 (or local) path to Engineereddata Parquet.")
    p.add_argument("--s3-bucket",     required=True, help="Target S3 bucket for uploads.")
    p.add_argument("--s3-prefix-out", required=True, help="S3 prefix for uploads (no trailing slash).")
    p.add_argument("--outdir",        default="/tmp/_out_gnn", help="Local output directory.")
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--lr",            type=float, default=0.002)
    p.add_argument("--weight-decay",  type=float, default=4e-4)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--dist-threshold",type=float, default=0.06, help="Lat_Bin/Lng_Bin radius in bin units.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--test-size",     type=float, default=0.20)
    p.add_argument("--no-upload",     action="store_true", help="Skip S3 upload step.")
    args = p.parse_args()

    return RunConfig(
        s3_input=args.s3_input,
        s3_bucket=args.s3_bucket,
        s3_prefix_out=args.s3_prefix_out,
        outdir=Path(args.outdir),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        dropout=args.dropout,
        dist_threshold=args.dist_threshold,
        seed=args.seed,
        test_size=args.test_size,
        upload=not args.no_upload
    )

if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
