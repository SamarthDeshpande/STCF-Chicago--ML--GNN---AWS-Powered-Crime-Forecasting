#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chicago Feature Engineering Pipeline
- Reads Parquet dataset from S3 (pyarrow + s3fs)
- Builds daily rolling/lag features and crime-type histograms
- Runs spatial clustering (DBSCAN) and hour-pattern clustering (KMeans)
- Optionally computes Node2Vec beat embeddings from a proximity graph
- Writes engineered dataset back to S3 as a single Parquet file

Usage (defaults mirror your original paths):
    python feature_engineering_pipeline.py \
        --s3-input s3://dataset-chicago/project-folder/Processed/Final/ \
        --s3-output s3://dataset-chicago/project-folder/Processed/Engineereddata/part-00000.parquet
"""

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances

import networkx as nx
import gc

# Optional: Node2Vec (skip if not installed)
try:
    from node2vec import Node2Vec
    _HAS_NODE2VEC = True
except Exception:
    _HAS_NODE2VEC = False


# ----------------------------- Config -----------------------------

@dataclass
class RunConfig:
    s3_input: str = "s3://dataset-chicago/project-folder/Processed/Final/"
    s3_output: str = "s3://dataset-chicago/project-folder/Processed/Engineereddata/part-00000.parquet"
    sample_n: int = 30000                 # for DBSCAN sampling
    dbscan_eps: float = 0.15
    dbscan_min_samples: int = 20
    hour_clusters: int = 5
    n2v_edge_km: float = 2.0
    n2v_dimensions: int = 8
    n2v_walk_length: int = 5
    n2v_num_walks: int = 50
    n2v_window: int = 3
    random_state: int = 42
    year_min: int = 2010
    # I/O
    engine: str = "pyarrow"
    storage_options: dict | None = None


# ----------------------------- Logging -----------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )


# ----------------------------- Helpers -----------------------------

def ensure_columns(df: pd.DataFrame, needed: list[str]):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def read_parquet_any(path: str, engine: str, storage_options: dict | None) -> pd.DataFrame:
    logging.info("Reading Parquet from: %s", path)
    df = pd.read_parquet(path, engine=engine, storage_options=storage_options or {"anon": False})
    logging.info("Loaded DataFrame shape: %s", df.shape)
    return df


def write_parquet_any(df: pd.DataFrame, path: str, engine: str, storage_options: dict | None):
    logging.info("Writing engineered Parquet to: %s", path)
    df.to_parquet(path, engine=engine, index=False, storage_options=storage_options or {"anon": False})
    logging.info("Write completed: %s rows, %s columns", df.shape[0], df.shape[1])


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    # Date conversions and sort
    df = df.copy()
    df["Date_TS"] = pd.to_datetime(df["Date_TS"], errors="coerce")
    df = df[df["Date_TS"].notna()]
    df = df[df["Date_TS"].dt.year >= 2010]
    df["Date_Only"] = pd.to_datetime(df["Date_TS"].dt.date)
    df.sort_values(by=["Beat", "Date_Only"], inplace=True)

    # Daily counts per Beat
    daily = df.groupby(["Beat", "Date_Only"]).size().reset_index(name="CrimeCountPerDay")
    threshold = daily["CrimeCountPerDay"].quantile(0.995)
    daily["Is_CrimeOutlier"] = daily["CrimeCountPerDay"] > threshold
    daily["CrimeCountPerDay_Capped"] = np.clip(daily["CrimeCountPerDay"], None, threshold)

    # Lags and rolling average
    for lag in [1, 7]:
        daily[f"Crime_Lag_{lag}Day_Beat"] = daily.groupby("Beat")["CrimeCountPerDay"].shift(lag)
    daily["Rolling_Avg_3Day_Beat"] = (
        daily.groupby("Beat")["CrimeCountPerDay"].transform(lambda x: x.shift(1).rolling(3).mean())
    )

    # Merge back
    df = df.merge(daily, on=["Beat", "Date_Only"], how="left")

    # CrimeType histogram per Beat (row-normalized)
    if "CrimeType" in df.columns:
        hist = df.groupby(["Beat", "CrimeType"]).size().unstack(fill_value=0)
        hist = hist.div(hist.sum(axis=1), axis=0).reset_index()
        df = df.merge(hist.drop_duplicates("Beat"), on="Beat", how="left")
    else:
        logging.warning("CrimeType column not found; skipping histogram features.")

    del daily
    gc.collect()
    return df


def spatial_dbscan(df: pd.DataFrame, sample_n: int, eps: float, min_samples: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    if not {"Latitude", "Longitude"}.issubset(df.columns):
        logging.warning("Latitude/Longitude not found; skipping DBSCAN spatial clustering.")
        df["Spatial_Cluster"] = -1
        return df

    coords_df = df[["Latitude", "Longitude"]].dropna()
    if coords_df.empty:
        logging.warning("No valid coordinates found; skipping DBSCAN.")
        df["Spatial_Cluster"] = -1
        return df

    n = min(sample_n, len(coords_df))
    sample_coords = coords_df.sample(n=n, random_state=seed)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(sample_coords)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)
    df["Spatial_Cluster"] = -1
    df.loc[sample_coords.index, "Spatial_Cluster"] = db.labels_.astype(int)
    return df


def hour_pattern_kmeans(df: pd.DataFrame, n_clusters: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    # Ensure Hour exists
    if "Hour" not in df.columns:
        if "Date_TS" in df.columns:
            df["Hour"] = pd.to_datetime(df["Date_TS"], errors="coerce").dt.hour
        else:
            logging.warning("Neither Hour nor Date_TS present; skipping hour-pattern clustering.")
            df["Hour_Cluster"] = -1
            return df

    beat_hour = df.groupby(["Beat", "Hour"]).size().unstack(fill_value=0)
    if beat_hour.empty:
        logging.warning("Beat x Hour table is empty; skipping hour-pattern clustering.")
        df["Hour_Cluster"] = -1
        return df

    beat_hour_dist = beat_hour.div(beat_hour.sum(axis=1), axis=0).fillna(0.0)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    beat_hour_dist["Hour_Cluster"] = km.fit_predict(beat_hour_dist)

    df = df.merge(beat_hour_dist[["Hour_Cluster"]].reset_index(), on="Beat", how="left")
    return df


def compute_node2vec_embeddings(
    df: pd.DataFrame,
    edge_km: float,
    dimensions: int,
    walk_length: int,
    num_walks: int,
    window: int,
    seed: int
) -> pd.DataFrame:
    df = df.copy()
    if not {"Latitude", "Longitude"}.issubset(df.columns):
        logging.warning("Latitude/Longitude not found; skipping Node2Vec embeddings.")
        return df
    if not _HAS_NODE2VEC:
        logging.warning("node2vec package not installed; skipping Node2Vec embeddings.")
        return df

    # Beat coordinates (mean per Beat)
    beat_coords = df.groupby("Beat")[["Latitude", "Longitude"]].mean().dropna().reset_index()
    if beat_coords.empty:
        logging.warning("No beat coordinates available; skipping Node2Vec.")
        return df

    beat_coords["Beat"] = beat_coords["Beat"].astype(str)
    coords_rad = np.radians(beat_coords[["Latitude", "Longitude"]].values)
    dist_km = haversine_distances(coords_rad) * 6371.0

    # Build edges for beats within threshold distance
    adj = (dist_km <= float(edge_km)).astype(int)
    np.fill_diagonal(adj, 0)

    edges = []
    idxs = beat_coords.index.to_numpy()
    for i in idxs:
        js = np.where(adj[i] == 1)[0]
        for j in js:
            if i != j:
                edges.append((beat_coords.iloc[i]["Beat"], beat_coords.iloc[j]["Beat"]))

    if not edges:
        logging.warning("No proximity edges found (edge_km=%.2f); skipping Node2Vec.", edge_km)
        return df

    # Build graph and log basic stats
    G = nx.from_edgelist(edges)
    iso = list(nx.isolates(G))
    logging.info("Graph: nodes=%d, edges=%d, isolates=%d, density=%.6f",
                 G.number_of_nodes(), G.number_of_edges(), len(iso), nx.density(G))

    # Train Node2Vec
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, workers=1, seed=seed)
    model = node2vec.fit(window=window, min_count=1)

    # Build embedding DataFrame
    embed_df = pd.DataFrame.from_dict(
        {n: model.wv[n] for n in model.wv.index_to_key}, orient="index"
    ).reset_index()
    embed_df.columns = ["Beat"] + [f"Embed_{i}" for i in range(dimensions)]
    embed_df["Beat"] = embed_df["Beat"].astype(str)

    # Merge into main df (remove prior Embed_* to avoid duplicates)
    df["Beat"] = df["Beat"].astype(str)
    embed_cols_existing = [c for c in df.columns if c.startswith("Embed_")]
    if embed_cols_existing:
        df = df.drop(columns=embed_cols_existing, errors="ignore")

    df = df.merge(embed_df.drop_duplicates("Beat"), on="Beat", how="left")
    # Drop rows without embeddings if needed (they are isolates or missing beats)
    embed_cols = [f"Embed_{i}" for i in range(dimensions)]
    df = df.dropna(subset=embed_cols)
    return df


def finalize_and_write(df: pd.DataFrame, output_uri: str, engine: str, storage_options: dict | None):
    # Label from CrimeType if present
    if "CrimeType" in df.columns:
        df["label"] = df["CrimeType"].astype("category").cat.codes
    else:
        logging.warning("CrimeType column missing; 'label' will not be created.")

    df = df.drop_duplicates()
    write_parquet_any(df, output_uri, engine=engine, storage_options=storage_options)


# ----------------------------- Main -----------------------------

def run(cfg: RunConfig):
    warnings.filterwarnings("ignore")
    setup_logging()
    np.random.seed(cfg.random_state)

    # Storage options default: assume IAM role attached to instance
    storage_opts = cfg.storage_options or {"anon": False}

    # Read
    df = read_parquet_any(cfg.s3_input, engine=cfg.engine, storage_options=storage_opts)

    # Required columns for core features
    ensure_columns(df, ["Beat", "Date_TS"])
    if "CrimeType" not in df.columns:
        logging.warning("CrimeType not present; histogram features and label will be skipped.")

    # Build daily features and histograms
    df = build_daily_features(df)
    logging.info("Daily features merged: shape=%s", df.shape)

    # Spatial clustering (DBSCAN) on a coordinate sample
    df = spatial_dbscan(df, cfg.sample_n, cfg.dbscan_eps, cfg.dbscan_min_samples, cfg.random_state)
    logging.info("Spatial clustering completed: shape=%s", df.shape)

    # Hour-pattern clustering (KMeans)
    df = hour_pattern_kmeans(df, cfg.hour_clusters, cfg.random_state)
    logging.info("Hour-pattern clustering completed: shape=%s", df.shape)

    # Node2Vec embeddings (optional)
    df = compute_node2vec_embeddings(
        df=df,
        edge_km=cfg.n2v_edge_km,
        dimensions=cfg.n2v_dimensions,
        walk_length=cfg.n2v_walk_length,
        num_walks=cfg.n2v_num_walks,
        window=cfg.n2v_window,
        seed=cfg.random_state
    )
    logging.info("Node2Vec embedding step completed: shape=%s", df.shape)

    # Write to S3
    finalize_and_write(df, cfg.s3_output, engine=cfg.engine, storage_options=storage_opts)
    logging.info("Pipeline finished successfully.")


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Chicago Feature Engineering Pipeline (S3 -> S3)")
    p.add_argument("--s3-input", default="s3://dataset-chicago/project-folder/Processed/Final/",
                   help="S3 (or local) path to source Parquet dataset (directory or file).")
    p.add_argument("--s3-output", default="s3://dataset-chicago/project-folder/Processed/Engineereddata/part-00000.parquet",
                   help="S3 path to write engineered Parquet file.")
    p.add_argument("--sample-n", type=int, default=30000, help="DBSCAN coordinate sample size.")
    p.add_argument("--dbscan-eps", type=float, default=0.15, help="DBSCAN eps (on standardized coords).")
    p.add_argument("--dbscan-min-samples", type=int, default=20, help="DBSCAN min_samples.")
    p.add_argument("--hour-clusters", type=int, default=5, help="KMeans clusters for hour patterns.")
    p.add_argument("--n2v-edge-km", type=float, default=2.0, help="Proximity graph radius in kilometers.")
    p.add_argument("--n2v-dimensions", type=int, default=8, help="Node2Vec embedding dimensions.")
    p.add_argument("--n2v-walk-length", type=int, default=5, help="Node2Vec walk length.")
    p.add_argument("--n2v-num-walks", type=int, default=50, help="Node2Vec number of walks.")
    p.add_argument("--n2v-window", type=int, default=3, help="Word2Vec window size for Node2Vec.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--engine", default="pyarrow", help="Parquet engine (default: pyarrow).")
    p.add_argument("--no-iam", action="store_true",
                   help="If set, do not pass storage_options={'anon': False}.")
    args = p.parse_args()

    return RunConfig(
        s3_input=args.s3_input,
        s3_output=args.s3_output,
        sample_n=args.sample_n,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        hour_clusters=args.hour_clusters,
        n2v_edge_km=args.n2v_edge_km,
        n2v_dimensions=args.n2v_dimensions,
        n2v_walk_length=args.n2v_walk_length,
        n2v_num_walks=args.n2v_num_walks,
        n2v_window=args.n2v_window,
        random_state=args.random_state,
        engine=args.engine,
        storage_options=(None if args.no_iam else {"anon": False})
    )


if __name__ == "__main__":
    run(parse_args())
