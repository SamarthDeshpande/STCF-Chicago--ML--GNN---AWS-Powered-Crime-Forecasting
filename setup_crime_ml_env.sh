#!/usr/bin/env bash
# setup_crime_ml_env.sh
# Creates a SageMaker-friendly conda env for all 4 pipelines:
#   1) Feature Engineering
#   2) Count Forecasts
#   3) GNN Hotspots
#   4) XGB Multiclass
# and writes a unified, pinned requirements.txt.
#
# Usage:
#   chmod +x setup_crime_ml_env.sh
#   ./setup_crime_ml_env.sh                # CPU by default, installs PyG (graph) stack too
#   INSTALL_PYG=no ./setup_crime_ml_env.sh # skip PyG (GNN code will use MLP fallback)
#   ENV_NAME=myenv ./setup_crime_ml_env.sh # custom env name
#
# Notes:
# - Pins are aligned for Python 3.10 + Torch 2.2.2.
# - For GPU nodes, this script auto-detects CUDA (nvidia-smi) and installs torch-cuda 12.1.
# - PyG wheels are installed from the official index matching Torch 2.2.* (CPU or cu121).

set -euo pipefail

ENV_NAME="${ENV_NAME:-crime-ml}"
INSTALL_PYG="${INSTALL_PYG:-yes}"

echo "==> Checking conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Use a SageMaker image with conda or install Miniconda."
  exit 1
fi

# Activate conda
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Detect GPU (CUDA) presence
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected (nvidia-smi found) -> using CUDA builds."
  HAS_GPU=1
else
  echo "==> No GPU detected -> using CPU builds."
  HAS_GPU=0
fi

echo "==> Creating/Updating conda env: ${ENV_NAME}"
# Create minimal env first to avoid solver thrash
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Env ${ENV_NAME} already exists; will install/ensure packages."
else
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

# Prefer mamba if present for speed
if command -v mamba >/dev/null 2>&1; then
  CONDA_CMD="mamba"
else
  CONDA_CMD="conda"
fi

echo "==> Installing core scientific + IO stack (conda-forge preferred)"
${CONDA_CMD} install -y -c conda-forge \
  numpy=1.26.4 \
  pandas=2.2.2 \
  scipy=1.12.0 \
  pyarrow=15.0.2 \
  s3fs=2024.5.0 \
  boto3=1.34.122 \
  joblib=1.4.2 \
  matplotlib=3.8.4 \
  seaborn=0.13.2 \
  scikit-learn=1.4.2 \
  statsmodels=0.14.3 \
  xgboost=2.0.3 \
  networkx=3.2.1 \
  gensim=4.3.2

echo "==> Installing pip-only deps"
# holidays via pip (small lib), node2vec via pip (depends on gensim/networkx already present)
pip install --no-deps --no-cache-dir holidays==0.56
pip install --no-cache-dir node2vec==0.4.6

echo "==> Installing PyTorch 2.2.2..."
if [[ "${HAS_GPU}" -eq 1 ]]; then
  # GPU build with CUDA 12.1
  ${CONDA_CMD} install -y -c pytorch -c nvidia pytorch=2.2.2 pytorch-cuda=12.1
else
  # CPU build
  ${CONDA_CMD} install -y -c pytorch pytorch=2.2.2 cpuonly
fi

if [[ "${INSTALL_PYG}" == "yes" ]]; then
  echo "==> Installing PyTorch Geometric (PyG) stack..."
  if [[ "${HAS_GPU}" -eq 1 ]]; then
    # CUDA 12.1 wheels for torch 2.2.* (cu121)
    pip install --no-cache-dir \
      torch-geometric==2.5.3 \
      torch-scatter==2.1.2 \
      torch-sparse==0.6.18 \
      torch-cluster==1.6.3 \
      torch-spline-conv==1.2.2 \
      -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
  else
    # CPU wheels for torch 2.2.*
    pip install --no-cache-dir \
      torch-geometric==2.5.3 \
      torch-scatter==2.1.2 \
      torch-sparse==0.6.18 \
      torch-cluster==1.6.3 \
      torch-spline-conv==1.2.2 \
      -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
  fi
else
  echo "==> Skipping PyG per INSTALL_PYG=no (GNN will use MLP fallback)."
fi

echo "==> Writing unified, pinned requirements.txt"
cat > requirements.txt <<'REQS'
# Combined requirements for:
# 1) Feature Engineering (DBSCAN/KMeans, Node2Vec, NetworkX)
# 2) Count Forecasting (Poisson/GBM/XGB + Quantiles)
# 3) GNN Hotspot Prediction (GAT via torch-geometric; MLP fallback if PyG absent)
# 4) XGBoost Multiclass (CrimeType)

# --- Core numeric / IO ---
numpy==1.26.4
pandas==2.2.2
scipy==1.12.0
pyarrow==15.0.2
s3fs==2024.5.0
boto3==1.34.122
joblib==1.4.2

# --- Plotting ---
matplotlib==3.8.4
seaborn==0.13.2

# --- ML ---
scikit-learn==1.4.2
xgboost==2.0.3
statsmodels==0.14.3
holidays==0.56

# --- Graph + Embedding (Feature Engineering) ---
networkx==3.2.1
gensim==4.3.2
node2vec==0.4.6

# --- PyTorch base (CPU by default). For GPU, swap the extra-index to cu121 wheels. ---
# CPU:
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.2

# GPU alternative (comment CPU extra-index above and uncomment below if using pip on GPU boxes):
# --extra-index-url https://download.pytorch.org/whl/cu121
# torch==2.2.2

# --- Optional: PyTorch Geometric stack (only if you need GAT; otherwise comment these out) ---
# CPU wheels for Torch 2.2.*:
-f https://data.pyg.org/whl/torch-2.2.0+cpu.html
torch-scatter==2.1.2
torch-sparse==0.6.18
torch-cluster==1.6.3
torch-spline-conv==1.2.2
torch-geometric==2.5.3

# GPU wheels for Torch 2.2.* + cu121 (use instead of the CPU block above):
# -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
# torch-scatter==2.1.2
# torch-sparse==0.6.18
# torch-cluster==1.6.3
# torch-spline-conv==1.2.2
# torch-geometric==2.5.3
REQS

echo "==> Quick smoke test (imports)..."
python - <<'PY'
import numpy, pandas, pyarrow, s3fs, boto3, joblib, matplotlib, seaborn
import sklearn, statsmodels.api as sm, xgboost, torch
import networkx, gensim
try:
    import node2vec
    print("node2vec OK")
except Exception as e:
    print("node2vec import failed:", e)
print("Core imports OK")
PY

echo "==> Environment ready."
echo "    conda activate ${ENV_NAME}"
echo "    (requirements.txt written to $(pwd)/requirements.txt)"
