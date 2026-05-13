#!/bin/bash

# ==========================================
# 📋 SLURM SETTINGS
# ==========================================
#SBATCH --job-name=Fusion_Exp0          # Name of the job
#SBATCH --output=run_output_%j.log     # Standard output log (%j inserts the JobID)
#SBATCH --error=run_error_%j.log       # Error log file
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Allocate 4 CPU cores for this task
#SBATCH --mem=32G                      # Total RAM requested (Note: consider increasing to 128G for large CSVs)

# --- 🔑 ACCOUNT & PRIVILEGES ---
#SBATCH --account=tamirtul-users_v2    # Research group account
#SBATCH --qos=public                   # Quality of Service level

# --- 🚀 GPU & PARTITION ---
#SBATCH --partition=gpu-general-pool   # Target partition for GPU jobs
#SBATCH --gres=gpu:1                   # Request 1 GPU

# Targeted list of compute nodes (Optimized for A100/H100 compatibility)
#SBATCH --nodelist=compute-0-103,compute-0-62,compute-0-390,compute-0-10,compute-0-282,compute-0-283,compute-0-284,compute-0-196

#SBATCH --time=12:00:00                # Maximum wall-clock time (HH:MM:SS)

# ==========================================
# 🛠️ ENVIRONMENT SETUP
# ==========================================

# Set local cache directory on Scratch storage to prevent filling up Home directory
export MY_CACHE_DIR="/scratch200/tallilo/my_cluster_caches"

# Ensure all necessary subdirectories exist before execution
mkdir -p $MY_CACHE_DIR/huggingface
mkdir -p $MY_CACHE_DIR/wandb_config
mkdir -p $MY_CACHE_DIR/wandb_cache
mkdir -p $MY_CACHE_DIR/wandb_dir
mkdir -p $MY_CACHE_DIR/triton
mkdir -p $MY_CACHE_DIR/tmp

# Redirect library cache paths to Scratch storage
export HF_HOME="$MY_CACHE_DIR/huggingface"
export WANDB_CONFIG_DIR="$MY_CACHE_DIR/wandb_config"
export WANDB_CACHE_DIR="$MY_CACHE_DIR/wandb_cache"
export WANDB_DATA_DIR="$MY_CACHE_DIR/wandb_dir"  # Local storage for Weights & Biases runs
export TRITON_CACHE_DIR="$MY_CACHE_DIR/triton"
export TMPDIR="$MY_CACHE_DIR/tmp"

# Put you WANDB  Key here
export WANDB_API_KEY=Wandb_key

# Activate the specific Anaconda environment for this project
source /powerapps/share/centos7/python-anaconda3-2022.10/bin/activate /scratch200/tallilo/conda_envs/dna_env

# Ensure Python output is flushed in real-time to the log file
export PYTHONUNBUFFERED=1

# Log hardware and execution details
echo "🚀 Job started on node: $SLURMD_NODENAME"
echo "🎮 GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ==========================================
# 🏃 EXECUTION
# ==========================================

# Execute the training script with the specified configuration
python -u   train_55_reproduction.py  --config configs/config_0_R.json

echo "✅ Job Finished!"