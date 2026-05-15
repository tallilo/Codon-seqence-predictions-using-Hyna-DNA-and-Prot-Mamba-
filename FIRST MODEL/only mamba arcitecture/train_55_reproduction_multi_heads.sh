#!/bin/bash

# ==========================================
# 📋 SLURM SETTINGS
# ==========================================
# Set the name of the job for easy identification in the SLURM queue
#SBATCH --job-name=Fusion_Exp0

# Define paths for standard output (stdout) and standard error (stderr)
# The '%j' wildcard is dynamically replaced by the actual SLURM Job ID
#SBATCH --output=run_output_%j.log
#SBATCH --error=run_error_%j.log

# Request 1 task (standard for single-node training)
#SBATCH --ntasks=1

# Allocate 4 CPU cores for this task (useful for data loading and preprocessing)
#SBATCH --cpus-per-task=4

# Request 32 GB of RAM (sufficient to hold the parsed dataset in memory)
#SBATCH --mem=32G

# --- 🔑 ACCOUNT & PRIVILEGES ---
# Billing account/group for cluster resource allocation
#SBATCH --account=tamirtul-users_v2

# Quality of Service (QoS) tier
#SBATCH --qos=public

# --- 🚀 GPU & PARTITION ---
# Submit the job to the general GPU queue
#SBATCH --partition=gpu-general-pool

# Request exactly 1 GPU for training
#SBATCH --gres=gpu:1

# Explicitly exclude these specific compute nodes from running the job
# (useful if you know certain nodes are faulty or slow)
#SBATCH --exclude=compute-0-300,compute-0-282,compute-0-58

# Set a strict 12-hour time limit. The job will automatically TIMEOUT if it exceeds this.
#SBATCH --time=12:00:00

# ==========================================
# 🛠️ ENVIRONMENT SETUP
# ==========================================
# Define the base directory for all temporary and cache files on the scratch drive
# This prevents filling up the limited home directory quota during training
export MY_CACHE_DIR="/scratch200/tallilo/my_cluster_caches"

# Create the necessary cache subdirectories if they do not already exist
mkdir -p $MY_CACHE_DIR/huggingface
mkdir -p $MY_CACHE_DIR/wandb_config
mkdir -p $MY_CACHE_DIR/wandb_cache
mkdir -p $MY_CACHE_DIR/wandb_dir
mkdir -p $MY_CACHE_DIR/triton
mkdir -p $MY_CACHE_DIR/tmp

# Redirect Hugging Face, Triton (GPU compiler), and OS temporary files to the scratch drive
export HF_HOME="$MY_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$MY_CACHE_DIR/triton"
export TMPDIR="$MY_CACHE_DIR/tmp"

#  WANDB CONFIGURATION 
export WANDB_CONFIG_DIR="$MY_CACHE_DIR/wandb_config"
export WANDB_CACHE_DIR="$MY_CACHE_DIR/wandb_cache"

# 🔥 Here WandB will save the run directories and local logs before syncing
export WANDB_DIR="$MY_CACHE_DIR/wandb_dir"  

# Set the WandB API key for automatic authentication (NOTE: Keep this key private)
export WANDB_API_KEY="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"

# Activate the dedicated Conda environment containing PyTorch and required dependencies
source /powerapps/share/centos7/python-anaconda3-2022.10/bin/activate /scratch200/tallilo/conda_envs/dna_env

# Force Python to print logs immediately to the output file instead of buffering them
export PYTHONUNBUFFERED=1

# Print diagnostic information to the output log to confirm where the job landed
echo "🚀 Job started on node: $SLURMD_NODENAME"
echo "🎮 GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ==========================================
# 🏃 EXECUTION
# ==========================================
echo "🔥 Running Experiment 0 with configs/config_0_R.json..."

# Execute the main PyTorch training script (without pre-trained ProtMamba weights)
# using the specified multi-head configuration file
python -u training_without_prot_mamba_wights.py --config configs/config_0_R_multi_heads.json

echo "✅ Job Finished!"