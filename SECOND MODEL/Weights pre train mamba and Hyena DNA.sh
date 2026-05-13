#!/bin/bash

# ==========================================
# 📋 SLURM SETTINGS
# ==========================================
# Descriptive name for the job in the queue
#SBATCH --job-name=Fusion_Exp0
# File paths for standard output and error logs (%j inserts the Job ID)
#SBATCH --output=run_output_%j.log
#SBATCH --error=run_error_%j.log
# Number of tasks and resources per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Total memory requested for the node
#SBATCH --mem=32G

# --- 🔑 ACCOUNT & PRIVILEGES ---
# Lab account associated with Professor Tamir Tuller
#SBATCH --account=tamirtul-users_v2
# Quality of Service (scheduling priority policy)
#SBATCH --qos=public

# --- 🚀 GPU & PARTITION ---
# Target partition for GPU jobs
#SBATCH --partition=gpu-general-pool

# Request 1 generic GPU. 
# We request a generic GPU but use exclusions below to avoid older architectures.
#SBATCH --gres=gpu:1

# --- 🚫 HARDWARE CONSTRAINTS ---
# Exclude specific nodes that use older Tesla V100 cards (e.g., compute-0-300).
# Modern architectures (A100/H100) are required for Flash Attention compatibility.
#SBATCH --exclude=compute-0-300,compute-0-100,compute-0-58,compute-0-53

# Maximum wall-clock time for the job (HH:MM:SS)
#SBATCH --time=12:00:00

# ==========================================
# 🛠️ ENVIRONMENT SETUP
# ==========================================

# Define a centralized cache directory on the scratch storage to avoid Home Directory quota limits
export MY_CACHE_DIR="/scratch200/tallilo/my_cluster_caches"

# Ensure all necessary subdirectories exist
mkdir -p $MY_CACHE_DIR/huggingface
mkdir -p $MY_CACHE_DIR/wandb_config
mkdir -p $MY_CACHE_DIR/wandb_cache
mkdir -p $MY_CACHE_DIR/wandb_dir
mkdir -p $MY_CACHE_DIR/triton
mkdir -p $MY_CACHE_DIR/tmp

# Redirect library cache paths to the high-capacity scratch directory
export HF_HOME="$MY_CACHE_DIR/huggingface"
export WANDB_CONFIG_DIR="$MY_CACHE_DIR/wandb_config"
export WANDB_CACHE_DIR="$MY_CACHE_DIR/wandb_cache"
export WANDB_DATA_DIR="$MY_CACHE_DIR/wandb_dir"
export TRITON_CACHE_DIR="$MY_CACHE_DIR/triton"
export TMPDIR="$MY_CACHE_DIR/tmp"

# Weights & Biases Authentication
export WANDB_API_KEY=wandb_Key

# Activate the specific Conda environment for bioinformatics and deep learning
source /powerapps/share/centos7/python-anaconda3-2022.10/bin/activate /scratch200/tallilo/conda_envs/dna_env

# Disable Python output buffering to ensure logs appear in real-time in the .log files
export PYTHONUNBUFFERED=1

# Log hardware and job details for troubleshooting
echo "🚀 Job started on node: $SLURMD_NODENAME"
echo "🎮 GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ==========================================
# 🏃 EXECUTION
# ==========================================

# Execute the training script using the provided JSON configuration
# The -u flag ensures unbuffered output for cleaner logging
python -u Testing_gradients_decoder_lead_run_away_platue_stage_6_trans_loss_1.0_less_noise_restart_8_lr_DNA_GATE.py \
    --config configs/Testing_gradients_decoder_lead_run_away_platue_stage_6_trans_loss_1.0_less_noise_restart_8_lr_DNA_GATE.json

echo "✅ Job Finished!"