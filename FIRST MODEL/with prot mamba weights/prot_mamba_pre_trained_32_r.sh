#!/bin/bash

# ==========================================
# 📋 SLURM SETTINGS
# ==========================================
# Set the name of the job (useful for identifying it in the SLURM queue)
#SBATCH --job-name=Fusion_Exp0

# Define where standard output (stdout) and standard error (stderr) will be saved. 
# The '%j' wildcard is dynamically replaced by the actual SLURM Job ID.
#SBATCH --output=run_output_%j.log
#SBATCH --error=run_error_%j.log

# Request 1 task (standard for single-node training)
#SBATCH --ntasks=1

# Allocate 4 CPU cores for this task (essential for fast data loading/preprocessing)
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

# Request exactly 1 GPU for training (e.g., NVIDIA A100)
#SBATCH --gres=gpu:1

# Restrict the job to run ONLY on these specific compute nodes
#SBATCH --nodelist=compute-0-103,compute-0-62,compute-0-390,compute-0-10,compute-0-282,compute-0-283,compute-0-284,compute-0-196

# Set a strict 24-hour time limit. The job will automatically TIMEOUT if it exceeds this limit.
#SBATCH --time=24:00:00

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

# Redirect Weights & Biases (WandB) cache and configuration files
export WANDB_CONFIG_DIR="$MY_CACHE_DIR/wandb_config"
export WANDB_CACHE_DIR="$MY_CACHE_DIR/wandb_cache"

# 🔥 This is the local directory where WandB will save the run logs and metrics before syncing
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

# Execute the main PyTorch training script for the Multimodal model
# using the specified LoRA configuration file
python -u training_with_prot_mamba_weights.py --config configs/prot_mamba_pre_trained_32_r.json

echo "✅ Job Finished!"