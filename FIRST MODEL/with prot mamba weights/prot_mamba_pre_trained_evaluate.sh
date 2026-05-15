#!/bin/bash

# ==========================================
# 📋 SLURM SETTINGS
# ==========================================
# Set the name of the job for easy identification in the SLURM queue
#SBATCH --job-name=eval_model

# Define where standard output (stdout) and standard error (stderr) will be saved
# The '%j' wildcard is dynamically replaced by the actual SLURM Job ID
#SBATCH --output=/scratch200/tallilo/deep_learning_project/eval_output_%j.log
#SBATCH --error=/scratch200/tallilo/deep_learning_project/eval_error_%j.log

# Set a strict 3-hour time limit. The job will automatically TIMEOUT if it exceeds this limit.
#SBATCH --time=03:00:00

# Note on QOS and Partitions: 
# If QOS is set to 'public', the partition is usually 'tuller' or 'public'. 
# Try 'tuller' if you are using lab-specific resources, or 'public' for general cluster resources.

# --- 🔑 ACCOUNT & PRIVILEGES ---
# Billing account/group for cluster resource allocation
#SBATCH --account=tamirtul-users_v2

# Quality of Service (QoS) tier
#SBATCH --qos=public

# --- 🚀 GPU & PARTITION ---
# Submit the job to the general GPU queue
#SBATCH --partition=gpu-general-pool

# Request exactly 1 GPU for inference
#SBATCH --gres=gpu:1

# Allocate 4 CPU cores for data loading and preprocessing tasks
#SBATCH --cpus-per-task=4

# Request 32 GB of RAM (sufficient for holding the test dataset and model weights in memory)
#SBATCH --mem=32G

# ==========================================
# 🛠️ SETUP ENVIRONMENT
# ==========================================
# Print diagnostic information to the output log to confirm where the job landed
echo "🚀 Job started on node: $(hostname)"
echo "🎮 GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Activate the dedicated Conda environment containing PyTorch and required dependencies
source /powerapps/share/centos7/python-anaconda3-2022.10/bin/activate /scratch200/tallilo/conda_envs/dna_env

# Redirect Triton (GPU compiler) cache to the scratch drive 
# This prevents filling up the limited home directory quota during model compilation
export TRITON_CACHE_DIR="/scratch200/tallilo/deep_learning_project/.triton_cache"
mkdir -p $TRITON_CACHE_DIR

# ==========================================
# 🏃 RUN INFERENCE SCRIPT
# ==========================================


# Force Python to print logs immediately to the output file instead of buffering them
export PYTHONUNBUFFERED=1

# Run CUDA operations synchronously (forces CPU to wait for GPU). 
# This slows down the code slightly but provides exact tracebacks if a GPU error occurs.
export CUDA_LAUNCH_BLOCKING=1

echo "🔥 Starting evaluation on TEST set..."

# Execute the evaluation script (the '-u' flag also enforces unbuffered output)
# python -u evaluate_test_with_csv_of_pred.py  <-- (Commented out previous version)
python -u inference_with_prot_mamba_weights.py

echo "✅ Evaluation Finished!"