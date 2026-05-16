#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --output=/scratch200/tallilo/deep_learning_project/eval_output_%j.log
#SBATCH --error=/scratch200/tallilo/deep_learning_project/eval_error_%j.log
#SBATCH --time=03:00:00


# --- 🔑 ACCOUNT & PRIVILEGES ---
#SBATCH --account=tamirtul-users_v2
#SBATCH --qos=public

# --- 🚀 GPU & PARTITION ---
#SBATCH --partition=gpu-general-pool

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ==========================================
# 🛠️ SETUP ENVIRONMENT
# ==========================================
echo "🚀 Job started on node: $(hostname)"
echo "🎮 GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /powerapps/share/centos7/python-anaconda3-2022.10/bin/activate /scratch200/tallilo/conda_envs/dna_env

# Define the home directory of the project ./
export MY_PROJECT_DIR="/scratch200/tallilo/deep_learning_project"

export TRITON_CACHE_DIR="/scratch200/tallilo/deep_learning_project/.triton_cache"
mkdir -p $TRITON_CACHE_DIR
# ==========================================
# 🏃 RUN INFERENCE SCRIPT
# ==========================================


export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

echo "🔥 Starting evaluation on TEST set..."

python -u "$MY_PROJECT_DIR/inference_without_prot_mamba_weights.py"

echo "✅ Evaluation Finished!"
