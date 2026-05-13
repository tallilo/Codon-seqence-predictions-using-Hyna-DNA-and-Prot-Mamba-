#!/bin/bash

# ==========================================
# 📋 SLURM SETTINGS
# ==========================================
#SBATCH --job-name=Fusion_Exp0
#SBATCH --output=run_output_%j.log
#SBATCH --error=run_error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# --- 🔑 ACCOUNT & PRIVILEGES ---
#SBATCH --account=tamirtul-users_v2
#SBATCH --qos=public

# --- 🚀 GPU & PARTITION ---
#SBATCH --partition=gpu-general-pool

#SBATCH --gres=gpu:1

#SBATCH --nodelist=compute-0-103,compute-0-62,compute-0-390,compute-0-10,compute-0-282,compute-0-283,compute-0-284,compute-0-196

#SBATCH --time=12:00:00

# ==========================================
# 🛠️ ENVIRONMENT SETUP
# ==========================================

export MY_CACHE_DIR="/scratch200/tallilo/my_cluster_caches"


mkdir -p $MY_CACHE_DIR/huggingface
mkdir -p $MY_CACHE_DIR/wandb_config
mkdir -p $MY_CACHE_DIR/wandb_cache
mkdir -p $MY_CACHE_DIR/wandb_dir
mkdir -p $MY_CACHE_DIR/triton
mkdir -p $MY_CACHE_DIR/tmp


export HF_HOME="$MY_CACHE_DIR/huggingface"
export WANDB_CONFIG_DIR="$MY_CACHE_DIR/wandb_config"
export WANDB_CACHE_DIR="$MY_CACHE_DIR/wandb_cache"
export WANDB_DATA_DIR="$MY_CACHE_DIR/wandb_dir"
export TRITON_CACHE_DIR="$MY_CACHE_DIR/triton"
export TMPDIR="$MY_CACHE_DIR/tmp"

export WANDB_API_KEY="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"


source /powerapps/share/centos7/python-anaconda3-2022.10/bin/activate /scratch200/tallilo/conda_envs/dna_env

export PYTHONUNBUFFERED=1

echo "🚀 Job started on node: $SLURMD_NODENAME"
echo "🎮 GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ==========================================
# 🏃 EXECUTION
# ==========================================
echo "🔥 Running Experiment 0 with configs/config_0_0.0001_lr_long_warmup_512_depth.json..."


python -u   train_55_reproduction.py  --config configs/config_0_R.json



echo "✅ Job Finished!"
