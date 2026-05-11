#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --output=/scratch200/tallilo/deep_learning_project/eval_output_%j.log
#SBATCH --error=/scratch200/tallilo/deep_learning_project/eval_error_%j.log
#SBATCH --time=03:00:00

# 🔥 עדכון המחיצה וה-QOS 🔥
# אם ה-QOS הוא public, לרוב המחיצה תהיה tuller או public.
# נסה tuller אם אתה משתמש במשאבי המעבדה, או public למשאבים כלליים.


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

# הפניית הקבצים הזמניים של Triton ל-scratch
export TRITON_CACHE_DIR="/scratch200/tallilo/deep_learning_project/.triton_cache"
mkdir -p $TRITON_CACHE_DIR
# ==========================================
# 🏃 RUN INFERENCE SCRIPT
# ==========================================
PYTHON_SCRIPT="/scratch200/tallilo/deep_learning_project/inference_eval.py"

echo "🔥 Starting evaluation on TEST set..."
python $PYTHON_SCRIPT

echo "✅ Evaluation Finished!"
