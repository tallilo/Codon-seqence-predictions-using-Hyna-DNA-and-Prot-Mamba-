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

# אנחנו מבקשים GPU אחד מכל סוג שתומך ב-Flash Attention
# הדרך הכי טובה היא לבקש גנרית ולחסום את ה-V100
#SBATCH --gres=gpu:1

# 🔥 חוסם את המכונות הישנות (V100) או כאלו שלא תואמות
# המכונה compute-0-300 היא V100 - אנחנו לא רוצים אותה.
# בקשת GPU ספציפי (H100 הוא האידיאלי ל-Batch גדול)


# או אם אתה רוצה A100 (גם מצוין)

# במידה ואתה רוצה להחריג את כל ה-Nodes החלשים/ישנים:
#SBATCH --exclude=compute-0-300,compute-0-100,compute-0-58,compute-0-53

#SBATCH --time=12:00:00

# ==========================================
# 🛠️ ENVIRONMENT SETUP
# ==========================================
# טעינת אנקונדה בצורה חסינה לתוך ה-Job
export MY_CACHE_DIR="/scratch200/tallilo/my_cluster_caches"

# יצירת התיקיות מראש כדי למנוע שגיאות
mkdir -p $MY_CACHE_DIR/huggingface
mkdir -p $MY_CACHE_DIR/wandb_config
mkdir -p $MY_CACHE_DIR/wandb_cache
mkdir -p $MY_CACHE_DIR/wandb_dir
mkdir -p $MY_CACHE_DIR/triton
mkdir -p $MY_CACHE_DIR/tmp

# הפניית משתני הסביבה של כל הספריות ל-Scratch
export HF_HOME="$MY_CACHE_DIR/huggingface"
export WANDB_CONFIG_DIR="$MY_CACHE_DIR/wandb_config"
export WANDB_CACHE_DIR="$MY_CACHE_DIR/wandb_cache"
export WANDB_DATA_DIR="$MY_CACHE_DIR/wandb_dir"
export TRITON_CACHE_DIR="$MY_CACHE_DIR/triton"
export TMPDIR="$MY_CACHE_DIR/tmp"
# 2. פתרון הקריסה של WandB (החלף את ה-XXX במפתח האמיתי שלך!)
export WANDB_API_KEY="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"

# 3. פתרון שגיאת ה-Conda - הפעלה ישירה עם Source

#conda activate /scratch200/tallilo/conda_envs/dna_env
source /powerapps/share/centos7/python-anaconda3-2022.10/bin/activate /scratch200/tallilo/conda_envs/dna_env
# מבטיח שהפלט יודפס בזמן אמת ללוג
export PYTHONUNBUFFERED=1

echo "🚀 Job started on node: $SLURMD_NODENAME"
echo "🎮 GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ==========================================
# 🏃 EXECUTION
# ==========================================
echo "🔥 Running Experiment 0 with configs/config_0_0.0001_lr_long_warmup_512_depth.json..."

# הרצת המודל עם הקונפיגורציה הספציפית

#python -u    zero_masking_Heirecy_pooler.py --config configs/config_0_Hirercial_pooling_Complex_0_Masking.json
python -u  Testing_gradients_decoder_lead_run_away_platue_stage_6_trans_loss_1.0_less_noise_restart_8_lr_DNA_GATE.py   --config configs/Testing_gradients_decoder_lead_run_away_platue_stage_6_trans_loss_1.0_less_noise_restart_8_lr_DNA_GATE.json


#python -u  Complex_Herircial_pooler.py  --config configs/config_0_ALL_Open_AdamW_Warmup_good_checkpoint_up_to_0.83.json

#python -u   overfit_test_classic.py  --config configs/config_0_ALL_Open_AdamW_Warmup_good_checkpoint_up_to_0.83.json
#python -u   overfit_test_bad_resampler.py  --config configs/config_0_ALL_Open_AdamW_Warmup_good_checkpoint_up_to_0.83.json
echo "✅ Job Finished!"
