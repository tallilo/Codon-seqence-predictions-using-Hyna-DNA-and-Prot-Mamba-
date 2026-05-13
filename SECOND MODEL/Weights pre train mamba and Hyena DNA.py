import sys
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch._dynamo
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import math
import torch
import torch.distributed as dist
from collections import defaultdict
from torch import Tensor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import sys
import numpy as np
import json
import csv
import os
import shutil
import argparse
import time
import glob
from datetime import datetime
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup, PretrainedConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from transformers import PretrainedConfig
from transformers import PreTrainedModel





# ==========================================
# 🛠️ 0. CONFIGURATION & SETUP
# ==========================================
# Change to your relevent paths
BASE_TMP_PATH = "/scratch200/tallilo/deep_learning_project"
TRAIN_FILE = f"{BASE_TMP_PATH}/data/TRAIN_HOMOLOGY_SPLIT.csv"
VAL_FILE = f"{BASE_TMP_PATH}/data/TEST_HOMOLOGY_SPLIT.csv"
PATH_DNA_CHECKPOINT = f"{BASE_TMP_PATH}/models/hyena_dna/HYENA_DNA_weights.ckpt"
PATH_PROT_MODEL_DIR = f"{BASE_TMP_PATH}/models/protmamba"
HYENA_PATH = f"{BASE_TMP_PATH}/models/hyena_dna"


# ---  ORGANISM MAPPING & COUNTS (HARDCODED FOR STABILITY)  ---
ORG_LIST = [
    'Bacillus_subtilis', 'Cryptococcus_neoformans', 'Deinococcus_radiodurans',
    'Dictyostelium_discoideum', 'E_coli', 'Halobacterium_salinarum',
    'Helicobacter_pylori', 'Methanocaldococcus_jannaschii', 'Mycobacterium_tuberculosis',
    'Mycoplasma_pneumoniae', 'Neurospora_crassa', 'Pseudomonas_aeruginosa',
    'Saccharomyces_cerevisiae', 'Salmonella_typhimurium', 'Schizosaccharomyces_pombe',
    'Staphylococcus_aureus', 'Streptococcus_pneumoniae'
]
ORG_TO_ID = {name: i for i, name in enumerate(ORG_LIST)}



# Counts from The specific training set

TRAIN_COUNTS = {
    'Bacillus_subtilis': 3143,
    'Cryptococcus_neoformans': 1998,
    'Deinococcus_radiodurans': 1656,
    'Dictyostelium_discoideum': 5736,
    'E_coli': 3280,
    'Halobacterium_salinarum': 1327,
    'Helicobacter_pylori': 1243,
    'Methanocaldococcus_jannaschii': 621,
    'Mycobacterium_tuberculosis': 2729,
    'Mycoplasma_pneumoniae': 311,
    'Neurospora_crassa': 4497,
    'Pseudomonas_aeruginosa': 3532,
    'Saccharomyces_cerevisiae': 3136,
    'Salmonella_typhimurium': 2476,
    'Schizosaccharomyces_pombe': 3434,
    'Staphylococcus_aureus': 1671,
    'Streptococcus_pneumoniae': 1300
}


############ moun adamW  optimisers

# -----------------------------------------------------------------------------
# """
# Good old AdamW optimizer, fused kernel.
# https://arxiv.org/abs/1711.05101
# """

# #@torch.compile(dynamic=False, fullgraph=True)
# #@torch._dynamo.disable
# @torch.compile(dynamic=False, fullgraph=True)
# def adamw_step_fused(
#     p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor,
#     step_t: Tensor, lr_t: Tensor, beta1_t: Tensor, beta2_t: Tensor,
#     eps_t: Tensor, wd_t: Tensor
# ) -> None:
#     # 1. חילוץ ה-dtype וה-device של הפרמטר (כנראה bfloat16 ו-cuda:0)
#     dtype = p.dtype
#     device = p.device

#     # 2. העברת כל הנתונים ל-dtype ול-device הנכונים כדי למנוע את ה-RuntimeError
#     lr = lr_t.to(device=device, dtype=dtype)
#     wd = wd_t.to(device=device, dtype=dtype)
#     beta1 = beta1_t.to(device=device, dtype=dtype)
#     beta2 = beta2_t.to(device=device, dtype=dtype)
#     eps = eps_t.to(device=device, dtype=dtype)

#     # 3. ביצוע החישובים (כאן הכל כבר באותו dtype, אז זה יעבור חלק)
#     # Weight decay (Decoupled)
#     p.mul_(1 - lr * wd)

#     # עדכון מומנטום (First and Second Moments)
#     # הערה: (1 - beta1) עכשיו ייווצר כ-bfloat16
#     exp_avg.lerp_(grad, 1 - beta1)
#     exp_avg_sq.lerp_(grad.square(), 1 - beta2)

#     # תיקוני Bias (Bias Correction)
#     # משתמשים ב-.pow(step_t) כדי להישאר בתוך עולם הטאנסורים
#     bias_corr1 = 1 - beta1_t.pow(step_t).to(device=device, dtype=dtype)
#     bias_corr2 = 1 - beta2_t.pow(step_t).to(device=device, dtype=dtype)

#     # חישוב הצעד הסופי
#     # step_size = lr / bias_corr1
#     step_size = lr / bias_corr1

#     # denom = sqrt(exp_avg_sq / bias_corr2) + eps
#     denom = (exp_avg_sq / bias_corr2).sqrt().add_(eps)

#     # עדכון הפרמטר: p = p - (step_size * exp_avg / denom)
#     # addcdiv_ עושה בדיוק את זה בבת אחת ובצורה אופטימלית
#     p.addcdiv_(exp_avg, denom, value=-step_size)

# # -----------------------------------------------------------------------------
# """
# Muon optimizer adapted and simplified from modded-nanogpt.
# https://github.com/KellerJordan/modded-nanogpt

# Background:
# Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
# quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
# of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
# zero even beyond the point where the iteration no longer converges all the way to one everywhere
# on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
# where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
# performance at all relative to UV^T, where USV^T = G is the SVD.

# Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
# Polar Express Sign Method for orthogonalization.
# https://arxiv.org/pdf/2505.16932
# by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

# NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
# update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
# https://arxiv.org/pdf/2510.05491

# Some of the changes in nanochat implementation:
# - Uses a simpler, more general approach to parameter grouping and stacking
# - Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
# - Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
# """

# # Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# # From https://arxiv.org/pdf/2505.16932
# polar_express_coeffs = [
#     (8.156554524902461, -22.48329292557795, 15.878769915207462),
#     (4.042929935166739, -2.808917465908714, 0.5000178451051316),
#     (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
#     (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
#     (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
# ]


# # @torch.compile(dynamic=True, fullgraph=True)
# #@torch._dynamo.disable
# @torch.compile(dynamic=False, fullgraph=True)
# def muon_step_fused(
#     stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
#     lr_t, momentum_t, wd_t, beta2_t, ns_steps, red_dim
# ):
#     # 1. חילוץ המכשיר והסוג (Device & Dtype)
#     device = stacked_params.device
#     dtype = stacked_params.dtype

#     # 2. 🔥 העברת ההיפר-פרמטרים ל-GPU 🔥
#     lr = lr_t.to(device=device, dtype=dtype)
#     wd = wd_t.to(device=device, dtype=dtype)
#     beta2 = beta2_t.to(device=device, dtype=dtype)
#     momentum = momentum_t.to(device=device, dtype=dtype)

#     # 3. Nesterov Momentum
#     momentum_buffer.lerp_(stacked_grads, 1 - momentum)
#     g = stacked_grads.lerp_(momentum_buffer, momentum)

#     # 4. 🔥 Variance Reduction (Pre-conditioning) 🔥
#     # חייב לקרות *לפני* ה-Polar Express!
#     v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
#     red_dim_size = g.size(red_dim)

#     v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
#     v_norm = v_norm_sq.sqrt()

#     # עדכון הבאפר השני (מוודאים Dtype כדי למנוע קריסות)
#     second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)

#     step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
#     scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
#     v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()

#     final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
#     g = g * final_scale.to(g.dtype)

#     # 5. 🔥 Polar Express ב-Float32 ליציבות 🔥
#     X = g.to(torch.float32)

#     # נורמליזציה ראשונית למניעת פיצוץ
#     gnorm = X.norm(dim=(-2, -1), keepdim=True)
#     X = X / (gnorm * 1.02 + 1e-9)

#     # איטרציות Newton-Schulz / Polar Express
#     for a, b, c in polar_express_coeffs[:ns_steps]:
#         if X.size(-2) > X.size(-1): # Tall matrix
#             A = X.mT @ X
#             B = b * A + c * (A @ A)
#             X = a * X + X @ B
#         else: # Wide matrix
#             A = X @ X.mT
#             B = b * A + c * (A @ A)
#             X = a * X + B @ X

#     # 6. 🔥 Cautious Update + Weight Decay 🔥
#     # מחזירים ל-dtype המקורי (bfloat16)
#     update = X.to(dtype)

#     # Cautious Mask: מפעילים את ה-Weight Decay רק כשהכיוונים תואמים
#     mask = (update * stacked_params) >= 0

#     # העדכון היחיד והסופי: p = p - (lr * update + lr * wd * p * mask)
#     stacked_params.sub_(lr * update + lr * wd * stacked_params * mask)

# # -----------------------------------------------------------------------------
# # Single GPU version of the MuonAdamW optimizer.
# # Used mostly for reference, debugging and testing.

# class MuonAdamW(torch.optim.Optimizer):
#     """
#     Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

#     AdamW - Fused AdamW optimizer step.

#     Muon - MomentUm Orthogonalized by Newton-schulz
#     https://kellerjordan.github.io/posts/muon/

#     Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
#     processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
#     matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
#     the advantage that it can be stably run in bfloat16 on the GPU.

#     Some warnings:
#     - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
#     or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
#     - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

#     Arguments:
#         param_groups: List of dicts, each containing:
#             - 'params': List of parameters
#             - 'kind': 'adamw' or 'muon'
#             - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
#             - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
#     """
#     def __init__(self, param_groups: list[dict]):
#         super().__init__(param_groups, defaults={})
#         # 0-D CPU tensors to avoid torch.compile recompilation when values change
#         # AdamW tensors
#         self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         # Muon tensors
#         self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
#         self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

#     def _step_adamw(self, group: dict) -> None:
#         """
#         AdamW update for each param in the group individually.
#         Lazy init the state, fill in all 0-D tensors, call the fused kernel.
#         """
#         for p in group['params']:
#             if p.grad is None:
#                 continue
#             grad = p.grad
#             state = self.state[p]

#             # State init
#             if not state:
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p)
#                 state['exp_avg_sq'] = torch.zeros_like(p)
#             exp_avg = state['exp_avg']
#             exp_avg_sq = state['exp_avg_sq']
#             state['step'] += 1

#             # Fill 0-D tensors with current values
#             self._adamw_step_t.fill_(state['step'])
#             self._adamw_lr_t.fill_(group['lr'])
#             self._adamw_beta1_t.fill_(group['betas'][0])
#             self._adamw_beta2_t.fill_(group['betas'][1])
#             self._adamw_eps_t.fill_(group['eps'])
#             self._adamw_wd_t.fill_(group['weight_decay'])

#             # Fused update: weight_decay -> momentum -> bias_correction -> param_update
#             adamw_step_fused(
#                 p, grad, exp_avg, exp_avg_sq,
#                 self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
#                 self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
#             )

#     def _step_muon(self, group: dict) -> None:
#         params: list[Tensor] = [p for p in group['params'] if p.grad is not None]
#         if not params: return

#         # 1. מילוי ערכי ה-CPU
#         self._muon_momentum_t.fill_(group["momentum"])
#         self._muon_wd_t.fill_(group["weight_decay"])
#         self._muon_beta2_t.fill_(group.get("beta2", 0.0))

#         # 2. 🔥 קיבוץ פרמטרים לפי גודל (Shapes) 🔥
#         # ככה נהנה מה-Stacking על כל קבוצה של שכבות זהות
#         from collections import defaultdict
#         shape_groups = defaultdict(list)
#         for p in params:
#             shape_groups[p.shape].append(p)

#         for shape, p_list in shape_groups.items():
#             num_in_group = len(p_list)

#             # אתחול באפרים ספציפיים לגודל הזה בתוך ה-State של הפרמטר הראשון
#             p0 = p_list[0]
#             state = self.state[p0]

#             if "momentum_buffer" not in state:
#                 state["momentum_buffer"] = torch.zeros(num_in_group, *shape, dtype=p0.dtype, device=p0.device)
#                 # באפר מצומצם (Factored) ל-Second Momentum
#                 s_shape = (num_in_group, shape[-2], 1) if shape[-2] >= shape[-1] else (num_in_group, 1, shape[-1])
#                 state["second_momentum_buffer"] = torch.zeros(s_shape, dtype=p0.dtype, device=p0.device)

#             # עדכון ה-LR לפי ה-Scale של המטריצה הספציפית
#             lr_scale = max(1.0, shape[-2] / shape[-1])**0.5
#             self._muon_lr_t.fill_(group["lr"] * lr_scale)
#             red_dim = -1 if shape[-2] >= shape[-1] else -2

#             # 3. ביצוע ה-Stack ועדכון ה-Kernel בבת אחת לקבוצה הזו
#             stacked_grads = torch.stack([p.grad for p in p_list])
#             stacked_params = torch.stack(p_list)

#             muon_step_fused(
#                 stacked_grads, stacked_params, state["momentum_buffer"],
#                 state["second_momentum_buffer"], self._muon_momentum_t,
#                 self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
#                 group["ns_steps"], red_dim
#             )

#             # 4. החזרה למשקולות המקוריות
#             for i, p in enumerate(p_list):
#                 p.copy_(stacked_params[i])

#     @torch.no_grad()
#     def step(self):
#         for group in self.param_groups:
#             if group['kind'] == 'adamw':
#                 self._step_adamw(group)
#             elif group['kind'] == 'muon':
#                 self._step_muon(group)
#             else:
#                 raise ValueError(f"Unknown optimizer kind: {group['kind']}")






######################################################### Utils functions
# ==========================================
# 📊 UTILITY FUNCTIONS & STABILITY METRICS
# ==========================================

def calculate_entropy_loss(logits, targets):
    """
    In order to stabilize learning and prevent the model from becoming overly 
    confident too quickly (which leads to mode collapse or 'spiky' distributions), 
    we apply an entropy loss penalty.
    """
    # Convert logits to probabilities and log-probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Calculate Shannon entropy: H = - sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1) 

    # Create a mask to ignore padding tokens (where target == 0)
    valid_mask = (targets != 0).float()
  
    # Average the entropy only over valid tokens to avoid skewing the loss
    valid_entropy = (entropy * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
    return valid_entropy


def get_class_weights():
    """
    Calculates smoothed inverse frequency weights for class stability.
    This prevents the model from ignoring rare organisms (e.g., Mycoplasma) 
    in favor of overrepresented ones (e.g., Dictyostelium).
    """
    # Extract the hardcoded training counts for each organism in the exact order of ORG_LIST
    counts = np.array([TRAIN_COUNTS.get(name, 1) for name in ORG_LIST])
 
    # Use square root inverse scaling. This is smoother than pure inverse scaling 
    # and prevents extremely rare classes from receiving aggressively high weights.
    weights = 1.0 / np.sqrt(counts)

    # Normalize the weights so their mean equals 1.0 (maintaining overall learning rate scale)
    weights = weights / weights.sum() * len(counts)
    smart_print(f"⚖️ Max weight: {weights.max():.2f} | Min weight: {weights.min():.2f}")
    
    return torch.tensor(weights, dtype=torch.float32)


def smart_print(msg):
    """Utility to print messages with timestamps and force immediate buffer flush."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ==========================================
# 🚀 DYNAMIC IMPORT & MODEL DEFINITION
# ==========================================

# --- Import Custom Hyena DNA Backbone ---
if HYENA_PATH not in sys.path:
    sys.path.append(HYENA_PATH)

try:
    # Attempt to load the custom Hyena implementation
    from modeling_hyena import HyenaDNAModel
    smart_print("✅ Successfully imported HyenaDNAModel from new path")
except ImportError as e:
    # Fallback to local directory if the scratch path fails
    smart_print(f"❌ Failed to import modeling_hyena: {e}")
    sys.path.append(".")
    from modeling_hyena import HyenaDNAModel


# --- Import Custom ProtMamba Backbone ---
if PATH_PROT_MODEL_DIR not in sys.path:
    sys.path.append(PATH_PROT_MODEL_DIR)

try:
    # Attempt to load the specialized Protein Mamba implementation
    from prot_mamba_modules import MambaLMHeadModelwithPosids
    smart_print("✅ Successfully imported ProtMamba modules from path")
except ImportError as e:
    smart_print(f"❌ Failed to import prot_mamba_modules: {e}")
    

def get_grad_stats(model):
    """
    Diagnostic tool to track gradient norms across different architectural components.
    Crucial for detecting exploding/vanishing gradients or checking if 
    the spatial heads are overwhelming the backbones.
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Calculate the L2 norm of the gradients for this parameter tensor
        grad_norm = param.grad.norm().item()

        # Route the norm to the appropriate component category based on the layer name
        if "dna_pooling" in name: key = "pooler_grad"
        elif "cds_decoder" in name: key = "decoder_grad"
        elif "hyena" in name: key = "dna_backbone_grad"
        elif "mamba" in name: key = "prot_backbone_grad"
        else: key = "other_grad"

        stats.setdefault(key, []).append(grad_norm)

    # Return the average gradient norm for each component
    return {k: (sum(v) / len(v)) for k, v in stats.items()}


# ==========================================
# ⚙️ CUSTOM HUGGINGFACE CONFIGURATIONS
# ==========================================

# --- 1. Hyena DNA Configuration ---
class CustomHyenaConfig(PretrainedConfig):
    """
    Custom configuration class extending HuggingFace's PretrainedConfig.
    It registers all unique hyperparameters specific to the Hyena block architecture
    (e.g., filter orders, short filters, inner MLPs) so they can be saved and loaded 
    using standard `from_pretrained` pipelines.
    """
    model_type = "custom_hyena"
    def __init__(
        self, d_model=256, n_layer=8, d_inner=1024, vocab_size=12, max_seq_len=32000,
        resid_dropout=0.0, embed_dropout=0.1, fused_mlp=False, fused_dropout_add_ln=True,
        checkpoint_mixer=True, checkpoint_mlp=True, residual_in_fp32=True,
        pad_vocab_size_multiple=8, return_hidden_state=True, layer=None,
        hyena_order=2, short_filter_order=3, num_inner_mlps=2,
        hyena_dropout=0.0, hyena_filter_dropout=0.0, layer_norm_epsilon=1e-5, initializer_range=0.02, emb_dim=3,
        use_bias=True, train_freq=True, filter_order=64, activation_freq=1, 
        **kwargs
    ):
        # Base architecture dimensions
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_inner = d_inner
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Regularization & Optimization flags
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.fused_mlp = fused_mlp
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.checkpoint_mixer = checkpoint_mixer
        self.checkpoint_mlp = checkpoint_mlp
        self.residual_in_fp32 = residual_in_fp32
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.return_hidden_state = return_hidden_state

        # Hyena Operator specific parameters
        self.hyena_order = hyena_order
        self.short_filter_order = short_filter_order
        self.num_inner_mlps = num_inner_mlps
        self.hyena_dropout = hyena_dropout
        self.hyena_filter_dropout = hyena_filter_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.emb_dim = emb_dim
        self.use_bias = use_bias
        self.train_freq = train_freq
        self.filter_order = filter_order
        self.activation_freq = activation_freq

        # Initialize the specific layer dictionary if not provided
        if layer is None:
            self.layer = {
                "_name_": "hyena", "emb_dim": 5, "filter_order": 64,
                "local_order": 3, "l_max": 1000002, "modulate": True,
                "w": 10, "lr": 6e-4, "wd": 0.0, "lr_pos_emb": 0.0
            }
        else:
            self.layer = layer
        super().__init__(**kwargs)

# --- 2. ProtMamba Configuration ---
class ProtMambaConfig(PretrainedConfig):
    """
    Configuration for the Mamba-based protein backbone. 
    Handles State Space Model (SSM) specific hyperparameters like time steps, 
    state sizes, and convolution kernels.
    """
    model_type = "prot_mamba"
    def __init__(
        self, d_model=1024, n_layer=16, vocab_size=38, ssm_cfg=None, rms_norm=True,
        use_mambapy=False, residual_in_fp32=False, fused_add_norm=True, pad_vocab_size_multiple=8,
        max_position_embeddings=2048, layer_norm_epsilon=1e-5, initializer_range=0.02,
        state_size=16, expand=2, conv_kernel=4, time_step_rank=64, use_bias=False, use_conv_bias=True,
        intermediate_size=2048, hidden_act="silu",
        time_step_scale=1.0, time_step_min=0.001, time_step_max=0.1, time_step_init_scheme="random", time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        max_seq_position_embeddings=512,
        add_position_ids="1d",
        max_msa_len=32768,
        fim_strategy="multiple_span",
        always_mask=True,
        compute_only_fim_loss=True,
        **kwargs
    ):
        # Base architecture dimensions (mapped to standard HF aliases)
        self.d_model = d_model
        self.hidden_size = d_model
        self.n_layer = n_layer
        self.num_hidden_layers = n_layer
        self.vocab_size = vocab_size
        
        # Mamba/SSM specific parameters
        self.ssm_cfg = ssm_cfg if ssm_cfg is not None else {}
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_mambapy = use_mambapy

        # Inner State dynamics
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.state_size = state_size
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.time_step_rank = time_step_rank
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias

        # Discretization Step dynamics
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual

        # Protein/MSA specific sequence constraints
        self.max_seq_position_embeddings = max_seq_position_embeddings
        self.add_position_ids = add_position_ids
        self.max_msa_len = max_msa_len
        self.fim_strategy = fim_strategy
        self.always_mask = always_mask
        self.compute_only_fim_loss = compute_only_fim_loss

        super().__init__(**kwargs)


# --- 3. ProtMamba Wrapper Model ---
class ProtMambaModel(PreTrainedModel):
    """
    A HuggingFace wrapper around the raw MambaLMHeadModelwithPosids.
    This allows the custom model to be instantiated via standard HF APIs 
    like AutoModel.from_pretrained().
    """
    config_class = ProtMambaConfig

    def __init__(self, config):
        super().__init__(config)
        # Instantiate the underlying specialized Mamba architecture
        self.model = MambaLMHeadModelwithPosids(config)

    def forward(self, input_ids, position_ids=None, **kwargs):
        """
        Forward pass. Auto-generates position_ids if they are missing, 
        ensuring the SSM knows the sequence order.
        """
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Extract hidden states from the backbone (ignoring the LM head)
        hidden_states = self.model.backbone(
            input_ids=input_ids,
            position_ids=position_ids
        )

        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=hidden_states)

    def get_input_embeddings(self):
        """Helper to retrieve the embedding layer."""
        return self.model.backbone.embedding

    def set_input_embeddings(self, value):
        """Helper to override the embedding layer."""
        self.model.backbone.embedding = value


# ==========================================
# 🔌 HUGGINGFACE REGISTRY INJECTION
# ==========================================
# Explicitly link the custom configs to the model classes.
HyenaDNAModel.config_class = CustomHyenaConfig

# Register the new architectures into the HuggingFace Auto system.
# This makes them behave like native transformers models (e.g., GPT2, BERT).
AutoConfig.register("custom_hyena", CustomHyenaConfig)
AutoConfig.register("prot_mamba", ProtMambaConfig)
AutoModel.register(CustomHyenaConfig, HyenaDNAModel)
AutoModel.register(ProtMambaConfig, ProtMambaModel)


# --- TOKENIZERS ---
# ==========================================
# 🔡 1. TOKENIZERS
# ==========================================

class CodonTokenizer:
    """
    Tokenizes DNA sequences into Codons (triplets of bases).
    Mainly used for the target output of the decoder.
    """
    def __init__(self):
        self.bases = ['A', 'C', 'G', 'T']
        # Generate all 64 possible 3-letter combinations (4^3)
        self.codons = [a+b+c for a in self.bases for b in self.bases for c in self.bases]
        
        # 1-based indexing for standard codons to reserve 0 for padding
        self.vocab = {codon: i+1 for i, codon in enumerate(self.codons)}
        
        # Special tokens
        self.vocab["<pad>"] = 0
        self.vocab["<unk>"] = 65  # Unknown triplet
        self.vocab["<eos>"] = 66  # End of Sequence
        self.vocab["<bos>"] = 67  # Beginning of Sequence

    def __call__(self, text, max_codons):
        if pd.isna(text): text = ""
      
        # Truncate text leaving room for <BOS> and <EOS> tokens
        text = str(text).upper()[:(max_codons-2)*3]
        
        # Split sequence into triplets (codons)
        triplets = [text[i:i+3] for i in range(0, len(text), 3)]

        # Build the sequence: [BOS] + [Codons...] + [EOS]
        ids = [self.vocab["<bos>"]]
        ids += [self.vocab.get(t, 65) for t in triplets] # Default to <unk> (65) if not found
        ids.append(self.vocab["<eos>"])

        # Pad sequence to the exact max_codons length
        if len(ids) < max_codons:
            ids += [0] * (max_codons - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class HyenaDNATokenizer:
    """
    Ultra-fast byte-level tokenizer for DNA sequences using numpy lookups.
    Maps nucleotides to specific IDs required by the pre-trained HyenaDNA backbone.
    """
    def __init__(self):
        self.vocab = {"<pad>": 0, "A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
        
        # Creating a fast transform table for ASCII characters (size 256)
        # Default all characters to 'N' (Unknown/Any base -> ID 11)
        self.lookup = np.full(256, 11, dtype=np.int64) 
        
        # Map specific upper and lower case bases to their IDs
        for char, idx in [('A', 7), ('C', 8), ('G', 9), ('T', 10), 
                          ('a', 7), ('c', 8), ('g', 9), ('t', 10)]:
            self.lookup[ord(char)] = idx

    def __call__(self, text, max_length):
        if pd.isna(text): text = ""
        text = str(text)[:max_length]
        actual_len = len(text)

        if actual_len > 0:
            # Convert string to ASCII byte array for O(1) numpy array indexing
            byte_array = np.frombuffer(text.encode('ascii'), dtype=np.uint8)
            ids = self.lookup[byte_array].tolist()
        else:
            ids = []

        # Pad sequence and create attention mask (False = real token, True = pad)
        input_ids = torch.tensor(ids + [0]*(max_length - actual_len), dtype=torch.long)
        attention_mask = torch.ones(max_length, dtype=torch.bool)
        attention_mask[:actual_len] = False
        
        return input_ids, attention_mask


class ProtMambaTokenizer:
    """
    Ultra-fast byte-level tokenizer for Amino Acid (Protein) sequences.
    Maps amino acids to specific IDs for the pre-trained ProtMamba backbone.
    """
    def __init__(self):
        self.vocab = {c: i+1 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.vocab["<pad>"] = 0
        self.vocab["<unk>"] = 21

        # Fast lookup table for all ASCII characters, defaulting to <unk> (21)
        self.lookup = np.full(256, 21, dtype=np.int64) 
        for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY"):
            self.lookup[ord(c)] = i + 1
            self.lookup[ord(c.lower())] = i + 1

    def __call__(self, text, max_length):
        if pd.isna(text): text = ""
        text = str(text)[:max_length]
        actual_len = len(text)
    
        if actual_len > 0:
            byte_array = np.frombuffer(text.encode('ascii'), dtype=np.uint8)
            ids = self.lookup[byte_array].tolist()
        else:
            ids = []

        input_ids = torch.tensor(ids + [0]*(max_length - actual_len), dtype=torch.long)
        attention_mask = torch.ones(max_length, dtype=torch.bool)
        attention_mask[:actual_len] = False
        return input_ids, attention_mask


# ==========================================
# 🏗️ 2. DATASET
# ==========================================

class GenomicDataset(Dataset):
    """
    Custom PyTorch Dataset handling the loading, tokenization, and dynamic
    Masked Language Modeling (MLM) logic for DNA and Protein sequences.
    """
    def __init__(self, csv_file, config, is_train=True):
        self.config, self.is_train = config, is_train
        self.dna_tok, self.prot_tok, self.codon_tok = HyenaDNATokenizer(), ProtMambaTokenizer(), CodonTokenizer()

        smart_print(f"📂 Loading {'TRAIN' if is_train else 'VAL'} Dataset...")
        # Handle different potential CSV delimiters gracefully
        try:
            self.data = pd.read_csv(csv_file)
        except:
            self.data = pd.read_csv(csv_file, sep="\t")

        self.data.fillna("", inplace=True)

        # Filter out sequences that are too short to be meaningful
        if 'codon_sequence' in self.data.columns:
            self.data = self.data[self.data['codon_sequence'].str.len() > 10].copy()

        # Map string organism names to integer IDs for classification targets
        self.data['org_id'] = self.data['organism'].map(ORG_TO_ID).fillna(0).astype(int)

        # Calculate sample weights for balancing classes during training
        if self.is_train:
            counts = self.data['organism'].map(TRAIN_COUNTS)
            counts = counts.fillna(1)
            # Use inverse square root scaling to balance common vs rare organisms safely
            self.sample_weights = 1.0 / np.sqrt(counts.values)
            self.sample_weights = self.sample_weights / self.sample_weights.sum()
            
        smart_print(f"✅ Loaded {len(self.data)} samples.")

    def __len__(self): 
        return len(self.data)

    def apply_span_mask(self, ids, mask_id, span_len=1, mask_prob=0.15):
        """ 
        MLM Masking Strategy: Dynamically replaces real tokens with a mask token
        during training to force the model to learn contextual representations.
        """
        if not self.is_train: 
            return ids
            
        masked_ids = ids.clone()
        # Count actual tokens ignoring padding
        real_len = (ids != 0).sum().item()
        if real_len == 0: 
            return masked_ids

        # Determine how many contiguous spans to mask
        num_spans = int(real_len * mask_prob / span_len)
        for _ in range(num_spans):
            start = torch.randint(0, max(1, real_len - span_len), (1,)).item()
            masked_ids[start : start + span_len] = mask_id
            
        return masked_ids

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Retrieve dynamic masking configurations
        dna_span = self.config.get('span_length', 6)
        dna_prob = self.config.get('mask_prob', 0.15)
        prot_span = self.config.get('prot_span_length', 2)
        prot_prob = self.config.get('prot_mask_prob', 0.05)

        # --- 1. DNA Processing (Upstream, Downstream, Introns) ---
        up_ids, up_mask = self.dna_tok(row.get('upstream_sequence', ''), 32000)
        down_ids, down_mask = self.dna_tok(row.get('downstream_sequence', ''), 32000)

        # Process up to 5 introns, stacking them into a single tensor
        intron_results = [self.dna_tok(row.get(f'intron_{i}_sequence', ''), 2048) for i in range(1, 6)]
        int_ids = torch.stack([self.apply_span_mask(r[0], 11, dna_span, dna_prob) for r in intron_results])
        int_masks = torch.stack([r[1] for r in intron_results]) # Shape: (5, 2048)

        # Apply MLM masking to upstream and downstream sequences (11 is 'N' in DNA vocab)
        up = self.apply_span_mask(up_ids, 11, dna_span, dna_prob)
        down = self.apply_span_mask(down_ids, 11, dna_span, dna_prob)

        # --- 2. Protein Processing ---
        aa_ids, aa_mask = self.prot_tok(row.get('aa_sequence', ''), 2048)
        # Apply MLM masking to protein sequence (21 is '<unk>' in Protein vocab)
        aa = self.apply_span_mask(aa_ids, 21, prot_span, prot_prob)

        # --- 3. Decoder Target Processing (CDS) ---
        cds_out = self.codon_tok(str(row.get('codon_sequence', '')), 2048)

        # Handle different potential output formats from the tokenizer gracefully
        if isinstance(cds_out, dict):
            cds_ids = cds_out.get('input_ids', cds_out.get('ids'))
            cds_mask = cds_out.get('attention_mask', cds_out.get('mask'))
        elif isinstance(cds_out, (list, tuple)):
            cds_ids = cds_out[0]
            cds_mask = cds_out[1]
        else:
            cds_ids = cds_out
            cds_mask = (cds_ids == 0).bool()

        # --- 4. Target Classes ---
        # Calculate the Protein Abundance (PA) class by taking log10 (max 5 classes)
        raw_pa = float(row.get('protein_abundance', 0))
        pa_class = 0 if raw_pa <= 0 else min(5, int(np.log10(raw_pa + 1)))

        return {
            'up': up, 'down': down, 'introns': int_ids, 'aa': aa, 'cds': cds_ids,
            'up_mask': up_mask, 'down_mask': down_mask, 'int_masks': int_masks,
            'aa_mask': aa_mask, 'cds_mask': cds_mask,
            'org': torch.tensor(row['org_id'], dtype=torch.long),
            'pa': torch.tensor(pa_class, dtype=torch.long)
        }


# ==========================================
# 🧠 3. ARCHITECTURE
# ==========================================

class LatentPoolingBlock(nn.Module):
    """
    A Perceiver-style pooling block.
    Uses Cross-Attention to compress a long, variable-length sequence into a 
    fixed-size latent representation, followed by positional mixing and self-attention.
    """
    def __init__(self, d_model, n_latents, n_heads=8, dropout=0.1,
                 num_self_attn_layers=2, layerdrop=0.1):
        super().__init__()

        # Learnable latent queries to extract information from the long sequence
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.05)

        # Normalizations for the Cross-Attention step
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Cross Attention: Latents (Queries) look at the Sequence (Keys/Values)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                batch_first=True, dropout=dropout)

        # Gating and Residual Scaling to stabilize early training
        self.gate_param = nn.Parameter(torch.tensor(0.6))
        self.res_scale = nn.Parameter(torch.tensor(0.75))

        # Feed Forward Network (FFN)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # Positional Mixing using Depthwise 1D Convolution
        # kernel=9 provides a wider receptive field, good for genomic sequence motifs
        self.norm_pos = nn.LayerNorm(d_model)
        self.pos_conv = nn.Conv1d(d_model, d_model, kernel_size=9,
                                  padding=4, groups=d_model)

        # Self-Attention layers to allow latents to communicate with each other
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_self_attn_layers)
        ])
        
        # LayerDrop probabilistically drops entire transformer layers during training for regularization
        self.layerdrop = layerdrop
        self.latent_dropout = nn.Dropout(dropout)
        
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, x_mask=None):
        B = x.shape[0]
        # Expand the learnable latents to match the batch size
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # --- 1. Cross Attention ---
        q = self.norm_q(latents)
        k = self.norm_kv(x)
        v = self.norm_kv(x)

        # Perform cross-attention (handling padding masks dynamically)
        attn_out, attn_weights = self.cross_attn(query=q, key=k, value=v, key_padding_mask=x_mask)
        self.last_attn_weights = attn_weights

        # Compute dynamic gating factor
        gate = torch.sigmoid(self.gate_param)

        # Apply a scaled and gated residual connection
        latents = latents + 0.3 * self.res_scale * gate * attn_out

        # --- 2. FFN Block ---
        latents = latents + 0.5 * self.ffn(self.norm_ffn(latents))

        # --- 3. Positional Mixing ---
        # Transpose needed because Conv1d expects (Batch, Channels, Length)
        pos_in = self.norm_pos(latents).transpose(1, 2)
        pos_out = self.pos_conv(pos_in).transpose(1, 2)

        # Scaled residual addition for positional information
        latents = latents + 0.3 * pos_out

        # --- 4. Self-Attention & LayerDrop ---
        latents = self.latent_dropout(latents)

        for layer in self.self_attn_layers:
            # Randomly skip layers during training based on layerdrop probability
            if self.training and torch.rand(1).item() < self.layerdrop:
                continue

            # Apply "Soft Residual" logic (downscaling the layer output impact)
            latents = latents + 0.5 * (layer(latents) - latents)

        # Final normalization before passing to the next module
        latents = self.final_norm(latents)

        return latents
    

class HierarchicalLatentPooling(nn.Module):
    """
    Gradually compresses long DNA sequences into a smaller, fixed-size latent space.
    This prevents the Transformer Decoder from being overwhelmed by sequence lengths 
    (e.g., ~5000+ tokens) and aligns the DNA temporal resolution with the Protein sequence.
    """
    def __init__(self, d_model=512):
        super().__init__()
        
        # Stage 1: Compress original sequence down to 2048 latents
        self.stage1 = LatentPoolingBlock(d_model, n_latents=2048,
                                        num_self_attn_layers=2, layerdrop=0.1)
        # Stage 2: Compress 2048 down to 1024 latents
        self.stage2 = LatentPoolingBlock(d_model, n_latents=1024,
                                        num_self_attn_layers=1, layerdrop=0.1)
        # Stage 3: Compress 1024 down to 512 latents
        self.stage3 = LatentPoolingBlock(d_model, n_latents=512,
                                        num_self_attn_layers=0, layerdrop=0.1)

    def forward(self, dna_ctx, dna_mask=None):
        # Apply the original padding mask only at the first stage
        x1 = self.stage1(dna_ctx, x_mask=dna_mask)
        x1 = F.layer_norm(x1, x1.shape[-1:]) # Stabilize distribution between stages

        # Subsequent stages don't need masks because the latents are fixed-size
        x2 = self.stage2(x1, x_mask=None)
        x2 = F.layer_norm(x2, x2.shape[-1:])

        x3 = self.stage3(x2, x_mask=None)
        return x3


# ==========================================
# 🧠 2. MULTIMODAL DECODER
# ==========================================

class FlashMultimodalDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder that predicts the Coding Sequence (CDS) codon-by-codon.
    It attends to the combined DNA + Protein memory space via Cross-Attention.
    """
    def __init__(self, d_model=512, vocab_size=68, max_len=4096, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        # Standard Sinusoidal Positional Encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe.unsqueeze(0))

        self.emb_dropout = nn.Dropout(dropout)

        # 6-Layer Transformer Decoder
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model, nhead=8, dim_feedforward=d_model * 4,
                batch_first=True, norm_first=True,
                activation='gelu', dropout=dropout
            ) for _ in range(6)
        ])

        # Global Context Gate: Dynamically injects the pooled summary vector into the stream
        self.context_gate = nn.Linear(d_model, d_model)

        # Auxiliary Head (used for deep supervision in the middle of the network)
        self.aux_head = nn.Linear(d_model, vocab_size, bias=False)

        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, cds_ids, memory, context_vec, cds_padding_mask=None, memory_padding_mask=None):
        B, L = cds_ids.shape
        device = cds_ids.device
        dtype = memory.dtype

        # Create Causal Mask to prevent the model from looking ahead at future codons
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=device, dtype=dtype), diagonal=1)

        x = self.embed(cds_ids) + self.pos_encoding[:, :L, :]

        # Inject noise into embeddings during training to improve robustness
        if self.training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        x = self.emb_dropout(x)
        aux_logits = None
        context_vec = context_vec.unsqueeze(1) 

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                memory,
                tgt_mask=causal_mask,
                tgt_is_causal=True,
                tgt_key_padding_mask=cds_padding_mask,
                memory_key_padding_mask=memory_padding_mask
            )

            # Gated Context Injection: Allows the network to selectively blend the 
            # global pooled sequence context into the autoregressive hidden states
            gate = torch.sigmoid(self.context_gate(x))
            x = x + gate * context_vec

            # Extract predictions from an intermediate layer to provide a shorter gradient path
            if i == 2:
                aux_logits = self.aux_head(x)

        x = self.final_norm(x)
        return self.output_head(x), aux_logits


# ==========================================
# 🏗️ 3. MAIN FUSION MODEL
# ==========================================

class OptimizedFusionModel(nn.Module):
    """
    The Master Architecture. Combines Hyena (DNA) and ProtMamba (Amino Acids), 
    aligns their dimensionalities, applies LoRA, and routes them into the Decoder.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        smart_print("🔧 Initializing OptimizedFusionModel (Smart ProtMamba Split)...")
        dropout_val = config.get('dropout', 0.1)
     
        # Register the translation matrix used for the biological fidelity loss
        self.register_buffer('codon_to_aa_matrix', self._build_translation_matrix())
        
        # --- DNA Backbone (Hyena) Initialization ---
        self.dna_pooling = HierarchicalLatentPooling(d_model=512)
        h_cfg = CustomHyenaConfig(max_seq_len=1000002, emb_dim=5)
        raw_hyena = HyenaDNAModel(h_cfg)
        self._load_hyena_weights(raw_hyena)
        self.hyena = self._apply_lora(raw_hyena, "hyena")

        # Fix embedding pointer for weight tying/initialization
        if hasattr(self.hyena.model, 'backbone') and hasattr(self.hyena.model.backbone, 'embeddings'):
            self.hyena.get_input_embeddings = lambda: self.hyena.model.backbone.embeddings.word_embeddings

        # --- Protein Backbone (ProtMamba) Initialization ---
        m_cfg = ProtMambaConfig()
        raw_mamba = AutoModel.from_pretrained(PATH_PROT_MODEL_DIR, config=m_cfg, trust_remote_code=True, local_files_only=True)
        self._load_mamba_weights(raw_mamba)
        self.mamba = self._apply_lora(raw_mamba, "mamba")
        
        # Memory optimization: Trade compute for VRAM
        self.hyena.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Boost gradient flow to Mamba to prevent it from learning too slowly compared to Hyena
        for name, param in self.mamba.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: grad * 3.0)

        # --- Projection & Downsampling Layers ---
        # 1D Convolution to drastically reduce the temporal length of the DNA embeddings
        self.dna_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=16, stride=16),
            nn.GroupNorm(1, 256),
            nn.GELU()
        )

        # Project both modalities to a shared latent dimension (d_model = 512)
        self.DNA_proj = nn.Sequential(nn.Linear(256, 512, bias=False), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1))
        self.prot_proj = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1))

        self.norm_dna = nn.LayerNorm(512)
        self.norm_prot = nn.LayerNorm(512)
        
        # Modality Gating Parameters: Allows the network to learn how much to trust DNA vs Protein
        # Notice initialized DNA gate is very small (0.001) to force early reliance on Protein
        self.dna_gate = nn.Parameter(torch.tensor(0.001))
        self.prot_gate = nn.Parameter(torch.tensor(1.0))

        # --- GPS & Region System ---
        # Modal embeddings (0 for DNA, 1 for Protein) to help the decoder distinguish sources
        self.modal_emb = nn.Embedding(2, 512)
        
        # Region embeddings (0=Upstream, 1=Intron, 2=Downstream)
        self.region_emb = nn.Embedding(3, 512)
        
        self.dna_pos_emb = nn.Embedding(8192, 512)
        self.aa_pos_emb = nn.Embedding(2048, 512)
        self.emb_dropout = nn.Dropout(dropout_val)

        self._init_sinusoidal_pos_emb(self.dna_pos_emb)
        self._init_sinusoidal_pos_emb(self.aa_pos_emb)

        # --- Output Heads ---
        self.cds_decoder = FlashMultimodalDecoder(512, vocab_size=68, max_len=2048, dropout=dropout_val)
        
        # Classification heads for Global Objectives
        self.head_org = nn.Sequential(nn.Dropout(dropout_val), nn.Linear(512, 17)) # 17 Organism Classes
        self.head_pa = nn.Sequential(nn.Dropout(dropout_val), nn.Linear(512, 6))  # 6 Protein Abundance Classes

    def _init_sinusoidal_pos_emb(self, emb_layer):
        """Initializes an embedding layer with fixed sinusoidal weights."""
        n_pos, d_model = emb_layer.num_embeddings, emb_layer.embedding_dim
        position = torch.arange(n_pos).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(n_pos, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        emb_layer.weight.data.copy_(pe)
        emb_layer.weight.requires_grad = True # Allow fine-tuning of the encodings

   
    def _build_translation_matrix(self):
        """
        Creates a binary mapping matrix from Codon IDs to Amino Acid IDs.
        Used to calculate Translation Fidelity Loss (ensuring predicted codons 
        biologically translate to the correct amino acids).
        """
        codon_tok = CodonTokenizer()
        prot_tok = ProtMambaTokenizer()

        standard_code = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
        }

        vocab_cds_size = len(codon_tok.vocab) 
        vocab_aa_size = len(prot_tok.vocab)   

        matrix = torch.zeros(vocab_cds_size, vocab_aa_size)

        for codon, codon_id in codon_tok.vocab.items():
            if codon in standard_code:
                aa = standard_code[codon]
                if aa == '*':
                    # Stop codons are mapped to <unk> in this specific protein tokenizer setup
                    aa_id = prot_tok.vocab.get('<unk>', 21)
                else:
                    aa_id = prot_tok.vocab.get(aa, prot_tok.vocab.get('<unk>', 21))

                matrix[codon_id, aa_id] = 1.0

        return matrix

    def _apply_lora(self, model, name):
        """Injects Low-Rank Adaptation (LoRA) matrices into the model for efficient fine-tuning."""
        target_mods = self.config.get('target_modules', ["in_proj", "x_proj", "dt_proj", "out_proj", "dense", "fc1", "fc2"])

        model = get_peft_model(model, LoraConfig(
            r=self.config.get('lora_r', 32),
            lora_alpha=self.config.get('lora_alpha', 64),
            target_modules=target_mods,
            lora_dropout=0.1,
            task_type=None
        ))

        # Explicitly unfreeze normalization and convolutional layers which LoRA usually ignores
        unfreeze_keywords = ["conv", "filter", "norm", "ln"]

        print(f"\n--- 🔓 Unfreezing specific layers for {name} ---")
        for n, p in model.named_parameters():
            if any(key in n.lower() for key in unfreeze_keywords):
                if "lora" not in n.lower():
                    p.requires_grad = True
                    print(f"Unfrozen: {n}")

        return model

    def _load_hyena_weights(self, raw_model):
        """Strictly loads and maps pre-trained Hyena weights, cleaning up legacy module names."""
        smart_print("⏳ Attempting STRICT load of Hyena DNA weights...")
        checkpoint = torch.load(PATH_DNA_CHECKPOINT, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        clean_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")
            new_key = new_key.replace(".mixer.layer.", ".mixer.")
            new_key = new_key.replace(".mlp.layer.", ".mlp.")
            if "torchmetrics" in new_key or "lm_head" in new_key:
                continue
            clean_sd[new_key] = value

        raw_model.load_state_dict(clean_sd, strict=True)
        smart_print("✅ Hyena DNA weights loaded and matched PERFECTLY!")

    def _load_mamba_weights(self, raw_model):
        """
        Advanced state_dict manipulation to load pre-trained Mamba weights.
        Handles shape mismatches by truncating/padding and ignores missing artifact layers.
        """
        smart_print("⏳ Manually remapping keys for a STRICT load...")
        weights_path = f"{PATH_PROT_MODEL_DIR}/pytorch_model.bin"
        original_state_dict = torch.load(weights_path, map_location='cpu')
        
        target_state_dict = raw_model.model.state_dict()
        target_keys = set(target_state_dict.keys())
        fixed_state_dict = {}

        for key, value in original_state_dict.items():
            new_key = key
            
            # Align nested 'model' prefixes
            if new_key.startswith("model."):
                new_key = new_key.replace("model.", "", 1)

            # Map legacy 'ckpt_layer' naming conventions
            if "ckpt_layer." in new_key and new_key not in target_keys:
                new_key_no_ckpt = new_key.replace("ckpt_layer.", "")
                if new_key_no_ckpt in target_keys:
                    new_key = new_key_no_ckpt
            elif "mixer." in new_key and "ckpt_layer." not in new_key:
                new_key_with_ckpt = new_key.replace("mixer.", "mixer.ckpt_layer.")
                if new_key_with_ckpt in target_keys:
                    new_key = new_key_with_ckpt

            # Handle dimension mismatches (e.g., if vocab size changed)
            if new_key in target_keys:
                target_shape = target_state_dict[new_key].shape
              
                if value.shape != target_shape:
                    new_tensor = torch.zeros(target_shape, dtype=value.dtype, device=value.device)
                
                    if len(target_shape) == 1:
                        min_len = min(value.size(0), target_shape[0])
                        new_tensor[:min_len] = value[:min_len]
                    elif len(target_shape) == 2:
                        min_rows = min(value.size(0), target_shape[0])
                        min_cols = min(value.size(1), target_shape[1])
                        new_tensor[:min_rows, :min_cols] = value[:min_rows, :min_cols]

                    value = new_tensor
                    smart_print(f"✂️ Resized {new_key} to match model exactly: {tuple(target_shape)}")

                fixed_state_dict[new_key] = value

        missing_keys = target_keys - set(fixed_state_dict.keys())

        if missing_keys:
            smart_print(f"⚠️ Warning: {len(missing_keys)} missing keys found!")
            smart_print("🔍 LIST OF MISSING KEYS (Exist in Config/Model but NOT in Weight File):")
            smart_print("-" * 60)
         
            for mk in sorted(list(missing_keys)):
                shape = tuple(target_state_dict[mk].shape)
                print(f"❌ MISSING ARTIFACT: {mk:<50} | Shape: {shape}")

            smart_print("-" * 60)
            smart_print(f"💡 Action: These layers will be initialized to default and FROZEN to prevent training distortion.")

            # Backfill missing keys with default initialized tensors
            for mk in missing_keys:
                fixed_state_dict[mk] = target_state_dict[mk].clone()

        try:
            raw_model.model.load_state_dict(fixed_state_dict, strict=True)
            smart_print("✅ PERFECT MATCH! All weights successfully loaded with strict=True.")
        except RuntimeError as e:
            smart_print(f"❌ Strict load failed tragically: {e}")
            raise e

        # Freeze artifact layers that weren't in the pre-trained checkpoint
        if missing_keys:
            frozen_count = 0
            for name, param in raw_model.model.named_parameters():
                if name in missing_keys:
                    param.requires_grad = False
                    frozen_count += 1
            smart_print(f"❄️ SECURITY LOCK: Successfully froze {frozen_count} artifact tensors! They will act as Identity layers forever.")


    def forward(self, up, down, introns, aa, cds_target=None, dna_mask=None, prot_mask=None, cds_mask=None, debug_step=False):
        """
        The Main Forward Pass.
        Routes DNA and Protein inputs through their respective backbones, aligns them,
        pools the DNA, merges the latents, and decodes the final outputs.
        """
        device = up.device
        B = up.shape[0]
        debug_info = {} 

        # 1. Mask Downsampling (Adjust attention mask size due to Convolutional pooling)
        d_mask = dna_mask[:, ::16] if dna_mask is not None else None

        # 2. DNA Processing (Pass regions through Hyena and immediately apply 1D Conv to compress temporal dimension)
        e_up_conv = self.dna_conv(self.hyena(up).last_hidden_state.transpose(1,2)).transpose(1,2)
        e_down_conv = self.dna_conv(self.hyena(down).last_hidden_state.transpose(1,2)).transpose(1,2)

        # Process multiple introns in batch, then reshape back
        _, _, L_int = introns.shape
        e_int_raw = self.dna_conv(self.hyena(introns.reshape(-1, L_int)).last_hidden_state.transpose(1,2)).transpose(1,2)
        e_int_conv = e_int_raw.reshape(B, -1, 256)

        # Concatenate the compressed DNA regions into one continuous context
        dna_ctx = torch.cat([e_up_conv, e_int_conv, e_down_conv], dim=1)
        seq_len = dna_ctx.size(1)

        # Create Region Identifiers (0 for up, 1 for intron, 2 for down)
        len_up, len_int, len_down = e_up_conv.size(1), e_int_conv.size(1), e_down_conv.size(1)
        r_up = torch.full((len_up,), 0, device=device)
        r_int = torch.full((len_int,), 1, device=device)
        r_down = torch.full((len_down,), 2, device=device)
        region_ids = torch.cat([r_up, r_int, r_down]).unsqueeze(0).expand(B, -1)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=self.dna_pos_emb.num_embeddings - 1)

        # Apply embeddings and projections to the DNA stream
        direct_dna = self.DNA_proj(dna_ctx)
        direct_dna = direct_dna + self.region_emb(region_ids)
        direct_dna = direct_dna + self.dna_pos_emb(positions)
        direct_dna = self.emb_dropout(direct_dna)
        # Add Modality Identifier (0)
        direct_dna = self.norm_dna(direct_dna) + self.modal_emb(torch.zeros(1, device=device).long())

        # Pad or truncate the downsampled mask to match the new DNA tensor size
        if d_mask is not None:
            if d_mask.size(1) > direct_dna.size(1):
                d_mask = d_mask[:, :direct_dna.size(1)]
            elif d_mask.size(1) < direct_dna.size(1):
                pad = torch.ones(B, direct_dna.size(1) - d_mask.size(1), device=device, dtype=torch.bool)
                d_mask = torch.cat([d_mask, pad], dim=1)

     
        if debug_step:
            debug_info['before_pooler_std'] = direct_dna.std().item()

        # 3. Latent Pooling: Compress the concatenated DNA sequence down to 512 tokens
        summarized_dna = self.dna_pooling(direct_dna, dna_mask=d_mask)

    
        if debug_step:
            debug_info['after_pooler_std'] = summarized_dna.std().item()
            
            # Log entropy of attention weights to monitor if pooling is collapsing
            w1 = self.dna_pooling.stage1.last_attn_weights
            w2 = self.dna_pooling.stage2.last_attn_weights
            if w1 is not None:
                debug_info['entropy_stage1'] = -(w1 * torch.log(w1 + 1e-8)).sum(dim=-1).mean().item()
            if w2 is not None:
                debug_info['entropy_stage2'] = -(w2 * torch.log(w2 + 1e-8)).sum(dim=-1).mean().item()

        # --- 4. PROTEIN PROCESSING ---
        mamba_out = self.mamba(aa).last_hidden_state
        if prot_mask is not None:
            mamba_out = mamba_out.masked_fill(prot_mask.unsqueeze(-1), 0.0)

        e_aa = self.prot_proj(mamba_out)
        if prot_mask is not None:
            e_aa = e_aa.masked_fill(prot_mask.unsqueeze(-1), 0.0)

        positions_aa = torch.arange(e_aa.size(1), device=device).unsqueeze(0).expand(B, -1)
        positions_aa = positions_aa.clamp(max=self.aa_pos_emb.num_embeddings - 1)

        # Apply embeddings and projections to the Protein stream
        e_aa = e_aa + self.aa_pos_emb(positions_aa)
        # Add Modality Identifier (1)
        e_aa = e_aa + self.modal_emb(torch.ones(1, device=device).long())
        e_aa = self.emb_dropout(e_aa)
        current_prot = self.norm_prot(e_aa)

        # Modality Dropout Strategy: Randomly drop DNA stream 35% of the time to force 
        # the model to learn robust protein-only representations (prevents modality dominance)
        current_dna = summarized_dna
        if self.training and torch.rand(1).item() < 0.35:
            current_dna = current_dna * 0.0

        # ========================================================
        # 🔬 ALIGNMENT CHECK: Global Cosine Similarity & Norms
        # ========================================================
        dna_vec = summarized_dna.detach().mean(dim=1)
        prot_vec = current_prot.detach().mean(dim=1)

        # Check if DNA and Protein embeddings are collapsing into the same space
        cos_sim = F.cosine_similarity(dna_vec, prot_vec, dim=-1)
        
        debug_info['cos_align_mean'] = cos_sim.mean().item()
        debug_info['cos_align_std'] = cos_sim.std().item()
        debug_info['dna_vec_norm'] = dna_vec.norm(dim=-1).mean().item()
        debug_info['prot_vec_norm'] = prot_vec.norm(dim=-1).mean().item()

        # --- 5. FUSION & GATING ---
        # Apply learnable scaling gates to the two modalities
        gated_dna = current_dna * self.dna_gate
        gated_prot = current_prot * self.prot_gate

        # Concatenate modalities horizontally to form the final Multimodal Memory
        memory = torch.cat([gated_dna, gated_prot], dim=1)

        # Since DNA latents are fixed size, they don't need a padding mask (all False)
        pooled_dna_mask = torch.zeros(B, summarized_dna.size(1), dtype=torch.bool, device=device)

        if prot_mask is not None:
            current_prot_mask = prot_mask
            if current_prot_mask.size(1) > e_aa.size(1):
                current_prot_mask = current_prot_mask[:, :e_aa.size(1)]
            elif current_prot_mask.size(1) < e_aa.size(1):
                pad = torch.ones(B, e_aa.size(1) - current_prot_mask.size(1), device=device, dtype=torch.bool)
                current_prot_mask = torch.cat([current_prot_mask, pad], dim=1)
            # Combine masks for the unified memory tensor
            full_mem_mask = torch.cat([pooled_dna_mask, current_prot_mask], dim=1)
        else:
            full_mem_mask = None

        # Create a collapsed, 1D context vector representing the entire memory
        # Used for classification heads and gated injection in the decoder
        if full_mem_mask is not None:
            valid_mask = (~full_mem_mask).unsqueeze(-1).float()
            sum_memory = (memory * valid_mask).sum(dim=1)
            valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled_memory = sum_memory / valid_counts
        else:
            pooled_memory = memory.mean(dim=1)

        # --- 6. DECODER INFERENCE ---
        logits_cds, aux_logits = self.cds_decoder(
            cds_target, memory, context_vec=pooled_memory,
            cds_padding_mask=cds_mask, memory_padding_mask=full_mem_mask
        )

        if debug_step:
            with torch.no_grad():
                debug_info['dna_gate'] = self.dna_gate.item()
                debug_info['prot_gate'] = self.prot_gate.item()

                # Ablation Test: Measure how much the Decoder relies on DNA vs Protein
                # Zeros out DNA memory and measures the absolute difference in logits
                memory_no_dna = memory.clone()
                memory_no_dna[:, :summarized_dna.size(1), :] = 0.0
                logits_no_dna, _ = self.cds_decoder(
                    cds_target, memory_no_dna, context_vec=pooled_memory,
                    cds_padding_mask=cds_mask, memory_padding_mask=full_mem_mask
                )
                debug_info['dna_influence_diff'] = (logits_cds - logits_no_dna).abs().mean().item()

        # Return Predictions: Organism, Protein Abundance, Target Codons, Deep Supervision Codons
        return self.head_org(pooled_memory), self.head_pa(pooled_memory), logits_cds, aux_logits, debug_info



# ==========================================
# 🚀 5. EVAL & MAIN
# ==========================================

def print_predictions(p_cds, target_cds, tokenizer, num_examples=3):
    """
    Utility function to print a side-by-side comparison of the model's 
    predicted codons versus the ground truth codons during evaluation.
    This provides an immediate, human-readable sanity check on model performance.
    """
    smart_print("\n--- 🧬 LIVE PREDICTION SAMPLE ---")
    
    # Get the highest probability token indices for each position
    # p_cds shape: (Batch, Length, Vocab_Size) -> preds shape: (Batch, Length)
    preds = p_cds.argmax(-1) 

    # Reverse the vocabulary mapping (ID -> String) to decode the sequences
    id_to_codon = {i: c for c, i in tokenizer.vocab.items()}

    for i in range(min(num_examples, preds.size(0))):
        # Decode the first 15 valid (non-padding) tokens from the prediction
        p_seq = [id_to_codon.get(idx.item(), "?") for idx in preds[i][:15] if idx != 0]
        
        # Decode the corresponding first 15 ground truth tokens (skipping the <BOS> token at index 0)
        t_seq = [id_to_codon.get(idx.item(), "?") for idx in target_cds[i][1:16] if idx != 0]

        smart_print(f"Example {i+1}:")
        smart_print(f"  REAL: {' '.join(t_seq)}")
        smart_print(f"  PRED: {' '.join(p_seq)}")
    smart_print("---------------------------------\n")


def evaluate(model, loader, device, dtype, tokenizer): 
    """
    Standard PyTorch evaluation loop. Computes loss and accuracies over 
    the entire validation/test dataset without tracking gradients.
    """
    # Set model to evaluation mode (disables dropout, changes batchnorm behavior, etc.)
    model.eval()
    
    # Initialize metric accumulators
    m = {'loss': 0, 'acc_cds': 0, 'acc_org': 0, 'acc_pa': 0}

    is_first = True 

    # Disable gradient computation to save memory and compute time
    with torch.no_grad():
        for b in loader:
            # Move all batch tensors to the target device (GPU) asynchronously
            for k,v in b.items(): 
                b[k] = v.to(device, non_blocking=True) 

            # Flatten and concatenate the spatial masks to match the compressed DNA context
            introns_flat = b['int_masks'].view(b['up_mask'].size(0), -1)
            dna_mask = torch.cat([b['up_mask'], introns_flat, b['down_mask']], dim=1)

            # Use Automatic Mixed Precision (AMP) for faster inference
            with torch.amp.autocast('cuda', dtype=dtype):
                p_org, p_pa, p_cds, _, _ = model(
                    b['up'], b['down'], b['introns'], b['aa'], b['cds'][:, :-1],
                    dna_mask=dna_mask,
                    prot_mask=b['aa_mask'],
                    cds_mask=b['cds_mask'][:, :-1],
                    debug_step=False
                )

            # Print a visual sample of predictions only for the very first batch
            if is_first:
                print_predictions(p_cds, b['cds'], tokenizer)
                is_first = False 

            # --- Calculate Validation Metrics ---
            # Main CDS Cross Entropy loss (ignoring padding tokens)
            m['loss'] += F.cross_entropy(p_cds.reshape(-1, 68), b['cds'][:, 1:].reshape(-1), ignore_index=0).item()

            # Sequence accuracy: Mean accuracy of non-padding tokens
            m['acc_cds'] += (p_cds.argmax(-1) == b['cds'][:, 1:])[b['cds'][:, 1:] != 0].float().mean().item()
            
            # Global classification accuracies
            m['acc_org'] += (p_org.argmax(-1) == b['org']).float().mean().item()
            m['acc_pa'] += (p_pa.argmax(-1) == b['pa']).float().mean().item()

    # Average the accumulated metrics over the total number of batches
    return {k: v/len(loader) for k,v in m.items()}


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path, batch_idx, best_val_loss):
    """
    Safely saves the model, training state, and optimizer state.
    Includes memory optimizations (saving only trainable parameters) and 
    data corruption safeguards (atomic saves and NaN checks).
    """
    # Unwrap model from DistributedDataParallel (if used)
    model_to_save = model.module if hasattr(model, 'module') else model

    # Unwrap model from torch.compile() optimizations to access original weights
    model_to_save = model_to_save._orig_mod if hasattr(model_to_save, '_orig_mod') else model_to_save

    # Filter out frozen weights. Since we use massive pre-trained backbones, 
    # saving only the weights that require gradients saves gigabytes of disk space.
    trainable_state_dict = {n: p.detach().cpu() for n, p in model_to_save.named_parameters() if p.requires_grad}

    # --- Safety Check: Checkpoint Poisoning Prevention ---
    # Scan all trainable parameters for Not-a-Number (NaN) or Infinity values.
    # If the model collapsed, we abort the save so we don't overwrite a good checkpoint with trash.
    for name, param in trainable_state_dict.items():
        if torch.isnan(param).any() or torch.isinf(param).any():
            smart_print(f"❌ CRITICAL: NaN/Inf detected in parameter '{name}'. SAVE ABORTED to prevent checkpoint poisoning.")
            return False 
    # ----------------------------------

    os.makedirs(path, exist_ok=True)

    # If using PEFT/LoRA, save the adapter weights via their native save methods
    if hasattr(model_to_save, 'hyena') and hasattr(model_to_save.hyena, 'save_pretrained'):
        model_to_save.hyena.save_pretrained(f"{path}/hyena_lora")
    if hasattr(model_to_save, 'mamba') and hasattr(model_to_save.mamba, 'save_pretrained'):
        model_to_save.mamba.save_pretrained(f"{path}/mamba_lora")

    # Bundle all necessary state variables to allow for exact experiment resumption
    state = {
        'epoch': epoch,
        'step': step,
        'batch_idx': batch_idx,
        'best_val_loss': best_val_loss,
        'trainable_weights': trainable_state_dict,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    # --- Atomic Save Strategy ---
    # Write to a temporary file first, then rapidly rename it to the final filename.
    # If the process is killed (e.g., Slurm timeout) right in the middle of saving, 
    # the existing good checkpoint isn't corrupted.
    temp_path = f"{path}/training_state.pth.tmp"
    final_path = f"{path}/training_state.pth"

    torch.save(state, temp_path)
    os.replace(temp_path, final_path) 

    smart_print(f"💾 Checkpoint safely saved at Epoch {epoch}, Step {step}")

    return True



def test_overfit_single_batch(model, train_loader, optimizer, device, dtype, config):
    """
    Diagnostic Test: Overfit a Single Batch.
    Goal: Prove that the model has the capacity to learn and that gradients flow correctly.
    If the model cannot achieve near 100% accuracy on a single batch, there is a major bug.
    """
    smart_print("🔬 ========================================================")
    smart_print("🔬 STARTING 'OVERFIT ON 1 BATCH' DIAGNOSTIC TEST")
    smart_print("🔬 ========================================================")
    
    # Ensure the model is in training mode (enables dropout, batchnorm, etc., though 
    # we usually want to turn dropout off for overfitting, keeping it on train() is standard)
    model.train()

    # --- 1. Disable Random Masking ---
    # We want the model to memorize the exact same inputs. If we keep random masking on,
    # the input changes every step, making it much harder to memorize.
    original_mask_prob = config.get('mask_prob', 0.15)
    config['mask_prob'] = 0.0

    # Grab exactly ONE batch from the dataloader
    single_batch = next(iter(train_loader))

    # Restore the original mask probability in the config for later use
    config['mask_prob'] = original_mask_prob

    # Move all tensors in the single batch to the GPU
    for k, v in single_batch.items():
        single_batch[k] = v.to(device, non_blocking=True)

    # --- 2. Construct the DNA Mask ---
    # Flatten the intron masks and concatenate them with upstream and downstream masks
    # This creates a unified attention mask for the entire sequence.
    introns_flat = single_batch['int_masks'].view(single_batch['up_mask'].size(0), -1)
    dna_mask = torch.cat([single_batch['up_mask'], introns_flat, single_batch['down_mask']], dim=1)

    # Initialize the gradient scaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    # --- 3. Aggressive Learning Rate ---
    # Overwrite the scheduler/optimizer learning rate to a high value. 
    # We want aggressive memorization, not slow, generalized learning.
    for pg in optimizer.param_groups:
        pg['lr'] = 5e-4

    # --- 4. The Memorization Loop ---
    # Try to overfit for a maximum of 150 steps.
    for step in range(1, 151):
        # Clear old gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass using Mixed Precision
        with torch.amp.autocast('cuda', dtype=dtype):
            p_org, p_pa, p_cds = model(
                single_batch['up'], single_batch['down'], single_batch['introns'], single_batch['aa'], single_batch['cds'][:, :-1],
                dna_mask=dna_mask,
                prot_mask=single_batch['aa_mask'],
                cds_mask=single_batch['cds_mask'][:, :-1]
            )

            # --- Calculate Losses ---
            # Main objective: CDS prediction. ignore_index=0 skips padding tokens.
            l_cds = F.cross_entropy(p_cds.reshape(-1, 68), single_batch['cds'][:, 1:].reshape(-1), ignore_index=0)
            
            # Global objectives: Organism and Protein Abundance prediction
            l_org = F.cross_entropy(p_org, single_batch['org'])
            l_pa = F.cross_entropy(p_pa, single_batch['pa'])

            # Combine losses based on configuration weights
            w_org = config.get('l_org_weight', 0.1)
            w_pa = config.get('l_pa_weight', 0.1)
            loss = l_cds + w_org * l_org + w_pa * l_pa

        # --- Backward Pass & Optimization ---
        if dtype == torch.float16:
            # AMP backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Prevent exploding gradients
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard float32 backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        # --- Evaluate Accuracy on the fly ---
        # Create a boolean mask to ignore padding tokens (0) in accuracy calculation
        valid_tokens_mask = single_batch['cds'][:, 1:] != 0
        
        # Calculate how many non-padding tokens were predicted perfectly
        acc_cds = (p_cds.argmax(-1) == single_batch['cds'][:, 1:])[valid_tokens_mask].float().mean().item()

        # Print progress every 10 steps to monitor the loss dropping
        if step % 10 == 0:
            smart_print(f"🎯 Test Step {step:03d} | Total Loss: {loss.item():.4f} | CDS Loss: {l_cds.item():.4f} | CDS Acc: {acc_cds:.2%}")

        # --- Early Stopping on Success ---
        # If the model memorizes 98% of the batch, the architecture is fundamentally sound.
        if acc_cds >= 0.98:
            smart_print("✅ TEST PASSED! Model successfully memorized the batch (Acc > 98%). The architecture has capacity!")
            break

    # --- 5. Hard Stop ---
    # This is a diagnostic test. We intentionally kill the script here so it doesn't 
    # accidentally proceed to standard training with a corrupted (overfitted) model state.
    smart_print("🛑 OVERFIT TEST FINISHED.")
    smart_print("🛑 PLEASE STOP THE SCRIPT AND ANALYZE THE RESULTS.")
    sys.exit(0)

def main():
    # Increase PyTorch Dynamo's cache limit. 
    # This prevents recompilation warnings when using `torch.compile` on dynamic shapes.
    torch._dynamo.config.cache_size_limit = 128
    
    # Parse command line arguments to allow passing a custom JSON config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()


    # =========================================================
    # ⚙️ CONFIGURATION SETUP
    # =========================================================
    DEFAULT_CONFIG = {
        "experiment_id": "h100_final",
        'rate_adam_old_component': 0.5,
        'rate_adam_new_component': 5,
        "rate_LoRA": 5,
        "lr": 0.8e-4,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "mask_prob": 0.15,
        "span_length": 1,
        "dropout": 0.1,
        "l_org_weight": 0.1,
        "l_pa_weight": 0.1,
        "lora_dropout": 0.1,
        "use_rslora": False,
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 1,
        "grad_accum": 32,
        "epochs": 5
    }

    # Attempt to load custom configuration from a JSON file if provided via arguments
    if args.config:
        try:
            with open(args.config, 'r') as f:
                CONFIG = json.load(f)
            EXP_ID = CONFIG.get("experiment_id", "manual")
            smart_print(f"✅ Successfully loaded config: {args.config}")
        except Exception as e: 
            smart_print(f"❌ ERROR LOADING CONFIG: {e}")
            CONFIG = DEFAULT_CONFIG
            EXP_ID = "fallback"
    else:
        CONFIG = DEFAULT_CONFIG
        EXP_ID = "default"

    # =========================================================
    # 🚀 DEVICE & CUDA OPTIMIZATIONS
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIG['gpu_model'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
 
    if torch.cuda.is_available():
        # Enable TensorFloat-32 (TF32) for faster matrix multiplications on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        
        # Enable cuDNN benchmarking to automatically find the most efficient convolutional algorithms
        torch.backends.cudnn.benchmark = True

        # Explicitly enable PyTorch's optimized Scaled Dot-Product Attention (Flash Attention)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        smart_print("⚡ CUDA Optimizations: TF32, cuDNN Benchmark, and Flash Attention ENABLED.")

    # =========================================================
    # 📁 DIRECTORY & ENVIRONMENT SETUP
    # =========================================================
    # Define the experiment directory path, appending a timestamp if using default IDs
    if EXP_ID == "default" or EXP_ID == "fallback":
        EXPERIMENT_DIR = f"{BASE_TMP_PATH}/runs/{EXP_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        EXPERIMENT_DIR = f"{BASE_TMP_PATH}/runs/{EXP_ID}"

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    # Save the active configuration to the experiment folder for reproducibility
    with open(f"{EXPERIMENT_DIR}/config_used.json", "w") as f: 
        json.dump(CONFIG, f, indent=4)

    # Optimize PyTorch memory allocation to prevent fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Use bfloat16 for mixed precision if hardware supports it, otherwise fallback to float32
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    smart_print(f"🖥️ Running on: {CONFIG['gpu_model']} | Exp Dir: {EXPERIMENT_DIR}")

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="Deep_Genomic_Fusion", 
        name=EXP_ID,                 
        config=CONFIG,               
        dir=EXPERIMENT_DIR           
    )
    smart_print(f"🚀 W&B Initialized. Run ID: {wandb.run.id if wandb.run else 'None'}")

    # =========================================================
    # 📊 DATA LOADERS & SAMPLING
    # =========================================================
    # --- 🔥 SETUP BALANCED LOADER 🔥 ---
    train_ds = GenomicDataset(TRAIN_FILE, CONFIG, is_train=True)

    # Use WeightedRandomSampler to handle class imbalances in the training set
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=12,
        pin_memory=True, # Speeds up host-to-device transfers
        prefetch_factor=4,
        persistent_workers=True # Keeps workers alive between epochs to reduce overhead
    )

    val_loader = DataLoader(
        GenomicDataset(VAL_FILE, CONFIG, is_train=False), 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,  
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=False
    )

    # Fetch and apply dynamic class weights for the organism classification loss
    org_weights = get_class_weights().to(device)
    smart_print(f"⚖️ Organism Loss Weights applied (Mean: {org_weights.mean():.2f})")

    model = OptimizedFusionModel(CONFIG).to(device).to(dtype)

    # Disable caching in the backbone during training to save memory
    model.hyena.model.config.use_cache = False
    
    # Use PyTorch 2.0 compiler to optimize the custom CDS decoder operations
    smart_print("⚡ Compiling the CDS Decoder for Ada Lovelace optimization...")
    model.cds_decoder = torch.compile(model.cds_decoder)

    # =========================================================
    # ❄️ PROGRESSIVE UNFREEZING
    # =========================================================
    smart_print("❄️ APPLYING PROGRESSIVE UNFREEZING: Base Backbone vs Spatial/Norms & LoRA")

    frozen_params = 0
    unfrozen_params = 0

    # Keywords identifying layers that should remain trainable
    unfreeze_keywords = ["conv", "filter", "norm", "ln"]

    for n, p in model.named_parameters():
        is_lora = 'lora_' in n.lower()
        # Identify components that are newly added (not part of pre-trained Hyena/Mamba)
        is_new_component = not any(k in n.lower() for k in ['hyena', 'mamba'])
        # Identify spatial or normalization layers within the backbone
        is_spatial_or_norm = any(k in n.lower() for k in unfreeze_keywords) and not is_lora

        # Unfreeze LoRA adapters, newly initialized heads/decoders, and specific spatial/norm layers
        if is_lora or is_new_component or is_spatial_or_norm:
            p.requires_grad = True
            unfrozen_params += p.numel()
        # Freeze the rest of the massive pre-trained backbones
        else:
            p.requires_grad = False
            frozen_params += p.numel()

    smart_print(f"📊 Freeze Stats: Unfrozen Params: {unfrozen_params:,} | Frozen Params: {frozen_params:,}")

    # =========================================================
    # 🎛️ OPTIMIZER PARAMETER GROUPS & DIFFERENTIAL LEARNING RATES
    # =========================================================
    dec_wd, dec_no_wd = [], []
    lora_wd, lora_no_wd = [], []
    spatial_prot_wd, spatial_prot_no_wd = [], [] 
    spatial_dna_wd, spatial_dna_no_wd = [], []   
    back_wd, back_no_wd = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        
        # Do not apply weight decay to biases, layernorms, or embeddings to prevent underfitting
        no_decay = any(nd in n.lower() for nd in ["bias", "norm", "ln_f", "embeddings", "position_embedding"])
        
        is_new_component = not any(k in n.lower() for k in ['hyena', 'mamba'])
        is_lora = 'lora_' in n.lower()
        is_spatial_or_norm = any(k in n.lower() for k in unfreeze_keywords) and not is_lora

        # Route parameters into their respective lists based on component type and decay necessity
        if is_lora:
            if no_decay: lora_no_wd.append(p)
            else: lora_wd.append(p)
        elif is_new_component:
            if no_decay: dec_no_wd.append(p)
            else: dec_wd.append(p)
        elif is_spatial_or_norm:
            if 'hyena' in n.lower():
                if no_decay: spatial_dna_no_wd.append(p)
                else: spatial_dna_wd.append(p)
            else:
                if no_decay: spatial_prot_no_wd.append(p)
                else: spatial_prot_wd.append(p)
        else:
            if no_decay: back_no_wd.append(p)
            else: back_wd.append(p)

    # Define base learning rate and scale it down for specific delicate components 
    # if you want the lr from the configue insted you need to change it
    base_lr = 7.5e-5
    lr_lora = base_lr / 2      # 1.5e-5 (LoRA needs smaller steps to fine-tune effectively)
    lr_prot = base_lr / 12     # 7.5e-6 (Protein backbone spatial layers)
    lr_dna  = base_lr / 12     # 7.5e-6 (DNA backbone spatial layers)

    # Construct the final parameter groups mapping specific LR and WD rules
    param_groups = [
        {'params': dec_wd, 'lr': base_lr, 'weight_decay': CONFIG['weight_decay']},
        {'params': dec_no_wd, 'lr': base_lr, 'weight_decay': 0.0},
        {'params': lora_wd, 'lr': lr_lora, 'weight_decay': CONFIG['weight_decay']},
        {'params': lora_no_wd, 'lr': lr_lora, 'weight_decay': 0.0},
        {'params': spatial_prot_wd, 'lr': lr_prot, 'weight_decay': CONFIG['weight_decay']},
        {'params': spatial_prot_no_wd, 'lr': lr_prot, 'weight_decay': 0.0},
        {'params': spatial_dna_wd, 'lr': lr_dna, 'weight_decay': CONFIG['weight_decay']},
        {'params': spatial_dna_no_wd, 'lr': lr_dna, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
    current_decoder_lr = base_lr 

    # =========================================================
    # 📉 LEARNING RATE SCHEDULER
    # =========================================================
    steps_per_epoch = len(train_loader) // CONFIG['grad_accum']
    actual_warmup = int(CONFIG.get("warmup_steps", 800)) 
    total_steps = steps_per_epoch * CONFIG['epochs']
    decay_steps = total_steps - actual_warmup

    smart_print(f"🌡️ Warmup: {actual_warmup} steps | Total: {total_steps}")

    # Stage 1: Linear Warmup to stabilize early training
    warmup_sched = LinearLR(
        optimizer,
        start_factor=6e-5,
        end_factor=1.0,
        total_iters=actual_warmup
    )

    # Stage 2: Cosine Annealing for a smooth decay phase
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=1e-5 
    )

    # Chain the schedulers together
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[actual_warmup]
    )

    smart_print(f"✅ Combined Scheduler: Warmup ({actual_warmup} steps) -> Slow Cosine Decay")

    # Initialize AMP GradScaler for stable mixed-precision training
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    # =========================================================
    # 🔄 CHECKPOINT RESUME LOGIC (HOT START)
    # =========================================================
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    resumed_batch_idx = -1
    resume_path = f"{EXPERIMENT_DIR}/latest_checkpoint"

    if resume_path and os.path.exists(f"{resume_path}/training_state.pth"):
        smart_print(f"🔄 Found checkpoint at {resume_path}. Resuming FULL STATE (Hot Start)...")
        try:
            state = torch.load(f"{resume_path}/training_state.pth", map_location='cpu', weights_only=False)

            # Safety check: Detect corrupted checkpoints (NaN weights)
            if 'trainable_weights' in state:
                if any(torch.isnan(v).any() for v in state['trainable_weights'].values()):
                    smart_print("❌ CRITICAL: NaN detected in checkpoint! Aborting resume.")
                    raise ValueError("Poisoned Checkpoint Detected")

            # Restore global tracking variables
            best_val_loss = state.get('best_val_loss', float('inf'))
            start_epoch = state.get('epoch', 0)
            global_step = state.get('step', 0)
            resumed_batch_idx = state.get('batch_idx', -1)

            # Reload PEFT LoRA adapters specifically
            if os.path.exists(f"{resume_path}/hyena_lora"):
                model.hyena.load_adapter(f"{resume_path}/hyena_lora", "default")
            if os.path.exists(f"{resume_path}/mamba_lora"):
                model.mamba.load_adapter(f"{resume_path}/mamba_lora", "default")

            # Restore standard trainable weights mapping keys correctly
            if 'trainable_weights' in state:
                loaded_weights = state['trainable_weights']
                current_model_dict = model.state_dict()
                new_state_dict = {}

                # Helper function to align keys by stripping compilation prefixes (e.g., '_orig_mod.')
                def clean_key(k):
                    return k.replace('_orig_mod.', '').replace('module.', '')

                current_clean_to_real = {clean_key(k): k for k in current_model_dict.keys()}

                matched_count = 0
                for k, v in loaded_weights.items():
                    clean_loaded_key = clean_key(k)

                    if clean_loaded_key in current_clean_to_real:
                        target_key = current_clean_to_real[clean_loaded_key]

                        # Ensure tensor dimensions match before injecting to prevent dimension mismatch errors
                        if v.shape == current_model_dict[target_key].shape:
                            new_state_dict[target_key] = v
                            matched_count += 1
                        else:
                            smart_print(f"⚠️ Size mismatch for {clean_loaded_key}: {v.shape} vs {current_model_dict[target_key].shape}. Skipping.")

                # Load only the matched parameters, ignoring missing keys (strict=False)
                model.load_state_dict(new_state_dict, strict=False)
                smart_print(f"✅ Successfully matched and restored {matched_count} tensors (including Decoder).")

            # Restore Scheduler
            if 'scheduler' in state and scheduler is not None:
                scheduler.load_state_dict(state['scheduler'])
                smart_print("✅ Scheduler state restored successfully (Learning Rate kept).")
            else:
                smart_print("⚠️ Scheduler state NOT found in checkpoint!")

            # Restore Optimizer (Wrapped in try/except in case architecture/groups changed)
            if 'optimizer' in state and optimizer is not None:
                try:
                    optimizer.load_state_dict(state['optimizer'])
                    smart_print("✅ Optimizer state restored successfully.")
                except Exception as opt_e:
                    smart_print(f"⚠️ Optimizer structure changed. Using fresh momentum. (Error: {opt_e})")

            # Restore AMP Scaler state
            if 'scaler' in state and scaler is not None:
                try:
                    scaler.load_state_dict(state['scaler'])
                    smart_print("✅ Scaler state restored successfully.")
                except:
                    pass

            smart_print(f"🚀 Fully resumed! Starting from Epoch {start_epoch}, Step {global_step}")

        except Exception as e:
            # Fallback if checkpoint is fundamentally corrupted or loading fails
            smart_print(f"⚠️ Resume failed or aborted: {e}. Starting from scratch.")
            start_epoch, global_step, resumed_batch_idx = 0, 0, -1




    # steps_per_epoch = len(train_loader) // CONFIG['grad_accum']
    # start_epoch = state.get('epoch', 0)
    # actual_warmup = int(CONFIG.get("warmup_steps", 800)) # נמשוך מהקונפיג
    # total_steps = steps_per_epoch * CONFIG['epochs']
    # decay_steps = total_steps - actual_warmup -(start_epoch * steps_per_epoch)


    # smart_print(f"🌡️ Warmup: {actual_warmup} steps | Total: {total_steps}")

    # # כדי להגיע בדיוק ל-eta_min בצעד האחרון, decay_steps חייב להיות שווה ליתרת הצעדים

    # # --- 🔥 שלב 2: יצירת הסקדיולרים 🔥 ---

    # # Warmup: עולה מ-0 (כמעט) עד ל-Base LR שהגדרת באופטימייזר
    # warmup_sched = LinearLR(
    #     optimizer,
    #     start_factor=6e-5,
    #     end_factor=1.0,
    #     total_iters=actual_warmup
    # )

    # # Cosine Decay: יורד מהשיא עד ל-0.00001
    # cosine_sched = CosineAnnealingLR(
    #     optimizer,
    #     T_max=decay_steps,
    #     eta_min=1e-5  # 🔥 ע 0.000001
    # )

    # # חיבור הסקדיולרים
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_sched, cosine_sched],
    #     milestones=[actual_warmup]
    # )



    # smart_print(f"✅ Combined Scheduler: Warmup ({actual_warmup} steps) -> Slow Cosine Decay")

    # scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))



    # === 1. הגדרת יעדי הטיפוס ===
    # saved_lr_decoder = optimizer.param_groups[0]['lr']
    
    # target_max_decoder = base_lr           # 9e-5
    # target_max_lora = lr_lora              # 4.5e-5
    # target_max_prot = lr_prot              # 7.5e-6
    # target_max_dna = lr_dna                # 7.5e-6

    # target_min_decoder = 1e-5
    # min_ratio = target_min_decoder / target_max_decoder 

    # smart_print(f"🔄 Resuming Decoder from saved LR: {saved_lr_decoder:.6e}. Climbing to {target_max_decoder:.6e}...")

    # # === 2. דריסה חכמה ששומרת על היחסים בין הקבוצות ===
    # for i, param_group in enumerate(optimizer.param_groups):
    #     if i in [0, 1]:    # קבוצת הדיקודר
    #         target_lr = target_max_decoder
    #     elif i in [2, 3]:  # קבוצת ה-LoRA
    #         target_lr = target_max_lora
    #     elif i in [4, 5]:  # קבוצת ה-Spatial Prot
    #         target_lr = target_max_prot
    #     else:              # קבוצות ה-Spatial DNA (אינדקסים 6, 7)
    #         target_lr = target_max_dna

    #     # מקבעים את המקסימום כדי שה-Warmup וה-Lambda יוכלו לעבוד עליו
    #     param_group['lr'] = target_lr
    #     param_group['initial_lr'] = target_lr

    # # === 3. בניית ה-Recovery Warmup ===
    # recovery_steps = 100
    # start_ratio = min(1.0, saved_lr_decoder / target_max_decoder)

    # recovery_warmup = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=start_ratio,
    #     end_factor=1.0,
    #     total_iters=recovery_steps
    # )

    # # === 4. בניית ה-Cosine Decay היחסי ===
    # total_steps = (len(train_loader) // CONFIG['grad_accum']) * CONFIG['epochs']
    # remaining_steps = total_steps - global_step
    # decay_steps = remaining_steps - recovery_steps

    # if decay_steps <= 0:
    #     decay_steps = 1 

    # # פונקציה שמחזירה אחוז (מ-1.0 יורד עד min_ratio)
    # def proportional_cosine_decay(step):
    #     import math # ממוקם פה ליתר ביטחון
    #     progress = min(1.0, step / decay_steps)
    #     cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
    #     return min_ratio + (1.0 - min_ratio) * cosine_val

    # cosine_decay = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=proportional_cosine_decay
    # )

    # # === 5. חיבור שני השלבים לסקדיולר אחד ===
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[recovery_warmup, cosine_decay],
    #     milestones=[recovery_steps]
    # )

    # smart_print(f"📉 Schedulers connected: {recovery_steps} steps Recovery Warmup -> {decay_steps} steps Proportional Cosine Decay.")






    # steps_per_epoch = len(train_loader) // CONFIG['grad_accum']
    # actual_warmup = int(CONFIG.get("warmup_steps", 800))
    # total_steps = steps_per_epoch * CONFIG['epochs']
    # decay_steps = total_steps - actual_warmup

    # smart_print(f"🌡️ Warmup: {actual_warmup} steps | Total: {total_steps}")

    # warmup_sched = LinearLR(
    #     optimizer,
    #     start_factor=6e-5,
    #     end_factor=1.0,
    #     total_iters=actual_warmup
    # )

    # cosine_sched = CosineAnnealingLR(
    #     optimizer,
    #     T_max=decay_steps,
    #     eta_min=1e-5
    # )

    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_sched, cosine_sched],
    #     milestones=[actual_warmup]
    # )

    # smart_print(f"✅ Combined Scheduler: Warmup ({actual_warmup} steps) -> Slow Cosine Decay")

    # # =================================================================
    # # --- 🔥 שלב 3: הרצת הסקדיולר "על ריק" עד לנקודה שעצרנו בה 🔥 ---
    # # =================================================================
    # if global_step > 0:
    #     smart_print(f"⏩ Fast-forwarding the new Scheduler to step {global_step}...")
    #     for _ in range(global_step):
    #         scheduler.step()

    # # Log of the loss
    # log_mode = 'a' if global_step > 0 else 'w'




    # === 1. הגדרת יעדי הטיפוס לפי היחסים (Ratios) ===
    # בגלל שטענו את האופטימייזר בהצלחה, אנחנו יכולים לשלוף את ה-LR האמיתי!
    # === 1. הגדרת יעדי הטיפוס (Targets) ===
    # בגלל שטענו את האופטימייזר בהצלחה, אנחנו יכולים לשלוף את ה-LR האמיתי שממנו עצרנו!
    

    # =========================================================
    # 🕒 PURE PROPORTIONAL DECAY (NO WARMUP!)
    # יורד ישירות מה-LR הנוכחי באופטימייזר ושומר על יחס פי 10
    # =========================================================
    # =========================================================
    # 🕒 PURE PROPORTIONAL DECAY (FROM SAVED OPTIMIZER STATE)
    # =========================================================
    # steps_per_epoch = len(train_loader) // CONFIG['grad_accum']
    # total_steps = steps_per_epoch * CONFIG['epochs']
    # remaining_steps = max(1, total_steps - global_step)

    # target_decoder_min = 1e-5  # ה-LR המינימלי שאליו הדיקודר יגיע בסוף

    # # 1. שולפים את ה-LR המדויק שיש עכשיו באופטימייזר של הדיקודר (שנטען מהצ'קפוינט!)
    # current_decoder_lr = optimizer.param_groups[0]['lr']

    # # 🔥 התיקון: כופים את היחסים החדשים על שאר הקבוצות כדי למחוק את היחס הישן מהצ'קפוינט! 🔥
    # for i, param_group in enumerate(optimizer.param_groups):
    #     if i in [0, 1]:  # קבוצות הדיקודר
    #         param_group['lr'] = current_decoder_lr
    #     elif i in [2, 3]:  # קבוצות ה-LoRA
    #         param_group['lr'] = current_decoder_lr * 0.625  # 0.625 מקצב הדיקודר
    #     else:  # קבוצות ה-Spatial וה-Backbone (אינדקסים 4, 5, 6, 7)
    #         param_group['lr'] = current_decoder_lr / 16  # פי 8 לאט יותר מקצב הדיקודר

    #     # מקבעים את קו הזינוק החדש ל-PyTorch כדי שלא יקפוץ אחורה
    #     param_group['initial_lr'] = param_group['lr']

    # smart_print(f"📉 Pure Cosine Decay: Dec: {current_decoder_lr:.6e} | LoRA: {current_decoder_lr * 0.75:.6e} | Backbone: {current_decoder_lr / 8.0:.6e} downwards for {remaining_steps} steps.")

    # # 2. חישוב יחס הירידה (נקבע לפי הדיקודר וחל על כולם באופן יחסי)
    # min_ratio = min(1.0, max(0.0, target_decoder_min / current_decoder_lr))

    # # 3. פונקציית הירידה
    # def proportional_decay(step):
    #     progress = step / remaining_steps
    #     cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
    #     return min_ratio + (1.0 - min_ratio) * cosine_val

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=proportional_decay)





    # --- Logging Setup ---
    # Determine if we should Append ('a') to an existing log or start a New ('w') one 
    # based on whether we are resuming from a previous global step.
    log_mode = 'a' if global_step > 0 else 'w'
    with open(f"{EXPERIMENT_DIR}/training_log.csv", log_mode, newline='') as f:
        # Write headers only if we are starting a fresh training run
        if log_mode == 'w': 
            csv.writer(f).writerow(['epoch', 'step', 'loss_total', 'loss_cds', 'loss_org', 
                                    'loss_pa', 'val_loss', 'val_acc_cds', 'val_acc_org', 
                                    'val_acc_pa', 'sps'])

    t0 = time.time()
    # stage_start_epoch serves as a reference point for curriculum learning adjustments
    stage_start_epoch = 86
    
    smart_print("🔥 STARTING TRAINING LOOP...")
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        
        # Initialize counter for gradient accumulation cycles
        accum_counter = 0

        for i, b in enumerate(train_loader):
         
            # --- Auto-Resume Logic ---
            # If the script crashed and restarted, skip the batches that were 
            # already processed in the current epoch to maintain data consistency.
            if epoch == start_epoch and i <= resumed_batch_idx:
                if i % 50 == 0:
                    smart_print(f"⏩ Skipping batch {i} / {resumed_batch_idx}...")
                continue

            # Move batch tensors to GPU; non_blocking=True allows host-to-device 
            # transfers to overlap with CPU execution.
            for k,v in b.items(): 
                b[k] = v.to(device, non_blocking=True)

            # --- DNA Mask Construction ---
            # Flatten intron masks and concatenate them with upstream/downstream masks 
            # to create a unified mask for the entire DNA input sequence.
            introns_flat = b['int_masks'].view(b['up_mask'].size(0), -1)
            dna_mask = torch.cat([b['up_mask'], introns_flat, b['down_mask']], dim=1)

            # --- Diagnostic Trigger ---
            # Only trigger heavy debugging/logging at the end of a gradient accumulation 
            # cycle every 50 global steps to save on computation time.
            should_debug = (accum_counter == (CONFIG['grad_accum'] - 1)) and ((global_step + 1) % 50 == 0)

            # --- Forward Pass ---
            # Use Automatic Mixed Precision (AMP) to speed up training and reduce 
            # VRAM usage on supported GPUs (like H100/A100).
            with torch.amp.autocast('cuda', dtype=dtype):
                p_org, p_pa, p_cds, aux_logits, debug_info = model(
                    b['up'], b['down'], b['introns'], b['aa'], b['cds'][:, :-1],
                    dna_mask=dna_mask,
                    prot_mask=b['aa_mask'],
                    cds_mask=b['cds_mask'][:, :-1],
                    debug_step=should_debug
                )

                # --- Stochastic Logit Noise ---
                # Inject a tiny amount of Gaussian noise into the CDS predictions 
                # to act as a regularizer and prevent the model from becoming over-confident.
                noise = torch.randn_like(p_cds) * 0.005
                p_cds = p_cds + noise


                # --- Temperature Scaling ---
                temp = 1.2
                scaled_p_cds = p_cds / temp  # Soften logits to improve gradient flow

                # ========================================================
                # 🎯 Curriculum Learning: Balanced Anti-Collapse Logic
                # ========================================================
                # This section dynamically decides which tokens the model should 
                # focus on to prevent it from "collapsing" into easy/repetitive predictions.
                stage_epoch = -1
                target_cds = b['cds'][:, 1:].clone()
                loss_targets = target_cds.clone()

                # Calculate confidence levels and predicted tokens
                probs = torch.softmax(p_cds.detach(), dim=-1)
                confidence, pred_tokens = probs.max(dim=-1)

                # Identify tokens for the "Learning Mask"
                random_mask = torch.rand_like(confidence) < 0.10  # 10% random exploration
                uncertain_mask = confidence < 0.65               # Focus on low-confidence tokens
                
                # Arrogant Error: Tokens where the model is very confident but WRONG
                arrogant_error_mask = (pred_tokens != target_cds) & (confidence >= 0.75)

                # Repeat Mask: Identify tokens where the model is repeating the same prediction
                repeat_mask = (pred_tokens[:, 1:] == pred_tokens[:, :-1])
                repeat_mask = F.pad(repeat_mask, pad=(1, 0), value=False)

                # Currently forcing learning on all tokens (can be swapped with logic above)
                learn_mask = torch.ones_like(confidence).bool()
                entropy_weight = 0.005
                
                # Ensure the first token is always learned (usually a Start token)
                learn_mask[:, 0] = True

                # Mask out tokens that shouldn't contribute to loss (set to -100 for ignore_index)
                IGNORE_INDEX = -100
                loss_targets[~learn_mask] = IGNORE_INDEX
                loss_targets[target_cds == 0] = IGNORE_INDEX # Ignore Padding

                # Set loss weights: Increase weight for difficult "Arrogant Errors" and repetitions
                weights = torch.ones_like(target_cds, dtype=torch.float)
                weights[arrogant_error_mask] = 2.0
                weights[repeat_mask] = 2.0
                weights[~learn_mask] = 0.0
                weights[target_cds == 0] = 0.0

                # --- Main CDS Cross Entropy Loss ---
                l_cds_raw = F.cross_entropy(scaled_p_cds.reshape(-1, 68), loss_targets.reshape(-1), 
                                            ignore_index=IGNORE_INDEX, label_smoothing=0.05, reduction='none')
                flat_weights = weights.reshape(-1)
                l_cds = (l_cds_raw * flat_weights).sum() / (flat_weights.sum() + 1e-8)

                # --- Auxiliary Loss (if provided by the model) ---
                l_aux = 0.0
                if aux_logits is not None:
                    scaled_aux = aux_logits / temp
                    l_aux_raw = F.cross_entropy(scaled_aux.reshape(-1, 68), loss_targets.reshape(-1), 
                                                ignore_index=IGNORE_INDEX, label_smoothing=0.05, reduction='none')
                    l_aux = (l_aux_raw * flat_weights).sum() / (flat_weights.sum() + 1e-8)

                # ========================================================
                # 🧬 Translation Fidelity Loss & Entropy
                # ========================================================
                # Biologically motivated: Ensures codon predictions translate to the correct Amino Acids (AA)
                sharp_temp = 0.85
                codon_probs_sharp = F.softmax(scaled_p_cds / sharp_temp, dim=-1)

                # Map Codon probabilities to AA probabilities using a pre-defined matrix
                aa_probs = torch.matmul(codon_probs_sharp, model.codon_to_aa_matrix.to(device))
                log_aa_probs = torch.log(aa_probs.clamp(min=1e-7))

                seq_len = min(log_aa_probs.size(1), b['aa'].size(1) - 1)
                target_aa = b['aa'][:, 1:seq_len+1].clone()
                pred_log_aa = log_aa_probs[:, :seq_len, :]

                # Mask invalid AAs (Padding or Unknowns)
                valid_aa_mask = (target_aa != 0) & (target_aa != 21)
                final_aa_mask = valid_aa_mask

                loss_aa_targets = target_aa.clone()
                loss_aa_targets[~final_aa_mask] = IGNORE_INDEX

                # Calculate Negative Log Likelihood for the translation
                l_translation_raw = F.nll_loss(pred_log_aa.reshape(-1, 22), loss_aa_targets.reshape(-1), 
                                               ignore_index=IGNORE_INDEX, reduction='none')
                aa_weights = torch.ones_like(l_translation_raw)
                l_translation = (l_translation_raw * aa_weights).sum() / (aa_weights.sum() + 1e-8)

                w_trans = 1.0 # Weight for biological fidelity constraint
                
                # --- Entropy Loss ---
                # Encourages the model to explore and prevents peaky, overconfident distributions
                codon_probs_normal = F.softmax(scaled_p_cds, dim=-1)
                entropy = -(codon_probs_normal * torch.log(codon_probs_normal.clamp(min=1e-8))).sum(dim=-1)
                l_entropy = -entropy.mean()
                

                # ========================================================
                # 🌍 Global Losses & Final Combination
                # ========================================================
                # l_org: Organism classification loss
                # l_pa: Protein Abundance classification loss
                l_org = F.cross_entropy(p_org, b['org'], weight=org_weights)
                l_pa = F.cross_entropy(p_pa, b['pa'])
                w_org = CONFIG.get('l_org_weight', 0.1)
                w_pa = CONFIG.get('l_pa_weight', 0.1)

                # Final weighted loss sum
                loss = (l_cds + 0.1 * l_aux + w_trans * l_translation + 
                        entropy_weight * l_entropy + w_org * l_org + w_pa * l_pa) / CONFIG['grad_accum']

            # --- Backward Pass ---
            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_counter += 1

            # --- Optimizer Step (Gradient Accumulation) ---
            if accum_counter == CONFIG['grad_accum']:
                if dtype == torch.float16:
                    scaler.unscale_(optimizer)

                # Capture gradient statistics if debugging is enabled
                if should_debug:
                    grad_dict = get_grad_stats(model)
                    debug_info.update(grad_dict)

                # Clip gradients to prevent exploding gradient issues
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

                if dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Cleanup for next accumulation cycle
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                accum_counter = 0

                # --- Logging (Every 10 steps) ---
                if global_step % 10 == 0:
                    dt = time.time() - t0
                    sps = (10 * CONFIG['batch_size'] * CONFIG['grad_accum']) / dt if dt > 0 else 0

                    # Extract Learning Rates for different parameter groups
                    lr_dec = optimizer.param_groups[0]['lr']   # Decoder LR
                    lr_lora = optimizer.param_groups[2]['lr']  # LoRA adapter LR
                    lr_prot = optimizer.param_groups[4]['lr']  # Protein backbone LR
                    lr_dna = optimizer.param_groups[6]['lr']   # DNA backbone LR

                    smart_print(f"Step {global_step} | Total: {loss.item()*CONFIG['grad_accum']:.4f} | "
                                f"CDS: {l_cds.item():.4f} | LR_Dec: {lr_dec:.8f} | SPS: {sps:.2f}")
                    t0 = time.time()

                    # Report metrics to Weights & Biases
                    wandb_dict = {
                        "train/total_loss": loss.item() * CONFIG['grad_accum'],
                        "train/cds_loss": l_cds.item(),
                        "train/org_loss": l_org.item(),      
                        "train/pa_loss": l_pa.item(),       
                        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "train/lr_decoder": lr_dec,
                        "train/lr_lora": lr_lora,
                        "train/lr_prot": lr_prot,           
                        "train/lr_dna": lr_dna,             
                        "train/sps": sps,
                        "global_step": global_step
                    }
                    wandb.log(wandb_dict)

                # --- Diagnostic Prints ---
                if should_debug:
                    smart_print("\n" + "🔥"*25)
                    smart_print(f"🔬 DEEP DIAGNOSTICS (Triggered at Step {global_step}):")
                    smart_print(f"  • Variance Before Pooler: {debug_info.get('before_pooler_std', 0):.4f} | After: {debug_info.get('after_pooler_std', 0):.4f}")
                    smart_print(f"  • Attention Entropy - Stage 1: {debug_info.get('entropy_stage1', 0):.4f} | Stage 2: {debug_info.get('entropy_stage2', 0):.4f}")
                    smart_print(f"  • DNA Influence on Decoder: {debug_info.get('dna_influence_diff', 0):.4f}")
                    smart_print(f"  • Gradient Flow (Norms):")
                    smart_print(f"      - Decoder: {debug_info.get('decoder_grad', 0):.4f}")
                    smart_print(f"      - Pooler: {debug_info.get('pooler_grad', 0):.4f}")
                    smart_print(f"      - Backbone DNA: {debug_info.get('dna_backbone_grad', 0):.4f}")
                    smart_print(f"      - Backbone Prot: {debug_info.get('prot_backbone_grad', 0):.4f}")
                    smart_print("🔥"*25 + "\n")
                    smart_print(f"  • Cosine Alignment: Mean = {debug_info.get('cos_align_mean', 0):.4f} | Std = {debug_info.get('cos_align_std', 0):.4f}")
                    smart_print(f"  • Vector Norms: DNA = {debug_info.get('dna_vec_norm', 0):.4f} | Prot = {debug_info.get('prot_vec_norm', 0):.4f}")

                    wandb.log(debug_info)

                # --- Periodic Checkpointing ---
                if global_step % 100 == 0:
                    smart_print(f"💾 Auto-saving mid-epoch checkpoint at Step {global_step} (Batch {i})...")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        step=global_step,
                        path=f"{EXPERIMENT_DIR}/latest_checkpoint",
                        batch_idx=i,
                        best_val_loss=best_val_loss
                    )

        # Reset batch index tracking for next epoch
        resumed_batch_idx = -1

        # --- End of Epoch Validation ---
        smart_print(f"🔍 End of Epoch {epoch+1} Validation...")
        vr = evaluate(model, val_loader, device, dtype, train_ds.codon_tok)

        # Log validation metrics to W&B
        wandb.log({
            "val/loss": vr['loss'],
            "val/acc_cds": vr['acc_cds'],
            "val/acc_org": vr['acc_org'],
            "val/acc_pa": vr['acc_pa'],
            "epoch": epoch + 1
        })

        lr_dec = optimizer.param_groups[0]['lr']
        smart_print(f"📊 Epoch {epoch+1} Val: Loss {vr['loss']:.4f} | CDS {vr['acc_cds']:.2%} | "
                    f"Org {vr['acc_org']:.2%} | PA {vr['acc_pa']:.2%} | LR_Dec: {lr_dec:.6f}")
        
        # Save validation results to CSV
        with open(f"{EXPERIMENT_DIR}/training_log.csv", 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, global_step, None, None, None, None, 
                                     vr['loss'], vr['acc_cds'], vr['acc_org'], vr['acc_pa'], None])

        # End of epoch state save
        save_checkpoint(model, optimizer, scheduler, scaler, epoch+1, global_step, f"{EXPERIMENT_DIR}/latest_checkpoint", -1, best_val_loss)

        # Handle Best Model tracking
        if vr['loss'] < best_val_loss:
            best_val_loss = vr['loss']
            smart_print(f"🌟 New Best Model! Loss: {best_val_loss:.4f}")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch+1, global_step, f"{EXPERIMENT_DIR}/best_model_checkpoint", -1, best_val_loss)

if __name__ == "__main__": main()