import torch
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
from datetime import datetime
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup, PretrainedConfig, get_cosine_schedule_with_warmup, PreTrainedModel
from peft import LoraConfig, get_peft_model
import wandb # Weights & Biases for experiment tracking

# ==========================================
# 🛠️ 0. CONFIGURATION & SETUP
# ==========================================
# Define base paths for the cluster environment
BASE_TMP_PATH = BASE_TMP_PATH = os.environ.get("BASE_TMP_PATH", "/scratch200/tallilo/deep_learning_project")
TRAIN_FILE = f"{BASE_TMP_PATH}/data/TRAIN_HOMOLOGY_SPLIT.csv"
VAL_FILE = f"{BASE_TMP_PATH}/data/TEST_HOMOLOGY_SPLIT.csv"
PATH_DNA_CHECKPOINT = f"{BASE_TMP_PATH}/models/hyena_dna/HYENA_DNA_weights.ckpt"
PATH_PROT_MODEL_DIR = f"{BASE_TMP_PATH}/models/protmamba"


HYENA_PATH = f"{BASE_TMP_PATH}/models/hyena_dna"

# Append custom model directories to sys.path for local imports
sys.path.append(f"{BASE_TMP_PATH}/models/hyena_dna")
sys.path.append(f"{BASE_TMP_PATH}/models/protmamba")

# Hardcoded organism list to ensure consistent ID mapping across runs
ORG_LIST = [
    'Bacillus_subtilis', 'Cryptococcus_neoformans', 'Deinococcus_radiodurans',
    'Dictyostelium_discoideum', 'E_coli', 'Halobacterium_salinarum',
    'Helicobacter_pylori', 'Methanocaldococcus_jannaschii', 'Mycobacterium_tuberculosis',
    'Mycoplasma_pneumoniae', 'Neurospora_crassa', 'Pseudomonas_aeruginosa',
    'Saccharomyces_cerevisiae', 'Salmonella_typhimurium', 'Schizosaccharomyces_pombe',
    'Staphylococcus_aureus', 'Streptococcus_pneumoniae'
]
ORG_TO_ID = {name: i for i, name in enumerate(ORG_LIST)}

# Specific sample counts from the training set to calculate class weights
TRAIN_COUNTS = {
    'Bacillus_subtilis': 3143, 'Cryptococcus_neoformans': 1998, 'Deinococcus_radiodurans': 1656,
    'Dictyostelium_discoideum': 5736, 'E_coli': 3280, 'Halobacterium_salinarum': 1327,
    'Helicobacter_pylori': 1243, 'Methanocaldococcus_jannaschii': 621, 'Mycobacterium_tuberculosis': 2729,
    'Mycoplasma_pneumoniae': 311, 'Neurospora_crassa': 4497, 'Pseudomonas_aeruginosa': 3532,
    'Saccharomyces_cerevisiae': 3136, 'Salmonella_typhimurium': 2476, 'Schizosaccharomyces_pombe': 3434,
    'Staphylococcus_aureus': 1671, 'Streptococcus_pneumoniae': 1300
}

def get_class_weights():
    """Calculates inverse frequency weights to handle class imbalance during training."""
    counts = np.array([TRAIN_COUNTS.get(name, 1) for name in ORG_LIST])
    weights = 1.0 / counts
    return torch.tensor(weights / weights.sum() * len(counts), dtype=torch.float32)

def smart_print(msg):
    """Utility function to print messages with a timestamp and force buffer flush."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path, batch_idx=-1):
    """Safely saves the training state, including the specific batch index for precise resuming."""
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    torch.save({
        'full_model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'step': step,
        'batch_idx': batch_idx, # Saves exact position within the epoch
    }, f"{path}/training_state.pth")
    smart_print(f"💾 Full checkpoint saved at Epoch {epoch}, Step {step}, Batch {batch_idx}")



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    Useful for verifying LoRA injections.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    smart_print(
        f"🔍 PARAMETER CHECK: \n"
        f"   - Trainable params: {trainable_params:,}\n"
        f"   - Total params: {all_param:,}\n"
        f"   - Percentage Trainable: {100 * trainable_params / all_param:.4f}%"
    )


# ==========================================
#  1. CUSTOM CONFIGURATIONS & MODELS
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
#  CUSTOM HUGGINGFACE CONFIGURATIONS
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
    
    
    supports_gradient_checkpointing = True 

    def __init__(self, config):
        super().__init__(config)
        self.model = MambaLMHeadModelwithPosids(config)

   
    def _set_gradient_checkpointing(self, module, value=False):
        pass

    def forward(self, input_ids, position_ids=None, **kwargs):
        """
        Forward pass. Auto-generates position_ids if they are missing, 
        ensuring the SSM knows the sequence order, SAFELY clamped to model limits.
        """
        if position_ids is None:
            seq_len = input_ids.shape[1]
            pos = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            
           
            max_pos = getattr(self.config, 'max_position_embeddings', 2048) - 1
            pos = torch.clamp(pos, max=max_pos)
            
            position_ids = pos.unsqueeze(0).expand_as(input_ids)

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
#  HUGGINGFACE REGISTRY INJECTION
# ==========================================
# Explicitly link the custom configs to the model classes.
HyenaDNAModel.config_class = CustomHyenaConfig

# Register the new architectures into the HuggingFace Auto system.
# This makes them behave like native transformers models (e.g., GPT2, BERT).
AutoConfig.register("custom_hyena", CustomHyenaConfig)
AutoConfig.register("prot_mamba", ProtMambaConfig)
AutoModel.register(CustomHyenaConfig, HyenaDNAModel)
AutoModel.register(ProtMambaConfig, ProtMambaModel)




# ==========================================
# 🔡 2. TOKENIZERS
# ==========================================

class CodonTokenizer:
    """Tokenizes sequences into DNA triplets (codons) for the decoder target."""
    def __init__(self):
        self.bases = ['A', 'C', 'G', 'T']
        # Generate all 64 possible DNA triplets
        self.codons = [a+b+c for a in self.bases for b in self.bases for c in self.bases]
        # Map codons to IDs 1-64; save 0 for padding
        self.vocab = {codon: i+1 for i, codon in enumerate(self.codons)}
        self.vocab["<pad>"] = 0; self.vocab["<unk>"] = 65; self.vocab["<eos>"] = 66
        
    def __call__(self, text, max_codons):
        text = str(text).upper() if not pd.isna(text) else ""
        # Slice the sequence into triplets (codons) up to the maximum allowed length
        triplets = [text[i:i+3] for i in range(0, min(len(text), (max_codons-1)*3), 3)]
        # Convert triplets to IDs and append the End of Sequence (EOS) token
        ids = [self.vocab.get(t, 65) for t in triplets] + [66]
        
        # Return a PyTorch tensor padded to the fixed max_codons length
        return torch.tensor(ids + [0] * max(0, max_codons - len(ids)), dtype=torch.long)

class HyenaDNATokenizer:
    """Byte-level tokenizer for the HyenaDNA backbone."""
    def __init__(self): 
        # Specific vocabulary mapping required by the pre-trained HyenaDNA weights
        self.vocab = {"<pad>": 0, "A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
        
    def __call__(self, text, max_length):
        text_str = str(text).upper() if not pd.isna(text) else ""
        # Map each character to its specific ID; default to 'N' (11) for unknown bases
        ids = [self.vocab.get(c, 11) for c in text_str[:max_length]]
        
        # Return as a PyTorch tensor with zero-padding
        return torch.tensor(ids + [0]*max(0, max_length-len(ids)), dtype=torch.long)

class ProtMambaTokenizer:
    """Byte-level tokenizer for the ProtMamba amino acid backbone."""
    def __init__(self):
        # Map the 20 standard amino acids to IDs 1-20
        self.vocab = {c: i+1 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.vocab["<pad>"] = 0; self.vocab["<unk>"] = 21
        
    def __call__(self, text, max_length):
        text_str = str(text).upper() if not pd.isna(text) else ""
        # Map amino acids to IDs; use 21 for any unknown characters
        ids = [self.vocab.get(c, 21) for c in text_str[:max_length]]
        
        # Return as a PyTorch tensor with zero-padding
        return torch.tensor(ids + [0]*max(0, max_length-len(ids)), dtype=torch.long)



# ==========================================
# 🏗️ 3. DATASET
# ==========================================
class GenomicDataset(Dataset):
    """Custom PyTorch dataset handling data loading, tokenization, and dynamic MLM masking."""
    
    def __init__(self, csv_file, config, is_train=True):
        self.config, self.is_train = config, is_train
        self.dna_tok, self.prot_tok, self.codon_tok = HyenaDNATokenizer(), ProtMambaTokenizer(), CodonTokenizer()
        
        smart_print(f"📂 Loading {'TRAIN' if is_train else 'VAL'} Dataset from {csv_file}...")
        
        # 1. Define exactly which columns we need. 
        # This prevents Pandas from loading unnecessary metadata into RAM, preventing OOM hangs.
        needed_cols = [
            'upstream_sequence', 'downstream_sequence', 'aa_sequence', 
            'codon_sequence', 'organism', 'protein_abundance'
        ]
        # Dynamically add the 5 intron columns
        needed_cols += [f'intron_{i}_sequence' for i in range(1, 6)]

        try:
            # 2. Use engine='c' for speed and low_memory=False to stop Pandas from guessing data types
            self.data = pd.read_csv(
                csv_file, 
                usecols=needed_cols, 
                engine='c', 
                low_memory=False
            )
            smart_print(f"✅ CSV Loaded successfully. Raw rows: {len(self.data)}")
        except Exception as e:
            smart_print(f"⚠️ Standard load failed, trying tab separator... Error: {e}")
            self.data = pd.read_csv(csv_file, sep="\t", usecols=needed_cols, engine='c', low_memory=False)

        # 3. Clean empty values immediately to stabilize the dataframe
        self.data.fillna("", inplace=True)
        
        # 4. Filter out sequences that are too short to process
        if 'codon_sequence' in self.data.columns: 
            # Using .copy() is crucial here. It forces Python to release the memory of the discarded rows.
            self.data = self.data[self.data['codon_sequence'].str.len() > 10].copy()
            smart_print(f"✅ Filtered short sequences. Remaining: {len(self.data)}")

        # Map organism strings to integer IDs
        self.data['org_id'] = self.data['organism'].map(ORG_TO_ID).fillna(0).astype(int)
        
        # Calculate sample weights for the WeightedRandomSampler
        if self.is_train:
            counts = self.data['organism'].map(TRAIN_COUNTS).fillna(TRAIN_COUNTS['E_coli'])
            self.sample_weights = 1.0 / counts.values
            
        smart_print(f"✅ Dataset Ready. Total samples: {len(self.data)}")

    def __len__(self): 
        return len(self.data)
    
    def apply_span_mask(self, ids, mask_id, span_len=1):
        """Applies dynamic Masked Language Modeling (MLM) to the input sequence."""
        if not self.is_train: 
            return ids
            
        masked_ids = ids.clone()
        real_len = (ids != 0).sum().item()
        
        if real_len == 0: 
            return masked_ids
            
        # Calculate how many spans to drop based on sequence length and probability
        num_spans = int(real_len * self.config.get('mask_prob', 0.15) / span_len)
        for _ in range(num_spans):
            start = torch.randint(0, max(1, real_len - span_len), (1,)).item()
            masked_ids[start : start + span_len] = mask_id
            
        return masked_ids
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dna_span = self.config.get('span_length', 1)
        
       
        up = self.apply_span_mask(self.dna_tok(row.get('upstream_sequence', ''), 32000), 11, dna_span)
        down = self.apply_span_mask(self.dna_tok(row.get('downstream_sequence', ''), 32000), 11, dna_span)
        
      
        introns = torch.stack([
            self.apply_span_mask(self.dna_tok(row.get(f'intron_{i}_sequence', ''), 2048), 11, dna_span) 
            for i in range(1, 6)
        ])
        
       
        aa = self.apply_span_mask(self.prot_tok(row.get('aa_sequence', ''), 4096), 21, 2)
        
        # Tokenize the target sequence
        cds = self.codon_tok(str(row.get('codon_sequence', '')), 4096)
        
        # Calculate Protein Abundance class target (Log10 scaling)
        raw_pa = float(row.get('protein_abundance', 0))
        pa_target = 0 if raw_pa <= 0 else min(5, int(np.log10(raw_pa + 1)))
        
        return {
            'up': up, 
            'down': down, 
            'introns': introns, 
            'aa': aa, 
            'cds': cds, 
            'org': torch.tensor(row['org_id'], dtype=torch.long), 
            'pa': torch.tensor(pa_target, dtype=torch.long)
        }
    
# ==========================================
# 🏗️ 4. ARCHITECTURE COMPONENTS
# ==========================================

class LatentResampler(nn.Module):
    """Compresses long sequences into a fixed number of latent vectors using Cross-Attention."""
    def __init__(self, d_model=256, n_latents=512, n_heads=8, dropout=0.1):
        super().__init__()
        # Initialize learnable latent queries
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.norm_latents = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        B = x.shape[0]
        # Expand learnable latents to match batch size
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        # Cross-Attention: Latents query the raw DNA context (x)
        attn_out, _ = self.cross_attn(query=self.norm_latents(latents), key=self.norm_ctx(x), value=self.norm_ctx(x))
        return latents + attn_out + self.ffn(latents + attn_out)

class FlashMultimodalDecoder(nn.Module):
    """Autoregressive Transformer Decoder to generate the final CDS sequence using Multi-Head Output."""
    def __init__(self, d_model=256, vocab_size=67, max_len=4096, dropout=0.1, num_species=17):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, 8, 1024, batch_first=True, norm_first=True, activation='gelu', dropout=dropout) 
            for _ in range(6)
        ])
        
        #  Architectural Change: 17 separate parallel heads instead of a single global head
        # This allows each species to maintain its own unique codon usage bias weights.
        self.species_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(num_species)])
        
    def forward(self, cds_ids, memory, org_ids): 
        # Added org_ids to the signature for task-specific routing
        B, L = cds_ids.shape
        # Causal mask ensures the model cannot look ahead during sequence generation
        mask = torch.triu(torch.ones(L, L, device=cds_ids.device, dtype=torch.bool), diagonal=1)
        
        x = self.embed(cds_ids) + self.pos_embed[:, :L, :]
        for layer in self.layers: 
            x = layer(x, memory, tgt_mask=mask)
            
        #  Physical Routing: Each sample in the batch is processed by its specific species head.
        # This ensures the model applies the correct 'dialect' for the specific organism.
        out = torch.zeros(B, L, 67, device=x.device, dtype=x.dtype)
        for i in range(B):
            species_id = org_ids[i].item() # Identify which organism head to activate
            out[i] = self.species_heads[species_id](x[i])
            
        return out

class OptimizedFusionModel(nn.Module):
    """The complete multimodal architecture bridging HyenaDNA and ProtMamba."""
    def __init__(self, config):
        super().__init__()
        self.config = config; dropout_val = config.get('dropout', 0.1)
        
        smart_print("🔧 Initializing Hyena...")
        h_cfg = CustomHyenaConfig(d_model=256, d_inner=1024, n_layer=8, vocab_size=12, max_seq_len=32000, emb_dim=5, filter_order=64, short_filter_order=3, activation_freq=10, pad_vocab_size_multiple=8, embed_dropout=dropout_val, resid_dropout=dropout_val)
        self.hyena = self._apply_lora(HyenaDNAModel(h_cfg), "hyena")
        if hasattr(self.hyena.model, 'backbone') and hasattr(self.hyena.model.backbone, 'embeddings'):
            self.hyena.get_input_embeddings = lambda: self.hyena.model.backbone.embeddings.word_embeddings
        self._load_hyena_weights()

        smart_print("🔧 Initializing ProtMamba with PRE-TRAINED Weights...")
        # 1. Load the original config from the directory (without overriding dimensions to 768!)
        mamba_cfg = ProtMambaConfig.from_pretrained(PATH_PROT_MODEL_DIR)
        
        # 2. Create the raw model
        raw_mamba = ProtMambaModel(mamba_cfg)
        
        # 3. Load the pre-trained weights
        self._load_mamba_weights(raw_mamba)
        
        # 4. Wrap the model with LoRA
        self.mamba = self._apply_lora(raw_mamba, "mamba")
        
        # Enable gradient checkpointing
        self.hyena.gradient_checkpointing_enable(); self.mamba.gradient_checkpointing_enable()
        
        # Temporal compression blocks
        self.dna_conv = nn.Conv1d(256, 256, 32, stride=32); self.prot_conv = nn.Conv1d(256, 256, 2, stride=2)
        
        # 5. ---  Critical Fix ---
        # Since we removed the compression to 768, we take the linear dimension directly from Mamba's config
        self.prot_proj = nn.Linear(mamba_cfg.d_model, 256); self.norm = nn.LayerNorm(256)

        self.modal_emb = nn.Embedding(2, 256)
        self.aa_pos_emb = nn.Embedding(2048, 256)
     
        self.species_emb = nn.Embedding(50, 256)
        
        self.resampler = LatentResampler(256, 512, dropout=dropout_val)
        self.cds_decoder = FlashMultimodalDecoder(256, vocab_size=67, max_len=4096, dropout=dropout_val)
        
        # Global classification heads
        self.head_org = nn.Sequential(nn.Dropout(dropout_val), nn.Linear(256, 17))
        self.head_pa = nn.Sequential(nn.Dropout(dropout_val), nn.Linear(256, 6))

    def _apply_lora(self, model, name):
        """Applies Low-Rank Adaptation (LoRA) to efficiently fine-tune the backbones."""
        target_mods = self.config.get('target_modules', ["in_proj", "out_proj", "query", "value", "key", "dense", "fc1", "fc2"])
        return get_peft_model(model, LoraConfig(r=self.config.get('lora_r', 32), lora_alpha=self.config.get('lora_alpha', 64), target_modules=target_mods, lora_dropout=self.config.get('lora_dropout', 0.1), use_rslora=self.config.get('use_rslora', False), task_type=None))

    def _load_hyena_weights(self):
        """Loads pre-trained weights for the HyenaDNA backbone."""
        if os.path.exists(PATH_DNA_CHECKPOINT):
            s = torch.load(PATH_DNA_CHECKPOINT, map_location='cpu', weights_only=False)
            self.hyena.load_state_dict({k.replace("model.backbone.", "backbone.").replace("model.", "").replace("hyena.", ""): v for k,v in s.get('state_dict', s).items()}, strict=False)
            smart_print("✅ Pre-trained Hyena weights loaded successfully.")

    def _load_mamba_weights(self, mamba_model):
        """Loads pre-trained weights for the ProtMamba backbone safely."""
        import os
        weight_path_bin = os.path.join(PATH_PROT_MODEL_DIR, "pytorch_model.bin")
        weight_path_safetensors = os.path.join(PATH_PROT_MODEL_DIR, "model.safetensors")

        try:
            if os.path.exists(weight_path_bin):
                state_dict = torch.load(weight_path_bin, map_location='cpu', weights_only=False)
            elif os.path.exists(weight_path_safetensors):
                from safetensors.torch import load_file
                state_dict = load_file(weight_path_safetensors)
            else:
                smart_print(f"⚠️ Could not find bin/safetensors in {PATH_PROT_MODEL_DIR}. Using random weights!")
                return

            # Adjust keys in case the weights were saved without the 'model.' prefix
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                # We wrap Mamba within self.model according to the code, so ensure keys match
                if not k.startswith("model."):
                    cleaned_state_dict[f"model.{k}"] = v
                else:
                    cleaned_state_dict[k] = v

            # Load the weights into the raw model (before LoRA!)
            mamba_model.load_state_dict(cleaned_state_dict, strict=False)
            smart_print("✅ Pre-trained ProtMamba weights loaded successfully!")
            
        except Exception as e:
            smart_print(f"❌ ERROR loading ProtMamba weights: {e}")

    def _safe_hyena(self, seq):
        """Runs the Hyena backbone safely, handling any NaN outputs and applying the padding mask."""
        out = self.hyena(seq).last_hidden_state
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        mask = (seq != 0).unsqueeze(-1).to(out.dtype)
        return out * mask
    

    def _safe_mamba(self, seq):
        """Runs the Mamba backbone safely, handling any NaN outputs and applying the padding mask."""
        out = self.mamba(seq) if isinstance(self.mamba, nn.Embedding) else self.mamba(seq).last_hidden_state
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        mask = (seq != 0).unsqueeze(-1).to(out.dtype)
        return out * mask

    
    def forward(self, up, down, introns, aa, org, cds_target=None):
        device = up.device
        
       
        e_up = self.dna_conv(self._safe_hyena(up).transpose(1,2)).transpose(1,2)
        e_down = self.dna_conv(self._safe_hyena(down).transpose(1,2)).transpose(1,2)
        B, N, L = introns.shape
        e_int = self.dna_conv(self._safe_hyena(introns.reshape(-1, L)).transpose(1,2)).transpose(1,2).reshape(B, -1, 256)

        dna_ctx = torch.cat([e_up, e_int, e_down], dim=1)
        resampled_dna = self.resampler(self.norm(dna_ctx)) + self.modal_emb(torch.zeros(1, device=device).long())

      
        e_aa = self._safe_mamba(aa)
        e_aa = self.prot_conv(self.prot_proj(e_aa).transpose(1,2)).transpose(1,2)
      
        seq_len = e_aa.shape[1]
        pos_indices = torch.arange(seq_len, device=device)
        pos_indices = torch.clamp(pos_indices, max=2047) 
        pos_indices = pos_indices.unsqueeze(0).expand(B, -1)
        
        e_aa = e_aa + self.aa_pos_emb(pos_indices) + self.modal_emb(torch.ones(1, device=device).long())

     
        org = org.reshape(-1)
       
        org = torch.clamp(org, min=0, max=self.species_emb.num_embeddings - 1)
        e_species = self.species_emb(org).unsqueeze(1)
        
        
        memory = torch.cat([e_species, resampled_dna, self.norm(e_aa)], dim=1)
        
       
        return self.head_org(memory.mean(1)), self.head_pa(memory.mean(1)), self.cds_decoder(cds_target, memory, org)
    
# ==========================================
#  5. EVAL & MAIN TRAINING LOOP
# ==========================================
def evaluate(model, loader, device, dtype):
    """Evaluates the model over the entire validation loader."""
    model.eval(); m = {'loss': 0, 'acc_cds': 0, 'acc_org': 0, 'acc_pa': 0}
    with torch.no_grad():
        for b in loader:
            for k,v in b.items(): b[k] = v.to(device)
            with torch.amp.autocast('cuda', dtype=dtype):
           
                p_org, p_pa, p_cds = model(b['up'], b['down'], b['introns'], b['aa'], b['org'], b['cds'][:, :-1])
            m['loss'] += F.cross_entropy(p_cds.reshape(-1, 67).float(), b['cds'][:, 1:].reshape(-1), ignore_index=0).item()
            # Calculate accuracy excluding padding tokens
            m['acc_cds'] += (p_cds.argmax(-1) == b['cds'][:, 1:])[b['cds'][:, 1:] != 0].float().mean().item()
            m['acc_org'] += (p_org.argmax(-1) == b['org']).float().mean().item()
            m['acc_pa'] += (p_pa.argmax(-1) == b['pa']).float().mean().item()
    return {k: v/len(loader) for k,v in m.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # Default Training Configuration
    CONFIG = {
        "experiment_id": "h100_final", "lr": 1e-4, "optimizer": "adamw", "weight_decay": 0.01,
        "warmup_steps": 500, "mask_prob": 0.15, "span_length": 1, "dropout": 0.1,
        "l_org_weight":0.1, "l_pa_weight":0.1, "lora_dropout": 0.1, "use_rslora": False,
        "lora_r": 32, "lora_alpha": 64, "batch_size": 1, "grad_accum": 32, "epochs": 5
    }

    # Load custom config if provided via command line args
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            CONFIG.update(json.load(f))

    EXP_ID = CONFIG.get("experiment_id", "default")
    EXPERIMENT_DIR = f"{BASE_TMP_PATH}/runs/{EXP_ID}"
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    latest_ckpt_path = f"{EXPERIMENT_DIR}/latest_checkpoint"
    best_ckpt_path = f"{EXPERIMENT_DIR}/best_model_checkpoint"
    
    # Resume Logic: Check if an interrupted training run exists
    is_resuming = os.path.exists(f"{latest_ckpt_path}/training_state.pth")
    if is_resuming:
        smart_print(f"🔄 Checkpoint found! Loading exact configuration from previous run...")
        with open(f"{EXPERIMENT_DIR}/config_used.json", "r") as f:
            CONFIG.update(json.load(f))
    else:
        # Save the current configuration for future resumes
        with open(f"{EXPERIMENT_DIR}/config_used.json", "w") as f:
            json.dump(CONFIG, f, indent=4)

    # Initialize Weights & Biases
    # Passing resume="allow" ensures W&B seamlessly connects to the previous run graph
    wandb.init(
        project="Deep_Genomic_Fusion", 
        name=EXP_ID,
        id=EXP_ID, 
        resume="allow",
        config=CONFIG,
        dir=EXPERIMENT_DIR
    )
    smart_print(f"🚀 W&B Initialized. Run ID: {wandb.run.id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Setup Datasets and Dataloaders
    train_ds = GenomicDataset(TRAIN_FILE, CONFIG, is_train=True)
    train_sampler = WeightedRandomSampler(weights=train_ds.sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(GenomicDataset(VAL_FILE, CONFIG, is_train=False), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = OptimizedFusionModel(CONFIG).to(device).to(dtype)
    print_trainable_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG.get('weight_decay', 0.01))
    
    # Cosine scheduler total steps calculation
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=CONFIG.get('warmup_steps', 500), num_training_steps=(len(train_loader) // CONFIG['grad_accum']) * CONFIG['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    start_epoch = 0
    global_step = 0
    resumed_batch_idx = -1
    best_val_loss = float('inf')
    org_weights = get_class_weights().to(device)

    #  Load State Dictionaries if Resuming
    if is_resuming:
        try:
            state = torch.load(f"{latest_ckpt_path}/training_state.pth", map_location=device)
            if 'full_model_state' in state:
                # 1. Extract the old weights from the previous single head
                old_weight = state['full_model_state'].pop('cds_decoder.output_head.weight', None)
                old_bias = state['full_model_state'].pop('cds_decoder.output_head.bias', None)
                
                # 2. Load the main model body (the backbone)
                model.load_state_dict(state['full_model_state'], strict=False)
                
                # 3. Copy the old weights to the new parallel heads (Warm Start)
                if old_weight is not None and old_bias is not None:
                    with torch.no_grad():
                        for i in range(17):
                            model.cds_decoder.species_heads[i].weight.copy_(old_weight)
                            model.cds_decoder.species_heads[i].bias.copy_(old_bias)
                    smart_print("✅ Warm Start: Transferred weights successfully!")

                # ---  Handle Optimizer State Transition ---
                try:
                    optimizer.load_state_dict(state['optimizer'])
                    scheduler.load_state_dict(state['scheduler'])
                    scaler.load_state_dict(state['scaler'])
                    smart_print("✅ Optimizer & Scheduler states loaded.")
                except:
                    smart_print("⚠️ Optimizer mismatch detected due to architecture change. Re-initializing optimizer from scratch (Normal for this transition).")
                # ----------------------------------

                start_epoch = state['epoch']
                global_step = state['step']
                resumed_batch_idx = state['batch_idx']
                
                smart_print(f"📉 Resuming EXACTLY from Epoch {start_epoch}, Step {global_step}, Batch {resumed_batch_idx}")
            else:
                smart_print("❌ ERROR: Old format detected. Please change experiment_id for a fresh start.")
                sys.exit(1)
        except Exception as e:
            smart_print(f"❌ ERROR loading checkpoint: {e}")
            sys.exit(1)
    else:
        smart_print("🆕 No checkpoints found. Starting fresh training.")

    
    # Prepare local CSV logging
    with open(f"{EXPERIMENT_DIR}/training_log.csv", 'a' if is_resuming else 'w', newline='') as f:
        writer = csv.writer(f)
        if not is_resuming:
            writer.writerow(['epoch', 'step', 'loss_total', 'loss_cds', 'loss_org', 'loss_pa', 'val_loss', 'val_acc_cds', 'val_acc_org', 'val_acc_pa'])

    smart_print(f"🔥 STARTING TRAINING LOOP...")
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        for i, b in enumerate(train_loader):
            
            #  Fast-forward logic: Skip batches that were already processed in a previous interrupted run
            if epoch == start_epoch and i <= resumed_batch_idx:
                if i % 100 == 0:
                    smart_print(f"⏩ Fast-forwarding... Skipping batch {i} / {resumed_batch_idx}")
                continue

            # Load batch to device
            for k,v in b.items(): b[k] = v.to(device)
            
            
            # Forward pass with Automatic Mixed Precision (AMP)
            with torch.amp.autocast('cuda', dtype=dtype):
          
                MAX_SAFE_LEN = 2000 
                
                safe_cds = b['cds']
                if safe_cds.shape[1] > MAX_SAFE_LEN:
                    safe_cds = safe_cds[:, :MAX_SAFE_LEN]
                
                # Forward pass עם ה-CDS המוגן
                p_org, p_pa, p_cds = model(b['up'], b['down'], b['introns'], b['aa'], b['org'], safe_cds[:, :-1])
                
            # Calculate primary CDS Generation loss
      
            valid_cds_tokens = (safe_cds[:, 1:] != 0).sum()
            if valid_cds_tokens > 0:
                l_cds = F.cross_entropy(p_cds.reshape(-1, 67).float(), safe_cds[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
            else:
                l_cds = torch.tensor(0.0, device=device, requires_grad=True)

            # Calculate Auxiliary Classification losses
            l_org = F.cross_entropy(p_org.float(), b['org'], weight=org_weights)
            l_pa = F.cross_entropy(p_pa.float(), b['pa'])
            
            # Combine losses and scale by gradient accumulation factor
            loss = (l_cds + CONFIG.get('l_org_weight', 0.1) * l_org + CONFIG.get('l_pa_weight', 0.1) * l_pa) / CONFIG['grad_accum']

            if torch.isnan(loss) or torch.isinf(loss): continue

            # Backward pass
            if dtype == torch.float16: scaler.scale(loss).backward()
            else: loss.backward()

            # Perform Optimization step only after accumulating enough gradients
            if (i + 1) % CONFIG['grad_accum'] == 0:
                # Gradient Sanity Check to prevent model collapse
                has_nans = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in model.parameters())
                if not has_nans:
                    if dtype == torch.float16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                
                # Fetch Current Learning Rate before scheduler step for logging
                current_lr = optimizer.param_groups[0]['lr']
                
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                #  Logging (Every 5 optimization steps)
                if global_step % 5 == 0:
                    smart_print(f"Step {global_step} | Total: {loss.item()*CONFIG['grad_accum']:.4f} | CDS: {l_cds.item():.4f}")
                    with open(f"{EXPERIMENT_DIR}/training_log.csv", 'a', newline='') as f: 
                        csv.writer(f).writerow([epoch+1, global_step, loss.item()*CONFIG['grad_accum'], l_cds.item(), l_org.item(), l_pa.item(), None, None, None, None])
                    
                    # W&B Logging
                    wandb.log({
                        "train/total_loss": loss.item() * CONFIG['grad_accum'],
                        "train/cds_loss": l_cds.item(),
                        "train/org_loss": l_org.item(),
                        "train/pa_loss": l_pa.item(),
                        "train/learning_rate": current_lr,
                        "epoch": epoch + (i / len(train_loader)) # Fractional epoch tracking
                    }, step=global_step)
                
                #  Checkpointing within the epoch (Every 150 optimization steps)
                if global_step % 150 == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, f"{EXPERIMENT_DIR}/latest_checkpoint", batch_idx=i)

        # ==========================================
        #  END OF EPOCH VALIDATION
        # ==========================================
        vr = evaluate(model, val_loader, device, dtype)
        smart_print(f"📊 Epoch {epoch+1} Val: Loss {vr['loss']:.4f}")
        
        # Log validation metrics to W&B
        wandb.log({
            "val/loss": vr['loss'],
            "val/acc_cds": vr['acc_cds'],
            "val/acc_org": vr['acc_org'],
            "val/acc_pa": vr['acc_pa'],
            "epoch_complete": epoch + 1
        }, step=global_step)
        
        # Reset the resumed batch index for the next epoch
        resumed_batch_idx = -1
        
        # Overwrite the latest checkpoint at the end of the epoch
        save_checkpoint(model, optimizer, scheduler, scaler, epoch+1, global_step, f"{EXPERIMENT_DIR}/latest_checkpoint", batch_idx=-1)
        
        # Keep track of the absolute best model
        if vr['loss'] < best_val_loss:
            best_val_loss = vr['loss']
            save_checkpoint(model, optimizer, scheduler, scaler, epoch+1, global_step, f"{EXPERIMENT_DIR}/best_model_checkpoint", batch_idx=-1)
            wandb.log({"val/best_loss": best_val_loss}, step=global_step)

    # Properly close the W&B run
    wandb.finish()

if __name__ == "__main__": main()