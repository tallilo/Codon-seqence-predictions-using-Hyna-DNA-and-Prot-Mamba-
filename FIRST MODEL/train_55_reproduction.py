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

# ==========================================
# 🛠️ 0. CONFIGURATION & SETUP
# ==========================================
BASE_TMP_PATH = "/scratch200/tallilo/deep_learning_project"
TRAIN_FILE = f"{BASE_TMP_PATH}/data/TRAIN_HOMOLOGY_SPLIT.csv"
VAL_FILE = f"{BASE_TMP_PATH}/data/TEST_HOMOLOGY_SPLIT.csv"
PATH_DNA_CHECKPOINT = f"{BASE_TMP_PATH}/models/hyena_dna/HYENA_DNA_weights.ckpt"
PATH_PROT_MODEL_DIR = f"{BASE_TMP_PATH}/models/protmamba"

sys.path.append(f"{BASE_TMP_PATH}/models/hyena_dna")
sys.path.append(f"{BASE_TMP_PATH}/models/protmamba")

ORG_LIST = [
    'Bacillus_subtilis', 'Cryptococcus_neoformans', 'Deinococcus_radiodurans',
    'Dictyostelium_discoideum', 'E_coli', 'Halobacterium_salinarum',
    'Helicobacter_pylori', 'Methanocaldococcus_jannaschii', 'Mycobacterium_tuberculosis',
    'Mycoplasma_pneumoniae', 'Neurospora_crassa', 'Pseudomonas_aeruginosa',
    'Saccharomyces_cerevisiae', 'Salmonella_typhimurium', 'Schizosaccharomyces_pombe',
    'Staphylococcus_aureus', 'Streptococcus_pneumoniae'
]
ORG_TO_ID = {name: i for i, name in enumerate(ORG_LIST)}

TRAIN_COUNTS = {
    'Bacillus_subtilis': 3143, 'Cryptococcus_neoformans': 1998, 'Deinococcus_radiodurans': 1656,
    'Dictyostelium_discoideum': 5736, 'E_coli': 3280, 'Halobacterium_salinarum': 1327,
    'Helicobacter_pylori': 1243, 'Methanocaldococcus_jannaschii': 621, 'Mycobacterium_tuberculosis': 2729,
    'Mycoplasma_pneumoniae': 311, 'Neurospora_crassa': 4497, 'Pseudomonas_aeruginosa': 3532,
    'Saccharomyces_cerevisiae': 3136, 'Salmonella_typhimurium': 2476, 'Schizosaccharomyces_pombe': 3434,
    'Staphylococcus_aureus': 1671, 'Streptococcus_pneumoniae': 1300
}

def get_class_weights():
    counts = np.array([TRAIN_COUNTS.get(name, 1) for name in ORG_LIST])
    weights = 1.0 / counts
    return torch.tensor(weights / weights.sum() * len(counts), dtype=torch.float32)

def smart_print(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

class ProtMambaConfig(PretrainedConfig):
    model_type = "prot_mamba"
    def __init__(self, **kwargs): super().__init__(**kwargs)

try:
    from prot_mamba_modules import MambaLMHeadModelwithPosids
    class ProtMambaModel(PreTrainedModel):
        config_class = ProtMambaConfig
        supports_gradient_checkpointing = True
        def __init__(self, config):
            super().__init__(config)
            self.model = MambaLMHeadModelwithPosids(config)
        def _set_gradient_checkpointing(self, module, value=False): pass
        def forward(self, input_ids, position_ids=None, **kwargs):
            if position_ids is None:
                pos = torch.clamp(torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device), max=getattr(self.config, 'max_position_embeddings', 2048) - 1)
                position_ids = pos.unsqueeze(0).expand_as(input_ids)
            out = self.model.backbone(input_ids=input_ids, position_ids=position_ids)
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=out)
except ImportError:
    smart_print("⚠️ Warning: Could not import prot_mamba_modules.")

class HyenaConfig(PretrainedConfig):
    model_type = "hyenadna"
    def __init__(self, vocab_size=12, d_model=256, d_inner=None, use_bias=True, train_freq=True,
                 max_seq_len=1024, emb_dim=3, n_layer=12, num_inner_mlps=2, hyena_order=2,
                 short_filter_order=3, filter_order=64, activation_freq=1, embed_dropout=0.1,
                 hyena_dropout=0.0, hyena_filter_dropout=0.0, layer_norm_epsilon=1e-5,
                 initializer_range=0.02, pad_vocab_size_multiple=8, **kwargs):
        self.vocab_size = vocab_size; self.d_model = d_model; self.d_inner = 4 * d_model if d_inner is None else d_inner
        self.use_bias = use_bias; self.train_freq = train_freq; self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim; self.n_layer = n_layer; self.hyena_order = hyena_order
        self.filter_order = filter_order; self.short_filter_order = short_filter_order
        self.activation_freq = activation_freq; self.num_inner_mlps = num_inner_mlps
        self.embed_dropout = embed_dropout; self.hyena_dropout = hyena_dropout
        self.hyena_filter_dropout = hyena_filter_dropout; self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range; self.pad_vocab_size_multiple = pad_vocab_size_multiple
        super().__init__(**kwargs)

try:
    from modeling_hyena import HyenaDNAModel
except ImportError: pass

class CodonTokenizer:
    def __init__(self):
        self.bases = ['A', 'C', 'G', 'T']
        self.codons = [a+b+c for a in self.bases for b in self.bases for c in self.bases]
        self.vocab = {codon: i+1 for i, codon in enumerate(self.codons)}
        self.vocab["<pad>"] = 0; self.vocab["<unk>"] = 65; self.vocab["<eos>"] = 66
    def __call__(self, text, max_codons):
        text = str(text).upper() if not pd.isna(text) else ""
        triplets = [text[i:i+3] for i in range(0, min(len(text), (max_codons-1)*3), 3)]
        ids = [self.vocab.get(t, 65) for t in triplets] + [66]
        return torch.tensor(ids + [0] * max(0, max_codons - len(ids)), dtype=torch.long)

class HyenaDNATokenizer:
    def __init__(self): self.vocab = {"<pad>": 0, "A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
    def __call__(self, text, max_length):
        ids = [self.vocab.get(c, 11) for c in str(text).upper()[:max_length]]
        return torch.tensor(ids + [0]*max(0, max_length-len(ids)), dtype=torch.long)

class ProtMambaTokenizer:
    def __init__(self):
        self.vocab = {c: i+1 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}; self.vocab["<pad>"] = 0; self.vocab["<unk>"] = 21
    def __call__(self, text, max_length):
        ids = [self.vocab.get(c, 21) for c in str(text).upper()[:max_length]]
        return torch.tensor(ids + [0]*max(0, max_length-len(ids)), dtype=torch.long)

class GenomicDataset(Dataset):
    def __init__(self, csv_file, config, is_train=True):
        self.config, self.is_train = config, is_train
        self.dna_tok, self.prot_tok, self.codon_tok = HyenaDNATokenizer(), ProtMambaTokenizer(), CodonTokenizer()
        smart_print(f"📂 Loading {'TRAIN' if is_train else 'VAL'} Dataset...")
        try: self.data = pd.read_csv(csv_file)
        except: self.data = pd.read_csv(csv_file, sep="\t")
        self.data.fillna("", inplace=True)
        if 'codon_sequence' in self.data.columns: self.data = self.data[self.data['codon_sequence'].str.len() > 10]
        self.data['org_id'] = self.data['organism'].map(ORG_TO_ID).fillna(0).astype(int)
        if self.is_train:
            counts = self.data['organism'].map(TRAIN_COUNTS).fillna(TRAIN_COUNTS['E_coli'])
            self.sample_weights = 1.0 / counts.values
        smart_print(f"✅ Loaded {len(self.data)} samples.")

    def __len__(self): return len(self.data)
    def apply_span_mask(self, ids, mask_id, span_len=1):
        if not self.is_train: return ids
        masked_ids = ids.clone(); real_len = (ids != 0).sum().item()
        if real_len == 0: return masked_ids
        for _ in range(int(real_len * self.config.get('mask_prob', 0.15) / span_len)):
            start = torch.randint(0, max(1, real_len - span_len), (1,)).item()
            masked_ids[start : start + span_len] = mask_id
        return masked_ids
    def __getitem__(self, idx):
        row = self.data.iloc[idx]; dna_span = self.config.get('span_length', 1)
        up = self.apply_span_mask(self.dna_tok(row.get('upstream_sequence', ''), 32000), 11, dna_span)
        down = self.apply_span_mask(self.dna_tok(row.get('downstream_sequence', ''), 32000), 11, dna_span)
        introns = torch.stack([self.apply_span_mask(self.dna_tok(row.get(f'intron_{i}_sequence', ''), 2048), 11, dna_span) for i in range(1, 6)])
        aa = self.apply_span_mask(self.prot_tok(row.get('aa_sequence', ''), 4096), 21, 2)
        cds = self.codon_tok(str(row.get('codon_sequence', '')), 4096)
        raw_pa = float(row.get('protein_abundance', 0))
        return {'up': up, 'down': down, 'introns': introns, 'aa': aa, 'cds': cds, 'org': torch.tensor(row['org_id'], dtype=torch.long), 'pa': torch.tensor(0 if raw_pa <= 0 else min(5, int(np.log10(raw_pa + 1))), dtype=torch.long)}

class LatentResampler(nn.Module):
    def __init__(self, d_model=256, n_latents=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.norm_ctx = nn.LayerNorm(d_model); self.norm_latents = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
    def forward(self, x):
        B = x.shape[0]; latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=self.norm_latents(latents), key=self.norm_ctx(x), value=self.norm_ctx(x))
        return latents + attn_out + self.ffn(latents + attn_out)

class FlashMultimodalDecoder(nn.Module):
    def __init__(self, d_model=256, vocab_size=67, max_len=4096, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model, 8, 1024, batch_first=True, norm_first=True, activation='gelu', dropout=dropout) for _ in range(6)])
        self.output_head = nn.Linear(d_model, vocab_size)
    def forward(self, cds_ids, memory):
        B, L = cds_ids.shape
        mask = torch.triu(torch.ones(L, L, device=cds_ids.device, dtype=torch.bool), diagonal=1)
        x = self.embed(cds_ids) + self.pos_embed[:, :L, :]
        for layer in self.layers: x = layer(x, memory, tgt_mask=mask)
        return self.output_head(x)

class OptimizedFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config; dropout_val = config.get('dropout', 0.1)
        
        smart_print("🔧 Initializing Hyena...")
        h_cfg = HyenaConfig(d_model=256, d_inner=1024, n_layer=8, vocab_size=12, max_seq_len=32000, emb_dim=5, filter_order=64, short_filter_order=3, activation_freq=10, pad_vocab_size_multiple=8, embed_dropout=dropout_val, resid_dropout=dropout_val)
        self.hyena = self._apply_lora(HyenaDNAModel(h_cfg), "hyena")
        if hasattr(self.hyena.model, 'backbone') and hasattr(self.hyena.model.backbone, 'embeddings'):
            self.hyena.get_input_embeddings = lambda: self.hyena.model.backbone.embeddings.word_embeddings
        self._load_hyena_weights()

        # 🔥 Mamba מותחל עם זבל אקראי (יצירה מקונפיגורציה בלבד) 🔥
        smart_print("🔧 Initializing ProtMamba (Random 'Garbage' Weights)...")
        mamba_cfg = ProtMambaConfig.from_pretrained(PATH_PROT_MODEL_DIR)
        mamba_cfg.d_model = 768; mamba_cfg.hidden_size = 768; mamba_cfg.n_embd = 768
        raw_mamba = ProtMambaModel(mamba_cfg)
        self.mamba = self._apply_lora(raw_mamba, "mamba")

        self.hyena.gradient_checkpointing_enable(); self.mamba.gradient_checkpointing_enable()
        self.dna_conv = nn.Conv1d(256, 256, 32, stride=32); self.prot_conv = nn.Conv1d(256, 256, 2, stride=2)
        self.prot_proj = nn.Linear(768, 256); self.norm = nn.LayerNorm(256)
        self.modal_emb = nn.Embedding(2, 256); self.aa_pos_emb = nn.Embedding(2048, 256)
        self.resampler = LatentResampler(256, 512, dropout=dropout_val)
        self.cds_decoder = FlashMultimodalDecoder(256, vocab_size=67, max_len=4096, dropout=dropout_val)
        self.head_org = nn.Sequential(nn.Dropout(dropout_val), nn.Linear(256, 17))
        self.head_pa = nn.Sequential(nn.Dropout(dropout_val), nn.Linear(256, 6))

    def _apply_lora(self, model, name):
        target_mods = self.config.get('target_modules', ["in_proj", "out_proj", "query", "value", "key", "dense", "fc1", "fc2"])
        return get_peft_model(model, LoraConfig(r=self.config.get('lora_r', 32), lora_alpha=self.config.get('lora_alpha', 64), target_modules=target_mods, lora_dropout=self.config.get('lora_dropout', 0.1), use_rslora=self.config.get('use_rslora', False), task_type=None))

    def _load_hyena_weights(self):
        if os.path.exists(PATH_DNA_CHECKPOINT):
            s = torch.load(PATH_DNA_CHECKPOINT, map_location='cpu', weights_only=False)
            self.hyena.load_state_dict({k.replace("model.backbone.", "backbone.").replace("model.", "").replace("hyena.", ""): v for k,v in s.get('state_dict', s).items()}, strict=False)
            smart_print("✅ Pre-trained Hyena weights loaded successfully.")

    def _safe_hyena(self, seq):
        out = self.hyena(seq).last_hidden_state
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        mask = (seq != 0).unsqueeze(-1).to(out.dtype)
        return out * mask

    def _safe_mamba(self, seq):
        out = self.mamba(seq) if isinstance(self.mamba, nn.Embedding) else self.mamba(seq).last_hidden_state
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        mask = (seq != 0).unsqueeze(-1).to(out.dtype)
        return out * mask

    def forward(self, up, down, introns, aa, cds_target=None):
        device = up.device
        e_up = self.dna_conv(self._safe_hyena(up).transpose(1,2)).transpose(1,2)
        e_down = self.dna_conv(self._safe_hyena(down).transpose(1,2)).transpose(1,2)
        B, N, L = introns.shape
        e_int = self.dna_conv(self._safe_hyena(introns.reshape(-1, L)).transpose(1,2)).transpose(1,2).reshape(B, -1, 256)

        dna_ctx = torch.cat([e_up, e_int, e_down], dim=1)
        resampled_dna = self.resampler(self.norm(dna_ctx)) + self.modal_emb(torch.zeros(1, device=device).long())

        e_aa = self._safe_mamba(aa)
        e_aa = self.prot_conv(self.prot_proj(e_aa).transpose(1,2)).transpose(1,2)
        e_aa = e_aa + self.aa_pos_emb(torch.arange(e_aa.shape[1], device=device).unsqueeze(0).expand(B, -1)) + self.modal_emb(torch.ones(1, device=device).long())

        memory = torch.cat([resampled_dna, self.norm(e_aa)], dim=1)
        return self.head_org(memory.mean(1)), self.head_pa(memory.mean(1)), self.cds_decoder(cds_target, memory)

# ==========================================
# 🚀 5. EVAL & MAIN
# ==========================================
def evaluate(model, loader, device, dtype):
    model.eval(); m = {'loss': 0, 'acc_cds': 0, 'acc_org': 0, 'acc_pa': 0}
    with torch.no_grad():
        for b in loader:
            for k,v in b.items(): b[k] = v.to(device)
            with torch.amp.autocast('cuda', dtype=dtype):
                p_org, p_pa, p_cds = model(b['up'], b['down'], b['introns'], b['aa'], b['cds'][:, :-1])
            m['loss'] += F.cross_entropy(p_cds.reshape(-1, 67).float(), b['cds'][:, 1:].reshape(-1), ignore_index=0).item()
            m['acc_cds'] += (p_cds.argmax(-1) == b['cds'][:, 1:])[b['cds'][:, 1:] != 0].float().mean().item()
            m['acc_org'] += (p_org.argmax(-1) == b['org']).float().mean().item()
            m['acc_pa'] += (p_pa.argmax(-1) == b['pa']).float().mean().item()
    return {k: v/len(loader) for k,v in m.items()}

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path):
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    torch.save({
        'full_model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'step': step,
    }, f"{path}/training_state.pth")
    smart_print(f"💾 Full checkpoint saved at step {step}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    CONFIG = {
        "experiment_id": "h100_final", "lr": 1e-4, "optimizer": "adamw", "weight_decay": 0.01,
        "warmup_steps": 500, "mask_prob": 0.15, "span_length": 1, "dropout": 0.1,
        "l_org_weight":0.1, "l_pa_weight":0.1, "lora_dropout": 0.1, "use_rslora": False,
        "lora_r": 32, "lora_alpha": 64, "batch_size": 1, "grad_accum": 32, "epochs": 5
    }

    if args.config:
        smart_print(f"🔍 Attempting to load config file: {args.config}")
        if os.path.exists(args.config):
            try:
                with open(args.config, 'r') as f:
                    loaded_conf = json.load(f)
                    CONFIG.update(loaded_conf)
                smart_print(f"✅ Successfully loaded config from {args.config}")
            except Exception as e:
                smart_print(f"❌ ERROR: Failed to parse JSON from {args.config}. Error: {e}")
                sys.exit(1)
        else:
            smart_print(f"❌ ERROR: Config file '{args.config}' not found! Please check the path.")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    EXP_ID = CONFIG.get("experiment_id", "default")
    EXPERIMENT_DIR = f"{BASE_TMP_PATH}/runs/{EXP_ID}"
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    with open(f"{EXPERIMENT_DIR}/config_used.json", "w") as f:
        json.dump(CONFIG, f, indent=4)

    train_ds = GenomicDataset(TRAIN_FILE, CONFIG, is_train=True)
    train_sampler = WeightedRandomSampler(weights=train_ds.sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(GenomicDataset(VAL_FILE, CONFIG, is_train=False), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = OptimizedFusionModel(CONFIG).to(device).to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG.get('weight_decay', 0.01))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=CONFIG.get('warmup_steps', 500), num_training_steps=(len(train_loader) // CONFIG['grad_accum']) * CONFIG['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    start_epoch = 0
    global_step = 0; best_val_loss = float('inf'); t0 = time.time()
    org_weights = get_class_weights().to(device)

    # 🔥 טעינה אוטומטית 🔥
    best_ckpt_path = f"{EXPERIMENT_DIR}/best_model_checkpoint"
    latest_ckpt_path = f"{EXPERIMENT_DIR}/latest_checkpoint"
    
    ckpt_to_load = None
    if os.path.exists(f"{best_ckpt_path}/training_state.pth"):
        ckpt_to_load = best_ckpt_path
    elif os.path.exists(f"{latest_ckpt_path}/training_state.pth"):
        ckpt_to_load = latest_ckpt_path

    if ckpt_to_load:
        smart_print(f"🔄 Resuming from full checkpoint: {ckpt_to_load}...")
        try:
            state = torch.load(f"{ckpt_to_load}/training_state.pth", map_location=device)
            if 'full_model_state' in state:
                model.load_state_dict(state['full_model_state'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                scaler.load_state_dict(state['scaler'])
                start_epoch = state['epoch']
                global_step = state['step']
                smart_print(f"✅ Resuming from Epoch {start_epoch}, Step {global_step}")
            else:
                smart_print("❌ ERROR: Old format detected. Please change experiment_id in your JSON for a fresh start.")
                sys.exit(1)
        except Exception as e:
            smart_print(f"❌ ERROR loading checkpoint: {e}")
            sys.exit(1)
    else:
        smart_print("🆕 No checkpoints found. Starting fresh training.")

    with open(f"{EXPERIMENT_DIR}/training_log.csv", 'a' if ckpt_to_load else 'w', newline='') as f:
        writer = csv.writer(f)
        if not ckpt_to_load:
            writer.writerow(['epoch', 'step', 'loss_total', 'loss_cds', 'loss_org', 'loss_pa', 'val_loss', 'val_acc_cds', 'val_acc_org', 'val_acc_pa'])

    smart_print(f"🔥 STARTING TRAINING LOOP...")
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        for i, b in enumerate(train_loader):
            for k,v in b.items(): b[k] = v.to(device)
            with torch.amp.autocast('cuda', dtype=dtype):
                p_org, p_pa, p_cds = model(b['up'], b['down'], b['introns'], b['aa'], b['cds'][:, :-1])

            valid_cds_tokens = (b['cds'][:, 1:] != 0).sum()
            if valid_cds_tokens > 0:
                l_cds = F.cross_entropy(p_cds.reshape(-1, 67).float(), b['cds'][:, 1:].reshape(-1), ignore_index=0)
            else:
                l_cds = torch.tensor(0.0, device=device, requires_grad=True)

            l_org = F.cross_entropy(p_org.float(), b['org'], weight=org_weights)
            l_pa = F.cross_entropy(p_pa.float(), b['pa'])
            loss = (l_cds + CONFIG.get('l_org_weight', 0.1) * l_org + CONFIG.get('l_pa_weight', 0.1) * l_pa) / CONFIG['grad_accum']

            if torch.isnan(loss) or torch.isinf(loss): continue

            if dtype == torch.float16: scaler.scale(loss).backward()
            else: loss.backward()

            if (i + 1) % CONFIG['grad_accum'] == 0:
                has_nans = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in model.parameters())
                if not has_nans:
                    if dtype == torch.float16:
                        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                optimizer.zero_grad(); scheduler.step(); global_step += 1

                if global_step % 5 == 0:
                    smart_print(f"Step {global_step} | Total: {loss.item()*CONFIG['grad_accum']:.4f} | CDS: {l_cds.item():.4f}")
                    with open(f"{EXPERIMENT_DIR}/training_log.csv", 'a', newline='') as f: csv.writer(f).writerow([epoch+1, global_step, loss.item()*CONFIG['grad_accum'], l_cds.item(), l_org.item(), l_pa.item(), None, None, None, None])
                if global_step % 150 == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, f"{EXPERIMENT_DIR}/latest_checkpoint")

        vr = evaluate(model, val_loader, device, dtype)
        smart_print(f"📊 Epoch {epoch+1} Val: Loss {vr['loss']:.4f}")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch+1, global_step, f"{EXPERIMENT_DIR}/latest_checkpoint")
        if vr['loss'] < best_val_loss:
            best_val_loss = vr['loss']
            save_checkpoint(model, optimizer, scheduler, scaler, epoch+1, global_step, f"{EXPERIMENT_DIR}/best_model_checkpoint")

if __name__ == "__main__": main()
