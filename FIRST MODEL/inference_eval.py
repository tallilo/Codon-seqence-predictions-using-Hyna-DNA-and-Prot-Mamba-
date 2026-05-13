import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import json
import sys
from tqdm import tqdm

# Importing custom architecture and dataset classes from the main training script
from train_55_reproduction import OptimizedFusionModel, GenomicDataset, smart_print, ORG_TO_ID, CodonTokenizer

# ==========================================
# 🛠️ SETUP & PATHS
# ==========================================
# Define base paths for the cluster environment
BASE_TMP_PATH = "/scratch200/tallilo/deep_learning_project"
TEST_FILE = f"{BASE_TMP_PATH}/data/TEST_HOMOLOGY_SPLIT.csv"

# Ensure this matches the exact experiment ID you want to evaluate
EXPERIMENT_ID = "0_high_lr_R" 
CHECKPOINT_PATH = f"{BASE_TMP_PATH}/runs/{EXPERIMENT_ID}/best_model_checkpoint/training_state.pth"
CONFIG_PATH = f"{BASE_TMP_PATH}/runs/{EXPERIMENT_ID}/config_used.json"

def run_inference():
    # Setup device (GPU if available) and dtype for mixed precision inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load the exact configuration used during training to ensure architectural match
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    smart_print(f"🚀 Starting Randomized Inference for Experiment: {EXPERIMENT_ID}")

    # Reverse the dictionaries to map predicted integer IDs back to readable strings
    ID_TO_ORG = {v: k for k, v in ORG_TO_ID.items()}
    codon_tok = CodonTokenizer()
    ID_TO_CODON = {v: k for k, v in codon_tok.vocab.items()}
    
    # Explicitly map special tokens for formatting
    ID_TO_CODON[0] = "<PAD>"
    ID_TO_CODON[66] = "<EOS>"

    # Initialize the model and load the pre-trained weights from the checkpoint
    model = OptimizedFusionModel(config).to(device).to(dtype)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state['full_model_state'])
    
    # Initialize the test dataset and dataloader
    test_ds = GenomicDataset(TEST_FILE, config, is_train=False)
    # Shuffle is set to True here so we get a random sample of 20 examples to display every time we run
    test_loader = DataLoader(test_ds, batch_size=config.get('batch_size', 1), shuffle=True, num_workers=4)

    # Set model to evaluation mode (disables dropout, fixes batchnorm statistics)
    model.eval()
    
    # Dictionary to accumulate global metrics across the entire test set
    metrics = {'correct_org': 0, 'correct_pa': 0, 'correct_cds': 0, 'total_cds_tokens': 0, 'total_samples': 0}
    
    # Variables to store a visual sample of predictions
    examples_to_show = 20 
    saved_examples = []

    # Disable gradient tracking to save VRAM and speed up inference
    with torch.no_grad():
        for b in tqdm(test_loader, file=sys.stdout):
            # Move all batch tensors to the GPU
            for k, v in b.items(): 
                b[k] = v.to(device)
            
            # Use Automatic Mixed Precision (AMP) for faster forward passes
            with torch.amp.autocast('cuda', dtype=dtype):
                # Forward pass: predict Organism, Protein Abundance, and Coding Sequence (CDS)
                p_org, p_pa, p_cds = model(b['up'], b['down'], b['introns'], b['aa'], b['cds'][:, :-1])

                # --- Extract Visual Examples ---
                # Only capture examples until we hit our limit of 20
                if len(saved_examples) < examples_to_show:
                    for idx in range(min(b['org'].size(0), examples_to_show - len(saved_examples))):
                        # Extract the first 20 ground truth and predicted tokens
                        gt_tokens = b['cds'][idx, 1:21].tolist()
                        pred_tokens = p_cds[idx].argmax(-1)[:20].tolist()
                        
                        # Store the decoded strings and classification results for pretty printing later
                        saved_examples.append({
                            'gt_org': ID_TO_ORG.get(b['org'][idx].item(), "Unknown"),
                            'pred_org': ID_TO_ORG.get(p_org[idx].argmax(-1).item(), "Unknown"),
                            'gt_pa': b['pa'][idx].item(),
                            'pred_pa': p_pa[idx].argmax(-1).item(),
                            'gt_cds': " ".join([ID_TO_CODON.get(t, "???") for t in gt_tokens if t != 0]),
                            # Safely extract predictions, ensuring we don't index out of bounds
                            'pred_cds': " ".join([ID_TO_CODON.get(t, "???") for t in pred_tokens[:len(gt_tokens)] if gt_tokens[gt_tokens.index(gt_tokens[0]) + gt_tokens.index(gt_tokens[-1])] !=0]) 
                        })

                # --- Accumulate Global Metrics ---
                # Calculate correct organism classifications
                metrics['correct_org'] += (p_org.argmax(-1) == b['org']).sum().item()
                # Calculate correct protein abundance classifications
                metrics['correct_pa'] += (p_pa.argmax(-1) == b['pa']).sum().item()
                metrics['total_samples'] += b['org'].size(0)
                
                # Calculate correct CDS codon predictions (ignoring padding tokens via the mask)
                mask = (b['cds'][:, 1:] != 0)
                metrics['correct_cds'] += ((p_cds.argmax(-1) == b['cds'][:, 1:]) & mask).sum().item()
                metrics['total_cds_tokens'] += mask.sum().item()

    # ==========================================
    # 📊 RESULTS OUTPUT
    # ==========================================
    print("\n" + "="*80, flush=True)
    print(f"🧬 SHUFFLED PREDICTIONS FROM TEST SET (20 Samples)", flush=True)
    print("="*80, flush=True)
    
    # Print the 20 stored examples with clear Success/Fail markers
    for i, ex in enumerate(saved_examples):
        org_res = "✅" if ex['gt_org'] == ex['pred_org'] else f"❌ (Is: {ex['pred_org']})"
        pa_res = "✅" if ex['gt_pa'] == ex['pred_pa'] else f"❌ (Is: {ex['pred_pa']})"
        
        print(f"[{i+1:02d}] Org: {ex['gt_org']:<22} {org_res}")
        print(f"     PA:  Level {ex['gt_pa']} {' '*15} {pa_res}")
        print(f"     GT:  {ex['gt_cds']}")
        print(f"     PR:  {ex['pred_cds']}")
        print("-" * 60, flush=True)

    # Print the final aggregated accuracies for the entire test dataset
    print(f"\n📈 OVERALL: Org Acc: {metrics['correct_org']/metrics['total_samples']:.2%}, PA Acc: {metrics['correct_pa']/metrics['total_samples']:.2%}, CDS Acc: {metrics['correct_cds']/metrics['total_cds_tokens']:.2%}", flush=True)

if __name__ == "__main__":
    run_inference()