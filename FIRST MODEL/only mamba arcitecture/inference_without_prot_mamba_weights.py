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

# Importing custom architecture and dataset classes
from train_55_reproduction_multiple_heads import OptimizedFusionModel, GenomicDataset, smart_print, ORG_TO_ID, CodonTokenizer

# ==========================================
#  SETUP & PATHS
# ==========================================
BASE_TMP_PATH = BASE_TMP_PATH = os.environ.get("BASE_TMP_PATH", "/scratch200/tallilo/deep_learning_project")
TEST_FILE = f"{BASE_TMP_PATH}/data/TEST_HOMOLOGY_SPLIT.csv"

# Make sure this points to your specific "Multi-Head per Organism" experiment
EXPERIMENT_ID = "0_high_lr_R_Task_Specific_Heads" 
CHECKPOINT_PATH = f"{BASE_TMP_PATH}/runs/{EXPERIMENT_ID}/best_model_checkpoint/training_state.pth"
CONFIG_PATH = f"{BASE_TMP_PATH}/runs/{EXPERIMENT_ID}/config_used.json"

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    smart_print(f"🚀 Starting Inference for TASK-SPECIFIC HEADS architecture: {EXPERIMENT_ID}")

    ID_TO_ORG = {v: k for k, v in ORG_TO_ID.items()}
    codon_tok = CodonTokenizer()
    ID_TO_CODON = {v: k for k, v in codon_tok.vocab.items()}
    ID_TO_CODON[0] = "" # Ignore PAD
    ID_TO_CODON[66] = "" # Ignore EOS
    ID_TO_CODON[65] = "<UNK>"

    model = OptimizedFusionModel(config).to(device).to(dtype)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state['full_model_state'], strict=False)
    
    test_ds = GenomicDataset(TEST_FILE, config, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=config.get('batch_size', 1), shuffle=False, num_workers=4)

    model.eval()
    
    # ---------------------------------------------------------
    #  METRICS TRACKING (Global + Per-Head)
    # ---------------------------------------------------------
    metrics = {'correct_org': 0, 'correct_pa': 0, 'correct_cds': 0, 'total_cds_tokens': 0, 'total_samples': 0}
    
    # acuracy for each Species head 
    per_head_metrics = {
        org_id: {'correct_cds': 0, 'total_cds_tokens': 0, 'samples': 0} 
        for org_id in ID_TO_ORG.keys()
    }
    
    csv_data = []

    # ---------------------------------------------------------
    #  HOOK 1: CAPTURE LATENT EMBEDDINGS (Shared Latent Space)
    # ---------------------------------------------------------
    saved_embeddings = []
    
    def capture_embeddings_hook(module, args):
        emb_array = args[0].detach().cpu().to(torch.float32).numpy()
        saved_embeddings.append(emb_array)
    
    # Assumption: head_org still reads from the shared fusion memory
    emb_hook_handle = model.head_org.register_forward_pre_hook(capture_embeddings_hook)

    # ---------------------------------------------------------
    #  HOOK 2: CAPTURE CROSS-ATTENTION (From Specific Heads)
    # ---------------------------------------------------------
    saved_attentions = []
    MAX_ATTN_SAMPLES = 100 
    attn_hook_handles = []
    
    def capture_attention_hook(module_name):
        def hook(module, inputs, output):
            if len(saved_attentions) < MAX_ATTN_SAMPLES:
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1] 
                    if attn_weights is not None:
                        attn_array = attn_weights.detach().cpu().to(torch.float16).numpy()
                        # Also save from which head (module_name) the attention map originated
                        saved_attentions.append({'head_name': module_name, 'weights': attn_array})
        return hook

    # Register a hook for all Attention layers in the model (covering all 17 heads)
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            handle = module.register_forward_hook(capture_attention_hook(name))
            attn_hook_handles.append(handle)
            
    smart_print(f"✅ Registered hooks on {len(attn_hook_handles)} Attention layers (for Task-Specific Heads).")

    # ---------------------------------------------------------
    #  MAIN INFERENCE LOOP
    # ---------------------------------------------------------
    with torch.no_grad():
        for b in tqdm(test_loader, file=sys.stdout):
            for k, v in b.items(): 
                b[k] = v.to(device)
            
            with torch.amp.autocast('cuda', dtype=dtype):
                MAX_SAFE_LEN = 2000
                safe_cds = b['cds']
                if safe_cds.shape[1] > MAX_SAFE_LEN:
                    safe_cds = safe_cds[:, :MAX_SAFE_LEN]

                # Run the model. 
                # Assumption: The forward function uses b['org'] to route the information to the appropriate head 
                # and returns a single p_cds tensor containing the predictions of the activated heads.
                p_org, p_pa, p_cds = model(b['up'], b['down'], b['introns'], b['aa'], b['org'], safe_cds[:, :-1])

                # Global metrics
                metrics['correct_org'] += (p_org.argmax(-1) == b['org']).sum().item()
                metrics['correct_pa'] += (p_pa.argmax(-1) == b['pa']).sum().item()
                metrics['total_samples'] += b['org'].size(0)

                # Loop over each sample in the batch to associate the results with the specific head that was active
                for idx in range(b['org'].size(0)):
                    current_org_id = b['org'][idx].item()
                    species_name = ID_TO_ORG.get(current_org_id, "Unknown")
                    
                    gt_tokens = safe_cds[idx, 1:]
                    pred_tokens = p_cds[idx].argmax(-1)
                    
                    # Calculate codon prediction accuracy for this specific sample
                    mask = (gt_tokens != 0) & (gt_tokens != 66)
                    correct_in_sample = ((pred_tokens == gt_tokens) & mask).sum().item()
                    total_in_sample = mask.sum().item()
                    
                    # Update global metrics
                    metrics['correct_cds'] += correct_in_sample
                    metrics['total_cds_tokens'] += total_in_sample
                    
                    # Update metrics for the specific head (the magic of the new architecture)
                    per_head_metrics[current_org_id]['correct_cds'] += correct_in_sample
                    per_head_metrics[current_org_id]['total_cds_tokens'] += total_in_sample
                    per_head_metrics[current_org_id]['samples'] += 1

                    # --- Preparation for CSV ---
                    gt_seq_list = [ID_TO_CODON.get(t, "") for t in gt_tokens.tolist() if t != 0 and t != 66]
                    pred_seq_list = [ID_TO_CODON.get(t, "") for t in pred_tokens.tolist()[:len(gt_seq_list)]]
                    
                    gt_seq_list = [c for c in gt_seq_list if c]
                    pred_seq_list = [c for c in pred_seq_list if c]
                    seq_length = len(gt_seq_list)
                    
                    acc_pct = (correct_in_sample / total_in_sample * 100) if total_in_sample > 0 else 0.0

                    sample_info = f"Species: {species_name} | Target Head: {current_org_id} | PA: {b['pa'][idx].item()} | Len: {seq_length}"

                    csv_data.append({
                        'Sample_Info': sample_info,
                        'Active_Decoder_Head': species_name,
                        'Ground_Truth_CDS': " ".join(gt_seq_list),
                        'Predicted_CDS': " ".join(pred_seq_list),
                        'Head_Specific_Accuracy_%': f"{acc_pct:.2f}%"
                    })

    # ---------------------------------------------------------
    #  CLEANUP AND SAVE ARTIFACTS
    # ---------------------------------------------------------
    emb_hook_handle.remove()
    for handle in attn_hook_handles:
        handle.remove()

    final_embeddings = np.concatenate(saved_embeddings, axis=0)
    emb_save_path = f"{BASE_TMP_PATH}/runs/{EXPERIMENT_ID}/latent_embeddings.npy"
    np.save(emb_save_path, final_embeddings)
    
    attn_save_path = ""
    if len(saved_attentions) > 0:
        # Save the dictionary (including the module/head name)
        attn_save_path = f"{BASE_TMP_PATH}/runs/{EXPERIMENT_ID}/task_specific_attention_maps.npy"
        np.save(attn_save_path, saved_attentions, allow_pickle=True)

    df_results = pd.DataFrame(csv_data)
    csv_save_path = f"{BASE_TMP_PATH}/runs/{EXPERIMENT_ID}/task_specific_predictions.csv"
    df_results.to_csv(csv_save_path, index=False, encoding='utf-8')

    # ---------------------------------------------------------
    #  PRINT RESULTS & PER-HEAD ANALYSIS
    # ---------------------------------------------------------
    smart_print("\n" + "="*80)
    smart_print(f"✅ SAVED {final_embeddings.shape[0]} EMBEDDINGS TO: {emb_save_path}")
    if attn_save_path:
        smart_print(f"✅ SAVED {len(saved_attentions)} HEAD-SPECIFIC ATTENTION MAPS TO: {attn_save_path}")
    smart_print(f"✅ SAVED CSV PREDICTIONS TO: {csv_save_path}")
    smart_print("="*80)
    
    smart_print(f"\n🌍 GLOBAL METRICS:")
    smart_print(f"➤ Org Classify Acc: {metrics['correct_org']/metrics['total_samples']:.2%}")
    smart_print(f"➤ PA Predict Acc:   {metrics['correct_pa']/metrics['total_samples']:.2%}")
    smart_print(f"➤ Global CDS Acc:   {metrics['correct_cds']/metrics['total_cds_tokens']:.2%}")
    
    smart_print(f"\n🧠 TASK-SPECIFIC HEADS PERFORMANCE:")
    smart_print(f"{'Organism (Head)':<35} | {'Samples':<10} | {'Accuracy'}")
    smart_print("-" * 65)
    
    for org_id, data in per_head_metrics.items():
        if data['samples'] > 0:
            head_acc = data['correct_cds'] / data['total_cds_tokens']
            org_name = ID_TO_ORG.get(org_id, f"ID_{org_id}")
            # Special highlighting for yeast and challenging bacteria
            marker = "⭐" if "cerevisiae" in org_name or "Streptococcus" in org_name else "  "
            smart_print(f"{marker} {org_name:<32} | {data['samples']:<10} | {head_acc:.2%}")

if __name__ == "__main__":
    run_inference()