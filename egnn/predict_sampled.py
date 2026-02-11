
#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np
import torch
import pandas as pd
from fia49k_new_dataset import FIA49kNewDataset, collate_fn
from qm9.models import EGNN
from qm9 import utils as qm9_utils
from torch.utils.data import DataLoader

def main():
    # Hardcoded paths
    # Assuming running from egnn directory or adjusting paths
    if os.path.exists("sampled_indices.json"):
        indices_path = "sampled_indices.json" # Run from root
        data_path = "egnn/fia49k_full_converted/fia49k_data.pt"
        model_path = "egnn/fia49k_new_logs/fia49k_new_exp/best_model.pt"
        output_xlsx = "egnn_sampled_predictions.xlsx"
        output_csv = "egnn_sampled_predictions.csv"
    else:
        # Run from egnn directory
        indices_path = "../sampled_indices.json" 
        data_path = "fia49k_full_converted/fia49k_data.pt"
        model_path = "fia49k_new_logs/fia49k_new_exp/best_model.pt"
        output_xlsx = "egnn_sampled_predictions.xlsx"
        output_csv = "egnn_sampled_predictions.csv"
    
    print(f"Using Model Weights from: {model_path}")

    torch.manual_seed(1)
    np.random.seed(1)

    
    print(f"Loading indices from {indices_path}...")
    with open(indices_path, 'r') as f:
        sampled_indices = json.load(f)
    print(f"Loaded {len(sampled_indices)} indices.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    print(f"Loading dataset from {data_path}...")
    # Initialize with 'test' split, but we will overwrite indices immediately
    dataset = FIA49kNewDataset(data_path, split='test')
    
    # OVERWRITE INDICES with our sampled set
    # Ensure indices are long tensor or list of ints
    dataset.indices = torch.tensor(sampled_indices, dtype=torch.long)
    print(f"Dataset indices replaced with {len(dataset)} sampled indices.")

    # Create DataLoader
    test_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    # Compute normalization stats (Need to load from training stats or compute from full dataset?)
    # The original script computes it from train_loader. 
    # To be consistent, we should ideally use the exact same stats used during training.
    # However, create_fia49k_new_dataloaders shuffles. 
    # Let's assume the 'mad' and 'mean' are stable enough or we can re-compute them from a "dummy" train loader
    # created with standard params.
    print("Computing normalization stats from default train split...")
    # Create a temporary dataset for stats
    train_ds = FIA49kNewDataset(data_path, split='train', train_ratio=0.8, val_ratio=0.1)
    train_loader_dummy = DataLoader(train_ds, batch_size=1024, collate_fn=collate_fn, shuffle=True)
    
    # Quick compute
    targets = []
    # Just take a subset to be fast if dataset is huge, but here 49k is manageable
    # Better to be accurate:
    all_targets = train_ds.data['targets'][train_ds.indices]
    meann = torch.mean(all_targets)
    mad = torch.mean(torch.abs(all_targets - meann))
    print(f"Mean: {meann}, MAD: {mad}")

    # Build Model
    n_atom_types = len(dataset.data["unique_atoms"])
    in_node_nf = n_atom_types * (2 + 1) # charge_power=2 default
    
    model = EGNN(
        in_node_nf=in_node_nf,
        in_edge_nf=0,
        hidden_nf=128,
        device=device,
        n_layers=7,
        coords_weight=1.0,
        attention=1,
        node_attr=0,
    )

    print(f"Loading weights from {model_path}...")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.eval()

    results = []
    charge_scale = 1.0
    charge_power = 2

    print("Starting prediction...")
    with torch.no_grad():
        for batch in test_loader:
            batch_size, n_nodes, _ = batch["positions"].size()
            atom_positions = batch["positions"].view(batch_size * n_nodes, -1).to(device)
            atom_mask = batch["atom_mask"].view(batch_size * n_nodes, -1).to(device)
            one_hot = batch["one_hot"].to(device)
            charges = batch["atomic_numbers"].to(device)

            nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, device)
            nodes = nodes.view(batch_size * n_nodes, -1)

            edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
            edge_mask = torch.ones(edges[0].shape[0], device=device, dtype=torch.float32).unsqueeze(1)
            label = batch["targets"].to(device)

            pred = model(
                h0=nodes,
                x=atom_positions,
                edges=edges,
                edge_attr=None,
                node_mask=atom_mask,
                edge_mask=edge_mask,
                n_nodes=n_nodes,
            )

            pred_denorm = mad * pred + meann

            for i in range(batch_size):
                true_val = float(label[i].item())
                pred_val = float(pred_denorm[i].item())
                abs_err = float(abs(pred_val - true_val))
                mol_id = str(batch["mol_ids"][i])

                results.append({
                    "mol_id": mol_id,
                    "predicted_value": pred_val,
                    "true_value": true_val,
                    "absolute_error": abs_err
                })

    # Save Results
    print(f"Saving {len(results)} results...")
    
    # Try Excel
    try:
        df = pd.DataFrame(results)
        df.to_excel(output_xlsx, index=False)
        print(f"Saved Excel to {output_xlsx}")
    except Exception as e:
        print(f"Could not save Excel: {e}")
        
    # Always Save CSV as backup
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV to {output_csv}")
    except Exception as e:
        print(f"Could not save CSV: {e}")

if __name__ == "__main__":
    main()
