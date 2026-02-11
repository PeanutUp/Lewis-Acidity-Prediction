#!/usr/bin/env python3
"""
Predict FIA49k test set with best EGNN weights and export JSON.
Outputs sorted results with true/pred/error plus recovered indices.
"""

import argparse
import json
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from fia49k_new_dataset import create_fia49k_new_dataloaders, FIA49kNewDataset, collate_fn
from qm9.models import EGNN
from qm9 import utils as qm9_utils


def load_csv_index_map(csv_path: str):
    if not csv_path or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "Compound" not in df.columns:
        return {}
    # Map Compound -> row index (0-based)
    return {str(row["Compound"]): int(idx) for idx, row in df.iterrows()}


def compute_train_stats(train_loader):
    targets = []
    for batch in train_loader:
        if "targets" in batch:
            targets.extend(batch["targets"].tolist())
        elif "target" in batch:
            targets.extend(batch["target"].tolist())
    targets = torch.tensor(targets, dtype=torch.float32)
    meann = torch.mean(targets)
    mad = torch.mean(torch.abs(targets - meann))
    return meann, mad


def main():
    parser = argparse.ArgumentParser(description="Predict FIA49k test set and export JSON")
    parser.add_argument("--data_path", type=str, default="/Users/apple/Desktop/大学/fall 2025/有机化学A(2)/教学论文/训练代码/egnn/fia49k_full_converted/fia49k_data.pt")
    parser.add_argument("--model_path", type=str, default="/Users/apple/Desktop/大学/fall 2025/有机化学A(2)/教学论文/训练代码/egnn/fia49k_new_logs/fia49k_new_exp/best_model.pt")
    parser.add_argument("--csv_path", type=str, default="/Users/apple/Desktop/大学/fall 2025/有机化学A(2)/教学论文/训练代码/egnn/ae_datasets/FIA49k.csv")
    parser.add_argument("--output_json", type=str, default="/Users/apple/Desktop/大学/fall 2025/有机化学A(2)/教学论文/训练代码/egnn/fia49k_new_logs/fia49k_new_exp/predictions_sampled.json")
    parser.add_argument("--output_xlsx", type=str, default="/Users/apple/Desktop/大学/fall 2025/有机化学A(2)/教学论文/训练代码/egnn_sampled_predictions.xlsx")
    parser.add_argument("--sampled_indices_json", type=str, default="/Users/apple/Desktop/大学/fall 2025/有机化学A(2)/教学论文/训练代码/sampled_indices.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--charge_power", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    args.data_path = os.path.abspath(args.data_path)
    args.model_path = os.path.abspath(args.model_path)
    args.csv_path = os.path.abspath(args.csv_path)
    args.output_json = os.path.abspath(args.output_json)
    args.output_xlsx = os.path.abspath(args.output_xlsx)
    args.sampled_indices_json = os.path.abspath(args.sampled_indices_json)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (for train stats only)
    train_loader, val_loader, test_loader = create_fia49k_new_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # Load sampled indices
    if not os.path.exists(args.sampled_indices_json):
        raise FileNotFoundError(f"Sampled indices file not found: {args.sampled_indices_json}")
    with open(args.sampled_indices_json, "r") as f:
        sampled_indices = json.load(f)
    if not isinstance(sampled_indices, list) or len(sampled_indices) == 0:
        raise ValueError("Sampled indices JSON is empty or invalid.")

    # Build sampled dataloader (no shuffle)
    sampled_dataset = FIA49kNewDataset(
        args.data_path,
        split="train",
        train_ratio=1.0,
        val_ratio=0.0,
    )
    sampled_dataset.indices = torch.tensor(sampled_indices, dtype=torch.long)
    sampled_loader = DataLoader(
        sampled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Compute normalization stats from train set (same as training script)
    meann, mad = compute_train_stats(train_loader)

    # Build model
    n_atom_types = len(train_loader.dataset.data["unique_atoms"])
    in_node_nf = n_atom_types * (args.charge_power + 1)
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

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model.eval()

    # Prepare csv index mapping
    csv_index_map = load_csv_index_map(args.csv_path)

    results = []
    dataset_index = 0
    charge_scale = 1.0

    with torch.no_grad():
        for batch in sampled_loader:
            batch_size, n_nodes, _ = batch["positions"].size()
            atom_positions = batch["positions"].view(batch_size * n_nodes, -1).to(device)
            atom_mask = batch["atom_mask"].view(batch_size * n_nodes, -1).to(device)
            one_hot = batch["one_hot"].to(device)
            charges = batch["atomic_numbers"].to(device)

            nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)
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
                rel_err = float(abs_err / abs(true_val) * 100) if true_val != 0 else 0.0

                mol_id = str(batch["mol_ids"][i])
                csv_row_index = csv_index_map.get(mol_id)
                actual_index = int(sampled_indices[dataset_index])

                results.append(
                    {
                        "dataset_index": int(dataset_index),
                        "actual_index": actual_index,
                        "csv_row_index": None if csv_row_index is None else int(csv_row_index),
                        "mol_id": mol_id,
                        "true_value": true_val,
                        "predicted_value": pred_val,
                        "absolute_error": abs_err,
                        "relative_error": rel_err,
                    }
                )
                dataset_index += 1

    # Sort by absolute error ascending
    results_sorted = sorted(results, key=lambda x: x["absolute_error"])

    output = {
        "count": len(results_sorted),
        "sorted_by": "absolute_error_asc",
        "data_path": args.data_path,
        "model_path": args.model_path,
        "csv_path": args.csv_path if os.path.exists(args.csv_path) else None,
        "sampled_indices_json": args.sampled_indices_json,
        "results": results_sorted,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    os.makedirs(os.path.dirname(args.output_xlsx), exist_ok=True)
    df = pd.DataFrame(results_sorted)
    df.to_excel(args.output_xlsx, index=False)

    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.output_xlsx}")


if __name__ == "__main__":
    main()
