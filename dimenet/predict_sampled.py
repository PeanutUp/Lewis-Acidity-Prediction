#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import yaml
import sys
import json
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider

def predict_sampled():
    # Force CPU to avoid Metal GPU OOM on macOS
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    # Paths (Run from dimenet directory or adjust)
    config_path = "config_fia49k.yaml"
    
    # Check if running from root or subdir
    if os.path.exists("data/sampled_2000.npz"):
        # Running from dimenet/
        dataset_path = "data/sampled_2000.npz"
        best_ckpt_path = "None/best/ckpt"
        saved_config_path = "None/config.yaml"
        output_xlsx = "dimenet_sampled_predictions.xlsx"
        output_csv = "dimenet_sampled_predictions.csv"
    else:
        # Running from root probably
        dataset_path = "dimenet/data/sampled_2000.npz"
        config_path = "dimenet/config_fia49k.yaml"
        best_ckpt_path = "dimenet/None/best/ckpt"
        saved_config_path = "dimenet/None/config.yaml"
        output_xlsx = "dimenet_sampled_predictions.xlsx"
        output_csv = "dimenet_sampled_predictions.csv"

    dataset_path = os.path.abspath(dataset_path)
    best_ckpt_path = os.path.abspath(best_ckpt_path)
    output_xlsx = os.path.abspath(output_xlsx)
    output_csv = os.path.abspath(output_csv)

    if os.path.exists(saved_config_path):
        print(f"Loading config from {saved_config_path}...")
        with open(saved_config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Load NPZ to get indices and IDs
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
    raw_data = np.load(dataset_path, allow_pickle=True)
    # Check keys
    if 'id' in raw_data:
        mol_ids = raw_data['id']
    else:
        print("Warning: 'id' key not found in NPZ. Using indices as IDs.")
        mol_ids = None

    # If id looks like numeric indices, try to load mol_id from sampled_metadata.csv
    if mol_ids is None or (np.issubdtype(np.asarray(mol_ids).dtype, np.number)):
        metadata_candidates = [
            os.path.abspath("sampled_metadata.csv"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sampled_metadata.csv"))
        ]
        metadata_path = next((p for p in metadata_candidates if os.path.exists(p)), None)
        if metadata_path is not None:
            try:
                meta_df = pd.read_csv(metadata_path)
                if "mol_id" in meta_df.columns and len(meta_df) == len(raw_data['N']):
                    mol_ids = meta_df["mol_id"].astype(str).values
                    print(f"Loaded mol_id from {metadata_path}")
                else:
                    print(f"Warning: {metadata_path} missing mol_id column or length mismatch")
            except Exception as e:
                print(f"Warning: Failed to read {metadata_path}: {e}")
        
    num_samples = len(raw_data['N'])
    sampled_indices = np.arange(num_samples)
    print(f"Loaded {num_samples} samples.")

    # Model Parameters
    model_name = config['model_name']
    emb_size = config['emb_size']
    out_emb_size = config.get('out_emb_size', None)
    int_emb_size = config.get('int_emb_size', None)
    basis_emb_size = config.get('basis_emb_size', None)
    num_blocks = config['num_blocks']
    num_bilinear = config.get('num_bilinear', None)
    num_spherical = config['num_spherical']
    num_radial = config['num_radial']
    output_init = config['output_init']
    extensive = config.get('extensive', False)
    num_before_skip = config['num_before_skip']
    num_after_skip = config['num_after_skip']
    num_dense_output = config['num_dense_output']
    cutoff = config['cutoff']
    envelope_exponent = config['envelope_exponent']
    targets = config['targets']
    
    # Load DataContainer
    data_container = DataContainer(dataset_path, cutoff=cutoff, target_keys=targets)
    
    # Use smaller batch size for stable inference
    pred_batch_size = min(16, config['batch_size'])

    # Initialize DataProvider
    data_provider = DataProvider(data_container, 1, 1, pred_batch_size, seed=42, randomized=False)
    
    # OVERWRITE TEST INDICES to be all indices
    data_provider.idx['test'] = sampled_indices
    data_provider.nsamples['test'] = len(sampled_indices)
    
    # Initialize Model
    print("Initializing model...")
    if model_name == "dimenet":
        model = DimeNet(
                emb_size=emb_size, num_blocks=num_blocks, num_bilinear=num_bilinear,
                num_spherical=num_spherical, num_radial=num_radial,
                cutoff=cutoff, envelope_exponent=envelope_exponent,
                num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                num_dense_output=num_dense_output, num_targets=len(targets),
                activation=swish, output_init=output_init)
    elif model_name == "dimenet++":
        model = DimeNetPP(
                emb_size=emb_size, out_emb_size=out_emb_size,
                int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
                num_blocks=num_blocks, num_spherical=num_spherical, num_radial=num_radial,
                cutoff=cutoff, envelope_exponent=envelope_exponent,
                num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                num_dense_output=num_dense_output, num_targets=len(targets),
                activation=swish, extensive=extensive, output_init=output_init)

    # Build model (forward pass)
    dummy_idx = sampled_indices[:min(len(sampled_indices), pred_batch_size)]
    dummy_inputs, _ = data_provider.idx_to_data(dummy_idx)
    _ = model(dummy_inputs, training=False)

    # Load Weights
    print(f"Loading weights from {best_ckpt_path}")
    try:
        # Prefer Keras load_weights when supported (TF checkpoint format)
        model.load_weights(best_ckpt_path)
        print("Weights loaded via model.load_weights().")
    except Exception as e:
        print(f"model.load_weights failed, falling back to Checkpoint restore: {e}")
        try:
            ckpt = tf.train.Checkpoint(model=model)
            status = ckpt.restore(best_ckpt_path)
            status.expect_partial()
            print("Checkpoint restore done (partial allowed).")

            # Manually assign weights that cannot be name-matched due to add_weight naming
            reader = tf.train.load_checkpoint(best_ckpt_path)
            ckpt_vars = reader.get_variable_to_shape_map()

            emb_key = "emb_block/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
            if emb_key in ckpt_vars:
                model.emb_block.embeddings.assign(reader.get_tensor(emb_key))

            freq_key = "rbf_layer/frequencies/.ATTRIBUTES/VARIABLE_VALUE"
            if freq_key in ckpt_vars:
                model.rbf_layer.frequencies.assign(reader.get_tensor(freq_key))
        except Exception as e2:
            print(f"Failed to load weights: {e2}")
            return

    # Predict
    print("Running predictions...")
    all_preds = []
    all_targets = []
    all_indices = []
    
    batch_size = pred_batch_size
    num_test = len(sampled_indices)
    steps = int(np.ceil(num_test / batch_size))
    
    data_provider.idx_in_epoch['test'] = 0
    
    for i in range(steps):
        batch_idx = data_provider.get_batch_idx('test')
        inputs, target_batch = data_provider.idx_to_data(batch_idx)
        preds = model(inputs, training=False)
        
        all_preds.append(preds.numpy())
        all_targets.append(target_batch.numpy())
        all_indices.append(batch_idx)
        
        if i % 10 == 0:
            print(f"Batch {i}/{steps}")

    all_preds = np.concatenate(all_preds, axis=0)[:num_test]
    all_targets = np.concatenate(all_targets, axis=0)[:num_test]
    all_indices = np.concatenate(all_indices, axis=0)[:num_test]
    
    y_true = all_targets[:, 0].flatten()
    y_pred = all_preds[:, 0].flatten()
    abs_error = np.abs(y_pred - y_true)
    
    results = []
    for i in range(len(all_indices)):
        idx = int(all_indices[i])
        if mol_ids is not None:
            if len(mol_ids) == len(all_indices):
                mid = str(mol_ids[i])
            else:
                mid = str(mol_ids[idx])
        else:
            mid = str(idx)
        
        results.append({
            "mol_id": mid,
            "predicted_value": float(y_pred[i]),
            "true_value": float(y_true[i]),
            "absolute_error": float(abs_error[i])
        })
        
    print(f"Saving {len(results)} results...")
    try:
        df = pd.DataFrame(results)
        df.to_excel(output_xlsx, index=False)
        print(f"Saved Excel to {output_xlsx}")
    except Exception as e:
        print(f"Excel save failed: {e}")

    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV to {output_csv}")
    except Exception as e:
        print(f"CSV save failed: {e}")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.get_logger().setLevel('WARN')
    predict_sampled()
