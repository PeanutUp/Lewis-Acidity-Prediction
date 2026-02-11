#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import yaml
import sys
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider

def predict_on_egnn_testset():
    # 1. 加载 EGNN 测试集信息
    egnn_results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'egnn/fia49k_new_logs/fia49k_new_exp/predictions_test_sorted.xlsx')
    
    if not os.path.exists(egnn_results_path):
        print(f"Error: EGNN results file not found at {egnn_results_path}")
        return

    print(f"Loading EGNN test set IDs from: {egnn_results_path}")
    egnn_df = pd.read_excel(egnn_results_path)
    target_mol_ids = set(egnn_df['mol_id'].values)
    print(f"Found {len(target_mol_ids)} molecules in EGNN test set")

    # 2. 查找模型和配置
    # 尝试查找当前目录下的 None 文件夹作为模型目录
    possible_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'None')
    if os.path.exists(possible_dir):
        latest_dir = possible_dir
        print(f"Force loading model from: {latest_dir}")
    else:
        # 默认日志目录
        logdir = './logs' 
        if not os.path.exists(logdir):
             # 尝试上一级目录
            logdir = '../logs'
        
        if not os.path.exists(logdir):
             # 使用默认配置里的 logdir
            config_path = os.path.join(os.path.dirname(__file__), 'config_fia49k.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logdir = config.get('logdir', './logs')

        # 找到最新的实验目录
        if os.path.exists(logdir):
            subdirs = [os.path.join(logdir, d) for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                print(f"Loading model from: {latest_dir}")
            else:
                print(f"Error: No training runs found in {logdir}")
                return
        else:
             print(f"Error: Log directory {logdir} not found")
             return

    # 加载配置
    saved_config_path = os.path.join(latest_dir, 'config.yaml')
    if os.path.exists(saved_config_path):
        print(f"Found saved configuration in {saved_config_path}")
        with open(saved_config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Fallback to default
        print("Warning: Saved config not found, using default config_fia49k.yaml")
        config_path = os.path.join(os.path.dirname(__file__), 'config_fia49k.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # 解析模型参数
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
    batch_size = config['batch_size']
    
    # 3. 加载数据集（使用刚刚转换的 EGNN 测试集）
    dataset_path = "data/temp_egnn_testset.npz"
    if not os.path.exists(dataset_path):
         print(f"Error: Converted dataset not found at {dataset_path}. Please run convert_egnn_test_to_dimenet.py first.")
         return
    
    if not os.path.isabs(dataset_path):
         dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)
    
    print(f"Loading dataset: {dataset_path}")
    data_container = DataContainer(dataset_path, cutoff=cutoff, target_keys=targets)
    
    # 4. 匹配 ID
    # 因为 temp_egnn_testset.npz 就是专门为测试集生成的，所以直接使用全部数据
    print("Using all molecules in the converted dataset...")
    matched_indices = np.arange(len(data_container))
    print(f"Total molecules to predict: {len(matched_indices)}")

    if len(matched_indices) == 0:
        print("Error: No molecules found.")
        return

    # 5. 初始化模型
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

    # 优先使用当前目录的 best 权重
    local_best_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best', 'ckpt')
    best_ckpt_file = local_best_ckpt if os.path.exists(local_best_ckpt + ".index") else os.path.join(latest_dir, 'best', 'ckpt')

    # 构建模型（Dummy pass）
    dummy_idx = matched_indices[:min(batch_size, len(matched_indices))]
    # 这里我们手动获取数据，因为 DataProvider 通常用于 split
    # 我们可以复用 DataProvider 的逻辑，但创建一个只包含 matched_indices 的 provider 可能更简单
    # 或者直接用 DataProvider.idx_to_data
    
    # 临时创建一个 DataProvider 实例来使用其 idx_to_data 方法
    # 参数 num_train/valid 并不重要，因为我们不使用 split 功能
    temp_provider = DataProvider(data_container, 1, 1, batch_size, seed=1, randomized=False)
    
    dummy_inputs, _ = temp_provider.idx_to_data(dummy_idx)
    _ = model(dummy_inputs, training=False)
    
    # 加载权重
    print(f"Loading weights from {best_ckpt_file}")
    try:
        # 尝试使用 load_weights
        model.load_weights(best_ckpt_file)
    except Exception as e:
        print(f"load_weights failed: {e}")
        print("Trying tf.train.Checkpoint.restore...")
        try:
            # 尝试使用 tf.train.Checkpoint
            ckpt = tf.train.Checkpoint(model=model)
            ckpt.restore(best_ckpt_file).expect_partial()
            print("Restored weights using tf.train.Checkpoint")
        except Exception as e2:
            print(f"Failed to restore weights: {e2}")
            return

    # 6. 预测
    print("Running predictions...")
    all_preds = []
    all_targets = []
    all_ids = []
    
    # 手动进行 batch 处理
    num_samples = len(matched_indices)
    steps = int(np.ceil(num_samples / batch_size))
    
    for i in range(steps):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)
        batch_idx = matched_indices[start:end]
        
        inputs, target_batch = temp_provider.idx_to_data(batch_idx)
        preds = model(inputs, training=False)
        
        all_preds.append(preds.numpy())
        all_targets.append(target_batch.numpy())
        all_ids.append(data_container.id[batch_idx])
        
        if i % 10 == 0:
            print(f"Batch {i}/{steps} processed")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)
    
    # 7. 计算指标和保存
    y_true = all_targets[:, 0].flatten()
    y_pred = all_preds[:, 0].flatten()
    residuals = y_pred - y_true
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("Test Results (on EGNN Test Set)")
    print("="*40)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    print("="*40)
    
    # 保存 Excel
    out_dir = 'predictions_on_egnn_testset'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    df = pd.DataFrame({
        'mol_id': all_ids,
        '真实值': y_true,
        '预测值': y_pred,
        '绝对误差': np.abs(residuals)
    })
    
    excel_path = os.path.join(out_dir, 'predictions.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"\nResults saved to: {excel_path}")
    
    # 绘图
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=15)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('True Values (DFT)')
    plt.ylabel('Predicted Values (DimeNet++)')
    plt.title(f'DimeNet++ on EGNN Test Set\nMAE={mae:.4f}, R²={r2:.4f}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'parity_plot.png'), dpi=300)
    
    plt.figure(figsize=(10, 6))
    plt.hist(np.abs(residuals), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'error_dist.png'), dpi=300)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.get_logger().setLevel('WARN')
    predict_on_egnn_testset()
