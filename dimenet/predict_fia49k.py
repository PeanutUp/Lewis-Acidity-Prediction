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

def predict_and_plot(config):
    # 配置
    logdir = config['logdir']
    
    # 尝试查找当前目录下的 None 文件夹作为模型目录
    possible_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'None')
    if os.path.exists(possible_dir):
        latest_dir = possible_dir
        print(f"Force loading model from: {latest_dir}")
    else:
        # 查找最新的训练目录
        if not os.path.exists(logdir):
            print(f"Error: Log directory {logdir} does not exist.")
            return

        # 找到最新的实验目录
        subdirs = [os.path.join(logdir, d) for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]
        if not subdirs:
            print(f"Error: No training runs found in {logdir}")
            return
        
        # 按时间排序
        latest_dir = max(subdirs, key=os.path.getmtime)
        print(f"Loading model from: {latest_dir}")
    
    # 优先使用训练目录中保存的配置，以确保参数完全对齐
    saved_config_path = os.path.join(latest_dir, 'config.yaml')
    if os.path.exists(saved_config_path):
        print(f"Found saved configuration in {saved_config_path}, using it instead of global config.")
        with open(saved_config_path, 'r') as f:
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
    
    # 强制使用正确的数据集路径
    dataset_path = "data/fia49k_full_processed.npz"

    num_train = config['num_train']
    num_valid = config['num_valid']
    data_seed = config['data_seed']
    batch_size = config['batch_size']
    targets = config['targets']
    
    # 优先使用当前目录的 best 权重
    local_best_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best', 'ckpt')
    best_ckpt_file = local_best_ckpt if os.path.exists(local_best_ckpt + ".index") else os.path.join(latest_dir, 'best', 'ckpt')

    # 加载数据
    if not os.path.isabs(dataset_path):
         dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)
    
    print(f"Loading dataset: {dataset_path}")
    data_container = DataContainer(dataset_path, cutoff=cutoff, target_keys=targets)
    
    # 划分数据
    data_provider = DataProvider(data_container, num_train, num_valid, batch_size,
                                 seed=data_seed, randomized=True)
    
    # 获取测试集信息
    num_test = data_provider.nsamples['test']
    
    print(f"Test samples: {num_test}")
    if num_test == 0:
        print("Error: No samples left for testing. Decrease num_train or num_valid.")
        return

    # 初始化模型
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
    
    # 构建模型（通过运行一次前向传播）
    test_indices = data_provider.idx['test']
    if len(test_indices) == 0:
        print("Error: No samples left for testing. Decrease num_train or num_valid.")
        return
    dummy_idx = test_indices[:batch_size]
    dummy_inputs, _ = data_provider.idx_to_data(dummy_idx)
    _ = model(dummy_inputs, training=False)
    
    # 加载权重
    print(f"Loading weights from {best_ckpt_file}")
    try:
        model.load_weights(best_ckpt_file)
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    # 预测
    print("Running predictions on test set...")
    all_preds = []
    all_targets = []
    all_indices = []
    
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
            print(f"Batch {i}/{steps} processed")

    all_preds = np.concatenate(all_preds, axis=0)[:num_test]
    all_targets = np.concatenate(all_targets, axis=0)[:num_test]
    all_indices = np.concatenate(all_indices, axis=0)[:num_test]
    
    # 计算指标
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    # 计算额外统计量
    if all_targets.shape[1] > 1:
        print(f"Warning: Multiple targets detected ({all_targets.shape[1]}). Using the first target for detailed export.")
    y_true = all_targets[:, 0].flatten()
    y_pred = all_preds[:, 0].flatten()
    residuals = y_pred - y_true
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    max_error = np.max(np.abs(residuals))
    
    print("\nDetailed Test Results:")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"MSE (Mean Squared Error):  {mse:.4f}")
    print(f"RMSE (Root Mean Sq Error): {rmse:.4f}")
    print(f"R2 Score:                  {r2:.4f}")
    print(f"Pearson Correlation:       {correlation:.4f}")
    print(f"Mean Residual (Bias):      {mean_res:.4f}")
    print(f"Std of Residuals:          {std_res:.4f}")
    print(f"Max Absolute Error:        {max_error:.4f}")
    
    # 保存结果
    results_dir = os.path.join(latest_dir, 'predictions')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    np.savez(os.path.join(results_dir, 'test_predictions.npz'),
             targets=all_targets, preds=all_preds)
    print(f"Predictions saved to {results_dir}")

    # 导出详细结果（JSON -> Excel）
    eps = 1e-8
    abs_error = np.abs(y_pred - y_true)
    rel_error = abs_error / (np.abs(y_true) + eps)
    if data_container.id is not None:
        true_index = data_container.id[all_indices]
    else:
        true_index = all_indices

    records = []
    for idx, t, p, ae, re in zip(true_index, y_true, y_pred, abs_error, rel_error):
        idx_dtype = np.asarray(idx).dtype
        idx_value = int(idx) if np.issubdtype(idx_dtype, np.integer) else idx
        records.append({
            "true_index": idx_value,
            "true_value": float(t),
            "pred_value": float(p),
            "abs_error": float(ae),
            "rel_error": float(re)
        })

    json_path = os.path.join(results_dir, "test_predictions_detail.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(records)
    df = df.sort_values(by="abs_error", ascending=True).reset_index(drop=True)
    excel_path = os.path.join(results_dir, "test_predictions_detail.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Detailed predictions saved to {excel_path}")

    # 绘图设置
    plt.switch_backend('Agg') # 使用非交互式后端
    plt.style.use('default')  # 重置样式
    
    # --- 图1: Prediction vs True Values ---
    plt.figure(figsize=(8, 8))
    
    # 绘制散点
    plt.scatter(y_true, y_pred, alpha=0.6, s=40, edgecolors='white', linewidth=0.5, color='#3498db', label='Data points')
    
    # 计算范围
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    limit_min = min_val - margin
    limit_max = max_val + margin
    
    # 绘制对角线
    plt.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', lw=2.5, label='Perfect Prediction')
    
    # 设置标签和标题
    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Prediction vs True Values\n$R^2$ = {r2:.4f}, Correlation = {correlation:.4f}', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='-', which='major')
    plt.legend(loc='upper left')
    
    plot_path_1 = os.path.join(results_dir, 'prediction_vs_true.png')
    plt.savefig(plot_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to {plot_path_1}")
    
    # --- 图2: Error Distribution ---
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(residuals, bins=50, color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 设置标签和标题
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Error Distribution\nMAE: {mae:.4f}', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='-', which='major')
    
    # 标记0刻度
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plot_path_2 = os.path.join(results_dir, 'error_distribution.png')
    plt.savefig(plot_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error distribution plot saved to {plot_path_2}")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.get_logger().setLevel('WARN')
    
    config_path = os.path.join(os.path.dirname(__file__), 'config_fia49k.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    predict_and_plot(config)
