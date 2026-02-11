#!/usr/bin/env python3
"""
FIA49k新数据集训练脚本
基于main_qm9.py的结构，使用新的FIA49k数据集训练EGNN模型
"""

from fia49k_new_dataset import create_fia49k_new_dataloaders
from qm9.models import EGNN
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
import utils
import json
import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser(description='FIA49k New Dataset Example')
parser.add_argument('--exp_name', type=str, default='fia49k_new_exp', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=10, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='fia49k_new_logs', metavar='N',
                    help='folder to output logs')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='hidden dimension')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the model')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='number of layers')
parser.add_argument('--property', type=str, default='fia_gas-DSDBLYP', metavar='N',
                    help='property to predict')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')
parser.add_argument('--data_path', type=str, default='fia49k_full_converted/fia49k_data.pt',
                    help='path to converted data')
parser.add_argument('--num_best', type=int, default=10, metavar='N',
                    help='number of best predictions to analyze (default: 10)')
parser.add_argument('--num_worst', type=int, default=50, metavar='N',
                    help='number of worst predictions to analyze (default: 50)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

utils.makedir(args.outf)
utils.makedir(args.outf + "/" + args.exp_name)

# 加载数据
train_loader, val_loader, test_loader = create_fia49k_new_dataloaders(args.data_path, args.batch_size, args.num_workers)
# 重命名键以匹配QM9格式
dataloaders = {
    'train': train_loader,
    'val': val_loader, 
    'test': test_loader
}
# 计算charge_scale
charge_scale = 1.0  # 对于FIA49k数据集，我们使用固定的charge_scale

# 计算均值和标准差 - 只计算一次，避免数据泄露
print("计算数据标准化参数...")
train_targets = []
for batch in dataloaders['train']:
    train_targets.extend(batch['targets'].tolist())
train_targets = torch.tensor(train_targets)
meann = torch.mean(train_targets)
ma = torch.abs(train_targets - meann)
mad = torch.mean(ma)
print(f"Target property {args.property} - Mean: {meann:.4f}, MAD: {mad:.4f}")
print(f"数据范围: [{torch.min(train_targets):.2f}, {torch.max(train_targets):.2f}]")
print(f"标准化后范围: [{torch.min((train_targets - meann) / mad):.2f}, {torch.max((train_targets - meann) / mad):.2f}]")
print("=" * 60)

# 创建模型 - 使用正确的输入维度
# 动态计算输入特征维度：原子类型数 * (charge_power + 1)
n_atom_types = len(train_loader.dataset.data['unique_atoms'])
in_node_nf = n_atom_types * (args.charge_power + 1)
print(f"Atom types: {n_atom_types}, Input node features: {in_node_nf}")
model = EGNN(in_node_nf=in_node_nf, in_edge_nf=0, hidden_nf=args.nf, device=device, n_layers=args.n_layers, coords_weight=1.0,
             attention=args.attention, node_attr=args.node_attr)

print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()


def train(epoch, loader, partition='train'):
    res = {'loss': 0, 'counter': 0, 'loss_arr':[], 'predictions': [], 'targets': []}
    
    if partition == 'train':
        model.train()
    else:
        model.eval()
    
    for i, data in enumerate(loader):
        if partition == 'train':
            optimizer.zero_grad()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['atomic_numbers'].to(device, dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)
        # preprocess_input返回形状为[batch_size, n_nodes, features]，需要reshape
        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        # 简化：使用所有边都有效的掩码，并扩展维度以匹配edge_feat
        edge_mask = torch.ones(edges[0].shape[0], device=device, dtype=dtype).unsqueeze(1)
        label = data['targets'].to(device, dtype)

        if partition == 'train':
            pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                         n_nodes=n_nodes)
            # 训练时使用标准化后的目标值
            loss = loss_l1(pred, (label - meann) / mad)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                             n_nodes=n_nodes)
                # 验证/测试时计算真实的loss（反标准化后）
                pred_denorm = mad * pred + meann
                loss = loss_l1(pred_denorm, label)
                # 收集预测结果用于可视化
                res['predictions'].extend(pred_denorm.detach().cpu().numpy())
                res['targets'].extend(label.detach().cpu().numpy())

        res['loss'] += loss.item()
        res['counter'] += 1
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    return res['loss'] / res['counter'], res


def visualize_predictions(predictions, targets, epoch, save_dir):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 预测vs真实值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.6, s=50)
    
    # 计算R²
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    # 计算残差平方和
    ss_res = np.sum((targets - predictions) ** 2)
    # 计算总平方和
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    # 计算R²
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 也计算相关系数作为参考
    correlation = np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0
    
    # 绘制对角线
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Prediction vs True Values (Epoch {epoch})\nR² = {r2:.4f}, Correlation = {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'prediction_scatter_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 误差分布直方图
    errors = np.array(predictions) - np.array(targets)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (Epoch {epoch})\nMAE: {np.mean(np.abs(errors)):.4f}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'error_distribution_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return r2, np.mean(np.abs(errors))


def analyze_best_worst_predictions(predictions, targets, epoch, num_best=10, num_worst=50):
    """分析预测最好和最差的分子"""
    # 计算绝对误差
    errors = np.abs(np.array(predictions) - np.array(targets))
    
    # 找到最好和最差的预测
    best_indices = np.argsort(errors)[:num_best]
    worst_indices = np.argsort(errors)[-num_worst:]
    
    # 分析最好的预测
    print(f"\n预测最好的 {num_best} 个分子 (Epoch {epoch}):")
    print("=" * 80)
    for i, idx in enumerate(best_indices):
        true_val = targets[idx]
        pred_val = predictions[idx]
        error = errors[idx]
        mae = abs(true_val - pred_val)
        
        relative_error = (mae/abs(true_val)*100) if true_val != 0 else 0.0
        print(f"  {i+1:2d}. 分子 {idx:5d} | 真实值: {true_val:8.4f} | 预测值: {pred_val:8.4f} | 误差: {mae:8.4f} | 相对误差: {relative_error:6.2f}%")
    
    # 分析最差的预测
    print(f"\n预测最差的 {num_worst} 个分子 (Epoch {epoch}):")
    print("=" * 80)
    for i, idx in enumerate(worst_indices):
        true_val = targets[idx]
        pred_val = predictions[idx]
        error = errors[idx]
        mae = abs(true_val - pred_val)
        
        relative_error = (mae/abs(true_val)*100) if true_val != 0 else 0.0
        print(f"  {i+1:2d}. 分子 {idx:5d} | 真实值: {true_val:8.4f} | 预测值: {pred_val:8.4f} | 误差: {mae:8.4f} | 相对误差: {relative_error:6.2f}%")
    
    # 统计信息
    best_mae = np.mean([errors[i] for i in best_indices])
    worst_mae = np.mean([errors[i] for i in worst_indices])
    
    print(f"\n统计信息:")
    print(f"  最好预测平均MAE: {best_mae:.6f}")
    print(f"  最差预测平均MAE: {worst_mae:.6f}")
    
    # 保存结果到JSON文件
    results = {
        'epoch': epoch,
        'best_predictions': [],
        'worst_predictions': []
    }
    
    # 保存最好的预测
    for i, idx in enumerate(best_indices):
        result = {
            'rank': i + 1,
            'index': int(idx),
            'true_value': float(targets[idx]),
            'predicted_value': float(predictions[idx]),
            'absolute_error': float(errors[idx]),
            'relative_error': float(errors[idx] / abs(targets[idx]) * 100) if targets[idx] != 0 else 0.0
        }
        results['best_predictions'].append(result)
    
    # 保存最差的预测
    for i, idx in enumerate(worst_indices):
        result = {
            'rank': i + 1,
            'index': int(idx),
            'true_value': float(targets[idx]),
            'predicted_value': float(predictions[idx]),
            'absolute_error': float(errors[idx]),
            'relative_error': float(errors[idx] / abs(targets[idx]) * 100) if targets[idx] != 0 else 0.0
        }
        results['worst_predictions'].append(result)
    
    # 保存到文件
    import os
    vis_dir = os.path.join(args.outf, args.exp_name, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    results_file = os.path.join(vis_dir, f'best_worst_predictions_epoch_{epoch}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"  结果已保存到: {results_file}")


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    # 创建可视化输出目录
    vis_dir = os.path.join(args.outf, args.exp_name, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    for epoch in range(0, args.epochs):
        train(epoch, dataloaders['train'], partition='train')
        # 每个epoch后更新学习率
        lr_scheduler.step()
        if epoch % args.test_interval == 0:
            val_loss, val_res = train(epoch, dataloaders['val'], partition='valid')
            test_loss, test_res = train(epoch, dataloaders['test'], partition='test')
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            # 计算真实的MAE（反标准化后）
            if len(val_res['predictions']) > 0 and len(test_res['predictions']) > 0:
                val_mae = np.mean(np.abs(np.array(val_res['predictions']) - np.array(val_res['targets'])))
                test_mae = np.mean(np.abs(np.array(test_res['predictions']) - np.array(test_res['targets'])))
            else:
                print("警告：验证/测试数据为空，跳过MAE计算")
                val_mae = 0.0
                test_mae = 0.0
            
            print("Val loss: %.4f (MAE: %.4f) \t test loss: %.4f (MAE: %.4f) \t epoch %d" % 
                  (val_loss, val_mae, test_loss, test_mae, epoch))
            
            # 基于真实MAE选择最佳模型
            if val_mae < res['best_val']:
                res['best_val'] = val_mae
                res['best_test'] = test_mae
                res['best_epoch'] = epoch
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(args.outf, args.exp_name, "best_model.pt"))
                print("新的最佳模型！Val MAE: %.4f, Test MAE: %.4f" % (val_mae, test_mae))
                
                # 为最佳模型生成可视化图片
                if len(test_res['predictions']) > 0:
                    r2, mae = visualize_predictions(test_res['predictions'], test_res['targets'], epoch, vis_dir)
                    print(f"最佳模型可视化已生成 - Test R²: {r2:.4f}, Test MAE: {mae:.4f}")
                    
                    # 分析最好和最差的预测
                    analyze_best_worst_predictions(test_res['predictions'], test_res['targets'], epoch, 
                                                 args.num_best, args.num_worst)
            
            print("Best: val MAE: %.4f \t test MAE: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
            
            # 可视化测试结果 - 只在特定epoch进行详细分析
            if len(test_res['predictions']) > 0:
                # 只在每10个epoch或最后几个epoch进行详细分析
                if epoch % 10 == 0 or epoch >= args.epochs - 3:
                    r2, mae = visualize_predictions(test_res['predictions'], test_res['targets'], epoch, vis_dir)
                    print(f"Test R²: {r2:.4f}, Test MAE: {mae:.4f}")
                    
                    # 分析最好和最差的预测
                    analyze_best_worst_predictions(test_res['predictions'], test_res['targets'], epoch, 
                                                 args.num_best, args.num_worst)
                else:
                    # 其他epoch只显示基本统计
                    test_mae = np.mean(np.abs(np.array(test_res['predictions']) - np.array(test_res['targets'])))
                    print(f"Test MAE: {test_mae:.4f}")

        # 只在测试时保存结果
        if epoch % args.test_interval == 0:
            # 转换numpy类型为Python原生类型，确保JSON可序列化
            res_serializable = {
                'epochs': [int(x) for x in res['epochs']],
                'losess': [float(x) for x in res['losess']],
                'best_val': float(res['best_val']),
                'best_test': float(res['best_test']),
                'best_epoch': int(res['best_epoch'])
            }
            json_object = json.dumps(res_serializable, indent=4)
            with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
                outfile.write(json_object)
    
    # 训练结束后，进行最终的可视化
    print("\n" + "="*60)
    print("训练完成！正在进行最终可视化...")
    print("="*60)
    
    # 加载最佳模型进行最终测试
    if os.path.exists(os.path.join(args.outf, args.exp_name, "best_model.pt")):
        print("加载最佳模型进行最终测试...")
        model.load_state_dict(torch.load(os.path.join(args.outf, args.exp_name, "best_model.pt")))
        model.eval()
        
        # 在测试集上进行最终预测
        final_test_loss, final_test_res = train(args.epochs - 1, dataloaders['test'], partition='test')
        
        if len(final_test_res['predictions']) > 0:
            # 生成最终的可视化
            final_r2, final_mae = visualize_predictions(final_test_res['predictions'], final_test_res['targets'], 
                                                       args.epochs - 1, vis_dir)
            print(f"最终模型性能 - Test R²: {final_r2:.4f}, Test MAE: {final_mae:.4f}")
            
            # 分析最终模型的最好和最差预测
            analyze_best_worst_predictions(final_test_res['predictions'], final_test_res['targets'], 
                                         args.epochs - 1, args.num_best, args.num_worst)
            
            print(f"最终可视化已保存到: {vis_dir}")
            print(f"   - prediction_scatter_epoch_{args.epochs-1}.png")
            print(f"   - error_distribution_epoch_{args.epochs-1}.png")
            print(f"   - best_worst_predictions_epoch_{args.epochs-1}.json")
        else:
            print("警告：无法获取最终测试结果，跳过最终可视化")
    else:
        print("警告：未找到最佳模型文件，跳过最终可视化")
    
    print("="*60)
    print("训练和可视化完成！")
    print("="*60)