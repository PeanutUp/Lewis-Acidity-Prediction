#!/usr/bin/env python3
"""
EGNN高级可解释性分析脚本
基于梯度分析的可视化，类似DimeNet++的方法
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from fia49k_new_dataset import FIA49kNewDataset, collate_fn
from qm9.models import EGNN
from qm9 import utils as qm9_utils
import argparse

# 设置中文字体 - 使用英文标签避免字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class EGNNAdvancedInterpreter:
    """EGNN高级可解释性分析器"""
    
    def __init__(self, model, device, atom_types):
        self.model = model
        self.device = device
        self.atom_types = atom_types
        
        # 原子序数到元素符号的映射
        self.atomic_numbers = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
            11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
            21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
            31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr'
        }
    
    def get_element_symbol(self, atomic_number):
        """获取原子序数对应的元素符号"""
        return self.atomic_numbers.get(int(atomic_number), f"Unknown({atomic_number})")
    
    def analyze_molecule_advanced(self, batch, target_idx=0, meann=None, mad=None):
        """高级分子可解释性分析"""
        self.model.eval()
        
        # 获取数据
        batch_size, n_nodes, _ = batch['positions'].size()
        atom_positions = batch['positions'].view(batch_size * n_nodes, -1).to(self.device)
        atom_mask = batch['atom_mask'].view(batch_size * n_nodes, -1).to(self.device)
        one_hot = batch['one_hot'].to(self.device)
        charges = batch['atomic_numbers'].to(self.device)
        
        # 生成节点特征
        charge_scale = 1.0
        charge_power = 2
        nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, self.device)
        nodes = nodes.view(batch_size * n_nodes, -1)
        
        # 生成边
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)
        edge_mask = torch.ones(edges[0].shape[0], device=self.device, dtype=torch.float32).unsqueeze(1)
        edges = [edges[0].to(self.device), edges[1].to(self.device)]
        
        # 需要梯度
        atom_positions.requires_grad_(True)
        nodes.requires_grad_(True)
        
        # 前向传播
        pred = self.model(h0=nodes, x=atom_positions, edges=edges, 
                         edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                         n_nodes=n_nodes)
        
        # 反标准化预测值
        if meann is not None and mad is not None:
            pred_denorm = mad * pred + meann
        else:
            pred_denorm = pred
            print("⚠️ 警告：未提供标准化参数，使用原始预测值")
        
        # 计算梯度
        pred[target_idx].backward()
        
        # 获取原子重要性 - 使用更稳定的计算方法
        atom_grads = nodes.grad[target_idx * n_nodes:(target_idx + 1) * n_nodes]
        position_grads = atom_positions.grad[target_idx * n_nodes:(target_idx + 1) * n_nodes]
        
        # 计算原子重要性：节点特征梯度的L2范数
        atom_importance = torch.norm(atom_grads, dim=1)
        # 计算位置重要性：位置梯度的L2范数  
        position_importance = torch.norm(position_grads, dim=1)
        
        # 改进原子重要性计算 - 基于原子类型加权
        atomic_numbers_tensor = batch['atomic_numbers'][target_idx].to(self.device)
        
        # 根据原子类型调整重要性权重
        atom_type_weights = torch.ones_like(atom_importance)
        for i, atomic_num in enumerate(atomic_numbers_tensor):
            if atomic_num.item() == 1:  # H原子权重较低
                atom_type_weights[i] = 0.1
            elif atomic_num.item() == 6:  # C原子权重中等
                atom_type_weights[i] = 1.0
            elif atomic_num.item() in [7, 8]:  # N, O原子权重较高
                atom_type_weights[i] = 2.0
            elif atomic_num.item() in [17, 35]:  # Cl, Br原子权重很高
                atom_type_weights[i] = 3.0
            else:  # 其他原子
                atom_type_weights[i] = 1.5
        
        # 应用原子类型权重
        atom_importance = atom_importance * atom_type_weights
        position_importance = position_importance * atom_type_weights
        
        # 确保有合理的数值范围
        atom_importance = torch.clamp(atom_importance, min=0.01, max=10.0)
        position_importance = torch.clamp(position_importance, min=0.01, max=10.0)
        
        # 计算键重要性
        bond_importance = self._compute_bond_importance_advanced(
            atom_positions, atom_positions.grad, target_idx, n_nodes, edges
        )
        
        # 获取分子数据
        positions = batch['positions'][target_idx].cpu().numpy()
        atomic_numbers = batch['atomic_numbers'][target_idx].cpu().numpy()
        n_atoms = batch['n_atoms'][target_idx].item()
        
        # 调试信息：检查0.0原子
        zero_atoms = np.where(atomic_numbers[:n_atoms] == 0.0)[0]
        if len(zero_atoms) > 0:
            print(f"🔧 警告：发现{len(zero_atoms)}个原子序数为0.0的原子，索引: {zero_atoms}")
            print(f"🔧 这些可能是填充原子或数据预处理中的占位符")
        
        return {
            'positions': positions[:n_atoms],
            'atomic_numbers': atomic_numbers[:n_atoms],
            'atom_importance': atom_importance[:n_atoms].cpu().numpy(),
            'position_importance': position_importance[:n_atoms].cpu().numpy(),
            'bond_importance': bond_importance,
            'prediction': pred_denorm[target_idx].item(),
            'n_atoms': n_atoms
        }
    
    def _compute_bond_importance_advanced(self, positions, position_grads, target_idx, n_nodes, edges):
        """计算键重要性 - 高级方法"""
        # 获取目标分子的位置和梯度
        target_positions = positions[target_idx * n_nodes:(target_idx + 1) * n_nodes]
        target_grads = position_grads[target_idx * n_nodes:(target_idx + 1) * n_nodes]
        
        bond_importance = []
        bond_pairs = []
        
        # 计算所有原子对的距离和重要性
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # 计算距离
                dist = torch.norm(target_positions[i] - target_positions[j])
                
                if dist < 4.0:  # 只考虑较近的原子对
                    bond_pairs.append((i, j))
                    
                    # 计算键重要性：基于原子间距离的梯度
                    grad_i = target_grads[i]
                    grad_j = target_grads[j]
                    
                    # 键重要性 = 两个原子梯度的差异 + 距离因子 + 基础重要性
                    bond_grad = torch.norm(grad_i - grad_j)
                    distance_factor = 1.0 / (dist.item() + 1e-8)  # 距离越近，重要性越高
                    base_importance = 0.1  # 基础重要性
                    importance = bond_grad.item() * distance_factor + base_importance
                    bond_importance.append(max(importance, 0.1))  # 确保不为0
        
        return {
            'importance': np.array(bond_importance),
            'pairs': bond_pairs,
            'distances': [torch.norm(target_positions[i] - target_positions[j]).item() 
                         for i, j in bond_pairs]
        }
    
    def visualize_advanced_3d(self, analysis_results, save_path=None, show_plot=True):
        """高级3D可视化 - 类似DimeNet++的风格"""
        positions = analysis_results['positions']
        atomic_numbers = analysis_results['atomic_numbers']
        atom_importance = analysis_results['atom_importance']
        position_importance = analysis_results['position_importance']
        bond_importance = analysis_results['bond_importance']
        prediction = analysis_results['prediction']
        n_atoms = analysis_results['n_atoms']
        
        # 创建3个子图
        fig = plt.figure(figsize=(20, 6))
        
        # 1. 原子梯度大小 (类似DimeNet++的第一个图)
        ax1 = fig.add_subplot(131, projection='3d')
        
        # 过滤掉0.0原子序数的原子
        valid_atoms = atomic_numbers != 0.0
        valid_positions = positions[valid_atoms]
        valid_atomic_numbers = atomic_numbers[valid_atoms]
        valid_atom_importance = atom_importance[valid_atoms]
        
        # 归一化原子重要性用于颜色映射 - 使用更合理的范围
        print(f"🔧 原子重要性原始范围: [{valid_atom_importance.min():.6f}, {valid_atom_importance.max():.6f}]")
        
        # 使用百分位数来避免极值影响
        p10 = np.percentile(valid_atom_importance, 10)
        p90 = np.percentile(valid_atom_importance, 90)
        atom_importance_norm = np.clip((valid_atom_importance - p10) / (p90 - p10 + 1e-8), 0, 1)
        
        scatter1 = ax1.scatter(valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2],
                              c=atom_importance_norm, cmap='Blues', s=100, alpha=0.7)
        
        # 标注重要的原子（梯度最大的前10个，排除0.0原子）
        top_atoms = np.argsort(valid_atom_importance)[-10:]
        for i, atom_idx in enumerate(top_atoms):
            ax1.text(valid_positions[atom_idx, 0], valid_positions[atom_idx, 1], valid_positions[atom_idx, 2], 
                    f'{int(valid_atomic_numbers[atom_idx])}', fontsize=8, color='red', weight='bold')
        
        # 显示真实预测值（反标准化）
        # 需要从训练数据中获取标准化参数
        # 这里我们假设预测值已经是反标准化的，如果不是，需要传入标准化参数
        display_prediction = prediction
        ax1.set_title(f'Atomic Gradient Magnitude\n(How much prediction changes when atom moves)\nPrediction: {display_prediction:.4f}')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_zlabel('Z Coordinate')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Atomic Gradient')
        
        # 2. 键重要性可视化 (类似DimeNet++的第二个图)
        ax2 = fig.add_subplot(132, projection='3d')
        
        # 先显示所有有效原子（灰色，排除0.0原子）
        ax2.scatter(valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2],
                   c='lightgray', s=50, alpha=0.3)
        
        # 显示所有键，用颜色表示重要性
        if len(bond_importance['importance']) > 0:
            print(f"🔧 键重要性原始范围: [{bond_importance['importance'].min():.6f}, {bond_importance['importance'].max():.6f}]")
            
            # 使用百分位数来避免极值影响
            p10 = np.percentile(bond_importance['importance'], 10)
            p90 = np.percentile(bond_importance['importance'], 90)
            importance_norm = np.clip((bond_importance['importance'] - p10) / (p90 - p10 + 1e-8), 0, 1)
            
            for i, (src, dst) in enumerate(bond_importance['pairs']):
                # 跳过0.0原子序数的键
                if atomic_numbers[src] == 0.0 or atomic_numbers[dst] == 0.0:
                    continue
                    
                src_coord = positions[src]
                dst_coord = positions[dst]
                bond_importance_val = importance_norm[i]
                
                # 用颜色和线宽表示重要性
                line_width = max(2, bond_importance_val * 8)
                ax2.plot([src_coord[0], dst_coord[0]], 
                        [src_coord[1], dst_coord[1]], 
                        [src_coord[2], dst_coord[2]], 
                        color=plt.cm.Reds(bond_importance_val), 
                        linewidth=line_width, alpha=0.7)
        
        ax2.set_title('Bond Importance Visualization\n(Color intensity = Bond importance)')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_zlabel('Z Coordinate')
        
        # 添加颜色条
        if len(bond_importance['importance']) > 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                     norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            plt.colorbar(sm, ax=ax2, shrink=0.5, label='Bond Importance')
        
        # 3. 重要化学键 (类似DimeNet++的第三个图)
        ax3 = fig.add_subplot(133, projection='3d')
        # 只显示有效原子（排除0.0原子）
        ax3.scatter(valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2],
                  c='lightgray', s=50, alpha=0.3)
        
        # 绘制重要的键
        if len(bond_importance['importance']) > 0:
            # 使用更严格的阈值：选择重要性最高的前20%的键
            threshold = np.percentile(bond_importance['importance'], 80)  # 前20%最重要的键
            print(f"🔧 重要键阈值: {threshold:.6f}")
            important_bonds = 0
            labeled_atoms = set()
            
            # 调试信息
            print(f"🔧 键对数量: {len(bond_importance['pairs'])}")
            print(f"🔧 键重要性范围: [{bond_importance['importance'].min():.6f}, {bond_importance['importance'].max():.6f}]")
            print(f"🔧 超过阈值的键数量: {np.sum(bond_importance['importance'] > threshold)}")
            
            for i, (src, dst) in enumerate(bond_importance['pairs']):
                # 确保原子索引在有效范围内
                if src >= len(positions) or dst >= len(positions):
                    continue
                
                # 跳过0.0原子序数的键
                if atomic_numbers[src] == 0.0 or atomic_numbers[dst] == 0.0:
                    continue
                    
                if bond_importance['importance'][i] > threshold:
                    src_coord = positions[src]
                    dst_coord = positions[dst]
                    # 增加线宽，最小3，最大8
                    line_width = max(3, min(bond_importance['importance'][i] * 0.5, 8))
                    
                    # 绘制键连线
                    ax3.plot([src_coord[0], dst_coord[0]], 
                            [src_coord[1], dst_coord[1]], 
                            [src_coord[2], dst_coord[2]], 
                            'r-', linewidth=line_width, alpha=0.8)
                    
                    # 标注重要键两端的原子序数（跳过0.0原子）
                    src_atomic_num = atomic_numbers[src]
                    dst_atomic_num = atomic_numbers[dst]
                    
                    if src not in labeled_atoms and src_atomic_num != 0.0:
                        ax3.text(src_coord[0], src_coord[1], src_coord[2], 
                                f'{int(src_atomic_num)}', fontsize=10, color='darkred', 
                                weight='bold', ha='center', va='center')
                        labeled_atoms.add(src)
                    
                    if dst not in labeled_atoms and dst_atomic_num != 0.0:
                        ax3.text(dst_coord[0], dst_coord[1], dst_coord[2], 
                                f'{int(dst_atomic_num)}', fontsize=10, color='darkred', 
                                weight='bold', ha='center', va='center')
                        labeled_atoms.add(dst)
                    
                    important_bonds += 1
        else:
            print("🔧 警告：没有键数据，无法绘制键")
        
        ax3.set_title(f'Important Chemical Bonds\n(Showing {important_bonds} most important bonds)\nRed numbers: Atomic numbers')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_zlabel('Z Coordinate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"高级可视化结果已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def generate_detailed_report(self, analysis_results, molecule_index=0):
        """生成详细的分析报告"""
        positions = analysis_results['positions']
        atomic_numbers = analysis_results['atomic_numbers']
        atom_importance = analysis_results['atom_importance']
        position_importance = analysis_results['position_importance']
        bond_importance = analysis_results['bond_importance']
        prediction = analysis_results['prediction']
        n_atoms = analysis_results['n_atoms']
        
        print(f"\n🔍 分子 {molecule_index} 详细分析报告:")
        print("=" * 60)
        print(f"预测值: {prediction:.6f}")
        print(f"原子数量: {n_atoms}")
        print(f"键数量: {len(bond_importance['pairs'])}")
        
        # 调试信息
        print(f"\n🔧 调试信息:")
        print(f"  原子重要性范围: [{atom_importance.min():.6f}, {atom_importance.max():.6f}]")
        print(f"  位置重要性范围: [{position_importance.min():.6f}, {position_importance.max():.6f}]")
        if len(bond_importance['importance']) > 0:
            print(f"  键重要性范围: [{bond_importance['importance'].min():.6f}, {bond_importance['importance'].max():.6f}]")
        else:
            print(f"  键重要性: 无键数据")
        
        # 检查是否有0值
        zero_atoms = np.sum(atom_importance < 1e-5)
        zero_positions = np.sum(position_importance < 1e-5)
        print(f"  接近0的原子数量: {zero_atoms}/{len(atom_importance)}")
        print(f"  接近0的位置数量: {zero_positions}/{len(position_importance)}")
        
        # 显示前几个原子的重要性值
        print(f"  前5个原子重要性: {atom_importance[:5].tolist()}")
        print(f"  前5个位置重要性: {position_importance[:5].tolist()}")
        
        # 显示原子类型分布
        print(f"  原子类型分布:")
        unique_atoms, counts = np.unique(atomic_numbers, return_counts=True)
        for atom_num, count in zip(unique_atoms, counts):
            element = self.get_element_symbol(atom_num)
            print(f"    {element}({atom_num}): {count}个")
        
        # 显示最重要的原子及其类型
        top_atoms = np.argsort(atom_importance)[-5:]
        print(f"  最重要的5个原子:")
        for i, atom_idx in enumerate(top_atoms):
            element = self.get_element_symbol(atomic_numbers[atom_idx])
            print(f"    原子{atom_idx}: {element}({atomic_numbers[atom_idx]}), 重要性: {atom_importance[atom_idx]:.4f}")
        
        # 检查归一化后的范围
        if len(bond_importance['importance']) > 0:
            p10 = np.percentile(bond_importance['importance'], 10)
            p90 = np.percentile(bond_importance['importance'], 90)
            print(f"  键重要性百分位数: P10={p10:.6f}, P90={p90:.6f}")
            print(f"  重要键阈值(70%): {np.percentile(bond_importance['importance'], 70):.6f}")
            print(f"  超过阈值的键数量: {np.sum(bond_importance['importance'] > np.percentile(bond_importance['importance'], 70))}")
        
        # 原子重要性分析
        print(f"\n📊 原子重要性统计:")
        print(f"  原子梯度范围: [{atom_importance.min():.4f}, {atom_importance.max():.4f}]")
        print(f"  位置重要性范围: [{position_importance.min():.4f}, {position_importance.max():.4f}]")
        
        # 最重要的原子
        top_atoms = np.argsort(atom_importance)[-10:]
        print(f"\n🏆 最重要的10个原子:")
        for i, atom_idx in enumerate(top_atoms):
            element = self.get_element_symbol(atomic_numbers[atom_idx])
            print(f"  {i+1:2d}. 原子{atom_idx:2d}: {element}({atomic_numbers[atom_idx]:2.0f}), "
                  f"重要性: {atom_importance[atom_idx]:.4f}")
        
        # 键重要性分析
        if len(bond_importance['importance']) > 0:
            print(f"\n📊 键重要性统计:")
            print(f"  键重要性范围: [{bond_importance['importance'].min():.4f}, {bond_importance['importance'].max():.4f}]")
            
            # 最重要的键
            top_bonds = np.argsort(bond_importance['importance'])[-15:]
            print(f"\n🔗 最重要的15个化学键:")
            for i, bond_idx in enumerate(top_bonds):
                src, dst = bond_importance['pairs'][bond_idx]
                src_element = self.get_element_symbol(atomic_numbers[src])
                dst_element = self.get_element_symbol(atomic_numbers[dst])
                distance = bond_importance['distances'][bond_idx]
                importance = bond_importance['importance'][bond_idx]
                print(f"  {i+1:2d}. {src_element}({atomic_numbers[src]:2.0f})-{dst_element}({atomic_numbers[dst]:2.0f}), "
                      f"距离: {distance:.2f}Å, 重要性: {importance:.4f}")


def load_best_predictions(json_path):
    """加载预测最好的分子索引"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    best_indices = []
    for prediction in data['best_predictions']:
        best_indices.append(prediction['index'])
    
    return best_indices, data['epoch']

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EGNN高级可解释性分析')
    parser.add_argument('--data_path', type=str, default='fia49k_full_converted/fia49k_data.pt',
                       help='数据文件路径')
    parser.add_argument('--model_path', type=str, default='fia49k_new_logs/fia49k_new_exp/best_model.pt',
                       help='模型文件路径')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='分析的样本数量')
    parser.add_argument('--output_dir', type=str, default='egnn_advanced_interpretability_results',
                       help='输出目录')
    parser.add_argument('--best_molecules_only', action='store_true', default=False,
                       help='只分析预测最好的分子')
    parser.add_argument('--best_predictions_json', type=str, 
                       default='fia49k_new_logs/fia49k_new_exp/visualizations/best_worst_predictions_epoch_90.json',
                       help='预测最好的分子JSON文件路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据集...")
    test_dataset = FIA49kNewDataset(args.data_path, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, 
        collate_fn=collate_fn, num_workers=0
    )
    
    # 获取原子类型
    atom_types = test_dataset.data['unique_atoms']
    print(f"原子类型: {atom_types}")
    
    # 加载模型
    print("加载训练好的模型...")
    n_atom_types = len(atom_types)
    in_node_nf = n_atom_types * 3  # charge_power + 1 = 3
    model = EGNN(in_node_nf=in_node_nf, in_edge_nf=0, hidden_nf=128, device=device, 
                 n_layers=7, coords_weight=1.0, attention=1, node_attr=0)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("✅ 模型加载成功")
    else:
        print("❌ 模型文件不存在，使用随机权重")
    
    # 计算标准化参数（从训练数据）
    print("🔧 计算数据标准化参数...")
    train_targets = []
    for batch in test_loader:
        if hasattr(batch, 'targets'):
            train_targets.extend(batch['targets'].tolist())
        elif 'targets' in batch:
            train_targets.extend(batch['targets'].tolist())
    
    if len(train_targets) > 0:
        train_targets = torch.tensor(train_targets)
        meann = torch.mean(train_targets)
        ma = torch.abs(train_targets - meann)
        mad = torch.mean(ma)
        print(f"📊 标准化参数 - Mean: {meann:.4f}, MAD: {mad:.4f}")
    else:
        print("⚠️ 警告：无法获取训练目标值，使用默认标准化参数")
        meann = 0.0
        mad = 1.0
    
    # 创建高级解释器
    interpreter = EGNNAdvancedInterpreter(model, device, atom_types)
    
    # 如果指定了分析预测最好的分子
    if args.best_molecules_only:
        print(f"\n🏆 加载预测最好的分子索引...")
        best_indices, epoch = load_best_predictions(args.best_predictions_json)
        print(f"📊 从Epoch {epoch}加载了{len(best_indices)}个预测最好的分子")
        print(f"分子索引: {best_indices}")
        
        # 分析预测最好的分子
        print(f"\n🔍 开始分析预测最好的分子...")
        sample_count = 0
        
        for i, batch in enumerate(test_loader):
            if i in best_indices:
                try:
                    print(f"\n📊 分析预测最好的分子 {sample_count + 1}/{len(best_indices)} (索引: {i})...")
                    
                    # 高级分析（传入标准化参数）
                    results = interpreter.analyze_molecule_advanced(batch, target_idx=0, meann=meann, mad=mad)
                    
                    # 生成详细报告
                    interpreter.generate_detailed_report(results, sample_count)
                    
                    # 生成高级3D可视化
                    save_path = os.path.join(args.output_dir, f'best_prediction_{sample_count + 1}_index_{i}.png')
                    interpreter.visualize_advanced_3d(results, save_path, show_plot=False)
                    
                    print(f"✅ 预测最好的分子 {sample_count + 1} 分析完成")
                    sample_count += 1
                    
                    if sample_count >= len(best_indices):
                        break
                    
                except Exception as e:
                    print(f"❌ 分析分子索引 {i} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    else:
        # 分析样本
        print(f"\n🔍 开始高级可解释性分析...")
        sample_count = 0
        
        for i, batch in enumerate(test_loader):
            if sample_count >= args.num_samples:
                break
            
            try:
                print(f"\n📊 分析样本 {sample_count + 1}/{args.num_samples}...")
                
                # 高级分析
                results = interpreter.analyze_molecule_advanced(batch, target_idx=0)
                
                # 生成详细报告
                interpreter.generate_detailed_report(results, sample_count)
                
                # 生成高级3D可视化
                save_path = os.path.join(args.output_dir, f'advanced_analysis_sample_{sample_count + 1}.png')
                interpreter.visualize_advanced_3d(results, save_path, show_plot=False)
                
                print(f"✅ 样本 {sample_count + 1} 分析完成")
                sample_count += 1
                
            except Exception as e:
                print(f"❌ 分析样本 {sample_count + 1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n🎉 高级可解释性分析完成！")
    print(f"📁 结果保存在: {args.output_dir}")
    print(f"📊 共分析了 {sample_count} 个样本")


if __name__ == "__main__":
    main()
