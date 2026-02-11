#!/usr/bin/env python3
"""
DimeNet++ 可解释性分析脚本
用于分析模型预测的可解释性
"""

import tensorflow as tf
import numpy as np
import os
import yaml
import ast
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 导入必要的模块
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider


class DimeNetExplainability:
    """DimeNet++ 可解释性分析器 - 基于RBF注意力机制"""
    
    def __init__(self, model, data_container):
        self.model = model
        self.data_container = data_container
    
    def analyze_molecule(self, inputs, target_class=0):
        """分析单个分子的可解释性"""
        
        # 1. 提取RBF注意力权重（无需梯度计算）
        rbf_attention = self._extract_rbf_attention(inputs)
        
        # 2. 计算原子坐标梯度
        atomic_gradients = self._compute_atomic_gradients(inputs, target_class)
        
        # 3. 计算键重要性
        bond_importance = self._compute_bond_importance(inputs, target_class, rbf_attention)
        
        # 4. 分子级分析
        molecular_contribution = self._analyze_molecular_contribution(inputs, target_class)
        
        return {
            'rbf_attention': rbf_attention,
            'atomic_gradients': atomic_gradients,
            'bond_importance': bond_importance,
            'molecular_contribution': molecular_contribution,
            'coordinates': inputs['R'],
            'atomic_numbers': inputs['Z'],
            'bond_indices': (inputs['idnb_i'], inputs['idnb_j']),
            'distances': self.model.calculate_interatomic_distances(
                inputs['R'], inputs['idnb_i'], inputs['idnb_j'])
        }
    
    def _extract_rbf_attention(self, inputs):
        """提取RBF注意力权重"""
        # 将numpy数组转换为TensorFlow张量
        R_tensor = tf.constant(inputs['R'], dtype=tf.float32)
        
        # 计算距离
        Dij = self.model.calculate_interatomic_distances(
            R_tensor, inputs['idnb_i'], inputs['idnb_j'])
        
        # 通过RBF层
        rbf = self.model.rbf_layer(Dij)
        
        # 通过输出层的RBF门控（使用第一个输出块）
        rbf_gate = self.model.output_blocks[0].dense_rbf(rbf)
        
        # 对RBF门控进行聚合，得到每个键的重要性
        rbf_attention = tf.reduce_sum(tf.abs(rbf_gate), axis=-1)
        
        return rbf_attention  # 这就是RBF注意力权重
    
    def _compute_atomic_gradients(self, inputs, target_class):
        """计算原子坐标梯度"""
        # 将numpy数组转换为TensorFlow张量
        R_tensor = tf.constant(inputs['R'], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(R_tensor)
            # 创建新的inputs字典，使用张量版本的R
            inputs_tensor = inputs.copy()
            inputs_tensor['R'] = R_tensor
            predictions = self.model(inputs_tensor, training=False)
            target_pred = predictions[:, target_class]
        
        gradients = tape.gradient(target_pred, R_tensor)
        return gradients
    
    def _compute_bond_importance(self, inputs, target_class, rbf_attention):
        """计算键重要性 - 结合RBF注意力和梯度"""
        # 将numpy数组转换为TensorFlow张量
        R_tensor = tf.constant(inputs['R'], dtype=tf.float32)
        
        # 计算距离
        Dij = self.model.calculate_interatomic_distances(
            R_tensor, inputs['idnb_i'], inputs['idnb_j'])
        
        # 计算距离梯度
        with tf.GradientTape() as tape:
            tape.watch(R_tensor)
            # 创建新的inputs字典，使用张量版本的R
            inputs_tensor = inputs.copy()
            inputs_tensor['R'] = R_tensor
            predictions = self.model(inputs_tensor, training=False)
            target_pred = predictions[:, target_class]
        
        gradients = tape.gradient(target_pred, R_tensor)
        
        # 计算每个键的梯度
        Ri = tf.gather(R_tensor, inputs['idnb_i'])
        Rj = tf.gather(R_tensor, inputs['idnb_j'])
        Gi = tf.gather(gradients, inputs['idnb_i'])
        Gj = tf.gather(gradients, inputs['idnb_j'])
        
        # 键梯度 = 两个原子梯度的组合
        bond_gradients = tf.norm(Gi - Gj, axis=-1)
        
        # 结合RBF注意力权重
        bond_importance = tf.abs(bond_gradients) * tf.abs(rbf_attention)
        
        return {
            'importance': bond_importance,
            'distances': Dij,
            'gradients': bond_gradients,
            'rbf_attention': rbf_attention
        }
    
    def _analyze_molecular_contribution(self, inputs, target_class):
        """分析分子级贡献"""
        predictions = self.model(inputs, training=False)
        return predictions[:, target_class]
    
    def visualize_explanations(self, analysis_results, molecule_index=0, 
                              save_path=None, show_plot=True):
        """可视化可解释性结果"""
        coords = analysis_results['coordinates']
        atomic_nums = analysis_results['atomic_numbers']
        rbf_attention = analysis_results['rbf_attention']
        atomic_grads = analysis_results['atomic_gradients']
        bond_importance = analysis_results['bond_importance']
        
        # 创建子图
        fig = plt.figure(figsize=(20, 6))
        
        # 1. 原子梯度可视化（使用原子级别的信息）
        ax1 = fig.add_subplot(131, projection='3d')
        grad_magnitude = tf.norm(atomic_grads, axis=-1)
        scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                              c=grad_magnitude, cmap='Blues', s=100, alpha=0.7)
        
        # 标注重要的原子（梯度最大的前5个）
        grad_magnitude_np = grad_magnitude.numpy()
        top_atoms = np.argsort(grad_magnitude_np)[-5:]  # 前5个最重要的原子
        for i, atom_idx in enumerate(top_atoms):
            ax1.text(coords[atom_idx, 0], coords[atom_idx, 1], coords[atom_idx, 2], 
                    f'{atomic_nums[atom_idx]}', fontsize=8, color='red', weight='bold')
        
        ax1.set_title('Atomic Gradient Magnitude\n(How much prediction changes when atom moves)')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_zlabel('Z Coordinate')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5)
        
        # 2. 键重要性可视化（直接显示键的重要性）
        ax2 = fig.add_subplot(132, projection='3d')
        
        # 先显示所有原子（灰色）
        ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c='lightgray', s=50, alpha=0.3)
        
        # 显示所有键，用颜色表示重要性
        bond_indices = analysis_results['bond_indices']
        importance = bond_importance['importance']
        
        # 绘制所有键，用颜色表示重要性
        for i, (src, dst) in enumerate(zip(bond_indices[0], bond_indices[1])):
            src_coord = coords[src]
            dst_coord = coords[dst]
            bond_importance_val = float(importance[i])
            
            # 用颜色表示重要性
            ax2.plot([src_coord[0], dst_coord[0]], 
                    [src_coord[1], dst_coord[1]], 
                    [src_coord[2], dst_coord[2]], 
                    color=plt.cm.Reds(bond_importance_val), 
                    linewidth=2, alpha=0.7)
        
        ax2.set_title('Bond Importance Visualization\n(Color intensity = Bond importance)')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_zlabel('Z Coordinate')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax2, shrink=0.5, label='Bond Importance')
        
        # 3. 重要键连接可视化
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                              c='lightgray', s=50, alpha=0.3)
        
        # 绘制重要的键
        threshold = float(tf.reduce_mean(importance)) * 1.5
        important_bonds = 0
        labeled_atoms = set()  # 记录已标注的原子，避免重复标注
        
        for i, (src, dst) in enumerate(zip(bond_indices[0], bond_indices[1])):
            if importance[i] > threshold:
                src_coord = coords[src]
                dst_coord = coords[dst]
                line_width = min(float(importance[i]) * 0.1, 5)  # 限制线宽
                ax3.plot([src_coord[0], dst_coord[0]], 
                        [src_coord[1], dst_coord[1]], 
                        [src_coord[2], dst_coord[2]], 
                        'r-', linewidth=line_width, alpha=0.7)
                
                # 标注重要键两端的原子序数
                src_atomic_num = atomic_nums[src]
                dst_atomic_num = atomic_nums[dst]
                
                # 标注源原子（如果还没标注过）
                if src not in labeled_atoms:
                    ax3.text(src_coord[0], src_coord[1], src_coord[2], 
                            f'{src_atomic_num}', fontsize=8, color='darkred', 
                            weight='bold', ha='center', va='center')
                    labeled_atoms.add(src)
                
                # 标注目标原子（如果还没标注过）
                if dst not in labeled_atoms:
                    ax3.text(dst_coord[0], dst_coord[1], dst_coord[2], 
                            f'{dst_atomic_num}', fontsize=8, color='darkred', 
                            weight='bold', ha='center', va='center')
                    labeled_atoms.add(dst)
                
                important_bonds += 1
        
        ax3.set_title(f'Important Chemical Bonds\n(Showing {important_bonds} most important bonds)\nRed numbers: Atomic numbers')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_zlabel('Z Coordinate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def generate_summary_report(self, analysis_results, molecule_index=0):
        """生成可解释性分析摘要报告"""
        rbf_attention = analysis_results['rbf_attention']
        atomic_grads = analysis_results['atomic_gradients']
        bond_importance = analysis_results['bond_importance']
        molecular_contribution = analysis_results['molecular_contribution']
        
        # 计算统计信息
        grad_magnitude = tf.norm(atomic_grads, axis=-1)
        
        report = {
            'molecule_index': molecule_index,
            'prediction': float(molecular_contribution),
            'rbf_attention_range': {
                'min': float(tf.reduce_min(rbf_attention)),
                'max': float(tf.reduce_max(rbf_attention))
            },
            'atomic_gradient_range': {
                'min': float(tf.reduce_min(grad_magnitude)),
                'max': float(tf.reduce_max(grad_magnitude))
            },
            'bond_importance_range': {
                'min': float(tf.reduce_min(bond_importance['importance'])),
                'max': float(tf.reduce_max(bond_importance['importance']))
            },
            'num_atoms': int(tf.shape(atomic_grads)[0]),
            'num_bonds': int(tf.shape(bond_importance['importance'])[0])
        }
        
        return report


def load_config(config_path='config_pp_minimal.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as c:
        config = yaml.safe_load(c)
    
    # 处理字符串类型的配置
    for key, val in config.items():
        if type(val) is str:
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
    
    return config


def load_model_and_data(config, best_weights_path):
    """加载模型和数据"""
    print("🔧 加载模型和数据...")
    
    # 加载数据
    data_container = DataContainer(
        config['dataset'], 
        cutoff=config['cutoff'], 
        target_keys=config['targets']
    )
    
    # 创建数据提供者
    data_provider = DataProvider(
        data_container, 
        config['num_train'], 
        config['num_valid'], 
        config['batch_size'],
        seed=config['data_seed'], 
        randomized=True
    )
    
    # 创建模型
    model = DimeNetPP(
        emb_size=config['emb_size'],
        out_emb_size=config['out_emb_size'],
        int_emb_size=config['int_emb_size'],
        basis_emb_size=config['basis_emb_size'],
        num_blocks=config['num_blocks'],
        num_spherical=config['num_spherical'],
        num_radial=config['num_radial'],
        cutoff=config['cutoff'],
        envelope_exponent=config['envelope_exponent'],
        num_before_skip=config['num_before_skip'],
        num_after_skip=config['num_after_skip'],
        num_dense_output=config['num_dense_output'],
        num_targets=len(config['targets']),
        activation=swish,
        extensive=config['extensive'],
        output_init=config['output_init'],
        minimal_output=config.get('minimal_output', False),
        freeze_features=config.get('freeze_features', False),
        pretrained_weights_path=config.get('pretrained_weights_path', None)
    )
    
    # 加载最佳权重
    if os.path.exists(best_weights_path):
        try:
            # 如果是tar文件，需要先解压
            if best_weights_path.endswith('.tar'):
                import tarfile
                import tempfile
                
                # 创建临时目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 解压tar文件
                    with tarfile.open(best_weights_path, 'r') as tar:
                        tar.extractall(temp_dir)
                    
                    # 查找best目录下的checkpoint文件
                    best_checkpoint_dir = os.path.join(temp_dir, 'best')
                    
                    if os.path.exists(best_checkpoint_dir):
                        print(f"🔍 找到best目录: {best_checkpoint_dir}")
                        
                        # 列出best目录下的所有文件
                        best_files = os.listdir(best_checkpoint_dir)
                        print(f"📁 best目录文件: {best_files}")
                        
                        # 使用ckpt文件加载（这是最可靠的方法）
                        ckpt_files = [f for f in best_files if f.startswith('ckpt') and f.endswith('.index')]
                        if ckpt_files:
                            ckpt_base = os.path.join(best_checkpoint_dir, ckpt_files[0].replace('.index', ''))
                            try:
                                print(f"🔄 加载ckpt文件: {ckpt_base}")
                                checkpoint = tf.train.Checkpoint(model=model)
                                checkpoint.restore(ckpt_base)
                                print(f"✅ 成功加载权重!")
                            except Exception as e:
                                print(f"❌ 加载失败: {e}")
                                print("将使用随机初始化权重")
                        else:
                            print("❌ 未找到ckpt文件，将使用随机初始化权重")
                    else:
                        print("❌ 未找到best目录，将使用随机初始化权重")
            else:
                # 直接加载checkpoint文件
                try:
                    print(f"🔄 加载checkpoint文件: {best_weights_path}")
                    
                    # 读取checkpoint文件内容
                    with open(best_weights_path, 'r') as f:
                        checkpoint_content = f.read()
                    print(f"📄 checkpoint内容: {checkpoint_content}")
                    
                    # 从checkpoint文件中提取实际的权重文件路径
                    import re
                    match = re.search(r'model_checkpoint_path: "([^"]+)"', checkpoint_content)
                    if match:
                        ckpt_base = os.path.join(os.path.dirname(best_weights_path), match.group(1))
                        print(f"🔍 权重文件基础路径: {ckpt_base}")
                        
                        # 检查是否存在ckpt文件（不包含扩展名）
                        if os.path.exists(ckpt_base):
                            actual_ckpt_path = ckpt_base
                            print(f"✅ 找到权重文件: {actual_ckpt_path}")
                        else:
                            # 检查是否存在ckpt.index文件
                            ckpt_index_path = ckpt_base + ".index"
                            if os.path.exists(ckpt_index_path):
                                actual_ckpt_path = ckpt_base
                                print(f"✅ 找到权重文件: {actual_ckpt_path}")
                            else:
                                print(f"❌ 权重文件不存在: {ckpt_base}")
                                print("将使用随机初始化权重")
                        
                        try:
                            checkpoint = tf.train.Checkpoint(model=model)
                            status = checkpoint.restore(actual_ckpt_path)
                            print(f"✅ 成功加载权重!")
                            print(f"📊 加载状态: {status}")
                        except Exception as e:
                            print(f"❌ 权重加载失败: {e}")
                            print("将使用随机初始化权重")
                    else:
                        print("❌ 无法从checkpoint文件中提取权重路径")
                        print("将使用随机初始化权重")
                        
                except Exception as e:
                    print(f"❌ 加载checkpoint失败: {e}")
                    print("将使用随机初始化权重")
        except Exception as e:
            print(f"⚠️ 加载权重失败: {e}")
            print("将使用随机初始化权重")
    else:
        print(f"⚠️ 未找到权重文件: {best_weights_path}")
        print("将使用随机初始化权重")
    
    return model, data_container, data_provider


def explain_single_molecule(model, data_container, molecule_index=0, save_dir=None):
    """解释单个分子的预测"""
    print(f"🔍 分析分子 {molecule_index} 的可解释性...")
    
    # 创建可解释性分析器
    explainer = DimeNetExplainability(model, data_container)
    
    # 获取分子数据
    inputs = data_container[molecule_index]
    
    # 进行可解释性分析
    results = explainer.analyze_molecule(inputs, target_class=0)
    
    # 生成摘要报告
    report = explainer.generate_summary_report(results, molecule_index)
    
    # 打印摘要
    print(f"\n📊 分子 {molecule_index} 可解释性分析摘要:")
    print(f"  预测值: {report['prediction']:.4f}")
    print(f"  原子数量: {report['num_atoms']}")
    print(f"  键数量: {report['num_bonds']}")
    print(f"  RBF注意力权重范围: [{report['rbf_attention_range']['min']:.4f}, {report['rbf_attention_range']['max']:.4f}]")
    print(f"  原子梯度大小范围: [{report['atomic_gradient_range']['min']:.4f}, {report['atomic_gradient_range']['max']:.4f}]")
    print(f"  键重要性范围: [{report['bond_importance_range']['min']:.4f}, {report['bond_importance_range']['max']:.4f}]")
    
    # 解释可视化内容
    print(f"\n🔍 可视化解释:")
    print(f"  图1 (蓝色): 原子梯度大小 - 显示每个原子位置变化对预测的影响程度")
    print(f"  图2 (红色): 原子键重要性 - 显示每个原子连接的化学键的重要性总和")
    print(f"  图3 (红色线条): 重要化学键 - 显示对预测最重要的化学键连接")
    print(f"  颜色越深 = 重要性越高，线条越粗 = 键越重要")
    
    # 详细分析重要原子和键
    print(f"\n🎯 重要原子和键分析:")
    
    # 分析重要原子（按梯度）
    grad_magnitude = tf.norm(results['atomic_gradients'], axis=-1).numpy()
    top_gradient_atoms = np.argsort(grad_magnitude)[-5:]
    print(f"  梯度最重要的5个原子:")
    for i, atom_idx in enumerate(top_gradient_atoms):
        atomic_num = results['atomic_numbers'][atom_idx]
        element = get_element_symbol(atomic_num)
        print(f"    {i+1}. 原子{atom_idx}: {element}({atomic_num}), 梯度大小: {grad_magnitude[atom_idx]:.2f}")
    
    # 分析重要键
    importance = results['bond_importance']['importance'].numpy()
    bond_indices = results['bond_indices']
    top_bonds = np.argsort(importance)[-10:]  # 前10个最重要的键
    print(f"  最重要的10个化学键:")
    for i, bond_idx in enumerate(top_bonds):
        src_atom = bond_indices[0][bond_idx]
        dst_atom = bond_indices[1][bond_idx]
        src_atomic_num = results['atomic_numbers'][src_atom]
        dst_atomic_num = results['atomic_numbers'][dst_atom]
        src_element = get_element_symbol(src_atomic_num)
        dst_element = get_element_symbol(dst_atomic_num)
        distance = results['distances'][bond_idx].numpy()
        print(f"    {i+1}. {src_element}(原子序数{src_atomic_num})-{dst_element}(原子序数{dst_atomic_num}), "
              f"距离: {distance:.2f}Å, 重要性: {importance[bond_idx]:.2f}")


    # 不在这里可视化，等所有分析完成后再统一生成图片
    # 只返回分析结果
    
    return results, report


def get_element_symbol(atomic_number):
    """获取原子序数对应的元素符号"""
    elements = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 13: 'Al', 16: 'S', 17: 'Cl', 35: 'Br'
    }
    return elements.get(atomic_number, f'Z{atomic_number}')


def predict_all_molecules(model, data_container, data_provider):
    """预测所有测试分子并返回预测结果"""
    print("🔮 开始预测所有测试分子...")
    
    predictions = []
    molecule_indices = []
    
    # 获取所有测试分子
    test_size = data_provider.nsamples['test']
    print(f"测试集总共有 {test_size} 个分子")
    
    # 批量预测所有测试分子
    for i in range(test_size):
        try:
            # 获取分子数据
            inputs = data_container[i]
            
            # 进行预测
            pred = model(inputs, training=False)
            prediction_value = float(pred[0])  # 假设是单输出
            
            predictions.append(prediction_value)
            molecule_indices.append(i)
            
            if (i + 1) % 100 == 0:
                print(f"已预测 {i + 1}/{test_size} 个分子...")
                
        except Exception as e:
            print(f"⚠️ 分子 {i} 预测失败: {e}")
            continue
    
    print(f"✅ 完成预测，共 {len(predictions)} 个分子")
    return predictions, molecule_indices


def find_top_predictions(predictions, molecule_indices, top_k=10):
    """找到预测值最高的前k个分子"""
    # 将预测值和分子索引配对
    pred_pairs = list(zip(predictions, molecule_indices))
    
    # 按预测值降序排序
    pred_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # 返回前k个
    top_predictions = pred_pairs[:top_k]
    
    print(f"\n🏆 预测最好的前{top_k}个分子:")
    print("=" * 60)
    for i, (pred_value, mol_idx) in enumerate(top_predictions):
        print(f"  {i+1:2d}. 分子索引: {mol_idx:4d} | 预测值: {pred_value:.6f}")
    print("=" * 60)
    
    # 额外统计信息
    all_predictions = [pred for pred, _ in pred_pairs]
    print(f"\n📊 预测统计信息:")
    print(f"  最高预测值: {max(all_predictions):.6f}")
    print(f"  最低预测值: {min(all_predictions):.6f}")
    print(f"  总分子数: {len(all_predictions)}")
    
    return top_predictions


def main():
    """主函数"""
    print("启动DimeNet++可解释性分析...")
    
    # 加载配置
    config = load_config()
    
    # 设置最佳权重路径（使用logs目录中的最新权重）
    best_weights_path = "logs/checkpoint"
    
    # 加载模型和数据（包含最佳权重）
    model, data_container, data_provider = load_model_and_data(config, best_weights_path)
    
    # 第一步：预测所有测试分子
    predictions, molecule_indices = predict_all_molecules(model, data_container, data_provider)
    
    # 第二步：找到预测最好的前10个分子
    top_10_molecules = find_top_predictions(predictions, molecule_indices, top_k=10)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"top10_predictions_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n结果将保存到: {save_dir}")
    
    # 第三步：对前10个最佳预测分子进行可解释性分析和可视化
    print(f"\n🔍 开始分析前10个最佳预测分子...")
    
    successful_analyses = 0
    failed_analyses = 0
    all_results = []  # 存储所有分析结果
    
    for i, (pred_value, mol_idx) in enumerate(top_10_molecules):
        try:
            print(f"\n[{i+1}/10] 分析第{i+1}名最佳预测分子:")
            print(f"  分子索引: {mol_idx}")
            print(f"  预测值: {pred_value:.6f}")
            print(f"  开始可解释性分析...")
            
            results, report = explain_single_molecule(
                model, data_container, mol_idx, 
                save_dir=None  # 不保存图片
            )
            all_results.append((mol_idx, results, report, pred_value))
            successful_analyses += 1
            print(f"✅ 第{i+1}名分子 {mol_idx} 分析完成")
            
        except Exception as e:
            failed_analyses += 1
            print(f"❌ 第{i+1}名分子 {mol_idx} 分析失败: {e}")
            continue
    
    # 第四步：生成可视化图片
    print(f"\n🎨 开始生成前10个最佳预测分子的可视化图片...")
    explainer = DimeNetExplainability(model, data_container)
    
    for i, (mol_idx, results, report, pred_value) in enumerate(all_results):
        try:
            save_path = os.path.join(save_dir, f"top_{i+1}_molecule_{mol_idx}_pred_{pred_value:.4f}.png")
            explainer.visualize_explanations(results, mol_idx, save_path, show_plot=False)
            print(f"✅ 第{i+1}名分子 {mol_idx} 可视化完成")
                
        except Exception as e:
            print(f"❌ 第{i+1}名分子 {mol_idx} 可视化失败: {e}")
            continue
    
    print(f"✅ 前10个最佳预测分子可视化图片生成完成！")
    
    # 保存预测结果到CSV文件
    results_data = []
    for i, (pred_value, mol_idx) in enumerate(top_10_molecules):
        results_data.append({
            '排名': i + 1,
            '分子索引': mol_idx,
            '预测值': pred_value
        })
    
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(save_dir, "top10_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 预测结果已保存到: {csv_path}")
    
    # 最终显示前10名分子详细信息
    print(f"\n🏆 最终结果 - 预测最好的前10个分子:")
    print("=" * 80)
    for i, (pred_value, mol_idx) in enumerate(top_10_molecules):
        status = "✅ 已分析" if i < successful_analyses else "❌ 分析失败"
        print(f"  第{i+1:2d}名: 分子索引 {mol_idx:4d} | 预测值 {pred_value:.6f} | {status}")
    print("=" * 80)
    
    print(f"\n🎉 前10个最佳预测分子可解释性分析完成！")
    print(f"📁 结果保存在: {save_dir}")
    print(f"📊 分析统计:")
    print(f"  成功分析: {successful_analyses} 个分子")
    print(f"  失败分析: {failed_analyses} 个分子")
    print(f"  总分子数: 10 个分子")
    print(f"  成功率: {100*successful_analyses/10:.1f}%")


if __name__ == "__main__":
    main()
