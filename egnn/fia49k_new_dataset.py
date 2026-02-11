#!/usr/bin/env python3
"""
FIA49k新数据集类
用于加载和处理转换后的FIA49k数据
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional

class FIA49kNewDataset(Dataset):
    """FIA49k新数据集类"""
    
    def __init__(self, data_path: str, split: str = 'train', 
                 train_ratio: float = 0.8, val_ratio: float = 0.1,
                 property_name: str = 'fia_gas-DSDBLYP'):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            split: 数据集分割 ('train', 'val', 'test')
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            property_name: 目标性质名称
        """
        self.data_path = data_path
        self.split = split
        self.property_name = property_name
        
        # 加载数据
        self.data = torch.load(data_path, map_location='cpu')
        
        # 获取数据统计
        self.n_molecules = len(self.data['targets'])
        self.max_atoms = self.data['positions'].shape[1]
        self.n_atom_types = len(self.data['unique_atoms'])
        
        # 分割数据集
        self._split_dataset(train_ratio, val_ratio)
        
        print(f"Loaded {self.n_molecules} molecules")
        print(f"Split '{split}': {len(self.indices)} molecules")
        print(f"Max atoms: {self.max_atoms}, Atom types: {self.n_atom_types}")
    
    def _split_dataset(self, train_ratio: float, val_ratio: float):
        """分割数据集"""
        n_total = self.n_molecules
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # 随机打乱索引
        indices = torch.randperm(n_total)
        
        if self.split == 'train':
            self.indices = indices[:n_train]
        elif self.split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        actual_idx = self.indices[idx]
        
        return {
            'positions': self.data['positions'][actual_idx],
            'one_hot': self.data['one_hot'][actual_idx],
            'atomic_numbers': self.data['atomic_numbers'][actual_idx],
            'atom_mask': self.data['atom_masks'][actual_idx],
            'target': self.data['targets'][actual_idx],
            'n_atoms': self.data['n_atoms'][actual_idx],
            'mol_id': self.data['mol_ids'][actual_idx]
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """获取数据集统计信息"""
        targets = self.data['targets'][self.indices]
        return {
            'mean': float(targets.mean()),
            'std': float(targets.std()),
            'min': float(targets.min()),
            'max': float(targets.max())
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """批处理函数"""
    # 获取批次大小
    batch_size = len(batch)
    
    # 获取最大原子数
    max_atoms = max([item['n_atoms'].item() for item in batch])
    
    # 初始化批次张量
    positions = torch.zeros(batch_size, max_atoms, 3)
    one_hot = torch.zeros(batch_size, max_atoms, batch[0]['one_hot'].shape[1])
    atomic_numbers = torch.zeros(batch_size, max_atoms)
    atom_masks = torch.zeros(batch_size, max_atoms)
    targets = torch.zeros(batch_size)
    n_atoms = torch.zeros(batch_size, dtype=torch.long)
    mol_ids = []
    
    for i, item in enumerate(batch):
        n_atoms_i = item['n_atoms'].item()
        
        # 填充数据
        positions[i, :n_atoms_i] = item['positions'][:n_atoms_i]
        one_hot[i, :n_atoms_i] = item['one_hot'][:n_atoms_i]
        atomic_numbers[i, :n_atoms_i] = item['atomic_numbers'][:n_atoms_i]
        atom_masks[i, :n_atoms_i] = 1.0
        targets[i] = item['target']
        n_atoms[i] = n_atoms_i
        mol_ids.append(item['mol_id'])
    
    # 创建边掩码
    edge_masks = torch.zeros(batch_size, max_atoms, max_atoms)
    for i in range(batch_size):
        n_atoms_i = n_atoms[i].item()
        edge_masks[i, :n_atoms_i, :n_atoms_i] = 1.0
    
    return {
        'positions': positions,
        'one_hot': one_hot,
        'atomic_numbers': atomic_numbers,
        'atom_mask': atom_masks,
        'edge_mask': edge_masks,
        'targets': targets,
        'n_atoms': n_atoms,
        'mol_ids': mol_ids
    }

def create_fia49k_new_dataloaders(data_path: str, 
                                 batch_size: int = 32,
                                 num_workers: int = 0,
                                 train_ratio: float = 0.8,
                                 val_ratio: float = 0.1,
                                 property_name: str = 'fia_gas-DSDBLYP') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建FIA49k新数据加载器
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        num_workers: 工作进程数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        property_name: 目标性质名称
    
    Returns:
        训练、验证、测试数据加载器
    """
    # 创建数据集
    train_dataset = FIA49kNewDataset(data_path, 'train', train_ratio, val_ratio, property_name)
    val_dataset = FIA49kNewDataset(data_path, 'val', train_ratio, val_ratio, property_name)
    test_dataset = FIA49kNewDataset(data_path, 'test', train_ratio, val_ratio, property_name)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

def test_dataset():
    """测试数据集功能"""
    print("Testing FIA49kNewDataset...")
    
    # 假设数据文件存在
    data_path = 'fia49k_new_converted/fia49k_data.pt'
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run the converter first.")
        return
    
    try:
        # 创建数据集
        dataset = FIA49kNewDataset(data_path, 'train')
        
        # 测试单个样本
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Positions shape: {sample['positions'].shape}")
        print(f"One hot shape: {sample['one_hot'].shape}")
        print(f"Target: {sample['target']}")
        
        # 测试数据加载器
        train_loader, val_loader, test_loader = create_fia49k_new_dataloaders(
            data_path, batch_size=4
        )
        
        # 测试批次
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch positions shape: {batch['positions'].shape}")
        print(f"Batch targets: {batch['targets']}")
        
        print("Dataset test passed!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")

if __name__ == "__main__":
    test_dataset()
