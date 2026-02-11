#!/usr/bin/env python3
"""
FIA49k新数据集转换脚本
处理CSV文件和JSON坐标文件，转换为EGNN训练格式
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import argparse
from tqdm import tqdm

def parse_xyz_content(xyz_content):
    """解析XYZ格式的坐标内容"""
    lines = xyz_content.strip().split('\n')
    n_atoms = int(lines[0])
    
    atoms = []
    coords = []
    
    for i in range(2, 2 + n_atoms):  # 跳过第一行(原子数)和第二行(注释)
        parts = lines[i].split()
        if len(parts) >= 4:
            atom = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(atom)
            coords.append([x, y, z])
    
    return atoms, np.array(coords)

def get_atomic_number(atom_symbol):
    """获取原子序数"""
    atomic_numbers = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'Al': 13
    }
    return atomic_numbers.get(atom_symbol, 0)

def load_json_coordinates(json_files):
    """加载JSON坐标文件"""
    all_coords = {}
    
    for json_file in json_files:
        print(f"Loading coordinates from {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for mol_id, mol_data in tqdm(data.items(), desc=f"Processing {json_file}"):
            # 优先使用la-PBEh3c坐标，如果没有则使用其他
            coords_data = None
            for key in ['la-PBEh3c', 'la-GFN2xTB', 'fa-PBEh3c', 'fa-GFN2xTB']:
                if key in mol_data:
                    coords_data = mol_data[key]
                    break
            
            if coords_data:
                try:
                    atoms, coords = parse_xyz_content(coords_data)
                    all_coords[mol_id] = {
                        'atoms': atoms,
                        'coords': coords
                    }
                except Exception as e:
                    print(f"Error parsing {mol_id}: {e}")
                    continue
    
    return all_coords

def load_csv_properties(csv_file):
    """加载CSV属性文件"""
    print(f"Loading properties from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 选择需要的列
    property_columns = [
        'Compound', 'fia_gas-DSDBLYP', 'fia_gas-PBEh3c', 
        'set_assignment', 'mol_type'
    ]
    
    # 检查列是否存在
    available_columns = [col for col in property_columns if col in df.columns]
    properties_df = df[available_columns].copy()
    
    print(f"Loaded {len(properties_df)} molecules with properties")
    print(f"Available columns: {available_columns}")
    
    return properties_df

def create_one_hot_encoding(atoms_list):
    """创建原子类型的一热编码"""
    unique_atoms = sorted(list(set(atoms_list)))
    atom_to_idx = {atom: i for i, atom in enumerate(unique_atoms)}
    
    return atom_to_idx, unique_atoms

def process_molecule(mol_id, atoms, coords, properties, atom_to_idx, max_atoms=100):
    """处理单个分子"""
    n_atoms = len(atoms)
    
    if n_atoms > max_atoms:
        return None  # 跳过过大的分子
    
    # 获取原子序数
    atomic_numbers = [get_atomic_number(atom) for atom in atoms]
    
    # 创建一热编码
    one_hot = np.zeros((max_atoms, len(atom_to_idx)))
    for i, atom in enumerate(atoms):
        if atom in atom_to_idx:
            one_hot[i, atom_to_idx[atom]] = 1
    
    # 填充坐标
    padded_coords = np.zeros((max_atoms, 3))
    padded_coords[:n_atoms] = coords
    
    # 填充原子序数
    padded_atomic_numbers = np.zeros(max_atoms)
    padded_atomic_numbers[:n_atoms] = atomic_numbers
    
    # 创建原子掩码
    atom_mask = np.zeros(max_atoms)
    atom_mask[:n_atoms] = 1
    
    # 获取性质值
    target_property = None
    if 'fia_gas-DSDBLYP' in properties.columns:
        target_property = properties[properties['Compound'] == mol_id]['fia_gas-DSDBLYP'].iloc[0]
    elif 'fia_gas-PBEh3c' in properties.columns:
        target_property = properties[properties['Compound'] == mol_id]['fia_gas-PBEh3c'].iloc[0]
    
    if target_property is None or pd.isna(target_property):
        return None
    
    return {
        'mol_id': mol_id,
        'positions': padded_coords,
        'one_hot': one_hot,
        'atomic_numbers': padded_atomic_numbers,
        'atom_mask': atom_mask,
        'n_atoms': n_atoms,
        'target': float(target_property)
    }

def main():
    parser = argparse.ArgumentParser(description='Convert FIA49k dataset to EGNN format')
    parser.add_argument('--csv_file', type=str, default='ae_datasets/FIA49k.csv',
                       help='Path to CSV properties file')
    parser.add_argument('--json_files', nargs='+', 
                       default=['ae_datasets/FIA49k_01.json', 'ae_datasets/FIA49k_02.json', 
                               'ae_datasets/FIA49k_03.json', 'ae_datasets/FIA49k_04.json'],
                       help='Paths to JSON coordinate files')
    parser.add_argument('--output_dir', type=str, default='fia49k_new_converted',
                       help='Output directory for converted data')
    parser.add_argument('--max_atoms', type=int, default=100,
                       help='Maximum number of atoms per molecule')
    parser.add_argument('--target_property', type=str, default='fia_gas-DSDBLYP',
                       help='Target property to predict')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("FIA49k新数据集转换")
    print("="*60)
    
    # 1. 加载CSV属性
    properties_df = load_csv_properties(args.csv_file)
    
    # 2. 加载JSON坐标
    coords_data = load_json_coordinates(args.json_files)
    
    print(f"Loaded coordinates for {len(coords_data)} molecules")
    
    # 3. 创建原子类型映射
    all_atoms = []
    for mol_data in coords_data.values():
        all_atoms.extend(mol_data['atoms'])
    
    atom_to_idx, unique_atoms = create_one_hot_encoding(all_atoms)
    print(f"Found {len(unique_atoms)} unique atom types: {unique_atoms}")
    
    # 4. 处理分子数据
    processed_molecules = []
    skipped_count = 0
    
    print("Processing molecules...")
    for mol_id in tqdm(coords_data.keys(), desc="Converting molecules"):
        if mol_id not in coords_data:
            skipped_count += 1
            continue
            
        mol_data = coords_data[mol_id]
        atoms = mol_data['atoms']
        coords = mol_data['coords']
        
        # 检查是否有对应的属性
        if mol_id not in properties_df['Compound'].values:
            skipped_count += 1
            continue
        
        processed_mol = process_molecule(
            mol_id, atoms, coords, properties_df, 
            atom_to_idx, args.max_atoms
        )
        
        if processed_mol is not None:
            processed_molecules.append(processed_mol)
        else:
            skipped_count += 1
    
    print(f"Processed {len(processed_molecules)} molecules")
    print(f"Skipped {skipped_count} molecules")
    
    # 5. 保存处理后的数据
    print("Saving processed data...")
    
    # 转换为PyTorch张量
    positions = torch.stack([torch.tensor(mol['positions'], dtype=torch.float32) 
                           for mol in processed_molecules])
    one_hot = torch.stack([torch.tensor(mol['one_hot'], dtype=torch.float32) 
                         for mol in processed_molecules])
    atomic_numbers = torch.stack([torch.tensor(mol['atomic_numbers'], dtype=torch.float32) 
                                for mol in processed_molecules])
    atom_masks = torch.stack([torch.tensor(mol['atom_mask'], dtype=torch.float32) 
                            for mol in processed_molecules])
    targets = torch.tensor([mol['target'] for mol in processed_molecules], dtype=torch.float32)
    n_atoms = torch.tensor([mol['n_atoms'] for mol in processed_molecules], dtype=torch.long)
    mol_ids = [mol['mol_id'] for mol in processed_molecules]
    
    # 保存数据
    data_dict = {
        'positions': positions,
        'one_hot': one_hot,
        'atomic_numbers': atomic_numbers,
        'atom_masks': atom_masks,
        'targets': targets,
        'n_atoms': n_atoms,
        'mol_ids': mol_ids,
        'atom_to_idx': atom_to_idx,
        'unique_atoms': unique_atoms
    }
    
    torch.save(data_dict, os.path.join(args.output_dir, 'fia49k_data.pt'))
    
    # 保存统计信息
    stats = {
        'total_molecules': len(processed_molecules),
        'skipped_molecules': skipped_count,
        'max_atoms': args.max_atoms,
        'unique_atoms': unique_atoms,
        'atom_to_idx': atom_to_idx,
        'target_property': args.target_property,
        'target_mean': float(targets.mean()),
        'target_std': float(targets.std()),
        'target_min': float(targets.min()),
        'target_max': float(targets.max())
    }
    
    with open(os.path.join(args.output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n转换完成!")
    print(f"输出目录: {args.output_dir}")
    print(f"数据文件: {os.path.join(args.output_dir, 'fia49k_data.pt')}")
    print(f"统计文件: {os.path.join(args.output_dir, 'stats.json')}")
    print(f"总分子数: {len(processed_molecules)}")
    print(f"跳过分子数: {skipped_count}")
    print(f"目标性质统计: 均值={stats['target_mean']:.3f}, 标准差={stats['target_std']:.3f}")

if __name__ == "__main__":
    main()
