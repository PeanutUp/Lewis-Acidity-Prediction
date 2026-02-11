#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理FIA49k和FIA49k_Al数据集，提取分子坐标和性质，生成与qm9_eV.npz相同格式的NPZ文件
"""

import numpy as np
import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Any

def parse_xyz_coordinates(xyz_string: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    解析XYZ格式的分子坐标字符串
    
    Args:
        xyz_string: XYZ格式的字符串
        
    Returns:
        tuple: (原子序数数组, 坐标数组)
    """
    lines = xyz_string.strip().split('\n')
    
    # 跳过第一行（原子数）和第二行（注释行）
    coord_lines = [line for line in lines[2:] if line.strip()]
    
    atomic_numbers = []
    coordinates = []
    
    # 原子序数映射
    atom_map = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'Al': 13
    }
    
    for line in coord_lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            atom_symbol = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            atomic_numbers.append(atom_map.get(atom_symbol, 0))
            coordinates.append([x, y, z])
    
    return np.array(atomic_numbers), np.array(coordinates)

def extract_molecule_data_from_fia49k_al(data_al: Dict) -> Dict[str, Dict]:
    """
    从FIA49k_Al数据中提取分子坐标和原子序数
    
    Args:
        data_al: FIA49k_Al JSON数据
        
    Returns:
        dict: 分子ID到分子数据的映射
    """
    molecules = {}
    
    for mol_id, mol_data in data_al.items():
        # 提取la-PBEh3c的坐标数据
        if 'la-PBEh3c' in mol_data:
            xyz_string = mol_data['la-PBEh3c']
            try:
                atomic_numbers, coordinates = parse_xyz_coordinates(xyz_string)
                molecules[mol_id] = {
                    'atomic_numbers': atomic_numbers,
                    'coordinates': coordinates,
                    'num_atoms': len(atomic_numbers)
                }
            except Exception as e:
                print(f"解析分子 {mol_id} 时出错: {e}")
                continue
    
    return molecules

def match_molecules(fia49k_df: pd.DataFrame, molecules_al: Dict) -> List[Dict]:
    """
    匹配FIA49k和FIA49k_Al中的分子
    
    Args:
        fia49k_df: FIA49k DataFrame
        molecules_al: FIA49k_Al中的分子数据
        
    Returns:
        list: 匹配的分子数据列表
    """
    matched_molecules = []
    
    for idx, row in fia49k_df.iterrows():
        compound_name = row['Compound']
        
        # 尝试不同的匹配方式
        possible_ids = [
            compound_name,
            f"Al_3_{compound_name}",
            compound_name.replace('_', '_3_')
        ]
        
        matched_mol = None
        for possible_id in possible_ids:
            if possible_id in molecules_al:
                matched_mol = molecules_al[possible_id]
                break
        
        if matched_mol is not None:
            matched_molecules.append({
                'compound': compound_name,
                'mol_id': compound_name,
                'atomic_numbers': matched_mol['atomic_numbers'],
                'coordinates': matched_mol['coordinates'],
                'num_atoms': matched_mol['num_atoms'],
                'fia_gas_dsdblyp': row['fia_gas-DSDBLYP']
            })
        else:
            print(f"未找到匹配的分子: {compound_name}")
    
    return matched_molecules

def create_npz_data(matched_molecules: List[Dict]) -> Dict[str, np.ndarray]:
    """
    创建与qm9_eV.npz相同格式的数据
    
    Args:
        matched_molecules: 匹配的分子数据列表
        
    Returns:
        dict: NPZ格式的数据字典
    """
    num_molecules = len(matched_molecules)
    
    # 计算累积原子数
    N = np.array([mol['num_atoms'] for mol in matched_molecules], dtype=np.int32)
    N_cumsum = np.concatenate([[0], np.cumsum(N)])
    
    # 初始化数组
    total_atoms = np.sum(N)
    R = np.zeros((total_atoms, 3), dtype=np.float32)
    Z = np.zeros(total_atoms, dtype=np.int32)
    id_array = np.zeros(num_molecules, dtype=np.int64)
    fia_gas_dsdblyp = np.zeros(num_molecules, dtype=np.float64)
    
    # 填充数据
    for i, mol in enumerate(matched_molecules):
        start_idx = N_cumsum[i]
        end_idx = N_cumsum[i + 1]
        
        # 坐标和原子序数
        R[start_idx:end_idx] = mol['coordinates']
        Z[start_idx:end_idx] = mol['atomic_numbers']
        
        # 分子ID和性质
        id_array[i] = i  # 使用索引作为ID
        fia_gas_dsdblyp[i] = mol['fia_gas_dsdblyp']
    
    # 创建与qm9_eV.npz相同格式的数据
    npz_data = {
        'R': R,
        'Z': Z,
        'N': N,
        'id': id_array,
        'fia_gas-DSDBLYP': fia_gas_dsdblyp,
        'meta': None  # 元数据
    }
    
    return npz_data

def main():
    """主函数"""
    print("开始处理FIA49k数据集...")
    
    # 1. 读取FIA49k数据集
    print("读取FIA49k数据集...")
    fia49k_df = pd.read_excel('data/FIA49k.xlsx')
    print(f"FIA49k数据集大小: {len(fia49k_df)}")
    
    # 2. 读取FIA49k_Al数据集
    print("读取FIA49k_Al数据集...")
    with open('data/FIA49k_Al.json', 'r') as f:
        data_al = json.load(f)
    print(f"FIA49k_Al数据集大小: {len(data_al)}")
    
    # 3. 提取FIA49k_Al中的分子坐标
    print("提取分子坐标...")
    molecules_al = extract_molecule_data_from_fia49k_al(data_al)
    print(f"成功提取 {len(molecules_al)} 个分子的坐标数据")
    
    # 4. 匹配分子
    print("匹配分子...")
    matched_molecules = match_molecules(fia49k_df, molecules_al)
    print(f"成功匹配 {len(matched_molecules)} 个分子")
    
    if len(matched_molecules) == 0:
        print("没有匹配到任何分子，请检查数据格式")
        return
    
    # 5. 创建NPZ数据
    print("创建NPZ格式数据...")
    npz_data = create_npz_data(matched_molecules)
    
    # 6. 保存NPZ文件
    output_file = 'data/fia49k_processed.npz'
    print(f"保存数据到 {output_file}...")
    np.savez(output_file, **npz_data)
    
    # 7. 验证数据
    print("\n数据验证:")
    print(f"分子数量: {len(npz_data['id'])}")
    print(f"总原子数: {len(npz_data['Z'])}")
    print(f"坐标数组形状: {npz_data['R'].shape}")
    print(f"原子序数数组形状: {npz_data['Z'].shape}")
    print(f"每个分子原子数: {npz_data['N'][:10]}...")
    print(f"前5个分子的fia_gas-DSDBLYP值: {npz_data['fia_gas-DSDBLYP'][:5]}")
    
    print(f"\n处理完成！数据已保存到 {output_file}")

if __name__ == "__main__":
    main()
