#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理完整的FIA49k数据集（包括01-05五个部分），提取分子坐标和性质，生成NPZ文件
"""

import numpy as np
import pandas as pd
import json
import os
import gc
from typing import Dict, List, Tuple, Any, Set

# 全局原子映射，会在处理过程中更新
ATOM_MAP = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63,
    'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86
}

# 用于记录已经警告过的未知原子，避免重复打印
seen_unknown_atoms = set()

def parse_xyz_coordinates(xyz_string: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    解析XYZ格式的分子坐标字符串
    """
    lines = xyz_string.strip().split('\n')
    
    # 跳过第一行（原子数）和第二行（注释行）
    # 注意：有时候 XYZ string 可能包含前导/尾随空白，需要小心处理
    coord_lines = [line for line in lines[2:] if line.strip()]
    
    atomic_numbers = []
    coordinates = []
    
    for line in coord_lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            atom_symbol = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                if atom_symbol not in ATOM_MAP:
                    if atom_symbol not in seen_unknown_atoms:
                        print(f"警告: 发现未知原子 {atom_symbol} (后续将不再针对此原子重复警告)")
                        seen_unknown_atoms.add(atom_symbol)
                    continue

                atomic_numbers.append(ATOM_MAP.get(atom_symbol, 0))
                coordinates.append([x, y, z])
            except ValueError:
                continue
    
    return np.array(atomic_numbers), np.array(coordinates)

def process_json_file(json_path: str, molecule_buffer: Dict[str, Any]):
    """
    读取单个JSON文件并将解析后的分子数据存入缓冲区
    """
    print(f"正在读取 {os.path.basename(json_path)} ...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        print(f"  包含 {len(data)} 个条目，开始解析...")
        count = 0
        for mol_id, mol_data in data.items():
            if 'la-PBEh3c' in mol_data:
                try:
                    atomic_numbers, coordinates = parse_xyz_coordinates(mol_data['la-PBEh3c'])
                    if len(atomic_numbers) > 0:
                        molecule_buffer[mol_id] = {
                            'atomic_numbers': atomic_numbers,
                            'coordinates': coordinates,
                            'num_atoms': len(atomic_numbers)
                        }
                        count += 1
                except Exception as e:
                    print(f"  解析分子 {mol_id} 失败: {e}")
        
        print(f"  成功提取 {count} 个分子数据")
        del data
        gc.collect()
        
    except Exception as e:
        print(f"读取文件 {json_path} 失败: {e}")

def match_and_create_dataset(fia49k_df: pd.DataFrame, molecules_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    匹配Excel数据和提取的分子坐标，生成最终数据集
    """
    print("开始匹配分子...")
    matched_molecules = []
    missing_count = 0
    
    for idx, row in fia49k_df.iterrows():
        compound_name = str(row['Compound']).strip()
        
        # 尝试匹配ID
        possible_ids = [
            compound_name,
            f"Al_3_{compound_name}",
            compound_name.replace('_', '_3_'),
            f"Al_{compound_name}" # 尝试其他可能的变体
        ]
        
        matched_mol = None
        for possible_id in possible_ids:
            if possible_id in molecules_data:
                matched_mol = molecules_data[possible_id]
                break
        
        if matched_mol is not None:
            # 获取目标值
            target_val = row.get('fia_gas-DSDBLYP', np.nan)
            if pd.isna(target_val):
                # 如果目标值缺失，可以跳过或者用其他方式处理
                # 这里暂时跳过
                continue

            matched_molecules.append({
                'id': idx, # 使用Excel的行索引作为ID
                'atomic_numbers': matched_mol['atomic_numbers'],
                'coordinates': matched_mol['coordinates'],
                'num_atoms': matched_mol['num_atoms'],
                'target': target_val
            })
        else:
            missing_count += 1
            if missing_count <= 10: # 只打印前10个缺失的
                print(f"  未找到匹配: {compound_name} (尝试了: {possible_ids})")

    print(f"匹配完成: 成功 {len(matched_molecules)} 个, 失败 {missing_count} 个")
    
    if len(matched_molecules) == 0:
        return None

    # 构建NPZ数据结构
    num_molecules = len(matched_molecules)
    N = np.array([m['num_atoms'] for m in matched_molecules], dtype=np.int32)
    N_cumsum = np.concatenate([[0], np.cumsum(N)])
    total_atoms = np.sum(N)
    
    R = np.zeros((total_atoms, 3), dtype=np.float32)
    Z = np.zeros(total_atoms, dtype=np.int32)
    id_array = np.zeros(num_molecules, dtype=np.int64)
    target_array = np.zeros(num_molecules, dtype=np.float64)
    
    for i, mol in enumerate(matched_molecules):
        start = N_cumsum[i]
        end = N_cumsum[i+1]
        R[start:end] = mol['coordinates']
        Z[start:end] = mol['atomic_numbers']
        id_array[i] = mol['id']
        target_array[i] = mol['target']
        
    return {
        'R': R,
        'Z': Z,
        'N': N,
        'id': id_array,
        'fia_gas-DSDBLYP': target_array,
        'meta': None
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # 1. 读取 CSV 文件 (从 Excel 改为 CSV)
    csv_path = os.path.join(data_dir, 'FIA49k.csv')
    print(f"读取 CSV 文件: {csv_path}")
    if not os.path.exists(csv_path):
        print("错误: 找不到 FIA49k.csv")
        return
    # 尝试不同的编码，以防中文路径或特殊字符问题
    try:
        fia49k_df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("utf-8 读取失败，尝试 gb18030...")
        fia49k_df = pd.read_csv(csv_path, encoding='gb18030')
        
    print(f"CSV 包含 {len(fia49k_df)} 行数据")
    
    # 2. 读取所有 JSON 文件
    json_files = [
        'FIA49k_01.json', 'FIA49k_02.json', 'FIA49k_03.json', 
        'FIA49k_04.json', 'FIA49k_05.json', 'FIA49k_Al.json'
    ]
    
    all_molecules = {}
    
    for json_file in json_files:
        path = os.path.join(data_dir, json_file)
        if os.path.exists(path):
            process_json_file(path, all_molecules)
        else:
            print(f"警告: 找不到文件 {json_file}")
            
    print(f"总共加载了 {len(all_molecules)} 个唯一的分子结构")
    
    # 3. 匹配并生成数据
    npz_data = match_and_create_dataset(fia49k_df, all_molecules)
    
    if npz_data:
        output_path = os.path.join(data_dir, 'fia49k_full_processed.npz')
        print(f"保存数据到 {output_path} ...")
        np.savez(output_path, **npz_data)
        
        print("\n数据概览:")
        print(f"  分子数量: {len(npz_data['id'])}")
        print(f"  原子总数: {len(npz_data['Z'])}")
        print(f"  保存成功！")
    else:
        print("错误: 未能生成有效数据")

if __name__ == '__main__':
    main()
