import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def get_neighbors(p_pos, all_pos, z_types, cutoff=2.0):
    """
    Identify bonded neighbors based on distance.
    Returns (indices, positions, atomic_numbers) of neighbors.
    """
    dists = cdist([p_pos], all_pos).flatten()
    # Exclude self (distance ~ 0)
    neighbor_mask = (dists < cutoff) & (dists > 0.1)
    return np.where(neighbor_mask)[0], all_pos[neighbor_mask], z_types[neighbor_mask]

def calculate_angle_sum(p_pos, neighbor_positions):
    """
    Calculate sum of bond angles around P.
    """
    if len(neighbor_positions) < 2:
        return 0.0
    
    angles = []
    num_neighbors = len(neighbor_positions)
    
    for i in range(num_neighbors):
        for j in range(i + 1, num_neighbors):
            v1 = neighbor_positions[i] - p_pos
            v2 = neighbor_positions[j] - p_pos
            
            val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            val = np.clip(val, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(val)))
            
    return sum(angles)

def calculate_lp_plane_alignment(p_pos, p_idx, neighbor_indices, mol_coords, mol_z):
    """
    Calculate alignment angles between P's lone pair and neighbors' conjugated planes.
    Returns list of angles (degrees) between LP vector and Neighbor Plane (0=In Plane, 90=Perpendicular).
    """
    neigh_pos = mol_coords[neighbor_indices]
    
    # 1. Estimate Lone Pair Vector pointing "out" from P
    # For pyramidal P, sum of bond vectors points away from LP.
    bond_vectors = [normalize(pos - p_pos) for pos in neigh_pos]
    sum_vec = np.sum(bond_vectors, axis=0)
    lp_vec = normalize(-sum_vec) # Points "up"
    
    alignments = []
    
    for n_idx in neighbor_indices:
        # Check if neighbor is capable of conjugation (C, N, maybe S/O if sp2)
        # We need its neighbors to define a plane
        n_pos = mol_coords[n_idx]
        nn_idxs, nn_pos, _ = get_neighbors(n_pos, mol_coords, mol_z, cutoff=2.0)
        
        # Filter out P itself from neighbors of neighbor
        other_nn = [pos for k, pos in enumerate(nn_pos) if nn_idxs[k] != p_idx]
        
        # We need at least 2 other neighbors to define a stable plane (indicating sp2/planar env)
        if len(other_nn) >= 2:
            v1 = other_nn[0] - n_pos
            v2 = other_nn[1] - n_pos
            
            # Plane normal
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) < 1e-3: continue # Collinear or degenerate
            normal = normalize(normal)
            
            # Angle between LP and Normal
            # We want angle between LP and Plane.
            # dot = cos(theta_normal)
            # theta_plane = 90 - theta_normal
            dot = np.abs(np.dot(lp_vec, normal))
            val = np.clip(dot, 0.0, 1.0)
            angle_with_normal = np.degrees(np.arccos(val)) # 0 to 90
            angle_with_plane = 90.0 - angle_with_normal # 0 means LP is IN plane
            
            alignments.append({
                'n_idx': n_idx,
                'n_z': mol_z[n_idx],
                'angle_with_plane': angle_with_plane
            })
            
    return alignments

# Load predictions
dimenet_df = pd.read_csv('dimenet_sampled_predictions.csv')
true_values = dimenet_df.set_index('mol_id')['true_value'].to_dict()
dimenet_preds = dimenet_df.set_index('mol_id')['predicted_value'].to_dict()

# Load structure data
data_path = 'dimenet/data/sampled_2000.npz'
data = np.load(data_path, allow_pickle=True)

R = data['R']
Z = data['Z']
N = data['N']
ids = data['id']

print(f"Loaded {len(N)} molecules structures.")

p_analysis_results = []

cursor = 0
for i, atom_count in enumerate(N):
    mol_id = ids[i] if isinstance(ids[i], (int, str)) else ids[i][0]
    
    mol_coords = R[cursor:cursor+atom_count]
    mol_z = Z[cursor:cursor+atom_count]
    cursor += atom_count
    
    if mol_id not in dimenet_preds: continue
        
    p_indices = np.where(mol_z == 15)[0]
    
    for p_idx in p_indices:
        p_pos = mol_coords[p_idx]
        neigh_idxs, neigh_pos, neigh_z = get_neighbors(p_pos, mol_coords, mol_z)
        
        if len(neigh_idxs) == 3:
            angle_sum = calculate_angle_sum(p_pos, neigh_pos)
            
            # Check alignment with neighbors
            alignments = calculate_lp_plane_alignment(p_pos, p_idx, neigh_idxs, mol_coords, mol_z)
            
            # Get minimum angle to plane (look for the most coplanar interaction)
            min_angle_plane = min([a['angle_with_plane'] for a in alignments]) if alignments else np.nan
            
            p_analysis_results.append({
                'mol_id': mol_id,
                'true_fia': true_values[mol_id],
                'pred_fia': dimenet_preds[mol_id],
                'error': dimenet_preds[mol_id] - true_values[mol_id],
                'angle_sum': angle_sum,
                'min_angle_plane': min_angle_plane,
                'num_conjugated_neighbors': len(alignments),
                'neighbor_signature': str(sorted([int(z) for z in neigh_z]))
            })

df = pd.DataFrame(p_analysis_results)

if not df.empty:
    print(f"Analyzed {len(df)} trivalent P centers.\n")
    
    # ========== TEST: MODALITY COMPETITION HYPOTHESIS ==========
    # If model over-relies on atom types (Z) and ignores geometry (angles),
    # we should see:
    # 1. Same neighbor types -> Similar predictions (low variance in pred_fia)
    # 2. Same neighbor types -> No correlation between angle_sum and pred_fia
    
    print("="*70)
    print("MODALITY COMPETITION ANALYSIS")
    print("="*70)
    
    # Group by neighbor types
    grouped = df.groupby('neighbor_signature')
    
    print(f"\nFound {len(grouped)} unique neighbor type combinations.\n")
    
    # For groups with >3 samples, check if geometry matters
    large_groups = [(name, group) for name, group in grouped if len(group) >= 3]
    
    if large_groups:
        print("--- Groups with ≥3 samples (Testing if Geometry affects Prediction) ---\n")
        
        modality_evidence = []
        
        for neighbor_sig, group in large_groups:
            # Within same atom types, does angle_sum correlate with prediction?
            corr_pred = group['angle_sum'].corr(group['pred_fia'])
            corr_true = group['angle_sum'].corr(group['true_fia'])
            
            # Variance in predictions vs variance in true values
            pred_std = group['pred_fia'].std()
            true_std = group['true_fia'].std()
            angle_range = group['angle_sum'].max() - group['angle_sum'].min()
            
            modality_evidence.append({
                'neighbors': neighbor_sig,
                'count': len(group),
                'angle_range': angle_range,
                'corr_angle_pred': corr_pred,
                'corr_angle_true': corr_true,
                'pred_std': pred_std,
                'true_std': true_std
            })
            
            print(f"Neighbors: {neighbor_sig} (n={len(group)})")
            print(f"  Angle Range: {angle_range:.1f}°")
            print(f"  Corr(Angle, Prediction): {corr_pred:.3f}")
            print(f"  Corr(Angle, True FIA):   {corr_true:.3f}")
            print(f"  Prediction Std: {pred_std:.1f}, True Std: {true_std:.1f}")
            
            if abs(corr_true) > 0.5 and abs(corr_pred) < 0.3:
                print(f"  ⚠️  MODALITY ISSUE: Model ignores geometry that strongly affects FIA!")
            print()
        
        # Overall statistics
        me_df = pd.DataFrame(modality_evidence)
        me_df = me_df[me_df['angle_range'] > 10]  # Only meaningful geometric variation
        
        if not me_df.empty:
            print("\n--- OVERALL MODALITY COMPETITION SCORE ---")
            print(f"Average |Corr(Angle, True)|:       {me_df['corr_angle_true'].abs().mean():.3f}")
            print(f"Average |Corr(Angle, Prediction)|: {me_df['corr_angle_pred'].abs().mean():.3f}")
            
            ratio = me_df['corr_angle_pred'].abs().mean() / (me_df['corr_angle_true'].abs().mean() + 1e-6)
            print(f"Geometry Sensitivity Ratio (Pred/True): {ratio:.3f}")
            print("  (Ratio < 0.5 suggests model under-utilizes geometry)\n")
    
    # Additional: Check prediction variance WITHIN vs BETWEEN neighbor types
    within_var = grouped['pred_fia'].std().mean()
    between_var = grouped['pred_fia'].mean().std()
    
    print(f"Prediction Variance WITHIN same atom types: {within_var:.2f}")
    print(f"Prediction Variance BETWEEN atom types:     {between_var:.2f}")
    print(f"Variance Ratio (Between/Within): {between_var/within_var:.2f}")
    print("  (High ratio suggests model mainly uses atom types, not geometry)\n")
    
    df.to_csv('phosphorus_modality_analysis.csv', index=False)
    print("\nSaved analysis to phosphorus_modality_analysis.csv")
else:
    print("No data.")
