"""Exploratory chemistry diagnostics of frozen predictions.

All test molecules are retained. Thresholds come from the training partition;
paired structures are diagnostic outcomes, never predictor inputs. Numerical
exports are small enough to inspect without restoring the trained models.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/chemistry'
TARGET = 'fia_gas-DSDBLYP'
MODELS = {
    'FIA-GNN': 'public_reference/fia_gnn_gas_{split}_predictions.csv',
    'DimeNet++': 'neural_predictions/dimenet_{split}.csv',
    'EGNN': 'neural_predictions/egnn-sum_{split}.csv',
    'Radial': 'neural_predictions/radial-site_{split}.csv',
    'Acid QM': 'neural_predictions/acid-qm_{split}.csv',
    'Refinement': 'refinement/delta_RDKit_LA_QM_{split}.csv',
    'xTB radial': 'geometry_shift/ensemble_{split}.csv',
}
CASE_IDS = ['B_3_ZT009074', 'P_3_ZT017132', 'As_3_ZT018788',
            'As_3_ZT018778', 'P_3_B014413', 'P_3_B014420',
            'P_3_B014554', 'P_3_ZB016799', 'P_3_ZT026756']


def central_angles(z, r, binding_info):
    """Assign first-shell atoms by reported element counts and distance.

    An element-matched nearest-neighbor assignment avoids a single absolute
    bond-length threshold across the periodic table. The maximum normalized
    distance is exported for independent inspection of this assignment.
    """
    pt = Chem.GetPeriodicTable()
    expected = Counter()
    for sym, count in re.findall(r'([A-Z][a-z]?)(?:\([^)]*\))?-(\d+)', binding_info):
        expected[pt.GetAtomicNumber(sym)] += int(count)
    if sum(expected.values()) != 3:
        return None
    chosen = []
    for number, count in expected.items():
        candidates = np.flatnonzero(z == number)
        candidates = candidates[candidates != 0]
        distances = np.linalg.norm(r[candidates] - r[0], axis=1)
        chosen.extend(candidates[np.argsort(distances)[:count]])
    if len(chosen) != 3:
        raise ValueError('Recorded first-shell element counts cannot be assigned')
    v = r[chosen] - r[0]
    distances = np.linalg.norm(v, axis=1)
    if np.any(distances <= 0):
        raise ValueError('Coincident first-shell atoms')
    u = v / distances[:, None]
    angles = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))[np.triu_indices(3, 1)]
    ratio = [distances[k] / (pt.GetRcovalent(int(z[0])) + pt.GetRcovalent(int(z[j])))
             for k, j in enumerate(chosen)]
    return dict(angle_sum=float(angles.sum()), min_angle=float(angles.min()),
                max_bond_radius_ratio=float(max(ratio)),
                neighbor_indices=';'.join(str(j) for j in chosen))


def interval(values, groups=None, seed=20261114, repeats=2000):
    """Mean and central-class-stratified molecular bootstrap interval."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires nonempty finite values')
    labels = np.zeros(len(values), dtype=int) if groups is None else np.asarray(groups)
    strata = [values[labels == k] for k in np.unique(labels)]
    rng = np.random.default_rng(seed)
    samples = np.zeros(repeats)
    for v in strata:
        samples += np.array([rng.choice(v, len(v), replace=True).sum()
                             for _ in range(repeats)]) / len(values)
    return float(values.mean()), *np.quantile(samples, [.025, .975]).tolist()


def attach_predictions(data, split):
    data = data.copy()
    for name, path in MODELS.items():
        p = pd.read_csv(ROOT/'reports'/path.format(split=split)).set_index('mol_id')
        if set(p.index) != set(data.index) or not p.index.is_unique:
            raise ValueError(f'Molecule mismatch: {name}, {split}')
        p = p.loc[data.index]
        if not np.allclose(p.y_true, data[TARGET], atol=1e-7, rtol=0):
            raise ValueError(f'Target mismatch: {name}, {split}')
        data[name] = p.y_pred
        data[name + '_error'] = p.y_pred - data[TARGET]
        data[name + '_ae'] = abs(data[name + '_error'])
    return data


def summarize(data, category, include_gain=True):
    rows = []
    for key, group in data.groupby(category, observed=True, sort=False):
        for model in MODELS:
            m, lo, hi = interval(group[model+'_ae'], group.ca_class)
            rows.append(dict(group=str(key), model=model, n=len(group), mae=m,
                             ci95_low=lo, ci95_high=hi,
                             mean_signed_error=group[model+'_error'].mean()))
        if include_gain:
            gain = group['FIA-GNN_ae']-group['DimeNet++_ae']
            m, lo, hi = interval(gain, group.ca_class)
            rows.append(dict(group=str(key), model='3D gain over FIA-GNN', n=len(group),
                             mae=m, ci95_low=lo, ci95_high=hi, mean_signed_error=np.nan))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT/'.data/FIA49k.csv.gz', low_memory=False).set_index('Compound')
    geom = pd.read_csv(ROOT/'reports/geometry_diagnostics.csv').set_index('Compound')
    data = data.join(geom, validate='one_to_one')
    with np.load(ROOT/'.data/la-PBEh3c.npz', allow_pickle=False) as p:
        ids, ns, zs, rs = [p[k] for k in ('mol_id', 'N', 'Z', 'R')]
    offsets = np.r_[0, np.cumsum(ns)]
    shape_rows = []
    for i, mid in enumerate(ids):
        sl = slice(offsets[i], offsets[i+1])
        shape = central_angles(zs[sl], rs[sl].astype(float),
                               data.loc[mid, 'atoms_binding_to_central_atom_info'])
        if shape is not None:
            shape_rows.append(dict(Compound=mid, **shape))
    shape = pd.DataFrame(shape_rows).set_index('Compound')
    if shape.max_bond_radius_ratio.max() >= 1.35:
        raise ValueError('Inspect first-shell assignments before analysis')
    shape.to_csv(OUT/'center_geometry.csv')
    data = data.join(shape, validate='one_to_one')
    train = data.loc[data.set_assignment.eq('train')]
    thresholds = {
        'reorganization': train.heavy_pair_distance_rms_change.quantile([.25, .5, .75, .9]).tolist(),
        'geometry_source': train.geometry_source_pair_rms.quantile([.5, .75, .9]).tolist(),
        'phosphorus_angles': train.loc[train.ca_class.eq('P(III)'), 'angle_sum'].quantile([.25, .5, .75]).tolist(),
    }
    test = attach_predictions(data.loc[data.set_assignment.eq('test')], 'test')
    summarize(test, 'ca_class').to_csv(OUT/'by_center.csv', index=False)
    summarize(test, 'denticity_class').to_csv(OUT/'by_denticity.csv', index=False)
    test['reorganization_bin'] = pd.cut(test.heavy_pair_distance_rms_change,
        [-np.inf]+thresholds['reorganization']+[np.inf], labels=['Q1', 'Q2', 'Q3', 'P75–90', 'P90–100'])
    summarize(test, 'reorganization_bin').to_csv(OUT/'by_reorganization.csv', index=False)
    test['source_bin'] = pd.cut(test.geometry_source_pair_rms,
        [-np.inf]+thresholds['geometry_source']+[np.inf], labels=['P0–50', 'P50–75', 'P75–90', 'P90–100'])
    source = summarize(test, 'source_bin', include_gain=False)
    source.to_csv(OUT/'by_geometry_source.csv', index=False)
    shift_rows = []
    for key, group in test.groupby('source_bin', observed=True, sort=False):
        m, lo, hi = interval(group['xTB radial_ae']-group['Radial_ae'], group.ca_class)
        shift_rows.append(dict(group=str(key), n=len(group), mae_increase=m,
                               ci95_low=lo, ci95_high=hi))
    pd.DataFrame(shift_rows).to_csv(OUT/'geometry_shift_penalty.csv', index=False)
    # Fixed within-center class standardization separates composition from the
    # source-shift contrast. Each center receives the same weight in both bins.
    test['source_tail'] = test.geometry_source_pair_rms > thresholds['geometry_source'][-1]
    penalty = test['xTB radial_ae']-test['Radial_ae']
    cells = test.assign(penalty=penalty).groupby(['ca_class', 'source_tail']).penalty.agg(['mean', 'size'])
    cells.to_csv(OUT/'geometry_shift_by_center.csv')
    valid = cells['mean'].unstack().dropna()
    standard = valid.mean().to_dict()
    ptest = test.loc[test.ca_class.eq('P(III)')].copy()
    ptest['angle_quartile'] = pd.cut(ptest.angle_sum,
        [-np.inf]+thresholds['phosphorus_angles']+[np.inf], labels=['Q1', 'Q2', 'Q3', 'Q4'])
    summarize(ptest, 'denticity_class').to_csv(OUT/'phosphorus_denticity.csv', index=False)
    summarize(ptest, 'angle_quartile').to_csv(OUT/'phosphorus_angles.csv', index=False)
    # Retain the full diagnostic table for reproducible exploration.
    cols = ['ca_class', 'denticity_class', 'angle_sum', 'heavy_pair_distance_rms_change',
            'geometry_source_pair_rms', 'source_bin', 'reorganization_bin', TARGET]
    cols += [m+s for m in MODELS for s in ('', '_error', '_ae')]
    test[cols].to_csv(OUT/'test_diagnostics.csv.gz', compression='gzip', index_label='Compound')
    cases = data.loc[CASE_IDS].copy()
    for model in MODELS:
        cases[model] = test[model].reindex(cases.index)
        # Validation examples use validation predictions, never training fits.
        val = cases.loc[cases.set_assignment.eq('validate')]
        val_path = ROOT/'reports'/MODELS[model].format(split='validate')
        if len(val) and val_path.exists():
            vp = pd.read_csv(val_path).set_index('mol_id')
            cases.loc[val.index, model] = vp.y_pred.reindex(val.index)
    case_cols = ['la_smiles', 'ca_class', 'set_assignment', 'ligand_info', TARGET,
                 'la-ca_mulliken_charge-PBEh3c', 'la-lumo_energy-PBEh3c',
                 'la-global_electrophilicity-PBEh3c',
                 'angle_sum', 'neighbor_indices', 'heavy_pair_distance_rms_change',
                 'center_near_distance_rms_change', 'added_f_distance'] + list(MODELS)
    cases[case_cols].to_csv(OUT/'molecular_cases.csv', index_label='Compound')
    manifest = dict(analysis='Exploratory diagnostics of fixed models, after model fitting',
        n_test=len(test), n_centers=len(test.ca_class.unique()),
        thresholds_from='Official training partition only', thresholds=thresholds,
        angle_assignment='Nearest atoms of each element matching recorded first-shell composition; atom 0 is center',
        n_angle_assignments=len(shape), max_bond_radius_ratio=float(shape.max_bond_radius_ratio.max()),
        intervals='2000 molecule bootstrap resamples, stratified by central class, seed 20261114; fixed models',
        geometry_shift_class_standardized={'non_tail': standard[False], 'tail': standard[True], 'n_classes': len(valid)},
        scope='Association and representation diagnostics; structural displacement is distinct from deformation energy; paired adduct geometry is diagnostic only',
        case_selection='Nine structurally diverse illustrations; all original selected identifiers retained; memberships explicit; no training predictions displayed')
    (OUT/'protocol.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
