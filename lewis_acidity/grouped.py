"""Canonical ligand-set holdout, distinct from the published FIA49k partition.

The complete disconnected ligand SMILES is canonicalized with stereochemistry;
the central element is deliberately excluded from the grouping key. A stable
seeded SHA256 hash assigns whole groups (70/15/15). Individual ligands may still
be shared: this is NOT an unseen-individual-ligand or generic-scaffold split.
Only original train/validate/test records enter this secondary experiment;
the five external test groups are reserved for the primary benchmark.
"""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .tabular import CATEGORICAL, META, QM, TARGET, stage
from .metrics import metrics

ROOT = Path(__file__).resolve().parents[1]


def group_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f'Unparseable ligand set: {smiles}')
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


def assignment(key):
    value = int(hashlib.sha256(('20261113|'+key).encode()).hexdigest()[:16], 16)/2**64
    return 'train' if value < .70 else ('validate' if value < .85 else 'test')


def scaffold_key(smiles):
    parts = []
    for fragment in Chem.GetMolFrags(Chem.MolFromSmiles(smiles), asMols=True):
        fragment = Chem.RemoveHs(fragment)
        scaffold = MurckoScaffold.GetScaffoldForMol(fragment)
        cyclic = scaffold.GetNumAtoms() > 0
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold if cyclic else fragment)
        parts.append(('ring:' if cyclic else 'acyclic:')+Chem.MolToSmiles(generic, isomericSmiles=False))
    return '.'.join(sorted(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grouping', choices=['ligand-set', 'scaffold'], default='ligand-set')
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    out = ROOT/('reports/grouped' if args.grouping == 'ligand-set' else 'reports/grouped_scaffold')
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT/'.data/FIA49k.csv.gz', low_memory=False)
    data['canonical_acid'] = data.la_smiles.map(group_key)
    data['ligand_set'] = data.ligand_smiles.map(group_key)
    # Audit canonical acid identities without moving records in the primary benchmark.
    overlap = data.groupby('canonical_acid').filter(lambda g: len(set(g.set_assignment))>1)
    overlap.to_csv(out/'canonical_identity_cross_partition.csv', index=False)
    full = data.copy()
    data = data.loc[data.set_assignment.isin(['train', 'validate', 'test'])].copy()
    data['original_assignment'] = data.set_assignment
    data['group_key'] = data.ligand_set if args.grouping == 'ligand-set' else data.ligand_smiles.map(scaffold_key)
    data['set_assignment'] = data.group_key.map(assignment)
    for key, g in data.groupby('group_key'):
        if g.set_assignment.nunique() != 1: raise AssertionError('Group leakage')
    train_ligands = {part for key in data.loc[data.set_assignment.eq('train'), 'ligand_set'] for part in key.split('.')}
    audit = dict(protocol=__doc__, grouping=args.grouping,
                 scaffold_definition='Sorted multiset of generic Bemis-Murcko ligand scaffolds; generic full ligand graph for acyclic fragments; center excluded',
                 n_groups=data.group_key.nunique(), largest_group=int(data.group_key.value_counts().max()),
                 n_molecules=len(data), n_ligand_sets=data.ligand_set.nunique(),
                 counts=data.set_assignment.value_counts().to_dict(),
                 canonical_acid_cross_partition_records=len(overlap),
                 individual_ligand_overlap={s: float(np.mean([any(part in train_ligands for part in key.split('.'))
                     for key in data.loc[data.set_assignment.eq(s), 'ligand_set']])) for s in ['validate','test']})
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    data[['Compound', 'original_assignment', 'set_assignment', 'ligand_set', 'group_key', 'ca_class']].to_csv(out/'manifest.csv', index=False)
    data.groupby(['set_assignment', 'ca_class']).size().to_csv(out/'class_counts.csv', header=['n'])
    with np.load(ROOT/'.data/la-PBEh3c.npz', allow_pickle=False) as source:
        if list(source['mol_id']) != list(full.Compound): raise ValueError('Geometry ID alignment differs')
        indices = data.index.to_numpy()
        offsets = np.r_[0, np.cumsum(source['N'])]
        atom_ix = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in indices])
        np.savez_compressed(ROOT/f'.data/{args.grouping}-PBEh3c.npz', mol_id=source['mol_id'][indices],
            N=source['N'][indices], Z=source['Z'][atom_ix], R=source['R'][atom_ix],
            y=source['y'][indices], center_index=source['center_index'][indices],
            split=data.set_assignment.to_numpy(dtype=str))
    desc = pd.read_csv(ROOT/'.data/rdkit_descriptors.csv.gz', index_col=0)
    if list(desc.index) != list(full.Compound): raise ValueError('Descriptor ID alignment differs')
    desc.index = full.index; desc.columns = ['rdkit_'+x for x in desc]
    data = pd.concat([data, desc.loc[data.index]], axis=1)
    tr, va = data.set_assignment.eq('train'), data.set_assignment.eq('validate')
    keep = [c for c in desc if data.loc[tr,c].notna().any() and data.loc[tr,c].nunique(dropna=True)>1]
    results, configs = [], []
    with threadpool_limits(limits=8):
        for name, numeric, target in [('RDKit2D', META+keep, data[TARGET]),
                                     ('delta_RDKit_LA_QM', META+keep+QM+['fia_gas-PBEh3c'], data[TARGET]-data['fia_gas-PBEh3c'])]:
            transform = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
                                           ('numeric', 'passthrough', numeric)])
            model = HistGradientBoostingRegressor(max_iter=1000, learning_rate=.05, max_leaf_nodes=31,
                l2_regularization=1, early_stopping=False, random_state=20261113)
            model.fit(transform.fit_transform(data.loc[tr]), target[tr])
            xv = transform.transform(data.loc[va])
            validation = {n: float(np.mean(abs(stage(model, xv, n)-target[va].to_numpy()))) for n in [100,400,1000]}
            selected = min(validation, key=lambda n: (validation[n], n))
            configs.append(dict(model=name, features=CATEGORICAL+numeric, validation_mae=validation,
                                selected_iteration=selected, estimator=model.get_params()))
            for split in ['validate','test']:
                g = data.loc[data.set_assignment.eq(split)]
                yp = stage(model, transform.transform(g), selected)
                if name.startswith('delta'): yp += g['fia_gas-PBEh3c'].to_numpy()
                frame = pd.DataFrame(dict(mol_id=g.Compound, y_true=g[TARGET], y_pred=yp))
                frame.to_csv(out/f'{name}_{split}.csv', index=False)
                vals = {r.mol_id:(r.y_true,r.y_pred) for r in frame.itertuples()}
                results.append(dict(model=name, split=split, **metrics(vals)))
    pd.DataFrame(results).to_csv(out/'metrics.csv', index=False)
    (out/'config.json').write_text(json.dumps(configs, indent=2)+'\n')
    print(json.dumps(audit, indent=2), flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__': main()
