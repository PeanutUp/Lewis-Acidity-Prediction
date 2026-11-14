"""Recompute original spreadsheet metrics and reconstruct the saved DimeNet split.

The membership reconstruction assumes the saved config, committed processed NPZ
and deterministic provider describe the historical run. It does not recover the
missing EGNN training manifest or silently equate official and historical splits.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import yaml
from lewis_acidity.metrics import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    meta = pd.read_csv(ROOT/'.data/FIA49k.csv.gz', low_memory=False).set_index('Compound')
    config = yaml.safe_load((ROOT/'dimenet/None/config.yaml').read_text())
    with np.load(ROOT/'dimenet/data/fia49k_full_processed.npz', allow_pickle=False) as original:
        with np.load(ROOT/'.data/la-PBEh3c.npz', allow_pickle=False) as current:
            assert np.array_equal(original['fia_gas-DSDBLYP'], current['y'])
            assert np.array_equal(original['N'], current['N'])
            assert np.array_equal(original['Z'], current['Z'])
            assert np.array_equal(original['R'], current['R'])
    perm = np.random.RandomState(config['data_seed']).permutation(len(meta))
    ntrain, nvalid = config['num_train'], config['num_valid']
    membership = {meta.index[i]: ('train' if j<ntrain else ('validate' if j<ntrain+nvalid else 'test'))
                  for j, i in enumerate(perm)}
    records = []
    for name in ['dimenet', 'egnn']:
        filename = f'Contrastive Learning/{name}_sampled_predictions.xlsx'
        frame = pd.read_excel(ROOT/filename)
        vals = {r.mol_id:(r.true_value, r.predicted_value) for r in frame.itertuples()}
        if len(vals) != len(frame): raise ValueError('Duplicate original IDs')
        records.append(dict(file=filename, **metrics(vals),
            official_assignments=meta.loc[frame.mol_id,'set_assignment'].value_counts().to_dict(),
            reconstructed_saved_dimenet_membership=pd.Series([membership[i] for i in frame.mol_id]).value_counts().to_dict(),
            max_label_rounding_difference=float(max(abs(meta.loc[frame.mol_id,'fia_gas-DSDBLYP'].to_numpy()-frame.true_value))),
            caveat='DimeNet membership reconstructed from saved config and verified NPZ ordering; EGNN membership remains unknown'))
    (ROOT/'work/original_review').mkdir(parents=True, exist_ok=True)
    (ROOT/'work/original_review/original_prediction_cohort_audit.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__': main()
