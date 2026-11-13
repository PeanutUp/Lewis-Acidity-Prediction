"""Train-only method-bias summaries and matched-ligand comparisons.

Matched series quantify a change in reference-method correction. They do not
isolate a lone-pair mechanism or validate absolute experimental FIA accuracy.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .grouped import group_key
from .tabular import TARGET

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_mean(x, seed=20261113, n_bootstrap=2000):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_bootstrap)])
    return float(x.mean()), *np.quantile(means, [.025, .975]).tolist()


def main():
    data = pd.read_csv(ROOT/'.data/FIA49k.csv.gz', low_memory=False)
    data['method_correction'] = data[TARGET]-data['fia_gas-PBEh3c']
    train = data.loc[data.set_assignment.eq('train')].copy()
    train.groupby('ca_class').method_correction.agg(['count','mean','std','median']).to_csv(ROOT/'reports/train_method_bias_by_class.csv')
    train['ligand_set'] = train.ligand_smiles.map(group_key)
    # Several entries may share both a class and ligand set; aggregate before pairing.
    series = train.groupby(['ligand_set','ca_class']).method_correction.mean().unstack()
    results = []
    for low, high in [('P(III)','As(III)'),('As(III)','Sb(III)'),('Sb(III)','Bi(III)'),
                      ('Si(IV)','Ge(IV)'),('Ge(IV)','Sn(IV)'),('Sn(IV)','Pb(IV)'),
                      ('Si(II)','Ge(II)'),('Ge(II)','Sn(II)'),('Sn(II)','Pb(II)')]:
        pair = series[[low,high]].dropna()
        if len(pair)<5: continue
        mean, lo, hi = bootstrap_mean(pair[high]-pair[low])
        results.append(dict(class_a=low, class_b=high, n_ligand_sets=len(pair),
            mean_correction_b_minus_a=mean, ci95_low=lo, ci95_high=hi))
    pd.DataFrame(results).to_csv(ROOT/'reports/train_matched_ligand_corrections.csv', index=False)
    (ROOT/'reports/chemical_analysis_protocol.json').write_text(json.dumps(dict(
        selection='Training partition only; all available exact canonical ligand sets for the declared neighboring-center pairs',
        units='kJ/mol', uncertainty='2000 paired ligand-set bootstrap samples; fixed seed20261113; descriptive, no multiple-testing claim',
        interpretation='Difference between two calculated FIA methods at matched ligand identity; not a causal orbital assignment'),indent=2)+'\n')
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__': main()
