"""A distinct, quantum-assisted tier: learn DSD-BLYP minus PBEh-3c FIA.

The lower-level FIA already requires acid AND fluoride-adduct calculations.
These results must not be advertised as a SMILES-only or acid-geometry-only
speedup. Raw lower-level FIA and simple train-only bias correction are controls.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .metrics import metrics
from .tabular import CATEGORICAL, META, QM, TARGET, stage

ROOT = Path(__file__).resolve().parents[1]
LOW = 'fia_gas-PBEh3c'


def main():
    out = ROOT/'reports/refinement'; out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT/'.data/FIA49k.csv.gz', low_memory=False)
    desc = pd.read_csv(ROOT/'.data/rdkit_descriptors.csv.gz', index_col=0)
    if list(desc.index) != list(data.Compound): raise ValueError('Descriptor ID alignment differs')
    desc.index = data.index; desc.columns = ['rdkit_'+c for c in desc.columns]
    data = pd.concat([data, desc], axis=1)
    tr, va = data.set_assignment.eq('train'), data.set_assignment.eq('validate')
    delta = data[TARGET]-data[LOW]
    if not np.isfinite(data[[TARGET, LOW]].to_numpy()).all(): raise ValueError('Missing quantum target')
    keep = [c for c in desc if data.loc[tr, c].notna().any() and data.loc[tr, c].nunique(dropna=True)>1]
    class_bias = delta[tr].groupby(data.loc[tr, 'ca_class']).mean()
    baseline = data[LOW].to_numpy()
    predictions = {'PBEh3c_raw': baseline, 'PBEh3c_global_bias': baseline+delta[tr].mean(),
                   'PBEh3c_class_bias': baseline+data.ca_class.map(class_bias).fillna(delta[tr].mean()).to_numpy()}
    configs = []; results = []
    with threadpool_limits(limits=8):
        for name, numeric in [('delta_metadata', META+[LOW]),
                              ('delta_RDKit_LA_QM', META+[LOW]+keep+QM)]:
            start = time.monotonic()
            transform = ColumnTransformer([
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
                ('numeric', 'passthrough', numeric)])
            xtr, xva = transform.fit_transform(data.loc[tr]), transform.transform(data.loc[va])
            model = HistGradientBoostingRegressor(max_iter=1000, learning_rate=.05,
                max_leaf_nodes=31, l2_regularization=1., early_stopping=False, random_state=20261113)
            model.fit(xtr, delta[tr])
            validation = {n: float(np.mean(abs(stage(model, xva, n)-delta[va].to_numpy())))
                          for n in [100, 400, 1000]}
            selected = min(validation, key=lambda n: (validation[n], n))
            predictions[name] = baseline+stage(model, transform.transform(data), selected)
            joblib.dump(dict(transform=transform, model=model, selected_iteration=selected), out/f'{name}.joblib')
            configs.append(dict(model=name, features=CATEGORICAL+numeric, target=f'{TARGET} minus {LOW}',
                selected_iteration=selected, validation_mae=validation, estimator=model.get_params(),
                sklearn_version=sklearn.__version__, seconds=time.monotonic()-start))
            print(name, validation, flush=True)
    for name, pred in predictions.items():
        for split in sorted(set(data.set_assignment)):
            if split == 'train': continue
            g = data.loc[data.set_assignment.eq(split)]
            vals = {key: (float(y), float(p)) for key, y, p in zip(g.Compound, g[TARGET], pred[g.index])}
            results.append(dict(model=name, split=split, input_tier='adduct-dependent quantum refinement', **metrics(vals)))
            pd.DataFrame(dict(mol_id=g.Compound, y_true=g[TARGET], y_pred=pred[g.index])).to_csv(out/f'{name}_{split}.csv', index=False)
    pd.DataFrame(results).to_csv(out/'metrics.csv', index=False)
    (out/'config.json').write_text(json.dumps(configs, indent=2)+'\n')
    print(pd.DataFrame(results)[['model', 'split', 'mae']].to_string(index=False), flush=True)


if __name__ == '__main__': main()
