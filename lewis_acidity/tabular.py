"""New CPU experiment, on the official FIA49k split, gas-phase target only.

Two HistGradientBoosting baselines: chemical metadata alone and metadata plus
six unbound-Lewis-acid PBEh-3c scalars. This is NOT a rerun of the user's XGBoost.
No fluoride-adduct feature, target-derived energy, or test-based selection is used.
"""
import itertools
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .metrics import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'qm_baseline'
CATEGORICAL=['ca','denticity_class']
META=['ca_ox_number','ca_period','ca_group','ligand_count']
QM=['la-ca_loewdin_charge-PBEh3c','la-ca_mulliken_charge-PBEh3c',
    'la-dipole-PBEh3c','la-global_electrophilicity-PBEh3c',
    'la-homo_energy-PBEh3c','la-lumo_energy-PBEh3c']
TARGET='fia_gas-DSDBLYP'


def stage(model, x, n):
    return next(itertools.islice(model.staged_predict(x),n-1,None))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'.data/FIA49k.csv.gz',low_memory=False)
    assert not data.Compound.duplicated().any()
    assert np.isfinite(data[TARGET]).all()
    tr=data.set_assignment.eq('train');va=data.set_assignment.eq('validate')
    assert not set(data.loc[tr,'Compound'])&set(data.loc[va,'Compound'])
    results=[];configs=[]
    with threadpool_limits(limits=8):
        for name,features in [('metadata',META),('metadata_plus_LA_QM',META+QM)]:
            start=time.monotonic()
            transform=ColumnTransformer([
                ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL),
                ('numeric','passthrough',features)])
            xtr=transform.fit_transform(data.loc[tr]);xva=transform.transform(data.loc[va])
            model=HistGradientBoostingRegressor(max_iter=1000,learning_rate=.05,
                       max_leaf_nodes=31,l2_regularization=1.,early_stopping=False,random_state=20261113)
            model.fit(xtr,data.loc[tr,TARGET])
            candidates=[100,400,1000]
            validation={n:float(np.mean(abs(stage(model,xva,n)-data.loc[va,TARGET].to_numpy())))
                        for n in candidates}
            best=min(candidates,key=lambda n:(validation[n],n))
            configs.append(dict(model=name,features=CATEGORICAL+features,
                estimator=model.get_params(),candidate_iterations=candidates,
                validation_mae=validation,selected_iteration=best,
                elapsed_seconds=time.monotonic()-start,sklearn_version=sklearn.__version__))
            print(name,'validation',validation,'selected',best,flush=True)
            for split in sorted(data.set_assignment.unique()):
                if not split.startswith('test'):continue
                g=data.loc[data.set_assignment.eq(split)]
                pred=stage(model,transform.transform(g),best)
                vals={i:(float(y),float(p)) for i,y,p in zip(g.Compound,g[TARGET],pred)}
                results.append(dict(model=name,split=split,phase='gas',**metrics(vals)))
                pd.DataFrame(dict(mol_id=g.Compound,y_true=g[TARGET],y_pred=pred)).to_csv(
                    OUT/f'{name}_{split}.csv',index=False)
                print(name,split,results[-1]['mae'],flush=True)
    pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'config.json').write_text(json.dumps(configs,indent=2)+'\n')
    print('Complete',flush=True)


if __name__=='__main__':main()
