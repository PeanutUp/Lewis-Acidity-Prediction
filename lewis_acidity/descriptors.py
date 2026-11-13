"""New RDKit 2D/HGB experiments, retaining all official test IDs.

Run with PYTHONPATH=.../work/chemdeps and <=8 CPU threads. Fits use only train;
100/400/1000 boosting stages are selected on validate, then tests are scored once.
Parsing/descriptor failures are exported; unavailable descriptors become NaN,
with the same metadata inputs retained for every molecule. No 3D input is used.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import sklearn
from .metrics import metrics
from .tabular import CATEGORICAL,META,QM,TARGET,stage

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'rdkit_baseline'
NAMES=[name for name,_ in Descriptors._descList]


def featurize(item):
    key,smiles=item
    RDLogger.DisableLog('rdApp.*')
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:return key,[np.nan]*len(NAMES),'parse_failure'
    result=Descriptors.CalcMolDescriptors(mol,missingVal=np.nan,silent=True)
    values=[]
    for n in NAMES:
        try:v=float(result.get(n,np.nan))
        except (ValueError,TypeError):v=np.nan
        values.append(v if np.isfinite(v) else np.nan)
    return key,values,''


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'.data/FIA49k.csv.gz',low_memory=False)
    assert d.Compound.is_unique
    path=ROOT/'.data/rdkit_descriptors.csv.gz'
    if path.exists():
        desc=pd.read_csv(path,index_col=0)
        assert list(desc.index)==list(d.Compound)
        assert list(desc.columns)==NAMES
    else:
        start=time.monotonic();rows=[];failures=[]
        with ProcessPoolExecutor(max_workers=8) as pool:
            for n,(key,vals,error) in enumerate(pool.map(featurize,zip(d.Compound,d.la_smiles),chunksize=32),1):
                rows.append(vals)
                if error:failures.append({'Compound':key,'error':error})
                if n%5000==0:print('Descriptors',n,'seconds',round(time.monotonic()-start),flush=True)
        desc=pd.DataFrame(rows,index=d.Compound,columns=NAMES)
        desc.to_csv(path,compression='gzip')
        pd.DataFrame(failures,columns=['Compound','error']).to_csv(OUT/'parse_failures.csv',index=False)
    desc.index=d.index
    desc.columns=['rdkit_'+x for x in NAMES]
    d=pd.concat([d,desc],axis=1)
    tr=d.set_assignment.eq('train');va=d.set_assignment.eq('validate')
    # Remove no-information columns using training data alone.
    keep=[c for c in desc.columns if d.loc[tr,c].notna().any() and d.loc[tr,c].nunique(dropna=True)>1]
    desc.isna().sum().to_csv(OUT/'descriptor_missing_counts.csv',header=['missing_count'])
    configs=[];results=[]
    with threadpool_limits(limits=8):
        for name,extra in [('RDKit2D',[]),('RDKit2D_plus_LA_QM',QM)]:
            start=time.monotonic();numeric=META+keep+extra
            transform=ColumnTransformer([
                ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL),
                ('numeric','passthrough',numeric)])
            xtr=transform.fit_transform(d.loc[tr]);xva=transform.transform(d.loc[va])
            model=HistGradientBoostingRegressor(max_iter=1000,learning_rate=.05,
                max_leaf_nodes=31,l2_regularization=1.,early_stopping=False,random_state=20261113)
            model.fit(xtr,d.loc[tr,TARGET])
            validation={n:float(np.mean(abs(stage(model,xva,n)-d.loc[va,TARGET].to_numpy())))
                        for n in [100,400,1000]}
            best=min(validation,key=lambda n:(validation[n],n))
            configs.append(dict(model=name,features=CATEGORICAL+numeric,estimator=model.get_params(),
                validation_mae=validation,selected_iteration=best,
                rdkit_version=rdkit.__version__,sklearn_version=sklearn.__version__,
                elapsed_seconds=time.monotonic()-start,
                missing_policy='NaN handled natively by HGB; parse failures retained with metadata inputs'))
            print(name,'validation',validation,'selected',best,flush=True)
            for split in sorted(d.set_assignment.unique()):
                if not split.startswith('test'):continue
                g=d.loc[d.set_assignment.eq(split)]
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
