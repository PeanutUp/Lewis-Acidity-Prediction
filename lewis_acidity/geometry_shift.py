"""Frozen-model robustness to xTB vs PBEh-3c geometry, NOT conformer ensembles.

Intentional geometry shift is explicit; identity, element ordering, targets and
partition labels must remain identical. Model selection/normalization are frozen.
Only geometry-only radial-site models are probed, so PBEh electronic descriptors
are not silently mixed into an alleged low-cost xTB screening workflow.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from .data import Molecules,collate
from .train import load_checkpoint,predict
from .metrics import metrics
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(4)
    acid=ROOT/'.data/la-PBEh3c.npz';xtb=ROOT/'.data/la-GFN2xTB.npz'
    with np.load(acid,allow_pickle=False) as a,np.load(xtb,allow_pickle=False) as b:
        for key in ['mol_id','N','Z','y','split','center_index']:
            if not np.array_equal(a[key],b[key]): raise ValueError(f'Geometry-source alignment differs: {key}')
    out=ROOT/'reports/geometry_shift';out.mkdir(parents=True,exist_ok=True)
    dataset=Molecules(xtb)
    rows=[];exports={}
    for seed in [1,2,3]:
        run=ROOT/f'runs/radial-site-s{seed}'
        model,ckpt=load_checkpoint(run/'best.pt',args.device)
        if ckpt['data_sha256'] != __import__('lewis_acidity.prepare',fromlist=['checksum']).checksum(acid):
            raise ValueError('Model was not trained on the verified PBEh geometry archive')
        if ckpt.get('feature_mode','none')!='none': raise ValueError('Only geometry-only models are allowed')
        if ckpt['train_ids']!=dataset.ids[dataset.splits['train']].tolist(): raise ValueError('Training IDs differ')
        for split,indices in dataset.splits.items():
            if not split.startswith('test'): continue
            loader=DataLoader(Subset(dataset,indices),batch_size=64,collate_fn=collate,num_workers=2)
            ix,_,yp=predict(model,loader,args.device,ckpt['mean'],ckpt['scale'])
            frame=pd.DataFrame(dict(mol_id=dataset.ids[ix],y_true=dataset.y[ix],y_pred=yp)).set_index('mol_id')
            original=pd.read_csv(run/f'{split}.csv').set_index('mol_id')
            if set(frame.index)!=set(original.index): raise ValueError('PBEh prediction IDs differ')
            frame=frame.loc[original.index]
            vals={k:(r.y_true,r.y_pred) for k,r in frame.iterrows()}
            base_mae=float(np.mean(abs(original.y_true-original.y_pred)))
            rows.append(dict(seed=seed,split=split,**metrics(vals),original_PBEh_mae=base_mae,
                mae_change=metrics(vals)['mae']-base_mae,
                mean_absolute_prediction_shift=float(np.mean(abs(frame.y_pred-original.y_pred)))))
            frame.reset_index().to_csv(out/f'seed{seed}_{split}.csv',index=False)
            exports[(seed,split)]=frame
    ensemble=[]
    for split in sorted(s for s in dataset.splits if s.startswith('test')):
        frame=exports[(1,split)].copy()
        frame['y_pred']=np.mean([exports[(seed,split)].y_pred.to_numpy() for seed in [1,2,3]],axis=0)
        frame.reset_index().to_csv(out/f'ensemble_{split}.csv',index=False)
        ensemble.append(dict(split=split,**metrics({k:(r.y_true,r.y_pred) for k,r in frame.iterrows()})))
    pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)
    pd.DataFrame(ensemble).to_csv(out/'ensemble_metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(experiment=__doc__,target='fia_gas-DSDBLYP unchanged',
        retraining=False,input_geometry='la-GFN2xTB',training_geometry='la-PBEh3c',
        interpretation='Deployment geometry-source shift; not an xTB-trained model, conformer uncertainty, or energy decomposition'),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
