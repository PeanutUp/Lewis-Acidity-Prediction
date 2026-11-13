"""Predict unlabeled unbound-acid XYZ files with frozen neural checkpoints.

Example: python -m lewis_acidity.predict --checkpoint runs/dimenet-s1/best.pt
  --xyz acid.xyz --center-index 0 --output predictions.csv

Use the same geometry method as training. An ensemble SD is model disagreement,
not a calibrated prediction interval. Test-set scoring remains in evaluate.py.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .prepare import parse_xyz, checksum
from .data import collate
from .train import load_checkpoint, move


def xyz_item(path, center_index, cutoff):
    z, r = parse_xyz(Path(path).read_text())
    if not 0 <= center_index < len(z): raise ValueError('Accepting-center index is out of range')
    distance2 = ((r[:,None]-r[None,:])**2).sum(-1)
    mask = ~np.eye(len(z),dtype=bool)
    if cutoff>0: mask &= distance2<cutoff**2
    return dict(z=torch.tensor(z,dtype=torch.long),r=torch.tensor(r,dtype=torch.float32),
                edges=torch.tensor(np.stack(np.where(mask)),dtype=torch.long),
                center=center_index,index=0,y=0.)


def predict_files(checkpoints, xyz_paths, center_index=0, feature_records=None, device='cpu'):
    paths=[Path(p) for p in xyz_paths];ids=[p.stem for p in paths]
    if len(set(ids))!=len(ids): raise ValueError('XYZ filename stems must be unique molecule IDs')
    if not paths or not checkpoints: raise ValueError('Need at least one XYZ and checkpoint')
    predictions=[]
    for checkpoint in checkpoints:
        model, state=load_checkpoint(checkpoint,device)
        items=[xyz_item(path,center_index,state['model_config']['cutoff']) for path in paths]
        features=state.get('feature_state')
        if features is not None:
            if feature_records is None: raise ValueError('This checkpoint requires acid-QM features')
            for item,key in zip(items,ids):
                record=feature_records[key]
                if set(record)!=set(features['columns']):
                    raise ValueError(f'{key}: provide exactly these features: {features["columns"]}')
                raw=np.array([np.nan if record[c] is None else float(record[c]) for c in features['columns']])
                vector=(np.where(np.isfinite(raw),raw,features['median'])-features['mean'])/features['scale']
                item['global']=torch.tensor(vector,dtype=torch.float32)
        elif feature_records is not None:
            raise ValueError('This checkpoint does not use global electronic features')
        values=[]
        with torch.no_grad():
            for batch in DataLoader(items,batch_size=16,collate_fn=collate):
                output=model(move(batch,device))*state['scale']+state['mean']
                values.extend(output.cpu().tolist())
        if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite prediction')
        predictions.append(values)
        del model
    pred=np.asarray(predictions)
    return pd.DataFrame(dict(mol_id=ids,predicted_fia_gas_DSDBLYP_kJ_mol=pred.mean(0),
                             ensemble_prediction_sd=pred.std(0,ddof=1) if len(pred)>1 else np.full(len(ids),np.nan),
                             n_models=len(pred),xyz_sha256=[checksum(p) for p in paths]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--xyz',type=Path,nargs='+',required=True)
    p.add_argument('--center-index',type=int,default=0,help='0-based accepting atom index, same for supplied XYZ files')
    p.add_argument('--features',type=Path,help='JSON mapping filename stem to the saved acid-QM feature columns')
    p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    feature_records=json.loads(args.features.read_text()) if args.features else None
    frame=predict_files(args.checkpoint,args.xyz,args.center_index,feature_records,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output,index=False)
    args.output.with_suffix('.provenance.json').write_text(json.dumps(dict(
        checkpoints={str(p):checksum(p) for p in args.checkpoint},
        accepting_center_index=args.center_index,feature_source_sha256=checksum(args.features) if args.features else None,
        target='FIA gas, dataset DSD-BLYP reference, kJ/mol',
        ensemble_sd='Model disagreement; not calibrated uncertainty',
        geometry_requirement='Unbound acid; same geometry-generation protocol as training'),indent=2)+'\n')
    print(frame.to_string(index=False))


if __name__=='__main__':main()
