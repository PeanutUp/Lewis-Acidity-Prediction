"""Raw quantum, global-bias and class-bias controls on each grouped partition."""
from pathlib import Path
import pandas as pd
from .metrics import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    meta = pd.read_csv(ROOT/'.data/FIA49k.csv.gz',low_memory=False).set_index('Compound',verify_integrity=True)
    for name in ['grouped','grouped_scaffold']:
        out = ROOT/'reports'/name
        manifest = pd.read_csv(out/'manifest.csv').set_index('Compound',verify_integrity=True)
        data = meta.loc[manifest.index].copy();data['set_assignment']=manifest.set_assignment
        tr = data.set_assignment.eq('train')
        delta = data['fia_gas-DSDBLYP']-data['fia_gas-PBEh3c']
        bias = delta[tr].groupby(data.loc[tr,'ca_class']).mean()
        variants = {'PBEh3c_raw':data['fia_gas-PBEh3c'],
                    'PBEh3c_global_bias':data['fia_gas-PBEh3c']+delta[tr].mean(),
                    'PBEh3c_class_bias':data['fia_gas-PBEh3c']+data.ca_class.map(bias).fillna(delta[tr].mean())}
        result = []
        for model,pred in variants.items():
            for split in ['validate','test']:
                g = data.loc[data.set_assignment.eq(split)]
                frame = pd.DataFrame(dict(mol_id=g.index,y_true=g['fia_gas-DSDBLYP'],y_pred=pred.loc[g.index]))
                frame.to_csv(out/f'{model}_{split}.csv',index=False)
                values = {r.mol_id:(r.y_true,r.y_pred) for r in frame.itertuples()}
                result.append(dict(model=model,split=split,**metrics(values)))
        pd.DataFrame(result).to_csv(out/'quantum_controls.csv',index=False)


if __name__=='__main__':main()
