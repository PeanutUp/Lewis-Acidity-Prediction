"""Strict comparisons, seed summaries, and paired descriptive uncertainty.

Run only after all planned training/evaluation jobs finish. --partial is useful
for development but is visibly marked and must not produce the final manuscript.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import load_predictions, metrics
ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {'radial-site':3, 'radial-sum':3, 'egnn-sum':3, 'dimenet':3, 'acid-qm':3,
            'scaffold-site':3, 'scaffold-qm':3, 'composition':1, 'composition-qm':1}


def aligned(reference, other):
    if set(reference) != set(other): raise ValueError('Incompatible prediction identities')
    keys = sorted(reference)
    if any(abs(reference[k][0]-other[k][0])>1e-6 for k in keys):
        raise ValueError('Incompatible target labels')
    return keys


def paired_bootstrap(reference, other, classes, n=2000):
    keys = aligned(reference, other)
    difference = np.array([abs(other[k][1]-other[k][0])-abs(reference[k][1]-reference[k][0]) for k in keys])
    labels = np.array([classes[k] for k in keys])
    groups = [np.flatnonzero(labels==c) for c in sorted(set(labels))]
    rng = np.random.default_rng(20261113)
    means = np.array([sum(difference[rng.choice(g,len(g),replace=True)].sum() for g in groups)/len(keys) for _ in range(n)])
    lo, hi = np.quantile(means,[.025,.975])
    return dict(delta_mae=float(difference.mean()),ci95_low=float(lo),ci95_high=float(hi),n=len(keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--partial',action='store_true')
    args = parser.parse_args()
    out = ROOT/'reports/neural';out.mkdir(parents=True,exist_ok=True)
    exports = ROOT/'reports/neural_predictions';exports.mkdir(parents=True,exist_ok=True)
    metadata = pd.read_csv(ROOT/'.data/FIA49k.csv.gz',low_memory=False).set_index('Compound')
    classes = metadata.ca_class.to_dict()
    per_seed, run_info, ensembles, available = [], [], {}, {}
    for family, expected in FAMILIES.items():
        runs = [ROOT/f'runs/{family}-s{i}' for i in range(1,expected+1)]
        completed = [run for run in runs if (run/'metrics.json').exists()]
        available[family] = len(completed)
        if len(completed) != expected and not args.partial:
            raise ValueError(f'{family}: expected {expected} evaluated runs, found {len(completed)}')
        if not completed: continue
        for run in completed:
            cfg = json.loads((run/'config.json').read_text())
            history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
            run_info.append(dict(cfg,run=run.name,epochs=len(history),
                total_epoch_seconds=sum(x['seconds'] for x in history),
                best_validation_mae=min(x['validation_mae'] for x in history),
                best_epoch=min(history,key=lambda r:r['validation_mae'])['epoch']))
            for row in json.loads((run/'metrics.json').read_text()):
                per_seed.append(dict(model=family,run=run.name,seed=cfg['seed'],**row))
        if len(completed) != expected: continue  # never call a partial ensemble a three-seed result
        for split in ['test','test_2','test_3','test_4','test_5','test_6','validate']:
            paths = [run/f'{split}.csv' for run in completed]
            if not all(p.exists() for p in paths): continue
            predictions = [load_predictions(path) for path in paths]
            keys = sorted(predictions[0])
            for pred in predictions[1:]: aligned(predictions[0],pred)
            mean = {k:(predictions[0][k][0],float(np.mean([p[k][1] for p in predictions]))) for k in keys}
            ensembles[(family,split)] = mean
            pd.DataFrame([dict(mol_id=k,y_true=v[0],y_pred=v[1],
                seed_prediction_sd=float(np.std([p[k][1] for p in predictions],ddof=1)) if expected>1 else 0.)
                for k,v in mean.items()]).to_csv(exports/f'{family}_{split}.csv',index=False)
    per = pd.DataFrame(per_seed)
    per.to_csv(out/'per_seed_metrics.csv',index=False)
    if len(per):
        per.groupby(['model','split']).agg(n=('n','first'),n_seeds=('seed','size'),
            mae_mean=('mae','mean'),mae_sd=('mae','std'),rmse_mean=('rmse','mean'),
            r2_mean=('r2','mean')).reset_index().to_csv(out/'seed_summary.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps(run_info,indent=2)+'\n')
    pd.DataFrame([dict(model=model,split=split,**metrics(values)) for (model,split),values in ensembles.items()]).to_csv(out/'ensemble_metrics.csv',index=False)
    confidence, subgroup = [], []
    for (family,split),pred in ensembles.items():
        if split=='validate': continue
        if family.startswith('scaffold'):
            reference = load_predictions(ROOT/'reports/grouped_scaffold'/f'RDKit2D_{split}.csv')
            baseline = 'RDKit2D_retrained_scaffold'
        else:
            reference = load_predictions(ROOT/'reports/public_reference'/f'fia_gnn_gas_{split}_predictions.csv')
            baseline = 'FIA-GNN_released'
        confidence.append(dict(model=family,split=split,baseline=baseline,**paired_bootstrap(reference,pred,classes)))
        for label in sorted({classes[k] for k in pred}):
            group = {k:v for k,v in pred.items() if classes[k]==label}
            subgroup.append(dict(model=family,split=split,ca_class=label,**metrics(group)))
    pd.DataFrame(confidence).to_csv(out/'paired_bootstrap.csv',index=False)
    pd.DataFrame(subgroup).to_csv(out/'by_class.csv',index=False)
    (out/'status.json').write_text(json.dumps(dict(partial=args.partial,available=available,
        uncertainty='2000 paired, central-class-stratified bootstrap samples of fixed test predictions; not training uncertainty or prospective coverage',
        ensembles='Unweighted mean of all declared seeds; seed mean/SD separately reported',
        timing='Observed epoch wall time on a shared host; no isolated architecture-speed ranking'),indent=2)+'\n')
    print(json.dumps(available,indent=2),flush=True)


if __name__=='__main__': main()
