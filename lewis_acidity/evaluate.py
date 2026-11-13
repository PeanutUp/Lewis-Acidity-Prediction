"""Evaluate a frozen checkpoint, enforcing its training data and identity manifest."""
import argparse
import json
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, Subset
from .data import Molecules, collate
from .train import load_checkpoint, predict
from .metrics import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--data', type=Path, default=Path('.data/la-PBEh3c.npz'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--output',type=Path,help='Prediction directory; defaults to the run directory for best.pt')
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    dataset = Molecules(args.data, cutoff=ckpt['model_config']['cutoff'],
                        feature_mode=ckpt.get('feature_mode','none'), feature_state=ckpt.get('feature_state'))
    if dataset.sha256 != ckpt['data_sha256'] and dataset.content_sha256 != ckpt.get('data_content_sha256'):
        raise ValueError('Dataset fingerprint differs from training checkpoint')
    if dataset.ids[dataset.splits['train']].tolist() != ckpt['train_ids']:
        raise ValueError('Training identity manifest differs')
    rows = []
    name=args.checkpoint.name.removesuffix('.gz').removesuffix('.pt')
    out=args.output or (args.checkpoint.parent if name=='best' else args.checkpoint.parent/f'{name}_evaluation')
    out.mkdir(parents=True,exist_ok=True)
    for split, indices in dataset.splits.items():
        if not split.startswith('test') and split != 'validate': continue
        loader = DataLoader(Subset(dataset, indices), batch_size=32 if ckpt['model_config']['kind']=='dimenet' else 64,
                            collate_fn=collate, num_workers=2)
        ix, y, yp = predict(model, loader, args.device, ckpt['mean'], ckpt['scale'])
        frame = pd.DataFrame(dict(mol_id=dataset.ids[ix], y_true=dataset.y[ix], y_pred=yp))
        vals = {r.mol_id: (r.y_true, r.y_pred) for r in frame.itertuples()}
        rows.append(dict(split=split, **metrics(vals)))
        frame.to_csv(out/f'{split}.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
