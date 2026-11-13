"""Train on the official partition; select checkpoints exclusively on validation.

Example: python -m lewis_acidity.train --output runs/radial-site-s1 --device cuda:4
Test evaluation is a separate command, so monitoring never consumes test labels.
"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .data import Molecules, collate
from .model import build_model


def move(batch, device): return {k: v.to(device) for k, v in batch.items()}


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, device, mean, scale):
    model.eval()
    indices, targets, predictions = [], [], []
    for data in loader:
        batch = move(data, device)
        pred = model(batch)*scale+mean
        indices.extend(data['index'].tolist())
        targets.extend(data['y'].tolist())
        predictions.extend(pred.cpu().tolist())
    return np.asarray(indices), np.asarray(targets), np.asarray(predictions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.data/la-PBEh3c.npz'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--kind', choices=['egnn', 'radial', 'dimenet'], default='radial')
    p.add_argument('--readout', choices=['sum', 'site'], default='site')
    p.add_argument('--cutoff', type=float, default=5.)
    p.add_argument('--no-geometry', action='store_true')
    p.add_argument('--features', choices=['none', 'acid-qm'], default='none')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if (args.output/'best.pt').exists():
        raise FileExistsError('Choose a new run directory; existing checkpoints are never overwritten')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        torch.cuda.set_per_process_memory_fraction(.5)
        torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = Molecules(args.data, cutoff=args.cutoff, feature_mode=args.features)
    tr, va = dataset.splits['train'], dataset.splits['validate']
    mean, scale = float(dataset.y[tr].mean()), float(dataset.y[tr].std())
    if not scale > 0: raise ValueError('Degenerate training targets')
    config = dict(hidden=args.hidden, layers=args.layers, kind=args.kind,
                  readout=args.readout, cutoff=args.cutoff, geometry=not args.no_geometry,
                  global_dim=0 if dataset.feature_state is None else len(dataset.feature_state['columns']))
    if args.kind == 'dimenet' and args.features != 'none':
        raise ValueError('Acid-QM fusion is implemented for invariant message-passing models only')
    model = build_model(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=8, min_lr=1e-5)
    loader_args = dict(batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers,
                       persistent_workers=args.workers > 0)
    train = DataLoader(Subset(dataset, tr), shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed), **loader_args)
    valid = DataLoader(Subset(dataset, va), shuffle=False, **loader_args)
    run_config = {**vars(args), 'data': str(args.data), 'output': str(args.output),
                  'data_sha256': dataset.sha256, 'mean': mean, 'scale': scale,
                  'torch': str(torch.__version__), 'numpy': str(np.__version__),
                  'parameters': sum(x.numel() for x in model.parameters()),
                  'selection': 'minimum official validation MAE; no test evaluation during training',
                  'determinism': 'seeded RNGs; CUDA index_add may have nondeterministic rounding'}
    (args.output/'config.json').write_text(json.dumps(run_config, indent=2)+'\n')
    np.savez_compressed(args.output/'split_manifest.npz', mol_id=dataset.ids, split=dataset.labels)
    best, best_epoch = float('inf'), 0
    with (args.output/'history.jsonl').open('w') as history:
        for epoch in range(1, args.epochs+1):
            start = time.monotonic(); model.train(); total, count = 0., 0
            for data in train:
                batch = move(data, args.device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch)
                loss = torch.nn.functional.l1_loss(pred, (batch['y']-mean)/scale)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                total += loss.item()*len(pred)*scale; count += len(pred)
            _, y, yp = predict(model, valid, args.device, mean, scale)
            mae = float(np.mean(abs(y-yp)))
            schedule.step(mae)
            if mae < best:
                best, best_epoch = mae, epoch
                torch.save(dict(model=model.state_dict(), model_config=config, mean=mean, scale=scale,
                                epoch=epoch, validation_mae=mae, data_sha256=dataset.sha256,
                                train_ids=dataset.ids[tr].tolist(), validation_ids=dataset.ids[va].tolist(),
                                feature_mode=args.features, feature_state=dataset.feature_state,
                                seed=args.seed), args.output/'best.pt')
            row = dict(epoch=epoch, train_mae=total/count, validation_mae=mae,
                       best_validation_mae=best, best_epoch=best_epoch,
                       lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic()-start)
            history.write(json.dumps(row)+'\n'); history.flush(); print(json.dumps(row), flush=True)
            if epoch-best_epoch >= args.patience: break


if __name__ == '__main__': main()
