"""Ragged molecular batches: no padding, atom cap, or implicit geometry fallback."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .prepare import checksum


def split_indices(labels):
    labels = np.asarray(labels)
    if not set(labels) <= {'train', 'validate', 'test', *[f'test_{i}' for i in range(2, 7)]}:
        raise ValueError('Unknown official split label')
    result = {s: np.flatnonzero(labels == s) for s in sorted(set(labels))}
    if not len(result.get('train', [])) or not len(result.get('validate', [])):
        raise ValueError('Training and validation partitions must be nonempty')
    return result


class Molecules(Dataset):
    def __init__(self, path, cutoff=5.0, feature_mode='none', feature_state=None):
        if feature_mode not in {'none', 'acid-qm'}:
            raise ValueError('Unknown feature mode')
        self.path = Path(path)
        self.sha256 = checksum(self.path)
        with np.load(path, allow_pickle=False) as f:
            self.ids, self.y, self.labels = f['mol_id'].copy(), f['y'].copy(), f['split'].copy()
            n, z, r, centers = f['N'], f['Z'], f['R'], f['center_index']
            if len(set(self.ids)) != len(self.ids) or len(n) != len(self.ids):
                raise ValueError('Nonunique or unaligned molecule IDs')
            if sum(n) != len(z) or r.shape != (len(z), 3) or (z < 1).any() or (z > 118).any():
                raise ValueError('Invalid atom arrays')
            if not np.isfinite(r).all() or not np.isfinite(self.y).all():
                raise ValueError('Nonfinite data')
            self.items = []
            start = 0
            for i, count in enumerate(n):
                zi = z[start:start+count].astype(np.int64)
                ri = r[start:start+count].copy()
                d2 = ((ri[:, None] - ri[None, :])**2).sum(-1)
                mask = ~np.eye(count, dtype=bool)
                if cutoff > 0:
                    mask &= d2 < cutoff**2
                edges = np.stack(np.where(mask))
                if not 0 <= centers[i] < count:
                    raise ValueError('Invalid accepting-center index')
                self.items.append(dict(z=torch.from_numpy(zi), r=torch.from_numpy(ri),
                                       edges=torch.from_numpy(edges), center=int(centers[i]),
                                       y=float(self.y[i]), index=i))
                start += count
        self.splits = split_indices(self.labels)
        self.feature_state = None
        if feature_mode != 'none':
            import pandas as pd
            from .tabular import QM, META
            source = self.path.parent/'FIA49k.csv.gz'
            metadata = pd.read_csv(source, low_memory=False).set_index('Compound', verify_integrity=True).loc[self.ids]
            if not np.allclose(metadata['fia_gas-DSDBLYP'].to_numpy(), self.y, atol=1e-8, rtol=0):
                raise ValueError('Quantum descriptor target/ID alignment differs')
            columns = QM+META
            raw = metadata[columns].to_numpy(dtype=np.float64)
            if feature_state is None:
                training = raw[self.splits['train']]
                median = np.nanmedian(np.where(np.isfinite(training), training, np.nan), axis=0)
                if not np.isfinite(median).all(): raise ValueError('Entirely missing training feature')
                filled = np.where(np.isfinite(training), training, median)
                mean, scale = filled.mean(0), filled.std(0)
                scale = np.maximum(scale, 1e-8)
                feature_state = dict(columns=columns, median=median.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                                     source_sha256=checksum(source))
            if feature_state['columns'] != columns or checksum(source) != feature_state['source_sha256']:
                raise ValueError('Feature provenance differs from checkpoint')
            x = (np.where(np.isfinite(raw), raw, feature_state['median'])-feature_state['mean'])/feature_state['scale']
            for item, vector in zip(self.items, x): item['global'] = torch.tensor(vector, dtype=torch.float32)
            self.feature_state = feature_state

    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.items[i]


def collate(items):
    counts = [len(item['z']) for item in items]
    offsets = np.r_[0, np.cumsum(counts)]
    result = dict(z=torch.cat([i['z'] for i in items]), r=torch.cat([i['r'] for i in items]),
                edges=torch.cat([i['edges']+int(o) for i, o in zip(items, offsets)], dim=1),
                batch=torch.repeat_interleave(torch.arange(len(items)), torch.tensor(counts)),
                center=torch.tensor([i['center']+int(o) for i, o in zip(items, offsets)]),
                counts=torch.tensor(counts), y=torch.tensor([i['y'] for i in items]),
                index=torch.tensor([i['index'] for i in items]))
    if 'global' in items[0]: result['global'] = torch.stack([i['global'] for i in items])
    return result
