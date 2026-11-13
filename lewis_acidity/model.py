"""Static-geometry invariant message passing with a controlled readout ablation.

EGNN feature messages follow Satorras et al. (ICML 2021). Like the official QM9
implementation, coordinates are fixed. The radial variant uses Gaussian filters
and a smooth cosine envelope, a standard continuous-filter construction. Neither
architecture nor the center-conditioned pooling is claimed as a new GNN family.
"""
import math
import torch
from torch import nn


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))


def scatter_sum(x, indices, n):
    return x.new_zeros((n, x.shape[-1])).index_add_(0, indices, x)


def build_model(config):
    if config['kind'] == 'dimenet':
        from .directional import DirectionalNet
        return DirectionalNet(**config)
    return AcidityNet(**config)


class Interaction(nn.Module):
    def __init__(self, hidden, radial, kind):
        super().__init__()
        self.kind = kind
        self.edge = mlp(2*hidden+1 if kind == 'egnn' else radial, hidden, hidden)
        self.node = mlp(2*hidden, hidden, hidden)
        self.gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, h, edges, distance, rbf, envelope):
        row, col = edges
        if self.kind == 'egnn':
            msg = self.edge(torch.cat([h[row], h[col], distance[:, None]**2], -1))
            msg = msg * self.gate(msg)
        else:
            msg = self.edge(rbf) * h[col]
        msg = msg * envelope[:, None]
        aggregate = scatter_sum(msg, row, len(h))
        return h + self.node(torch.cat([h, aggregate], -1))


class AcidityNet(nn.Module):
    def __init__(self, hidden=128, layers=4, radial=32, cutoff=5.0,
                 kind='radial', readout='site', geometry=True, global_dim=0):
        super().__init__()
        if kind not in {'egnn', 'radial'} or readout not in {'sum', 'site'}:
            raise ValueError('Unknown model or readout')
        if not geometry and cutoff > 0:
            raise ValueError('Geometry-free control needs a complete graph (cutoff=0)')
        self.cutoff, self.geometry, self.readout = cutoff, geometry, readout
        self.global_dim = global_dim
        self.embedding = nn.Embedding(119, hidden, padding_idx=0)
        self.register_buffer('rbf_centers', torch.linspace(0, cutoff if cutoff > 0 else 10, radial))
        self.rbf_width = radial / (cutoff if cutoff > 0 else 10)
        self.interactions = nn.ModuleList([Interaction(hidden, radial, kind) for _ in range(layers)])
        self.node_decoder = mlp(hidden, hidden, hidden)
        if global_dim: self.global_encoder = mlp(global_dim, hidden, hidden)
        self.output = mlp((hidden if readout == 'sum' else 3*hidden)+(hidden if global_dim else 0), hidden, 1)

    def forward(self, data):
        h = self.embedding(data['z'])
        row, col = data['edges']
        distance = (data['r'][row] - data['r'][col]).norm(dim=-1)
        if not self.geometry:
            distance = torch.zeros_like(distance)
        rbf = torch.exp(-((distance[:, None]-self.rbf_centers)*self.rbf_width)**2)
        envelope = (0.5*(torch.cos(math.pi*distance/self.cutoff)+1)).clamp(min=0) \
            if self.cutoff > 0 else torch.ones_like(distance)
        for interaction in self.interactions:
            h = interaction(h, data['edges'], distance, rbf, envelope)
        h = self.node_decoder(h)
        pooled = scatter_sum(h, data['batch'], len(data['counts']))
        if self.readout == 'site':
            pooled = torch.cat([h[data['center']], pooled/data['counts'][:, None], pooled], -1)
        if self.global_dim:
            pooled = torch.cat([pooled, self.global_encoder(data['global'])], -1)
        return self.output(pooled).squeeze(-1)
