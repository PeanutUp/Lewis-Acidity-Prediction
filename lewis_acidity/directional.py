"""Optional PyG DimeNet++ reference with the original repository's 4-block size.

No QM9 weights are used. max_num_neighbors=118 covers every atom in FIA49k
without the default 32-neighbor truncation. This is a fresh PyTorch baseline,
not a claim to reproduce the legacy TensorFlow checkpoint or training schedule.
"""
from torch import nn


class DirectionalNet(nn.Module):
    def __init__(self, hidden=128, layers=4, cutoff=5., readout='sum', geometry=True, **unused):
        super().__init__()
        if readout != 'sum' or not geometry or cutoff <= 0:
            raise ValueError('DimeNet++ reference uses geometric inputs and sum readout')
        from torch_geometric.nn.models import DimeNetPlusPlus
        self.network = DimeNetPlusPlus(hidden_channels=hidden, out_channels=1, num_blocks=layers,
            int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, cutoff=cutoff, max_num_neighbors=118)

    def forward(self, batch):
        if int(batch['z'].max()) >= 95:
            raise ValueError('PyG DimeNet atomic embedding supports Z<95')
        return self.network(batch['z'], batch['r'], batch['batch']).squeeze(-1)
