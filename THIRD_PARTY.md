EGNN source in `egnn/models/gcl.py`, `egnn/qm9/models.py`, and `egnn/qm9/utils.py`
was recovered from https://github.com/vgsatorras/egnn at commit
`3c079e7267dad0aa6443813ac1a12425c3717558`, as referenced by this repository.
The upstream MIT license is retained in `egnn/LICENSE`. Local changes provide
package imports, train-only normalization and a device-aware edge cache. The
upstream QM9 model fixes coordinates; it does not simulate physical relaxation.

The optional directional comparison uses PyTorch Geometric 2.6.1's DimeNet++
implementation (MIT), https://github.com/pyg-team/pytorch_geometric/tree/2.6.1.
It is initialized from scratch. The original TensorFlow DimeNet source and
license remain in `dimenet/`; the new comparison does not reuse old weights.

FIA49k target, geometry and released prediction data are by Lukas M. Sigmund
and collaborators, distributed under CC BY 4.0. Source DOIs:
https://doi.org/10.6084/m9.figshare.24635136.v1,
https://doi.org/10.6084/m9.figshare.24635157.v1,
https://doi.org/10.6084/m9.figshare.24635187.v1.
Use the attribution and scientific citation in the manuscript when reusing them.
Bulk files are downloaded with checksum verification, rather than copied into Git.
