# Prediction identities and exports

Run `python scripts/restore_artifacts.py` once after cloning. The helper validates
each storage chunk and the reconstructed file SHA256. `predictions/` contains
compressed, molecule-level exports with explicit partition labels. `manifests/`
contains the official and secondary grouping assignments. Metrics are calculated
from full-precision targets and aligned IDs; no test intersections are used.
`storage_manifest.json` specifies every restored path and checksum.
