# Frozen FIA49k models

Run `python scripts/restore_artifacts.py` from the repository root to restore and
verify the compressed checkpoints from their storage chunks. `manifest.json`
records each run, feature regime, validation selection, data fingerprint and
SHA256. All model tensors and normalizers match the completed training runs;
publication packaging adds a canonical data-array fingerprint and retains the
original checkpoint SHA256. Repacking identical arrays into another NPZ container
does not change their identity.

Use `.pt.gz` checkpoints directly with `lewis_acidity.predict` or
`lewis_acidity.evaluate`. The two `.joblib.gz` files contain the quantum-refinement
HGB models, encoders and selected stage counts. They require lower-level FIA and
have a different preparation cost from the geometry-only models.
