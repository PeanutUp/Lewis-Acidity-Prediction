# Lewis Acidity Prediction

Reproducible prediction of gas-phase fluoride ion affinity (FIA) across p-block
Lewis acids. The original EGNN and TensorFlow DimeNet work is retained in Git
history and the existing model directories. The revision integrates preparation,
training, validation, classical controls and chemical analysis into one package.

## Scientific status

The original paper's accuracy figures must not be interpreted as verified held-out
performance. The committed 2,000-molecule prediction spreadsheets reproduce its
rounded RMSE values, but the saved DimeNet configuration reconstructs 1,813 of
those molecules in training, 105 in validation and only 82 in test. The EGNN
training manifest is missing. See `scripts/audit_original.py` and
`reports/original_prediction_cohort_audit.json`; the reconstruction assumptions
are explicit. New runs use the official published partition and saved IDs.

This project uses one optimized geometry per molecule. It does not yet establish
conformer-ensemble modeling, causal orbital explanations, physical coordinate
relaxation or experimental accuracy. All new comparisons are to calculated
`fia_gas-DSDBLYP` labels in kJ/mol.

## Environment

Use Python 3.10 and your own virtual environment. Install the PyTorch 2.6 wheel
appropriate to your CPU/CUDA, then the remaining pinned dependencies in
`requirements-benchmark.txt`. The recorded neural environment uses CUDA 12.4.
For the optional PyG DimeNet++ reference install matching extension wheels with
`scripts/install_directional.sh`. `THIRD_PARTY.md` identifies code and data licenses.

Run commands below from the repository root. `python -m pytest -q` checks split
isolation, ID/label alignment, normalization, malformed geometries, geometric
invariance, batch independence and checkpoint restoration.

## Prepare the complete dataset

```bash
python -m lewis_acidity.prepare --data-dir .data
```

The downloader verifies the versioned public file checksums. All 48,986 molecules
are retained, with 22 atom types and up to 118 atoms. The default input is explicitly
`la-PBEh3c`, the unbound acid; `--geometry la-GFN2xTB` prepares a separate low-cost
geometry source. There is no fallback to fluoride-adduct coordinates, silent atom
truncation, unknown-element-to-zero conversion or row-index identity guessing.
The ragged NPZ, per-molecule manifest and SHA256 provenance are written to `.data/`.

## Train and evaluate

```bash
python -m lewis_acidity.train --output runs/radial-site-s1 --device cuda:0 --seed 1
python -m lewis_acidity.evaluate runs/radial-site-s1/best.pt --device cuda:0
```

The first command uses only 32,333 training and 7,001 validation molecules.
Checkpoints contain normalization, model configuration, data fingerprint and
training/validation IDs. Test evaluation is a separate command and rejects a
different dataset. All six published test groups are exported with molecule IDs;
no failing or inconvenient cases are intersected away.

Controlled variants:

```bash
# Same distance-filter network, extensive readout
python -m lewis_acidity.train --output runs/radial-sum-s1 --readout sum --device cuda:0
# Reconstructed EGNN feature-message baseline, 7 layers, fixed coordinates
python egnn/main_fia49k_new.py --output runs/egnn-s1 --device cuda:0
# Fresh 4-block PyG DimeNet++; no pretrained weights or 32-neighbor truncation
python -m lewis_acidity.train --output runs/dimenet-s1 --kind dimenet --readout sum --batch-size 32 --device cuda:0
# Composition control: complete graph, no coordinate information
python -m lewis_acidity.train --output runs/composition-s1 --no-geometry --cutoff 0 --device cuda:0
# Acid-only electronic descriptors fused with the geometric representation
python -m lewis_acidity.train --output runs/acid-qm-s1 --features acid-qm --device cuda:0
```

The site readout concatenates center, mean and sum representations. Gaussian
radial filters, invariant feature messages and center pooling are established
constructions; this repository does not claim a new GNN architecture. Seeds,
selection rules and all planned comparisons are in `reports/experiment_plan.json`.

## Classical controls and a separate quantum-refinement tier

```bash
python -m lewis_acidity.tabular
python -m lewis_acidity.descriptors
python -m lewis_acidity.refinement
```

Descriptor generation can take several minutes. The first two commands use
chemical metadata, RDKit descriptors and/or six **unbound-acid** PBEh-3c scalars.
The refinement command learns DSD-BLYP minus PBEh-3c FIA. Its lower-level FIA
requires calculations of **both acid and fluoride adduct**. Its better accuracy
must therefore be interpreted within that more expensive input tier, alongside
raw PBEh-3c and train-only global/class bias controls. It is not a SMILES-only
screening model. All stage selection uses validation only.

## Generalization and chemical diagnostics

```bash
python -m lewis_acidity.grouped --grouping scaffold
python -m lewis_acidity.train --data .data/scaffold-PBEh3c.npz --output runs/scaffold-s1 --device cuda:0
python -m lewis_acidity.evaluate runs/scaffold-s1/best.pt --data .data/scaffold-PBEh3c.npz --device cuda:0
python -m lewis_acidity.chemistry
python -m lewis_acidity.reorganization
```

The secondary split holds out whole multisets of generic ligand scaffolds across
central elements (33,460/6,379/6,227 train/validation/test). Cyclic ligands use
generic Bemis–Murcko scaffolds; acyclic ligands use generic full graphs. Individual
ligands can still overlap; all counts and the exact definition are reported. A
separate `--grouping ligand-set` experiment holds out exact complete ligand sets.
Published FIA-GNN weights are not evaluated as an unseen-scaffold baseline because
their training membership would be incompatible with this new split.

Matched-ligand method corrections use training data only. Geometry reorganization
uses paired acid/adduct structures solely for diagnostics, checks archive atom
order, and reports pair-distance changes in Angstrom—not deformation energies.
The original “Contrastive Learning” directory contains legacy error analysis,
not a contrastive-learning method.

## Results, manuscript and reproducibility

Small metrics/configuration/provenance reports are versioned under `reports/`.
Large prepared data, fitted checkpoints and raw per-molecule exports are generated
locally under `.data/`, `runs/` and the report subdirectories. Manuscript sources,
figures and journal variants are maintained under `manuscript/`. A publication
archive must include the frozen weights, prediction exports and manifests; the
Digital Discovery code/data DOI must be minted before publication.

Data: Sigmund et al., FIA49k, CC BY 4.0,
[targets](https://doi.org/10.6084/m9.figshare.24635136.v1),
[geometries](https://doi.org/10.6084/m9.figshare.24635157.v1),
[released predictions](https://doi.org/10.6084/m9.figshare.24635187.v1).
Scientific reference: Sigmund et al., *Angew. Chem. Int. Ed.* 2024, 63, e202401084,
[doi:10.1002/anie.202401084](https://doi.org/10.1002/anie.202401084).
