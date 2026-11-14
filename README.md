# Lewis Acidity Prediction

Reproducible prediction of gas-phase fluoride ion affinity (FIA) across p-block
Lewis acids, integrating geometry-based neural networks, quantum refinement,
chemical diagnostics and native LaTeX manuscript sources.

## Study and endpoint

The study uses all 48,986 FIA49k molecules and the published gas-phase
`fia_gas-DSDBLYP` reference in kJ/mol. Official and ligand-scaffold partitions,
frozen checkpoints, complete predictions and training histories accompany the
results. Each geometric input is one optimized acid structure. Chemical analyses
cover ligand denticity, P(III), constrained coordination environments,
matched-ligand periodic trends and geometry-source sensitivity.

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

For a new, **unlabeled unbound-acid** XYZ file, use the frozen normalizer directly:

```bash
python -m lewis_acidity.predict --checkpoint runs/dimenet-s1/best.pt --xyz acid.xyz --center-index 0 --output predictions.csv
```

Use the training geometry protocol and identify the accepting atom by its 0-based
index. Repeat `--checkpoint` for an unweighted ensemble. Its exported SD measures
model disagreement and is not a calibrated interval. Acid-QM checkpoints also
require `--features features.json`, mapping each XYZ filename stem to exactly the
10 feature names stored in the checkpoint's feature state.

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

The completed experiment includes 23 neural runs. Main-test MAE is in kJ/mol:

| Model/input regime | Seed mean ± sample SD | Three-seed ensemble |
| --- | ---: | ---: |
| DimeNet++, acid geometry | 11.21 ± 0.27 | 10.29 |
| EGNN feature messages, acid geometry | 12.08 ± 0.16 | 11.04 |
| Radial filters, center readout | 12.22 ± 0.17 | 11.15 |
| Geometry + acid electronic features | 12.18 ± 0.04 | 11.12 |
| Released FIA-GNN reference | 12.11 (released predictions) | — |

The separate adduct-dependent refinement tier gives MAE 3.29, compared with
23.33 for raw PBEh-3c FIA, 10.06 for a train-only global correction and 7.15
for a center-class correction. Its scaffold-holdout MAE is 3.35. These values
measure agreement with the DSD-BLYP target and require lower-level acid **and
adduct** calculations. They cannot establish equivalent cost or absolute accuracy.

Small metrics/configuration/provenance reports are versioned under `reports/`.
Frozen checkpoints and complete molecule-level exports are also included as
checksum-verified storage chunks. Restore them after cloning:

```bash
python scripts/restore_artifacts.py
python -m lewis_acidity.predict --checkpoint checkpoints/dimenet-s1.pt.gz --xyz acid.xyz --center-index 0 --output predictions.csv
```

See `checkpoints/manifest.json` and `artifacts/storage_manifest.json` for exact
contents and hashes. Prepared data and original run directories remain local
under `.data/` and `runs/`; regenerate the input arrays with the command above.
Manuscript sources, figures and journal variants are under `manuscript/`;
native PDFs are in `manuscript/compiled/`. Optional Word exports are generated
locally in the ignored `manuscript/review/` directory.
An immutable archive DOI must be minted before publication.

Regenerate the tables and manuscript from completed run directories with:

```bash
python scripts/evaluate_completed.py
python -m lewis_acidity.analysis
python -m lewis_acidity.chemistry_insights
python scripts/plot_results.py
PYTHONPATH=. python scripts/plot_chemistry.py
python scripts/build_manuscripts.py
bash scripts/compile_manuscripts.sh
```

The article generator refuses a partial analysis. LaTeX is the primary manuscript
format: the RSC article layout and ACS `achemso` variant share the scientific text,
BibTeX bibliography and vector figures. `manuscript/compiled/` contains PDFs
compiled directly with Tectonic. See `manuscript/README.md` for Overleaf and local
build instructions. Word files are secondary editable exports. Install `cairosvg` to rebuild the
chemical structure drawings from released SMILES. Chemical diagnostic tables
and source definitions are in `reports/chemistry/`.

Data: Sigmund et al., FIA49k, CC BY 4.0,
[targets](https://doi.org/10.6084/m9.figshare.24635136.v1),
[geometries](https://doi.org/10.6084/m9.figshare.24635157.v1),
[released predictions](https://doi.org/10.6084/m9.figshare.24635187.v1).
Scientific reference: Sigmund et al., *Angew. Chem. Int. Ed.* 2024, 63, e202401084,
[doi:10.1002/anie.202401084](https://doi.org/10.1002/anie.202401084).


## Writing and figure review resources

Manuscript organization, evidence checks and figure review consulted the
scientific-writing, scientific-critical-thinking and scientific-visualization
guidance from [Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills).
Reference: T. Kassis, V. Agarwal, Y. He, D. Patel and A. M. Brueckner,
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*,
arXiv (2026), [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
These resources support editing and review; numerical results come from the
versioned computational records in this repository.
