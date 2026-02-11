# GNN-based Lewis Acidity Prediction

This repository contains code and resources for a course project applying Graph Neural Networks (GNNs) and machine learning techniques to predict chemical properties and tasks. The project explores various deep learning models, including DimeNet and EGNN, and compares their performance on FIA49k.

## Attribution & Modifications

This project is based on the following open-source repositories:

- **dimenet/**: [DimeNet](https://github.com/klicperajo/dimenet) (original authors: Johannes Klicpera et al.)
- **egnn/**: [EGNN](https://github.com/vgsatorras/egnn) (original authors: Victor Garcia Satorras et al.)

These folders contain code and scripts from the original projects, used here for academic research and comparison. Please refer to their LICENSE files for usage terms.

### My Contributions

- **Contrastive Learning/**: Scripts and results for contrastive learning experiments and analysis (newly added).
- **ML_xgboost/**: Baseline machine learning models and results (newly added).
- Additional scripts for data analysis, alignment, and interpretability (e.g., `analyze_phosphorus.py`, CSV analysis files).
- Modifications to configuration, training, and evaluation scripts to adapt to the course project and dataset (e.g., `predict_fia49k.py`, `train_fia49k.py`)

If you use or reference this repository, please also cite the original DimeNet and EGNN papers and repositories.

## Project Structure

- **dimenet/**: DimeNet model code, training scripts, data processing, and configuration files.
- **egnn/**: EGNN model code, training and evaluation scripts, and utilities.
- **Contrastive Learning/**: Scripts and results for contrastive learning experiments and analysis.
- **ML_xgboost/**: Baseline machine learning models (e.g., XGBoost) and results.

## Main Features
- Training of GNN models for chemical property prediction.
- Data preprocessing and conversion scripts for chemical datasets.
- Analysis and comparison of model predictions and performance.
- Baseline machine learning models for benchmarking.

## Requirements
- Python 3.7+
- See `dimenet/environment.yml` for dependencies and environment setup.

## Usage
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```
2. Set up the environment:
   ```bash
   conda env create -f dimenet/environment.yml
   conda activate <env-name>
   ```
3. Run training or prediction scripts as needed. Example:
   ```bash
   python dimenet/train_fia49k.py
   ```

## Data
- **Note:** Raw and processed data files are not included in the repository. Please refer to the dataset documentation or contact the authors for access.

