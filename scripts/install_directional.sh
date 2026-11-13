#!/usr/bin/env bash
set -euo pipefail
# Run after installing torch 2.6.0+cu124 in your own virtual environment.
python -m pip install torch-geometric==2.6.1
python -m pip install --only-binary=:all: torch-cluster torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
