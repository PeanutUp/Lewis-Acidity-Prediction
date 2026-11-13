#!/usr/bin/env python3
"""Predict new acids using frozen, provenance-complete checkpoints.

This replaces the legacy arbitrary sampled-cohort export. Historical code is in
Git history. For independent benchmark scoring use lewis_acidity.evaluate;
for unlabeled XYZ input use the CLI below.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewis_acidity.predict import main
if __name__ == '__main__': main()
