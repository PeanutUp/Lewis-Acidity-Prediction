#!/usr/bin/env python3
"""Evaluate corrected checkpoints with their saved normalization and split IDs.

Usage: python egnn/predict_fia49k_new.py runs/egnn-s1/best.pt --device cuda:0
Legacy state-dict-only checkpoints lack the required provenance and must not
be evaluated using freshly drawn normalization or test partitions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewis_acidity.evaluate import main
if __name__ == '__main__': main()
