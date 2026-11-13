#!/usr/bin/env python3
"""Corrected FIA49k training entry point.

Uses the shared, ID-preserving official-split pipeline. The old implementation
is available in Git history (4f28d89); its random per-partition splitting and
padding-edge handling do not support a held-out performance claim.
New CLI: python egnn/main_fia49k_new.py --output runs/egnn-s1 --device cuda:0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewis_acidity.train import main

if __name__ == '__main__':
    for flag, value in [('--kind', 'egnn'), ('--layers', '7'), ('--readout', 'sum')]:
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    main()
