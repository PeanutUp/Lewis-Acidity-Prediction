"""Diagnostic geometry changes on fluoride binding, never screening inputs.

Archive atom-order equivalence is checked explicitly. A pair-distance RMS change
is alignment-free and measures structural displacement, NOT deformation energy.
Acid xTB/PBEh changes characterize geometry-source sensitivity, not conformers.
"""
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .prepare import parse_xyz

ROOT = Path(__file__).resolve().parents[1]


def distance_matrix(r):
    return np.linalg.norm(r[:, None]-r[None, :], axis=-1)


def main():
    rows = []
    with zipfile.ZipFile(ROOT/'.data/FIA49k.zip') as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.json'): continue
            for key, entry in json.loads(archive.read(name)).items():
                za, ra = parse_xyz(entry['la-PBEh3c'])
                zf, rf = parse_xyz(entry['fa-PBEh3c'])
                zx, rx = parse_xyz(entry['la-GFN2xTB'])
                if not (np.array_equal(za, zf[:-1]) and zf[-1] == 9 and np.array_equal(za, zx)):
                    raise ValueError(f'Atom-order correspondence unverified: {key}')
                heavy = za > 1
                da, df, dx = [distance_matrix(r[heavy]) for r in (ra, rf[:-1], rx)]
                upper = np.triu_indices(len(da), k=1)
                delta, source_delta = (df-da)[upper], (dx-da)[upper]
                # Atom 0 is the verified accepting center; exclude H when describing its shell.
                heavy_ix = np.flatnonzero(heavy); donor = heavy_ix[heavy_ix != 0]
                acid_dist = np.linalg.norm(ra[donor]-ra[0], axis=1)
                adduct_dist = np.linalg.norm(rf[donor]-rf[0], axis=1)
                near = acid_dist < 2.7
                rows.append(dict(Compound=key, heavy_pair_distance_rms_change=float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.,
                    geometry_source_pair_rms=float(np.sqrt(np.mean(source_delta**2))) if len(source_delta) else 0.,
                    center_near_heavy_count=int(near.sum()),
                    center_near_distance_rms_change=float(np.sqrt(np.mean((adduct_dist[near]-acid_dist[near])**2))) if near.any() else 0.,
                    added_f_distance=float(np.linalg.norm(rf[-1]-rf[0]))))
            print(name, len(rows), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT/'reports/geometry_diagnostics.csv', index=False)
    (ROOT/'reports/geometry_diagnostics.json').write_text(json.dumps(dict(
        n_checked=len(rows), correspondence='N_adduct=N_acid+1; all preceding Z equal; final Z=9; xTB Z equal',
        caveat='Element-sequence equality checks the archive order convention; identical-element atom permutations cannot be excluded by Z alone.',
        quantities='Pair distances of heavy atoms; reported in Angstrom. Not energies. Not model inputs.',
        shell='All non-H atoms within 2.7 Angstrom of verified central atom; geometric shell, not assigned covalent bonds.'), indent=2)+'\n')


if __name__ == '__main__': main()
