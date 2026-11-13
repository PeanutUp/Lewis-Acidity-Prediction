"""Prepare ID-aligned, unbound-acid geometries from versioned FIA49k files.

The original converters mixed geometry fallbacks, discarded molecule IDs, or
silently mapped unrecognized elements to zero. This path fails on those cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
           "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac "
           "Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
ATOMIC_NUMBER = {symbol: number for number, symbol in enumerate(SYMBOLS)}
PUBLIC_FILES = {
    "FIA49k.csv.gz": (43289211, "76b9018afda2f8b9c776d1555f55180f"),
    "FIA49k_predictions.csv.gz": (43289322, "7f0f55c3e876d34452abd7bbb0c0e657"),
    "FIA49k.zip": (43289292, "60e9ffd7280bdfd6baeb37d6a641f81b"),
}


def checksum(path: Path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(path: Path):
    file_id, md5 = PUBLIC_FILES[path.name]
    if path.exists() and checksum(path, "md5") == md5:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(f"https://ndownloader.figshare.com/files/{file_id}", timeout=60) as response:
        with partial.open("wb") as f:
            for chunk in iter(lambda: response.read(1024*1024), b""):
                f.write(chunk)
    if checksum(partial, "md5") != md5:
        raise ValueError(f"Checksum mismatch: {partial}")
    partial.replace(path)


def parse_xyz(text: str):
    lines = text.strip().splitlines()
    n = int(lines[0])
    if n < 1 or len(lines) != n + 2:
        raise ValueError("XYZ atom count does not match coordinate rows")
    numbers, positions = [], []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 4 or fields[0] not in ATOMIC_NUMBER:
            raise ValueError(f"Malformed atom record: {line!r}")
        number = ATOMIC_NUMBER[fields[0]]
        if number == 0:
            raise ValueError("Atomic number zero is reserved for padding")
        numbers.append(number)
        positions.append([float(x) for x in fields[1:]])
    numbers = np.asarray(numbers, dtype=np.int16)
    positions = np.asarray(positions, dtype=np.float32)
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite coordinates")
    return numbers, positions


def prepare(data_dir: Path, geometry: str = "la-PBEh3c"):
    if geometry not in {"la-PBEh3c", "la-GFN2xTB"}:
        raise ValueError("Training input must be an explicitly selected unbound-acid geometry")
    for name in PUBLIC_FILES:
        download(data_dir / name)
    metadata = pd.read_csv(data_dir / "FIA49k.csv.gz", low_memory=False)
    if not metadata.Compound.is_unique or metadata.Compound.isna().any():
        raise ValueError("Compound identifiers must be nonmissing and unique")
    metadata = metadata.set_index("Compound", drop=False)
    coordinates = {}
    with zipfile.ZipFile(data_dir / "FIA49k.zip") as archive:
        for filename in sorted(archive.namelist()):
            if not filename.endswith(".json"):
                continue
            with archive.open(filename) as f:
                part = json.load(f)
            for mol_id, entry in part.items():
                if mol_id not in metadata.index:
                    raise ValueError(f"Geometry without target metadata: {mol_id}")
                if mol_id in coordinates:
                    raise ValueError(f"Duplicate structure: {mol_id}")
                if geometry not in entry:
                    raise ValueError(f"Missing {geometry} for {mol_id}; no geometry fallback is used")
                z, r = parse_xyz(entry[geometry])
                center_z = ATOMIC_NUMBER[metadata.loc[mol_id, "ca"]]
                # The public XYZ convention places the accepting center first.
                # Check it for EVERY record, including molecules with repeated elements.
                if int(z[0]) != center_z:
                    raise ValueError(f"Center-first archive convention violated: {mol_id}")
                coordinates[mol_id] = (z, r)
            print(f"Parsed {filename}: {len(coordinates)} structures", flush=True)
    missing = set(metadata.index) - set(coordinates)
    if missing:
        raise ValueError(f"Missing geometries for {len(missing)} records: {sorted(missing)[:5]}")
    ids = metadata.Compound.to_numpy(dtype=str)
    counts = np.array([len(coordinates[i][0]) for i in ids], dtype=np.int32)
    target = metadata["fia_gas-DSDBLYP"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite targets")
    destination = data_dir / f"{geometry}.npz"
    np.savez_compressed(destination, mol_id=ids, N=counts,
                        Z=np.concatenate([coordinates[i][0] for i in ids]),
                        R=np.concatenate([coordinates[i][1] for i in ids]),
                        y=target, split=metadata.set_assignment.to_numpy(dtype=str),
                        center_index=np.zeros(len(ids), dtype=np.int32))
    audit = dict(geometry=geometry, geometry_fallback=False, center_index_convention="first atom; checked for every record",
                 n_molecules=len(ids), total_atoms=int(counts.sum()), max_atoms=int(counts.max()),
                 atom_types=sorted(set(int(v) for i in ids for v in coordinates[i][0])),
                 split_counts=metadata.set_assignment.value_counts().to_dict(),
                 input_checksums={name: checksum(data_dir/name) for name in PUBLIC_FILES},
                 output_sha256=checksum(destination))
    destination.with_suffix(".json").write_text(json.dumps(audit, indent=2)+"\n")
    # Exact molecule identity is portable and must never be reconstructed from row numbers later.
    metadata[["Compound", "set_assignment", "mol_type", "ca_class", "la_smiles", "ligand_smiles"]].to_csv(
        data_dir / "official_manifest.csv", index=False)
    print(json.dumps(audit, indent=2), flush=True)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path(".data"))
    p.add_argument("--geometry", choices=["la-PBEh3c", "la-GFN2xTB"], default="la-PBEh3c")
    args = p.parse_args()
    prepare(args.data_dir, args.geometry)


if __name__ == "__main__":
    main()
