import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Dict, List, Tuple, Optional

class FIA49kDimeNetDataset(Dataset):
    """
    Dataset class that loads data from DimeNet's processed .npz file
    and formats it for EGNN.
    """
    
    def __init__(self, npz_path: str, split: str = 'train', 
                 num_train: int = 45000, num_valid: int = 2000,
                 seed: int = 42, max_atoms: int = None,
                 fixed_atoms: List[int] = None, atom_order: List[int] = None):
        """
        Args:
            npz_path: Path to the .npz file (e.g., dimenet/data/fia49k_full_processed.npz)
            split: 'train', 'val', or 'test'
            num_train: Number of training samples
            num_valid: Number of validation samples
            seed: Random seed for splitting (must match DimeNet: 42)
            max_atoms: Maximum number of atoms for padding
            fixed_atoms: Optional list of atomic numbers to filter molecules.
            atom_order: Optional list of atomic numbers to define the one-hot encoding order.
                        If provided, it overrides the sorting logic of fixed_atoms.
        """
        self.npz_path = npz_path
        self.split = split
        self.max_atoms = max_atoms
        self.fixed_atoms = fixed_atoms
        self.atom_order = atom_order
        
        # Load data
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Data file not found: {npz_path}")
            
        print(f"Loading data from {npz_path}...")
        self.data = np.load(npz_path)
        
        # Extract arrays
        self.R = self.data['R']  # Coordinates (total_atoms, 3)
        self.Z = self.data['Z']  # Atomic numbers (total_atoms,)
        self.N = self.data['N']  # Num atoms per molecule (num_molecules,)
        self.max_atoms = int(self.N.max()) if max_atoms is None else max_atoms
        if int(self.N.max()) > self.max_atoms:
            raise ValueError('Atom cap would truncate a molecule; increase max_atoms')
        self.ids = self.data['id'] # Molecule IDs (num_molecules,)
        self.targets = self.data['fia_gas-DSDBLYP'] # Targets (num_molecules,)
        
        # Compute cumulative sum for indexing R and Z
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        
        self.num_molecules = len(self.N)
        
        # Determine unique atoms for one-hot encoding
        if self.fixed_atoms is not None:
            # DO NOT SORT if fixed_atoms is provided! 
            # We need to respect the order provided by the user (e.g. to match training data order)
            self.unique_atoms = np.array(self.fixed_atoms)
            print(f"Using fixed atom list (preserving order): {self.unique_atoms}")
        else:
            self.unique_atoms = np.unique(self.Z)
            
        self.atom_to_idx = {atom: i for i, atom in enumerate(self.unique_atoms)}
        self.n_atom_types = len(self.unique_atoms)
        
        # Split dataset using DimeNet's logic
        self._split_dataset(num_train, num_valid, seed)
        
        # Filter molecules if fixed_atoms is provided
        if self.fixed_atoms is not None:
            self._filter_molecules()
        
        print(f"Loaded {self.num_molecules} molecules total")
        print(f"Unique atoms: {self.unique_atoms}")
        print(f"Split '{split}': {len(self.indices)} molecules")

    def _split_dataset(self, num_train, num_valid, seed):
        """Replicate DimeNet's splitting logic"""
        random_state = np.random.RandomState(seed=seed)
        all_idx = np.arange(self.num_molecules)
        
        # DimeNet uses randomized=True by default in config
        all_idx = random_state.permutation(all_idx)
        
        if self.split == 'train':
            self.indices = all_idx[0:num_train]
        elif self.split == 'val':
            self.indices = all_idx[num_train:num_train+num_valid]
        elif self.split == 'test':
            self.indices = all_idx[num_train+num_valid:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
    def _filter_molecules(self):
        """Filter out molecules containing atoms not in fixed_atoms"""
        valid_indices = []
        dropped_count = 0
        
        valid_atom_set = set(self.fixed_atoms)
        
        for idx in self.indices:
            start = self.N_cumsum[idx]
            end = self.N_cumsum[idx+1]
            atoms = self.Z[start:end]
            
            is_valid = True
            for atom in atoms:
                if atom not in valid_atom_set:
                    is_valid = False
                    break
            
            if is_valid:
                valid_indices.append(idx)
            else:
                dropped_count += 1
                
        self.indices = np.array(valid_indices)
        if dropped_count > 0:
            raise ValueError(f'Atom vocabulary excludes {dropped_count} molecules; use the complete vocabulary')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get actual molecule index
        mol_idx = self.indices[idx]
        
        # Get atom range
        start = self.N_cumsum[mol_idx]
        end = self.N_cumsum[mol_idx+1]
        
        # Get data
        n_atoms = self.N[mol_idx]
        coords = self.R[start:end]
        atoms = self.Z[start:end]
        target = self.targets[mol_idx]
        mol_id = self.ids[mol_idx]
        
        # Filter if n_atoms > max_atoms (though DimeNet data might not need this if clean)
        if n_atoms > self.max_atoms:
            raise ValueError('Atom truncation is prohibited')
            
        # Prepare tensors with padding
        positions = torch.zeros((self.max_atoms, 3), dtype=torch.float32)
        positions[:n_atoms] = torch.tensor(coords, dtype=torch.float32)
        
        one_hot = torch.zeros((self.max_atoms, self.n_atom_types), dtype=torch.float32)
        for i, atom in enumerate(atoms):
            if atom in self.atom_to_idx:
                one_hot[i, self.atom_to_idx[atom]] = 1.0
                
        atomic_numbers = torch.zeros(self.max_atoms, dtype=torch.float32)
        atomic_numbers[:n_atoms] = torch.tensor(atoms, dtype=torch.float32)
        
        atom_mask = torch.zeros(self.max_atoms, dtype=torch.float32)
        atom_mask[:n_atoms] = 1.0
        
        return {
            'positions': positions,
            'one_hot': one_hot,
            'atomic_numbers': atomic_numbers,
            'atom_mask': atom_mask,
            'target': torch.tensor(target, dtype=torch.float32),
            'n_atoms': torch.tensor(n_atoms, dtype=torch.long),
            'mol_id': mol_id
        }

def collate_fn(batch):
    """Batch collation function"""
    batch_size = len(batch)
    
    # Check dimensions from first item
    max_atoms = batch[0]['positions'].shape[0]
    n_atom_types = batch[0]['one_hot'].shape[1]
    
    # Initialize batch tensors
    positions = torch.zeros(batch_size, max_atoms, 3)
    one_hot = torch.zeros(batch_size, max_atoms, n_atom_types)
    atomic_numbers = torch.zeros(batch_size, max_atoms)
    atom_mask = torch.zeros(batch_size, max_atoms)
    targets = torch.zeros(batch_size)
    n_atoms = torch.zeros(batch_size, dtype=torch.long)
    mol_ids = []
    
    for i, item in enumerate(batch):
        positions[i] = item['positions']
        one_hot[i] = item['one_hot']
        atomic_numbers[i] = item['atomic_numbers']
        atom_mask[i] = item['atom_mask']
        targets[i] = item['target']
        n_atoms[i] = item['n_atoms']
        mol_ids.append(item['mol_id'])
    
    # Create edge mask (fully connected)
    edge_mask = torch.zeros(batch_size, max_atoms, max_atoms)
    for i in range(batch_size):
        num = n_atoms[i].item()
        edge_mask[i, :num, :num] = 1.0
        
    return {
        'positions': positions,
        'one_hot': one_hot,
        'atomic_numbers': atomic_numbers,
        'atom_mask': atom_mask,
        'edge_mask': edge_mask,
        'targets': targets,
        'n_atoms': n_atoms,
        'mol_ids': mol_ids
    }

def create_dataloaders(npz_path: str, batch_size: int = 32, num_workers: int = 0,
                       num_train: int = 45000, num_valid: int = 2000, seed: int = 42,
                       fixed_atoms: List[int] = None):
    """Create DataLoaders for train, val, test"""
    
    train_dataset = FIA49kDimeNetDataset(npz_path, 'train', num_train, num_valid, seed, fixed_atoms=fixed_atoms)
    val_dataset = FIA49kDimeNetDataset(npz_path, 'val', num_train, num_valid, seed, fixed_atoms=fixed_atoms)
    test_dataset = FIA49kDimeNetDataset(npz_path, 'test', num_train, num_valid, seed, fixed_atoms=fixed_atoms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, collate_fn=collate_fn)
                             
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Test code
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz_path = os.path.join(base_path, 'dimenet', 'data', 'fia49k_full_processed.npz')
    
    if os.path.exists(npz_path):
        print("Testing FIA49kDimeNetDataset...")
        # Old atoms list
        old_atoms = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 31, 32, 33, 35, 49, 51, 53, 82, 83]
        # Not including 50 (Sn) and 52 (Te)
        
        train_loader, val_loader, test_loader = create_dataloaders(npz_path, batch_size=4, fixed_atoms=old_atoms)
        
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Positions shape: {batch['positions'].shape}")
        print(f"One-hot shape: {batch['one_hot'].shape}")
        print(f"Targets: {batch['targets']}")
        print("Test passed!")
    else:
        print(f"File not found: {npz_path}")
