import torch
import pytest
from egnn.fia49k_new_dataset import FIA49kNewDataset, collate_fn
from egnn.fia49k_new_converter import get_atomic_number, process_molecule
from egnn.qm9.utils import compute_mean_mad
from lewis_acidity.grouped import group_key, assignment, scaffold_key


def test_independent_dataset_constructors_share_one_partition(tmp_path):
    n = 30
    data = dict(positions=torch.zeros(n, 2, 3), unique_atoms=['H'], targets=torch.arange(n).float())
    path = tmp_path/'data.pt'; torch.save(data, path)
    parts = [FIA49kNewDataset(path, split=s) for s in ['train','val','test']]
    sets = [set(x.indices.tolist()) for x in parts]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert len(set.union(*sets)) == n
    assert torch.equal(parts[0].indices, FIA49kNewDataset(path).indices)


def test_train_normalization_excludes_validation_and_test():
    class Dataset:
        data = {'target': torch.tensor([2.,4.,1000.])}
        indices = torch.tensor([0,1])
    class Loader: dataset = Dataset()
    mean, mad = compute_mean_mad({'train':Loader()}, 'target')
    assert mean.item() == 3 and mad.item() == 1


def test_complete_periodic_table_and_unknown_failures():
    assert get_atomic_number('P') == 15 and get_atomic_number('Sn') == 50
    with pytest.raises(ValueError): get_atomic_number('Unknown')


def test_ligand_grouping_ignores_fragment_order_and_chemical_substitution_for_scaffolds():
    assert group_key('Cl.[O]C') == group_key('[O]C.Cl')
    assert assignment(group_key('Cl.[O]C')) == assignment(group_key('[O]C.Cl'))
    assert scaffold_key('Cc1ccccc1.[Cl]') == scaffold_key('Fc1ccccc1.[Br]')
