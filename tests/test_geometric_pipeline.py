import numpy as np
import pytest
import torch
from lewis_acidity.data import Molecules, collate, split_indices, content_fingerprint
from lewis_acidity.model import AcidityNet
from lewis_acidity.prepare import parse_xyz
from lewis_acidity.train import load_checkpoint


def molecule(z, r, center=0):
    z, r = torch.tensor(z), torch.tensor(r, dtype=torch.float32)
    mask = ~torch.eye(len(z), dtype=torch.bool)
    return dict(z=z, r=r, edges=torch.stack(torch.where(mask)), center=center, y=1., index=0)


@pytest.mark.parametrize('kind', ['radial', 'egnn'])
@pytest.mark.parametrize('readout', ['sum', 'site'])
def test_rotation_translation_permutation_batch_and_checkpoint(kind, readout, tmp_path):
    torch.manual_seed(1)
    cfg = dict(hidden=16, layers=2, cutoff=0, kind=kind, readout=readout)
    model = AcidityNet(**cfg).eval()
    a = molecule([13, 1, 9], [[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    b = molecule([5, 6], [[0, 0, 0], [0, 0, 1]])
    reference = model(collate([a]))
    assert torch.allclose(reference, model(collate([a, b]))[:1], atol=1e-5)
    perm = torch.tensor([2, 0, 1])
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    transformed = molecule(a['z'][perm].tolist(), (a['r'][perm]@q + 13).tolist(), center=1)
    assert torch.allclose(reference, model(collate([transformed])), atol=1e-5)
    path = tmp_path/'model.pt'
    torch.save(dict(model=model.state_dict(), model_config=cfg, mean=123., scale=17.), path)
    restored, ckpt = load_checkpoint(path)
    assert ckpt['mean'] == 123. and ckpt['scale'] == 17.
    assert torch.equal(reference, restored(collate([a])))


def test_geometry_parser_rejects_silent_corruption():
    for text in ['1\ncomment\nXx 0 0 0', '2\ncomment\nC 0 0 0', '1\ncomment\nB nan 0 0']:
        with pytest.raises(ValueError): parse_xyz(text)
    z, _ = parse_xyz('2\ncomment\nSn 0 0 0\nTe 1 0 0')
    assert z.tolist() == [50, 52]


def test_official_split_manifest_is_disjoint_and_reproducible():
    labels = ['train', 'test', 'validate', 'test_2', 'train']
    first, second = split_indices(labels), split_indices(labels)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert len(set(np.concatenate(list(first.values())))) == len(labels)
    with pytest.raises(ValueError): split_indices(['train', 'validation_typo'])


def test_composition_control_cannot_retain_radius_geometry():
    with pytest.raises(ValueError): AcidityNet(geometry=False, cutoff=5)


def test_quantum_features_fit_only_training_and_restore_frozen_preprocessing(tmp_path):
    import pandas as pd
    from lewis_acidity.tabular import QM, META
    ids = np.array(['a','b','c'])
    np.savez(tmp_path/'geometry.npz', mol_id=ids, y=np.array([1.,2.,3.]),
             N=np.array([1,1,1]), Z=np.array([5,5,5]), R=np.zeros((3,3),dtype=np.float32),
             split=np.array(['train','train','validate']), center_index=np.zeros(3,dtype=np.int32))
    frame = pd.DataFrame({'Compound':ids, 'fia_gas-DSDBLYP':[1.,2.,3.]})
    for name in QM+META: frame[name] = [2.,4.,1000.]
    frame.to_csv(tmp_path/'FIA49k.csv.gz',index=False)
    a = Molecules(tmp_path/'geometry.npz',feature_mode='acid-qm')
    assert a.feature_state['mean'] == [3.] * len(QM+META)
    assert a.feature_state['scale'] == [1.] * len(QM+META)
    assert torch.equal(a[2]['global'],torch.tensor([997.] * len(QM+META)))
    b = Molecules(tmp_path/'geometry.npz',feature_mode='acid-qm',feature_state=a.feature_state)
    assert torch.equal(a[2]['global'],b[2]['global'])
    frame.loc[2,QM[0]] = 0
    frame.to_csv(tmp_path/'FIA49k.csv.gz',index=False)
    with pytest.raises(ValueError):
        Molecules(tmp_path/'geometry.npz',feature_mode='acid-qm',feature_state=a.feature_state)


def test_unlabeled_xyz_inference_reuses_checkpoint_scaling(tmp_path):
    from lewis_acidity.predict import predict_files,xyz_item
    torch.manual_seed(7)
    cfg=dict(hidden=16,layers=2,cutoff=5,kind='radial',readout='site')
    model=AcidityNet(**cfg).eval()
    checkpoint=tmp_path/'model.pt'
    torch.save(dict(model=model.state_dict(),model_config=cfg,mean=123.,scale=17.),checkpoint)
    xyz=tmp_path/'acid.xyz';xyz.write_text('3\nexample\nB 0 0 0\nF 1 0 0\nH 0 1 0\n')
    result=predict_files([checkpoint],[xyz])
    expected=(model(collate([xyz_item(xyz,0,5)]))*17+123).item()
    assert result.predicted_fia_gas_DSDBLYP_kJ_mol.iloc[0]==expected
    assert result.n_models.iloc[0]==1
    with pytest.raises(ValueError): predict_files([checkpoint],[xyz,xyz])


def test_dataset_fingerprint_ignores_zip_container_but_detects_changed_values(tmp_path):
    arrays=dict(mol_id=np.array(['acid']),N=np.array([1]),Z=np.array([5]),
                R=np.zeros((1,3)),y=np.array([123.]),split=np.array(['train']),center_index=np.array([0]))
    a,b=tmp_path/'a.npz',tmp_path/'b.npz'
    np.savez(a,**arrays);np.savez_compressed(b,**arrays)
    assert a.read_bytes()!=b.read_bytes()
    assert content_fingerprint(a)==content_fingerprint(b)
    arrays['y'][0]+=1;np.savez_compressed(b,**arrays)
    assert content_fingerprint(a)!=content_fingerprint(b)
