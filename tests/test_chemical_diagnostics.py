"""Analytical geometry and population checks for the chemical analysis."""
import numpy as np
import pandas as pd
import pytest
from lewis_acidity.chemistry_insights import central_angles,attach_predictions,ROOT


def test_planar_and_pyramidal_first_shell():
    # A remote carbon must not displace the reported O attachment.
    z=np.array([15,6,6,8,6])
    planar=np.array([[0.,0.,0.],[1.,0.,0.],[-.5,np.sqrt(3)/2,0.],[-.5,-np.sqrt(3)/2,0.],[6.,0.,0.]])
    result=central_angles(z,planar,'C(SP2)-2_O(SP3)-1')
    assert result['angle_sum']==pytest.approx(360.)
    assert set(result['neighbor_indices'].split(';'))=={'1','2','3'}
    pyramidal=planar.copy();pyramidal[1:4]=np.eye(3)
    assert central_angles(z,pyramidal,'C(SP2)-2_O(SP3)-1')['angle_sum']==pytest.approx(270.)
    q,_=np.linalg.qr(np.random.default_rng(31).normal(size=(3,3)))
    transformed=pyramidal@q+np.array([32.,-7.,4.])
    assert central_angles(z,transformed,'C(SP2)-2_O(SP3)-1')['angle_sum']==pytest.approx(270.)


def test_incomplete_first_shell_is_rejected():
    with pytest.raises(ValueError):
        central_angles(np.array([15,6,6]),np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),'C(SP2)-2_O(SP3)-1')


def test_prediction_comparison_requires_identical_population(monkeypatch):
    # This deliberately incomplete population must fail before a metric is formed.
    data=pd.DataFrame({'fia_gas-DSDBLYP':[0.]},index=['missing-molecule'])
    monkeypatch.setattr(pd,'read_csv',lambda *a,**k:pd.DataFrame(
        {'mol_id':['another-molecule'],'y_true':[0.],'y_pred':[0.]}))
    with pytest.raises(ValueError,match='Molecule mismatch'):
        attach_predictions(data,'test')
