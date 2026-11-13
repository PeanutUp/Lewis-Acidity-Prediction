"""Recompute metrics from released FIA-GNN predictions; does not train a model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data'
OUT = ROOT / 'reports' / 'public_reference'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checksums = {'FIA49k.csv.gz': '76b9018afda2f8b9c776d1555f55180f',
                 'FIA49k_predictions.csv.gz': '7f0f55c3e876d34452abd7bbb0c0e657'}
    for name, expected in checksums.items():
        assert hashlib.md5((DATA/name).read_bytes()).hexdigest() == expected, name
    d = pd.read_csv(DATA/'FIA49k.csv.gz', low_memory=False)
    p = pd.read_csv(DATA/'FIA49k_predictions.csv.gz')
    assert not d.Compound.duplicated().any()
    assert not p.Compound.duplicated().any()
    assert set(p.Compound) == set(d.loc[d.set_assignment.str.startswith('test'), 'Compound'])
    aligned = d.set_index('Compound').loc[p.Compound]
    for phase in ['gas', 'solv']:
        target = f'fia_{phase}-DSDBLYP'
        assert np.isfinite(p[[target, f'pred_{target}']].to_numpy()).all()
        assert np.allclose(aligned[target], p[target], rtol=0, atol=1e-6)
    assert (aligned.set_assignment.to_numpy() == p.set_assignment.to_numpy()).all()
    rows = []
    for split, g in p.groupby('set_assignment'):
        for phase in ['gas', 'solv']:
            ycol = f'fia_{phase}-DSDBLYP'
            vals = {r.Compound: (getattr(r, 'truth'), getattr(r, 'prediction'))
                    for r in g.rename(columns={ycol:'truth', f'pred_{ycol}':'prediction'}).itertuples()}
            rows.append(dict(split=split, phase=phase, **metrics(vals)))
            export = g[['Compound', ycol, f'pred_{ycol}']].copy()
            export.columns = ['mol_id', 'y_true', 'y_pred']
            export.to_csv(OUT/f'fia_gnn_{phase}_{split}_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT/'public_fia_gnn_metrics.csv', index=False)
    summary = dict(rows=len(d), unique_compounds=int(d.Compound.nunique()),
                   unique_exact_smiles=int(d.la_smiles.nunique()),
                   exact_smiles_crossing_splits=int(d.groupby('la_smiles').set_assignment.nunique().gt(1).sum()),
                   canonical_identity_audit='Not performed; exact-string uniqueness is not a chemical leakage test.',
                   split_counts=d.set_assignment.value_counts().to_dict(),
                   source_split_counts=pd.crosstab(d.mol_type, d.set_assignment).to_dict(),
                   classes=d.ca_class.value_counts().to_dict(),
                   target_missing={k:int(d[k].isna().sum()) for k in ['fia_gas-DSDBLYP','fia_solv-DSDBLYP']},
                   provenance='Figshare v1, DOI 10.6084/m9.figshare.24635136.v1 and '
                              '10.6084/m9.figshare.24635187.v1; predictions recomputed, no retraining')
    (OUT/'public_dataset_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    ids = ['B_3_ZT009074','P_3_ZT017132','As_3_ZT018788','As_3_ZT018778',
           'P_3_B014413','P_3_B014420','P_3_B014554','P_3_ZB016799','P_3_ZT026756']
    d.set_index('Compound').loc[ids, ['fia_gas-DSDBLYP','fia_solv-DSDBLYP',
                                       'set_assignment','mol_type','ca_class']].to_csv(OUT/'original_case_provenance.csv')
    merged = p.merge(d[['Compound','ca_class']], on='Compound', validate='one_to_one')
    byclass=[]
    for (split,ca),g in merged.groupby(['set_assignment','ca_class']):
        vals = {str(i):(float(y),float(yp)) for i,y,yp in
                zip(g.Compound,g['fia_gas-DSDBLYP'],g['pred_fia_gas-DSDBLYP'])}
        byclass.append(dict(split=split, ca_class=ca, **metrics(vals)))
    pd.DataFrame(byclass).to_csv(OUT/'public_fia_gnn_gas_by_class.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(rows)
    splits=['test','test_2','test_3','test_4','test_5','test_6']
    labels=['FIA44k\n(n=6732)','CSD\n(n=1200)','Bimacrocycles\n(n=763)',
            'Four-membered rings\n(n=911)','PT catalysts\n(n=15)','Catalysts\n(n=31)']
    fig,ax=plt.subplots(figsize=(9.4,3.8),layout='constrained')
    x=np.arange(6)
    for phase,offset,color,label in [('gas',-.18,'#226A9C','Gas phase'),('solv',.18,'#C9862B','CH₂Cl₂ solution')]:
        v=table[table.phase==phase].set_index('split').loc[splits,'mae'].to_numpy()
        bars=ax.bar(x+offset,v,.34,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    ax.set_xticks(x,labels,fontsize=8)
    ax.set_ylabel('Mean absolute error / kJ mol⁻¹')
    ax.set_ylim(0,29)
    ax.legend(frameon=False,loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.yaxis.grid(alpha=.16)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'published_fia_gnn_baseline.{ext}',dpi=300)
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
