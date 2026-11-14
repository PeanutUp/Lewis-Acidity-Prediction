"""Rebuild publication figures from frozen tables, with separate text space."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/figures'
SPLITS=['test','test_2','test_3','test_4','test_5','test_6']
LABELS=['Main\nn = 6,732','CSD\nn = 1,200','Bimacrocycles\nn = 763','Four-membered\nn = 911','PT catalysts\nn = 15','Catalysts\nn = 31']
COLORS=['#0072B2','#D55E00','#009E73','#8B63A6','#B39A2E','#566573']
LIGHT=LinearSegmentedColormap.from_list('paper',['#FFFFFF','#DFEDF2','#A5CCD9'])

def save(fig,name):
 # Fixed page size is preserved for reliable final-size typography.
 for ext in ['pdf','svg','png']:
  fig.savefig(OUT/f'{name}.{ext}',dpi=400,facecolor='white')
 svg=OUT/f'{name}.svg'
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 plt.close(fig)

def style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9.5,
  'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
  'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
  'axes.titlepad':10,'lines.linewidth':1.3,'axes.linewidth':.7})

def grid(ax):
 ax.grid(axis='x',color='#E5E9EC',lw=.6);ax.set_axisbelow(True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');a=p.parse_args()
 status=json.loads((ROOT/'reports/neural/status.json').read_text())
 if status['partial'] and not a.preview:raise RuntimeError('Need complete experiments')
 OUT.mkdir(parents=True,exist_ok=True);style()
 seeds=pd.read_csv(ROOT/'reports/neural/seed_summary.csv')
 ens=pd.read_csv(ROOT/'reports/neural/ensemble_metrics.csv')
 ref=pd.read_csv(ROOT/'reports/public_reference/public_fia_gnn_metrics.csv').query('phase=="gas"')
 q=pd.read_csv(ROOT/'reports/refinement/metrics.csv')
 labels={'radial-site':'Radial / center','radial-sum':'Radial / sum','egnn-sum':'EGNN','dimenet':'DimeNet++','acid-qm':'Geometry + acid QM'}
 fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,3.4),layout='constrained',gridspec_kw={'width_ratios':[1.15,1]})
 models=list(labels)
 for i,m in enumerate(models):
  r=seeds[seeds.model.eq(m)&seeds.split.eq('test')].iloc[0]
  ax.errorbar(r.mae_mean,i,xerr=r.mae_sd,fmt='o',ms=4.5,color=COLORS[i],capsize=3,zorder=3)
  ax.text(13.65,i,f'{r.mae_mean:.2f}',ha='right',va='center',fontsize=8.5)
 ax.axvline(ref.loc[ref.split.eq('test'),'mae'].iloc[0],color='#737D85',ls='--',lw=1)
 ax.set_yticks(range(5),[labels[m] for m in models]);ax.set_ylim(4.6,-1.2);ax.set_xlim(10.1,13.9)
 ax.text(12.11186,-.65,'FIA-GNN',ha='center',va='center',fontsize=7.5,color='#56616A',bbox=dict(facecolor='white',edgecolor='none',pad=2))
 ax.set_xlabel('MAE / kJ mol⁻¹');ax.set_title('(a) Optimized acid as input',loc='left');grid(ax)
 names=['PBEh3c_raw','PBEh3c_global_bias','PBEh3c_class_bias','delta_metadata','delta_RDKit_LA_QM']
 texts=['PBEh-3c','Global bias','Center-class bias','Metadata Δ','Full Δ model']
 vals=q[q.split.eq('test')].set_index('model').loc[names,'mae']
 bx.barh(np.arange(5),vals,color=['#AAB5BC','#7E9BAC','#4F8DAA','#0072B2','#009E73'],height=.57,zorder=3)
 bx.set_yticks(np.arange(5),texts);bx.set_ylim(4.6,-1.2);bx.set_xlim(0,28)
 bx.set_xlabel('MAE / kJ mol⁻¹');bx.set_title('(b) Acid + adduct as input',loc='left');grid(bx)
 for i,v in enumerate(vals):bx.text(v+.65,i,f'{v:.2f}',va='center',fontsize=8.5)
 save(fig,'representation_and_refinement')
 # A matrix avoids connecting unrelated chemical categories and needs no legend.
 names=['dimenet','egnn-sum','radial-site','acid-qm']
 values=[ens[ens.model.eq(m)].set_index('split').loc[SPLITS,'mae'].values for m in names]
 values += [ref.set_index('split').loc[SPLITS,'mae'].values,
            q[q.model.eq('delta_RDKit_LA_QM')].set_index('split').loc[SPLITS,'mae'].values]
 fig,ax=plt.subplots(figsize=(7.15,3.35),layout='constrained')
 ax.imshow(values,cmap=LIGHT,vmin=0,vmax=26,aspect='auto')
 ax.set_yticks(range(6),['DimeNet++','EGNN','Radial / center','Geometry + acid QM','FIA-GNN','Refinement: acid + adduct'])
 ax.set_xticks(range(6),LABELS);ax.tick_params(length=0,pad=8)
 for i,row in enumerate(values):
  for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=9)
 ax.axhline(4.5,color='#697B87',lw=1.1)
 for spine in ax.spines.values():spine.set_visible(False)
 ax.set_title('Test-group MAE / kJ mol⁻¹',loc='left');save(fig,'external_groups')
 bias=pd.read_csv(ROOT/'reports/train_method_bias_by_class.csv')
 matched=pd.read_csv(ROOT/'reports/train_matched_ligand_corrections.csv')
 fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,4.3),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
 y=np.arange(len(bias));ax.errorbar(bias['mean'],y,xerr=bias['std'],fmt='o',ms=3,color=COLORS[0],ecolor='#9AAFB9',capsize=2,lw=.8)
 ax.set_yticks(y,bias.ca_class,fontsize=8);ax.invert_yaxis();ax.set_xlabel('DSD − PBEh-3c FIA / kJ mol⁻¹')
 ax.set_title('(a) Center-class mean ± SD',loc='left');grid(ax)
 x=matched.mean_correction_b_minus_a;y=np.arange(len(matched))
 bx.errorbar(x,y,xerr=np.vstack([x-matched.ci95_low,matched.ci95_high-x]),fmt='o',color=COLORS[2],capsize=3,ms=4)
 bx.axvline(0,color='#8B939A',lw=.8,ls='--')
 bx.set_yticks(y,[f'{r.class_b} − {r.class_a}\nn = {r.n_ligand_sets} ligand sets' for r in matched.itertuples()],fontsize=7.5)
 bx.set_ylim(len(y)-.6,-.6);bx.set_xlabel('Correction difference / kJ mol⁻¹')
 bx.set_title('(b) Matched ligands, 95% CI',loc='left');grid(bx)
 save(fig,'method_corrections')
 shift=pd.read_csv(ROOT/'reports/geometry_shift/ensemble_metrics.csv').set_index('split')
 original=ens[ens.model.eq('radial-site')].set_index('split')
 fig,ax=plt.subplots(figsize=(7.15,3.35),layout='constrained');x=np.arange(6)
 ax.bar(x-.18,original.loc[SPLITS,'mae'],.34,color=COLORS[0],label='PBEh-3c geometry')
 ax.bar(x+.18,shift.loc[SPLITS,'mae'],.34,color=COLORS[1],hatch='///',label='GFN2-xTB geometry')
 ax.set_xticks(x,LABELS);ax.set_ylabel('MAE / kJ mol⁻¹')
 ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 save(fig,'geometry_source_shift')
 fig,ax=plt.subplots(figsize=(3.25,1.75));ax.set_xlim(0,3);ax.set_ylim(0,1);ax.axis('off')
 boxes=[('Connectivity','Rapid\nscreening',COLORS[0]),('Acid geometry','Chemical\nrepresentation',COLORS[2]),('Acid + adduct','Quantum\nrefinement',COLORS[1])]
 for i,(top,bottom,color) in enumerate(boxes):
  ax.text(i+.5,.58,top,ha='center',va='center',fontsize=6.2,color='white',bbox={'boxstyle':'round,pad=.5','facecolor':color,'edgecolor':'none'})
  ax.text(i+.5,.25,bottom,ha='center',va='center',fontsize=6.2)
 ax.text(1.5,.95,'Fluoride binding across the p-block',ha='center',fontsize=8)
 save(fig,'toc_graphic')
 print('Scientific figures rebuilt',flush=True)
if __name__=='__main__':main()
