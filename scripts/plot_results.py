"""Rebuild vector scientific figures from the frozen result tables."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/figures'
SPLITS=['test','test_2','test_3','test_4','test_5','test_6']
LABELS=['FIA44k\n6,732','CSD\n1,200','Bimacrocycles\n763','Four-membered\nrings: 911','PT catalysts\n15','Catalysts\n31']
COLORS=['#2877A6','#D37828','#16806A','#9B5C93','#B69A25','#505D6B']

def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=400,bbox_inches='tight',facecolor='white')
 plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');a=p.parse_args()
 status=json.loads((ROOT/'reports/neural/status.json').read_text())
 if status['partial'] and not a.preview:raise RuntimeError('Need complete experiments')
 OUT.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 seeds=pd.read_csv(ROOT/'reports/neural/seed_summary.csv')
 ens=pd.read_csv(ROOT/'reports/neural/ensemble_metrics.csv')
 ref=pd.read_csv(ROOT/'reports/public_reference/public_fia_gnn_metrics.csv').query('phase=="gas"')
 q=pd.read_csv(ROOT/'reports/refinement/metrics.csv')
 labels={'radial-site':'Radial / center readout','radial-sum':'Radial / sum readout','egnn-sum':'EGNN feature messages','dimenet':'DimeNet++','acid-qm':'Geometry + acid QM'}
 fig,(ax,bx)=plt.subplots(1,2,figsize=(8.1,3.65),layout='constrained')
 models=[m for m in labels if ((seeds.model==m)&(seeds.split=='test')).any()]
 for i,m in enumerate(models):
  r=seeds[(seeds.model==m)&(seeds.split=='test')].iloc[0]
  ax.errorbar(r.mae_mean,i,xerr=0 if pd.isna(r.mae_sd) else r.mae_sd,fmt='o',color=COLORS[i],capsize=3)
  ax.annotate(f'{r.mae_mean:.2f}',(r.mae_mean,i),xytext=(8,0),textcoords='offset points',va='center',fontsize=8)
 ax.axvline(ref.loc[ref.split.eq('test'),'mae'].iloc[0],color='#697480',ls='--',lw=1,label='Released FIA-GNN')
 ax.set_yticks(range(len(models)),[labels[m] for m in models],fontsize=8)
 ax.set_ylim(len(models)-.5,-.5);ax.set_xlabel('MAE / kJ mol⁻¹');ax.set_title('(a) Acid-geometry predictors',loc='left',fontsize=10)
 values=seeds.loc[seeds.model.isin(models)&seeds.split.eq('test'),'mae_mean']
 ax.set_xlim(float(values.min())-.8,float(values.max())+1.2)
 ax.legend(frameon=False,fontsize=8,loc='lower left')
 names=['PBEh3c_raw','PBEh3c_global_bias','PBEh3c_class_bias','delta_metadata','delta_RDKit_LA_QM']
 texts=['Raw PBEh-3c','Global correction','Center-class correction','Metadata Δ model','2D + acid-QM Δ model']
 vals=q[q.split.eq('test')].set_index('model').loc[names,'mae']
 bx.barh(np.arange(5),vals,color=['#AAB1B7','#7C8C99','#547E94','#267C9C','#16806A'],height=.6)
 bx.set_yticks(np.arange(5),texts,fontsize=8);bx.invert_yaxis();bx.set_xlabel('MAE / kJ mol⁻¹')
 bx.set_title('(b) Acid + adduct quantum input',loc='left',fontsize=10);bx.set_xlim(0,27)
 for i,v in enumerate(vals):bx.text(v+.35,i,f'{v:.2f}',va='center',fontsize=8)
 save(fig,'representation_and_refinement')
 fig,ax=plt.subplots(figsize=(7.4,3.6),layout='constrained')
 for i,m in enumerate(['egnn-sum','dimenet','radial-site','acid-qm']):
  frame=ens[ens.model.eq(m)].set_index('split')
  if not set(SPLITS)<=set(frame.index):continue
  ax.plot(range(6),frame.loc[SPLITS,'mae'],marker='o',ms=4,lw=1.4,color=COLORS[i],label=labels[m])
 ax.plot(range(6),ref.set_index('split').loc[SPLITS,'mae'],'s--',color='#5A6470',lw=1.2,ms=4,label='FIA-GNN (released)')
 ax.plot(range(6),q[q.model.eq('delta_RDKit_LA_QM')].set_index('split').loc[SPLITS,'mae'],'D:',color='#B59B25',lw=1.5,ms=4,label='Δ refinement (adduct input)')
 ax.set_xticks(range(6),LABELS,fontsize=8);ax.set_ylabel('MAE / kJ mol⁻¹');ax.set_ylim(bottom=0)
 ax.legend(frameon=False,fontsize=7.5,ncol=2,loc='upper left');ax.grid(axis='y',alpha=.15);save(fig,'external_groups')
 bias=pd.read_csv(ROOT/'reports/train_method_bias_by_class.csv')
 matched=pd.read_csv(ROOT/'reports/train_matched_ligand_corrections.csv')
 fig,(ax,bx)=plt.subplots(2,1,figsize=(6.7,6.1),height_ratios=[1,1.15],layout='constrained')
 ax.bar(range(len(bias)),bias['mean'],yerr=bias['std'],capsize=1.3,width=.65,color='#2877A6',error_kw={'elinewidth':.65,'ecolor':'#7C8B95'})
 ax.set_xticks(range(len(bias)),bias.ca_class,rotation=90,fontsize=8)
 ax.set_ylabel('DSD − PBEh-3c FIA / kJ mol⁻¹');ax.set_title('(a) Training class means ± SD',loc='left',fontsize=10)
 x=matched.mean_correction_b_minus_a;y=np.arange(len(matched))
 bx.errorbar(x,y,xerr=np.vstack([x-matched.ci95_low,matched.ci95_high-x]),fmt='o',color='#16806A',capsize=3,ms=4)
 bx.axvline(0,color='#8B939A',lw=.8,ls='--')
 bx.set_yticks(y,[f'{r.class_b} − {r.class_a}  (n={r.n_ligand_sets})' for r in matched.itertuples()],fontsize=8)
 bx.invert_yaxis();bx.set_xlabel('Paired correction difference / kJ mol⁻¹');bx.set_title('(b) Matched ligands, 95% CI',loc='left',fontsize=10)
 save(fig,'method_corrections')
 shift=pd.read_csv(ROOT/'reports/geometry_shift/ensemble_metrics.csv').set_index('split')
 original=ens[ens.model.eq('radial-site')].set_index('split')
 fig,ax=plt.subplots(figsize=(7.4,3.2),layout='constrained');x=np.arange(6)
 ax.bar(x-.18,original.loc[SPLITS,'mae'],.34,color='#2877A6',label='PBEh-3c geometry')
 ax.bar(x+.18,shift.loc[SPLITS,'mae'],.34,color='#D37828',label='xTB geometry; frozen PBEh-trained model')
 ax.set_xticks(x,LABELS,fontsize=8);ax.set_ylabel('MAE / kJ mol⁻¹');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 save(fig,'geometry_source_shift')
 # Native schematic: no generated molecular image or unverifiable mechanistic arrow.
 fig,ax=plt.subplots(figsize=(3.25,1.75));ax.set_xlim(0,3);ax.set_ylim(0,1);ax.axis('off')
 boxes=[('Connectivity','FIA screening','#2877A6'),('Optimized acid','Geometry + electronics','#16806A'),('Acid + F⁻ adduct','Reference-level refinement','#D37828')]
 for i,(top,bottom,color) in enumerate(boxes):
  ax.text(i+.5,.58,top,ha='center',va='center',fontsize=6.5,color='white',bbox={'boxstyle':'round,pad=.5','facecolor':color,'edgecolor':'none'})
  ax.text(i+.5,.25,bottom.replace(' ','\n',1),ha='center',va='center',fontsize=6)
 ax.text(1.5,.95,'Fluoride ion affinity: input and prediction',ha='center',fontsize=8)
 save(fig,'toc_graphic')
 print('Scientific figures rebuilt',flush=True)
if __name__=='__main__':main()
